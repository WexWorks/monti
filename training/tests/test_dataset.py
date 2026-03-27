"""Tests for ExrDataset and transforms."""

import os
import tempfile

import numpy as np
import pytest
import torch

from deni_train.data.exr_dataset import ExrDataset
from deni_train.data.transforms import Compose, RandomCrop, RandomHorizontalFlip


@pytest.fixture
def synthetic_data_dir():
    """Generate a small synthetic dataset in a temp directory."""
    # Import here to avoid requiring scipy at module level
    from scripts.generate_synthetic_data import generate

    with tempfile.TemporaryDirectory() as tmpdir:
        generate(tmpdir, num_pairs=3, width=64, height=48, seed=123)
        yield tmpdir


class TestExrDataset:
    def test_finds_pairs(self, synthetic_data_dir):
        ds = ExrDataset(synthetic_data_dir)
        assert len(ds) == 3

    def test_input_tensor_shape(self, synthetic_data_dir):
        ds = ExrDataset(synthetic_data_dir)
        input_tensor, _, _, _, _ = ds[0]
        assert input_tensor.shape == (19, 48, 64)

    def test_target_tensor_shape(self, synthetic_data_dir):
        ds = ExrDataset(synthetic_data_dir)
        _, target_tensor, _, _, _ = ds[0]
        assert target_tensor.shape == (6, 48, 64)

    def test_tensor_dtype_fp16(self, synthetic_data_dir):
        ds = ExrDataset(synthetic_data_dir)
        input_tensor, target_tensor, albedo_d, albedo_s, hit_mask = ds[0]
        assert input_tensor.dtype == torch.float16
        assert target_tensor.dtype == torch.float16
        assert albedo_d.dtype == torch.float16
        assert albedo_s.dtype == torch.float16
        assert hit_mask.dtype == torch.float16

    def test_target_is_demodulated_irradiance(self, synthetic_data_dir):
        """Verify target channels are demodulated diffuse + specular irradiance."""
        import OpenEXR
        import Imath

        ds = ExrDataset(synthetic_data_dir)
        _, target_tensor, _, _, _ = ds[0]

        # Target should have 6 channels: demodulated diffuse RGB + specular RGB
        assert target_tensor.shape[0] == 6

        # Read raw EXR channels to verify demodulation
        input_path = ds.pairs[0][0]
        target_path = ds.pairs[0][1]
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        width, height = 64, 48

        inp_exr = OpenEXR.InputFile(input_path)
        tgt_exr = OpenEXR.InputFile(target_path)

        albedo_d_r = np.frombuffer(inp_exr.channel("albedo_d.R", pt), dtype=np.float32).reshape(height, width)
        hit_mask = np.frombuffer(inp_exr.channel("diffuse.A", pt), dtype=np.float32).reshape(height, width)
        hit_bool = hit_mask > 0.5
        diff_r = np.frombuffer(tgt_exr.channel("diffuse.R", pt), dtype=np.float32).reshape(height, width)
        inp_exr.close()
        tgt_exr.close()

        # Expected: demodulated diffuse R = diff_r / max(albedo_d_r, eps) where hit
        eps = 0.001
        expected_r = np.where(hit_bool, diff_r / np.maximum(albedo_d_r, eps), diff_r).astype(np.float16)
        actual_r = target_tensor[0].numpy()
        np.testing.assert_allclose(actual_r, expected_r, atol=1e-2)

    def test_missing_target_skipped(self):
        """Dataset skips input files that have no matching target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an orphan input file (empty — won't be loaded, just found)
            orphan_dir = os.path.join(tmpdir, "orphan")
            os.makedirs(orphan_dir)
            open(os.path.join(orphan_dir, "input.exr"), "w").close()
            with pytest.warns(UserWarning, match="Missing target"):
                ds = ExrDataset(tmpdir)
            assert len(ds) == 0

    def test_flat_naming_discovery(self, synthetic_data_dir):
        """Dataset discovers flat-named pairs (scene_ev+0.5_vp3_input.exr)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy an existing pair as flat-named files
            ds_orig = ExrDataset(synthetic_data_dir)
            assert len(ds_orig) >= 1
            import shutil
            src_input, src_target = ds_orig.pairs[0]
            shutil.copy2(src_input, os.path.join(tmpdir, "scene_ev+0.5_vp3_input.exr"))
            shutil.copy2(src_target, os.path.join(tmpdir, "scene_ev+0.5_vp3_target.exr"))
            ds = ExrDataset(tmpdir)
            assert len(ds) == 1

    def test_flat_naming_missing_target(self):
        """Dataset warns for flat-named input with no matching target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "scene_ev0.0_vp0_input.exr"), "w").close()
            with pytest.warns(UserWarning, match="Missing target"):
                ds = ExrDataset(tmpdir)
            assert len(ds) == 0

    def test_with_transform(self, synthetic_data_dir):
        transform = Compose([RandomCrop(32)])
        ds = ExrDataset(synthetic_data_dir, transform=transform)
        input_tensor, target_tensor, albedo_d, albedo_s, hit_mask = ds[0]
        assert input_tensor.shape == (19, 32, 32)
        assert target_tensor.shape == (6, 32, 32)
        assert albedo_d.shape == (3, 32, 32)
        assert albedo_s.shape == (3, 32, 32)
        assert hit_mask.shape == (1, 32, 32)


class TestRandomCrop:
    def test_output_size(self):
        inp = torch.randn(13, 48, 64, dtype=torch.float16)
        tgt = torch.randn(3, 48, 64, dtype=torch.float16)
        crop = RandomCrop(32)
        inp_c, tgt_c = crop((inp, tgt))
        assert inp_c.shape == (13, 32, 32)
        assert tgt_c.shape == (3, 32, 32)

    def test_spatial_alignment(self):
        """Input and target are cropped from the same spatial location."""
        # Use a unique pattern per pixel so we can verify alignment
        h, w = 48, 64
        coords = torch.arange(h * w, dtype=torch.float16).reshape(1, h, w)
        inp = coords.expand(13, -1, -1).clone()
        tgt = coords.expand(3, -1, -1).clone()

        crop = RandomCrop(16)
        inp_c, tgt_c = crop((inp, tgt))

        # All channels should see the same spatial patch
        torch.testing.assert_close(inp_c[0], tgt_c[0])

    def test_too_small_raises(self):
        inp = torch.randn(13, 16, 16, dtype=torch.float16)
        tgt = torch.randn(3, 16, 16, dtype=torch.float16)
        crop = RandomCrop(32)
        with pytest.raises(ValueError, match="smaller than crop"):
            crop((inp, tgt))


class TestRandomHorizontalFlip:
    def test_motion_x_negated_on_flip(self):
        """Motion vector X (channel 11) is negated when the image is flipped."""
        torch.manual_seed(0)  # Ensure flip happens (p=1.0)
        inp = torch.ones(13, 4, 4, dtype=torch.float16)
        inp[11] = 2.0  # motion X
        tgt = torch.ones(3, 4, 4, dtype=torch.float16)

        flip = RandomHorizontalFlip(p=1.0)
        inp_f, _ = flip((inp, tgt))

        # Motion X should be negated
        assert (inp_f[11] == -2.0).all()
        # Other channels should remain 1.0
        assert (inp_f[0] == 1.0).all()

    def test_no_flip_preserves_data(self):
        inp = torch.ones(13, 4, 4, dtype=torch.float16) * 3.0
        tgt = torch.ones(3, 4, 4, dtype=torch.float16) * 5.0

        flip = RandomHorizontalFlip(p=0.0)
        inp_f, tgt_f = flip((inp, tgt))

        torch.testing.assert_close(inp_f, inp)
        torch.testing.assert_close(tgt_f, tgt)


class TestCompose:
    def test_chains_transforms(self):
        inp = torch.randn(13, 48, 64, dtype=torch.float16)
        tgt = torch.randn(3, 48, 64, dtype=torch.float16)

        transform = Compose([RandomCrop(32), RandomHorizontalFlip(p=0.0)])
        inp_t, tgt_t = transform((inp, tgt))
        assert inp_t.shape == (13, 32, 32)
        assert tgt_t.shape == (3, 32, 32)
