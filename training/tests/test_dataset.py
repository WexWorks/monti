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
        input_tensor, _ = ds[0]
        assert input_tensor.shape == (13, 48, 64)

    def test_target_tensor_shape(self, synthetic_data_dir):
        ds = ExrDataset(synthetic_data_dir)
        _, target_tensor = ds[0]
        assert target_tensor.shape == (3, 48, 64)

    def test_tensor_dtype_fp16(self, synthetic_data_dir):
        ds = ExrDataset(synthetic_data_dir)
        input_tensor, target_tensor = ds[0]
        assert input_tensor.dtype == torch.float16
        assert target_tensor.dtype == torch.float16

    def test_target_is_sum_of_diffuse_and_specular(self, synthetic_data_dir):
        """Verify target = diffuse.RGB + specular.RGB from the target EXR."""
        import OpenEXR
        import Imath

        ds = ExrDataset(synthetic_data_dir)
        _, target_tensor = ds[0]

        # Read raw target EXR to verify summation
        target_path = ds.pairs[0][1]
        exr = OpenEXR.InputFile(target_path)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        width, height = 64, 48

        diff_r = np.frombuffer(exr.channel("diffuse.R", pt), dtype=np.float32).reshape(height, width)
        spec_r = np.frombuffer(exr.channel("specular.R", pt), dtype=np.float32).reshape(height, width)
        expected_r = (diff_r + spec_r).astype(np.float16)
        exr.close()

        actual_r = target_tensor[0].numpy()
        np.testing.assert_allclose(actual_r, expected_r, atol=1e-3)

    def test_missing_target_skipped(self):
        """Dataset skips input files that have no matching target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an orphan input file (empty — won't be loaded, just found)
            open(os.path.join(tmpdir, "frame_000000_input.exr"), "w").close()
            with pytest.warns(UserWarning, match="Missing target"):
                ds = ExrDataset(tmpdir)
            assert len(ds) == 0

    def test_with_transform(self, synthetic_data_dir):
        transform = Compose([RandomCrop(32)])
        ds = ExrDataset(synthetic_data_dir, transform=transform)
        input_tensor, target_tensor = ds[0]
        assert input_tensor.shape == (13, 32, 32)
        assert target_tensor.shape == (3, 32, 32)


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
