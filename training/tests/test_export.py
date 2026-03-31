"""Tests for F9-3: export scripts and metrics."""

import os
import struct

import numpy as np
import pytest
import torch

from deni_train.models.unet import DeniUNet
from deni_train.utils.metrics import compute_psnr, compute_ssim

# Import export helpers (scripts/ is not a package, so add to path)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from export_weights import write_denimodel, export_onnx


# ---------------------------------------------------------------------------
# .denimodel export tests
# ---------------------------------------------------------------------------

class TestDenimodelExport:
    @pytest.fixture
    def model_and_path(self, tmp_path):
        model = DeniUNet(in_channels=13, out_channels=3, base_channels=16)
        output_path = str(tmp_path / "test.denimodel")
        return model, output_path

    def test_write_and_parse_header(self, model_and_path):
        model, output_path = model_and_path
        write_denimodel(model.state_dict(), output_path)

        with open(output_path, "rb") as f:
            magic = f.read(4)
            assert magic == b"DENI"
            version = struct.unpack("<I", f.read(4))[0]
            assert version == 1
            num_layers = struct.unpack("<I", f.read(4))[0]
            assert num_layers == len(model.state_dict())
            total_bytes = struct.unpack("<I", f.read(4))[0]
            assert total_bytes > 0

    def test_roundtrip_weights_match(self, model_and_path):
        model, output_path = model_and_path
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        write_denimodel(model.state_dict(), output_path)

        # Read back and verify
        loaded = {}
        with open(output_path, "rb") as f:
            f.read(4)  # magic
            f.read(4)  # version
            num_layers = struct.unpack("<I", f.read(4))[0]
            f.read(4)  # total_bytes

            for _ in range(num_layers):
                name_len = struct.unpack("<I", f.read(4))[0]
                name = f.read(name_len).decode("utf-8")
                num_dims = struct.unpack("<I", f.read(4))[0]
                shape = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(num_dims))
                n_elements = 1
                for d in shape:
                    n_elements *= d
                data = np.frombuffer(f.read(n_elements * 4), dtype=np.float32).reshape(shape)
                loaded[name] = torch.from_numpy(data.copy())

        assert set(loaded.keys()) == set(original_state.keys())
        for name in original_state:
            torch.testing.assert_close(
                loaded[name], original_state[name].float(),
                msg=f"Weight mismatch for layer '{name}'"
            )

    def test_file_size_reasonable(self, model_and_path):
        model, output_path = model_and_path
        write_denimodel(model.state_dict(), output_path)
        file_size = os.path.getsize(output_path)
        num_params = sum(p.numel() for p in model.parameters())
        # File should be slightly larger than raw weights (header + per-layer metadata)
        expected_min = num_params * 4  # float32
        assert file_size > expected_min
        assert file_size < expected_min * 2  # not unreasonably large

    def test_layer_count_matches_state_dict(self, model_and_path):
        model, output_path = model_and_path
        write_denimodel(model.state_dict(), output_path)

        with open(output_path, "rb") as f:
            f.read(4)  # magic
            f.read(4)  # version
            num_layers = struct.unpack("<I", f.read(4))[0]

        assert num_layers == len(model.state_dict())


# ---------------------------------------------------------------------------
# ONNX export tests
# ---------------------------------------------------------------------------

class TestOnnxExport:
    def test_onnx_export_runs(self, tmp_path):
        model = DeniUNet(in_channels=19, out_channels=6, base_channels=16)
        onnx_path = str(tmp_path / "test.onnx")
        export_onnx(model, 19, onnx_path)
        assert os.path.exists(onnx_path)
        assert os.path.getsize(onnx_path) > 0


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_psnr_identical_inputs(self):
        x = torch.rand(1, 3, 32, 32) * 2.0
        psnr = compute_psnr(x, x.clone())
        assert psnr >= 80.0  # should be very high (capped at 100)

    def test_psnr_different_inputs(self):
        x = torch.rand(1, 3, 32, 32) * 2.0
        y = torch.rand(1, 3, 32, 32) * 2.0
        psnr = compute_psnr(x, y)
        assert 0.0 < psnr < 80.0

    def test_psnr_closer_is_higher(self):
        target = torch.rand(1, 3, 32, 32) * 2.0
        close = target + torch.randn_like(target) * 0.01
        far = target + torch.randn_like(target) * 0.5
        psnr_close = compute_psnr(close, target)
        psnr_far = compute_psnr(far, target)
        assert psnr_close > psnr_far

    def test_ssim_identical_inputs(self):
        x = torch.rand(1, 3, 32, 32) * 2.0
        ssim = compute_ssim(x, x.clone())
        assert ssim > 0.99

    def test_ssim_range(self):
        x = torch.rand(1, 3, 32, 32) * 2.0
        y = torch.rand(1, 3, 32, 32) * 2.0
        ssim = compute_ssim(x, y)
        assert -1.0 <= ssim <= 1.0

    def test_ssim_closer_is_higher(self):
        target = torch.rand(1, 3, 32, 32) * 2.0
        close = target + torch.randn_like(target) * 0.01
        far = target + torch.randn_like(target) * 0.5
        ssim_close = compute_ssim(close, target)
        ssim_far = compute_ssim(far, target)
        assert ssim_close > ssim_far
