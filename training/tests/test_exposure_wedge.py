"""Tests for exposure wedge generation in generate_training_data.py."""

import os
import sys
import tempfile

import numpy as np
import pytest

try:
    import OpenEXR
    import Imath
except ImportError:
    pytest.skip("OpenEXR not installed", allow_module_level=True)

# Add scripts/ to path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from generate_training_data import (
    _clamp_fp16_chromaticity,
    _read_exr_all_channels,
    _write_exr,
    apply_exposure_wedge,
    _FP16_MAX,
    _INPUT_RADIANCE_CHANNELS,
)


def _make_test_exr(path: str, channels: dict[str, np.ndarray], width: int, height: int):
    """Create a minimal EXR file with given channels."""
    header = OpenEXR.Header(width, height)
    ch_defs = {}
    for name in channels:
        ch_defs[name] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header["channels"] = ch_defs

    out = OpenEXR.OutputFile(path, header)
    try:
        data = {name: arr.astype(np.float32).tobytes() for name, arr in channels.items()}
        out.writePixels(data)
    finally:
        out.close()


def _make_input_exr(path: str, width: int, height: int,
                    diffuse_val: float = 1.0, specular_val: float = 0.5,
                    normal_val: float = 0.0, depth_val: float = 10.0):
    """Create a synthetic input EXR with all 21 channels."""
    channels = {}
    for c in ("R", "G", "B"):
        channels[f"diffuse.{c}"] = np.full((height, width), diffuse_val, dtype=np.float32)
    channels["diffuse.A"] = np.ones((height, width), dtype=np.float32)
    for c in ("R", "G", "B"):
        channels[f"specular.{c}"] = np.full((height, width), specular_val, dtype=np.float32)
    channels["specular.A"] = np.ones((height, width), dtype=np.float32)
    for c in ("X", "Y", "Z", "W"):
        channels[f"normal.{c}"] = np.full((height, width), normal_val, dtype=np.float32)
    channels["depth.Z"] = np.full((height, width), depth_val, dtype=np.float32)
    for c in ("X", "Y"):
        channels[f"motion.{c}"] = np.zeros((height, width), dtype=np.float32)
    for c in ("R", "G", "B"):
        channels[f"albedo_d.{c}"] = np.full((height, width), 0.8, dtype=np.float32)
    for c in ("R", "G", "B"):
        channels[f"albedo_s.{c}"] = np.full((height, width), 0.04, dtype=np.float32)
    _make_test_exr(path, channels, width, height)


def _make_target_exr(path: str, width: int, height: int,
                     diffuse_val: float = 1.0, specular_val: float = 0.5):
    """Create a synthetic target EXR with diffuse + specular RGBA channels."""
    channels = {}
    for c in ("R", "G", "B"):
        channels[f"diffuse.{c}"] = np.full((height, width), diffuse_val, dtype=np.float32)
    channels["diffuse.A"] = np.ones((height, width), dtype=np.float32)
    for c in ("R", "G", "B"):
        channels[f"specular.{c}"] = np.full((height, width), specular_val, dtype=np.float32)
    channels["specular.A"] = np.ones((height, width), dtype=np.float32)
    _make_test_exr(path, channels, width, height)


class TestExposureWedge:
    def test_ev0_preserves_values(self, tmp_path):
        """Offset 0 should produce identical radiance values."""
        w, h = 4, 4
        input_path = str(tmp_path / "input.exr")
        target_path = str(tmp_path / "target.exr")
        _make_input_exr(input_path, w, h, diffuse_val=0.5, specular_val=0.3)
        _make_target_exr(target_path, w, h, diffuse_val=0.5, specular_val=0.3)

        output_dir = str(tmp_path / "out")
        os.makedirs(output_dir)
        apply_exposure_wedge(input_path, target_path, output_dir, "test_vp0", [0])

        # Read back ev+0 files
        out_input = os.path.join(output_dir, "test_vp0_ev+0_input.exr")
        out_target = os.path.join(output_dir, "test_vp0_ev+0_target.exr")
        assert os.path.exists(out_input)
        assert os.path.exists(out_target)

        in_ch, _ = _read_exr_all_channels(out_input)
        np.testing.assert_allclose(in_ch["diffuse.R"], 0.5, atol=1e-5)
        np.testing.assert_allclose(in_ch["specular.R"], 0.3, atol=1e-5)

        tgt_ch, _ = _read_exr_all_channels(out_target)
        np.testing.assert_allclose(tgt_ch["diffuse.R"], 0.5, atol=1e-5)

    def test_ev_plus1_doubles_radiance(self, tmp_path):
        """Offset +1 should multiply radiance by 2."""
        w, h = 4, 4
        input_path = str(tmp_path / "input.exr")
        target_path = str(tmp_path / "target.exr")
        _make_input_exr(input_path, w, h, diffuse_val=1.0, specular_val=0.5)
        _make_target_exr(target_path, w, h, diffuse_val=1.0, specular_val=0.5)

        output_dir = str(tmp_path / "out")
        os.makedirs(output_dir)
        apply_exposure_wedge(input_path, target_path, output_dir, "test", [1])

        in_ch, _ = _read_exr_all_channels(
            os.path.join(output_dir, "test_ev+1_input.exr"))
        np.testing.assert_allclose(in_ch["diffuse.R"], 2.0, atol=1e-4)
        np.testing.assert_allclose(in_ch["specular.G"], 1.0, atol=1e-4)

        tgt_ch, _ = _read_exr_all_channels(
            os.path.join(output_dir, "test_ev+1_target.exr"))
        np.testing.assert_allclose(tgt_ch["diffuse.R"], 2.0, atol=1e-4)
        np.testing.assert_allclose(tgt_ch["specular.G"], 1.0, atol=1e-4)

    def test_ev_minus2_quarters_radiance(self, tmp_path):
        """Offset -2 should multiply radiance by 0.25."""
        w, h = 4, 4
        input_path = str(tmp_path / "input.exr")
        target_path = str(tmp_path / "target.exr")
        _make_input_exr(input_path, w, h, diffuse_val=4.0, specular_val=2.0)
        _make_target_exr(target_path, w, h, diffuse_val=4.0, specular_val=2.0)

        output_dir = str(tmp_path / "out")
        os.makedirs(output_dir)
        apply_exposure_wedge(input_path, target_path, output_dir, "test", [-2])

        in_ch, _ = _read_exr_all_channels(
            os.path.join(output_dir, "test_ev-2_input.exr"))
        np.testing.assert_allclose(in_ch["diffuse.R"], 1.0, atol=1e-4)
        np.testing.assert_allclose(in_ch["specular.R"], 0.5, atol=1e-4)

    def test_guide_channels_unchanged(self, tmp_path):
        """Non-radiance channels (normals, depth, motion, albedo) are unchanged."""
        w, h = 4, 4
        input_path = str(tmp_path / "input.exr")
        target_path = str(tmp_path / "target.exr")
        _make_input_exr(input_path, w, h, normal_val=0.577, depth_val=42.0)
        _make_target_exr(target_path, w, h)

        output_dir = str(tmp_path / "out")
        os.makedirs(output_dir)
        apply_exposure_wedge(input_path, target_path, output_dir, "test", [2])

        in_ch, _ = _read_exr_all_channels(
            os.path.join(output_dir, "test_ev+2_input.exr"))
        np.testing.assert_allclose(in_ch["normal.X"], 0.577, atol=1e-5)
        np.testing.assert_allclose(in_ch["depth.Z"], 42.0, atol=1e-4)
        np.testing.assert_allclose(in_ch["motion.X"], 0.0, atol=1e-6)
        np.testing.assert_allclose(in_ch["albedo_d.R"], 0.8, atol=1e-5)

    def test_five_steps_produces_five_pairs(self, tmp_path):
        """Steps=5 should produce 5 input/target pairs with correct names."""
        w, h = 4, 4
        input_path = str(tmp_path / "input.exr")
        target_path = str(tmp_path / "target.exr")
        _make_input_exr(input_path, w, h)
        _make_target_exr(target_path, w, h)

        output_dir = str(tmp_path / "out")
        os.makedirs(output_dir)
        offsets = [-2, -1, 0, 1, 2]
        created = apply_exposure_wedge(
            input_path, target_path, output_dir, "scene_vp0", offsets)

        assert len(created) == 10  # 5 input + 5 target
        for offset in offsets:
            assert os.path.exists(
                os.path.join(output_dir, f"scene_vp0_ev{offset:+d}_input.exr"))
            assert os.path.exists(
                os.path.join(output_dir, f"scene_vp0_ev{offset:+d}_target.exr"))

    def test_three_steps(self, tmp_path):
        """Steps=3 produces offsets [-1, 0, +1]."""
        w, h = 4, 4
        input_path = str(tmp_path / "input.exr")
        target_path = str(tmp_path / "target.exr")
        _make_input_exr(input_path, w, h)
        _make_target_exr(target_path, w, h)

        output_dir = str(tmp_path / "out")
        os.makedirs(output_dir)
        offsets = [-1, 0, 1]
        created = apply_exposure_wedge(
            input_path, target_path, output_dir, "scene_vp0", offsets)

        assert len(created) == 6  # 3 input + 3 target

    def test_fp16_overflow_protection(self, tmp_path):
        """Values near FP16 max with +2 EV (4x) should not produce Inf."""
        w, h = 4, 4
        input_path = str(tmp_path / "input.exr")
        target_path = str(tmp_path / "target.exr")
        _make_input_exr(input_path, w, h, diffuse_val=60000.0, specular_val=0.0)
        _make_target_exr(target_path, w, h, diffuse_val=60000.0, specular_val=0.0)

        output_dir = str(tmp_path / "out")
        os.makedirs(output_dir)
        apply_exposure_wedge(input_path, target_path, output_dir, "test", [2])

        in_ch, _ = _read_exr_all_channels(
            os.path.join(output_dir, "test_ev+2_input.exr"))
        for ch_name in _INPUT_RADIANCE_CHANNELS:
            assert np.isfinite(in_ch[ch_name]).all(), \
                f"Non-finite values in {ch_name} after +2 EV scaling"
            assert (in_ch[ch_name] <= _FP16_MAX).all(), \
                f"Values exceed FP16 max in {ch_name}"

    def test_fp16_clamp_preserves_chromaticity(self, tmp_path):
        """FP16 clamping should preserve R:G:B ratios."""
        w, h = 4, 4
        channels = {}
        # R=40000, G=20000, B=10000 → ratio 4:2:1
        channels["diffuse.R"] = np.full((h, w), 40000.0, dtype=np.float32)
        channels["diffuse.G"] = np.full((h, w), 20000.0, dtype=np.float32)
        channels["diffuse.B"] = np.full((h, w), 10000.0, dtype=np.float32)
        channels["specular.R"] = np.zeros((h, w), dtype=np.float32)
        channels["specular.G"] = np.zeros((h, w), dtype=np.float32)
        channels["specular.B"] = np.zeros((h, w), dtype=np.float32)

        # Simulate +1 EV (2x scale)
        for ch_name in _INPUT_RADIANCE_CHANNELS:
            channels[ch_name] = channels[ch_name] * 2.0

        _clamp_fp16_chromaticity(channels, _INPUT_RADIANCE_CHANNELS)

        r = channels["diffuse.R"][0, 0]
        g = channels["diffuse.G"][0, 0]
        b = channels["diffuse.B"][0, 0]

        # All should be finite and within FP16 range
        assert r <= _FP16_MAX
        # Ratios should be preserved: R/G ≈ 2, R/B ≈ 4
        assert abs(r / g - 2.0) < 0.01
        assert abs(r / b - 4.0) < 0.01


class TestExrRoundTrip:
    def test_write_read_roundtrip(self, tmp_path):
        """Write and read back an EXR, verify channel data survives."""
        w, h = 8, 8
        channels = {
            "R": np.random.rand(h, w).astype(np.float32),
            "G": np.random.rand(h, w).astype(np.float32),
            "B": np.random.rand(h, w).astype(np.float32),
        }

        # Create source EXR
        src_path = str(tmp_path / "src.exr")
        _make_test_exr(src_path, channels, w, h)

        # Read and write back
        read_channels, header = _read_exr_all_channels(src_path)
        out_path = str(tmp_path / "out.exr")
        _write_exr(out_path, read_channels, header)

        # Read final output
        final_channels, _ = _read_exr_all_channels(out_path)
        for name in channels:
            np.testing.assert_allclose(
                final_channels[name], channels[name], atol=1e-6,
                err_msg=f"Channel {name} round-trip mismatch")
