"""Tests for remove_invalid_viewpoints.py."""

import json
import os
import sys

import numpy as np
import pytest

try:
    import OpenEXR
    import Imath
except ImportError:
    pytest.skip("OpenEXR/Imath not available", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from remove_invalid_viewpoints import (
    _read_exr_channels,
    has_excessive_nans,
    is_near_black,
    parse_filename,
    run,
    _DEFAULT_DARK_FRACTION,
    _DEFAULT_NAN_FRACTION,
    _DEFAULT_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TARGET_CHANNEL_GROUPS = [
    ("diffuse", ["R", "G", "B", "A"], "FLOAT"),
    ("specular", ["R", "G", "B", "A"], "FLOAT"),
]


def _write_target_exr(
    path: str,
    width: int,
    height: int,
    diffuse_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5),
    specular_rgb: tuple[float, float, float] = (0.1, 0.1, 0.1),
) -> None:
    """Write a flat-color target EXR with the given diffuse/specular values."""
    header = OpenEXR.Header(width, height)
    channels = {}
    for prefix, suffixes, ptype in _TARGET_CHANNEL_GROUPS:
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        for s in suffixes:
            channels[f"{prefix}.{s}"] = Imath.Channel(pt)
    header["channels"] = channels

    pixels = width * height
    dr, dg, db = diffuse_rgb
    sr, sg, sb = specular_rgb

    data = {
        "diffuse.R": np.full(pixels, dr, dtype=np.float32).tobytes(),
        "diffuse.G": np.full(pixels, dg, dtype=np.float32).tobytes(),
        "diffuse.B": np.full(pixels, db, dtype=np.float32).tobytes(),
        "diffuse.A": np.ones(pixels, dtype=np.float32).tobytes(),
        "specular.R": np.full(pixels, sr, dtype=np.float32).tobytes(),
        "specular.G": np.full(pixels, sg, dtype=np.float32).tobytes(),
        "specular.B": np.full(pixels, sb, dtype=np.float32).tobytes(),
        "specular.A": np.ones(pixels, dtype=np.float32).tobytes(),
    }

    out = OpenEXR.OutputFile(path, header)
    out.writePixels(data)
    out.close()


def _write_input_exr(path: str, width: int, height: int) -> None:
    """Write a minimal input EXR (only needs to exist, content doesn't matter)."""
    _write_target_exr(path, width, height)


def _make_viewpoints(ids: list[str]) -> list[dict]:
    """Create a minimal viewpoint list with the given IDs."""
    return [
        {"id": vid, "position": [0.0, 0.0, 0.0], "target": [1.0, 0.0, 0.0]}
        for vid in ids
    ]


def _write_target_exr_arrays(
    path: str,
    width: int,
    height: int,
    channel_data: dict[str, np.ndarray],
) -> None:
    """Write a target EXR with arbitrary per-channel float32 data."""
    header = OpenEXR.Header(width, height)
    channels = {}
    for prefix, suffixes, _ in _TARGET_CHANNEL_GROUPS:
        for s in suffixes:
            channels[f"{prefix}.{s}"] = Imath.Channel(
                Imath.PixelType(Imath.PixelType.FLOAT),
            )
    header["channels"] = channels

    pixels = width * height
    data = {}
    for name in channels:
        if name in channel_data:
            data[name] = channel_data[name].astype(np.float32).tobytes()
        else:
            data[name] = np.zeros(pixels, dtype=np.float32).tobytes()

    out = OpenEXR.OutputFile(path, header)
    out.writePixels(data)
    out.close()


# ---------------------------------------------------------------------------
# Tests: parse_filename
# ---------------------------------------------------------------------------


class TestParseFilename:
    def test_basic_target(self):
        assert parse_filename("my_scene_a3f1c0b2_target") == ("my_scene", "a3f1c0b2")

    def test_basic_input(self):
        assert parse_filename("my_scene_a3f1c0b2_input") == ("my_scene", "a3f1c0b2")

    def test_multi_underscore_scene(self):
        assert parse_filename("a_beautiful_game_84fd8ce2_target") == (
            "a_beautiful_game", "84fd8ce2",
        )

    def test_no_suffix_returns_none(self):
        assert parse_filename("my_scene_a3f1c0b2") is None

    def test_no_id_returns_none(self):
        assert parse_filename("_target") is None

    def test_empty_returns_none(self):
        assert parse_filename("") is None


# ---------------------------------------------------------------------------
# Tests: is_near_black
# ---------------------------------------------------------------------------


class TestIsNearBlack:
    def test_bright_image_passes(self, tmp_path):
        path = str(tmp_path / "bright_target.exr")
        _write_target_exr(path, 32, 32, diffuse_rgb=(0.5, 0.5, 0.5))
        assert is_near_black(path, _DEFAULT_THRESHOLD, _DEFAULT_DARK_FRACTION) is False

    def test_black_image_fails(self, tmp_path):
        path = str(tmp_path / "black_target.exr")
        _write_target_exr(
            path, 32, 32,
            diffuse_rgb=(0.0, 0.0, 0.0),
            specular_rgb=(0.0, 0.0, 0.0),
        )
        assert is_near_black(path, _DEFAULT_THRESHOLD, _DEFAULT_DARK_FRACTION) is True

    def test_near_black_with_some_bright_pixels(self, tmp_path):
        """Image with 99% black pixels should still fail at 98% threshold."""
        path = str(tmp_path / "mostly_black_target.exr")
        width, height = 100, 100
        pixels = width * height

        header = OpenEXR.Header(width, height)
        channels = {}
        for prefix, suffixes, _ in _TARGET_CHANNEL_GROUPS:
            for s in suffixes:
                channels[f"{prefix}.{s}"] = Imath.Channel(
                    Imath.PixelType(Imath.PixelType.FLOAT),
                )
        header["channels"] = channels

        # 99% black, 1% bright
        bright_count = pixels // 100
        data_arrays = {}
        for prefix in ("diffuse", "specular"):
            for suffix in ("R", "G", "B"):
                arr = np.zeros(pixels, dtype=np.float32)
                arr[:bright_count] = 1.0
                data_arrays[f"{prefix}.{suffix}"] = arr.tobytes()
            data_arrays[f"{prefix}.A"] = np.ones(pixels, dtype=np.float32).tobytes()

        out = OpenEXR.OutputFile(path, header)
        out.writePixels(data_arrays)
        out.close()

        assert is_near_black(path, _DEFAULT_THRESHOLD, _DEFAULT_DARK_FRACTION) is True

    def test_custom_threshold(self, tmp_path):
        """Dim image passes with low threshold but fails with high threshold."""
        path = str(tmp_path / "dim_target.exr")
        # Luminance = 0.2126*0.01 + 0.7152*0.01 + 0.0722*0.01 ≈ 0.01
        _write_target_exr(
            path, 32, 32,
            diffuse_rgb=(0.01, 0.01, 0.01),
            specular_rgb=(0.0, 0.0, 0.0),
        )
        assert is_near_black(path, 0.001, 0.98) is False
        assert is_near_black(path, 0.1, 0.98) is True

    def test_nan_pixels_count_as_dark(self, tmp_path):
        """NaN pixels should be treated as dark in the near-black check."""
        path = str(tmp_path / "nan_target.exr")
        w, h = 32, 32
        pixels = w * h
        nan_arr = np.full(pixels, np.nan, dtype=np.float32)
        _write_target_exr_arrays(path, w, h, {
            "diffuse.R": nan_arr, "diffuse.G": nan_arr, "diffuse.B": nan_arr,
            "specular.R": nan_arr, "specular.G": nan_arr, "specular.B": nan_arr,
        })
        assert is_near_black(path, _DEFAULT_THRESHOLD, _DEFAULT_DARK_FRACTION) is True


# ---------------------------------------------------------------------------
# Tests: has_excessive_nans
# ---------------------------------------------------------------------------


class TestHasExcessiveNans:
    def test_clean_image_passes(self, tmp_path):
        path = str(tmp_path / "clean_target.exr")
        _write_target_exr(path, 32, 32)
        assert has_excessive_nans(path) is False

    def test_all_nan_fails(self, tmp_path):
        path = str(tmp_path / "nan_target.exr")
        w, h = 32, 32
        pixels = w * h
        nan_arr = np.full(pixels, np.nan, dtype=np.float32)
        _write_target_exr_arrays(path, w, h, {
            "diffuse.R": nan_arr, "diffuse.G": nan_arr, "diffuse.B": nan_arr,
            "specular.R": nan_arr, "specular.G": nan_arr, "specular.B": nan_arr,
        })
        assert has_excessive_nans(path) is True

    def test_all_inf_fails(self, tmp_path):
        path = str(tmp_path / "inf_target.exr")
        w, h = 32, 32
        pixels = w * h
        inf_arr = np.full(pixels, np.inf, dtype=np.float32)
        _write_target_exr_arrays(path, w, h, {
            "diffuse.R": inf_arr, "diffuse.G": inf_arr, "diffuse.B": inf_arr,
        })
        assert has_excessive_nans(path) is True

    def test_few_nans_passes(self, tmp_path):
        """A handful of NaN pixels (< 0.1%) should not trigger removal."""
        path = str(tmp_path / "few_nan_target.exr")
        w, h = 100, 100
        pixels = w * h
        # Put NaN in only 1 pixel of 1 channel out of 6 channels * 10000 pixels
        arr = np.zeros(pixels, dtype=np.float32)
        arr[0] = np.nan
        _write_target_exr_arrays(path, w, h, {"diffuse.R": arr})
        assert has_excessive_nans(path) is False

    def test_custom_nan_fraction(self, tmp_path):
        """With a very strict threshold, even a few NaNs should fail."""
        path = str(tmp_path / "some_nan_target.exr")
        w, h = 10, 10
        pixels = w * h
        # 10 NaN pixels in one channel = 10/600 ≈ 1.7% of total channel-pixels
        arr = np.zeros(pixels, dtype=np.float32)
        arr[:10] = np.nan
        _write_target_exr_arrays(path, w, h, {"diffuse.R": arr})
        # Default 0.1% threshold: 10/600 > 0.001 → fails
        assert has_excessive_nans(path, nan_fraction=0.001) is True
        # 5% threshold: 10/600 < 0.05 → passes
        assert has_excessive_nans(path, nan_fraction=0.05) is False


# ---------------------------------------------------------------------------
# Tests: _read_exr_channels (error handling)
# ---------------------------------------------------------------------------


class TestReadExrChannels:
    def test_missing_channel_raises(self, tmp_path):
        """Requesting a channel that doesn't exist should raise RuntimeError."""
        path = str(tmp_path / "target.exr")
        _write_target_exr(path, 16, 16)
        with pytest.raises(RuntimeError, match="missing EXR channels"):
            _read_exr_channels(path, ["nonexistent.channel"])

    def test_valid_channels_succeed(self, tmp_path):
        path = str(tmp_path / "target.exr")
        _write_target_exr(path, 16, 16)
        result = _read_exr_channels(path, ["diffuse.R", "specular.G"])
        assert "diffuse.R" in result
        assert "specular.G" in result
        assert result["diffuse.R"].shape == (16, 16)


# ---------------------------------------------------------------------------
# Tests: run (integration)
# ---------------------------------------------------------------------------


class TestRun:
    """Integration tests for the full run() pipeline."""

    @pytest.fixture
    def setup_dirs(self, tmp_path):
        """Create training_data/ and viewpoints/ with a mix of bright and dark pairs."""
        data_dir = tmp_path / "training_data"
        data_dir.mkdir()
        vp_dir = tmp_path / "viewpoints"
        vp_dir.mkdir()

        width, height = 16, 16

        # Bright pair
        _write_target_exr(str(data_dir / "scene_aaa11111_target.exr"), width, height)
        _write_input_exr(str(data_dir / "scene_aaa11111_input.exr"), width, height)

        # Dark pair
        _write_target_exr(
            str(data_dir / "scene_bbb22222_target.exr"), width, height,
            diffuse_rgb=(0.0, 0.0, 0.0), specular_rgb=(0.0, 0.0, 0.0),
        )
        _write_input_exr(str(data_dir / "scene_bbb22222_input.exr"), width, height)

        # Another bright pair
        _write_target_exr(str(data_dir / "scene_ccc33333_target.exr"), width, height)
        _write_input_exr(str(data_dir / "scene_ccc33333_input.exr"), width, height)

        # Viewpoints JSON
        vps = _make_viewpoints(["aaa11111", "bbb22222", "ccc33333"])
        (vp_dir / "scene.json").write_text(json.dumps(vps, indent=2))

        return data_dir, vp_dir

    def test_removes_dark_viewpoint(self, tmp_path, setup_dirs):
        data_dir, vp_dir = setup_dirs

        removed = run(str(data_dir), str(vp_dir))

        assert len(removed) == 1
        assert removed[0] == ("scene", "bbb22222")

        # Dark EXRs should be moved
        invalid_dir = tmp_path / "invalid_training_data"
        assert (invalid_dir / "scene_bbb22222_target.exr").exists()
        assert (invalid_dir / "scene_bbb22222_input.exr").exists()

        # Original dark EXRs should be gone
        assert not (data_dir / "scene_bbb22222_target.exr").exists()
        assert not (data_dir / "scene_bbb22222_input.exr").exists()

        # Bright EXRs should still be there
        assert (data_dir / "scene_aaa11111_target.exr").exists()
        assert (data_dir / "scene_ccc33333_target.exr").exists()

    def test_viewpoint_json_updated(self, tmp_path, setup_dirs):
        data_dir, vp_dir = setup_dirs

        run(str(data_dir), str(vp_dir))

        # Remaining viewpoints should exclude bbb22222
        with open(vp_dir / "scene.json") as f:
            remaining = json.load(f)
        ids = [vp["id"] for vp in remaining]
        assert "bbb22222" not in ids
        assert "aaa11111" in ids
        assert "ccc33333" in ids

    def test_invalid_viewpoints_logged(self, tmp_path, setup_dirs):
        data_dir, vp_dir = setup_dirs

        run(str(data_dir), str(vp_dir))

        log_path = tmp_path / "invalid_viewpoints" / "scene.json"
        assert log_path.exists()
        with open(log_path) as f:
            logged = json.load(f)
        assert len(logged) == 1
        assert logged[0]["id"] == "bbb22222"

    def test_dry_run_does_not_modify(self, tmp_path, setup_dirs):
        data_dir, vp_dir = setup_dirs

        removed = run(str(data_dir), str(vp_dir), dry_run=True)

        assert len(removed) == 1

        # Nothing should have moved
        assert (data_dir / "scene_bbb22222_target.exr").exists()
        assert (data_dir / "scene_bbb22222_input.exr").exists()

        # Viewpoints unchanged
        with open(vp_dir / "scene.json") as f:
            vps = json.load(f)
        assert len(vps) == 3

        # No invalid dirs created
        assert not (tmp_path / "invalid_training_data").exists()
        assert not (tmp_path / "invalid_viewpoints").exists()

    def test_no_target_files(self, tmp_path):
        data_dir = tmp_path / "empty_data"
        data_dir.mkdir()
        vp_dir = tmp_path / "viewpoints"
        vp_dir.mkdir()

        removed = run(str(data_dir), str(vp_dir))
        assert removed == []

    def test_appends_to_existing_log(self, tmp_path, setup_dirs):
        data_dir, vp_dir = setup_dirs

        # Pre-populate invalid_viewpoints log
        invalid_vp_dir = tmp_path / "invalid_viewpoints"
        invalid_vp_dir.mkdir()
        existing_entry = {"id": "old00000", "position": [0, 0, 0], "target": [1, 0, 0]}
        (invalid_vp_dir / "scene.json").write_text(json.dumps([existing_entry]))

        run(str(data_dir), str(vp_dir))

        with open(invalid_vp_dir / "scene.json") as f:
            logged = json.load(f)
        assert len(logged) == 2
        assert logged[0]["id"] == "old00000"
        assert logged[1]["id"] == "bbb22222"

    def test_multi_scene(self, tmp_path):
        """Files from different scenes are handled correctly."""
        data_dir = tmp_path / "training_data"
        data_dir.mkdir()
        vp_dir = tmp_path / "viewpoints"
        vp_dir.mkdir()

        w, h = 16, 16

        # Scene A: dark
        _write_target_exr(
            str(data_dir / "scene_a_aaaa0001_target.exr"), w, h,
            diffuse_rgb=(0.0, 0.0, 0.0), specular_rgb=(0.0, 0.0, 0.0),
        )
        _write_input_exr(str(data_dir / "scene_a_aaaa0001_input.exr"), w, h)
        (vp_dir / "scene_a.json").write_text(
            json.dumps(_make_viewpoints(["aaaa0001"])),
        )

        # Scene B: bright
        _write_target_exr(str(data_dir / "scene_b_bbbb0001_target.exr"), w, h)
        _write_input_exr(str(data_dir / "scene_b_bbbb0001_input.exr"), w, h)
        (vp_dir / "scene_b.json").write_text(
            json.dumps(_make_viewpoints(["bbbb0001"])),
        )

        removed = run(str(data_dir), str(vp_dir))

        assert len(removed) == 1
        assert removed[0] == ("scene_a", "aaaa0001")

        # Scene A viewpoint removed
        with open(vp_dir / "scene_a.json") as f:
            assert json.load(f) == []

        # Scene B viewpoint untouched
        with open(vp_dir / "scene_b.json") as f:
            vps = json.load(f)
        assert len(vps) == 1
        assert vps[0]["id"] == "bbbb0001"

    def test_removes_nan_viewpoint(self, tmp_path):
        """Integration: NaN-heavy images should be removed."""
        data_dir = tmp_path / "training_data"
        data_dir.mkdir()
        vp_dir = tmp_path / "viewpoints"
        vp_dir.mkdir()

        w, h = 16, 16
        pixels = w * h

        # Bright pair (valid)
        _write_target_exr(str(data_dir / "scene_aaa11111_target.exr"), w, h)
        _write_input_exr(str(data_dir / "scene_aaa11111_input.exr"), w, h)

        # NaN pair (invalid)
        nan_arr = np.full(pixels, np.nan, dtype=np.float32)
        _write_target_exr_arrays(
            str(data_dir / "scene_bbb22222_target.exr"), w, h,
            {"diffuse.R": nan_arr, "diffuse.G": nan_arr, "diffuse.B": nan_arr,
             "specular.R": nan_arr, "specular.G": nan_arr, "specular.B": nan_arr},
        )
        _write_input_exr(str(data_dir / "scene_bbb22222_input.exr"), w, h)

        vps = _make_viewpoints(["aaa11111", "bbb22222"])
        (vp_dir / "scene.json").write_text(json.dumps(vps, indent=2))

        removed = run(str(data_dir), str(vp_dir))

        assert len(removed) == 1
        assert removed[0] == ("scene", "bbb22222")

        # NaN EXRs moved
        invalid_dir = tmp_path / "invalid_training_data"
        assert (invalid_dir / "scene_bbb22222_target.exr").exists()

        # Valid pair untouched
        assert (data_dir / "scene_aaa11111_target.exr").exists()
