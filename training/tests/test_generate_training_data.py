"""Tests for generate_training_data.py helper functions."""

import json
import os
import sys
import tempfile

import pytest

# Add scripts/ to path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from generate_training_data import (
    _format_exposure,
    _load_viewpoints,
    _count_viewpoints_per_scene,
    _check_disk_space,
    _EXPOSURES,
    _GB_PER_PAIR,
)


class TestFormatExposure:
    def test_positive(self):
        assert _format_exposure(1.0) == "+1.0"
        assert _format_exposure(0.5) == "+0.5"

    def test_negative(self):
        assert _format_exposure(-1.0) == "-1.0"
        assert _format_exposure(-0.5) == "-0.5"

    def test_zero(self):
        assert _format_exposure(0.0) == "0.0"


class TestLoadViewpoints:
    def test_file_not_found_returns_none(self, tmp_path):
        result = _load_viewpoints(str(tmp_path), "nonexistent", None)
        assert result is None

    def test_loads_all_viewpoints(self, tmp_path):
        vps = [
            {"position": [1, 2, 3], "target": [0, 0, 0]},
            {"position": [4, 5, 6], "target": [0, 0, 0]},
            {"position": [7, 8, 9], "target": [0, 0, 0]},
        ]
        vp_file = tmp_path / "test_scene.json"
        vp_file.write_text(json.dumps(vps))

        result = _load_viewpoints(str(tmp_path), "test_scene", None)
        assert result is not None
        assert len(result) == 3
        assert result[0]["position"] == [1, 2, 3]

    def test_truncates_to_max_viewpoints(self, tmp_path):
        vps = [{"position": [i, 0, 0], "target": [0, 0, 0]} for i in range(24)]
        vp_file = tmp_path / "scene_a.json"
        vp_file.write_text(json.dumps(vps))

        result = _load_viewpoints(str(tmp_path), "scene_a", 2)
        assert len(result) == 2
        assert result[0]["position"] == [0, 0, 0]
        assert result[1]["position"] == [1, 0, 0]

    def test_max_viewpoints_larger_than_list(self, tmp_path):
        vps = [{"position": [0, 0, 0], "target": [0, 0, 0]}]
        vp_file = tmp_path / "small.json"
        vp_file.write_text(json.dumps(vps))

        result = _load_viewpoints(str(tmp_path), "small", 10)
        assert len(result) == 1


class TestCountViewpointsPerScene:
    def test_mixed_scenes(self, tmp_path):
        """Scenes with and without viewpoint files."""
        # Scene with viewpoint file
        vps = [{"position": [i, 0, 0], "target": [0, 0, 0]} for i in range(5)]
        (tmp_path / "scene_a.json").write_text(json.dumps(vps))

        scenes = [("scene_a", "scene_a.glb"), ("scene_b", "scene_b.glb")]
        counts, data = _count_viewpoints_per_scene(
            scenes, str(tmp_path), None)

        assert counts["scene_a"] == 5
        assert counts["scene_b"] == 1
        assert data["scene_a"] is not None
        assert data["scene_b"] is None

    def test_with_max_viewpoints(self, tmp_path):
        vps = [{"position": [i, 0, 0], "target": [0, 0, 0]} for i in range(24)]
        (tmp_path / "scene_a.json").write_text(json.dumps(vps))

        scenes = [("scene_a", "scene_a.glb")]
        counts, data = _count_viewpoints_per_scene(
            scenes, str(tmp_path), 3)

        assert counts["scene_a"] == 3
        assert len(data["scene_a"]) == 3

    def test_total_frames_calculation(self, tmp_path):
        """Verify the total frames formula used in generate_training_data."""
        vps_a = [{"position": [i, 0, 0], "target": [0, 0, 0]} for i in range(24)]
        vps_b = [{"position": [i, 0, 0], "target": [0, 0, 0]} for i in range(24)]
        (tmp_path / "scene_a.json").write_text(json.dumps(vps_a))
        (tmp_path / "scene_b.json").write_text(json.dumps(vps_b))

        scenes = [
            ("scene_a", "scene_a.glb"),
            ("scene_b", "scene_b.glb"),
            ("scene_c", "scene_c.glb"),  # no viewpoint file
        ]
        counts, _ = _count_viewpoints_per_scene(scenes, str(tmp_path), 2)

        total = sum(counts[name] * len(_EXPOSURES) for name, _ in scenes)
        # 2 vp × 5 exp + 2 vp × 5 exp + 1 auto × 5 exp = 25
        assert total == 25


class TestCheckDiskSpace:
    def test_no_warning_when_plenty_of_space(self, tmp_path, capsys):
        """Should not warn or prompt when space is abundant."""
        # 10 frames × 0.15 GB = 1.5 GB — should be fine on any dev machine
        _check_disk_space(str(tmp_path), 10, auto_yes=True)
        captured = capsys.readouterr()
        assert "WARNING" not in captured.err

    def test_prints_estimates(self, tmp_path, capsys):
        _check_disk_space(str(tmp_path), 100, auto_yes=True)
        captured = capsys.readouterr()
        assert "Estimated disk:" in captured.out
        assert "Free disk space:" in captured.out
        assert "15.0 GB" in captured.out  # 100 × 0.15


class TestDryRunIntegration:
    """Integration tests using subprocess to run the script with --dry-run."""

    def test_dry_run_with_no_scenes(self, tmp_path):
        """--dry-run should exit with error when no scenes exist."""
        import subprocess
        script = os.path.join(os.path.dirname(__file__), "..", "scripts",
                              "generate_training_data.py")
        result = subprocess.run(
            [sys.executable, script, "--dry-run", "--yes",
             "--scenes", str(tmp_path / "empty_scenes"),
             "--viewpoints-dir", str(tmp_path / "empty_vp")],
            capture_output=True, text=True)
        assert result.returncode != 0
        assert "No scene files found" in result.stderr

    def test_dry_run_with_fake_scenes(self, tmp_path):
        """--dry-run should succeed with fake scene files (doesn't need monti_datagen)."""
        import subprocess

        # Create fake scene files
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        (scenes_dir / "cornell_box.glb").write_bytes(b"fake")
        (scenes_dir / "DamagedHelmet.glb").write_bytes(b"fake")

        # Create a viewpoint file for one scene
        vp_dir = tmp_path / "vp"
        vp_dir.mkdir()
        vps = [{"position": [i, 0, 0], "target": [0, 0, 0]} for i in range(5)]
        (vp_dir / "cornell_box.json").write_text(json.dumps(vps))

        script = os.path.join(os.path.dirname(__file__), "..", "scripts",
                              "generate_training_data.py")
        result = subprocess.run(
            [sys.executable, script, "--dry-run", "--yes",
             "--scenes", str(scenes_dir),
             "--viewpoints-dir", str(vp_dir),
             "--max-viewpoints", "2"],
            capture_output=True, text=True)
        assert result.returncode == 0
        assert "Dry Run" in result.stdout
        # cornell_box: 2 vp × 5 exp = 10, damaged_helmet: 1 auto × 5 = 5 → total 15
        assert "Total frames:    15" in result.stdout
