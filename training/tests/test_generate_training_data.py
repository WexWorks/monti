"""Tests for generate_training_data.py helper functions."""

import json
import os
import sys
import tempfile

import pytest

# Add scripts/ to path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from generate_training_data import (
    _load_viewpoints,
    _group_viewpoints,
    _check_disk_space,
    _GB_PER_PAIR,
)


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

    def test_samples_to_max_viewpoints(self, tmp_path):
        vps = [{"position": [i, 0, 0], "target": [0, 0, 0]} for i in range(24)]
        vp_file = tmp_path / "scene_a.json"
        vp_file.write_text(json.dumps(vps))

        result = _load_viewpoints(str(tmp_path), "scene_a", 2)
        assert len(result) == 2
        # Verify sampled viewpoints come from the original set
        original_positions = {tuple(vp["position"]) for vp in vps}
        for vp in result:
            assert tuple(vp["position"]) in original_positions

    def test_sampling_is_deterministic(self, tmp_path):
        vps = [{"position": [i, 0, 0], "target": [0, 0, 0]} for i in range(24)]
        vp_file = tmp_path / "scene_a.json"
        vp_file.write_text(json.dumps(vps))

        result1 = _load_viewpoints(str(tmp_path), "scene_a", 2)
        result2 = _load_viewpoints(str(tmp_path), "scene_a", 2)
        assert result1 == result2

    def test_different_scenes_get_different_samples(self, tmp_path):
        vps = [{"position": [i, 0, 0], "target": [0, 0, 0]} for i in range(24)]
        (tmp_path / "scene_a.json").write_text(json.dumps(vps))
        (tmp_path / "scene_b.json").write_text(json.dumps(vps))

        result_a = _load_viewpoints(str(tmp_path), "scene_a", 2)
        result_b = _load_viewpoints(str(tmp_path), "scene_b", 2)
        assert result_a != result_b

    def test_max_viewpoints_larger_than_list(self, tmp_path):
        vps = [{"position": [0, 0, 0], "target": [0, 0, 0]}]
        vp_file = tmp_path / "small.json"
        vp_file.write_text(json.dumps(vps))

        result = _load_viewpoints(str(tmp_path), "small", 10)
        assert len(result) == 1


class TestGroupViewpoints:
    def test_single_group_no_env_no_lights(self):
        vps = [
            {"position": [0, 0, 0], "target": [1, 0, 0]},
            {"position": [1, 0, 0], "target": [0, 0, 0]},
        ]
        groups = _group_viewpoints(vps)
        assert len(groups) == 1
        key = ("", "")
        assert key in groups
        assert len(groups[key]) == 2

    def test_groups_by_environment(self):
        vps = [
            {"position": [0, 0, 0], "target": [1, 0, 0], "environment": "/envs/a.exr"},
            {"position": [1, 0, 0], "target": [0, 0, 0], "environment": "/envs/b.exr"},
            {"position": [2, 0, 0], "target": [0, 0, 0], "environment": "/envs/a.exr"},
        ]
        groups = _group_viewpoints(vps)
        assert len(groups) == 2
        assert len(groups[("/envs/a.exr", "")]) == 2
        assert len(groups[("/envs/b.exr", "")]) == 1

    def test_groups_by_lights(self):
        vps = [
            {"position": [0, 0, 0], "target": [1, 0, 0], "lights": "/rigs/overhead.json"},
            {"position": [1, 0, 0], "target": [0, 0, 0], "lights": "/rigs/kfr.json"},
        ]
        groups = _group_viewpoints(vps)
        assert len(groups) == 2

    def test_preserves_global_indices(self):
        vps = [
            {"position": [0, 0, 0], "target": [1, 0, 0], "environment": "/envs/a.exr"},
            {"position": [1, 0, 0], "target": [0, 0, 0], "environment": "/envs/b.exr"},
            {"position": [2, 0, 0], "target": [0, 0, 0], "environment": "/envs/a.exr"},
        ]
        groups = _group_viewpoints(vps)
        group_a = groups[("/envs/a.exr", "")]
        indices = [idx for idx, _ in group_a]
        assert indices == [0, 2]

    def test_mixed_env_and_lights(self):
        vps = [
            {"position": [0, 0, 0], "target": [1, 0, 0], "environment": "/envs/a.exr"},
            {"position": [1, 0, 0], "target": [0, 0, 0], "lights": "/rigs/o.json"},
            {"position": [2, 0, 0], "target": [0, 0, 0]},
        ]
        groups = _group_viewpoints(vps)
        assert len(groups) == 3


class TestCheckDiskSpace:
    def test_no_warning_when_plenty_of_space(self, tmp_path, capsys):
        """Should not warn or prompt when space is abundant."""
        # 10 frames × 0.037 GB = 0.37 GB — should be fine on any dev machine
        _check_disk_space(str(tmp_path), 10, skip_confirm=True)
        captured = capsys.readouterr()
        assert "WARNING" not in captured.err

    def test_prints_estimates(self, tmp_path, capsys):
        _check_disk_space(str(tmp_path), 100, skip_confirm=True)
        captured = capsys.readouterr()
        assert "Estimated disk:" in captured.out
        assert "Free disk space:" in captured.out
        assert "3.7 GB" in captured.out  # 100 × 0.037


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
        # cornell_box: 2 vp, DamagedHelmet: 1 auto → total 3
        assert "Total frames:    3" in result.stdout

    def test_dry_run_with_env_and_lights_in_viewpoints(self, tmp_path):
        """Viewpoints with different env/lights should show multiple groups."""
        import subprocess

        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        (scenes_dir / "TestModel.glb").write_bytes(b"fake")

        vp_dir = tmp_path / "vp"
        vp_dir.mkdir()
        vps = [
            {"position": [0, 0, 0], "target": [1, 0, 0], "environment": "/envs/a.exr"},
            {"position": [1, 0, 0], "target": [0, 0, 0], "environment": "/envs/a.exr"},
            {"position": [2, 0, 0], "target": [0, 0, 0], "environment": "/envs/b.exr"},
        ]
        (vp_dir / "TestModel.json").write_text(json.dumps(vps))

        script = os.path.join(os.path.dirname(__file__), "..", "scripts",
                              "generate_training_data.py")
        result = subprocess.run(
            [sys.executable, script, "--dry-run", "--yes",
             "--scenes", str(scenes_dir),
             "--viewpoints-dir", str(vp_dir)],
            capture_output=True, text=True)
        assert result.returncode == 0
        assert "Dry Run" in result.stdout
        assert "Total frames:    3" in result.stdout
        assert "2 group(s)" in result.stdout

    def test_dry_run_shows_parallel_jobs(self, tmp_path):
        """--jobs value should appear in configuration output."""
        import subprocess

        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        (scenes_dir / "Test.glb").write_bytes(b"fake")

        vp_dir = tmp_path / "vp"
        vp_dir.mkdir()

        script = os.path.join(os.path.dirname(__file__), "..", "scripts",
                              "generate_training_data.py")
        result = subprocess.run(
            [sys.executable, script, "--dry-run", "--yes",
             "--scenes", str(scenes_dir),
             "--viewpoints-dir", str(vp_dir),
             "--jobs", "5"],
            capture_output=True, text=True)
        assert result.returncode == 0
        assert "Parallel jobs:   5" in result.stdout

