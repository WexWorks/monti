"""Tests for prune_viewpoints.py."""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from prune_viewpoints import prune_viewpoints, _load_skipped_files


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _make_skipped_file(tmp_dir, filename, scene, skipped_entries):
    """Create a skipped-viewpoints JSON file in monti_datagen's format."""
    path = os.path.join(tmp_dir, filename)
    _write_json(path, {"scene": scene, "skipped": skipped_entries})
    return path


class TestLoadSkippedFiles:
    """Tests for _load_skipped_files()."""

    def test_single_file(self, tmp_dir):
        path = _make_skipped_file(tmp_dir, "s1.json", "Sponza", [
            {"viewpoint_id": "vp0", "reason": "near_black", "detail": 0.0001},
        ])
        entries = _load_skipped_files([path])
        assert len(entries) == 1
        assert entries[0]["scene"] == "Sponza"
        assert entries[0]["viewpoint_id"] == "vp0"

    def test_multiple_files(self, tmp_dir):
        p1 = _make_skipped_file(tmp_dir, "s1.json", "Sponza", [
            {"viewpoint_id": "vp0", "reason": "near_black", "detail": 0.0001},
        ])
        p2 = _make_skipped_file(tmp_dir, "s2.json", "Helmet", [
            {"viewpoint_id": "vp1", "reason": "excessive_nan", "detail": 0.05},
            {"viewpoint_id": "vp3", "reason": "near_black", "detail": 0.0002},
        ])
        entries = _load_skipped_files([p1, p2])
        assert len(entries) == 3
        scenes = {e["scene"] for e in entries}
        assert scenes == {"Sponza", "Helmet"}

    def test_empty_skipped_list(self, tmp_dir):
        path = _make_skipped_file(tmp_dir, "s1.json", "Sponza", [])
        entries = _load_skipped_files([path])
        assert entries == []


class TestPruneViewpoints:
    """Tests for prune_viewpoints()."""

    def test_empty_skipped_files(self, tmp_dir):
        """No-op when all skipped files have empty lists."""
        path = _make_skipped_file(tmp_dir, "s1.json", "Sponza", [])
        result = prune_viewpoints([path], tmp_dir)
        assert result == {}

    def test_removes_matching_viewpoints(self, tmp_dir):
        """Removes viewpoints whose id matches skipped entries."""
        vp_dir = os.path.join(tmp_dir, "viewpoints")
        os.makedirs(vp_dir)

        viewpoints = [
            {"id": "vp0", "position": [0, 0, 0], "target": [1, 0, 0]},
            {"id": "vp1", "position": [1, 0, 0], "target": [2, 0, 0]},
            {"id": "vp2", "position": [2, 0, 0], "target": [3, 0, 0]},
        ]
        _write_json(os.path.join(vp_dir, "Sponza.json"), viewpoints)

        skipped_path = _make_skipped_file(tmp_dir, "s1.json", "Sponza", [
            {"viewpoint_id": "vp1", "reason": "near_black", "detail": 0.0001},
        ])

        result = prune_viewpoints([skipped_path], vp_dir)

        assert result == {"Sponza": 1}
        remaining = _read_json(os.path.join(vp_dir, "Sponza.json"))
        assert len(remaining) == 2
        assert [vp["id"] for vp in remaining] == ["vp0", "vp2"]

    def test_removes_multiple_viewpoints_same_scene(self, tmp_dir):
        """Removes multiple viewpoints from the same scene."""
        vp_dir = os.path.join(tmp_dir, "viewpoints")
        os.makedirs(vp_dir)

        viewpoints = [
            {"id": f"vp{i}", "position": [i, 0, 0], "target": [i + 1, 0, 0]}
            for i in range(5)
        ]
        _write_json(os.path.join(vp_dir, "Helmet.json"), viewpoints)

        skipped_path = _make_skipped_file(tmp_dir, "s1.json", "Helmet", [
            {"viewpoint_id": "vp0", "reason": "near_black", "detail": 0.0002},
            {"viewpoint_id": "vp3", "reason": "excessive_nan", "detail": 0.05},
        ])

        result = prune_viewpoints([skipped_path], vp_dir)

        assert result == {"Helmet": 2}
        remaining = _read_json(os.path.join(vp_dir, "Helmet.json"))
        assert len(remaining) == 3
        assert [vp["id"] for vp in remaining] == ["vp1", "vp2", "vp4"]

    def test_multiple_scenes_single_file(self, tmp_dir):
        """prune_viewpoints works when entries span scenes from separate files."""
        vp_dir = os.path.join(tmp_dir, "viewpoints")
        os.makedirs(vp_dir)

        for scene in ["SceneA", "SceneB"]:
            viewpoints = [
                {"id": f"vp{i}", "position": [i, 0, 0], "target": [i + 1, 0, 0]}
                for i in range(3)
            ]
            _write_json(os.path.join(vp_dir, f"{scene}.json"), viewpoints)

        p1 = _make_skipped_file(tmp_dir, "s1.json", "SceneA", [
            {"viewpoint_id": "vp0", "reason": "near_black", "detail": 0.0001},
        ])
        p2 = _make_skipped_file(tmp_dir, "s2.json", "SceneB", [
            {"viewpoint_id": "vp2", "reason": "excessive_nan", "detail": 0.01},
        ])

        result = prune_viewpoints([p1, p2], vp_dir)

        assert result == {"SceneA": 1, "SceneB": 1}
        assert len(_read_json(os.path.join(vp_dir, "SceneA.json"))) == 2
        assert len(_read_json(os.path.join(vp_dir, "SceneB.json"))) == 2

    def test_multiple_files_same_scene(self, tmp_dir):
        """Merges entries from multiple files for the same scene."""
        vp_dir = os.path.join(tmp_dir, "viewpoints")
        os.makedirs(vp_dir)

        viewpoints = [
            {"id": f"vp{i}", "position": [i, 0, 0], "target": [i + 1, 0, 0]}
            for i in range(5)
        ]
        _write_json(os.path.join(vp_dir, "Sponza.json"), viewpoints)

        p1 = _make_skipped_file(tmp_dir, "s1.json", "Sponza", [
            {"viewpoint_id": "vp0", "reason": "near_black", "detail": 0.0001},
        ])
        p2 = _make_skipped_file(tmp_dir, "s2.json", "Sponza", [
            {"viewpoint_id": "vp3", "reason": "near_black", "detail": 0.0002},
        ])

        result = prune_viewpoints([p1, p2], vp_dir)

        assert result == {"Sponza": 2}
        remaining = _read_json(os.path.join(vp_dir, "Sponza.json"))
        assert [vp["id"] for vp in remaining] == ["vp1", "vp2", "vp4"]

    def test_dry_run_does_not_modify(self, tmp_dir):
        """Dry run reports changes but does not modify files."""
        vp_dir = os.path.join(tmp_dir, "viewpoints")
        os.makedirs(vp_dir)

        viewpoints = [
            {"id": "vp0", "position": [0, 0, 0], "target": [1, 0, 0]},
            {"id": "vp1", "position": [1, 0, 0], "target": [2, 0, 0]},
        ]
        _write_json(os.path.join(vp_dir, "Test.json"), viewpoints)

        skipped_path = _make_skipped_file(tmp_dir, "s1.json", "Test", [
            {"viewpoint_id": "vp0", "reason": "near_black", "detail": 0.0001},
        ])

        result = prune_viewpoints([skipped_path], vp_dir, dry_run=True)

        assert result == {"Test": 1}
        # File should be unmodified
        remaining = _read_json(os.path.join(vp_dir, "Test.json"))
        assert len(remaining) == 2

    def test_missing_scene_json(self, tmp_dir):
        """Warns when a scene's viewpoint JSON doesn't exist."""
        vp_dir = os.path.join(tmp_dir, "viewpoints")
        os.makedirs(vp_dir)

        skipped_path = _make_skipped_file(tmp_dir, "s1.json", "Missing", [
            {"viewpoint_id": "vp0", "reason": "near_black", "detail": 0.0001},
        ])

        result = prune_viewpoints([skipped_path], vp_dir)
        assert result == {}

    def test_no_matching_ids(self, tmp_dir):
        """No changes when viewpoint IDs don't match."""
        vp_dir = os.path.join(tmp_dir, "viewpoints")
        os.makedirs(vp_dir)

        viewpoints = [
            {"id": "vp0", "position": [0, 0, 0], "target": [1, 0, 0]},
        ]
        _write_json(os.path.join(vp_dir, "Test.json"), viewpoints)

        skipped_path = _make_skipped_file(tmp_dir, "s1.json", "Test", [
            {"viewpoint_id": "vp_nonexistent", "reason": "near_black", "detail": 0.0001},
        ])

        result = prune_viewpoints([skipped_path], vp_dir)
        assert result == {}
        remaining = _read_json(os.path.join(vp_dir, "Test.json"))
        assert len(remaining) == 1

    def test_preserves_viewpoint_data(self, tmp_dir):
        """All fields of kept viewpoints are preserved exactly."""
        vp_dir = os.path.join(tmp_dir, "viewpoints")
        os.makedirs(vp_dir)

        viewpoints = [
            {"id": "vp0", "position": [1, 2, 3], "target": [4, 5, 6],
             "fov": 60, "environment": "env.exr", "lights": "lights.json"},
            {"id": "vp1", "position": [7, 8, 9], "target": [10, 11, 12]},
        ]
        _write_json(os.path.join(vp_dir, "Test.json"), viewpoints)

        skipped_path = _make_skipped_file(tmp_dir, "s1.json", "Test", [
            {"viewpoint_id": "vp1", "reason": "near_black", "detail": 0.0001},
        ])

        prune_viewpoints([skipped_path], vp_dir)

        remaining = _read_json(os.path.join(vp_dir, "Test.json"))
        assert len(remaining) == 1
        assert remaining[0] == viewpoints[0]
