"""Tests for safetensors support in evaluate.py (Phase S4)."""

import pytest
import torch
from safetensors.torch import save_file

from deni_train.data.splits import detect_data_format
from deni_train.evaluate import _get_val_indices_from_files


class TestDetectDataFormat:
    def test_empty_dir_returns_exr(self, tmp_path):
        assert detect_data_format(str(tmp_path)) == "exr"

    def test_safetensors_detected(self, tmp_path):
        (tmp_path / "sample.safetensors").write_bytes(b"dummy")
        assert detect_data_format(str(tmp_path)) == "safetensors"

    def test_exr_only_returns_exr(self, tmp_path):
        (tmp_path / "sample_input.exr").write_bytes(b"dummy")
        assert detect_data_format(str(tmp_path)) == "exr"

    def test_nested_safetensors_detected(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "sample.safetensors").write_bytes(b"dummy")
        assert detect_data_format(str(tmp_path)) == "safetensors"


class TestGetValIndicesFromFiles:
    def _make_files(self, scene_counts: dict[str, int]) -> list[str]:
        files = []
        for scene, count in scene_counts.items():
            for i in range(count):
                hex_id = f"{i:08x}"
                files.append(f"data/{scene}_{hex_id}.safetensors")
        return sorted(files)

    def test_returns_val_indices(self):
        files = self._make_files({"SceneA": 20, "SceneB": 10})
        val_idx = _get_val_indices_from_files(files)
        assert len(val_idx) > 0
        assert all(0 <= i < len(files) for i in val_idx)

    def test_val_is_subset(self):
        files = self._make_files({"SceneA": 20})
        val_idx = _get_val_indices_from_files(files)
        assert len(val_idx) < len(files)
