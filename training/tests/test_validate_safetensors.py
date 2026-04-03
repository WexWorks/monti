"""Tests for safetensors support in validate_dataset.py (Phase S5)."""

import os
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

# validate_dataset.py is a script, not a package module — add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from validate_dataset import (
    ValidationResult,
    _detect_data_format,
    validate_safetensors_file,
    validate_dataset,
)


def _make_valid_safetensors(path: str, h: int = 16, w: int = 16) -> None:
    """Create a valid safetensors file with realistic tensor shapes."""
    inp = torch.rand(13, h, w, dtype=torch.float16) * 0.5
    tgt = torch.rand(3, h, w, dtype=torch.float16) * 0.5
    save_file({"input": inp, "target": tgt}, path)


class TestDetectDataFormat:
    def test_empty_dir_returns_exr(self, tmp_path):
        assert _detect_data_format(str(tmp_path)) == "exr"

    def test_safetensors_detected(self, tmp_path):
        _make_valid_safetensors(str(tmp_path / "sample.safetensors"))
        assert _detect_data_format(str(tmp_path)) == "safetensors"

    def test_exr_only_returns_exr(self, tmp_path):
        (tmp_path / "sample_input.exr").write_bytes(b"dummy")
        assert _detect_data_format(str(tmp_path)) == "exr"


class TestValidateSafetensorsFile:
    def test_valid_file(self, tmp_path):
        path = str(tmp_path / "Scene_abcd1234.safetensors")
        _make_valid_safetensors(path)
        result = ValidationResult()
        assert validate_safetensors_file(path, result, str(tmp_path)) is True
        assert len(result.errors) == 0

    def test_missing_input_key(self, tmp_path):
        path = str(tmp_path / "bad.safetensors")
        save_file({"target": torch.rand(3, 8, 8, dtype=torch.float16)}, path)
        result = ValidationResult()
        assert validate_safetensors_file(path, result) is False
        assert any("Missing 'input'" in e for e in result.errors)

    def test_missing_target_key(self, tmp_path):
        path = str(tmp_path / "bad.safetensors")
        save_file({"input": torch.rand(13, 8, 8, dtype=torch.float16)}, path)
        result = ValidationResult()
        assert validate_safetensors_file(path, result) is False
        assert any("Missing 'target'" in e for e in result.errors)

    def test_wrong_input_shape(self, tmp_path):
        path = str(tmp_path / "bad.safetensors")
        save_file({
            "input": torch.rand(10, 8, 8, dtype=torch.float16),
            "target": torch.rand(3, 8, 8, dtype=torch.float16),
        }, path)
        result = ValidationResult()
        assert validate_safetensors_file(path, result) is False
        assert any("Input shape" in e for e in result.errors)

    def test_wrong_target_shape(self, tmp_path):
        path = str(tmp_path / "bad.safetensors")
        save_file({
            "input": torch.rand(13, 8, 8, dtype=torch.float16),
            "target": torch.rand(6, 8, 8, dtype=torch.float16),
        }, path)
        result = ValidationResult()
        assert validate_safetensors_file(path, result) is False
        assert any("Target shape" in e for e in result.errors)

    def test_nan_detected(self, tmp_path):
        path = str(tmp_path / "nan.safetensors")
        inp = torch.rand(13, 8, 8, dtype=torch.float16)
        inp[0, 0, 0] = float("nan")
        save_file({"input": inp, "target": torch.rand(3, 8, 8, dtype=torch.float16)}, path)
        result = ValidationResult()
        assert validate_safetensors_file(path, result) is False
        assert any("NaN" in e for e in result.errors)

    def test_inf_detected(self, tmp_path):
        path = str(tmp_path / "inf.safetensors")
        tgt = torch.rand(3, 8, 8, dtype=torch.float16)
        tgt[0, 0, 0] = float("inf")
        save_file({"input": torch.rand(13, 8, 8, dtype=torch.float16), "target": tgt}, path)
        result = ValidationResult()
        assert validate_safetensors_file(path, result) is False
        assert any("Inf" in e for e in result.errors)

    def test_low_variance_warning(self, tmp_path):
        path = str(tmp_path / "zero.safetensors")
        inp = torch.zeros(13, 8, 8, dtype=torch.float16)
        tgt = torch.rand(3, 8, 8, dtype=torch.float16) * 0.5
        save_file({"input": inp, "target": tgt}, path)
        result = ValidationResult()
        validate_safetensors_file(path, result, str(tmp_path))
        assert any("zero variance" in w for w in result.warnings)

    def test_thumbnail_generated(self, tmp_path):
        path = str(tmp_path / "thumb.safetensors")
        _make_valid_safetensors(path)
        result = ValidationResult()
        validate_safetensors_file(path, result, str(tmp_path))
        assert len(result.thumbnails) == 1
        assert result.thumbnails[0]["name"] == "thumb"

    def test_display_name_strips_extension(self, tmp_path):
        sub = tmp_path / "SceneA"
        sub.mkdir()
        path = str(sub / "var_001.safetensors")
        _make_valid_safetensors(path)
        result = ValidationResult()
        validate_safetensors_file(path, result, str(tmp_path))
        assert result.thumbnails[0]["name"] == "SceneA/var_001"


class TestValidateDatasetSafetensors:
    def test_auto_detect_and_validate(self, tmp_path):
        _make_valid_safetensors(str(tmp_path / "a_00000001.safetensors"))
        _make_valid_safetensors(str(tmp_path / "a_00000002.safetensors"))
        result = validate_dataset(str(tmp_path), gallery_path=str(tmp_path / "g.html"))
        assert result.total_pairs == 2
        assert len(result.errors) == 0
        assert os.path.exists(str(tmp_path / "g.html"))

    def test_force_safetensors_format(self, tmp_path):
        _make_valid_safetensors(str(tmp_path / "x.safetensors"))
        result = validate_dataset(str(tmp_path), data_format="safetensors")
        assert result.total_pairs == 1
        assert len(result.errors) == 0

    def test_max_variations(self, tmp_path):
        scene = tmp_path / "SceneA"
        scene.mkdir()
        for i in range(5):
            _make_valid_safetensors(str(scene / f"SceneA_{i:08x}.safetensors"))
        result = validate_dataset(str(tmp_path), max_variations=2)
        assert result.total_pairs == 2

    def test_empty_dir(self, tmp_path):
        result = validate_dataset(str(tmp_path), data_format="safetensors")
        assert result.total_pairs == 0
