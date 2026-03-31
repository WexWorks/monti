"""Tests for safetensors support in train.py (Phase S3)."""

import os
import sys

import pytest
import torch
from safetensors.torch import save_file

from deni_train.train import _build_dataloaders, _Config


def _make_sample(path: str, h: int = 48, w: int = 64):
    """Write a minimal safetensors sample."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(
        {
            "input": torch.randn(19, h, w, dtype=torch.float16),
            "target": torch.randn(7, h, w, dtype=torch.float16),
        },
        path,
    )


class TestBuildDataloadersFormat:
    """Test that _build_dataloaders selects the right dataset class."""

    def _make_cfg(self, data_dir: str, data_format: str = "auto") -> _Config:
        return _Config({
            "data": {
                "data_dir": data_dir,
                "data_format": data_format,
                "crop_size": 16,
                "batch_size": 1,
                "num_workers": 0,
            }
        })

    def test_auto_uses_safetensors(self, tmp_path):
        for i in range(3):
            _make_sample(str(tmp_path / f"SceneA_{i:08x}.safetensors"), h=32, w=32)
        cfg = self._make_cfg(str(tmp_path))
        train_loader, val_loader = _build_dataloaders(cfg)
        assert len(train_loader.dataset) + len(val_loader.dataset) == 3

    def test_force_safetensors(self, tmp_path):
        for i in range(3):
            _make_sample(str(tmp_path / f"SceneA_{i:08x}.safetensors"), h=32, w=32)
        cfg = self._make_cfg(str(tmp_path), data_format="safetensors")
        train_loader, val_loader = _build_dataloaders(cfg)
        assert len(train_loader.dataset) + len(val_loader.dataset) == 3

    def test_empty_safetensors_raises(self, tmp_path):
        cfg = self._make_cfg(str(tmp_path), data_format="safetensors")
        with pytest.raises(RuntimeError, match="No safetensors files"):
            _build_dataloaders(cfg)

    def test_empty_exr_raises(self, tmp_path):
        cfg = self._make_cfg(str(tmp_path), data_format="exr")
        with pytest.raises(RuntimeError, match="No EXR pairs"):
            _build_dataloaders(cfg)


class TestWindowsNumWorkers:
    """Test that safetensors skips the Windows num_workers=0 override."""

    def _make_cfg(self, data_dir: str, data_format: str, num_workers: int = 4) -> _Config:
        return _Config({
            "data": {
                "data_dir": data_dir,
                "data_format": data_format,
                "crop_size": 16,
                "batch_size": 1,
                "num_workers": num_workers,
            }
        })

    def test_safetensors_keeps_num_workers_on_windows(self, tmp_path, monkeypatch):
        """Safetensors should NOT override num_workers on Windows."""
        for i in range(3):
            _make_sample(str(tmp_path / f"SceneA_{i:08x}.safetensors"), h=32, w=32)
        monkeypatch.setattr(sys, "platform", "win32")
        cfg = self._make_cfg(str(tmp_path), data_format="safetensors", num_workers=4)
        train_loader, _ = _build_dataloaders(cfg)
        assert train_loader.num_workers == 4

    def test_exr_format_selected_on_windows(self, tmp_path, monkeypatch):
        """EXR code path is entered when data_format='exr' on Windows."""
        monkeypatch.setattr(sys, "platform", "win32")
        cfg = self._make_cfg(str(tmp_path), data_format="exr", num_workers=4)
        with pytest.raises(RuntimeError, match="No EXR pairs"):
            _build_dataloaders(cfg)

    def test_safetensors_keeps_num_workers_on_linux(self, tmp_path, monkeypatch):
        """Safetensors should keep num_workers on non-Windows too."""
        for i in range(3):
            _make_sample(str(tmp_path / f"SceneA_{i:08x}.safetensors"), h=32, w=32)
        monkeypatch.setattr(sys, "platform", "linux")
        cfg = self._make_cfg(str(tmp_path), data_format="safetensors", num_workers=4)
        train_loader, _ = _build_dataloaders(cfg)
        assert train_loader.num_workers == 4
