"""Tests for SafetensorsDataset loader."""

import os
import tempfile

import pytest
import torch

from deni_train.data.exr_dataset import ExrDataset
from deni_train.data.safetensors_dataset import SafetensorsDataset
from deni_train.data.transforms import Compose, RandomCrop
from scripts.convert_to_safetensors import convert


@pytest.fixture
def exr_and_st_dirs():
    """Generate synthetic EXR data and convert to safetensors."""
    from scripts.generate_synthetic_data import generate

    with tempfile.TemporaryDirectory() as exr_dir:
        generate(exr_dir, num_pairs=3, width=64, height=48, seed=789)
        with tempfile.TemporaryDirectory() as st_dir:
            success = convert(exr_dir, st_dir, verify=False,
                              delete_exr=False, jobs=1)
            assert success
            yield exr_dir, st_dir


class TestSafetensorsDataset:
    def test_finds_all_files(self, exr_and_st_dirs):
        _, st_dir = exr_and_st_dirs
        ds = SafetensorsDataset(st_dir)
        assert len(ds) == 3

    def test_input_shape_and_dtype(self, exr_and_st_dirs):
        _, st_dir = exr_and_st_dirs
        ds = SafetensorsDataset(st_dir)
        input_tensor, _, _, _ = ds[0]
        assert input_tensor.shape == (19, 48, 64)
        assert input_tensor.dtype == torch.float16

    def test_target_shape_and_dtype(self, exr_and_st_dirs):
        _, st_dir = exr_and_st_dirs
        ds = SafetensorsDataset(st_dir)
        _, target_tensor, _, _ = ds[0]
        assert target_tensor.shape == (6, 48, 64)
        assert target_tensor.dtype == torch.float16

    def test_matches_exr_dataset(self, exr_and_st_dirs):
        """Safetensors dataset must return identical tensors to EXR dataset."""
        exr_dir, st_dir = exr_and_st_dirs
        exr_ds = ExrDataset(exr_dir)
        st_ds = SafetensorsDataset(st_dir)
        assert len(exr_ds) == len(st_ds)

        for i in range(len(exr_ds)):
            exr_input, exr_target, _, _ = exr_ds[i]
            st_input, st_target, _, _ = st_ds[i]
            assert torch.equal(st_input, exr_input), f"Input mismatch at index {i}"
            assert torch.equal(st_target, exr_target), f"Target mismatch at index {i}"

    def test_with_transform(self, exr_and_st_dirs):
        _, st_dir = exr_and_st_dirs
        transform = Compose([RandomCrop(32)])
        ds = SafetensorsDataset(st_dir, transform=transform)
        input_tensor, target_tensor, _, _ = ds[0]
        assert input_tensor.shape == (19, 32, 32)
        assert target_tensor.shape == (6, 32, 32)

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = SafetensorsDataset(tmpdir)
            assert len(ds) == 0

    def test_files_attribute_exposed(self, exr_and_st_dirs):
        """The files list must be accessible for scene name extraction."""
        _, st_dir = exr_and_st_dirs
        ds = SafetensorsDataset(st_dir)
        assert len(ds.files) == 3
        assert all(f.endswith(".safetensors") for f in ds.files)
