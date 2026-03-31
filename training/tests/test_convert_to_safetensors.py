"""Tests for convert_to_safetensors.py conversion script."""

import os
import tempfile

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from deni_train.data.exr_dataset import ExrDataset

# Import from the scripts directory
from scripts.convert_to_safetensors import (
    _build_tensors,
    _discover_exr_pairs,
    _output_path_for_pair,
    convert,
)


@pytest.fixture
def synthetic_data_dir():
    """Generate a small synthetic dataset in a temp directory."""
    from scripts.generate_synthetic_data import generate

    with tempfile.TemporaryDirectory() as tmpdir:
        generate(tmpdir, num_pairs=3, width=64, height=48, seed=456)
        yield tmpdir


@pytest.fixture
def converted_dir(synthetic_data_dir):
    """Convert synthetic EXR data to safetensors and return both directories."""
    with tempfile.TemporaryDirectory() as outdir:
        success = convert(synthetic_data_dir, outdir, verify=False,
                          delete_exr=False, jobs=1)
        assert success
        yield synthetic_data_dir, outdir


class TestDiscoverPairs:
    def test_finds_directory_based_pairs(self, synthetic_data_dir):
        pairs = _discover_exr_pairs(synthetic_data_dir)
        assert len(pairs) == 3

    def test_finds_flat_named_pairs(self, synthetic_data_dir):
        """Verify discovery works for flat-named files too."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import shutil
            pairs = _discover_exr_pairs(synthetic_data_dir)
            src_input, src_target = pairs[0]
            shutil.copy2(src_input, os.path.join(tmpdir, "scene_hash_input.exr"))
            shutil.copy2(src_target, os.path.join(tmpdir, "scene_hash_target.exr"))
            flat_pairs = _discover_exr_pairs(tmpdir)
            assert len(flat_pairs) == 1

    def test_empty_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pairs = _discover_exr_pairs(tmpdir)
            assert len(pairs) == 0


class TestBuildTensors:
    def test_input_shape_and_dtype(self, synthetic_data_dir):
        pairs = _discover_exr_pairs(synthetic_data_dir)
        input_tensor, _ = _build_tensors(*pairs[0])
        assert input_tensor.shape == (19, 48, 64)
        assert input_tensor.dtype == torch.float16

    def test_target_shape_and_dtype(self, synthetic_data_dir):
        pairs = _discover_exr_pairs(synthetic_data_dir)
        _, target_tensor = _build_tensors(*pairs[0])
        assert target_tensor.shape == (7, 48, 64)
        assert target_tensor.dtype == torch.float16

    def test_matches_exr_dataset(self, synthetic_data_dir):
        """Tensors from _build_tensors must exactly match ExrDataset.__getitem__."""
        ds = ExrDataset(synthetic_data_dir)
        pairs = _discover_exr_pairs(synthetic_data_dir)
        for i in range(len(ds)):
            exr_input, exr_target, _, _, exr_hit = ds[i]
            st_input, st_target = _build_tensors(*pairs[i])
            assert torch.equal(st_input, exr_input), f"Input mismatch at index {i}"
            exr_target_full = torch.cat([exr_target, exr_hit], dim=0)
            assert torch.equal(st_target, exr_target_full), f"Target mismatch at index {i}"


class TestOutputPath:
    def test_directory_based_naming(self):
        path = _output_path_for_pair(
            "/data/SceneName/var_001/input.exr", "/data", "/out"
        )
        assert path == os.path.join("/out", "SceneName", "var_001.safetensors")

    def test_flat_naming(self):
        path = _output_path_for_pair(
            "/data/SceneName_abc123_input.exr", "/data", "/out"
        )
        assert path == os.path.join("/out", "SceneName_abc123.safetensors")


class TestConvert:
    def test_creates_correct_number_of_files(self, converted_dir):
        _, outdir = converted_dir
        st_files = [f for f in os.listdir(outdir) if f.endswith(".safetensors")]
        # Directory-based pairs produce files in subdirs
        all_st = []
        for root, _, files in os.walk(outdir):
            all_st.extend(f for f in files if f.endswith(".safetensors"))
        assert len(all_st) == 3

    def test_safetensors_contain_expected_keys(self, converted_dir):
        _, outdir = converted_dir
        for root, _, files in os.walk(outdir):
            for f in files:
                if not f.endswith(".safetensors"):
                    continue
                tensors = load_file(os.path.join(root, f))
                assert "input" in tensors
                assert "target" in tensors

    def test_safetensors_shapes_and_dtypes(self, converted_dir):
        _, outdir = converted_dir
        for root, _, files in os.walk(outdir):
            for f in files:
                if not f.endswith(".safetensors"):
                    continue
                tensors = load_file(os.path.join(root, f))
                assert tensors["input"].shape == (19, 48, 64)
                assert tensors["target"].shape == (7, 48, 64)
                assert tensors["input"].dtype == torch.float16
                assert tensors["target"].dtype == torch.float16

    def test_converted_matches_exr_source(self, converted_dir):
        """Round-trip: converted safetensors must exactly match EXR source tensors."""
        exr_dir, outdir = converted_dir
        ds = ExrDataset(exr_dir)
        pairs = _discover_exr_pairs(exr_dir)

        for i in range(len(ds)):
            exr_input, exr_target, _, _, exr_hit = ds[i]
            out_path = _output_path_for_pair(pairs[i][0], exr_dir, outdir)
            tensors = load_file(out_path)

            assert torch.equal(tensors["input"], exr_input), \
                f"Input mismatch for pair {i}"
            exr_target_full = torch.cat([exr_target, exr_hit], dim=0)
            assert torch.equal(tensors["target"], exr_target_full), \
                f"Target mismatch for pair {i}"

    def test_verify_mode_passes(self, synthetic_data_dir):
        with tempfile.TemporaryDirectory() as outdir:
            success = convert(synthetic_data_dir, outdir, verify=True,
                              delete_exr=False, jobs=1)
            assert success

    def test_returns_false_on_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.TemporaryDirectory() as outdir:
                success = convert(tmpdir, outdir, verify=False,
                                  delete_exr=False, jobs=1)
                assert not success

    def test_no_nan_in_converted_tensors(self, converted_dir):
        _, outdir = converted_dir
        for root, _, files in os.walk(outdir):
            for f in files:
                if not f.endswith(".safetensors"):
                    continue
                tensors = load_file(os.path.join(root, f))
                assert not torch.isnan(tensors["input"]).any(), f"NaN in input: {f}"
                assert not torch.isnan(tensors["target"]).any(), f"NaN in target: {f}"
