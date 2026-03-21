"""Per-scene stratified train/validation splitting for EXR training pairs."""

import glob
import os
import re
from collections import defaultdict


# Matches: <SceneName>_<8-hex-id>_input.exr
_FLAT_PATTERN = re.compile(r"^(.+)_([0-9a-f]{8})_(?:input|target)\.exr$", re.IGNORECASE)

# Matches: <SceneName>_<8-hex-id>.safetensors
_SAFETENSORS_FLAT_PATTERN = re.compile(r"^(.+)_([0-9a-f]{8})\.safetensors$", re.IGNORECASE)


def scene_name_from_pair(pair: tuple[str, str]) -> str:
    """Extract scene name from an (input_path, target_path) pair.

    Supports flat naming: <SceneName>_<8-hex-id>_input.exr
    Falls back to parent directory name for directory-based naming.
    """
    input_path = pair[0]
    basename = os.path.basename(input_path)

    m = _FLAT_PATTERN.match(basename)
    if m:
        return m.group(1)

    # Directory-based: <scene>/input.exr
    if basename == "input.exr":
        return os.path.basename(os.path.dirname(input_path))

    return "unknown"


def scene_name_from_file(path: str) -> str:
    """Extract scene name from a single safetensors file path.

    Supports flat naming: <SceneName>_<8-hex-id>.safetensors
    Falls back to parent directory name.
    """
    basename = os.path.basename(path)

    m = _SAFETENSORS_FLAT_PATTERN.match(basename)
    if m:
        return m.group(1)

    # Directory-based: <scene>/variation.safetensors
    return os.path.basename(os.path.dirname(path)) or "unknown"


def _stratified_split_by_names(scene_names: list[str]) -> tuple[list[int], list[int]]:
    """Core stratified split: hold out last ~10% per scene (minimum 1)."""
    scene_indices: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(scene_names):
        scene_indices[name].append(i)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for scene in sorted(scene_indices.keys()):
        indices = scene_indices[scene]
        n = len(indices)
        n_val = max(1, n // 10)
        n_train = n - n_val
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:])

    return train_indices, val_indices


def stratified_split(pairs: list[tuple[str, str]]) -> tuple[list[int], list[int]]:
    """Split pairs into train/val indices, holding out ~10% per scene.

    Groups pairs by scene name, then for each scene takes the last 10%
    (minimum 1) of sorted pairs as validation. Returns (train_indices,
    val_indices) where indices refer to positions in the input pairs list.
    """
    scene_names = [scene_name_from_pair(pair) for pair in pairs]
    return _stratified_split_by_names(scene_names)


def stratified_split_files(files: list[str]) -> tuple[list[int], list[int]]:
    """Split safetensors files into train/val indices, holding out ~10% per scene."""
    scene_names = [scene_name_from_file(f) for f in files]
    return _stratified_split_by_names(scene_names)


def detect_data_format(data_dir: str) -> str:
    """Return 'safetensors' if any .safetensors files exist in data_dir, else 'exr'."""
    if glob.glob(os.path.join(data_dir, "**", "*.safetensors"), recursive=True):
        return "safetensors"
    return "exr"
