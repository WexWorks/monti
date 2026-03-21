"""Per-scene stratified train/validation splitting for EXR training pairs."""

import os
import re
from collections import defaultdict


# Matches: <SceneName>_<8-hex-id>_input.exr
_FLAT_PATTERN = re.compile(r"^(.+)_([0-9a-f]{8})_(?:input|target)\.exr$", re.IGNORECASE)


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


def stratified_split(pairs: list[tuple[str, str]]) -> tuple[list[int], list[int]]:
    """Split pairs into train/val indices, holding out ~10% per scene.

    Groups pairs by scene name, then for each scene takes the last 10%
    (minimum 1) of sorted pairs as validation. Returns (train_indices,
    val_indices) where indices refer to positions in the input pairs list.
    """
    # Group indices by scene name
    scene_indices: dict[str, list[int]] = defaultdict(list)
    for i, pair in enumerate(pairs):
        name = scene_name_from_pair(pair)
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
