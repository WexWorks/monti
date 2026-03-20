"""Remove invalid viewpoints from training data.

Scans rendered training EXR files, identifies images that fail validation
checks (currently: near-black detection), and:
  1. Moves invalid EXR file pairs to a sibling directory for manual inspection.
  2. Removes corresponding viewpoints from source viewpoint JSON files.
  3. Logs removed viewpoints for auditability and potential restoration.

Usage:
    python scripts/remove_invalid_viewpoints.py \
        --training-data <dir> \
        --viewpoints-dir <dir> \
        [--threshold <float>] \
        [--dark-fraction <float>] \
        [--dry-run]
"""

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    import OpenEXR
    import Imath
except ImportError:
    print("Error: OpenEXR and Imath packages required.")
    sys.exit(1)

_DEFAULT_THRESHOLD = 0.001
_DEFAULT_DARK_FRACTION = 0.98
_DEFAULT_NAN_FRACTION = 0.001

_TARGET_RADIANCE_CHANNELS = [
    "diffuse.R", "diffuse.G", "diffuse.B",
    "specular.R", "specular.G", "specular.B",
]

# Rec. 709 luminance coefficients
_LUMA_R = 0.2126
_LUMA_G = 0.7152
_LUMA_B = 0.0722


def _read_exr_channels(path: str, channel_names: list[str]) -> dict[str, np.ndarray]:
    """Read specified channels from an EXR file as float32 arrays.

    Raises ``RuntimeError`` if any requested channel is missing from the file.
    """
    exr = OpenEXR.InputFile(path)
    try:
        header = exr.header()
        available = set(header["channels"].keys())
        missing = [ch for ch in channel_names if ch not in available]
        if missing:
            raise RuntimeError(
                f"{path}: missing EXR channels {missing} "
                f"(available: {sorted(available)})"
            )
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        pt_float = Imath.PixelType(Imath.PixelType.FLOAT)
        result: dict[str, np.ndarray] = {}
        for name in channel_names:
            raw = exr.channel(name, pt_float)
            result[name] = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
        return result
    finally:
        exr.close()


def is_near_black(target_path: str, threshold: float, dark_fraction: float) -> bool:
    """Return True if the target EXR is near-black.

    Loads diffuse + specular radiance channels, computes per-pixel luminance,
    and checks whether the fraction of dark pixels exceeds *dark_fraction*.
    NaN pixels are counted as dark (they carry no useful radiance).
    """
    channels = _read_exr_channels(target_path, _TARGET_RADIANCE_CHANNELS)

    r = channels["diffuse.R"] + channels["specular.R"]
    g = channels["diffuse.G"] + channels["specular.G"]
    b = channels["diffuse.B"] + channels["specular.B"]

    luminance = _LUMA_R * r + _LUMA_G * g + _LUMA_B * b

    total_pixels = luminance.size
    # NaN < threshold is False, so count NaN pixels separately
    dark_pixels = int(np.count_nonzero(luminance < threshold))
    nan_pixels = int(np.count_nonzero(np.isnan(luminance)))
    fraction = (dark_pixels + nan_pixels) / total_pixels

    return fraction >= dark_fraction


def has_excessive_nans(
    target_path: str, nan_fraction: float = _DEFAULT_NAN_FRACTION,
) -> bool:
    """Return True if the target EXR has too many NaN or Inf pixels.

    Checks all radiance channels. A small number of NaN/Inf pixels can occur
    from fireflies or edge cases in path tracing, but a high fraction
    indicates a corrupted render.
    """
    channels = _read_exr_channels(target_path, _TARGET_RADIANCE_CHANNELS)

    total_pixels = 0
    bad_pixels = 0
    for arr in channels.values():
        total_pixels += arr.size
        bad_pixels += int(np.count_nonzero(~np.isfinite(arr)))

    return (bad_pixels / total_pixels) >= nan_fraction


def parse_filename(stem: str) -> tuple[str, str] | None:
    """Extract (scene_name, viewpoint_id) from a file stem.

    Expected format: ``<scene>_<id>_{input,target}``
    The ID is the last segment before the suffix, and the scene name is
    everything before that.

    Returns None if the stem does not match the expected pattern.
    """
    # Strip the input/target suffix
    for suffix in ("_input", "_target"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    else:
        return None

    # The ID is the last underscore-delimited segment
    last_underscore = stem.rfind("_")
    if last_underscore <= 0:
        return None

    scene_name = stem[:last_underscore]
    vp_id = stem[last_underscore + 1 :]
    if not vp_id:
        return None

    return scene_name, vp_id


def _find_target_files(training_data_dir: str) -> list[str]:
    """Return sorted list of *_target.exr files in the training data directory."""
    pattern = os.path.join(training_data_dir, "*_target.exr")
    return sorted(glob.glob(pattern))


def _move_invalid_exrs(
    training_data_dir: str,
    invalid_dir: str,
    scene_name: str,
    vp_id: str,
    dry_run: bool,
) -> list[str]:
    """Move input+target EXR pair for (scene_name, vp_id) to *invalid_dir*.

    Returns the list of files moved (or that would be moved in dry-run mode).
    """
    moved = []
    for suffix in ("input", "target"):
        filename = f"{scene_name}_{vp_id}_{suffix}.exr"
        src = os.path.join(training_data_dir, filename)
        if os.path.isfile(src):
            dst = os.path.join(invalid_dir, filename)
            if not dry_run:
                os.makedirs(invalid_dir, exist_ok=True)
                shutil.move(src, dst)
            moved.append(src)
    return moved


def _remove_viewpoint_from_json(
    viewpoints_dir: str,
    invalid_viewpoints_dir: str,
    scene_name: str,
    vp_id: str,
    dry_run: bool,
) -> dict | None:
    """Remove the viewpoint with *vp_id* from the scene's JSON file.

    The removed entry is appended to the invalid-viewpoints log file.
    Returns the removed entry, or None if not found.
    """
    vp_path = os.path.join(viewpoints_dir, f"{scene_name}.json")
    if not os.path.isfile(vp_path):
        return None

    with open(vp_path, "r") as f:
        viewpoints: list[dict] = json.load(f)

    removed_entry = None
    remaining = []
    for vp in viewpoints:
        if vp.get("id") == vp_id:
            removed_entry = vp
        else:
            remaining.append(vp)

    if removed_entry is None:
        return None

    if not dry_run:
        # Write remaining viewpoints back
        with open(vp_path, "w") as f:
            json.dump(remaining, f, indent=2)
            f.write("\n")

        # Append removed entry to invalid-viewpoints log
        os.makedirs(invalid_viewpoints_dir, exist_ok=True)
        log_path = os.path.join(invalid_viewpoints_dir, f"{scene_name}.json")
        existing_log: list[dict] = []
        if os.path.isfile(log_path):
            with open(log_path, "r") as f:
                existing_log = json.load(f)
        existing_log.append(removed_entry)
        with open(log_path, "w") as f:
            json.dump(existing_log, f, indent=2)
            f.write("\n")

    return removed_entry


def _check_image(target_path: str, threshold: float, dark_fraction: float,
                  nan_fraction: float) -> str | None:
    """Run all validation checks on a target EXR.

    Returns a reason string if the image is invalid, or None if it passes.
    """
    if has_excessive_nans(target_path, nan_fraction):
        return "excessive NaN/Inf"
    if is_near_black(target_path, threshold, dark_fraction):
        return "near-black"
    return None


def run(
    training_data_dir: str,
    viewpoints_dir: str,
    threshold: float = _DEFAULT_THRESHOLD,
    dark_fraction: float = _DEFAULT_DARK_FRACTION,
    nan_fraction: float = _DEFAULT_NAN_FRACTION,
    dry_run: bool = False,
) -> list[tuple[str, str]]:
    """Scan training data and remove invalid viewpoints.

    Returns a list of (scene_name, vp_id) tuples that were removed.
    """
    target_files = _find_target_files(training_data_dir)
    if not target_files:
        print(f"No *_target.exr files found in {training_data_dir}")
        return []

    # Compute sibling directories
    data_parent = os.path.dirname(os.path.normpath(training_data_dir))
    data_basename = os.path.basename(os.path.normpath(training_data_dir))
    invalid_data_dir = os.path.join(data_parent, f"invalid_{data_basename}")

    vp_parent = os.path.dirname(os.path.normpath(viewpoints_dir))
    invalid_vp_dir = os.path.join(vp_parent, "invalid_viewpoints")

    removed: list[tuple[str, str]] = []
    total = len(target_files)

    for i, target_path in enumerate(target_files, 1):
        stem = Path(target_path).stem  # e.g. "scene_name_a3f1c0b2_target"
        parsed = parse_filename(stem)
        if parsed is None:
            print(f"  [{i}/{total}] SKIP (unparseable name): {os.path.basename(target_path)}")
            continue

        scene_name, vp_id = parsed

        reason = _check_image(target_path, threshold, dark_fraction, nan_fraction)
        if reason is None:
            continue

        prefix = "[DRY RUN] " if dry_run else ""
        print(f"  [{i}/{total}] {prefix}INVALID ({reason}): {scene_name} id={vp_id}")

        moved = _move_invalid_exrs(
            training_data_dir, invalid_data_dir, scene_name, vp_id, dry_run,
        )
        for f in moved:
            print(f"    {prefix}Move: {os.path.basename(f)}")

        entry = _remove_viewpoint_from_json(
            viewpoints_dir, invalid_vp_dir, scene_name, vp_id, dry_run,
        )
        if entry is not None:
            print(f"    {prefix}Removed viewpoint from {scene_name}.json")
        else:
            print(f"    WARNING: viewpoint id={vp_id} not found in {scene_name}.json")

        removed.append((scene_name, vp_id))

    action = "would remove" if dry_run else "removed"
    print(f"\nDone: {action} {len(removed)} invalid viewpoint(s) out of {total} checked.")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove invalid viewpoints from training data.",
    )
    parser.add_argument(
        "--training-data", required=True,
        help="Directory containing *_{input,target}.exr training pairs.",
    )
    parser.add_argument(
        "--viewpoints-dir", required=True,
        help="Directory containing <scene>.json viewpoint files.",
    )
    parser.add_argument(
        "--threshold", type=float, default=_DEFAULT_THRESHOLD,
        help=f"Luminance threshold below which a pixel is 'dark' (default: {_DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--dark-fraction", type=float, default=_DEFAULT_DARK_FRACTION,
        help=f"Fraction of dark pixels to classify image as near-black (default: {_DEFAULT_DARK_FRACTION}).",
    )
    parser.add_argument(
        "--nan-fraction", type=float, default=_DEFAULT_NAN_FRACTION,
        help=f"Fraction of NaN/Inf pixels to classify image as corrupted (default: {_DEFAULT_NAN_FRACTION}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be removed without modifying anything.",
    )
    args = parser.parse_args()

    run(
        training_data_dir=args.training_data,
        viewpoints_dir=args.viewpoints_dir,
        threshold=args.threshold,
        dark_fraction=args.dark_fraction,
        nan_fraction=args.nan_fraction,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
