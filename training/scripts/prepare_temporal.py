"""Temporal training data crop extractor.

Groups frames by path_id, builds sliding temporal windows, and outputs
(W, C, H, W) tensors where W is the window size. All frames in a window
are cropped at the same spatial position for temporal consistency.

Input EXR format (from generate_training_data.py / monti_datagen):
    *_input.exr:  19 named channels (diffuse, specular, normals, depth, motion, albedo)
    *_target.exr:  6 named channels (diffuse.RGB, specular.RGB — high-SPP reference)

Output format:
    input:  float16, (W, 19, crop_size, crop_size)
    target: float16, (W,  6, crop_size, crop_size)

Usage:
    python scripts/prepare_temporal.py \\
        --input-dir ../training_data/ \\
        --output-dir ../training_data_temporal_st/ \\
        --window 16 --stride 8 --crops 4 --crop-size 384 \\
        --workers 4
"""

import argparse
import glob
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from safetensors.torch import save_file

import OpenEXR
import Imath

# EXR channel constants (previously in exr_dataset.py)
_INPUT_CHANNELS = [
    ("diffuse.R", "diffuse.G", "diffuse.B"),
    ("specular.R", "specular.G", "specular.B"),
    ("normal.X", "normal.Y", "normal.Z"),
    ("normal.W",),
    ("depth.Z",),
    ("motion.X", "motion.Y"),
    ("albedo_d.R", "albedo_d.G", "albedo_d.B"),
    ("albedo_s.R", "albedo_s.G", "albedo_s.B"),
]
_INPUT_CHANNEL_NAMES = [name for group in _INPUT_CHANNELS for name in group]
_TARGET_DIFFUSE = ("diffuse.R", "diffuse.G", "diffuse.B")
_TARGET_SPECULAR = ("specular.R", "specular.G", "specular.B")
_DEMOD_EPS = 0.001


def _read_exr_channels(path: str, channel_names: list[str]) -> dict[str, np.ndarray]:
    """Read specified channels from an EXR file as float32 numpy arrays."""
    exr = OpenEXR.InputFile(path)
    try:
        header = exr.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        result = {}
        pt_float = Imath.PixelType(Imath.PixelType.FLOAT)
        for name in channel_names:
            if name not in header["channels"]:
                raise KeyError(f"Channel '{name}' not found in {path}")
            raw = exr.channel(name, pt_float)
            arr = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
            result[name] = arr
    finally:
        exr.close()
    return result

# Depth sentinel for background (miss) pixels, matching kSentinelDepth in
# constants.glsl.  Background pixels have this exact value in channel 10.
_SENTINEL_DEPTH = 1e4

# Minimum fraction of pixels that must hit geometry for a crop to be kept.
# Geometry is detected by depth below the sentinel (input channel 10).
# Crops below this threshold are mostly background/sky and add no useful
# denoising signal.
_MIN_COVERAGE = 0.3

# Over-sample factor: attempt this many candidate crops per requested crop to
# compensate for rejections by the coverage check.
_OVERSAMPLE_FACTOR = 3

# Regex for parsing EXR filenames produced by generate_training_data.py.
# Pattern: {scene}_{path_id}_{frame}_input.exr
_FNAME_RE = re.compile(r"^(.+)_([0-9a-f]{8})_(\d{4})_input\.exr$")


def _load_exr_pair(
    input_exr: str, target_exr: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load an EXR pair and return (input, target) float16 tensors.

    Preprocessing applied:
      - Reads 19 input channels as float32
      - Demodulates diffuse irradiance (ch 0-2) by albedo_d (ch 13-15)
      - Demodulates specular irradiance (ch 3-5) by albedo_s (ch 16-18)
      - Clips to float16 range [-65504.0, 65504.0] and casts to float16
      - Target (6 ch) demodulated using same albedo from input EXR

    Returns:
        input:  float16 (19, H, W)
        target: float16 (6,  H, W)
    """
    input_data = _read_exr_channels(input_exr, _INPUT_CHANNEL_NAMES)
    input_arrays = np.stack(
        [input_data[name] for name in _INPUT_CHANNEL_NAMES], axis=0
    )  # (19, H, W) float32

    albedo_d = input_arrays[13:16]
    albedo_s = input_arrays[16:19]

    input_arrays[0:3] = input_arrays[0:3] / np.maximum(albedo_d, _DEMOD_EPS)
    input_arrays[3:6] = input_arrays[3:6] / np.maximum(albedo_s, _DEMOD_EPS)
    np.clip(input_arrays, -65504.0, 65504.0, out=input_arrays)
    input_tensor = torch.from_numpy(input_arrays).to(torch.float16)

    target_channel_names = list(_TARGET_DIFFUSE) + list(_TARGET_SPECULAR)
    target_data = _read_exr_channels(target_exr, target_channel_names)

    target_d = np.stack([target_data[n] for n in _TARGET_DIFFUSE], axis=0)
    target_s = np.stack([target_data[n] for n in _TARGET_SPECULAR], axis=0)

    target_d = target_d / np.maximum(albedo_d, _DEMOD_EPS)
    target_s = target_s / np.maximum(albedo_s, _DEMOD_EPS)
    np.clip(target_d, -65504.0, 65504.0, out=target_d)
    np.clip(target_s, -65504.0, 65504.0, out=target_s)

    target_arrays = np.concatenate([target_d, target_s], axis=0)  # (6, H, W)
    target_tensor = torch.from_numpy(target_arrays).to(torch.float16)

    return input_tensor, target_tensor


def _windows(frames: list[int], window: int, stride: int) -> list[list[int]]:
    """Build sliding windows of frame indices."""
    return [frames[i:i + window] for i in range(0, len(frames) - window + 1, stride)]


def _process_temporal_window(
    input_dir: str,
    scene_dir: str,
    path_id: str,
    frame_numbers: list[int],
    frame_paths: dict[int, str],
    output_dir: str,
    n_crops: int,
    crop_size: int,
    min_coverage: float = _MIN_COVERAGE,
) -> tuple[int, int, int]:
    """Extract crops from one temporal window (W consecutive frames).

    All frames in the window are cropped at the same spatial position for
    temporal consistency. Output tensors have shape (W, C, H, W).

    Returns (crops_saved, crops_discarded, bytes_written).
    """
    window_size = len(frame_numbers)
    window_start = frame_numbers[0]

    # Load all frames in the window
    all_inp = []
    all_tgt = []
    for frame_num in frame_numbers:
        inp_exr = frame_paths[frame_num]
        tgt_exr = inp_exr[: -len("_input.exr")] + "_target.exr"
        inp, tgt = _load_exr_pair(inp_exr, tgt_exr)
        all_inp.append(inp)
        all_tgt.append(tgt)

    _, h, w = all_inp[0].shape

    # Stack into (W, C, H, W) for slicing
    stacked_inp = torch.stack(all_inp)  # (W, 19, H, W)
    stacked_tgt = torch.stack(all_tgt)  # (W, 6, H, W)

    # Output base: {scene_dir}/{path_id}_{window_start:04d}
    out_base = os.path.join(output_dir, scene_dir, f"{path_id}_{window_start:04d}")
    os.makedirs(os.path.join(output_dir, scene_dir), exist_ok=True)

    # If source is already crop-sized or smaller, save as-is
    if h <= crop_size and w <= crop_size:
        # Coverage check on the first frame's normals
        depth = stacked_inp[0, 10]  # linear depth of first frame
        coverage = (depth < _SENTINEL_DEPTH).float().mean().item()
        if coverage < min_coverage:
            return 0, 1, 0
        out_path = f"{out_base}_crop0.safetensors"
        save_file({
            "input": stacked_inp.contiguous(),
            "target": stacked_tgt.contiguous(),
        }, out_path)
        return 1, 0, os.path.getsize(out_path)

    # Deterministic RNG seeded by path_id + window start
    rng = random.Random(f"{path_id}_{window_start}")

    max_attempts = n_crops * _OVERSAMPLE_FACTOR
    candidates = [
        (rng.randint(0, w - crop_size), rng.randint(0, h - crop_size))
        for _ in range(max_attempts)
    ]

    saved = 0
    discarded = 0
    bytes_written = 0

    for cx, cy in candidates:
        if saved >= n_crops:
            break

        # Crop all frames at the same position
        crop_inp = stacked_inp[:, :, cy:cy + crop_size, cx:cx + crop_size].contiguous()
        crop_tgt = stacked_tgt[:, :, cy:cy + crop_size, cx:cx + crop_size].contiguous()

        # Coverage check on the first frame's normals
        depth = crop_inp[0, 10]  # linear depth of first frame
        coverage = (depth < _SENTINEL_DEPTH).float().mean().item()
        if coverage < min_coverage:
            discarded += 1
            continue

        out_path = f"{out_base}_crop{saved}.safetensors"
        save_file({"input": crop_inp, "target": crop_tgt}, out_path)
        bytes_written += os.path.getsize(out_path)
        saved += 1

    return saved, discarded, bytes_written


def preprocess_temporal(
    input_dir: str,
    output_dir: str,
    n_crops: int,
    crop_size: int,
    window: int,
    stride: int,
    workers: int,
    min_coverage: float = _MIN_COVERAGE,
) -> None:
    """Extract temporal window crops from full-resolution EXR pairs.

    Groups files by (scene, path_id), sorts by frame, builds sliding windows,
    and extracts crops where all frames share the same spatial position.
    """
    all_files = sorted(
        glob.glob(os.path.join(input_dir, "**", "*_input.exr"), recursive=True)
    )
    if not all_files:
        print(f"No *_input.exr files found in {input_dir}")
        return

    # Group files by (scene_dir, path_id) -> {frame_number: path}
    groups: dict[tuple[str, str], dict[int, str]] = {}
    ungrouped = 0
    for fpath in all_files:
        rel = os.path.relpath(fpath, input_dir).replace("\\", "/")
        parts = rel.rsplit("/", 1)
        if len(parts) == 2:
            scene_dir, fname = parts
        else:
            scene_dir, fname = "", parts[0]

        m = _FNAME_RE.match(fname)
        if not m:
            ungrouped += 1
            continue

        path_id = m.group(2)
        frame_num = int(m.group(3))
        key = (scene_dir, path_id)
        if key not in groups:
            groups[key] = {}
        groups[key][frame_num] = fpath

    # Build all windows
    window_tasks: list[tuple[str, str, list[int], dict[int, str]]] = []
    for (scene_dir, path_id), frame_map in sorted(groups.items()):
        sorted_frames = sorted(frame_map.keys())
        if len(sorted_frames) < window:
            continue
        wins = _windows(sorted_frames, window, stride)
        for win_frames in wins:
            window_tasks.append((scene_dir, path_id, win_frames, frame_map))

    total_windows = len(window_tasks)
    total_paths = len(groups)
    print(f"Found {len(all_files)} EXR input files in {input_dir}")
    if ungrouped:
        print(f"  Skipped {ungrouped} files (filename doesn't match path_id pattern)")
    print(f"  Paths: {total_paths}, Windows: {total_windows}")
    print(f"  Window size: {window}, Stride: {stride}")
    print(f"Output directory: {output_dir}")
    print(f"Crops per window: {n_crops}, crop size: {crop_size}x{crop_size}")
    print(f"Min coverage: {min_coverage:.0%}, oversample: {_OVERSAMPLE_FACTOR}x")
    print(f"Workers: {workers}")

    if total_windows == 0:
        print("No windows to process (paths too short for window size).")
        return

    start_time = time.monotonic()
    total_saved = 0
    total_discarded = 0
    total_bytes = 0
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for scene_dir, path_id, win_frames, frame_map in window_tasks:
            future = executor.submit(
                _process_temporal_window,
                input_dir, scene_dir, path_id, win_frames, frame_map,
                output_dir, n_crops, crop_size, min_coverage,
            )
            futures[future] = f"{scene_dir}/{path_id}_{win_frames[0]:04d}"

        for future in as_completed(futures):
            label = futures[future]
            try:
                saved, discarded, nbytes = future.result()
                total_saved += saved
                total_discarded += discarded
                total_bytes += nbytes
            except Exception as e:
                print(f"  ERROR processing {label}: {e}")
            done += 1
            if done % 50 == 0 or done == total_windows:
                elapsed = time.monotonic() - start_time
                print(f"  [{done}/{total_windows}] {elapsed:.1f}s")

    elapsed = time.monotonic() - start_time
    size_mb = total_bytes / (1024 * 1024)
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Windows processed: {total_windows}")
    print(f"  Crops saved:       {total_saved}")
    print(f"  Crops discarded:   {total_discarded} (below {min_coverage:.0%} coverage)")
    print(f"  Output size:       {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Extract pre-cropped temporal safetensors from full-resolution training data.",
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing EXR training data (*_input.exr / *_target.exr pairs)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for pre-cropped .safetensors files",
    )
    parser.add_argument(
        "--crops", type=int, default=4,
        help="Number of crops to extract per temporal window (default: 4)",
    )
    parser.add_argument(
        "--crop-size", type=int, default=384,
        help="Crop spatial size in pixels (default: 384)",
    )
    parser.add_argument(
        "--window", type=int, default=16,
        help="Temporal window size (default: 16). "
             "Groups frames by path_id and outputs (W, C, H, W) tensors.",
    )
    parser.add_argument(
        "--stride", type=int, default=8,
        help="Sliding window stride (default: 8)",
    )
    parser.add_argument(
        "--workers", type=int, default=min(os.cpu_count() or 1, 8),
        help="Number of parallel workers (default: min(cpu_count, 8))",
    )
    parser.add_argument(
        "--min-coverage", type=float, default=_MIN_COVERAGE,
        help=f"Minimum geometry coverage fraction to keep a crop (default: {_MIN_COVERAGE})",
    )
    args = parser.parse_args()

    preprocess_temporal(
        args.input_dir, args.output_dir, args.crops, args.crop_size,
        args.window, args.stride, args.workers, args.min_coverage,
    )


if __name__ == "__main__":
    main()
