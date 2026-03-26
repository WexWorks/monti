"""Convert EXR training data pairs to safetensors format.

Usage:
    python scripts/convert_to_safetensors.py --data_dir training_data/ --output_dir training_data_st/
    python scripts/convert_to_safetensors.py --data_dir training_data/ --output_dir training_data_st/ --verify
"""

import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np
import torch
from safetensors.torch import load_file, save_file

# Reuse the EXR reading infrastructure from the training pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deni_train.data.exr_dataset import (
    _INPUT_CHANNEL_NAMES,
    _HIT_MASK_CHANNEL,
    _TARGET_DIFFUSE,
    _TARGET_SPECULAR,
    _DEMOD_EPS,
    _read_exr_channels,
)


def _discover_exr_pairs(data_dir: str) -> list[tuple[str, str]]:
    """Discover EXR input/target pairs using the same logic as ExrDataset."""
    dir_files = glob.glob(os.path.join(data_dir, "**", "input.exr"), recursive=True)
    flat_files = glob.glob(os.path.join(data_dir, "**", "*_input.exr"), recursive=True)
    input_files = sorted(set(dir_files + flat_files))

    pairs: list[tuple[str, str]] = []
    for input_path in input_files:
        basename = os.path.basename(input_path)
        if basename == "input.exr":
            target_path = os.path.join(os.path.dirname(input_path), "target.exr")
        else:
            target_path = input_path[: -len("_input.exr")] + "_target.exr"
        if os.path.exists(target_path):
            pairs.append((input_path, target_path))
        else:
            warnings.warn(f"Missing target for {input_path}, skipping")
    return pairs


def _build_tensors(
    input_path: str, target_path: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read an EXR pair and return (input, target) tensors matching ExrDataset."""
    # Input: 19-channel float16 (demodulated irradiance + aux + albedo)
    all_input_names = _INPUT_CHANNEL_NAMES + [_HIT_MASK_CHANNEL]
    input_data = _read_exr_channels(input_path, all_input_names)

    hit_mask = input_data[_HIT_MASK_CHANNEL]
    hit_bool = hit_mask > 0.5

    input_arrays = np.stack([input_data[name] for name in _INPUT_CHANNEL_NAMES], axis=0)

    # Extract albedo
    albedo_d = input_arrays[13:16]
    albedo_s = input_arrays[16:19]

    # Demodulate diffuse (channels 0-2)
    raw_d = input_arrays[0:3]
    input_arrays[0:3] = np.where(hit_bool, raw_d / np.maximum(albedo_d, _DEMOD_EPS), raw_d)

    # Demodulate specular (channels 3-5)
    raw_s = input_arrays[3:6]
    input_arrays[3:6] = np.where(hit_bool, raw_s / np.maximum(albedo_s, _DEMOD_EPS), raw_s)

    input_tensor = torch.from_numpy(input_arrays).to(torch.float16)

    # Target: 6-channel demodulated irradiance + 1-channel hit mask (7 total)
    target_channel_names = list(_TARGET_DIFFUSE) + list(_TARGET_SPECULAR)
    target_data = _read_exr_channels(target_path, target_channel_names)

    target_d = np.stack([target_data[n] for n in _TARGET_DIFFUSE], axis=0)
    target_s = np.stack([target_data[n] for n in _TARGET_SPECULAR], axis=0)

    target_d = np.where(hit_bool, target_d / np.maximum(albedo_d, _DEMOD_EPS), target_d)
    target_s = np.where(hit_bool, target_s / np.maximum(albedo_s, _DEMOD_EPS), target_s)

    target_with_mask = np.concatenate(
        [target_d, target_s, hit_mask[np.newaxis]], axis=0
    )
    target_tensor = torch.from_numpy(target_with_mask).to(torch.float16)

    return input_tensor, target_tensor


def _output_path_for_pair(
    input_path: str, data_dir: str, output_dir: str
) -> str:
    """Compute the output .safetensors path preserving relative structure."""
    rel = os.path.relpath(input_path, data_dir)
    basename = os.path.basename(rel)

    # Strip _input.exr or input.exr suffix
    if basename == "input.exr":
        # Directory-based: SceneName/variation/input.exr → SceneName/variation.safetensors
        parent = os.path.dirname(rel)
        st_rel = parent + ".safetensors"
    else:
        # Flat naming: SceneName_hash_input.exr → SceneName_hash.safetensors
        st_rel = os.path.join(
            os.path.dirname(rel),
            basename[: -len("_input.exr")] + ".safetensors",
        )

    return os.path.join(output_dir, st_rel)


def convert(data_dir: str, output_dir: str, verify: bool) -> bool:
    """Convert all EXR pairs to safetensors. Returns True on success."""
    pairs = _discover_exr_pairs(data_dir)
    if not pairs:
        print(f"No EXR pairs found in {data_dir}")
        return False

    print(f"Found {len(pairs)} EXR pairs in {data_dir}")
    print(f"Output directory: {output_dir}")

    total_bytes = 0
    start_time = time.monotonic()
    errors = 0

    for i, (input_path, target_path) in enumerate(pairs):
        out_path = _output_path_for_pair(input_path, data_dir, output_dir)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        try:
            input_tensor, target_tensor = _build_tensors(input_path, target_path)
            save_file({"input": input_tensor, "target": target_tensor}, out_path)
            total_bytes += os.path.getsize(out_path)
        except Exception as e:
            print(f"  ERROR converting {input_path}: {e}")
            errors += 1
            continue

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
            elapsed = time.monotonic() - start_time
            rate = (i + 1) / elapsed
            print(f"  [{i + 1}/{len(pairs)}] {rate:.1f} files/s, "
                  f"{total_bytes / (1024 ** 3):.2f} GB written")

    elapsed = time.monotonic() - start_time
    print(f"\nConversion complete: {len(pairs) - errors}/{len(pairs)} files, "
          f"{total_bytes / (1024 ** 3):.2f} GB, {elapsed:.1f}s")

    if errors:
        print(f"  {errors} errors encountered")

    # Verification pass
    if verify and errors == 0:
        print("\nVerifying converted files...")
        verify_errors = 0
        for i, (input_path, target_path) in enumerate(pairs):
            out_path = _output_path_for_pair(input_path, data_dir, output_dir)
            try:
                st_tensors = load_file(out_path)
                exr_input, exr_target = _build_tensors(input_path, target_path)

                if not torch.equal(st_tensors["input"], exr_input):
                    print(f"  MISMATCH input: {out_path}")
                    verify_errors += 1
                if not torch.equal(st_tensors["target"], exr_target):
                    print(f"  MISMATCH target: {out_path}")
                    verify_errors += 1
            except Exception as e:
                print(f"  ERROR verifying {out_path}: {e}")
                verify_errors += 1

            if (i + 1) % 100 == 0 or (i + 1) == len(pairs):
                print(f"  [{i + 1}/{len(pairs)}] verified")

        if verify_errors:
            print(f"\nVerification FAILED: {verify_errors} errors")
            return False
        else:
            print(f"\nVerification PASSED: all {len(pairs)} files match")

    return errors == 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert EXR training data pairs to safetensors format."
    )
    parser.add_argument(
        "--data_dir", required=True, help="Path to EXR training data directory."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Path to output safetensors directory."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After conversion, verify each file matches the EXR source.",
    )
    args = parser.parse_args()

    success = convert(args.data_dir, args.output_dir, args.verify)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
