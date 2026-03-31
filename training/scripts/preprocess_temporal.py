"""Offline training data crop extractor.

Phase 4A (static): Extracts random crops from full-resolution safetensors
(produced by convert_to_safetensors.py) into smaller pre-cropped safetensors.
This eliminates the disk I/O bottleneck during training — each sample is ~5MB
instead of ~75MB at full resolution.

Phase 4B (temporal, not yet implemented): Will extend to group frames by
path_id, build sliding temporal windows, and output (T, C, H, W) tensors.

Input safetensors format (from convert_to_safetensors.py):
    input:  float16, (19, H, W)
    target: float16, (7, H, W)

Output safetensors format (same layout, smaller spatial size):
    input:  float16, (19, crop_size, crop_size)
    target: float16, (7, crop_size, crop_size)

Output is compatible with the existing SafetensorsDataset — no new dataset
loader required. Use `precropped: true` in the training config to skip
RandomCrop (data is already crop-sized).

Usage:
    python scripts/preprocess_temporal.py \\
        --input-dir ../training_data_st/ \\
        --output-dir ../training_data_cropped_st/ \\
        --crops 8 --crop-size 384 \\
        --workers 4
"""

import argparse
import glob
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from safetensors.torch import load_file, save_file

# Minimum fraction of pixels that must hit geometry (hit_mask > 0.5) for a
# crop to be kept.  Crops below this threshold are mostly background/sky and
# add no useful denoising signal.
_MIN_COVERAGE = 0.1

# Over-sample factor: attempt this many candidate crops per requested crop to
# compensate for rejections by the coverage check.
_OVERSAMPLE_FACTOR = 3


def _process_one(
    input_path: str,
    rel_path: str,
    output_dir: str,
    n_crops: int,
    crop_size: int,
) -> tuple[int, int, int]:
    """Extract crops from one safetensors file.

    Returns (crops_saved, crops_discarded, bytes_written).
    """
    tensors = load_file(input_path)
    inp = tensors["input"]   # (19, H, W) float16
    tgt = tensors["target"]  # (7, H, W) float16

    _, h, w = inp.shape

    # Derive output base name: strip .safetensors extension
    stem = rel_path[: -len(".safetensors")] if rel_path.endswith(".safetensors") else rel_path

    # If source is already crop-sized or smaller in both dimensions, copy as-is
    if h <= crop_size and w <= crop_size:
        hit_mask = tgt[6]
        coverage = (hit_mask > 0.5).float().mean().item()
        if coverage < _MIN_COVERAGE:
            return 0, 1, 0
        out_path = os.path.join(output_dir, f"{stem}_crop0.safetensors")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_file({"input": inp.contiguous(), "target": tgt.contiguous()}, out_path)
        return 1, 0, os.path.getsize(out_path)

    # Deterministic RNG seeded by relative path for reproducibility
    rng = random.Random(rel_path)

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

        crop_inp = inp[:, cy:cy + crop_size, cx:cx + crop_size].contiguous()
        crop_tgt = tgt[:, cy:cy + crop_size, cx:cx + crop_size].contiguous()

        # Coverage check: discard crops with <10% geometry coverage
        hit_mask = crop_tgt[6]  # (crop_size, crop_size), 1.0=hit 0.0=miss
        coverage = (hit_mask > 0.5).float().mean().item()
        if coverage < _MIN_COVERAGE:
            discarded += 1
            continue

        out_path = os.path.join(output_dir, f"{stem}_crop{saved}.safetensors")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_file({"input": crop_inp, "target": crop_tgt}, out_path)
        bytes_written += os.path.getsize(out_path)
        saved += 1

    return saved, discarded, bytes_written


def preprocess(
    input_dir: str,
    output_dir: str,
    n_crops: int,
    crop_size: int,
    workers: int,
) -> None:
    """Extract crops from all safetensors files in input_dir."""
    all_files = sorted(
        glob.glob(os.path.join(input_dir, "**", "*.safetensors"), recursive=True)
    )
    if not all_files:
        print(f"No .safetensors files found in {input_dir}")
        return

    print(f"Found {len(all_files)} safetensors files in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Crops per image: {n_crops}, crop size: {crop_size}x{crop_size}")
    print(f"Min coverage: {_MIN_COVERAGE:.0%}, oversample: {_OVERSAMPLE_FACTOR}x")
    print(f"Workers: {workers}")

    start_time = time.monotonic()
    total_saved = 0
    total_discarded = 0
    total_bytes = 0
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for fpath in all_files:
            rel = os.path.relpath(fpath, input_dir)
            future = executor.submit(
                _process_one, fpath, rel, output_dir, n_crops, crop_size,
            )
            futures[future] = rel

        for future in as_completed(futures):
            rel = futures[future]
            try:
                saved, discarded, nbytes = future.result()
                total_saved += saved
                total_discarded += discarded
                total_bytes += nbytes
            except Exception as e:
                print(f"  ERROR processing {rel}: {e}")
            done += 1
            if done % 50 == 0 or done == len(all_files):
                elapsed = time.monotonic() - start_time
                print(f"  [{done}/{len(all_files)}] {elapsed:.1f}s")

    elapsed = time.monotonic() - start_time
    size_mb = total_bytes / (1024 * 1024)
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Files processed: {len(all_files)}")
    print(f"  Crops saved:     {total_saved}")
    print(f"  Crops discarded: {total_discarded} (below {_MIN_COVERAGE:.0%} coverage)")
    print(f"  Output size:     {size_mb:.1f} MB")


def verify(input_dir: str, output_dir: str, crop_size: int) -> bool:
    """Verify that output crops match corresponding regions of source data.

    Recomputes crop positions deterministically (same RNG seed) and checks
    that each saved crop is bit-identical to the expected source region.
    Returns True if all checks pass.

    Uses numpy (via safetensors.numpy) so torch is not required locally.
    """
    import re

    import numpy as np
    from safetensors.numpy import load_file as np_load_file

    def _bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
        """Bit-exact comparison that treats NaN == NaN (same bit pattern)."""
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return a.view(np.uint8).tobytes() == b.view(np.uint8).tobytes()

    # Map source stem → source path
    src_files = sorted(
        glob.glob(os.path.join(input_dir, "**", "*.safetensors"), recursive=True)
    )
    src_by_stem: dict[str, str] = {}
    for fpath in src_files:
        rel = os.path.relpath(fpath, input_dir)
        stem = rel[: -len(".safetensors")] if rel.endswith(".safetensors") else rel
        src_by_stem[stem] = fpath

    # Find all output crops
    crop_files = sorted(
        glob.glob(os.path.join(output_dir, "**", "*.safetensors"), recursive=True)
    )
    if not crop_files:
        print("No crop files found to verify.")
        return False

    crop_re = re.compile(r"^(.+)_crop(\d+)\.safetensors$")
    errors = 0
    checked = 0
    total_crops = len(crop_files)
    t0 = time.time()

    print(f"\nVerifying {total_crops} crops...")
    for i, crop_path in enumerate(crop_files):
        if (i + 1) % 500 == 0 or (i + 1) == total_crops:
            print(f"  [{i+1}/{total_crops}] {time.time()-t0:.1f}s  errors={errors}", flush=True)
        rel = os.path.relpath(crop_path, output_dir)
        m = crop_re.match(rel.replace("\\", "/"))
        if not m:
            m = crop_re.match(rel)
        if not m:
            print(f"  WARN: cannot parse crop filename: {rel}")
            continue

        src_stem = m.group(1)
        crop_idx = int(m.group(2))

        if src_stem not in src_by_stem:
            print(f"  FAIL: source not found for {rel} (stem={src_stem})")
            errors += 1
            continue

        # Load source and recompute crop position
        src_tensors = np_load_file(src_by_stem[src_stem])
        src_inp = src_tensors["input"]
        src_tgt = src_tensors["target"]
        _, h, w = src_inp.shape

        crop_tensors = np_load_file(crop_path)
        crop_inp = crop_tensors["input"]
        crop_tgt = crop_tensors["target"]

        # Small-image passthrough: crop0 should be the entire source
        if h <= crop_size and w <= crop_size:
            if not _bytes_equal(crop_inp, src_inp) or not _bytes_equal(crop_tgt, src_tgt):
                print(f"  FAIL: passthrough mismatch: {rel}")
                errors += 1
            else:
                checked += 1
            continue

        # Recompute candidates with same RNG seed.
        # We don't know n_crops used during extraction, but crop_idx tells us
        # at least crop_idx+1 valid crops were found. Generate enough candidates
        # to cover any reasonable n_crops value (up to 32).
        src_rel = src_stem + ".safetensors"
        rng = random.Random(src_rel)
        max_attempts = 32 * _OVERSAMPLE_FACTOR
        candidates = [
            (rng.randint(0, w - crop_size), rng.randint(0, h - crop_size))
            for _ in range(max_attempts)
        ]

        # Walk candidates to find which one produced crop_idx
        valid_idx = 0
        matched = False
        for cx, cy in candidates:
            region_tgt = src_tgt[:, cy:cy + crop_size, cx:cx + crop_size]
            hit_mask = region_tgt[6]
            coverage = float(np.mean(hit_mask > 0.5))
            if coverage < _MIN_COVERAGE:
                continue
            if valid_idx == crop_idx:
                region_inp = np.ascontiguousarray(src_inp[:, cy:cy + crop_size, cx:cx + crop_size])
                region_tgt = np.ascontiguousarray(region_tgt)
                if not _bytes_equal(crop_inp, region_inp):
                    print(f"  FAIL: input mismatch: {rel} at ({cx},{cy})")
                    errors += 1
                elif not _bytes_equal(crop_tgt, region_tgt):
                    print(f"  FAIL: target mismatch: {rel} at ({cx},{cy})")
                    errors += 1
                else:
                    checked += 1
                matched = True
                break
            valid_idx += 1

        if not matched:
            print(f"  FAIL: could not reconstruct crop position for {rel}")
            errors += 1

    print(f"\nVerification: {checked} crops checked, {errors} errors")
    return errors == 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract pre-cropped safetensors from full-resolution training data.",
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing full-resolution .safetensors files",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for pre-cropped .safetensors files",
    )
    parser.add_argument(
        "--crops", type=int, default=8,
        help="Number of crops to extract per source image (default: 8)",
    )
    parser.add_argument(
        "--crop-size", type=int, default=384,
        help="Crop spatial size in pixels (default: 384)",
    )
    parser.add_argument(
        "--workers", type=int, default=min(os.cpu_count() or 1, 8),
        help="Number of parallel workers (default: min(cpu_count, 8))",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="After extraction, verify each crop matches the source region",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Skip extraction, only verify existing crops against sources",
    )
    args = parser.parse_args()
    if not args.verify_only:
        preprocess(args.input_dir, args.output_dir, args.crops, args.crop_size, args.workers)
    if args.verify or args.verify_only:
        ok = verify(args.input_dir, args.output_dir, args.crop_size)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
