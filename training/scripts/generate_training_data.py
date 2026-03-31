"""Invoke monti_datagen for each training scene to generate EXR pairs.

Reads viewpoints (with embedded environment settings) from per-scene
JSON files produced by monti_view path tracking. Groups viewpoints by
environment to minimize reloads, then invokes monti_datagen once
per group.

Usage:
    python scripts/generate_training_data.py [--monti-datagen <path>] [--output <dir>]
                                              [--scenes <dir>] [--width N] [--height N]
                                              [--spp N] [--ref-frames N]
                                              [--viewpoints-dir <dir>]
                                              [--max-viewpoints N]
                                              [--dry-run] [--skip-confirm]
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np

try:
    import OpenEXR
    import Imath
except ImportError:
    print("Error: OpenEXR and Imath packages required.", file=sys.stderr)
    sys.exit(1)


# Estimated size per EXR pair (input + target) in GB, by compression mode.
# Measured at 960x540: uncompressed ~37 MB, ZIP ~10 MB.
_GB_PER_PAIR = {"none": 0.037, "zip": 0.010}

_FP16_MAX = 65504.0

# Radiance channels in input EXR (FP16 — need overflow protection)
_INPUT_RADIANCE_CHANNELS = [
    "diffuse.R", "diffuse.G", "diffuse.B",
    "specular.R", "specular.G", "specular.B",
]

# Radiance channels in target EXR (FP32 — no overflow protection needed)
_TARGET_RADIANCE_CHANNELS = [
    "diffuse.R", "diffuse.G", "diffuse.B",
    "specular.R", "specular.G", "specular.B",
]


def _read_exr_all_channels(path: str) -> tuple[dict[str, np.ndarray], dict]:
    """Read all channels from an EXR file. Returns (channel_data, header)."""
    exr = OpenEXR.InputFile(path)
    try:
        header = exr.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        pt_float = Imath.PixelType(Imath.PixelType.FLOAT)
        channels: dict[str, np.ndarray] = {}
        for name in header["channels"]:
            raw = exr.channel(name, pt_float)
            channels[name] = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
        return channels, header
    finally:
        exr.close()


def _write_exr(path: str, channels: dict[str, np.ndarray], header: dict) -> None:
    """Write channels to an EXR file preserving the original header layout."""
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Build output header with same channel definitions and compression
    out_header = OpenEXR.Header(width, height)
    out_header["channels"] = header["channels"]
    if "compression" in header:
        out_header["compression"] = header["compression"]

    # Copy custom attributes
    for key, value in header.items():
        if key not in ("channels", "compression", "dataWindow", "displayWindow",
                       "lineOrder", "pixelAspectRatio", "screenWindowCenter",
                       "screenWindowWidth", "type"):
            out_header[key] = value

    out = OpenEXR.OutputFile(path, out_header)
    try:
        ch_defs = header["channels"]
        pt_half = Imath.PixelType(Imath.PixelType.HALF)
        channel_data = {}
        for name, arr in channels.items():
            if name in ch_defs and ch_defs[name].type == pt_half:
                channel_data[name] = arr.astype(np.float16).tobytes()
            else:
                channel_data[name] = arr.astype(np.float32).tobytes()
        out.writePixels(channel_data)
    finally:
        out.close()


def _clamp_fp16_chromaticity(channels: dict[str, np.ndarray],
                              radiance_names: list[str]) -> None:
    """Chromaticity-preserving clamp for FP16 radiance channels in-place.

    For each (R, G, B) triplet, if any channel exceeds FP16 max after scaling,
    all three channels are scaled by the same factor to fit.
    """
    # Process in diffuse/specular triplets
    for start in range(0, len(radiance_names), 3):
        triplet = radiance_names[start:start + 3]
        if len(triplet) < 3:
            break
        r, g, b = channels[triplet[0]], channels[triplet[1]], channels[triplet[2]]
        max_val = np.maximum(np.maximum(np.abs(r), np.abs(g)), np.abs(b))
        overflow = max_val > _FP16_MAX
        if overflow.any():
            scale = np.ones_like(max_val)
            scale[overflow] = _FP16_MAX / max_val[overflow]
            channels[triplet[0]] = r * scale
            channels[triplet[1]] = g * scale
            channels[triplet[2]] = b * scale


def apply_exposure_wedge(
    input_path: str,
    target_path: str,
    output_dir: str,
    base_name: str,
    offsets: list[int],
) -> list[str]:
    """Generate exposure-shifted copies of an input/target EXR pair.

    For each offset s in offsets, scales radiance channels by 2^s.
    The s=0 pair is copied without modification.

    Returns list of output file paths created.
    """
    input_channels, input_header = _read_exr_all_channels(input_path)
    target_channels, target_header = _read_exr_all_channels(target_path)

    created: list[str] = []

    for offset in offsets:
        out_input_name = f"{base_name}_ev{offset:+d}_input.exr"
        out_target_name = f"{base_name}_ev{offset:+d}_target.exr"
        out_input_path = os.path.join(output_dir, out_input_name)
        out_target_path = os.path.join(output_dir, out_target_name)

        if offset == 0:
            # No scaling needed — copy channels as-is
            _write_exr(out_input_path, input_channels, input_header)
            _write_exr(out_target_path, target_channels, target_header)
        else:
            scale = 2.0 ** offset

            # Scale input radiance (FP16 — needs overflow protection)
            scaled_input = dict(input_channels)
            for ch_name in _INPUT_RADIANCE_CHANNELS:
                if ch_name in scaled_input:
                    scaled_input[ch_name] = input_channels[ch_name] * scale
            _clamp_fp16_chromaticity(scaled_input, _INPUT_RADIANCE_CHANNELS)
            _write_exr(out_input_path, scaled_input, input_header)

            # Scale target radiance (FP32 — no clamping needed)
            scaled_target = dict(target_channels)
            for ch_name in _TARGET_RADIANCE_CHANNELS:
                if ch_name in scaled_target:
                    scaled_target[ch_name] = target_channels[ch_name] * scale
            _write_exr(out_target_path, scaled_target, target_header)

        created.append(out_input_path)
        created.append(out_target_path)

    return created


def _select_exposure_offsets(candidates: list[int], n_select: int, seed: str) -> list[int]:
    """Select n_select offsets from candidates, always including EV=0.

    When n_select >= len(candidates), returns all candidates sorted.
    Otherwise randomly samples n_select values (EV=0 plus n_select-1 others)
    using a deterministic seed derived from the viewpoint identity.
    """
    if n_select >= len(candidates):
        return sorted(candidates)
    non_zero = [x for x in candidates if x != 0]
    rng = random.Random(seed)
    chosen = rng.sample(non_zero, n_select - 1)
    return sorted([0] + chosen)


def _viewpoint_is_complete(
    output_dir: str,
    scene_name: str,
    vp: dict,
    exposure_candidates: list[int],
    exposure_count: int,
) -> bool:
    """Check whether all output EXR files for a viewpoint already exist."""
    vp_id = f"{vp['path_id']}_{vp['frame']:04d}"
    base_name = f"{scene_name}_{vp_id}"
    offsets = _select_exposure_offsets(
        exposure_candidates, exposure_count, f"{scene_name}_{vp_id}"
    )
    for offset in offsets:
        input_path = os.path.join(output_dir, f"{base_name}_ev{offset:+d}_input.exr")
        target_path = os.path.join(output_dir, f"{base_name}_ev{offset:+d}_target.exr")
        if not os.path.isfile(input_path) or not os.path.isfile(target_path):
            return False
    return True


def _scene_name_from_path(path: str) -> str:
    """Derive a scene name from a file path.

    For .glb files the stem is used directly.
    For .gltf files with a companion .bin, the parent directory name is used
    when it differs from the stem (e.g. BistroInterior/scene.gltf -> BistroInterior).
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if path.lower().endswith(".gltf"):
        parent = os.path.dirname(path)
        if parent:
            dir_name = os.path.basename(parent)
            bin_path = os.path.join(parent, stem + ".bin")
            if os.path.isfile(bin_path) and stem != dir_name:
                return dir_name
    return stem


def _discover_scenes(scenes_dirs: str | list[str]) -> list[tuple[str, str]]:
    """Discover all scenes in one or more scene directories.

    Returns list of (scene_name, scene_path) tuples. When multiple directories
    are provided, results are merged and deduplicated by scene name (first
    occurrence wins).
    """
    if isinstance(scenes_dirs, str):
        scenes_dirs = [scenes_dirs]

    seen: set[str] = set()
    scenes: list[tuple[str, str]] = []

    for scenes_dir in scenes_dirs:
        if not os.path.isdir(scenes_dir):
            continue

        # GLB files in scenes/ root
        for entry in sorted(os.listdir(scenes_dir)):
            path = os.path.join(scenes_dir, entry)
            if entry.lower().endswith(".glb") and os.path.isfile(path):
                name = _scene_name_from_path(entry)
                if name not in seen:
                    seen.add(name)
                    scenes.append((name, path))

        # Multi-file glTF subdirectories
        for entry in sorted(os.listdir(scenes_dir)):
            subdir = os.path.join(scenes_dir, entry)
            if not os.path.isdir(subdir):
                continue
            gltf_path = os.path.join(subdir, f"{entry}.gltf")
            if not os.path.isfile(gltf_path):
                gltf_files = [
                    f for f in os.listdir(subdir)
                    if f.lower().endswith(".gltf") and os.path.isfile(os.path.join(subdir, f))
                ]
                if len(gltf_files) == 1:
                    gltf_path = os.path.join(subdir, gltf_files[0])
                else:
                    continue
            name = _scene_name_from_path(gltf_path)
            if name not in seen:
                seen.add(name)
                scenes.append((name, gltf_path))

    return scenes


def _load_viewpoints(
    viewpoints_dir: str,
    scene_name: str,
    max_viewpoints: Optional[int],
) -> Optional[list]:
    """Load viewpoint JSON for a scene, sampled to max_viewpoints if set.

    Uses deterministic random sampling (seeded by scene_name) for variety.
    Returns None if no viewpoint file exists (fall back to auto-fit).
    """
    vp_path = os.path.join(viewpoints_dir, f"{scene_name}.json")
    if not os.path.isfile(vp_path):
        return None
    with open(vp_path, "r") as f:
        viewpoints = json.load(f)
    if max_viewpoints is not None and max_viewpoints < len(viewpoints):
        rng = random.Random(scene_name)
        viewpoints = rng.sample(viewpoints, max_viewpoints)
    return viewpoints


def _group_viewpoints(
    viewpoints: list[dict],
    batch_size: Optional[int] = None,
) -> dict[str, list[tuple[int, dict]]]:
    """Group viewpoints by environment for efficient batching.

    Returns dict mapping env_path (with optional batch suffix) ->
    list of (global_index, viewpoint).  When *batch_size* is set,
    groups larger than that are split into sub-batches keyed as
    ``env_path#1``, ``env_path#2``, etc.
    """
    raw_groups: dict[str, list[tuple[int, dict]]] = {}
    for i, vp in enumerate(viewpoints):
        key = vp.get("environment", "")
        if key not in raw_groups:
            raw_groups[key] = []
        raw_groups[key].append((i, vp))

    if batch_size is None or batch_size <= 0:
        return raw_groups

    groups: dict[str, list[tuple[int, dict]]] = {}
    for key, entries in raw_groups.items():
        if len(entries) <= batch_size:
            groups[key] = entries
        else:
            for chunk_idx in range(0, len(entries), batch_size):
                chunk = entries[chunk_idx:chunk_idx + batch_size]
                batch_key = f"{key}#{chunk_idx // batch_size + 1}"
                groups[batch_key] = chunk
    return groups


def _check_disk_space(output_dir: str, total_frames: int, skip_confirm: bool,
                      exr_compression: str = "none") -> None:
    """Check if the output volume has enough free space. Warn and prompt if not."""
    gb_per_pair = _GB_PER_PAIR.get(exr_compression, _GB_PER_PAIR["none"])
    estimated_gb = total_frames * gb_per_pair
    print(f"  Estimated disk:  {estimated_gb:.1f} GB ({total_frames} pairs x {gb_per_pair * 1000:.0f} MB)")

    # Resolve the volume root for the output directory
    check_path = os.path.abspath(output_dir)
    while not os.path.exists(check_path):
        check_path = os.path.dirname(check_path)

    usage = shutil.disk_usage(check_path)
    free_gb = usage.free / (1024 ** 3)
    print(f"  Free disk space: {free_gb:.1f} GB on {check_path}")

    if estimated_gb > free_gb * 0.9:
        print(f"\n  WARNING: Estimated {estimated_gb:.1f} GB exceeds 90% of "
              f"available {free_gb:.1f} GB free space.", file=sys.stderr)
        if skip_confirm:
            print("  --skip-confirm flag set, continuing anyway.", file=sys.stderr)
        else:
            response = input("  Continue anyway? [y/N] ").strip().lower()
            if response not in ("y", "yes"):
                print("Aborted.", file=sys.stderr)
                sys.exit(1)


def _run_render(
    cmd: list[str],
    inv_tmp: str,
) -> tuple[bool, str, Optional[dict]]:
    """Execute monti_datagen and collect timing data.

    Returns (success, error_message, timing_data).  Does NOT move or
    post-process output files — that is handled by _postprocess_outputs.
    """
    os.makedirs(inv_tmp, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        stdout = result.stdout.decode(errors="replace")
        return False, f"exit code {result.returncode}\n{stdout}\n{stderr}", None

    timing = None
    timing_path = os.path.join(inv_tmp, "timing.json")
    if os.path.isfile(timing_path):
        with open(timing_path) as f:
            timing = json.load(f)

    return True, "", timing


def _postprocess_outputs(
    inv_tmp: str,
    output_dir: str,
    scene_name: str,
    group_entries: list[tuple[int, dict]],
    exposure_candidates: list[int],
    exposure_count: int,
) -> None:
    """Apply exposure wedge and move outputs to the final directory."""
    for local_idx, (global_idx, vp) in enumerate(group_entries):
        vp_id = f"{vp['path_id']}_{vp['frame']:04d}"
        src_dir = os.path.join(inv_tmp, f"vp_{local_idx}")
        input_src = os.path.join(src_dir, "input.exr")
        target_src = os.path.join(src_dir, "target.exr")

        if os.path.exists(input_src) and os.path.exists(target_src):
            base_name = f"{scene_name}_{vp_id}"
            offsets = _select_exposure_offsets(
                exposure_candidates, exposure_count, f"{scene_name}_{vp_id}"
            )
            apply_exposure_wedge(
                input_src, target_src, output_dir, base_name, offsets,
            )

    # Clean up this batch's temp directory to free disk space
    shutil.rmtree(inv_tmp, ignore_errors=True)


def _print_timing_summary(
    all_timings: list[dict],
    total_viewpoints: int,
    wall_time_sec: float,
    jobs: int,
) -> None:
    """Print aggregated timing summary from collected timing.json data."""
    if not all_timings:
        return

    # Aggregate setup timings
    setup_keys = [
        "vulkan_init_ms", "scene_load_ms", "env_load_ms",
        "renderer_create_ms", "mesh_upload_ms", "gbuffer_create_ms",
    ]
    setup_sums: dict[str, list[float]] = {k: [] for k in setup_keys}
    for t in all_timings:
        setup = t.get("setup", {})
        for k in setup_keys:
            if k in setup:
                setup_sums[k].append(setup[k])

    # Aggregate per-viewpoint timings
    vp_keys = ["render_noisy_ms", "render_reference_ms", "write_exr_ms", "total_ms"]
    vp_sums: dict[str, list[float]] = {k: [] for k in vp_keys}
    for t in all_timings:
        for vp in t.get("viewpoints", []):
            for k in vp_keys:
                if k in vp:
                    vp_sums[k].append(vp[k])

    n_vp = len(vp_sums["total_ms"])
    if n_vp == 0:
        return

    avg_total = sum(vp_sums["total_ms"]) / n_vp

    setup_labels = {
        "vulkan_init_ms": "Vulkan init",
        "scene_load_ms": "Scene load",
        "env_load_ms": "Environment",
        "renderer_create_ms": "Renderer create",
        "mesh_upload_ms": "Mesh upload",
        "gbuffer_create_ms": "G-buffer create",
    }

    print()
    print("=== Timing Summary ===")
    print(f"  Setup (avg per invocation, {len(all_timings)} invocations):")
    for k in setup_keys:
        vals = setup_sums[k]
        if vals:
            avg = sum(vals) / len(vals)
            lo, hi = min(vals), max(vals)
            range_str = f" (range: {lo:.0f}-{hi:.0f}ms)" if len(vals) > 1 else ""
            print(f"    {setup_labels[k]+':':<20s} {avg:>7.0f}ms{range_str}")

    print()
    vp_labels = {
        "render_noisy_ms": "Render noisy",
        "render_reference_ms": "Render reference",
        "write_exr_ms": "Write EXR",
    }
    print(f"  Per-viewpoint averages (across {n_vp} viewpoints):")
    for k in ["render_noisy_ms", "render_reference_ms", "write_exr_ms"]:
        vals = vp_sums[k]
        if vals:
            avg = sum(vals) / len(vals)
            pct = avg / avg_total * 100 if avg_total > 0 else 0
            print(f"    {vp_labels[k]+':':<20s} {avg:>7.1f}ms   ({pct:.1f}%)")

    # Throughput
    print()
    vp_per_min = total_viewpoints / wall_time_sec * 60 if wall_time_sec > 0 else 0
    h, rem = divmod(int(wall_time_sec), 3600)
    m, s = divmod(rem, 60)

    # Serial sum of all viewpoint times
    serial_total_sec = sum(vp_sums["total_ms"]) / 1000
    # Add setup times
    for t in all_timings:
        setup = t.get("setup", {})
        for k in setup_keys:
            serial_total_sec += setup.get(k, 0) / 1000

    speedup = serial_total_sec / wall_time_sec if wall_time_sec > 0 else 0
    efficiency_pct = speedup / jobs * 100 if jobs > 0 else 0

    print("  Throughput:")
    print(f"    Total wall time:     {h:02d}:{m:02d}:{s:02d}")
    print(f"    Viewpoints/min:      {vp_per_min:.1f}")
    print(f"    Parallel efficiency: {efficiency_pct:.0f}%"
          f"  ({jobs} jobs, {speedup:.1f}x speedup)")

    # Identify dominant bottleneck
    if vp_sums["render_reference_ms"]:
        ref_avg = sum(vp_sums["render_reference_ms"]) / n_vp
        ref_pct = ref_avg / avg_total * 100 if avg_total > 0 else 0
        if ref_pct > 50:
            print(f"\n  Bottleneck: render_reference accounts for {ref_pct:.0f}%"
                  " of per-viewpoint time.")


def _write_aggregate_timing(
    output_dir: str,
    all_timings: list[dict],
    total_viewpoints: int,
    wall_time_sec: float,
    width: int,
    height: int,
    spp: int,
    ref_frames: int,
    jobs: int,
    exr_compression: str,
    ref_spp: Optional[int] = None,
) -> None:
    """Write aggregate timing data to generation_timing.json."""
    if not all_timings:
        return

    effective_ref_spp = ref_spp if ref_spp is not None else spp
    aggregate = {
        "config": {
            "resolution": [width, height],
            "spp": spp,
            "ref_spp": effective_ref_spp,
            "ref_frames": ref_frames,
            "jobs": jobs,
            "exr_compression": exr_compression,
        },
        "wall_time_sec": round(wall_time_sec, 2),
        "total_viewpoints": total_viewpoints,
        "invocations": all_timings,
    }

    timing_path = os.path.join(output_dir, "generation_timing.json")
    try:
        with open(timing_path, "w") as f:
            json.dump(aggregate, f, indent=2)
            f.write("\n")
        print(f"\n  Timing data: {timing_path}")
    except OSError as e:
        print(f"Warning: failed to write {timing_path}: {e}", file=sys.stderr)


def generate_training_data(
    monti_datagen: str,
    output_dir: str,
    scenes_dir: str | list[str],
    width: int,
    height: int,
    spp: int,
    ref_frames: int,
    viewpoints_dir: str,
    max_viewpoints: Optional[int],
    dry_run: bool,
    skip_confirm: bool,
    jobs: int = 3,
    exr_compression: str = "none",
    exposure_steps: int = 5,
    ref_spp: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> None:
    """Run monti_datagen for all discovered scenes.

    Viewpoint JSON files are the canonical source for camera positions
    and environment maps. Viewpoints sharing the same environment are
    batched into a single monti_datagen invocation to minimize resource
    reloads.  When *batch_size* is set, large groups are split so that
    no invocation renders more than *batch_size* viewpoints.

    After monti_datagen produces normalized EXR pairs, an exposure wedge
    is applied: each pair is replicated at *exposure_steps* symmetric
    EV offsets (e.g. steps=5 → -2, -1, 0, +1, +2).
    """
    # Validate resolution divisible by 4 (required by U-Net 2-level 2x MaxPool)
    if width % 4 != 0 or height % 4 != 0:
        print(f"Error: Resolution {width}x{height} must be divisible by 4",
              file=sys.stderr)
        sys.exit(1)

    # Validate monti_datagen exists
    monti_datagen = os.path.abspath(monti_datagen)
    if not dry_run and not os.path.isfile(monti_datagen):
        print(f"Error: monti_datagen not found: {monti_datagen}", file=sys.stderr)
        print("Build monti_datagen first, or specify path with --monti-datagen",
              file=sys.stderr)
        sys.exit(1)

    # Discover scenes dynamically
    scenes = _discover_scenes(scenes_dir)
    if not scenes:
        print("Error: No scene files found. Cannot generate training data.",
              file=sys.stderr)
        sys.exit(1)

    # Validate scene files exist
    missing = []
    available_scenes = []
    for name, scene_path in scenes:
        if os.path.isfile(scene_path):
            available_scenes.append((name, scene_path))
        else:
            missing.append(scene_path)

    if missing:
        print(f"Warning: Missing {len(missing)} scene files:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)

    if not available_scenes:
        print("Error: No scene files found. Cannot generate training data.",
              file=sys.stderr)
        sys.exit(1)

    # Build per-scene plan
    scene_plans: dict[str, dict] = {}
    total_frames = 0
    total_invocations = 0

    for scene_name, scene_path in available_scenes:
        vps = _load_viewpoints(viewpoints_dir, scene_name, max_viewpoints)
        if vps is None:
            # No viewpoints file — single auto-fit invocation
            scene_plans[scene_name] = {
                "groups": {"": [(0, {})]},
                "n_viewpoints": 1,
                "has_viewpoints": False,
            }
            total_frames += 1
            total_invocations += 1
        else:
            groups = _group_viewpoints(vps, batch_size)
            n_vp = len(vps)
            scene_plans[scene_name] = {
                "groups": groups,
                "n_viewpoints": n_vp,
                "has_viewpoints": True,
            }
            total_frames += n_vp
            total_invocations += len(groups)

    # Compute exposure wedge candidates and count.
    # For even n: pool is the balanced wedge of n+1 (range -n//2..+n//2); n are randomly
    # selected per viewpoint (always including EV=0).  For odd n: pool == count, so all
    # offsets are used deterministically.
    half = exposure_steps // 2
    exposure_candidates = list(range(-half, half + 1))
    exposure_count = exposure_steps
    total_output_pairs = total_frames * exposure_steps

    # Print configuration
    effective_ref_spp = ref_spp if ref_spp is not None else spp
    ref_total_spp = ref_frames * effective_ref_spp
    print("=== Monti Training Data Generation ===")
    print(f"  monti_datagen:   {monti_datagen}")
    print(f"  Output:          {output_dir}")
    print(f"  Scenes:          {', '.join(scenes_dir) if isinstance(scenes_dir, list) else scenes_dir}")
    print(f"  Viewpoints dir:  {viewpoints_dir}")
    print(f"  Resolution:      {width}x{height}")
    print(f"  Noisy SPP:       {spp}")
    print(f"  Reference SPP:   {ref_total_spp} ({ref_frames} frames x {effective_ref_spp})")
    print(f"  Parallel jobs:   {jobs}")
    if batch_size is not None:
        print(f"  Batch size:      {batch_size} viewpoints/invocation")
    print(f"  Invocations:     {total_invocations}")
    is_random_wedge = exposure_count < len(exposure_candidates)
    wedge_desc = (f"random {exposure_count} of {exposure_candidates}"
                  if is_random_wedge else str(exposure_candidates))
    print(f"  Exposure wedge:  {exposure_steps} steps ({wedge_desc})")
    if max_viewpoints is not None:
        print(f"  Max viewpoints:  {max_viewpoints}")
    print(f"  Total viewpoints: {total_frames} ({len(available_scenes)} scenes)")
    print(f"  Total pairs:     {total_output_pairs} ({total_frames} viewpoints x {exposure_steps} EV steps)")
    if missing:
        print(f"  Skipped:         {len(missing)} missing scenes")

    _check_disk_space(output_dir, total_output_pairs, skip_confirm, exr_compression)

    # Per-scene breakdown
    print("\n  Per-scene plan:")
    for scene_name, _ in available_scenes:
        plan = scene_plans[scene_name]
        n_vp = plan["n_viewpoints"]
        n_groups = len(plan["groups"])
        vp_source = "viewpoints JSON" if plan["has_viewpoints"] else "auto-fit"
        groups_note = f", {n_groups} group(s)" if n_groups > 1 else ""
        print(f"    {scene_name}: {n_vp} viewpoint(s) ({vp_source}{groups_note})")
    print()

    if dry_run:
        print("=== Dry Run — no data generated ===")
        return

    # Build invocation task list
    start_time = time.monotonic()
    tmp_dir = tempfile.mkdtemp(prefix="monti_vp_")
    os.makedirs(output_dir, exist_ok=True)

    # Filter out already-completed viewpoints for resume support
    skipped_complete = 0
    tasks = []
    for scene_name, scene_path in available_scenes:
        plan = scene_plans[scene_name]

        for group_key, group_entries in plan["groups"].items():
            # Remove viewpoints whose output files already exist
            remaining = [
                (gi, vp) for gi, vp in group_entries
                if not plan["has_viewpoints"] or not _viewpoint_is_complete(
                    output_dir, scene_name, vp,
                    exposure_candidates, exposure_count,
                )
            ]
            n_skipped = len(group_entries) - len(remaining)
            skipped_complete += n_skipped
            if not remaining:
                continue

            inv_idx = len(tasks) + 1
            inv_tmp = os.path.join(tmp_dir, f"inv_{inv_idx}")

            cmd = [
                monti_datagen,
                "--output", inv_tmp,
                "--width", str(width),
                "--height", str(height),
                "--spp", str(spp),
                "--ref-frames", str(ref_frames),
                "--exr-compression", exr_compression,
            ]

            if ref_spp is not None:
                cmd.extend(["--ref-spp", str(ref_spp)])

            if plan["has_viewpoints"]:
                group_vps = [vp for _, vp in remaining]
                vp_tmp_path = os.path.join(
                    tmp_dir, f"{scene_name}_{inv_idx}.json")
                with open(vp_tmp_path, "w") as f:
                    json.dump(group_vps, f)
                cmd.extend(["--viewpoints", vp_tmp_path])

            cmd.append(scene_path)

            # Construct unique --skipped-path in output_dir (survives temp cleanup)
            skipped_filename = f"skipped-{scene_name}-{inv_idx}.json"
            skipped_out_path = os.path.join(output_dir, skipped_filename)
            cmd.extend(["--skipped-path", skipped_out_path])

            env_label = ""
            if group_key:
                # Strip batch suffix (e.g. "path/env.exr#2" -> "path/env.exr")
                display_key = group_key.rsplit("#", 1)[0] if "#" in group_key else group_key
                env_label = f" env={os.path.basename(display_key)}"
                if "#" in group_key:
                    env_label += f" batch {group_key.rsplit('#', 1)[1]}"

            tasks.append({
                "scene_name": scene_name,
                "env_label": env_label,
                "n_frames": len(remaining),
                "cmd": cmd,
                "inv_tmp": inv_tmp,
                "group_entries": remaining,
            })

    remaining_frames = sum(t["n_frames"] for t in tasks)
    if skipped_complete > 0:
        print(f"  Resuming: {skipped_complete} viewpoints already complete, "
              f"{remaining_frames} remaining")
    if not tasks:
        print("\n=== All viewpoints already rendered — nothing to do ===")
        return

    # Execute rendering with pipelined post-processing.
    # GPU rendering runs in `render_pool` (max `jobs` concurrent).
    # Exposure-wedge post-processing runs in `postprocess_pool` so that
    # the GPU can start the next batch while EXR I/O happens in the background.
    completed_count = 0
    frames_done = 0
    failed = False
    all_timings: list[dict] = []
    postprocess_futures: list = []

    try:
        with (ThreadPoolExecutor(max_workers=jobs) as render_pool,
              ThreadPoolExecutor(max_workers=max(jobs, 2)) as postprocess_pool):
            future_to_task = {
                render_pool.submit(
                    _run_render, t["cmd"], t["inv_tmp"],
                ): t
                for t in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    success, error_msg, timing = future.result()
                except Exception as exc:
                    success, error_msg, timing = False, str(exc), None
                completed_count += 1
                frames_done += task["n_frames"]
                label = f"{task['scene_name']}{task['env_label']}"
                if success:

                    # Kick off post-processing in the background
                    pp_future = postprocess_pool.submit(
                        _postprocess_outputs,
                        task["inv_tmp"], output_dir,
                        task["scene_name"], task["group_entries"],
                        exposure_candidates, exposure_count,
                    )
                    postprocess_futures.append((pp_future, label))

                    # Build live progress line with timing info
                    elapsed = time.monotonic() - start_time
                    vp_per_min = frames_done / elapsed * 60 if elapsed > 0 else 0
                    timing_suffix = ""
                    if timing and "summary" in timing:
                        inv_ms = timing["summary"].get("total_ms", 0)
                        avg_vp = timing["summary"].get("avg_viewpoint_ms", 0)
                        avg_ref = timing["summary"].get("avg_render_reference_ms", 0)
                        timing_suffix = (
                            f"  {inv_ms / 1000:.1f}s"
                            f"  avg {avg_vp / 1000:.2f}s/vp"
                            f"  ref {avg_ref / 1000:.2f}s/vp"
                        )
                    print(f"  [{completed_count}/{len(tasks)}] "
                          f"{label} ({task['n_frames']} vp)"
                          f"{timing_suffix}")

                    # Running progress line
                    remaining_vp = remaining_frames - frames_done
                    if vp_per_min > 0 and remaining_vp > 0:
                        eta_sec = remaining_vp / vp_per_min * 60
                        eta_m, eta_s = divmod(int(eta_sec), 60)
                        eta_h, eta_m = divmod(eta_m, 60)
                        el_m, el_s = divmod(int(elapsed), 60)
                        el_h, el_m = divmod(el_m, 60)
                        print(f"  Progress: {frames_done}/{remaining_frames} viewpoints"
                              f"  |  elapsed {el_h:02d}:{el_m:02d}:{el_s:02d}"
                              f"  |  ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
                              f"  |  {vp_per_min:.1f} vp/min")

                    if timing:
                        timing["_scene_name"] = task["scene_name"]
                        all_timings.append(timing)
                else:
                    print(f"Error: monti_datagen failed for {label}:\n"
                          f"  {error_msg}", file=sys.stderr)
                    failed = True
                    render_pool.shutdown(wait=False, cancel_futures=True)
                    break

            # Wait for all background post-processing to finish
            if not failed:
                for pp_future, label in postprocess_futures:
                    try:
                        pp_future.result()
                    except Exception as exc:
                        print(f"Error: post-processing failed for {label}:\n"
                              f"  {exc}", file=sys.stderr)
                        failed = True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if failed:
        sys.exit(1)

    elapsed = time.monotonic() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    print()
    print("=== Complete ===")
    pairs_done = frames_done * exposure_steps
    print(f"  Generated {pairs_done} EXR pairs ({frames_done} viewpoints x "
          f"{exposure_steps} EV steps) in {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"  Output: {output_dir}")

    # Print timing summary if timing data was collected
    _print_timing_summary(all_timings, frames_done, elapsed, jobs)

    # Write aggregate timing JSON
    _write_aggregate_timing(
        output_dir, all_timings, frames_done, elapsed,
        width, height, spp, ref_frames, jobs, exr_compression, ref_spp,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data using monti_datagen")
    parser.add_argument(
        "--monti-datagen",
        default=os.path.join("..", "build", "Release", "monti_datagen.exe"),
        help="Path to monti_datagen executable")
    parser.add_argument("--output", default="training_data",
                        help="Output directory (default: training_data)")
    scenes_root = os.path.join("..", "scenes")
    parser.add_argument("--scenes", nargs="+",
                        default=[os.path.join(scenes_root, "khronos"),
                                 os.path.join(scenes_root, "training")],
                        help="Scene directories (default: ../scenes/khronos ../scenes/training)")
    parser.add_argument("--viewpoints-dir", default="viewpoints",
                        help="Directory containing per-scene viewpoint JSONs (default: viewpoints)")
    parser.add_argument("--max-viewpoints", type=int, default=None,
                        help="Max viewpoints per scene (default: use all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print generation plan without running monti_datagen")
    parser.add_argument("--skip-confirm", "--yes", "-y", action="store_true",
                        help="Skip disk space confirmation prompt")
    parser.add_argument("--jobs", "-j", type=int, default=3,
                        help="Max parallel monti_datagen invocations (default: 3)")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--spp", type=int, default=4,
                        help="Samples per pixel for noisy input")
    parser.add_argument("--ref-frames", type=int, default=64,
                        help="Accumulation frames for reference target")
    parser.add_argument("--ref-spp", type=int, default=None,
                        help="Samples per pixel per reference frame "
                             "(default: same as --spp)")
    parser.add_argument("--exr-compression", default="zip",
                        choices=["none", "zip"],
                        help="EXR compression mode (default: zip)")
    parser.add_argument("--exposure-steps", type=int, default=5,
                        help="Number of exposure wedge steps (default: 5 → offsets -2..+2).\n"
                             "Odd values use a full symmetric wedge. Even values randomly\n"
                             "sample N from a balanced pool of N+1, always including EV=0.")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Max viewpoints per monti_datagen invocation. Large groups "
                             "are split into batches for more frequent progress updates "
                             "(default: 50). Use 0 for unlimited.")
    args = parser.parse_args()

    generate_training_data(
        monti_datagen=args.monti_datagen,
        output_dir=args.output,
        scenes_dir=args.scenes,
        width=args.width,
        height=args.height,
        spp=args.spp,
        ref_frames=args.ref_frames,
        viewpoints_dir=args.viewpoints_dir,
        max_viewpoints=args.max_viewpoints,
        dry_run=args.dry_run,
        skip_confirm=args.skip_confirm,
        jobs=args.jobs,
        exr_compression=args.exr_compression,
        exposure_steps=args.exposure_steps,
        ref_spp=args.ref_spp,
        batch_size=args.batch_size or None,
    )


if __name__ == "__main__":
    main()
