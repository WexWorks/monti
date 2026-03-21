"""Invoke monti_datagen for each training scene to generate EXR pairs.

Reads viewpoints (with embedded environment/lights/exposure) from per-scene
JSON files produced by generate_viewpoints.py. Groups viewpoints by shared
environment and lights to minimize reloads, then invokes monti_datagen once
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

# Reuse scene discovery from generate_viewpoints
from generate_viewpoints import _discover_scenes

# Estimated size per EXR pair (input + target) in GB — uncompressed default
_GB_PER_PAIR = 0.04


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
) -> dict[tuple[str, str], list[tuple[int, dict]]]:
    """Group viewpoints by (environment, lights) key for efficient batching.

    Returns dict mapping (env_path, lights_path) -> list of (global_index, viewpoint).
    """
    groups: dict[tuple[str, str], list[tuple[int, dict]]] = {}
    for i, vp in enumerate(viewpoints):
        key = (vp.get("environment", ""), vp.get("lights", ""))
        if key not in groups:
            groups[key] = []
        groups[key].append((i, vp))
    return groups


def _check_disk_space(output_dir: str, total_frames: int, skip_confirm: bool) -> None:
    """Check if the output volume has enough free space. Warn and prompt if not."""
    estimated_gb = total_frames * _GB_PER_PAIR
    print(f"  Estimated disk:  {estimated_gb:.1f} GB ({total_frames} pairs x {_GB_PER_PAIR * 1000:.0f} MB)")

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


def _run_invocation(
    cmd: list[str],
    inv_tmp: str,
    output_dir: str,
    scene_name: str,
    group_entries: list[tuple[int, dict]],
) -> tuple[bool, str, Optional[dict]]:
    """Execute a single monti_datagen invocation and move outputs.

    Renames monti_datagen's `vp_N/{input,target}.exr` output to flat
    `<scene>_<id>_{input,target}.exr` files in output_dir.

    Returns (success, error_message, timing_data).
    """
    os.makedirs(inv_tmp, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        stdout = result.stdout.decode(errors="replace")
        return False, f"exit code {result.returncode}\n{stdout}\n{stderr}", None

    # Collect timing.json before moving files
    timing = None
    timing_path = os.path.join(inv_tmp, "timing.json")
    if os.path.isfile(timing_path):
        with open(timing_path) as f:
            timing = json.load(f)

    for local_idx, (global_idx, vp) in enumerate(group_entries):
        vp_id = vp.get("id", f"vp{global_idx}")
        src_dir = os.path.join(inv_tmp, f"vp_{local_idx}")
        for suffix in ("input", "target"):
            src = os.path.join(src_dir, f"{suffix}.exr")
            dst = os.path.join(output_dir, f"{scene_name}_{vp_id}_{suffix}.exr")
            if os.path.exists(src):
                shutil.move(src, dst)

    return True, "", timing


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
) -> None:
    """Write aggregate timing data to generation_timing.json."""
    if not all_timings:
        return

    aggregate = {
        "config": {
            "resolution": [width, height],
            "spp": spp,
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
    scenes_dir: str,
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
) -> None:
    """Run monti_datagen for all discovered scenes.

    Viewpoint JSON files are the canonical source for camera positions,
    exposure, environment maps, and light rigs. Viewpoints sharing the same
    environment and lights are batched into a single monti_datagen invocation
    to minimize resource reloads.
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
                "groups": {("", ""): [(0, {})]},
                "n_viewpoints": 1,
                "has_viewpoints": False,
            }
            total_frames += 1
            total_invocations += 1
        else:
            groups = _group_viewpoints(vps)
            n_vp = len(vps)
            scene_plans[scene_name] = {
                "groups": groups,
                "n_viewpoints": n_vp,
                "has_viewpoints": True,
            }
            total_frames += n_vp
            total_invocations += len(groups)

    # Print configuration
    ref_spp = ref_frames * spp
    effective_parallelism = min(jobs, total_invocations) if total_invocations > 0 else 1
    estimated_time_min = total_frames * 0.5 / effective_parallelism
    estimated_time_max = total_frames * 1.0 / effective_parallelism

    print("=== Monti Training Data Generation ===")
    print(f"  monti_datagen:   {monti_datagen}")
    print(f"  Output:          {output_dir}")
    print(f"  Scenes:          {scenes_dir}")
    print(f"  Viewpoints dir:  {viewpoints_dir}")
    print(f"  Resolution:      {width}x{height}")
    print(f"  Noisy SPP:       {spp}")
    print(f"  Reference SPP:   {ref_spp} ({ref_frames} frames x {spp})")
    print(f"  Parallel jobs:   {jobs}")
    if max_viewpoints is not None:
        print(f"  Max viewpoints:  {max_viewpoints}")
    print(f"  Total frames:    {total_frames} ({len(available_scenes)} scenes)")
    if missing:
        print(f"  Skipped:         {len(missing)} missing scenes")

    _check_disk_space(output_dir, total_frames, skip_confirm)

    est_min_h, est_min_m = divmod(int(estimated_time_min), 60)
    est_max_h, est_max_m = divmod(int(estimated_time_max), 60)
    print(f"  Estimated time:  {est_min_h}h{est_min_m:02d}m – {est_max_h}h{est_max_m:02d}m")

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

    tasks = []
    for scene_name, scene_path in available_scenes:
        plan = scene_plans[scene_name]

        for group_key, group_entries in plan["groups"].items():
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

            if plan["has_viewpoints"]:
                group_vps = [vp for _, vp in group_entries]
                vp_tmp_path = os.path.join(
                    tmp_dir, f"{scene_name}_{inv_idx}.json")
                with open(vp_tmp_path, "w") as f:
                    json.dump(group_vps, f)
                cmd.extend(["--viewpoints", vp_tmp_path])

            cmd.append(scene_path)

            env_label = ""
            if group_key[0]:
                env_label = f" env={os.path.basename(group_key[0])}"
            if group_key[1]:
                env_label = f" lights={os.path.basename(group_key[1])}"

            tasks.append({
                "scene_name": scene_name,
                "env_label": env_label,
                "n_frames": len(group_entries),
                "cmd": cmd,
                "inv_tmp": inv_tmp,
                "group_entries": group_entries,
            })

    # Execute invocations in parallel
    completed_count = 0
    frames_done = 0
    failed = False
    all_timings: list[dict] = []

    try:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            future_to_task = {
                pool.submit(
                    _run_invocation,
                    t["cmd"], t["inv_tmp"], output_dir,
                    t["scene_name"], t["group_entries"],
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
                    remaining_vp = total_frames - frames_done
                    if vp_per_min > 0 and remaining_vp > 0:
                        eta_sec = remaining_vp / vp_per_min * 60
                        eta_m, eta_s = divmod(int(eta_sec), 60)
                        eta_h, eta_m = divmod(eta_m, 60)
                        el_m, el_s = divmod(int(elapsed), 60)
                        el_h, el_m = divmod(el_m, 60)
                        print(f"  Progress: {frames_done}/{total_frames} viewpoints"
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
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if failed:
        sys.exit(1)

    elapsed = time.monotonic() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    print()
    print("=== Complete ===")
    print(f"  Generated {frames_done} EXR pairs in {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"  Output: {output_dir}")

    # Print timing summary if timing data was collected
    _print_timing_summary(all_timings, frames_done, elapsed, jobs)

    # Write aggregate timing JSON
    _write_aggregate_timing(
        output_dir, all_timings, frames_done, elapsed,
        width, height, spp, ref_frames, jobs, exr_compression,
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
    parser.add_argument("--scenes", default="scenes",
                        help="Scenes directory (default: scenes)")
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
    parser.add_argument("--exr-compression", default="none",
                        choices=["none", "zip"],
                        help="EXR compression mode (default: none)")
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
    )


if __name__ == "__main__":
    main()
