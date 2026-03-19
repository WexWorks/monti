"""Invoke monti_datagen for each training scene at multiple exposure levels.

Generates EXR input/target pairs for ML denoiser training.

Usage:
    python scripts/generate_training_data.py [--monti-datagen <path>] [--output <dir>]
                                              [--scenes <dir>] [--width N] [--height N]
                                              [--spp N] [--ref-frames N]
                                              [--viewpoints-dir <dir>] [--env <path>]
                                              [--max-viewpoints N] [--dry-run] [--yes]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Optional

# Scene definitions: (directory_name, scene_file_relative_path)
# GLB scenes use just the filename; multi-file glTF uses subdir/filename.
_SCENES = [
    ("cornell_box", "cornell_box.glb"),
    ("damaged_helmet", "DamagedHelmet.glb"),
    ("dragon_attenuation", "DragonAttenuation.glb"),
    ("water_bottle", "WaterBottle.glb"),
    ("antique_camera", "AntiqueCamera.glb"),
    ("lantern", "Lantern.glb"),
    ("toy_car", "ToyCar.glb"),
    ("a_beautiful_game", "ABeautifulGame.glb"),
    ("mosquito_in_amber", "MosquitoInAmber.glb"),
    ("glass_hurricane_candle_holder", "GlassHurricaneCandleHolder.glb"),
    ("boom_box", "BoomBox.glb"),
    ("sheen_chair", "SheenChair.glb"),
    ("flight_helmet", os.path.join("FlightHelmet", "FlightHelmet.gltf")),
    ("sponza", os.path.join("Sponza", "Sponza.gltf")),
]

# Exposure levels (EV100)
_EXPOSURES = [-1.0, -0.5, 0.0, 0.5, 1.0]

# Estimated size per EXR pair (input + target) in GB
_GB_PER_PAIR = 0.15


def _format_exposure(ev: float) -> str:
    """Format exposure value as a signed string for directory naming."""
    if ev > 0:
        return f"+{ev:.1f}"
    elif ev < 0:
        return f"{ev:.1f}"
    return "0.0"


def _load_viewpoints(
    viewpoints_dir: str,
    scene_name: str,
    max_viewpoints: Optional[int],
) -> Optional[list]:
    """Load viewpoint JSON for a scene, truncated to max_viewpoints if set.

    Returns None if no viewpoint file exists (fall back to auto-fit).
    """
    vp_path = os.path.join(viewpoints_dir, f"{scene_name}.json")
    if not os.path.isfile(vp_path):
        return None
    with open(vp_path, "r") as f:
        viewpoints = json.load(f)
    if max_viewpoints is not None:
        viewpoints = viewpoints[:max_viewpoints]
    return viewpoints


def _count_viewpoints_per_scene(
    available_scenes: list,
    viewpoints_dir: str,
    max_viewpoints: Optional[int],
) -> tuple[dict, dict]:
    """Return (counts, viewpoints) dicts keyed by scene_name.

    counts maps scene_name -> number of viewpoints.
    viewpoints maps scene_name -> list or None.
    """
    counts = {}
    viewpoints = {}
    for name, _ in available_scenes:
        vps = _load_viewpoints(viewpoints_dir, name, max_viewpoints)
        counts[name] = len(vps) if vps is not None else 1
        viewpoints[name] = vps
    return counts, viewpoints


def _check_disk_space(output_dir: str, total_frames: int, auto_yes: bool) -> None:
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
        if auto_yes:
            print("  --yes flag set, continuing anyway.", file=sys.stderr)
        else:
            response = input("  Continue anyway? [y/N] ").strip().lower()
            if response not in ("y", "yes"):
                print("Aborted.", file=sys.stderr)
                sys.exit(1)


def generate_training_data(
    monti_datagen: str,
    output_dir: str,
    scenes_dir: str,
    width: int,
    height: int,
    spp: int,
    ref_frames: int,
    viewpoints_dir: str,
    env_map: Optional[str],
    max_viewpoints: Optional[int],
    dry_run: bool,
    auto_yes: bool,
) -> None:
    """Run monti_datagen for all scenes, viewpoints, and exposure levels."""

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

    # Validate env map if specified
    if env_map is not None:
        env_map = os.path.abspath(env_map)
        if not os.path.isfile(env_map):
            print(f"Error: Environment map not found: {env_map}", file=sys.stderr)
            sys.exit(1)

    # Validate all scene files exist — warn but don't abort for missing optional scenes
    missing = []
    available_scenes = []
    for name, filename in _SCENES:
        scene_path = os.path.join(scenes_dir, filename)
        if not os.path.isfile(scene_path):
            missing.append(filename)
        else:
            available_scenes.append((name, filename))

    if missing:
        print(f"Warning: Missing {len(missing)} scene files in {scenes_dir}:",
              file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print("Run these scripts to download scenes:", file=sys.stderr)
        print(f"  python scripts/export_cornell_box.py --output {scenes_dir}/cornell_box.glb",
              file=sys.stderr)
        print(f"  python scripts/download_scenes.py --output {scenes_dir}/",
              file=sys.stderr)

    if not available_scenes:
        print("Error: No scene files found. Cannot generate training data.",
              file=sys.stderr)
        sys.exit(1)

    # Count viewpoints per scene
    vp_counts, vp_data = _count_viewpoints_per_scene(
        available_scenes, viewpoints_dir, max_viewpoints)
    total_frames = sum(
        vp_counts[name] * len(_EXPOSURES) for name, _ in available_scenes)

    # Print configuration
    ref_spp = ref_frames * spp
    exposure_strs = ", ".join(_format_exposure(e) for e in _EXPOSURES)
    estimated_time_min = total_frames * 0.5  # ~30s per frame low estimate
    estimated_time_max = total_frames * 1.0  # ~60s per frame high estimate

    print("=== Monti Training Data Generation ===")
    print(f"  monti_datagen:   {monti_datagen}")
    print(f"  Output:          {output_dir}")
    print(f"  Scenes:          {scenes_dir}")
    print(f"  Viewpoints dir:  {viewpoints_dir}")
    if env_map:
        print(f"  Environment map: {env_map}")
    print(f"  Resolution:      {width}x{height}")
    print(f"  Noisy SPP:       {spp}")
    print(f"  Reference SPP:   {ref_spp} ({ref_frames} frames x {spp})")
    print(f"  Exposures:       {exposure_strs} EV")
    if max_viewpoints is not None:
        print(f"  Max viewpoints:  {max_viewpoints}")
    print(f"  Total frames:    {total_frames} ({len(available_scenes)} scenes)")
    if missing:
        print(f"  Skipped:         {len(missing)} missing scenes")

    _check_disk_space(output_dir, total_frames, auto_yes)

    est_min_h, est_min_m = divmod(int(estimated_time_min), 60)
    est_max_h, est_max_m = divmod(int(estimated_time_max), 60)
    print(f"  Estimated time:  {est_min_h}h{est_min_m:02d}m – {est_max_h}h{est_max_m:02d}m")
    print()

    # Per-scene breakdown
    print("  Per-scene plan:")
    for name, _ in available_scenes:
        n_vp = vp_counts[name]
        n_frames = n_vp * len(_EXPOSURES)
        vp_source = "viewpoints JSON" if vp_data[name] is not None else "auto-fit"
        print(f"    {name}: {n_vp} viewpoint(s) x {len(_EXPOSURES)} exposures = {n_frames} frames ({vp_source})")
    print()

    if dry_run:
        print("=== Dry Run — no data generated ===")
        return

    # Generate data — loop over scenes × exposures.
    # monti_datagen handles all viewpoints per invocation, creating vp_N/ subdirs.
    invocation_count = 0
    total_invocations = len(available_scenes) * len(_EXPOSURES)
    frames_rendered = 0
    start_time = time.monotonic()
    tmp_dir = tempfile.mkdtemp(prefix="monti_vp_")

    try:
        for scene_name, scene_file in available_scenes:
            scene_path = os.path.abspath(os.path.join(scenes_dir, scene_file))
            viewpoints = vp_data[scene_name]

            # Write truncated viewpoints to temp file for monti_datagen
            vp_tmp_path = None
            if viewpoints is not None:
                vp_tmp_path = os.path.join(tmp_dir, f"{scene_name}.json")
                with open(vp_tmp_path, "w") as f:
                    json.dump(viewpoints, f)

            n_vp = vp_counts[scene_name]

            for exposure in _EXPOSURES:
                invocation_count += 1
                ev_str = _format_exposure(exposure)
                out_subdir = os.path.join(output_dir, scene_name, f"ev_{ev_str}")
                os.makedirs(out_subdir, exist_ok=True)

                frames_rendered += n_vp
                print(f"[{invocation_count}/{total_invocations}] "
                      f"{scene_name} @ {ev_str} EV ({n_vp} viewpoint(s), "
                      f"{frames_rendered}/{total_frames} frames)")

                cmd = [
                    monti_datagen,
                    "--output", out_subdir,
                    "--width", str(width),
                    "--height", str(height),
                    "--spp", str(spp),
                    "--ref-frames", str(ref_frames),
                    "--exposure", str(exposure),
                ]

                if vp_tmp_path is not None:
                    cmd.extend(["--viewpoints", vp_tmp_path])

                if env_map is not None:
                    cmd.extend(["--env", env_map])

                cmd.append(scene_path)

                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"Error: monti_datagen failed for {scene_name} "
                          f"@ {ev_str} EV (exit code {result.returncode})",
                          file=sys.stderr)
                    sys.exit(1)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.monotonic() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    print()
    print("=== Complete ===")
    print(f"  Generated {frames_rendered} EXR pairs in {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data using monti_datagen")
    parser.add_argument(
        "--monti-datagen",
        default=os.path.join("..", "build", "app", "datagen", "Release", "monti_datagen.exe"),
        help="Path to monti_datagen executable")
    parser.add_argument("--output", default="training_data",
                        help="Output directory (default: training_data)")
    parser.add_argument("--scenes", default="scenes",
                        help="Scenes directory (default: scenes)")
    parser.add_argument("--viewpoints-dir", default="viewpoints",
                        help="Directory containing per-scene viewpoint JSONs (default: viewpoints)")
    parser.add_argument("--env", default=None,
                        help="Path to HDR environment map (.exr) forwarded to monti_datagen")
    parser.add_argument("--max-viewpoints", type=int, default=None,
                        help="Max viewpoints per scene (default: use all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print generation plan without running monti_datagen")
    parser.add_argument("--yes", action="store_true",
                        help="Skip disk space confirmation prompt")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--spp", type=int, default=4,
                        help="Samples per pixel for noisy input")
    parser.add_argument("--ref-frames", type=int, default=64,
                        help="Accumulation frames for reference target")
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
        env_map=args.env,
        max_viewpoints=args.max_viewpoints,
        dry_run=args.dry_run,
        auto_yes=args.yes,
    )


if __name__ == "__main__":
    main()
