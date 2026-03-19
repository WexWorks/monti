"""Invoke monti_datagen for each training scene at multiple exposure levels.

Generates EXR input/target pairs for ML denoiser training.

Usage:
    python scripts/generate_training_data.py [--monti-datagen <path>] [--output <dir>]
                                              [--scenes <dir>] [--width N] [--height N]
                                              [--spp N] [--ref-frames N]
"""

import argparse
import os
import subprocess
import sys
import time

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


def _format_exposure(ev: float) -> str:
    """Format exposure value as a signed string for directory naming."""
    if ev > 0:
        return f"+{ev:.1f}"
    elif ev < 0:
        return f"{ev:.1f}"
    return "0.0"


def generate_training_data(
    monti_datagen: str,
    output_dir: str,
    scenes_dir: str,
    width: int,
    height: int,
    spp: int,
    ref_frames: int,
) -> None:
    """Run monti_datagen for all scenes and exposure levels."""

    # Validate resolution divisible by 4 (required by U-Net 2-level 2x MaxPool)
    if width % 4 != 0 or height % 4 != 0:
        print(f"Error: Resolution {width}x{height} must be divisible by 4",
              file=sys.stderr)
        sys.exit(1)

    # Validate monti_datagen exists
    monti_datagen = os.path.abspath(monti_datagen)
    if not os.path.isfile(monti_datagen):
        print(f"Error: monti_datagen not found: {monti_datagen}", file=sys.stderr)
        print("Build monti_datagen first, or specify path with --monti-datagen",
              file=sys.stderr)
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

    # Print configuration
    total_pairs = len(available_scenes) * len(_EXPOSURES)
    ref_spp = ref_frames * spp
    exposure_strs = ", ".join(_format_exposure(e) for e in _EXPOSURES)

    print("=== Monti Training Data Generation ===")
    print(f"  monti_datagen:  {monti_datagen}")
    print(f"  Output:         {output_dir}")
    print(f"  Scenes:         {scenes_dir}")
    print(f"  Resolution:     {width}x{height}")
    print(f"  Noisy SPP:      {spp}")
    print(f"  Reference SPP:  {ref_spp} ({ref_frames} frames x {spp})")
    print(f"  Exposures:      {exposure_strs} EV")
    print(f"  Total pairs:    {total_pairs} ({len(available_scenes)} scenes x {len(_EXPOSURES)} exposures)")
    if missing:
        print(f"  Skipped:        {len(missing)} missing scenes")
    print()

    # Generate data
    pair_count = 0
    start_time = time.monotonic()

    for scene_name, scene_file in available_scenes:
        scene_path = os.path.abspath(os.path.join(scenes_dir, scene_file))

        for exposure in _EXPOSURES:
            pair_count += 1
            ev_str = _format_exposure(exposure)
            out_subdir = os.path.join(output_dir, scene_name, f"ev_{ev_str}")
            os.makedirs(out_subdir, exist_ok=True)

            print(f"[{pair_count}/{total_pairs}] {scene_name} @ {ev_str} EV")

            cmd = [
                monti_datagen,
                "--output", out_subdir,
                "--width", str(width),
                "--height", str(height),
                "--spp", str(spp),
                "--ref-frames", str(ref_frames),
                "--exposure", str(exposure),
                scene_path,
            ]

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"Error: monti_datagen failed for {scene_name} @ {ev_str} EV "
                      f"(exit code {result.returncode})", file=sys.stderr)
                sys.exit(1)

    elapsed = time.monotonic() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    print()
    print("=== Complete ===")
    print(f"  Generated {pair_count} EXR pairs in {hours:02d}:{minutes:02d}:{seconds:02d}")
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
    )


if __name__ == "__main__":
    main()
