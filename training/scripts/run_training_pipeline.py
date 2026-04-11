#!/usr/bin/env python3
"""End-to-end temporal training pipeline: clean → render → crop → train → export.

Assumes viewpoint JSONs already exist in viewpoints/ (recorded via monti_view).
Run from the training/ directory:

    python scripts/run_training_pipeline.py
    python scripts/run_training_pipeline.py --skip-clean --skip-render  # resume from crop step
    python scripts/run_training_pipeline.py --dry-run                   # preview commands without running
"""

import argparse
import subprocess
import sys
import time


def _run(args: list[str], description: str, *, dry_run: bool = False) -> None:
    """Run a subprocess, printing the command and timing it."""
    cmd_str = " ".join(args)
    print(f"\n{'=' * 72}")
    print(f"  {description}")
    print(f"  {cmd_str}")
    print(f"{'=' * 72}\n", flush=True)

    if dry_run:
        print("  [dry-run] skipped\n")
        return

    t0 = time.time()
    result = subprocess.run(args)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n*** FAILED: {description} (exit code {result.returncode}, {elapsed:.0f}s) ***")
        sys.exit(result.returncode)

    print(f"\n  Done: {description} ({elapsed:.0f}s)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full temporal training pipeline end-to-end.",
    )
    # Paths
    parser.add_argument(
        "--monti-datagen", default=r"..\build\Release\monti_datagen.exe",
        help="Path to monti_datagen executable",
    )
    parser.add_argument(
        "--scenes", nargs="+",
        default=[r"..\scenes\khronos", r"..\scenes\training", r"..\scenes\extended\Cauldron-Media"],
        help="Scene directories to render",
    )
    parser.add_argument(
        "--viewpoints-dir", default="viewpoints",
        help="Directory containing per-scene viewpoint JSONs",
    )
    parser.add_argument(
        "--config", default="configs/temporal.yaml",
        help="Training config YAML (default: configs/temporal.yaml)",
    )

    # Rendering
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--spp", type=int, default=4)
    parser.add_argument("--ref-frames", type=int, default=256)
    parser.add_argument("--render-jobs", type=int, default=8, help="Parallel monti_datagen invocations")

    # Temporal crop extraction
    parser.add_argument("--window", type=int, default=16, help="Temporal window size")
    parser.add_argument("--stride", type=int, default=8, help="Sliding window stride")
    parser.add_argument("--crops", type=int, default=4, help="Crops per temporal window")
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--crop-workers", type=int, default=4)

    # Skip flags
    parser.add_argument("--skip-clean", action="store_true", help="Skip the clean step")
    parser.add_argument("--skip-render", action="store_true", help="Skip rendering (use existing EXRs)")
    parser.add_argument("--skip-crop", action="store_true", help="Skip crop extraction")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--skip-export", action="store_true", help="Skip export and golden ref")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training")

    # General
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")

    args = parser.parse_args()
    dry = args.dry_run
    t_start = time.time()

    # ── 1. Clean previous artifacts ──────────────────────────────────────
    if not args.skip_clean:
        _run(
            [sys.executable, r"scripts\clean_training_run.py", "--yes"],
            "Step 1/6: Clean previous training artifacts",
            dry_run=dry,
        )

    # ── 2. Render EXR training pairs ─────────────────────────────────────
    if not args.skip_render:
        render_cmd = [
            sys.executable, r"scripts\generate_training_data.py",
            "--monti-datagen", args.monti_datagen,
            "--scenes", *args.scenes,
            "--viewpoints-dir", args.viewpoints_dir,
            "--output", "training_data",
            "--width", str(args.width),
            "--height", str(args.height),
            "--spp", str(args.spp),
            "--ref-frames", str(args.ref_frames),
            "--jobs", str(args.render_jobs),
            "--skip-confirm",
        ]
        _run(render_cmd, "Step 2/6: Render EXR training pairs", dry_run=dry)

    # ── 3. Extract temporal pre-cropped safetensors ──────────────────────
    if not args.skip_crop:
        _run(
            [
                sys.executable, r"scripts\prepare_temporal.py",
                "--input-dir", "training_data",
                "--output-dir", "training_data_temporal_st",
                "--window", str(args.window),
                "--stride", str(args.stride),
                "--crops", str(args.crops),
                "--crop-size", str(args.crop_size),
                "--workers", str(args.crop_workers),
            ],
            "Step 3/6: Extract temporal pre-cropped safetensors from EXR",
            dry_run=dry,
        )

    # ── 4. Train temporal model ──────────────────────────────────────────
    if not args.skip_train:
        _run(
            [sys.executable, "-m", "deni_train.train_temporal", "--config", args.config],
            "Step 4/6: Train temporal denoiser model",
            dry_run=dry,
        )

    # ── 4b. Evaluate (optional) ──────────────────────────────────────────
    if args.evaluate and not args.skip_train:
        _run(
            [sys.executable, "-m", "deni_train.evaluate_temporal", "--config", args.config],
            "Step 4b: Evaluate temporal denoiser model",
            dry_run=dry,
        )

    # ── 5. Export weights ────────────────────────────────────────────────
    if not args.skip_export:
        _run(
            [
                sys.executable, r"scripts\export_weights.py",
                "--checkpoint", r"configs\checkpoints\model_best.pt",
                "--output", r"models\deni_v3.denimodel",
                "--install",
            ],
            "Step 5/6: Export and install model weights",
            dry_run=dry,
        )

        # ── 6. Regenerate golden reference ───────────────────────────────
        _run(
            [
                sys.executable, r"..\tests\generate_golden_reference.py",
                "--output", r"..\tests\data\golden_ref_v3.bin",
            ],
            "Step 6/6: Regenerate golden reference for GPU tests",
            dry_run=dry,
        )

    elapsed = time.time() - t_start
    print(f"\n{'=' * 72}")
    print(f"  Pipeline complete! Total time: {elapsed:.0f}s")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
