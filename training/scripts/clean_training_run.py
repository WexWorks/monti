#!/usr/bin/env python3
"""clean_training_run.py — Permanently delete all artifacts from a previous training run.

All deletions bypass the Recycle Bin and free disk space immediately.

Run from the training/ directory before starting a new training run:

    python scripts/clean_training_run.py

Use --dry-run to preview what would be deleted without removing anything.
Use --light-rigs to also remove the auto-generated light_rigs/ directory.
"""

import argparse
import shutil
import sys
from pathlib import Path


def _format_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def _dir_stats(path: Path) -> tuple[int, int]:
    """Return (file_count, total_bytes) for all files under path."""
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def _remove(path: Path) -> None:
    """Permanently delete a file or directory tree, bypassing the Recycle Bin."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Permanently delete all artifacts from a previous training run. "
            "Deletions bypass the Recycle Bin and free disk space immediately. "
            "Auto-generated viewpoints are removed while viewpoints/manual/ is preserved."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without deleting anything.",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    parser.add_argument(
        "--light-rigs",
        action="store_true",
        help="Also remove auto-generated light rigs (light_rigs/).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    training_dir = script_dir.parent  # training/scripts/../ = training/

    # -------------------------------------------------------------------------
    # Define cleanup targets.
    # Each entry: (path, label, remove_contents_only)
    #   remove_contents_only=True  → delete children, keep the directory itself
    #   remove_contents_only=False → delete the entire directory
    # -------------------------------------------------------------------------
    targets: list[tuple[Path, str, bool]] = [
        (training_dir / "training_data",      "Training data (EXR/JSON/HTML)",  True),
        (training_dir / "training_data_test", "Training test data",             False),
        (training_dir / "training_data_st",   "Safetensors training data",      False),
        (training_dir / "configs/checkpoints","Model checkpoints (*.pt)",       True),
        (training_dir / "configs/runs",       "TensorBoard run logs",           True),
        (training_dir / "models",             "Exported models (*.denimodel)",  True),
        (training_dir / "results",            "Evaluation results",             True),
    ]

    if args.light_rigs:
        targets.append((training_dir / "light_rigs", "Auto-generated light rigs", True))

    # Auto-generated viewpoints: *.json directly in viewpoints/ (not in manual/)
    viewpoints_dir = training_dir / "viewpoints"
    auto_viewpoints = sorted(viewpoints_dir.glob("*.json")) if viewpoints_dir.exists() else []

    # -------------------------------------------------------------------------
    # Collect items to delete and compute sizes for the summary.
    # -------------------------------------------------------------------------
    total_bytes = 0
    has_anything = bool(auto_viewpoints)

    print("Scanning for previous training artifacts...\n")

    rows: list[str] = []

    for path, label, contents_only in targets:
        if not path.exists():
            rows.append(f"  {'(not found)':<28}  {label}")
            continue

        if contents_only:
            children = [c for c in path.iterdir()]
            if not children:
                rows.append(f"  {'(empty)':<28}  {label}")
                continue
            count, size = _dir_stats(path)
            rows.append(
                f"  {str(path.relative_to(training_dir) / '*'):<28}"
                f"  {label}  ({count} files, {_format_size(size)})"
            )
            total_bytes += size
            has_anything = True
        else:
            count, size = _dir_stats(path)
            rows.append(
                f"  {str(path.relative_to(training_dir)):<28}"
                f"  {label}  ({count} files, {_format_size(size)})"
            )
            total_bytes += size
            has_anything = True

    if auto_viewpoints:
        vp_size = sum(f.stat().st_size for f in auto_viewpoints)
        total_bytes += vp_size
        rows.append(
            f"  {'viewpoints/*.json':<28}"
            f"  Auto-generated viewpoints  ({len(auto_viewpoints)} files,"
            f" {_format_size(vp_size)})  [manual/ preserved]"
        )

    for row in rows:
        print(row)

    print()

    if not has_anything:
        print("Nothing to clean — no previous training artifacts found.")
        return 0

    print(f"Total to free: ~{_format_size(total_bytes)}")
    print()
    print("WARNING: Deletions are permanent and bypass the Recycle Bin.")
    print()

    if args.dry_run:
        print("[dry-run] No files were deleted.")
        return 0

    if not args.yes:
        try:
            response = input("Permanently delete all of the above? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        if response != "y":
            print("Aborted.")
            return 1

    # -------------------------------------------------------------------------
    # Delete.
    # -------------------------------------------------------------------------
    errors = 0

    for path, label, contents_only in targets:
        if not path.exists():
            continue
        if contents_only:
            for child in list(path.iterdir()):
                try:
                    _remove(child)
                except OSError as e:
                    print(f"  ERROR: {e}", file=sys.stderr)
                    errors += 1
        else:
            try:
                _remove(path)
            except OSError as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                errors += 1

    for vp in auto_viewpoints:
        try:
            vp.unlink()
        except OSError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            errors += 1

    if errors:
        print(f"\nDone with {errors} error(s). Some files may not have been removed.")
        return 1

    print("Done. Previous training artifacts removed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
