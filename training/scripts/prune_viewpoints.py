"""Remove skipped viewpoints from viewpoint JSON files.

Reads one or more skipped-viewpoints JSON files produced by monti_datagen
(via --skipped-path) and removes the corresponding entries from per-scene
viewpoint JSON files.

Each skipped file has the format:
    {"scene": "<name>", "skipped": [{"viewpoint_id": "...", "reason": "...", "detail": ...}]}

This is an optional cleanup step — monti_datagen already skips invalid
viewpoints during rendering (no EXR output). Pruning prevents re-rendering
them on subsequent runs.

Usage:
    python scripts/prune_viewpoints.py \
        --skipped training_data/skipped-*.json \
        --viewpoints-dir viewpoints \
        [--dry-run]
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict


def _load_skipped_files(skipped_paths: list[str]) -> list[dict]:
    """Load and merge entries from multiple skipped-viewpoints JSON files.

    Each file has {"scene": ..., "skipped": [...]}.  Returns a flat list
    of dicts with "scene" and "viewpoint_id" keys (plus "reason" and "detail").
    """
    entries: list[dict] = []
    for path in skipped_paths:
        with open(path) as f:
            data = json.load(f)
        scene = data["scene"]
        for skip in data.get("skipped", []):
            entries.append({
                "scene": scene,
                "viewpoint_id": skip["viewpoint_id"],
                "reason": skip.get("reason", ""),
                "detail": skip.get("detail", 0.0),
            })
    return entries


def prune_viewpoints(
    skipped_paths: list[str],
    viewpoints_dir: str,
    dry_run: bool = False,
) -> dict[str, int]:
    """Remove skipped viewpoints from scene JSON files.

    Args:
        skipped_paths: Paths to skipped-viewpoints JSON files from monti_datagen.
        viewpoints_dir: Directory containing per-scene viewpoint JSON files.
        dry_run: If True, print what would be removed without modifying files.

    Returns:
        Dict mapping scene name to number of viewpoints removed.
    """
    entries = _load_skipped_files(skipped_paths)

    if not entries:
        print("No skipped viewpoints to prune.")
        return {}

    # Group by scene
    by_scene: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        by_scene[entry["scene"]].add(entry["viewpoint_id"])

    removed: dict[str, int] = {}

    for scene_name, vp_ids in sorted(by_scene.items()):
        vp_path = os.path.join(viewpoints_dir, f"{scene_name}.json")
        if not os.path.isfile(vp_path):
            print(f"  Warning: {vp_path} not found, skipping", file=sys.stderr)
            continue

        with open(vp_path) as f:
            viewpoints = json.load(f)

        original_count = len(viewpoints)

        def _vp_id(vp: dict) -> str:
            return f"{vp['path_id']}_{vp.get('frame', 0):04d}"

        kept = [vp for vp in viewpoints if _vp_id(vp) not in vp_ids]
        pruned_count = original_count - len(kept)

        if pruned_count == 0:
            print(f"  {scene_name}: no matching viewpoints found "
                  f"(IDs: {sorted(vp_ids)})")
            continue

        if dry_run:
            print(f"  {scene_name}: would remove {pruned_count}/{original_count} "
                  f"viewpoints (IDs: {sorted(vp_ids)})")
        else:
            with open(vp_path, "w") as f:
                json.dump(kept, f, indent=2)
                f.write("\n")
            print(f"  {scene_name}: removed {pruned_count}/{original_count} "
                  f"viewpoints (IDs: {sorted(vp_ids)})")

        removed[scene_name] = pruned_count

    total = sum(removed.values())
    action = "Would remove" if dry_run else "Removed"
    print(f"\n{action} {total} viewpoint(s) across {len(removed)} scene(s).")
    return removed


def main():
    parser = argparse.ArgumentParser(
        description="Remove skipped viewpoints from viewpoint JSON files")
    parser.add_argument("--skipped", nargs="+", required=True,
                        help="Paths or glob patterns for skipped-*.json files")
    parser.add_argument("--viewpoints-dir", default="viewpoints",
                        help="Directory with per-scene viewpoint JSONs "
                             "(default: viewpoints)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without modifying files")
    args = parser.parse_args()

    # Expand globs
    resolved: list[str] = []
    for pattern in args.skipped:
        matches = glob.glob(pattern)
        if matches:
            resolved.extend(matches)
        elif os.path.isfile(pattern):
            resolved.append(pattern)
        else:
            print(f"Warning: {pattern} matched no files", file=sys.stderr)

    if not resolved:
        print("Error: no skipped-viewpoints files found", file=sys.stderr)
        sys.exit(1)

    prune_viewpoints(resolved, args.viewpoints_dir, args.dry_run)


if __name__ == "__main__":
    main()
