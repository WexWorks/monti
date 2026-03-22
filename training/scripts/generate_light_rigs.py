"""Generate per-scene area light rig JSON files for training data diversity.

Creates two lighting rigs per scene (overhead + key-fill-rim) using bounding
box geometry from GLB/glTF files. Output JSONs match the --lights format
expected by monti_datagen.

Usage:
    python scripts/generate_light_rigs.py --scenes-dir scenes/ --output light_rigs/
                                          [--seed 42]
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Optional

# Reuse bounding box and scene discovery from generate_viewpoints
from generate_viewpoints import (
    compute_bounding_box,
    _center_and_radius_from_aabb,
    _discover_scenes,
    _scene_has_emissive_lights,
)


def _make_quad_facing(
    center: list[float],
    normal: list[float],
    width: float,
    height: float,
) -> tuple[list[float], list[float], list[float]]:
    """Create a quad (corner, edge_a, edge_b) centered at `center` facing `normal`.

    Returns (corner, edge_a, edge_b) where cross(edge_a, edge_b) points along
    the provided normal direction.
    """
    # Find a vector not parallel to normal for cross product
    nx, ny, nz = normal
    n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
    if n_len < 1e-8:
        raise ValueError("Normal vector is zero-length")
    nx, ny, nz = nx / n_len, ny / n_len, nz / n_len

    # Choose an up vector not parallel to normal
    if abs(ny) < 0.9:
        up = [0.0, 1.0, 0.0]
    else:
        up = [1.0, 0.0, 0.0]

    # edge_a = normalize(up × normal) × width
    ax = up[1] * nz - up[2] * ny
    ay = up[2] * nx - up[0] * nz
    az = up[0] * ny - up[1] * nx
    a_len = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / a_len, ay / a_len, az / a_len

    # edge_b = normalize(normal × edge_a) × height
    bx = ny * az - nz * ay
    by = nz * ax - nx * az
    bz = nx * ay - ny * ax
    b_len = math.sqrt(bx * bx + by * by + bz * bz)
    bx, by, bz = bx / b_len, by / b_len, bz / b_len

    edge_a = [ax * width, ay * width, az * width]
    edge_b = [bx * height, by * height, bz * height]

    # Corner = center - 0.5 * edge_a - 0.5 * edge_b
    corner = [
        center[0] - 0.5 * edge_a[0] - 0.5 * edge_b[0],
        center[1] - 0.5 * edge_a[1] - 0.5 * edge_b[1],
        center[2] - 0.5 * edge_a[2] - 0.5 * edge_b[2],
    ]

    return corner, edge_a, edge_b


def _spherical_to_cartesian(
    center: list[float],
    radius: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> list[float]:
    """Convert spherical coordinates to world position around center."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cos_el = math.cos(el)
    x = center[0] + radius * cos_el * math.sin(az)
    y = center[1] + radius * math.sin(el)
    z = center[2] + radius * cos_el * math.cos(az)
    return [x, y, z]


def _direction_toward(from_pos: list[float], to_pos: list[float]) -> list[float]:
    """Compute unit direction vector from from_pos toward to_pos."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    dz = to_pos[2] - from_pos[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-8:
        return [0.0, -1.0, 0.0]
    return [dx / length, dy / length, dz / length]


def generate_overhead_rig(
    bbox_min: list[float],
    bbox_max: list[float],
    rng: random.Random,
) -> list[dict]:
    """Generate an overhead area light rig for a scene.

    Single quad area light centered above the bounding box.
    """
    center = [(bbox_min[i] + bbox_max[i]) / 2.0 for i in range(3)]
    extent = [bbox_max[i] - bbox_min[i] for i in range(3)]
    extent_y = extent[1]
    horizontal_extent = max(extent[0], extent[2], 0.1)

    # Height above scene
    light_y = bbox_max[1] + 1.5 * extent_y
    light_center = [center[0], light_y, center[2]]

    # Random size
    size_scale = rng.uniform(0.5, 2.0)
    light_width = size_scale * horizontal_extent
    light_height = size_scale * horizontal_extent

    # Normal faces downward
    normal = [0.0, -1.0, 0.0]
    corner, edge_a, edge_b = _make_quad_facing(light_center, normal,
                                                light_width, light_height)

    # Warm white radiance with random scale
    radiance_scale = rng.uniform(0.5, 2.0)
    radiance = [5.0 * radiance_scale, 4.8 * radiance_scale, 4.5 * radiance_scale]

    return [{
        "corner": corner,
        "edge_a": edge_a,
        "edge_b": edge_b,
        "radiance": radiance,
        "two_sided": False,
    }]


def generate_key_fill_rim_rig(
    bbox_min: list[float],
    bbox_max: list[float],
    rng: random.Random,
) -> list[dict]:
    """Generate a three-point lighting rig (key, fill, rim) for a scene.

    Three area lights positioned around the bounding box at varying azimuth
    and elevation angles.
    """
    center, bbox_radius = _center_and_radius_from_aabb(bbox_min, bbox_max)
    extent = [bbox_max[i] - bbox_min[i] for i in range(3)]
    light_size = 0.5 * max(extent[0], extent[1], extent[2], 0.1)
    light_distance = 2.0 * bbox_radius

    # Light definitions: (nominal_azimuth, nominal_elevation, base_radiance)
    light_defs = [
        # Key: 45° azimuth, 30° elevation
        (45.0, 30.0, [8.0, 7.5, 7.0]),
        # Fill: -45° azimuth, 15° elevation, ~0.3× key, cooler
        (-45.0, 15.0, [2.0, 2.2, 2.5]),
        # Rim: 180° azimuth, 45° elevation, ~0.5× key, warm
        (180.0, 45.0, [4.0, 3.8, 3.5]),
    ]

    lights = []
    for nom_az, nom_el, base_rad in light_defs:
        # Perturbation
        az = nom_az + rng.uniform(-10.0, 10.0)
        el = nom_el + rng.uniform(-5.0, 5.0)
        dist = light_distance * rng.uniform(0.9, 1.1)

        light_pos = _spherical_to_cartesian(center, dist, az, el)
        normal = _direction_toward(light_pos, center)

        corner, edge_a, edge_b = _make_quad_facing(light_pos, normal,
                                                    light_size, light_size)

        # Radiance with ±15% variation
        radiance = [r * rng.uniform(0.85, 1.15) for r in base_rad]

        lights.append({
            "corner": corner,
            "edge_a": edge_a,
            "edge_b": edge_b,
            "radiance": radiance,
            "two_sided": False,
        })

    return lights


def generate_light_rigs(
    scenes_dir: str | list[str],
    output_dir: str,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Generate light rig JSONs for all scenes.

    Returns dict mapping scene_name -> list of generated rig file paths.
    """
    rng = random.Random(seed)
    scenes = _discover_scenes(scenes_dir)

    if not scenes:
        dirs_str = ', '.join(scenes_dir) if isinstance(scenes_dir, list) else scenes_dir
        print(f"No scenes found in {dirs_str}", file=sys.stderr)
        return {}

    results: dict[str, list[str]] = {}

    for scene_name, scene_path in scenes:
        # Skip scenes with built-in emissive light sources
        if _scene_has_emissive_lights(scene_path):
            print(f"  {scene_name}: skipped (built-in light sources)")
            continue

        try:
            bbox_min, bbox_max = compute_bounding_box(scene_path)
        except (ValueError, OSError) as e:
            print(f"  WARNING: {scene_name}: {e}", file=sys.stderr)
            continue

        os.makedirs(output_dir, exist_ok=True)

        rig_paths = []

        # Overhead rig
        overhead = generate_overhead_rig(bbox_min, bbox_max, rng)
        overhead_path = os.path.join(output_dir, f"{scene_name}_overhead.json")
        with open(overhead_path, "w", encoding="utf-8") as f:
            json.dump(overhead, f, indent=2)
        rig_paths.append(overhead_path)

        # Key-fill-rim rig
        kfr = generate_key_fill_rim_rig(bbox_min, bbox_max, rng)
        kfr_path = os.path.join(output_dir, f"{scene_name}_keyfillrim.json")
        with open(kfr_path, "w", encoding="utf-8") as f:
            json.dump(kfr, f, indent=2)
        rig_paths.append(kfr_path)

        results[scene_name] = rig_paths
        print(f"  {scene_name}: 2 rigs -> {output_dir}/")

    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scenes_root = os.path.join(script_dir, "..", "..", "scenes")
    default_scenes = [
        os.path.join(scenes_root, "khronos"),
        os.path.join(scenes_root, "training"),
    ]
    default_output = os.path.join(script_dir, "..", "light_rigs")

    parser = argparse.ArgumentParser(
        description="Generate area light rig JSONs for training scenes")
    parser.add_argument("--scenes-dir", nargs="+", default=default_scenes,
                        help="Scene directories (default: scenes/khronos/ scenes/training/)")
    parser.add_argument("--output", default=default_output,
                        help="Output directory for light rig JSONs (default: light_rigs/)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    print("=== Light Rig Generation ===")
    results = generate_light_rigs(args.scenes_dir, args.output, args.seed)

    total_rigs = sum(len(v) for v in results.values())
    print(f"\nDone: {len(results)} scenes, {total_rigs} rig files")


if __name__ == "__main__":
    main()
