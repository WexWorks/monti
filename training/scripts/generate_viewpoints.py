"""Generate camera viewpoints for training scene data generation.

Computes bounding boxes from GLB/glTF accessor min/max, then generates orbit
and hemisphere camera viewpoints around each scene. Outputs JSON files matching
the --viewpoints format expected by monti_datagen (F9-6a).

Usage:
    python scripts/generate_viewpoints.py [--scenes scenes/] [--output viewpoints/]
"""

import argparse
import json
import math
import os
import re
import struct
import sys
from typing import Optional


def compute_bounding_box_glb(glb_path: str) -> tuple[list[float], list[float]]:
    """Extract scene AABB from GLB accessor min/max for POSITION attributes.

    Returns (aabb_min, aabb_max) as 3-element lists.
    """
    with open(glb_path, "rb") as f:
        # GLB header: magic(4) + version(4) + length(4)
        magic = f.read(4)
        if magic != b"glTF":
            raise ValueError(f"Not a GLB file: {glb_path}")

        _version, _length = struct.unpack("<II", f.read(8))

        # First chunk must be JSON
        chunk_length, chunk_type = struct.unpack("<II", f.read(8))
        if chunk_type != 0x4E4F534A:  # "JSON" in little-endian
            raise ValueError("First GLB chunk is not JSON")

        json_data = f.read(chunk_length)

    gltf = json.loads(json_data)
    return _extract_position_bounds(gltf)


def compute_bounding_box_gltf(gltf_path: str) -> tuple[list[float], list[float]]:
    """Extract scene AABB from glTF accessor min/max for POSITION attributes.

    Returns (aabb_min, aabb_max) as 3-element lists.
    """
    with open(gltf_path, "r", encoding="utf-8") as f:
        gltf = json.load(f)

    return _extract_position_bounds(gltf)


def _extract_position_bounds(gltf: dict) -> tuple[list[float], list[float]]:
    """Extract scene-wide AABB from POSITION accessor min/max in a parsed glTF."""
    scene_min = [float("inf")] * 3
    scene_max = [float("-inf")] * 3

    accessors = gltf.get("accessors", [])
    meshes = gltf.get("meshes", [])

    # Find all POSITION accessors referenced by mesh primitives
    position_accessor_indices: set[int] = set()
    for mesh in meshes:
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            if "POSITION" in attrs:
                position_accessor_indices.add(attrs["POSITION"])

    if not position_accessor_indices:
        raise ValueError("No POSITION attributes found in any mesh primitive")

    for idx in position_accessor_indices:
        accessor = accessors[idx]
        acc_min = accessor.get("min")
        acc_max = accessor.get("max")

        if acc_min is None or acc_max is None:
            continue
        if len(acc_min) < 3 or len(acc_max) < 3:
            continue

        for i in range(3):
            scene_min[i] = min(scene_min[i], acc_min[i])
            scene_max[i] = max(scene_max[i], acc_max[i])

    if any(v == float("inf") for v in scene_min):
        raise ValueError("Could not determine bounding box: no valid accessor min/max")

    return scene_min, scene_max


def compute_bounding_box(scene_path: str) -> tuple[list[float], list[float]]:
    """Compute scene AABB from a .glb or .gltf file.

    Returns (aabb_min, aabb_max) as 3-element lists.
    """
    if scene_path.lower().endswith(".glb"):
        return compute_bounding_box_glb(scene_path)
    elif scene_path.lower().endswith(".gltf"):
        return compute_bounding_box_gltf(scene_path)
    else:
        raise ValueError(f"Unsupported file format: {scene_path}")


def compute_orbit_viewpoints(
    center: list[float],
    radius: float,
    num_views: int,
    elevation_deg: float,
) -> list[dict]:
    """Generate evenly-spaced orbit camera positions around a center point.

    Args:
        center: [x, y, z] look-at target
        radius: Distance from center to camera
        num_views: Number of viewpoints around the orbit
        elevation_deg: Camera elevation angle in degrees (0 = horizontal,
                       positive = above, negative = below)

    Returns:
        List of {"position": [x,y,z], "target": [x,y,z]} dicts.
    """
    viewpoints = []
    elev_rad = math.radians(elevation_deg)
    cos_elev = math.cos(elev_rad)
    sin_elev = math.sin(elev_rad)

    for i in range(num_views):
        angle = 2.0 * math.pi * i / num_views
        x = center[0] + radius * cos_elev * math.cos(angle)
        y = center[1] + radius * sin_elev
        z = center[2] + radius * cos_elev * math.sin(angle)
        viewpoints.append({
            "position": [x, y, z],
            "target": list(center),
        })

    return viewpoints


def compute_hemisphere_viewpoints(
    center: list[float],
    radius: float,
    num_views: int,
) -> list[dict]:
    """Generate quasi-uniform viewpoints on a hemisphere using Fibonacci spiral.

    Generates points on the upper hemisphere (y >= center_y) for a natural
    camera distribution above the scene.

    Args:
        center: [x, y, z] look-at target
        radius: Distance from center to camera
        num_views: Number of viewpoints to generate

    Returns:
        List of {"position": [x,y,z], "target": [x,y,z]} dicts.
    """
    viewpoints = []
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0

    for i in range(num_views):
        # Fibonacci sphere sampling on upper hemisphere
        # theta: azimuthal angle, phi: polar angle from zenith
        theta = 2.0 * math.pi * i / golden_ratio
        # Map i to upper hemisphere: cos(phi) in [0, 1]
        cos_phi = 1.0 - (i + 0.5) / num_views
        sin_phi = math.sqrt(max(0.0, 1.0 - cos_phi * cos_phi))

        x = center[0] + radius * sin_phi * math.cos(theta)
        y = center[1] + radius * cos_phi
        z = center[2] + radius * sin_phi * math.sin(theta)

        viewpoints.append({
            "position": [x, y, z],
            "target": list(center),
        })

    return viewpoints


# Cornell box uses hardcoded parameters since it's a programmatic scene
_CORNELL_BOX_CONFIG = {
    "center": [0.0, 1.0, 0.0],
    "radius": 3.5,
    "orbit_views": 8,
    "elevations": [0.0, 20.0, -10.0],
}

# Default viewpoint generation parameters for auto-computed scenes
_DEFAULT_ORBIT_VIEWS = 8
_DEFAULT_ELEVATIONS = [0.0, 20.0, -10.0]
_DEFAULT_RADIUS_MULTIPLIER = 2.5  # radius = multiplier × half-diagonal


def _scene_name_from_path(path: str) -> str:
    """Derive a scene name from a file path.

    GLB: "DamagedHelmet.glb" -> "damaged_helmet"
    glTF directory: "FlightHelmet/FlightHelmet.gltf" -> "flight_helmet"
    """
    basename = os.path.splitext(os.path.basename(path))[0]
    # Convert PascalCase/camelCase to snake_case
    # Insert underscore between lowercase and uppercase: "damagedHelmet" -> "damaged_Helmet"
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", basename)
    # Insert underscore between consecutive uppercase and uppercase+lowercase:
    # "ABeautiful" -> "A_Beautiful"
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    return s.lower()


def _center_and_radius_from_aabb(
    aabb_min: list[float], aabb_max: list[float]
) -> tuple[list[float], float]:
    """Compute scene center and orbit radius from AABB."""
    center = [(aabb_min[i] + aabb_max[i]) / 2.0 for i in range(3)]
    extents = [aabb_max[i] - aabb_min[i] for i in range(3)]
    half_diagonal = math.sqrt(sum(e * e for e in extents)) / 2.0
    radius = _DEFAULT_RADIUS_MULTIPLIER * half_diagonal
    # Ensure a minimum radius so we don't end up inside tiny models
    radius = max(radius, 0.5)
    return center, radius


def generate_viewpoints_for_scene(
    scene_path: str,
    scene_name: Optional[str] = None,
) -> list[dict]:
    """Generate viewpoints for a single scene.

    For cornell_box.glb, uses hardcoded parameters. For all other scenes,
    auto-computes center/radius from the GLB/glTF bounding box.

    Returns list of {"position": [x,y,z], "target": [x,y,z]} dicts.
    """
    if scene_name is None:
        scene_name = _scene_name_from_path(scene_path)

    if scene_name == "cornell_box":
        cfg = _CORNELL_BOX_CONFIG
        viewpoints = []
        for elev in cfg["elevations"]:
            viewpoints.extend(compute_orbit_viewpoints(
                cfg["center"], cfg["radius"], cfg["orbit_views"], elev
            ))
        return viewpoints

    # Auto-compute from bounding box
    aabb_min, aabb_max = compute_bounding_box(scene_path)
    center, radius = _center_and_radius_from_aabb(aabb_min, aabb_max)

    viewpoints = []
    for elev in _DEFAULT_ELEVATIONS:
        viewpoints.extend(compute_orbit_viewpoints(
            center, radius, _DEFAULT_ORBIT_VIEWS, elev
        ))

    return viewpoints


def _discover_scenes(scenes_dir: str) -> list[tuple[str, str]]:
    """Discover all downloadable scenes in the scenes directory.

    Returns list of (scene_name, scene_path) tuples.
    """
    scenes = []

    # GLB files in scenes/ root
    for entry in sorted(os.listdir(scenes_dir)):
        path = os.path.join(scenes_dir, entry)
        if entry.lower().endswith(".glb") and os.path.isfile(path):
            scenes.append((_scene_name_from_path(entry), path))

    # Multi-file glTF subdirectories (e.g., FlightHelmet/, Sponza/)
    for entry in sorted(os.listdir(scenes_dir)):
        subdir = os.path.join(scenes_dir, entry)
        if not os.path.isdir(subdir):
            continue
        # Look for a .gltf file with the same name as the directory
        gltf_path = os.path.join(subdir, f"{entry}.gltf")
        if os.path.isfile(gltf_path):
            scenes.append((_scene_name_from_path(entry), gltf_path))

    return scenes


def generate_all_viewpoints(
    scenes_dir: str, output_dir: str
) -> dict[str, int]:
    """Generate viewpoint JSONs for all scenes in scenes_dir.

    Returns dict mapping scene_name -> viewpoint_count.
    """
    os.makedirs(output_dir, exist_ok=True)

    scenes = _discover_scenes(scenes_dir)
    if not scenes:
        print(f"No scenes found in {scenes_dir}", file=sys.stderr)
        return {}

    results: dict[str, int] = {}

    for scene_name, scene_path in scenes:
        try:
            viewpoints = generate_viewpoints_for_scene(scene_path, scene_name)
        except (ValueError, OSError) as e:
            print(f"  WARNING: {scene_name}: {e}", file=sys.stderr)
            continue

        output_path = os.path.join(output_dir, f"{scene_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(viewpoints, f, indent=2)

        results[scene_name] = len(viewpoints)
        print(f"  {scene_name}: {len(viewpoints)} viewpoints -> {output_path}")

    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_scenes = os.path.join(script_dir, "..", "scenes")
    default_output = os.path.join(script_dir, "..", "viewpoints")

    parser = argparse.ArgumentParser(
        description="Generate camera viewpoints for training scenes")
    parser.add_argument("--scenes", default=default_scenes,
                        help="Scenes directory (default: scenes/)")
    parser.add_argument("--output", default=default_output,
                        help="Output directory for viewpoint JSONs (default: viewpoints/)")
    args = parser.parse_args()

    print("=== Viewpoint Generation ===")
    results = generate_all_viewpoints(args.scenes, args.output)

    total_vps = sum(results.values())
    print(f"\nDone: {len(results)} scenes, {total_vps} total viewpoints")


if __name__ == "__main__":
    main()
