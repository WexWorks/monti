"""Generate camera viewpoints for training scene data generation.

Computes bounding boxes from GLB/glTF accessor min/max, then generates orbit
and hemisphere camera viewpoints around each scene. Outputs JSON files matching
the --viewpoints format expected by monti_datagen (F9-6a).

Supports optional seed viewpoints (hand-authored via monti_view) with random
variation generation for richer training data.

Usage:
    python scripts/generate_viewpoints.py [--scenes scenes/] [--output viewpoints/]
    python scripts/generate_viewpoints.py --seeds viewpoints/manual/ --variations-per-seed 4
"""

import argparse
import json
import math
import os
import random
import struct
import sys
import uuid
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


def _parse_gltf_json_from_glb(glb_path: str) -> dict:
    """Parse the JSON chunk from a GLB file. Returns empty dict on failure."""
    with open(glb_path, "rb") as f:
        magic = f.read(4)
        if magic != b"glTF":
            return {}
        _version, _length = struct.unpack("<II", f.read(8))
        chunk_length, chunk_type = struct.unpack("<II", f.read(8))
        if chunk_type != 0x4E4F534A:  # "JSON" in little-endian
            return {}
        json_data = f.read(chunk_length)
    return json.loads(json_data)


def _scene_has_emissive_lights(scene_path: str) -> bool:
    """Check if a glTF/GLB scene contains bright emissive materials.

    Detects materials where emissiveFactor × emissiveStrength > 1.0,
    indicating an intentional area light source rather than decorative glow.
    """
    try:
        if scene_path.lower().endswith(".glb"):
            gltf = _parse_gltf_json_from_glb(scene_path)
        elif scene_path.lower().endswith(".gltf"):
            with open(scene_path, "r", encoding="utf-8") as f:
                gltf = json.load(f)
        else:
            return False

        for material in gltf.get("materials", []):
            emissive = material.get("emissiveFactor", [0.0, 0.0, 0.0])
            if not any(v > 0.0 for v in emissive):
                continue
            extensions = material.get("extensions", {})
            strength_ext = extensions.get("KHR_materials_emissive_strength", {})
            strength = strength_ext.get("emissiveStrength", 1.0)
            max_radiance = max(v * strength for v in emissive)
            if max_radiance > 1.0:
                return True
    except (ValueError, OSError, json.JSONDecodeError, KeyError, struct.error):
        return False
    return False


def _discover_envs(envs_dir: str) -> list[str]:
    """Discover all .exr files in envs_dir, sorted alphabetically.

    Returns list of absolute paths.
    """
    if not os.path.isdir(envs_dir):
        return []
    exrs = sorted(
        f for f in os.listdir(envs_dir) if f.lower().endswith(".exr")
    )
    return [os.path.abspath(os.path.join(envs_dir, f)) for f in exrs]


def _discover_light_rigs_for_scene(
    lights_dir: str, scene_name: str
) -> list[tuple[str, str]]:
    """Discover light rig JSON files for a scene using flat naming convention.

    Looks for files matching ``<scene_name>_<rig_type>.json`` in lights_dir.
    Returns list of (rig_type, absolute_path) tuples, sorted by rig type.
    """
    if not os.path.isdir(lights_dir):
        return []
    prefix = f"{scene_name}_"
    rigs = []
    for f in sorted(os.listdir(lights_dir)):
        if f.startswith(prefix) and f.lower().endswith(".json"):
            rig_type = f[len(prefix):-len(".json")]
            rigs.append((rig_type, os.path.abspath(os.path.join(lights_dir, f))))
    return rigs


def _assign_environment_and_lights(
    viewpoints: list[dict],
    scene_name: str,
    envs: list[str],
    rigs: list[tuple[str, str]],
) -> None:
    """Assign environment map or light rig to viewpoints without existing assignment.

    Each viewpoint receives either an environment map OR a light rig (not both).
    Assignment cycles through available options deterministically seeded by scene name.
    """
    pool: list[tuple[str, str]] = []
    for env_path in envs:
        pool.append(("environment", env_path))
    for _, rig_path in rigs:
        pool.append(("lights", rig_path))

    if not pool:
        return

    rng = random.Random(scene_name)

    for vp in viewpoints:
        if "environment" in vp or "lights" in vp:
            continue
        choice_type, choice_path = rng.choice(pool)
        vp[choice_type] = choice_path


def _amplify_env_intensities(
    viewpoints: list[dict], intensities: list[float]
) -> list[dict]:
    """Expand environment-lit viewpoints into copies at different intensity levels.

    Viewpoints that already have an "environmentIntensity" field or that use
    light rigs (no "environment" key) are kept unchanged.
    Each eligible viewpoint produces len(intensities) copies.
    """
    if not intensities:
        return viewpoints
    result = []
    for vp in viewpoints:
        if "environmentIntensity" in vp or "environment" not in vp:
            result.append(vp)
            continue
        for intensity in intensities:
            amplified = dict(vp)
            amplified["environmentIntensity"] = intensity
            result.append(amplified)
    return result


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


# ---------------------------------------------------------------------------
# Seed viewpoint variation generation
# ---------------------------------------------------------------------------

def _vec3_distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _vec3_add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(3)]


def _vec3_lerp(a: list[float], b: list[float], t: float) -> list[float]:
    return [a[i] + t * (b[i] - a[i]) for i in range(3)]


def _random_in_sphere(rng: random.Random, radius: float) -> list[float]:
    """Generate a random point uniformly within a sphere of given radius."""
    while True:
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)
        z = rng.uniform(-1.0, 1.0)
        if x * x + y * y + z * z <= 1.0:
            return [x * radius, y * radius, z * radius]


def _cartesian_to_spherical(
    position: list[float], target: list[float]
) -> tuple[float, float, float]:
    """Convert position to spherical coordinates (azimuth, elevation, distance) around target."""
    dx = position[0] - target[0]
    dy = position[1] - target[1]
    dz = position[2] - target[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist < 1e-12:
        return 0.0, 0.0, 0.0
    azimuth = math.atan2(dz, dx)
    elevation = math.asin(max(-1.0, min(1.0, dy / dist)))
    return azimuth, elevation, dist


def _spherical_to_cartesian(
    target: list[float], azimuth: float, elevation: float, dist: float
) -> list[float]:
    """Convert spherical coordinates around target back to Cartesian position."""
    cos_elev = math.cos(elevation)
    x = target[0] + dist * cos_elev * math.cos(azimuth)
    y = target[1] + dist * math.sin(elevation)
    z = target[2] + dist * cos_elev * math.sin(azimuth)
    return [x, y, z]


def _vary_position_jitter(
    seed: dict, rng: random.Random, jitter_frac: float
) -> dict:
    """Jitter camera position; keep target fixed."""
    dist = _vec3_distance(seed["position"], seed["target"])
    offset = _random_in_sphere(rng, jitter_frac * dist)
    vp = {
        "position": _vec3_add(seed["position"], offset),
        "target": list(seed["target"]),
    }
    for field in ("fov", "environment", "lights", "environmentBlur", "environmentIntensity"):
        if field in seed:
            vp[field] = seed[field]
    return vp


def _vary_target_jitter(
    seed: dict, rng: random.Random
) -> dict:
    """Jitter look-at target (±5% of camera-to-target distance); keep camera position fixed."""
    dist = _vec3_distance(seed["position"], seed["target"])
    target_jitter = 0.05
    offset = _random_in_sphere(rng, target_jitter * dist)
    vp = {
        "position": list(seed["position"]),
        "target": _vec3_add(seed["target"], offset),
    }
    for field in ("fov", "environment", "lights", "environmentBlur", "environmentIntensity"):
        if field in seed:
            vp[field] = seed[field]
    return vp


def _vary_interpolation(
    seed_a: dict, seed_b: dict, rng: random.Random
) -> dict:
    """Interpolate position and target between two seed viewpoints."""
    t = rng.uniform(0.2, 0.8)
    vp = {
        "position": _vec3_lerp(seed_a["position"], seed_b["position"], t),
        "target": _vec3_lerp(seed_a["target"], seed_b["target"], t),
    }
    # Blend FOV if both seeds have it
    if "fov" in seed_a and "fov" in seed_b:
        vp["fov"] = seed_a["fov"] + t * (seed_b["fov"] - seed_a["fov"])
    elif "fov" in seed_a:
        vp["fov"] = seed_a["fov"]
    for field in ("environment", "lights", "environmentBlur", "environmentIntensity"):
        if field in seed_a:
            vp[field] = seed_a[field]
        elif field in seed_b:
            vp[field] = seed_b[field]
    return vp


def _vary_orbit_perturbation(
    seed: dict, rng: random.Random
) -> dict:
    """Perturb azimuth, elevation, and distance in spherical coordinates around the target."""
    azimuth, elevation, dist = _cartesian_to_spherical(
        seed["position"], seed["target"]
    )
    azimuth += math.radians(rng.uniform(-15.0, 15.0))
    elevation += math.radians(rng.uniform(-10.0, 10.0))
    # Clamp elevation to avoid flipping
    elevation = max(math.radians(-89.0), min(math.radians(89.0), elevation))
    dist *= rng.uniform(0.9, 1.1)

    vp = {
        "position": _spherical_to_cartesian(seed["target"], azimuth, elevation, dist),
        "target": list(seed["target"]),
    }
    if "fov" in seed:
        vp["fov"] = seed["fov"] + rng.uniform(-2.0, 2.0)
    for field in ("environment", "lights", "environmentBlur", "environmentIntensity"):
        if field in seed:
            vp[field] = seed[field]
    return vp


def generate_seed_variations(
    seeds: list[dict],
    variations_per_seed: int = 4,
    jitter_frac: float = 0.15,
    rng_seed: str = "",
) -> list[dict]:
    """Generate viewpoint variations from seed viewpoints.

    Each seed is included verbatim in the output. Additional variations are
    generated to reach ``variations_per_seed`` total viewpoints per seed.

    Args:
        seeds: List of seed viewpoints (each with "position", "target", optional "fov").
        variations_per_seed: Target total viewpoints per seed (including the original).
        jitter_frac: Max position jitter as fraction of camera-to-target distance.
        rng_seed: Seed string for reproducible random generation.

    Returns:
        List of viewpoints including originals and variations.
    """
    if not seeds:
        return []

    rng = random.Random(rng_seed)
    result: list[dict] = []

    # Strategies that work with a single seed
    single_strategies = ["position_jitter", "target_jitter", "orbit_perturbation"]
    # Interpolation requires >= 2 seeds
    has_interpolation = len(seeds) >= 2

    for seed_idx, seed in enumerate(seeds):
        # Always include the original seed
        result.append(dict(seed))

        # Generate additional variations
        extras_needed = max(0, variations_per_seed - 1)
        for _ in range(extras_needed):
            if has_interpolation:
                strategy = rng.choice(single_strategies + ["interpolation"])
            else:
                strategy = rng.choice(single_strategies)

            if strategy == "position_jitter":
                result.append(_vary_position_jitter(seed, rng, jitter_frac))
            elif strategy == "target_jitter":
                result.append(_vary_target_jitter(seed, rng))
            elif strategy == "orbit_perturbation":
                result.append(_vary_orbit_perturbation(seed, rng))
            elif strategy == "interpolation":
                # Pick a different seed to interpolate with
                other_idx = rng.choice(
                    [i for i in range(len(seeds)) if i != seed_idx]
                )
                result.append(
                    _vary_interpolation(seed, seeds[other_idx], rng)
                )

    return result


def load_seed_viewpoints(seeds_dir: str) -> dict[str, list[dict]]:
    """Load seed viewpoint JSON files from a directory.

    Returns a dict mapping scene_name -> list of seed viewpoints.
    File names are matched by convention: ``sponza.json`` matches the ``sponza`` scene.
    """
    seed_map: dict[str, list[dict]] = {}
    if not os.path.isdir(seeds_dir):
        return seed_map

    for entry in sorted(os.listdir(seeds_dir)):
        if not entry.lower().endswith(".json"):
            continue
        scene_name = os.path.splitext(entry)[0]
        path = os.path.join(seeds_dir, entry)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            seed_map[scene_name] = data

    return seed_map


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
_DEFAULT_RADIUS_MULTIPLIER = 1.5  # radius = multiplier × half-diagonal
_DEFAULT_ZOOM_FACTORS = [0.8, 1.0, 1.4]  # Relative zoom levels for framing variety


def _assign_viewpoint_ids(viewpoints: list[dict]) -> None:
    """Assign a unique 8-hex-char ID to each viewpoint that lacks one.

    Modifies viewpoints in place. Existing IDs are preserved.
    On the extremely unlikely collision, regenerates the colliding ID.
    """
    existing_ids: set[str] = set()
    for vp in viewpoints:
        if "id" in vp:
            existing_ids.add(vp["id"])

    for vp in viewpoints:
        if "id" in vp:
            continue
        while True:
            new_id = uuid.uuid4().hex[:8]
            if new_id not in existing_ids:
                break
        vp["id"] = new_id
        existing_ids.add(new_id)


def _scene_name_from_path(path: str) -> str:
    """Derive a scene name from a file path.

    For ``.glb`` files the stem is always used directly::

        DamagedHelmet.glb  ->  DamagedHelmet

    For ``.gltf`` files that live inside a directory with a companion
    ``.bin`` file (multi-file glTF scenes), the filename stem is compared
    against the parent directory name.  When they differ — e.g.
    ``BistroInterior/scene.gltf`` or ``Brutalism/BrutalistHall.gltf`` —
    the parent directory name is used instead, because it is a more
    meaningful identifier for the scene::

        BistroInterior/scene.gltf             ->  BistroInterior
        Brutalism/BrutalistHall.gltf          ->  Brutalism
        FlightHelmet/FlightHelmet.gltf        ->  FlightHelmet   (stem == dir)
    """
    stem = os.path.splitext(os.path.basename(path))[0]

    if path.lower().endswith(".gltf"):
        parent = os.path.dirname(path)
        if parent:
            dir_name = os.path.basename(parent)
            # Multi-file glTF: companion .bin exists alongside the .gltf
            bin_path = os.path.join(parent, stem + ".bin")
            if os.path.isfile(bin_path) and stem != dir_name:
                return dir_name

    return stem


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
    seeds: Optional[list[dict]] = None,
    variations_per_seed: int = 4,
    seed_jitter: float = 0.15,
) -> list[dict]:
    """Generate viewpoints for a single scene.

    When *seeds* are provided, generates variations from them instead of
    auto-generated orbit viewpoints. Otherwise falls back to the existing
    auto-generation (hardcoded config for cornell_box, bounding-box orbit
    for everything else).

    Returns list of {"position": [x,y,z], "target": [x,y,z]} dicts.
    """
    if scene_name is None:
        scene_name = _scene_name_from_path(scene_path)

    # Seed viewpoints replace auto-generation for this scene
    if seeds:
        return generate_seed_variations(
            seeds,
            variations_per_seed=variations_per_seed,
            jitter_frac=seed_jitter,
            rng_seed=scene_name,
        )

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
    center, base_radius = _center_and_radius_from_aabb(aabb_min, aabb_max)

    viewpoints = []
    for zoom in _DEFAULT_ZOOM_FACTORS:
        radius = max(base_radius * zoom, 0.5)
        for elev in _DEFAULT_ELEVATIONS:
            viewpoints.extend(compute_orbit_viewpoints(
                center, radius, _DEFAULT_ORBIT_VIEWS, elev
            ))

    return viewpoints


def _discover_scenes(scenes_dirs: str | list[str]) -> list[tuple[str, str]]:
    """Discover all downloadable scenes in one or more scene directories.

    Returns list of (scene_name, scene_path) tuples.  When multiple directories
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

        # Multi-file glTF subdirectories (e.g., FlightHelmet/, Sponza/)
        for entry in sorted(os.listdir(scenes_dir)):
            subdir = os.path.join(scenes_dir, entry)
            if not os.path.isdir(subdir):
                continue
            # Prefer a .gltf file matching the directory name
            gltf_path = os.path.join(subdir, f"{entry}.gltf")
            if not os.path.isfile(gltf_path):
                # Fall back to any single .gltf file in the directory
                gltf_files = [
                    f for f in os.listdir(subdir)
                    if f.lower().endswith(".gltf") and os.path.isfile(
                        os.path.join(subdir, f))
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


def generate_all_viewpoints(
    scenes_dir: str | list[str],
    output_dir: str,
    seeds_dir: Optional[str] = None,
    variations_per_seed: int = 4,
    seed_jitter: float = 0.15,
    envs_dir: Optional[str] = None,
    lights_dir: Optional[str] = None,
    env_intensities: Optional[list[float]] = None,
) -> dict[str, int]:
    """Generate viewpoint JSONs for all scenes in scenes_dir.

    When *seeds_dir* is provided, seed viewpoint files are loaded and matched
    to scenes by name. Scenes with matching seeds use variation generation;
    scenes without seeds fall back to auto-generation.

    When *envs_dir* and/or *lights_dir* are provided, each viewpoint is assigned
    either an environment map or a light rig (not both). Emissive scenes are
    skipped for environment/lights assignment.

    Returns dict mapping scene_name -> viewpoint_count.
    """
    os.makedirs(output_dir, exist_ok=True)

    scenes = _discover_scenes(scenes_dir)
    if not scenes:
        dirs_str = ', '.join(scenes_dir) if isinstance(scenes_dir, list) else scenes_dir
        print(f"No scenes found in {dirs_str}", file=sys.stderr)
        return {}

    seed_map: dict[str, list[dict]] = {}
    if seeds_dir:
        seed_map = load_seed_viewpoints(seeds_dir)
        if seed_map:
            print(f"  Loaded seeds for: {', '.join(sorted(seed_map.keys()))}")

    # Discover environment maps
    envs: list[str] = []
    if envs_dir:
        envs = _discover_envs(envs_dir)
        if envs:
            print(f"  Environments: {len(envs)} HDRIs from {envs_dir}")

    if env_intensities:
        print(f"  Env intensities: {len(env_intensities)} levels "
              f"({', '.join(f'{v:.1f}' for v in env_intensities)})")

    results: dict[str, int] = {}

    for scene_name, scene_path in scenes:
        try:
            scene_seeds = seed_map.get(scene_name)
            viewpoints = generate_viewpoints_for_scene(
                scene_path,
                scene_name,
                seeds=scene_seeds,
                variations_per_seed=variations_per_seed,
                seed_jitter=seed_jitter,
            )
        except (ValueError, OSError) as e:
            print(f"  WARNING: {scene_name}: {e}", file=sys.stderr)
            continue

        # Assign environment/lights (skip emissive scenes)
        is_emissive = _scene_has_emissive_lights(scene_path)
        if not is_emissive:
            rigs = _discover_light_rigs_for_scene(lights_dir, scene_name) if lights_dir else []
            _assign_environment_and_lights(viewpoints, scene_name, envs, rigs)

        # Amplify environment intensities
        if env_intensities:
            viewpoints = _amplify_env_intensities(viewpoints, env_intensities)

        # Assign unique IDs
        _assign_viewpoint_ids(viewpoints)

        output_path = os.path.join(output_dir, f"{scene_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(viewpoints, f, indent=2)

        source = "seeds" if scene_name in seed_map else "auto"
        emissive_tag = " [emissive]" if is_emissive else ""
        results[scene_name] = len(viewpoints)
        print(f"  {scene_name}: {len(viewpoints)} viewpoints ({source}){emissive_tag} -> {output_path}")

    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scenes_root = os.path.join(script_dir, "..", "..", "scenes")
    default_scenes = [
        os.path.join(scenes_root, "khronos"),
        os.path.join(scenes_root, "training"),
    ]
    default_output = os.path.join(script_dir, "..", "viewpoints")
    default_envs = os.path.join(script_dir, "..", "environments")
    default_lights = os.path.join(script_dir, "..", "light_rigs")

    parser = argparse.ArgumentParser(
        description="Generate camera viewpoints for training scenes")
    parser.add_argument("--scenes", nargs="+", default=default_scenes,
                        help="Scene directories (default: scenes/khronos/ scenes/training/)")
    parser.add_argument("--output", default=default_output,
                        help="Output directory for viewpoint JSONs (default: viewpoints/)")
    parser.add_argument("--seeds", default=None,
                        help="Directory of seed viewpoint JSONs (matched to scenes by name)")
    parser.add_argument("--variations-per-seed", type=int, default=4,
                        help="Target total viewpoints per seed including original (default: 4)")
    parser.add_argument("--seed-jitter", type=float, default=0.15,
                        help="Max position jitter as fraction of camera-to-target distance (default: 0.15)")
    parser.add_argument("--envs-dir", default=None,
                        help="Directory of .exr environment maps (default: environments/ if it exists)")
    parser.add_argument("--lights-dir", default=None,
                        help="Directory of light rig JSONs (default: light_rigs/ if it exists)")
    parser.add_argument("--env-intensities", type=float, nargs="+",
                        default=None,
                        help="Environment intensity multipliers for amplification "
                             "(e.g., 1.0 3.0 10.0)")
    args = parser.parse_args()

    # Use default directories if they exist and user didn't specify
    envs_dir = args.envs_dir
    if envs_dir is None and os.path.isdir(default_envs):
        envs_dir = default_envs

    lights_dir = args.lights_dir
    if lights_dir is None and os.path.isdir(default_lights):
        lights_dir = default_lights

    print("=== Viewpoint Generation ===")
    results = generate_all_viewpoints(
        args.scenes,
        args.output,
        seeds_dir=args.seeds,
        variations_per_seed=args.variations_per_seed,
        seed_jitter=args.seed_jitter,
        envs_dir=envs_dir,
        lights_dir=lights_dir,
        env_intensities=args.env_intensities,
    )

    total_vps = sum(results.values())
    print(f"\nDone: {len(results)} scenes, {total_vps} total viewpoints")


if __name__ == "__main__":
    main()
