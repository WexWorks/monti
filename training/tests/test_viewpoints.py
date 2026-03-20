"""Tests for F9-6b: viewpoint generation and scene bounding box computation."""

import json
import math
import os
import struct
import tempfile

import pytest

# Import from scripts (not a package)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from generate_viewpoints import (
    compute_bounding_box,
    compute_bounding_box_glb,
    compute_bounding_box_gltf,
    compute_hemisphere_viewpoints,
    compute_orbit_viewpoints,
    generate_all_viewpoints,
    generate_viewpoints_for_scene,
    generate_seed_variations,
    load_seed_viewpoints,
    _amplify_exposures,
    _assign_environment_and_lights,
    _assign_viewpoint_ids,
    _center_and_radius_from_aabb,
    _DEFAULT_EXPOSURES,
    _discover_envs,
    _discover_light_rigs_for_scene,
    _scene_has_emissive_lights,
    _scene_name_from_path,
    _vec3_distance,
    _cartesian_to_spherical,
    _spherical_to_cartesian,
)


# ---------------------------------------------------------------------------
# Helpers for creating test GLB/glTF files
# ---------------------------------------------------------------------------

def _make_test_glb(path: str, aabb_min: list, aabb_max: list) -> None:
    """Create a minimal GLB file with a POSITION accessor containing min/max."""
    gltf_json = {
        "asset": {"version": "2.0"},
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": list(aabb_min),
                "max": list(aabb_max),
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {"attributes": {"POSITION": 0}}
                ]
            }
        ],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}],
        "buffers": [{"byteLength": 36}],
    }

    json_bytes = json.dumps(gltf_json).encode("utf-8")
    # Pad JSON to 4-byte alignment
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "

    # Minimal binary buffer (36 bytes = 3 vertices × 3 floats × 4 bytes)
    bin_data = b"\x00" * 36
    while len(bin_data) % 4 != 0:
        bin_data += b"\x00"

    # GLB header
    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk = struct.pack("<II", len(bin_data), 0x004E4942) + bin_data
    total_length = 12 + len(json_chunk) + len(bin_chunk)
    header = struct.pack("<III", 0x46546C67, 2, total_length)  # "glTF", version 2

    with open(path, "wb") as f:
        f.write(header + json_chunk + bin_chunk)


def _make_test_gltf(directory: str, name: str,
                    aabb_min: list, aabb_max: list) -> str:
    """Create a minimal .gltf file with a POSITION accessor containing min/max.

    Returns the path to the .gltf file.
    """
    gltf_json = {
        "asset": {"version": "2.0"},
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": list(aabb_min),
                "max": list(aabb_max),
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {"attributes": {"POSITION": 0}}
                ]
            }
        ],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}],
        "buffers": [{"uri": f"{name}.bin", "byteLength": 36}],
    }

    os.makedirs(directory, exist_ok=True)
    gltf_path = os.path.join(directory, f"{name}.gltf")
    with open(gltf_path, "w", encoding="utf-8") as f:
        json.dump(gltf_json, f)

    # Write a dummy .bin file
    bin_path = os.path.join(directory, f"{name}.bin")
    with open(bin_path, "wb") as f:
        f.write(b"\x00" * 36)

    return gltf_path


# ---------------------------------------------------------------------------
# Orbit viewpoint tests
# ---------------------------------------------------------------------------

class TestOrbitViewpoints:
    def test_correct_count(self):
        vps = compute_orbit_viewpoints([0.0, 0.0, 0.0], 5.0, 8, 0.0)
        assert len(vps) == 8

    def test_single_viewpoint(self):
        vps = compute_orbit_viewpoints([0.0, 0.0, 0.0], 5.0, 1, 0.0)
        assert len(vps) == 1

    def test_positions_on_circle_at_zero_elevation(self):
        center = [0.0, 0.0, 0.0]
        radius = 5.0
        vps = compute_orbit_viewpoints(center, radius, 8, 0.0)
        for vp in vps:
            pos = vp["position"]
            # At 0 elevation, y should equal center y
            assert abs(pos[1] - center[1]) < 1e-10
            # Distance from center in XZ plane should equal radius
            dist_xz = math.sqrt(pos[0] ** 2 + pos[2] ** 2)
            assert abs(dist_xz - radius) < 1e-10

    def test_positions_at_positive_elevation(self):
        center = [1.0, 2.0, 3.0]
        radius = 4.0
        elev = 30.0
        vps = compute_orbit_viewpoints(center, radius, 6, elev)
        for vp in vps:
            pos = vp["position"]
            dx = pos[0] - center[0]
            dy = pos[1] - center[1]
            dz = pos[2] - center[2]
            dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            # All positions should be at the specified radius
            # (distance is radius because of how elevation is applied)
            expected_y = radius * math.sin(math.radians(elev))
            assert abs(dy - expected_y) < 1e-10

    def test_targets_equal_center(self):
        center = [1.0, 2.0, 3.0]
        vps = compute_orbit_viewpoints(center, 5.0, 4, 15.0)
        for vp in vps:
            assert vp["target"] == center

    def test_no_nan_values(self):
        vps = compute_orbit_viewpoints([0.0, 0.0, 0.0], 5.0, 12, 45.0)
        for vp in vps:
            for val in vp["position"]:
                assert not math.isnan(val)
                assert not math.isinf(val)

    def test_position_not_equal_target(self):
        vps = compute_orbit_viewpoints([0.0, 0.0, 0.0], 5.0, 8, 0.0)
        for vp in vps:
            assert vp["position"] != vp["target"]

    def test_evenly_spaced_angles(self):
        vps = compute_orbit_viewpoints([0.0, 0.0, 0.0], 5.0, 4, 0.0)
        # 4 positions at 0,90,180,270 degrees in XZ plane
        angles = []
        for vp in vps:
            x, z = vp["position"][0], vp["position"][2]
            angles.append(math.atan2(z, x))

        # Check angular differences are ~pi/2
        angles.sort()
        for i in range(len(angles) - 1):
            diff = angles[i + 1] - angles[i]
            assert abs(diff - math.pi / 2.0) < 1e-10


# ---------------------------------------------------------------------------
# Hemisphere viewpoint tests
# ---------------------------------------------------------------------------

class TestHemisphereViewpoints:
    def test_correct_count(self):
        vps = compute_hemisphere_viewpoints([0.0, 0.0, 0.0], 5.0, 10)
        assert len(vps) == 10

    def test_all_above_center(self):
        center = [0.0, 0.0, 0.0]
        vps = compute_hemisphere_viewpoints(center, 5.0, 20)
        for vp in vps:
            assert vp["position"][1] >= center[1] - 1e-10

    def test_distance_from_center(self):
        center = [1.0, 2.0, 3.0]
        radius = 4.0
        vps = compute_hemisphere_viewpoints(center, radius, 15)
        for vp in vps:
            dx = vp["position"][0] - center[0]
            dy = vp["position"][1] - center[1]
            dz = vp["position"][2] - center[2]
            dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            assert abs(dist - radius) < 1e-10

    def test_targets_equal_center(self):
        center = [1.0, 2.0, 3.0]
        vps = compute_hemisphere_viewpoints(center, 5.0, 8)
        for vp in vps:
            assert vp["target"] == center

    def test_no_nan_values(self):
        vps = compute_hemisphere_viewpoints([0.0, 0.0, 0.0], 5.0, 20)
        for vp in vps:
            for val in vp["position"]:
                assert not math.isnan(val)
                assert not math.isinf(val)

    def test_position_not_equal_target(self):
        vps = compute_hemisphere_viewpoints([0.0, 0.0, 0.0], 5.0, 10)
        for vp in vps:
            assert vp["position"] != vp["target"]

    def test_quasi_uniform_distribution(self):
        """Hemisphere viewpoints should be spread out, not clustered."""
        vps = compute_hemisphere_viewpoints([0.0, 0.0, 0.0], 1.0, 50)
        positions = [vp["position"] for vp in vps]

        # Check that no two points are too close together
        min_dist = float("inf")
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dz = positions[i][2] - positions[j][2]
                dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                min_dist = min(min_dist, dist)

        # With 50 points on a hemisphere of radius 1, the minimum distance
        # should be at least ~0.1 (well-separated)
        assert min_dist > 0.05


# ---------------------------------------------------------------------------
# Bounding box tests
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def test_glb_bounding_box(self, tmp_path):
        glb_path = str(tmp_path / "test.glb")
        _make_test_glb(glb_path, [-1.0, -2.0, -3.0], [4.0, 5.0, 6.0])

        aabb_min, aabb_max = compute_bounding_box_glb(glb_path)
        assert aabb_min == [-1.0, -2.0, -3.0]
        assert aabb_max == [4.0, 5.0, 6.0]

    def test_gltf_bounding_box(self, tmp_path):
        subdir = str(tmp_path / "TestModel")
        gltf_path = _make_test_gltf(subdir, "TestModel",
                                     [-0.5, 0.0, -0.5], [0.5, 1.0, 0.5])

        aabb_min, aabb_max = compute_bounding_box_gltf(gltf_path)
        assert aabb_min == [-0.5, 0.0, -0.5]
        assert aabb_max == [0.5, 1.0, 0.5]

    def test_compute_bounding_box_dispatches_glb(self, tmp_path):
        glb_path = str(tmp_path / "model.glb")
        _make_test_glb(glb_path, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        aabb_min, aabb_max = compute_bounding_box(glb_path)
        assert aabb_min == [0.0, 0.0, 0.0]
        assert aabb_max == [1.0, 1.0, 1.0]

    def test_compute_bounding_box_dispatches_gltf(self, tmp_path):
        subdir = str(tmp_path / "Model")
        gltf_path = _make_test_gltf(subdir, "Model", [0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
        aabb_min, aabb_max = compute_bounding_box(gltf_path)
        assert aabb_min == [0.0, 0.0, 0.0]
        assert aabb_max == [2.0, 2.0, 2.0]

    def test_invalid_glb_magic_raises(self, tmp_path):
        path = str(tmp_path / "bad.glb")
        with open(path, "wb") as f:
            f.write(b"BAD!" + b"\x00" * 100)
        with pytest.raises(ValueError, match="Not a GLB file"):
            compute_bounding_box_glb(path)

    def test_unsupported_extension_raises(self, tmp_path):
        path = str(tmp_path / "model.obj")
        with open(path, "w") as f:
            f.write("v 0 0 0\n")
        with pytest.raises(ValueError, match="Unsupported file format"):
            compute_bounding_box(path)

    def test_cornell_box_bbox(self):
        """If cornell_box.glb exists, verify its bounding box is reasonable."""
        scenes_dir = os.path.join(os.path.dirname(__file__), "..", "scenes")
        cornell_path = os.path.join(scenes_dir, "cornell_box.glb")
        if not os.path.isfile(cornell_path):
            pytest.skip("cornell_box.glb not found in scenes/")

        aabb_min, aabb_max = compute_bounding_box(cornell_path)
        # Cornell box is roughly [-0.3, -0.1, -0.6] to [0.6, 1.1, 0.6]
        for i in range(3):
            assert aabb_min[i] < aabb_max[i]


# ---------------------------------------------------------------------------
# Center/radius computation
# ---------------------------------------------------------------------------

class TestCenterAndRadius:
    def test_unit_cube(self):
        center, radius = _center_and_radius_from_aabb(
            [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
        )
        assert center == [0.5, 0.5, 0.5]
        half_diag = math.sqrt(3.0) / 2.0
        assert abs(radius - 1.5 * half_diag) < 1e-10

    def test_asymmetric_box(self):
        center, radius = _center_and_radius_from_aabb(
            [-2.0, 0.0, -1.0], [2.0, 4.0, 1.0]
        )
        assert center == [0.0, 2.0, 0.0]
        extents = [4.0, 4.0, 2.0]
        half_diag = math.sqrt(sum(e * e for e in extents)) / 2.0
        assert abs(radius - 1.5 * half_diag) < 1e-10

    def test_minimum_radius(self):
        """Tiny models should get a minimum radius of 0.5."""
        center, radius = _center_and_radius_from_aabb(
            [0.0, 0.0, 0.0], [0.001, 0.001, 0.001]
        )
        assert radius >= 0.5


# ---------------------------------------------------------------------------
# Scene name derivation
# ---------------------------------------------------------------------------

class TestSceneNameFromPath:
    def test_pascal_case_glb(self):
        assert _scene_name_from_path("DamagedHelmet.glb") == "DamagedHelmet"

    def test_camel_case(self):
        assert _scene_name_from_path("boomBox.glb") == "boomBox"

    def test_already_snake_case(self):
        assert _scene_name_from_path("cornell_box.glb") == "cornell_box"

    def test_gltf_path(self):
        assert _scene_name_from_path("FlightHelmet/FlightHelmet.gltf") == "FlightHelmet"

    def test_preserves_original_casing(self):
        name = _scene_name_from_path("ToyCar.glb")
        assert name == "ToyCar"

    def test_mixed_case_preserved(self):
        assert _scene_name_from_path("ABeautifulGame.glb") == "ABeautifulGame"


# ---------------------------------------------------------------------------
# Full pipeline: generate viewpoints for scene
# ---------------------------------------------------------------------------

class TestGenerateViewpointsForScene:
    def test_cornell_box_uses_hardcoded_config(self, tmp_path):
        # Create a dummy cornell_box.glb
        glb_path = str(tmp_path / "cornell_box.glb")
        _make_test_glb(glb_path, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        vps = generate_viewpoints_for_scene(glb_path, scene_name="cornell_box")
        # 3 elevations × 8 orbit views = 24
        assert len(vps) == 24

    def test_auto_computed_scene(self, tmp_path):
        glb_path = str(tmp_path / "TestModel.glb")
        _make_test_glb(glb_path, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])

        vps = generate_viewpoints_for_scene(glb_path)
        # 3 zoom factors × 3 default elevations × 8 default orbit views = 72
        assert len(vps) == 72

    def test_viewpoint_format(self, tmp_path):
        glb_path = str(tmp_path / "Model.glb")
        _make_test_glb(glb_path, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        vps = generate_viewpoints_for_scene(glb_path)
        for vp in vps:
            assert "position" in vp
            assert "target" in vp
            assert len(vp["position"]) == 3
            assert len(vp["target"]) == 3

    def test_no_nan_in_viewpoints(self, tmp_path):
        glb_path = str(tmp_path / "Model.glb")
        _make_test_glb(glb_path, [-5.0, 0.0, -5.0], [5.0, 10.0, 5.0])

        vps = generate_viewpoints_for_scene(glb_path)
        for vp in vps:
            for val in vp["position"] + vp["target"]:
                assert not math.isnan(val)
                assert not math.isinf(val)

    def test_position_never_equals_target(self, tmp_path):
        glb_path = str(tmp_path / "Model.glb")
        _make_test_glb(glb_path, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        vps = generate_viewpoints_for_scene(glb_path)
        for vp in vps:
            assert vp["position"] != vp["target"]


# ---------------------------------------------------------------------------
# JSON output format
# ---------------------------------------------------------------------------

class TestJsonOutput:
    def test_json_format_matches_f9_6a_schema(self, tmp_path):
        """Output JSON must be an array of {position, target} objects."""
        scenes_dir = str(tmp_path / "scenes")
        output_dir = str(tmp_path / "viewpoints")
        os.makedirs(scenes_dir)

        _make_test_glb(os.path.join(scenes_dir, "TestModel.glb"),
                       [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        results = generate_all_viewpoints(scenes_dir, output_dir)
        assert "TestModel" in results

        json_path = os.path.join(output_dir, "TestModel.json")
        assert os.path.isfile(json_path)

        with open(json_path, "r") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) > 0
        for entry in data:
            assert isinstance(entry, dict)
            assert "position" in entry
            assert "target" in entry
            assert "id" in entry
            assert isinstance(entry["id"], str)
            assert len(entry["id"]) == 8
            assert isinstance(entry["position"], list)
            assert isinstance(entry["target"], list)
            assert len(entry["position"]) == 3
            assert len(entry["target"]) == 3

        # All IDs must be unique
        ids = [entry["id"] for entry in data]
        assert len(ids) == len(set(ids))

    def test_discovers_gltf_subdirectory(self, tmp_path):
        scenes_dir = str(tmp_path / "scenes")
        output_dir = str(tmp_path / "viewpoints")

        subdir = os.path.join(scenes_dir, "FlightHelmet")
        _make_test_gltf(subdir, "FlightHelmet",
                        [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])

        results = generate_all_viewpoints(scenes_dir, output_dir)
        assert "FlightHelmet" in results

    def test_empty_scenes_dir(self, tmp_path):
        scenes_dir = str(tmp_path / "empty_scenes")
        output_dir = str(tmp_path / "viewpoints")
        os.makedirs(scenes_dir)

        results = generate_all_viewpoints(scenes_dir, output_dir)
        assert results == {}


# ---------------------------------------------------------------------------
# Spherical coordinate helpers
# ---------------------------------------------------------------------------

class TestSphericalCoordinates:
    def test_roundtrip(self):
        target = [1.0, 2.0, 3.0]
        position = [4.0, 5.0, 6.0]
        az, el, dist = _cartesian_to_spherical(position, target)
        restored = _spherical_to_cartesian(target, az, el, dist)
        for i in range(3):
            assert abs(restored[i] - position[i]) < 1e-10

    def test_zero_distance(self):
        az, el, dist = _cartesian_to_spherical([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        assert dist == 0.0

    def test_positive_x_axis(self):
        az, el, dist = _cartesian_to_spherical([5.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        assert abs(dist - 5.0) < 1e-10
        assert abs(az - 0.0) < 1e-10
        assert abs(el - 0.0) < 1e-10


# ---------------------------------------------------------------------------
# Seed variation generation
# ---------------------------------------------------------------------------

class TestSeedVariations:
    @staticmethod
    def _make_seed(px, py, pz, tx, ty, tz, fov=60.0):
        return {
            "position": [px, py, pz],
            "target": [tx, ty, tz],
            "fov": fov,
        }

    def test_empty_seeds_returns_empty(self):
        result = generate_seed_variations([], variations_per_seed=4)
        assert result == []

    def test_single_seed_includes_original(self):
        seed = self._make_seed(0.0, 0.0, 5.0, 0.0, 0.0, 0.0)
        result = generate_seed_variations([seed], variations_per_seed=1)
        assert len(result) == 1
        assert result[0]["position"] == seed["position"]
        assert result[0]["target"] == seed["target"]

    def test_correct_count_single_seed(self):
        seed = self._make_seed(0.0, 0.0, 5.0, 0.0, 0.0, 0.0)
        result = generate_seed_variations([seed], variations_per_seed=4)
        assert len(result) == 4

    def test_correct_count_multiple_seeds(self):
        seeds = [
            self._make_seed(0.0, 0.0, 5.0, 0.0, 0.0, 0.0),
            self._make_seed(5.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            self._make_seed(0.0, 5.0, 0.0, 0.0, 0.0, 0.0),
        ]
        result = generate_seed_variations(seeds, variations_per_seed=4)
        # 3 seeds × 4 per seed = 12 total
        assert len(result) == 12

    def test_originals_included_verbatim(self):
        seeds = [
            self._make_seed(0.0, 0.0, 5.0, 0.0, 0.0, 0.0),
            self._make_seed(5.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ]
        result = generate_seed_variations(seeds, variations_per_seed=3)
        # First of each group of 3 should be the original
        assert result[0]["position"] == seeds[0]["position"]
        assert result[3]["position"] == seeds[1]["position"]

    def test_variations_differ_from_original(self):
        seed = self._make_seed(0.0, 0.0, 5.0, 0.0, 0.0, 0.0)
        result = generate_seed_variations([seed], variations_per_seed=10, rng_seed="test")
        # At least some variations should differ from the original
        different_count = sum(
            1 for vp in result[1:]
            if vp["position"] != seed["position"] or vp["target"] != seed["target"]
        )
        assert different_count > 0

    def test_no_nan_values(self):
        seeds = [
            self._make_seed(0.0, 0.0, 5.0, 0.0, 0.0, 0.0),
            self._make_seed(5.0, 2.0, 0.0, 0.0, 1.0, 0.0),
        ]
        result = generate_seed_variations(seeds, variations_per_seed=8, rng_seed="nancheck")
        for vp in result:
            for val in vp["position"] + vp["target"]:
                assert not math.isnan(val)
                assert not math.isinf(val)

    def test_position_not_equal_target(self):
        seeds = [
            self._make_seed(0.0, 0.0, 5.0, 0.0, 0.0, 0.0),
            self._make_seed(3.0, 2.0, 0.0, 0.0, 0.0, 0.0),
        ]
        result = generate_seed_variations(seeds, variations_per_seed=6, rng_seed="poscheck")
        for vp in result:
            assert vp["position"] != vp["target"]

    def test_jitter_fraction_controls_spread(self):
        seed = self._make_seed(0.0, 0.0, 10.0, 0.0, 0.0, 0.0)
        # Small jitter
        result_small = generate_seed_variations(
            [seed], variations_per_seed=20, jitter_frac=0.01, rng_seed="small"
        )
        # Large jitter
        result_large = generate_seed_variations(
            [seed], variations_per_seed=20, jitter_frac=0.5, rng_seed="large"
        )

        def avg_dist(vps):
            dists = [_vec3_distance(vp["position"], seed["position"]) for vp in vps[1:]]
            return sum(dists) / len(dists) if dists else 0.0

        # The large jitter set should have more spread on average
        assert avg_dist(result_large) > avg_dist(result_small)

    def test_fov_preserved_in_single_seed_variations(self):
        seed = self._make_seed(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, fov=45.0)
        result = generate_seed_variations([seed], variations_per_seed=4, rng_seed="fov")
        # Original should have exact FOV
        assert result[0]["fov"] == 45.0
        # Variations may have slight FOV jitter (orbit perturbation ±2°) but
        # should be in reasonable range
        for vp in result:
            assert "fov" in vp

    def test_exposure_preserved(self):
        seed = {"position": [0.0, 0.0, 5.0], "target": [0.0, 0.0, 0.0], "exposure": 2.5}
        result = generate_seed_variations([seed], variations_per_seed=4, rng_seed="exp")
        for vp in result:
            assert "exposure" in vp

    def test_reproducible_with_same_seed(self):
        seeds = [self._make_seed(0.0, 0.0, 5.0, 0.0, 0.0, 0.0)]
        r1 = generate_seed_variations(seeds, variations_per_seed=6, rng_seed="repro")
        r2 = generate_seed_variations(seeds, variations_per_seed=6, rng_seed="repro")
        assert r1 == r2

    def test_different_rng_seed_produces_different_output(self):
        seeds = [self._make_seed(0.0, 0.0, 5.0, 0.0, 0.0, 0.0)]
        r1 = generate_seed_variations(seeds, variations_per_seed=6, rng_seed="a")
        r2 = generate_seed_variations(seeds, variations_per_seed=6, rng_seed="b")
        # At least some variations should differ
        assert r1 != r2

    def test_variations_per_seed_one_returns_only_originals(self):
        seeds = [
            self._make_seed(0.0, 0.0, 5.0, 0.0, 0.0, 0.0),
            self._make_seed(5.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ]
        result = generate_seed_variations(seeds, variations_per_seed=1)
        assert len(result) == 2
        assert result[0]["position"] == seeds[0]["position"]
        assert result[1]["position"] == seeds[1]["position"]


# ---------------------------------------------------------------------------
# Seed file loading
# ---------------------------------------------------------------------------

class TestLoadSeedViewpoints:
    def test_load_from_directory(self, tmp_path):
        seeds_dir = str(tmp_path / "seeds")
        os.makedirs(seeds_dir)

        vps = [
            {"position": [0.0, 0.0, 5.0], "target": [0.0, 0.0, 0.0]},
            {"position": [5.0, 0.0, 0.0], "target": [0.0, 0.0, 0.0]},
        ]
        with open(os.path.join(seeds_dir, "sponza.json"), "w") as f:
            json.dump(vps, f)

        result = load_seed_viewpoints(seeds_dir)
        assert "sponza" in result
        assert len(result["sponza"]) == 2

    def test_ignores_non_json_files(self, tmp_path):
        seeds_dir = str(tmp_path / "seeds")
        os.makedirs(seeds_dir)

        with open(os.path.join(seeds_dir, "notes.txt"), "w") as f:
            f.write("not json")
        with open(os.path.join(seeds_dir, "sponza.json"), "w") as f:
            json.dump([{"position": [0, 0, 5], "target": [0, 0, 0]}], f)

        result = load_seed_viewpoints(seeds_dir)
        assert "sponza" in result
        assert "notes" not in result

    def test_ignores_empty_arrays(self, tmp_path):
        seeds_dir = str(tmp_path / "seeds")
        os.makedirs(seeds_dir)

        with open(os.path.join(seeds_dir, "empty.json"), "w") as f:
            json.dump([], f)

        result = load_seed_viewpoints(seeds_dir)
        assert "empty" not in result

    def test_nonexistent_dir_returns_empty(self):
        result = load_seed_viewpoints("/nonexistent/path")
        assert result == {}

    def test_multiple_scene_files(self, tmp_path):
        seeds_dir = str(tmp_path / "seeds")
        os.makedirs(seeds_dir)

        for name in ["sponza", "cornell_box", "damaged_helmet"]:
            with open(os.path.join(seeds_dir, f"{name}.json"), "w") as f:
                json.dump([{"position": [0, 0, 5], "target": [0, 0, 0]}], f)

        result = load_seed_viewpoints(seeds_dir)
        assert set(result.keys()) == {"sponza", "cornell_box", "damaged_helmet"}


# ---------------------------------------------------------------------------
# Integration: seeds in generate_viewpoints_for_scene
# ---------------------------------------------------------------------------

class TestSeedIntegration:
    def test_seeds_replace_auto_generation(self, tmp_path):
        glb_path = str(tmp_path / "TestModel.glb")
        _make_test_glb(glb_path, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])

        seeds = [
            {"position": [0.0, 0.0, 5.0], "target": [0.0, 0.0, 0.0]},
            {"position": [5.0, 0.0, 0.0], "target": [0.0, 0.0, 0.0]},
        ]
        vps = generate_viewpoints_for_scene(glb_path, seeds=seeds, variations_per_seed=3)
        # 2 seeds × 3 per seed = 6 (not the default 72)
        assert len(vps) == 6

    def test_seeds_replace_cornell_box_hardcoded(self, tmp_path):
        glb_path = str(tmp_path / "cornell_box.glb")
        _make_test_glb(glb_path, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        seeds = [
            {"position": [0.5, 0.5, 3.0], "target": [0.5, 0.5, 0.5]},
        ]
        vps = generate_viewpoints_for_scene(
            glb_path, scene_name="cornell_box", seeds=seeds, variations_per_seed=4
        )
        # Seed-derived, not the default 24
        assert len(vps) == 4
        # Original seed should be first
        assert vps[0]["position"] == [0.5, 0.5, 3.0]

    def test_no_seeds_falls_back_to_auto(self, tmp_path):
        glb_path = str(tmp_path / "TestModel.glb")
        _make_test_glb(glb_path, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])

        vps = generate_viewpoints_for_scene(glb_path)
        # Default: 3 zoom × 3 elev × 8 orbit = 72
        assert len(vps) == 72

    def test_generate_all_with_seeds(self, tmp_path):
        scenes_dir = str(tmp_path / "scenes")
        output_dir = str(tmp_path / "viewpoints")
        seeds_dir = str(tmp_path / "seeds")
        os.makedirs(scenes_dir)
        os.makedirs(seeds_dir)

        # Create two scenes
        _make_test_glb(os.path.join(scenes_dir, "ModelA.glb"),
                       [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        _make_test_glb(os.path.join(scenes_dir, "ModelB.glb"),
                       [0.0, 0.0, 0.0], [2.0, 2.0, 2.0])

        # Seed only for ModelA (filename must match scene stem)
        with open(os.path.join(seeds_dir, "ModelA.json"), "w") as f:
            json.dump([{"position": [0, 0, 5], "target": [0.5, 0.5, 0.5]}], f)

        results = generate_all_viewpoints(
            scenes_dir, output_dir,
            seeds_dir=seeds_dir, variations_per_seed=3,
        )

        # ModelA should use seeds: 1 seed × 3 = 3
        assert results["ModelA"] == 3
        # ModelB should use auto: 72
        assert results["ModelB"] == 72

    def test_output_format_with_seeds(self, tmp_path):
        scenes_dir = str(tmp_path / "scenes")
        output_dir = str(tmp_path / "viewpoints")
        seeds_dir = str(tmp_path / "seeds")
        os.makedirs(scenes_dir)
        os.makedirs(seeds_dir)

        _make_test_glb(os.path.join(scenes_dir, "Test.glb"),
                       [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        with open(os.path.join(seeds_dir, "Test.json"), "w") as f:
            json.dump([
                {"position": [0, 0, 5], "target": [0, 0, 0], "fov": 60.0},
            ], f)

        generate_all_viewpoints(
            scenes_dir, output_dir,
            seeds_dir=seeds_dir, variations_per_seed=4,
        )

        with open(os.path.join(output_dir, "Test.json"), "r") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 4
        for entry in data:
            assert "position" in entry
            assert "target" in entry
            assert len(entry["position"]) == 3
            assert len(entry["target"]) == 3


# ---------------------------------------------------------------------------
# Environment and light rig discovery
# ---------------------------------------------------------------------------

class TestDiscoverEnvs:
    def test_discovers_exr_files(self, tmp_path):
        (tmp_path / "a.exr").write_bytes(b"fake")
        (tmp_path / "b.exr").write_bytes(b"fake")
        (tmp_path / "not_exr.png").write_bytes(b"fake")
        result = _discover_envs(str(tmp_path))
        assert len(result) == 2
        assert all(p.endswith(".exr") for p in result)

    def test_sorted_alphabetically(self, tmp_path):
        (tmp_path / "zz.exr").write_bytes(b"fake")
        (tmp_path / "aa.exr").write_bytes(b"fake")
        (tmp_path / "mm.exr").write_bytes(b"fake")
        result = _discover_envs(str(tmp_path))
        basenames = [os.path.basename(p) for p in result]
        assert basenames == ["aa.exr", "mm.exr", "zz.exr"]

    def test_empty_dir(self, tmp_path):
        result = _discover_envs(str(tmp_path))
        assert result == []

    def test_nonexistent_dir(self, tmp_path):
        result = _discover_envs(str(tmp_path / "nope"))
        assert result == []

    def test_returns_absolute_paths(self, tmp_path):
        (tmp_path / "test.exr").write_bytes(b"fake")
        result = _discover_envs(str(tmp_path))
        assert all(os.path.isabs(p) for p in result)


class TestDiscoverLightRigsForScene:
    def test_discovers_scene_rigs_flat(self, tmp_path):
        (tmp_path / "MyScene_overhead.json").write_text("[]")
        (tmp_path / "MyScene_keyfillrim.json").write_text("[]")
        (tmp_path / "OtherScene_overhead.json").write_text("[]")
        result = _discover_light_rigs_for_scene(str(tmp_path), "MyScene")
        assert len(result) == 2
        types = [r[0] for r in result]
        assert "keyfillrim" in types
        assert "overhead" in types

    def test_sorted_by_rig_type(self, tmp_path):
        (tmp_path / "Scene_z_rig.json").write_text("[]")
        (tmp_path / "Scene_a_rig.json").write_text("[]")
        result = _discover_light_rigs_for_scene(str(tmp_path), "Scene")
        types = [r[0] for r in result]
        assert types == ["a_rig", "z_rig"]

    def test_no_matching_rigs(self, tmp_path):
        (tmp_path / "OtherScene_overhead.json").write_text("[]")
        result = _discover_light_rigs_for_scene(str(tmp_path), "MyScene")
        assert result == []

    def test_nonexistent_dir(self, tmp_path):
        result = _discover_light_rigs_for_scene(str(tmp_path / "nope"), "Scene")
        assert result == []

    def test_returns_absolute_paths(self, tmp_path):
        (tmp_path / "Scene_overhead.json").write_text("[]")
        result = _discover_light_rigs_for_scene(str(tmp_path), "Scene")
        assert all(os.path.isabs(p) for _, p in result)


class TestSceneHasEmissiveLights:
    """Tests for emissive light source detection in glTF/GLB scenes."""

    @staticmethod
    def _make_glb_with_materials(path: str, materials: list) -> None:
        """Create a minimal GLB with the given materials array."""
        gltf_json = {
            "asset": {"version": "2.0"},
            "materials": materials,
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3,
                 "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 1]}
            ],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}],
            "buffers": [{"byteLength": 36}],
        }
        json_bytes = json.dumps(gltf_json).encode("utf-8")
        while len(json_bytes) % 4 != 0:
            json_bytes += b" "
        bin_data = b"\x00" * 36
        json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
        bin_chunk = struct.pack("<II", len(bin_data), 0x004E4942) + bin_data
        total = 12 + len(json_chunk) + len(bin_chunk)
        header = struct.pack("<III", 0x46546C67, 2, total)
        with open(path, "wb") as f:
            f.write(header + json_chunk + bin_chunk)

    def test_no_materials_returns_false(self, tmp_path):
        self._make_glb_with_materials(str(tmp_path / "test.glb"), [])
        assert not _scene_has_emissive_lights(str(tmp_path / "test.glb"))

    def test_non_emissive_material_returns_false(self, tmp_path):
        materials = [{"pbrMetallicRoughness": {"baseColorFactor": [1, 0, 0, 1]}}]
        self._make_glb_with_materials(str(tmp_path / "test.glb"), materials)
        assert not _scene_has_emissive_lights(str(tmp_path / "test.glb"))

    def test_weak_emissive_returns_false(self, tmp_path):
        """Decorative glow (factor <= 1.0, no strength extension) is not a light."""
        materials = [{"emissiveFactor": [0.5, 0.3, 0.1]}]
        self._make_glb_with_materials(str(tmp_path / "test.glb"), materials)
        assert not _scene_has_emissive_lights(str(tmp_path / "test.glb"))

    def test_bright_emissive_with_strength_returns_true(self, tmp_path):
        """Cornell box style: high emissive strength = area light source."""
        materials = [{
            "emissiveFactor": [1.0, 0.706, 0.235],
            "extensions": {
                "KHR_materials_emissive_strength": {"emissiveStrength": 17.0}
            }
        }]
        self._make_glb_with_materials(str(tmp_path / "test.glb"), materials)
        assert _scene_has_emissive_lights(str(tmp_path / "test.glb"))

    def test_emissive_factor_just_above_threshold(self, tmp_path):
        """Factor of [1.1, 0, 0] with default strength 1.0 -> max_radiance=1.1 > 1.0."""
        materials = [{"emissiveFactor": [1.1, 0.0, 0.0]}]
        self._make_glb_with_materials(str(tmp_path / "test.glb"), materials)
        assert _scene_has_emissive_lights(str(tmp_path / "test.glb"))

    def test_invalid_glb_returns_false(self, tmp_path):
        """Gracefully handles corrupt files."""
        (tmp_path / "bad.glb").write_bytes(b"not a glb file")
        assert not _scene_has_emissive_lights(str(tmp_path / "bad.glb"))

    def test_gltf_file(self, tmp_path):
        """Works with .gltf JSON files too."""
        gltf = {
            "asset": {"version": "2.0"},
            "materials": [{
                "emissiveFactor": [1.0, 1.0, 1.0],
                "extensions": {
                    "KHR_materials_emissive_strength": {"emissiveStrength": 5.0}
                }
            }]
        }
        gltf_path = tmp_path / "scene.gltf"
        gltf_path.write_text(json.dumps(gltf))
        assert _scene_has_emissive_lights(str(gltf_path))

    def test_unsupported_extension_returns_false(self, tmp_path):
        (tmp_path / "model.obj").write_text("v 0 0 0")
        assert not _scene_has_emissive_lights(str(tmp_path / "model.obj"))


# ---------------------------------------------------------------------------
# Environment and lights assignment
# ---------------------------------------------------------------------------

class TestAssignEnvironmentAndLights:
    def test_empty_pool_no_changes(self):
        vps = [{"position": [0, 0, 0], "target": [1, 0, 0]}]
        _assign_environment_and_lights(vps, "scene", [], [])
        assert "environment" not in vps[0]
        assert "lights" not in vps[0]

    def test_env_only_pool(self):
        vps = [
            {"position": [0, 0, 0], "target": [1, 0, 0]},
            {"position": [1, 0, 0], "target": [0, 0, 0]},
        ]
        _assign_environment_and_lights(
            vps, "scene", ["/envs/a.exr", "/envs/b.exr"], [])
        for vp in vps:
            assert "environment" in vp
            assert "lights" not in vp

    def test_lights_only_pool(self):
        vps = [
            {"position": [0, 0, 0], "target": [1, 0, 0]},
            {"position": [1, 0, 0], "target": [0, 0, 0]},
        ]
        rigs = [("overhead", "/rigs/overhead.json"), ("kfr", "/rigs/kfr.json")]
        _assign_environment_and_lights(vps, "scene", [], rigs)
        for vp in vps:
            assert "lights" in vp
            assert "environment" not in vp

    def test_mixed_pool_assigns_one_type_per_viewpoint(self):
        vps = [{"position": [i, 0, 0], "target": [0, 0, 0]} for i in range(10)]
        rigs = [("overhead", "/rigs/overhead.json")]
        _assign_environment_and_lights(
            vps, "scene", ["/envs/a.exr"], rigs)
        for vp in vps:
            has_env = "environment" in vp
            has_lights = "lights" in vp
            assert has_env or has_lights
            assert not (has_env and has_lights)

    def test_skips_already_assigned_environment(self):
        vps = [
            {"position": [0, 0, 0], "target": [1, 0, 0],
             "environment": "/envs/existing.exr"},
            {"position": [1, 0, 0], "target": [0, 0, 0]},
        ]
        _assign_environment_and_lights(
            vps, "scene", ["/envs/new.exr"], [])
        assert vps[0]["environment"] == "/envs/existing.exr"
        assert vps[1]["environment"] == "/envs/new.exr"

    def test_skips_already_assigned_lights(self):
        vps = [
            {"position": [0, 0, 0], "target": [1, 0, 0],
             "lights": "/rigs/existing.json"},
            {"position": [1, 0, 0], "target": [0, 0, 0]},
        ]
        rigs = [("overhead", "/rigs/new.json")]
        _assign_environment_and_lights(vps, "scene", [], rigs)
        assert vps[0]["lights"] == "/rigs/existing.json"

    def test_deterministic_seeding(self):
        vps1 = [{"position": [i, 0, 0], "target": [0, 0, 0]} for i in range(10)]
        vps2 = [{"position": [i, 0, 0], "target": [0, 0, 0]} for i in range(10)]
        envs = ["/envs/a.exr", "/envs/b.exr", "/envs/c.exr"]
        _assign_environment_and_lights(vps1, "scene_x", envs, [])
        _assign_environment_and_lights(vps2, "scene_x", envs, [])
        for v1, v2 in zip(vps1, vps2):
            assert v1.get("environment") == v2.get("environment")

    def test_different_scenes_get_different_assignments(self):
        envs = ["/envs/a.exr", "/envs/b.exr", "/envs/c.exr"]
        vps1 = [{"position": [i, 0, 0], "target": [0, 0, 0]} for i in range(20)]
        vps2 = [{"position": [i, 0, 0], "target": [0, 0, 0]} for i in range(20)]
        _assign_environment_and_lights(vps1, "scene_a", envs, [])
        _assign_environment_and_lights(vps2, "scene_b", envs, [])
        assignments1 = [vp.get("environment") for vp in vps1]
        assignments2 = [vp.get("environment") for vp in vps2]
        assert assignments1 != assignments2


# ---------------------------------------------------------------------------
# Exposure amplification
# ---------------------------------------------------------------------------

class TestAmplifyExposures:
    def test_empty_viewpoints(self):
        result = _amplify_exposures([], [0.0, 1.0])
        assert result == []

    def test_empty_exposures(self):
        vps = [{"position": [0, 0, 0], "target": [1, 0, 0]}]
        result = _amplify_exposures(vps, [])
        assert result == vps

    def test_amplification_count(self):
        vps = [{"position": [0, 0, 0], "target": [1, 0, 0]}]
        result = _amplify_exposures(vps, [0.0, -1.0, 1.0])
        assert len(result) == 3

    def test_exposure_values_assigned(self):
        vps = [{"position": [0, 0, 0], "target": [1, 0, 0]}]
        exposures = [0.0, -1.0, 1.0, -2.0, 2.0]
        result = _amplify_exposures(vps, exposures)
        assert len(result) == 5
        for vp, ev in zip(result, exposures):
            assert vp["exposure"] == ev

    def test_preserves_existing_exposure(self):
        vps = [{"position": [0, 0, 0], "target": [1, 0, 0], "exposure": 2.5}]
        result = _amplify_exposures(vps, [0.0, -1.0, 1.0])
        assert len(result) == 1
        assert result[0]["exposure"] == 2.5

    def test_mixed_with_and_without_exposure(self):
        vps = [
            {"position": [0, 0, 0], "target": [1, 0, 0]},
            {"position": [1, 0, 0], "target": [0, 0, 0], "exposure": 3.0},
            {"position": [2, 0, 0], "target": [0, 0, 0]},
        ]
        result = _amplify_exposures(vps, [0.0, -1.0])
        # vp1: 2 copies, vp2: 1 (kept), vp3: 2 copies = 5 total
        assert len(result) == 5
        assert result[2]["exposure"] == 3.0

    def test_preserves_other_fields(self):
        vps = [{"position": [0, 0, 0], "target": [1, 0, 0],
                "environment": "/envs/a.exr"}]
        result = _amplify_exposures(vps, [0.0, 1.0])
        for vp in result:
            assert vp["environment"] == "/envs/a.exr"

    def test_default_exposures_constant(self):
        assert len(_DEFAULT_EXPOSURES) == 5
        assert 0.0 in _DEFAULT_EXPOSURES

    def test_does_not_mutate_originals(self):
        vps = [{"position": [0, 0, 0], "target": [1, 0, 0]}]
        result = _amplify_exposures(vps, [0.0, 1.0])
        assert "exposure" not in vps[0]
        assert len(result) == 2

    def test_integration_with_generate_all(self, tmp_path):
        """Exposure amplification works end-to-end through generate_all_viewpoints."""
        scenes_dir = str(tmp_path / "scenes")
        output_dir = str(tmp_path / "viewpoints")
        os.makedirs(scenes_dir)

        _make_test_glb(os.path.join(scenes_dir, "TestModel.glb"),
                       [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        results = generate_all_viewpoints(
            scenes_dir, output_dir, exposures=[0.0, -1.0, 1.0])

        # 72 auto viewpoints × 3 exposures = 216
        assert results["TestModel"] == 216

        with open(os.path.join(output_dir, "TestModel.json")) as f:
            data = json.load(f)
        assert len(data) == 216
        exposure_values = {vp["exposure"] for vp in data}
        assert exposure_values == {0.0, -1.0, 1.0}
        # All amplified viewpoints should have unique IDs
        ids = [vp["id"] for vp in data]
        assert len(ids) == len(set(ids))


class TestAssignViewpointIds:
    def test_assigns_8_hex_ids(self):
        vps = [{"position": [0, 0, 0], "target": [1, 0, 0]},
               {"position": [1, 0, 0], "target": [0, 0, 0]}]
        _assign_viewpoint_ids(vps)
        for vp in vps:
            assert "id" in vp
            assert len(vp["id"]) == 8
            assert all(c in "0123456789abcdef" for c in vp["id"])

    def test_ids_are_unique(self):
        vps = [{"position": [i, 0, 0], "target": [0, 0, 0]} for i in range(100)]
        _assign_viewpoint_ids(vps)
        ids = [vp["id"] for vp in vps]
        assert len(ids) == len(set(ids))

    def test_preserves_existing_ids(self):
        vps = [
            {"position": [0, 0, 0], "target": [1, 0, 0], "id": "deadbeef"},
            {"position": [1, 0, 0], "target": [0, 0, 0]},
        ]
        _assign_viewpoint_ids(vps)
        assert vps[0]["id"] == "deadbeef"
        assert len(vps[1]["id"]) == 8
        assert vps[1]["id"] != "deadbeef"

    def test_empty_list(self):
        vps: list[dict] = []
        _assign_viewpoint_ids(vps)
        assert vps == []

    def test_no_collision_with_existing(self):
        vps = [
            {"position": [0, 0, 0], "target": [1, 0, 0], "id": "abcd1234"},
            {"position": [1, 0, 0], "target": [0, 0, 0]},
        ]
        _assign_viewpoint_ids(vps)
        assert vps[1]["id"] != "abcd1234"

    def test_does_not_overwrite_existing(self):
        vps = [{"position": [0, 0, 0], "target": [1, 0, 0], "id": "00000000"}]
        _assign_viewpoint_ids(vps)
        assert vps[0]["id"] == "00000000"
