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
    _center_and_radius_from_aabb,
    _scene_name_from_path,
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
        assert abs(radius - 2.5 * half_diag) < 1e-10

    def test_asymmetric_box(self):
        center, radius = _center_and_radius_from_aabb(
            [-2.0, 0.0, -1.0], [2.0, 4.0, 1.0]
        )
        assert center == [0.0, 2.0, 0.0]
        extents = [4.0, 4.0, 2.0]
        half_diag = math.sqrt(sum(e * e for e in extents)) / 2.0
        assert abs(radius - 2.5 * half_diag) < 1e-10

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
        assert _scene_name_from_path("DamagedHelmet.glb") == "damaged_helmet"

    def test_camel_case(self):
        assert _scene_name_from_path("boomBox.glb") == "boom_box"

    def test_already_snake_case(self):
        assert _scene_name_from_path("cornell_box.glb") == "cornell_box"

    def test_gltf_path(self):
        assert _scene_name_from_path("FlightHelmet/FlightHelmet.gltf") == "flight_helmet"

    def test_consecutive_uppercase(self):
        # "GLB" should not insert underscore between consecutive upper
        name = _scene_name_from_path("ToyCar.glb")
        assert name == "toy_car"

    def test_leading_consecutive_uppercase(self):
        assert _scene_name_from_path("ABeautifulGame.glb") == "a_beautiful_game"


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
        # 3 default elevations × 8 default orbit views = 24
        assert len(vps) == 24

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
        assert "test_model" in results

        json_path = os.path.join(output_dir, "test_model.json")
        assert os.path.isfile(json_path)

        with open(json_path, "r") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) > 0
        for entry in data:
            assert isinstance(entry, dict)
            assert "position" in entry
            assert "target" in entry
            assert isinstance(entry["position"], list)
            assert isinstance(entry["target"], list)
            assert len(entry["position"]) == 3
            assert len(entry["target"]) == 3

    def test_discovers_gltf_subdirectory(self, tmp_path):
        scenes_dir = str(tmp_path / "scenes")
        output_dir = str(tmp_path / "viewpoints")

        subdir = os.path.join(scenes_dir, "FlightHelmet")
        _make_test_gltf(subdir, "FlightHelmet",
                        [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])

        results = generate_all_viewpoints(scenes_dir, output_dir)
        assert "flight_helmet" in results

    def test_empty_scenes_dir(self, tmp_path):
        scenes_dir = str(tmp_path / "empty_scenes")
        output_dir = str(tmp_path / "viewpoints")
        os.makedirs(scenes_dir)

        results = generate_all_viewpoints(scenes_dir, output_dir)
        assert results == {}
