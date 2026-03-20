"""Tests for generate_light_rigs.py."""

import json
import math
import os
import random
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from generate_light_rigs import (
    generate_overhead_rig,
    generate_key_fill_rim_rig,
    generate_light_rigs,
    _make_quad_facing,
    _direction_toward,
)


def _cross(a, b):
    """Cross product of two 3-element lists."""
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _normalize(v):
    length = math.sqrt(sum(x * x for x in v))
    return [x / length for x in v] if length > 1e-8 else v


def _light_normal(light: dict) -> list[float]:
    """Compute the front-face normal from cross(edge_a, edge_b)."""
    return _normalize(_cross(light["edge_a"], light["edge_b"]))


# Sample bounding boxes for testing
_UNIT_BBOX = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
_TALL_BBOX = ([0.0, 0.0, 0.0], [1.0, 5.0, 1.0])
_WIDE_BBOX = ([-5.0, 0.0, -5.0], [5.0, 1.0, 5.0])


class TestMakeQuadFacing:
    def test_downward_facing_quad(self):
        corner, edge_a, edge_b = _make_quad_facing(
            [0.0, 5.0, 0.0], [0.0, -1.0, 0.0], 2.0, 2.0)
        normal = _normalize(_cross(edge_a, edge_b))
        # Normal should point roughly downward
        assert normal[1] < -0.9

    def test_quad_centered_at_position(self):
        center = [3.0, 7.0, -2.0]
        corner, edge_a, edge_b = _make_quad_facing(
            center, [0.0, -1.0, 0.0], 1.0, 1.0)
        # Center of quad = corner + 0.5*edge_a + 0.5*edge_b
        computed_center = [
            corner[i] + 0.5 * edge_a[i] + 0.5 * edge_b[i]
            for i in range(3)
        ]
        for i in range(3):
            assert abs(computed_center[i] - center[i]) < 1e-6

    def test_edge_lengths(self):
        corner, edge_a, edge_b = _make_quad_facing(
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 3.0, 5.0)
        len_a = math.sqrt(sum(x * x for x in edge_a))
        len_b = math.sqrt(sum(x * x for x in edge_b))
        assert abs(len_a - 3.0) < 1e-6
        assert abs(len_b - 5.0) < 1e-6


class TestOverheadRig:
    def test_returns_single_light(self):
        lights = generate_overhead_rig(*_UNIT_BBOX, random.Random(42))
        assert len(lights) == 1

    def test_light_above_scene(self):
        lights = generate_overhead_rig(*_UNIT_BBOX, random.Random(42))
        light = lights[0]
        # Light center should be above bbox_max Y
        center_y = light["corner"][1] + 0.5 * light["edge_a"][1] + 0.5 * light["edge_b"][1]
        assert center_y > 1.0  # Above bbox_max_y

    def test_normal_faces_downward(self):
        lights = generate_overhead_rig(*_UNIT_BBOX, random.Random(42))
        normal = _light_normal(lights[0])
        assert normal[1] < -0.9

    def test_positive_radiance(self):
        lights = generate_overhead_rig(*_UNIT_BBOX, random.Random(42))
        for comp in lights[0]["radiance"]:
            assert comp > 0.0

    def test_not_two_sided(self):
        lights = generate_overhead_rig(*_UNIT_BBOX, random.Random(42))
        assert lights[0]["two_sided"] is False

    def test_schema_matches_lights_json(self):
        lights = generate_overhead_rig(*_UNIT_BBOX, random.Random(42))
        light = lights[0]
        assert "corner" in light and len(light["corner"]) == 3
        assert "edge_a" in light and len(light["edge_a"]) == 3
        assert "edge_b" in light and len(light["edge_b"]) == 3
        assert "radiance" in light and len(light["radiance"]) == 3
        assert "two_sided" in light

    def test_reproducible_with_same_seed(self):
        lights1 = generate_overhead_rig(*_UNIT_BBOX, random.Random(42))
        lights2 = generate_overhead_rig(*_UNIT_BBOX, random.Random(42))
        assert lights1 == lights2


class TestKeyFillRimRig:
    def test_returns_three_lights(self):
        lights = generate_key_fill_rim_rig(*_UNIT_BBOX, random.Random(42))
        assert len(lights) == 3

    def test_all_normals_face_scene_center(self):
        bbox_min, bbox_max = _UNIT_BBOX
        center = [(bbox_min[i] + bbox_max[i]) / 2.0 for i in range(3)]
        lights = generate_key_fill_rim_rig(bbox_min, bbox_max, random.Random(42))

        for light in lights:
            normal = _light_normal(light)
            # Compute light center
            light_center = [
                light["corner"][i] + 0.5 * light["edge_a"][i] + 0.5 * light["edge_b"][i]
                for i in range(3)
            ]
            # Direction from light to scene center
            to_center = _normalize(_direction_toward(light_center, center))
            # Normal should roughly point toward scene center
            dot = _dot(normal, to_center)
            assert dot > 0.5, f"Light normal {normal} doesn't face center {center} (dot={dot})"

    def test_positive_radiance(self):
        lights = generate_key_fill_rim_rig(*_UNIT_BBOX, random.Random(42))
        for light in lights:
            for comp in light["radiance"]:
                assert comp > 0.0

    def test_not_two_sided(self):
        lights = generate_key_fill_rim_rig(*_UNIT_BBOX, random.Random(42))
        for light in lights:
            assert light["two_sided"] is False

    def test_schema_matches_lights_json(self):
        lights = generate_key_fill_rim_rig(*_UNIT_BBOX, random.Random(42))
        for light in lights:
            assert "corner" in light and len(light["corner"]) == 3
            assert "edge_a" in light and len(light["edge_a"]) == 3
            assert "edge_b" in light and len(light["edge_b"]) == 3
            assert "radiance" in light and len(light["radiance"]) == 3
            assert "two_sided" in light

    def test_key_is_brightest(self):
        lights = generate_key_fill_rim_rig(*_UNIT_BBOX, random.Random(42))
        key_brightness = sum(lights[0]["radiance"])
        fill_brightness = sum(lights[1]["radiance"])
        rim_brightness = sum(lights[2]["radiance"])
        assert key_brightness > fill_brightness
        assert key_brightness > rim_brightness


class TestGenerateLightRigs:
    def _make_test_glb(self, path, bbox_min, bbox_max):
        """Create a minimal GLB with POSITION accessor min/max."""
        gltf = {
            "asset": {"version": "2.0"},
            "meshes": [{
                "primitives": [{
                    "attributes": {"POSITION": 0}
                }]
            }],
            "accessors": [{
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": list(bbox_min),
                "max": list(bbox_max),
            }],
        }
        json_bytes = json.dumps(gltf).encode("utf-8")
        # Pad to 4-byte boundary
        while len(json_bytes) % 4 != 0:
            json_bytes += b" "

        with open(path, "wb") as f:
            # GLB header
            total = 12 + 8 + len(json_bytes)
            f.write(b"glTF")
            f.write(struct.pack("<II", 2, total))
            # JSON chunk
            f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
            f.write(json_bytes)

    def test_generates_two_rigs_per_scene(self, tmp_path):
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        self._make_test_glb(
            str(scenes_dir / "TestModel.glb"),
            [0, 0, 0], [1, 1, 1])

        output_dir = tmp_path / "light_rigs"
        results = generate_light_rigs(str(scenes_dir), str(output_dir))

        assert len(results) == 1
        scene_name = list(results.keys())[0]
        assert len(results[scene_name]) == 2

        # Verify files exist and contain valid JSON
        for rig_path in results[scene_name]:
            assert os.path.isfile(rig_path)
            with open(rig_path) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) > 0

    def test_overhead_and_kfr_files(self, tmp_path):
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        self._make_test_glb(
            str(scenes_dir / "TestModel.glb"),
            [-2, 0, -2], [2, 3, 2])

        output_dir = tmp_path / "light_rigs"
        results = generate_light_rigs(str(scenes_dir), str(output_dir))

        scene_name = list(results.keys())[0]
        rig_names = [os.path.basename(p) for p in results[scene_name]]
        assert "TestModel_overhead.json" in rig_names
        assert "TestModel_keyfillrim.json" in rig_names

    def test_reproducible_with_same_seed(self, tmp_path):
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        self._make_test_glb(
            str(scenes_dir / "TestModel.glb"),
            [0, 0, 0], [1, 1, 1])

        out1 = tmp_path / "rigs1"
        out2 = tmp_path / "rigs2"
        results1 = generate_light_rigs(str(scenes_dir), str(out1), seed=42)
        results2 = generate_light_rigs(str(scenes_dir), str(out2), seed=42)

        for name in results1:
            for p1, p2 in zip(results1[name], results2[name]):
                with open(p1) as f1, open(p2) as f2:
                    assert json.load(f1) == json.load(f2)

    def test_no_scenes_returns_empty(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        results = generate_light_rigs(str(empty_dir), str(tmp_path / "out"))
        assert results == {}
