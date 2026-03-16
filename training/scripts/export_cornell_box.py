"""Export a Cornell Box scene to .glb matching tests/scenes/CornellBox.cpp.

Generates a glTF 2.0 binary file with:
  - 7 meshes: floor, ceiling, back_wall, left_wall, right_wall, short_box, tall_box
  - 4 materials: white diffuse, red diffuse, green diffuse, light emissive
  - Area light encoded as emissive ceiling light mesh
  - Camera at canonical viewpoint

Usage:
    python scripts/export_cornell_box.py [--output scenes/cornell_box.glb]
"""

import argparse
import math
import os
import struct
import sys

import numpy as np

try:
    import pygltflib
except ImportError:
    print("Error: pygltflib required. Install with: pip install pygltflib")
    sys.exit(1)


def _make_quad_vertices(v0, v1, v2, v3, normal):
    """Create 4 vertices for a quad (matching MakeQuad in CornellBox.cpp).

    Returns (positions, normals) as lists of 3-tuples.
    """
    positions = [v0, v1, v2, v3]
    normals = [normal] * 4
    return positions, normals


def _make_box_vertices(bmin, bmax):
    """Create 24 vertices for a box (matching MakeBox in CornellBox.cpp).

    6 faces × 4 vertices each. Returns (positions, normals).
    """
    positions = []
    normals = []

    faces = [
        # Front face (+Z)
        ((bmin[0], bmin[1], bmax[2]), (bmax[0], bmin[1], bmax[2]),
         (bmax[0], bmax[1], bmax[2]), (bmin[0], bmax[1], bmax[2]),
         (0.0, 0.0, 1.0)),
        # Back face (-Z)
        ((bmax[0], bmin[1], bmin[2]), (bmin[0], bmin[1], bmin[2]),
         (bmin[0], bmax[1], bmin[2]), (bmax[0], bmax[1], bmin[2]),
         (0.0, 0.0, -1.0)),
        # Right face (+X)
        ((bmax[0], bmin[1], bmax[2]), (bmax[0], bmin[1], bmin[2]),
         (bmax[0], bmax[1], bmin[2]), (bmax[0], bmax[1], bmax[2]),
         (1.0, 0.0, 0.0)),
        # Left face (-X)
        ((bmin[0], bmin[1], bmin[2]), (bmin[0], bmin[1], bmax[2]),
         (bmin[0], bmax[1], bmax[2]), (bmin[0], bmax[1], bmin[2]),
         (-1.0, 0.0, 0.0)),
        # Top face (+Y)
        ((bmin[0], bmax[1], bmax[2]), (bmax[0], bmax[1], bmax[2]),
         (bmax[0], bmax[1], bmin[2]), (bmin[0], bmax[1], bmin[2]),
         (0.0, 1.0, 0.0)),
        # Bottom face (-Y)
        ((bmin[0], bmin[1], bmin[2]), (bmax[0], bmin[1], bmin[2]),
         (bmax[0], bmin[1], bmax[2]), (bmin[0], bmin[1], bmax[2]),
         (0.0, -1.0, 0.0)),
    ]

    for v0, v1, v2, v3, n in faces:
        positions.extend([v0, v1, v2, v3])
        normals.extend([n] * 4)

    return positions, normals


def _quad_indices(base=0):
    """Return 6 indices for a quad starting at vertex base."""
    return [base, base + 1, base + 2, base, base + 2, base + 3]


def _box_indices():
    """Return 36 indices for a 6-face box."""
    indices = []
    for face in range(6):
        indices.extend(_quad_indices(face * 4))
    return indices


def _pack_f32(values):
    """Pack a flat list of floats as little-endian float32 bytes."""
    return struct.pack(f"<{len(values)}f", *values)


def _pack_u16(values):
    """Pack a flat list of ints as little-endian uint16 bytes."""
    return struct.pack(f"<{len(values)}H", *values)


def _flatten_vec3_list(vecs):
    """Flatten list of 3-tuples to a flat list of floats."""
    result = []
    for v in vecs:
        result.extend(v)
    return result


def _compute_min_max(positions):
    """Compute bounding box min/max from a list of 3-tuples."""
    arr = np.array(positions, dtype=np.float32)
    return arr.min(axis=0).tolist(), arr.max(axis=0).tolist()


def build_cornell_box_glb(output_path: str) -> None:
    """Build and write a Cornell Box .glb file."""

    # ── Mesh definitions (matching CornellBox.cpp) ──

    meshes_def = []

    # Floor (white, material 0) — Y=0 plane
    pos, nrm = _make_quad_vertices(
        (0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1), (0, 1, 0))
    meshes_def.append(("floor", pos, nrm, _quad_indices(), 0))

    # Ceiling (white, material 0) — Y=1 plane
    pos, nrm = _make_quad_vertices(
        (0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0), (0, -1, 0))
    meshes_def.append(("ceiling", pos, nrm, _quad_indices(), 0))

    # Back wall (white, material 0) — Z=0 plane
    pos, nrm = _make_quad_vertices(
        (0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 1))
    meshes_def.append(("back_wall", pos, nrm, _quad_indices(), 0))

    # Left wall (red, material 1) — X=0 plane
    pos, nrm = _make_quad_vertices(
        (0, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 0), (1, 0, 0))
    meshes_def.append(("left_wall", pos, nrm, _quad_indices(), 1))

    # Right wall (green, material 2) — X=1 plane
    pos, nrm = _make_quad_vertices(
        (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1), (-1, 0, 0))
    meshes_def.append(("right_wall", pos, nrm, _quad_indices(), 2))

    # Short box (white, material 0)
    pos, nrm = _make_box_vertices((0.13, 0.0, 0.065), (0.43, 0.33, 0.38))
    meshes_def.append(("short_box", pos, nrm, _box_indices(), 0))

    # Tall box (white, material 0)
    pos, nrm = _make_box_vertices((0.53, 0.0, 0.37), (0.83, 0.66, 0.67))
    meshes_def.append(("tall_box", pos, nrm, _box_indices(), 0))

    # ── Ceiling light mesh (emissive, material 3) ──
    # Area light: corner=(0.35, 0.999, 0.35), edge_a=(0.3,0,0), edge_b=(0,0,0.3)
    light_y = 0.999
    pos, nrm = _make_quad_vertices(
        (0.35, light_y, 0.35),
        (0.65, light_y, 0.35),
        (0.65, light_y, 0.65),
        (0.35, light_y, 0.65),
        (0, -1, 0))
    meshes_def.append(("ceiling_light", pos, nrm, _quad_indices(), 3))

    # ── Build binary buffer ──
    buffer_data = bytearray()
    accessors = []
    buffer_views = []

    for _name, positions, normals, indices, _mat_idx in meshes_def:
        # Index buffer
        idx_bytes = _pack_u16(indices)
        idx_offset = len(buffer_data)
        buffer_data.extend(idx_bytes)

        buffer_views.append(pygltflib.BufferView(
            buffer=0,
            byteOffset=idx_offset,
            byteLength=len(idx_bytes),
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        ))
        accessors.append(pygltflib.Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=pygltflib.UNSIGNED_SHORT,
            count=len(indices),
            type=pygltflib.SCALAR,
            max=[max(indices)],
            min=[min(indices)],
        ))
        idx_accessor = len(accessors) - 1

        # Position buffer
        flat_pos = _flatten_vec3_list(positions)
        pos_bytes = _pack_f32(flat_pos)
        pos_offset = len(buffer_data)
        buffer_data.extend(pos_bytes)

        pos_min, pos_max = _compute_min_max(positions)
        buffer_views.append(pygltflib.BufferView(
            buffer=0,
            byteOffset=pos_offset,
            byteLength=len(pos_bytes),
            target=pygltflib.ARRAY_BUFFER,
        ))
        accessors.append(pygltflib.Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=pygltflib.FLOAT,
            count=len(positions),
            type=pygltflib.VEC3,
            max=pos_max,
            min=pos_min,
        ))
        pos_accessor = len(accessors) - 1

        # Normal buffer
        flat_nrm = _flatten_vec3_list(normals)
        nrm_bytes = _pack_f32(flat_nrm)
        nrm_offset = len(buffer_data)
        buffer_data.extend(nrm_bytes)

        buffer_views.append(pygltflib.BufferView(
            buffer=0,
            byteOffset=nrm_offset,
            byteLength=len(nrm_bytes),
            target=pygltflib.ARRAY_BUFFER,
        ))
        accessors.append(pygltflib.Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=pygltflib.FLOAT,
            count=len(normals),
            type=pygltflib.VEC3,
        ))
        nrm_accessor = len(accessors) - 1

        # Store accessor indices on the mesh def for later use
        # (Overwriting the tuple isn't clean, so we just track by order)

    # ── Materials ──
    # White diffuse: base_color = (0.73, 0.73, 0.73)
    white_mat = pygltflib.Material(
        name="white",
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[0.73, 0.73, 0.73, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
    )
    # Red diffuse: base_color = (0.65, 0.05, 0.05)
    red_mat = pygltflib.Material(
        name="red",
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[0.65, 0.05, 0.05, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
    )
    # Green diffuse: base_color = (0.12, 0.45, 0.15)
    green_mat = pygltflib.Material(
        name="green",
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[0.12, 0.45, 0.15, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
    )
    # Light emissive: base_color = (1,1,1), emissive = (17, 12, 4)
    # glTF KHR_materials_emissive_strength extension for emissive > 1.0
    light_mat = pygltflib.Material(
        name="light",
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
        emissiveFactor=[1.0, 12.0 / 17.0, 4.0 / 17.0],
        extensions={
            "KHR_materials_emissive_strength": {
                "emissiveStrength": 17.0,
            }
        },
    )

    materials = [white_mat, red_mat, green_mat, light_mat]

    # ── Meshes and Nodes ──
    gltf_meshes = []
    gltf_nodes = []

    for i, (name, positions, normals, indices, mat_idx) in enumerate(meshes_def):
        # Each mesh has 3 accessors: indices, positions, normals
        base_accessor = i * 3
        idx_acc = base_accessor
        pos_acc = base_accessor + 1
        nrm_acc = base_accessor + 2

        gltf_meshes.append(pygltflib.Mesh(
            name=name,
            primitives=[pygltflib.Primitive(
                attributes=pygltflib.Attributes(
                    POSITION=pos_acc,
                    NORMAL=nrm_acc,
                ),
                indices=idx_acc,
                material=mat_idx,
            )],
        ))
        gltf_nodes.append(pygltflib.Node(name=name, mesh=i))

    # ── Camera ──
    # CornellBox.cpp: position=(0.5, 0.5, 1.94), target=(0.5, 0.5, 0),
    # vfov=39.3077°, near=0.01, far=10.0
    # glTF cameras don't store position — that's on the node.
    # glTF camera looks down -Z in local space, so we need a rotation to
    # point from (0.5,0.5,1.94) toward (0.5,0.5,0), which is already -Z.
    vfov_rad = math.radians(39.3077)

    gltf_camera = pygltflib.Camera(
        name="default",
        type="perspective",
        perspective=pygltflib.Perspective(
            aspectRatio=960.0 / 540.0,
            yfov=vfov_rad,
            znear=0.01,
            zfar=10.0,
        ),
    )
    camera_node = pygltflib.Node(
        name="camera",
        camera=0,
        translation=[0.5, 0.5, 1.94],
    )
    gltf_nodes.append(camera_node)

    # ── Scene ──
    node_indices = list(range(len(gltf_nodes)))
    scene = pygltflib.Scene(name="CornellBox", nodes=node_indices)

    # ── Assemble GLTF ──
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[scene],
        nodes=gltf_nodes,
        meshes=gltf_meshes,
        materials=materials,
        cameras=[gltf_camera],
        accessors=accessors,
        bufferViews=buffer_views,
        buffers=[pygltflib.Buffer(byteLength=len(buffer_data))],
        extensionsUsed=["KHR_materials_emissive_strength"],
    )
    gltf.set_binary_blob(bytes(buffer_data))

    # Write .glb
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    gltf.save(output_path)
    print(f"Exported Cornell Box: {output_path} ({len(buffer_data)} bytes geometry)")
    print(f"  8 meshes, 4 materials, 1 camera")
    print(f"  Emissive light: (17, 12, 4) via KHR_materials_emissive_strength")


def main():
    parser = argparse.ArgumentParser(
        description="Export Cornell Box scene to .glb")
    parser.add_argument("--output", default="scenes/cornell_box.glb",
                        help="Output .glb file path (default: scenes/cornell_box.glb)")
    args = parser.parse_args()
    build_cornell_box_glb(args.output)


if __name__ == "__main__":
    main()
