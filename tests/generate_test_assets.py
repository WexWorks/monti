#!/usr/bin/env python3
"""Generate minimal glTF binary (.glb) test assets for monti unit tests.

Uses only Python standard library (struct, json). No third-party dependencies.

Generated files:
  - NoNormals.glb: quad + degenerate mesh without normals accessor
  - DiffuseTransmission.glb: triangle with KHR_materials_diffuse_transmission
  - NoMaterial.glb: triangle with no material reference
"""

import json
import os
import struct

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "..", "scenes", "debug")


def write_glb(path, json_obj, bin_data):
    """Write a GLB 2.0 file from a JSON dict and binary buffer bytes."""
    json_str = json.dumps(json_obj, separators=(",", ":"))
    # Pad JSON to 4-byte boundary with spaces
    json_pad = (4 - len(json_str) % 4) % 4
    json_bytes = json_str.encode("ascii") + b" " * json_pad
    # Pad binary to 4-byte boundary with null bytes
    bin_pad = (4 - len(bin_data) % 4) % 4
    bin_bytes = bin_data + b"\x00" * bin_pad

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)

    with open(path, "wb") as f:
        # GLB header: magic, version, total length
        f.write(struct.pack("<III", 0x46546C67, 2, total_length))
        # JSON chunk
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        # BIN chunk
        f.write(struct.pack("<II", len(bin_bytes), 0x004E4942))
        f.write(bin_bytes)

    print(f"  {os.path.basename(path)} ({total_length} bytes)")


def generate_no_normals():
    """NoNormals.glb: unit quad + degenerate triangle, no normals accessor.

    Binary layout (136 bytes):
      [0..12)    quad indices      6 x uint16
      [12..18)   degen indices     3 x uint16
      [18..20)   padding
      [20..68)   quad positions    4 x vec3
      [68..100)  quad texcoords    4 x vec2
      [100..136) degen positions   3 x vec3
    """
    buf = bytearray()

    # Quad indices: two triangles (CCW winding viewed from -Y)
    buf += struct.pack("<6H", 0, 1, 2, 0, 2, 3)
    # Degenerate indices
    buf += struct.pack("<3H", 0, 1, 2)
    # Padding to 4-byte alignment for floats
    buf += b"\x00" * 2

    # Quad positions (XZ plane, Y=0)
    buf += struct.pack("<3f", -0.5, 0.0, -0.5)
    buf += struct.pack("<3f",  0.5, 0.0, -0.5)
    buf += struct.pack("<3f",  0.5, 0.0,  0.5)
    buf += struct.pack("<3f", -0.5, 0.0,  0.5)

    # Quad texcoords
    buf += struct.pack("<2f", 0.0, 0.0)
    buf += struct.pack("<2f", 1.0, 0.0)
    buf += struct.pack("<2f", 1.0, 1.0)
    buf += struct.pack("<2f", 0.0, 1.0)

    # Degenerate positions (all at origin)
    buf += struct.pack("<3f", 0.0, 0.0, 0.0)
    buf += struct.pack("<3f", 0.0, 0.0, 0.0)
    buf += struct.pack("<3f", 0.0, 0.0, 0.0)

    assert len(buf) == 136

    gltf = {
        "asset": {"version": "2.0", "generator": "monti-test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "NoNormals"}],
        "meshes": [{
            "primitives": [
                {"attributes": {"POSITION": 2, "TEXCOORD_0": 3}, "indices": 0},
                {"attributes": {"POSITION": 4}, "indices": 1}
            ]
        }],
        "accessors": [
            {"bufferView": 0, "componentType": 5123, "count": 6,
             "type": "SCALAR", "max": [3], "min": [0]},
            {"bufferView": 1, "componentType": 5123, "count": 3,
             "type": "SCALAR", "max": [2], "min": [0]},
            {"bufferView": 2, "componentType": 5126, "count": 4,
             "type": "VEC3", "max": [0.5, 0.0, 0.5], "min": [-0.5, 0.0, -0.5]},
            {"bufferView": 3, "componentType": 5126, "count": 4,
             "type": "VEC2", "max": [1.0, 1.0], "min": [0.0, 0.0]},
            {"bufferView": 4, "componentType": 5126, "count": 3,
             "type": "VEC3", "max": [0.0, 0.0, 0.0], "min": [0.0, 0.0, 0.0]}
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0,   "byteLength": 12, "target": 34963},
            {"buffer": 0, "byteOffset": 12,  "byteLength": 6,  "target": 34963},
            {"buffer": 0, "byteOffset": 20,  "byteLength": 48, "target": 34962},
            {"buffer": 0, "byteOffset": 68,  "byteLength": 32, "target": 34962},
            {"buffer": 0, "byteOffset": 100, "byteLength": 36, "target": 34962}
        ],
        "buffers": [{"byteLength": 136}]
    }

    write_glb(os.path.join(ASSETS_DIR, "NoNormals.glb"), gltf, bytes(buf))


def generate_diffuse_transmission():
    """DiffuseTransmission.glb: triangle with KHR_materials_diffuse_transmission.

    Binary layout (44 bytes):
      [0..6)   indices      3 x uint16
      [6..8)   padding
      [8..44)  positions    3 x vec3
    """
    buf = bytearray()

    buf += struct.pack("<3H", 0, 1, 2)
    buf += b"\x00" * 2

    buf += struct.pack("<3f", -1.0, 0.0,  0.0)
    buf += struct.pack("<3f",  1.0, 0.0,  0.0)
    buf += struct.pack("<3f",  0.0, 0.0, -1.0)

    assert len(buf) == 44

    gltf = {
        "asset": {"version": "2.0", "generator": "monti-test"},
        "extensionsUsed": ["KHR_materials_diffuse_transmission"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "DiffuseTransmission"}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 1},
                "indices": 0,
                "material": 0
            }]
        }],
        "materials": [{
            "name": "diffuse_transmission_mat",
            "extensions": {
                "KHR_materials_diffuse_transmission": {
                    "diffuseTransmissionFactor": 0.75,
                    "diffuseTransmissionColorFactor": [0.8, 0.2, 0.1]
                }
            }
        }],
        "accessors": [
            {"bufferView": 0, "componentType": 5123, "count": 3,
             "type": "SCALAR", "max": [2], "min": [0]},
            {"bufferView": 1, "componentType": 5126, "count": 3,
             "type": "VEC3", "max": [1.0, 0.0, 0.0], "min": [-1.0, 0.0, -1.0]}
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 6, "target": 34963},
            {"buffer": 0, "byteOffset": 8, "byteLength": 36, "target": 34962}
        ],
        "buffers": [{"byteLength": 44}]
    }

    write_glb(os.path.join(ASSETS_DIR, "DiffuseTransmission.glb"), gltf, bytes(buf))


def generate_no_material():
    """NoMaterial.glb: triangle with no material reference.

    Binary layout (44 bytes):
      [0..6)   indices      3 x uint16
      [6..8)   padding
      [8..44)  positions    3 x vec3
    """
    buf = bytearray()

    buf += struct.pack("<3H", 0, 1, 2)
    buf += b"\x00" * 2

    buf += struct.pack("<3f", -1.0, 0.0,  0.0)
    buf += struct.pack("<3f",  1.0, 0.0,  0.0)
    buf += struct.pack("<3f",  0.0, 0.0, -1.0)

    assert len(buf) == 44

    gltf = {
        "asset": {"version": "2.0", "generator": "monti-test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "NoMaterial"}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 1},
                "indices": 0
            }]
        }],
        "accessors": [
            {"bufferView": 0, "componentType": 5123, "count": 3,
             "type": "SCALAR", "max": [2], "min": [0]},
            {"bufferView": 1, "componentType": 5126, "count": 3,
             "type": "VEC3", "max": [1.0, 0.0, 0.0], "min": [-1.0, 0.0, -1.0]}
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 6, "target": 34963},
            {"buffer": 0, "byteOffset": 8, "byteLength": 36, "target": 34962}
        ],
        "buffers": [{"byteLength": 44}]
    }

    write_glb(os.path.join(ASSETS_DIR, "NoMaterial.glb"), gltf, bytes(buf))


if __name__ == "__main__":
    os.makedirs(ASSETS_DIR, exist_ok=True)
    print("Generating test assets...")
    generate_no_normals()
    generate_diffuse_transmission()
    generate_no_material()
    print("Done.")
