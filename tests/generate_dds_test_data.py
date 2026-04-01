#!/usr/bin/env python3
"""Generate DDS test textures and glTF test scenes for Phase 8N tests.

Creates:
  - test_bc1.dds  (64x64, 7 mip levels, BC1/DXT1 format)
  - test_bc5.dds  (64x64, 7 mip levels, BC5/ATI2 format)
  - test_bc7.dds  (64x64, 7 mip levels, BC7 format)
  - dds_quad.gltf (glTF scene referencing Cloth_BaseColor.dds)
  - dds_normal_sphere.gltf (glTF scene referencing Cloth_Normal.dds)

The DDS files contain valid headers and deterministic block data.
They are small (~5-22 KB each) and used for GPU integration tests.
"""

import argparse
import json
import math
import os
import struct


def write_dds_dx10(path: str, width: int, height: int, mip_levels: int,
                   dxgi_format: int, block_size: int):
    """Write a DDS file with DX10 extended header and deterministic block data."""
    # DDS magic
    magic = b'DDS '

    # Compute pitch for first mip (used in header)
    bx = max((width + 3) // 4, 1)
    pitch = bx * block_size

    # DDS_HEADER (124 bytes)
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000  # CAPS|HEIGHT|WIDTH|PIXELFORMAT|MIPMAPCOUNT|LINEARSIZE
    caps = 0x1000 | 0x8 | 0x400000  # TEXTURE | COMPLEX | MIPMAP

    # Compute total linear size (all mips)
    total_size = 0
    w, h = width, height
    for _ in range(mip_levels):
        bx_m = max((w + 3) // 4, 1)
        by_m = max((h + 3) // 4, 1)
        total_size += bx_m * by_m * block_size
        w = max(w // 2, 1)
        h = max(h // 2, 1)

    header = struct.pack('<IIIIIII',
        124,          # dwSize
        flags,        # dwFlags
        height,       # dwHeight
        width,        # dwWidth
        pitch,        # dwPitchOrLinearSize
        0,            # dwDepth
        mip_levels,   # dwMipMapCount
    )
    header += b'\x00' * (11 * 4)  # dwReserved1[11]

    # DDS_PIXELFORMAT (32 bytes) — use FOURCC 'DX10' for extended header
    pf = struct.pack('<II4sIIIII',
        32,                    # dwSize
        0x4,                   # dwFlags (FOURCC)
        b'DX10',               # dwFourCC
        0, 0, 0, 0, 0         # remaining fields unused
    )
    header += pf

    header += struct.pack('<IIIII',
        caps,          # dwCaps
        0, 0, 0, 0    # dwCaps2-4, dwReserved2
    )

    # DDS_HEADER_DXT10 (20 bytes)
    dx10 = struct.pack('<IIIII',
        dxgi_format,   # dxgiFormat
        3,             # resourceDimension = D3D10_RESOURCE_DIMENSION_TEXTURE2D
        0,             # miscFlag
        1,             # arraySize
        0,             # miscFlags2
    )

    # Generate deterministic block data
    data = bytearray()
    w, h = width, height
    for mip in range(mip_levels):
        bx_m = max((w + 3) // 4, 1)
        by_m = max((h + 3) // 4, 1)
        num_blocks = bx_m * by_m
        for block_idx in range(num_blocks):
            # Fill each block with a deterministic pattern based on mip + block index
            seed = (mip * 10007 + block_idx * 31) & 0xFFFFFFFF
            block_bytes = bytearray(block_size)
            for i in range(block_size):
                block_bytes[i] = ((seed >> (i % 4) * 8) + i * 37) & 0xFF
            data.extend(block_bytes)
        w = max(w // 2, 1)
        h = max(h // 2, 1)

    with open(path, 'wb') as f:
        f.write(magic)
        f.write(header)
        f.write(dx10)
        f.write(bytes(data))


def write_gltf_with_dds_texture(path: str, dds_filename: str,
                                is_normal_map: bool):
    """Write a minimal glTF file that references a DDS texture via URI."""
    # Simple quad or sphere geometry as base64-embedded buffer
    if is_normal_map:
        # UV sphere with tangents for normal mapping
        gltf = _make_sphere_gltf(dds_filename, is_normal_map=True)
    else:
        # Simple textured quad
        gltf = _make_quad_gltf(dds_filename)

    with open(path, 'w') as f:
        json.dump(gltf, f, indent=2)


def _make_quad_gltf(dds_filename: str) -> dict:
    """Create a glTF with a textured quad referencing a DDS file."""
    import base64

    # Quad vertices: position(3) + normal(3) + texcoord(2) = 8 floats per vertex
    positions = [
        -1, -1, 0,
         1, -1, 0,
         1,  1, 0,
        -1,  1, 0,
    ]
    normals = [0, 0, 1] * 4
    texcoords = [0, 1, 1, 1, 1, 0, 0, 0]
    tangents = [1, 0, 0, 1] * 4
    indices = [0, 1, 2, 0, 2, 3]

    # Pack binary data
    buf = bytearray()

    # Indices (uint16)
    idx_offset = len(buf)
    for i in indices:
        buf.extend(struct.pack('<H', i))
    idx_size = len(buf) - idx_offset
    # Pad to 4-byte alignment
    while len(buf) % 4:
        buf.append(0)

    # Positions (float32)
    pos_offset = len(buf)
    for v in positions:
        buf.extend(struct.pack('<f', v))
    pos_size = len(buf) - pos_offset

    # Normals (float32)
    norm_offset = len(buf)
    for v in normals:
        buf.extend(struct.pack('<f', v))
    norm_size = len(buf) - norm_offset

    # Texcoords (float32)
    tc_offset = len(buf)
    for v in texcoords:
        buf.extend(struct.pack('<f', v))
    tc_size = len(buf) - tc_offset

    # Tangents (float32)
    tan_offset = len(buf)
    for v in tangents:
        buf.extend(struct.pack('<f', v))
    tan_size = len(buf) - tan_offset

    b64 = base64.b64encode(bytes(buf)).decode('ascii')

    return {
        "asset": {"version": "2.0", "generator": "monti-test-gen"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 1,
                    "NORMAL": 2,
                    "TEXCOORD_0": 3,
                    "TANGENT": 4,
                },
                "indices": 0,
                "material": 0,
            }]
        }],
        "materials": [{
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 0.5,
            }
        }],
        "textures": [{"source": 0}],
        "images": [{"uri": dds_filename}],
        "accessors": [
            {"bufferView": 0, "componentType": 5123, "count": 6,
             "type": "SCALAR", "max": [3], "min": [0]},
            {"bufferView": 1, "componentType": 5126, "count": 4,
             "type": "VEC3", "max": [1, 1, 0], "min": [-1, -1, 0]},
            {"bufferView": 2, "componentType": 5126, "count": 4,
             "type": "VEC3", "max": [0, 0, 1], "min": [0, 0, 1]},
            {"bufferView": 3, "componentType": 5126, "count": 4,
             "type": "VEC2", "max": [1, 1], "min": [0, 0]},
            {"bufferView": 4, "componentType": 5126, "count": 4,
             "type": "VEC4", "max": [1, 0, 0, 1], "min": [1, 0, 0, 1]},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": idx_offset, "byteLength": idx_size},
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": pos_size},
            {"buffer": 0, "byteOffset": norm_offset, "byteLength": norm_size},
            {"buffer": 0, "byteOffset": tc_offset, "byteLength": tc_size},
            {"buffer": 0, "byteOffset": tan_offset, "byteLength": tan_size},
        ],
        "buffers": [{
            "uri": f"data:application/octet-stream;base64,{b64}",
            "byteLength": len(buf),
        }],
    }


def _make_sphere_gltf(dds_filename: str, is_normal_map: bool) -> dict:
    """Create a glTF with a UV sphere referencing a DDS normal map."""
    import base64

    # Generate UV sphere
    stacks, slices = 16, 32
    positions = []
    normals = []
    texcoords = []
    tangents = []
    indices_list = []

    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        for j in range(slices + 1):
            theta = 2.0 * math.pi * j / slices
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            positions.extend([x, y, z])
            normals.extend([x, y, z])
            texcoords.extend([j / slices, i / stacks])
            # Tangent along theta direction
            tx = -math.sin(theta)
            ty = 0.0
            tz = math.cos(theta)
            tangents.extend([tx, ty, tz, 1.0])

    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            indices_list.extend([a, b, a + 1, b, b + 1, a + 1])

    num_vertices = (stacks + 1) * (slices + 1)
    num_indices = len(indices_list)

    # Pack binary data
    buf = bytearray()

    # Indices (uint16)
    idx_offset = len(buf)
    for i in indices_list:
        buf.extend(struct.pack('<H', i))
    idx_size = len(buf) - idx_offset
    while len(buf) % 4:
        buf.append(0)

    # Positions
    pos_offset = len(buf)
    for v in positions:
        buf.extend(struct.pack('<f', v))
    pos_size = len(buf) - pos_offset

    # Normals
    norm_offset = len(buf)
    for v in normals:
        buf.extend(struct.pack('<f', v))
    norm_size = len(buf) - norm_offset

    # Texcoords
    tc_offset = len(buf)
    for v in texcoords:
        buf.extend(struct.pack('<f', v))
    tc_size = len(buf) - tc_offset

    # Tangents
    tan_offset = len(buf)
    for v in tangents:
        buf.extend(struct.pack('<f', v))
    tan_size = len(buf) - tan_offset

    b64 = base64.b64encode(bytes(buf)).decode('ascii')

    # Compute min/max for positions
    pos_min = [min(positions[i::3]) for i in range(3)]
    pos_max = [max(positions[i::3]) for i in range(3)]
    tc_min = [min(texcoords[i::2]) for i in range(2)]
    tc_max = [max(texcoords[i::2]) for i in range(2)]

    material = {
        "pbrMetallicRoughness": {
            "metallicFactor": 0.0,
            "roughnessFactor": 0.5,
        },
        "normalTexture": {"index": 0},
    } if is_normal_map else {
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicFactor": 0.0,
            "roughnessFactor": 0.5,
        }
    }

    # For normal map material, normalTexture is a property of the material, not pbrMetallicRoughness
    gltf = {
        "asset": {"version": "2.0", "generator": "monti-test-gen"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 1,
                    "NORMAL": 2,
                    "TEXCOORD_0": 3,
                    "TANGENT": 4,
                },
                "indices": 0,
                "material": 0,
            }]
        }],
        "materials": [material],
        "textures": [{"source": 0}],
        "images": [{"uri": dds_filename}],
        "accessors": [
            {"bufferView": 0, "componentType": 5123, "count": num_indices,
             "type": "SCALAR", "max": [num_vertices - 1], "min": [0]},
            {"bufferView": 1, "componentType": 5126, "count": num_vertices,
             "type": "VEC3", "max": pos_max, "min": pos_min},
            {"bufferView": 2, "componentType": 5126, "count": num_vertices,
             "type": "VEC3"},
            {"bufferView": 3, "componentType": 5126, "count": num_vertices,
             "type": "VEC2", "max": tc_max, "min": tc_min},
            {"bufferView": 4, "componentType": 5126, "count": num_vertices,
             "type": "VEC4"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": idx_offset, "byteLength": idx_size},
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": pos_size},
            {"buffer": 0, "byteOffset": norm_offset, "byteLength": norm_size},
            {"buffer": 0, "byteOffset": tc_offset, "byteLength": tc_size},
            {"buffer": 0, "byteOffset": tan_offset, "byteLength": tan_size},
        ],
        "buffers": [{
            "uri": f"data:application/octet-stream;base64,{b64}",
            "byteLength": len(buf),
        }],
    }
    return gltf


# DXGI format constants
DXGI_FORMAT_BC1_UNORM = 71
DXGI_FORMAT_BC5_UNORM = 83
DXGI_FORMAT_BC7_UNORM = 98


def main():
    parser = argparse.ArgumentParser(description='Generate Phase 8N test DDS assets')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for generated files')
    parser.add_argument('--bistro-dir', default=None,
                        help='Path to BistroInterior scene (for Cloth DDS textures)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate small test DDS textures (64x64, 7 mip levels)
    write_dds_dx10(os.path.join(args.output_dir, 'test_bc1.dds'),
                   64, 64, 7, DXGI_FORMAT_BC1_UNORM, block_size=8)
    write_dds_dx10(os.path.join(args.output_dir, 'test_bc5.dds'),
                   64, 64, 7, DXGI_FORMAT_BC5_UNORM, block_size=16)
    write_dds_dx10(os.path.join(args.output_dir, 'test_bc7.dds'),
                   64, 64, 7, DXGI_FORMAT_BC7_UNORM, block_size=16)

    print(f"Generated test DDS textures in {args.output_dir}")

    # Generate glTF test scenes if Bistro DDS textures are available
    if args.bistro_dir:
        cloth_base = os.path.join(args.bistro_dir, 'Cloth_BaseColor.dds')
        cloth_normal = os.path.join(args.bistro_dir, 'Cloth_Normal.dds')

        if os.path.exists(cloth_base):
            # Copy or symlink the DDS texture, then write the glTF
            import shutil
            dst = os.path.join(args.output_dir, 'Cloth_BaseColor.dds')
            if not os.path.exists(dst):
                shutil.copy2(cloth_base, dst)
            write_gltf_with_dds_texture(
                os.path.join(args.output_dir, 'dds_quad.gltf'),
                'Cloth_BaseColor.dds', is_normal_map=False)
            print(f"Generated dds_quad.gltf with Cloth_BaseColor.dds")
        else:
            print(f"WARNING: {cloth_base} not found, skipping dds_quad.gltf")

        if os.path.exists(cloth_normal):
            import shutil
            dst = os.path.join(args.output_dir, 'Cloth_Normal.dds')
            if not os.path.exists(dst):
                shutil.copy2(cloth_normal, dst)
            write_gltf_with_dds_texture(
                os.path.join(args.output_dir, 'dds_normal_sphere.gltf'),
                'Cloth_Normal.dds', is_normal_map=True)
            print(f"Generated dds_normal_sphere.gltf with Cloth_Normal.dds")
        else:
            print(f"WARNING: {cloth_normal} not found, skipping dds_normal_sphere.gltf")
    else:
        print("No --bistro-dir provided, skipping glTF DDS scene generation")


if __name__ == '__main__':
    main()
