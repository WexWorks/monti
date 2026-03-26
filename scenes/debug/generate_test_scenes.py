#!/usr/bin/env python3
"""Generate diagnostic test scenes for energy conservation validation.

Creates minimal glTF scenes and a uniform white EXR environment map
that can be loaded in Monti and vk_gltf_renderer for numerical comparison.

Test scenes:
  1. white_furnace.gltf  - White diffuse sphere (albedo=1, roughness=1, metallic=0)
     Expected result under uniform env: ~0.96 (= 1 - F0 with F0=0.04)
  2. grey_furnace.gltf   - Grey diffuse sphere (albedo=0.5, roughness=1, metallic=0)
     Expected result: ~0.48
  3. mirror_sphere.gltf   - Perfect mirror sphere (albedo=1, roughness=0, metallic=1)
     Expected result: should reflect env perfectly (value ~1.0)
  4. roughness_spheres.gltf - 5 spheres at roughness 0.0, 0.25, 0.5, 0.75, 1.0
     All white metallic=0, tests energy conservation across roughness range

Environment:
  uniform_white.exr - 16x8 uniform white (1.0, 1.0, 1.0) EXR

Usage:
  python generate_test_scenes.py [--output-dir DIR]

Default output: scenes/debug/
"""

import argparse
import base64
import json
import math
import os
import struct
import sys


# ─── UV sphere mesh generation ───────────────────────────────────────────────

def generate_uv_sphere(radius=1.0, lat_segments=32, lon_segments=64):
    """Generate a UV sphere with positions, normals, and triangle indices."""
    positions = []
    normals = []
    indices = []

    for lat in range(lat_segments + 1):
        theta = math.pi * lat / lat_segments
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        for lon in range(lon_segments + 1):
            phi = 2.0 * math.pi * lon / lon_segments
            x = sin_theta * math.cos(phi)
            y = cos_theta
            z = sin_theta * math.sin(phi)

            positions.extend([x * radius, y * radius, z * radius])
            normals.extend([x, y, z])

    for lat in range(lat_segments):
        for lon in range(lon_segments):
            a = lat * (lon_segments + 1) + lon
            b = a + lon_segments + 1

            indices.extend([a, a + 1, b])
            indices.extend([b, a + 1, b + 1])

    return positions, normals, indices


# ─── Binary buffer helpers ────────────────────────────────────────────────────

def pack_floats(values):
    return struct.pack(f"<{len(values)}f", *values)


def pack_indices(values):
    if max(values) > 65535:
        return struct.pack(f"<{len(values)}I", *values), 5125  # UNSIGNED_INT
    return struct.pack(f"<{len(values)}H", *values), 5123  # UNSIGNED_SHORT


def compute_min_max(values, stride):
    """Compute per-component min/max for accessor bounds."""
    mins = [float("inf")] * stride
    maxs = [float("-inf")] * stride
    for i in range(0, len(values), stride):
        for j in range(stride):
            v = values[i + j]
            mins[j] = min(mins[j], v)
            maxs[j] = max(maxs[j], v)
    return mins, maxs


def pad_to_4(data):
    """Pad byte data to 4-byte alignment."""
    remainder = len(data) % 4
    if remainder:
        data += b"\x00" * (4 - remainder)
    return data


# ─── glTF creation ────────────────────────────────────────────────────────────

def create_sphere_gltf(materials, sphere_positions=None, radius=0.5):
    """Create a glTF with one sphere per material.

    Args:
        materials: list of dicts with 'name', 'baseColorFactor', 'metallicFactor',
                   'roughnessFactor'.
        sphere_positions: list of [x, y, z] for each sphere. Defaults to origin.
        radius: sphere radius.
    """
    if sphere_positions is None:
        sphere_positions = [[0.0, 0.0, 0.0]] * len(materials)

    positions, normals, indices = generate_uv_sphere(radius=radius)
    pos_bytes = pack_floats(positions)
    norm_bytes = pack_floats(normals)
    idx_bytes, idx_component_type = pack_indices(indices)
    idx_bytes = pad_to_4(idx_bytes)

    pos_min, pos_max = compute_min_max(positions, 3)
    norm_min, norm_max = compute_min_max(normals, 3)

    vertex_count = len(positions) // 3
    index_count = len(indices)

    # Build binary buffer: [indices | positions | normals]
    buffer_data = idx_bytes + pos_bytes + norm_bytes
    buffer_base64 = base64.b64encode(buffer_data).decode("ascii")
    buffer_uri = f"data:application/octet-stream;base64,{buffer_base64}"

    idx_byte_length = len(idx_bytes)
    pos_byte_length = len(pos_bytes)
    norm_byte_length = len(norm_bytes)

    # glTF structure
    gltf = {
        "asset": {"version": "2.0", "generator": "monti_test_scene_generator"},
        "scene": 0,
        "scenes": [{"name": "TestScene", "nodes": list(range(len(materials)))}],
        "nodes": [],
        "meshes": [],
        "accessors": [
            {   # 0: indices
                "bufferView": 0,
                "componentType": idx_component_type,
                "count": index_count,
                "type": "SCALAR",
                "max": [max(indices)],
                "min": [0],
            },
            {   # 1: positions
                "bufferView": 1,
                "componentType": 5126,  # FLOAT
                "count": vertex_count,
                "type": "VEC3",
                "max": pos_max,
                "min": pos_min,
            },
            {   # 2: normals
                "bufferView": 2,
                "componentType": 5126,
                "count": vertex_count,
                "type": "VEC3",
                "max": norm_max,
                "min": norm_min,
            },
        ],
        "bufferViews": [
            {   # 0: indices
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": idx_byte_length,
                "target": 34963,  # ELEMENT_ARRAY_BUFFER
            },
            {   # 1: positions
                "buffer": 0,
                "byteOffset": idx_byte_length,
                "byteLength": pos_byte_length,
                "target": 34962,  # ARRAY_BUFFER
            },
            {   # 2: normals
                "buffer": 0,
                "byteOffset": idx_byte_length + pos_byte_length,
                "byteLength": norm_byte_length,
                "target": 34962,
            },
        ],
        "buffers": [
            {
                "uri": buffer_uri,
                "byteLength": len(buffer_data),
            }
        ],
        "materials": [],
    }

    for i, mat in enumerate(materials):
        gltf["nodes"].append({
            "name": mat["name"],
            "mesh": i,
            "translation": sphere_positions[i],
        })
        gltf["meshes"].append({
            "name": f"Sphere_{mat['name']}",
            "primitives": [{
                "attributes": {"POSITION": 1, "NORMAL": 2},
                "indices": 0,
                "material": i,
            }],
        })
        gltf["materials"].append({
            "name": mat["name"],
            "pbrMetallicRoughness": {
                "baseColorFactor": mat["baseColorFactor"],
                "metallicFactor": mat["metallicFactor"],
                "roughnessFactor": mat["roughnessFactor"],
            },
        })

    return gltf


# ─── Minimal OpenEXR writer ──────────────────────────────────────────────────

def write_exr_uniform(path, width, height, r, g, b):
    """Write a minimal OpenEXR 2.0 file with uniform RGB half-float pixels.

    Uses scanline, no compression, HALF pixel type.
    """
    import struct as st

    def float_to_half(f):
        """Convert a Python float to IEEE 754 half-precision (16-bit)."""
        # Use struct to get the 32-bit float bits
        bits = st.unpack(">I", st.pack(">f", f))[0]
        sign = (bits >> 31) & 1
        exp = (bits >> 23) & 0xFF
        frac = bits & 0x7FFFFF

        if exp == 0:
            # Zero or denorm
            return sign << 15
        elif exp == 0xFF:
            # Inf or NaN
            if frac:
                return (sign << 15) | 0x7E00  # NaN
            return (sign << 15) | 0x7C00  # Inf

        new_exp = exp - 127 + 15
        if new_exp >= 31:
            return (sign << 15) | 0x7C00  # Overflow -> Inf
        if new_exp <= 0:
            if new_exp < -10:
                return sign << 15  # Too small -> zero
            frac = (frac | 0x800000) >> (1 - new_exp)
            return (sign << 15) | (frac >> 13)

        return (sign << 15) | (new_exp << 10) | (frac >> 13)

    def write_null_terminated(buf, s):
        buf.extend(s.encode("ascii"))
        buf.append(0)

    def write_attr(buf, name, type_name, value_bytes):
        write_null_terminated(buf, name)
        write_null_terminated(buf, type_name)
        buf.extend(st.pack("<I", len(value_bytes)))
        buf.extend(value_bytes)

    # Channel descriptor: name\0 + (pixel_type:i32, pLinear:u8, reserved:3bytes, xSamp:i32, ySamp:i32)
    def channel_entry(name, pixel_type=1):
        """pixel_type: 0=UINT, 1=HALF, 2=FLOAT"""
        entry = name.encode("ascii") + b"\x00"
        entry += st.pack("<i", pixel_type)  # pixelType = HALF
        entry += st.pack("<B", 1)  # pLinear = 1 (linear data)
        entry += b"\x00\x00\x00"  # reserved
        entry += st.pack("<ii", 1, 1)  # xSampling, ySampling
        return entry

    header = bytearray()

    # Channels attribute (must be sorted alphabetically: B, G, R)
    chlist_data = channel_entry("B") + channel_entry("G") + channel_entry("R") + b"\x00"
    write_attr(header, "channels", "chlist", chlist_data)

    # Compression: 0 = NO_COMPRESSION
    write_attr(header, "compression", "compression", st.pack("<B", 0))

    # Data window: (xMin, yMin, xMax, yMax)
    write_attr(header, "dataWindow", "box2i",
               st.pack("<iiii", 0, 0, width - 1, height - 1))

    # Display window
    write_attr(header, "displayWindow", "box2i",
               st.pack("<iiii", 0, 0, width - 1, height - 1))

    # Line order: 0 = INCREASING_Y
    write_attr(header, "lineOrder", "lineOrder", st.pack("<B", 0))

    # Pixel aspect ratio
    write_attr(header, "pixelAspectRatio", "float", st.pack("<f", 1.0))

    # Screen window center
    write_attr(header, "screenWindowCenter", "v2f", st.pack("<ff", 0.0, 0.0))

    # Screen window width
    write_attr(header, "screenWindowWidth", "float", st.pack("<f", 1.0))

    # End of header
    header.append(0)

    # Magic number + version
    magic = st.pack("<I", 20000630)  # OpenEXR magic
    version = st.pack("<I", 2)  # Version 2, flags=0 (single-part scanline)

    # Compute offsets
    header_size = len(magic) + len(version) + len(header)

    # Scanline data: each scanline has 3 channels (B, G, R) of width half-floats
    bytes_per_channel = width * 2  # 2 bytes per half
    bytes_per_scanline = 3 * bytes_per_channel
    # Each scanline block: y_coord(i32) + pixel_data_size(i32) + pixel_data
    block_header_size = 8
    block_size = block_header_size + bytes_per_scanline

    # Offset table: one uint64 per scanline
    offset_table_size = height * 8
    scanline_data_start = header_size + offset_table_size

    # Build offset table
    offset_table = bytearray()
    for y in range(height):
        offset = scanline_data_start + y * block_size
        offset_table.extend(st.pack("<Q", offset))

    # Build scanline data
    half_r = float_to_half(r)
    half_g = float_to_half(g)
    half_b = float_to_half(b)

    # Pre-pack one scanline of each channel
    b_channel = st.pack(f"<{width}H", *([half_b] * width))
    g_channel = st.pack(f"<{width}H", *([half_g] * width))
    r_channel = st.pack(f"<{width}H", *([half_r] * width))
    scanline_pixels = b_channel + g_channel + r_channel

    scanline_data = bytearray()
    for y in range(height):
        scanline_data.extend(st.pack("<i", y))  # y coordinate
        scanline_data.extend(st.pack("<i", bytes_per_scanline))  # data size
        scanline_data.extend(scanline_pixels)

    # Write file
    with open(path, "wb") as f:
        f.write(magic)
        f.write(version)
        f.write(header)
        f.write(offset_table)
        f.write(scanline_data)


# ─── Radiance HDR writer ─────────────────────────────────────────────────────

def write_hdr_uniform(path, width, height, r, g, b):
    """Write a Radiance RGBE (.hdr) file with uniform RGB pixels."""

    def float_to_rgbe(r, g, b):
        """Convert linear RGB floats to RGBE 4-byte encoding."""
        v = max(r, g, b)
        if v < 1e-32:
            return bytes([0, 0, 0, 0])
        import math as m
        mantissa, exp = m.frexp(v)
        # Scale so that max channel maps to mantissa * 256
        scale = mantissa * 256.0 / v
        return bytes([
            max(0, min(255, int(r * scale))),
            max(0, min(255, int(g * scale))),
            max(0, min(255, int(b * scale))),
            exp + 128,
        ])

    rgbe = float_to_rgbe(r, g, b)

    with open(path, "wb") as f:
        # Radiance header
        f.write(b"#?RADIANCE\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n")
        f.write(b"\n")  # Empty line ends header
        # Resolution string: -Y height +X width (top-to-bottom, left-to-right)
        f.write(f"-Y {height} +X {width}\n".encode("ascii"))
        # Uncompressed scanline data (no RLE for small images)
        for _ in range(height):
            for _ in range(width):
                f.write(rgbe)


# ─── Scene definitions ────────────────────────────────────────────────────────

def generate_all(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. White furnace test
    white_furnace = create_sphere_gltf([{
        "name": "WhiteDiffuse",
        "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 1.0,
    }])
    write_gltf(os.path.join(output_dir, "white_furnace.gltf"), white_furnace)
    print("  white_furnace.gltf  - white diffuse sphere (expect ~0.96)")

    # 2. Grey furnace test
    grey_furnace = create_sphere_gltf([{
        "name": "GreyDiffuse",
        "baseColorFactor": [0.5, 0.5, 0.5, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 1.0,
    }])
    write_gltf(os.path.join(output_dir, "grey_furnace.gltf"), grey_furnace)
    print("  grey_furnace.gltf   - grey diffuse sphere (expect ~0.48)")

    # 3. Mirror sphere
    mirror_sphere = create_sphere_gltf([{
        "name": "PerfectMirror",
        "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
        "metallicFactor": 1.0,
        "roughnessFactor": 0.0,
    }])
    write_gltf(os.path.join(output_dir, "mirror_sphere.gltf"), mirror_sphere)
    print("  mirror_sphere.gltf  - perfect mirror metal (expect ~1.0)")

    # 4. Roughness ladder - 5 spheres at different roughness values
    roughness_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    roughness_mats = []
    roughness_positions = []
    spacing = 1.5
    offset = -(len(roughness_values) - 1) * spacing / 2.0

    for i, r in enumerate(roughness_values):
        roughness_mats.append({
            "name": f"Dielectric_R{r:.2f}",
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": r,
        })
        roughness_positions.append([offset + i * spacing, 0.0, 0.0])

    roughness_ladder = create_sphere_gltf(roughness_mats, roughness_positions)
    write_gltf(os.path.join(output_dir, "roughness_ladder.gltf"), roughness_ladder)
    print("  roughness_ladder.gltf - 5 white dielectric spheres, roughness 0..1")

    # 5. Metal roughness ladder
    metal_mats = []
    metal_positions = []
    for i, r in enumerate(roughness_values):
        metal_mats.append({
            "name": f"Metal_R{r:.2f}",
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": 1.0,
            "roughnessFactor": r,
        })
        metal_positions.append([offset + i * spacing, 0.0, 0.0])

    metal_ladder = create_sphere_gltf(metal_mats, metal_positions)
    write_gltf(os.path.join(output_dir, "metal_roughness_ladder.gltf"), metal_ladder)
    print("  metal_roughness_ladder.gltf - 5 white metal spheres, roughness 0..1")

    # 6. Uniform white EXR environment map (16x8, constant 1.0)
    exr_path = os.path.join(output_dir, "uniform_white.exr")
    write_exr_uniform(exr_path, 16, 8, 1.0, 1.0, 1.0)
    print("  uniform_white.exr   - 16x8 uniform white (1,1,1) environment")

    # 7. Uniform grey EXR (0.5) for secondary tests
    exr_grey_path = os.path.join(output_dir, "uniform_grey.exr")
    write_exr_uniform(exr_grey_path, 16, 8, 0.5, 0.5, 0.5)
    print("  uniform_grey.exr    - 16x8 uniform grey (0.5) environment")

    # 8. Uniform white HDR (for vk_gltf_renderer which loads .hdr)
    hdr_white_path = os.path.join(output_dir, "uniform_white.hdr")
    write_hdr_uniform(hdr_white_path, 16, 8, 1.0, 1.0, 1.0)
    print("  uniform_white.hdr   - 16x8 uniform white (1,1,1) Radiance HDR")

    # 9. Uniform grey HDR
    hdr_grey_path = os.path.join(output_dir, "uniform_grey.hdr")
    write_hdr_uniform(hdr_grey_path, 16, 8, 0.5, 0.5, 0.5)
    print("  uniform_grey.hdr    - 16x8 uniform grey (0.5) Radiance HDR")

    print(f"\nAll files written to: {os.path.abspath(output_dir)}")
    print("\nUsage examples:")
    print(f"  monti_view {output_dir}/white_furnace.gltf --env {output_dir}/uniform_white.exr")
    print(f"  monti_view {output_dir}/roughness_ladder.gltf --env {output_dir}/uniform_white.exr")
    print("\nExpected values under uniform white env (no tone mapping / debug mode):")
    print("  White diffuse:   ~0.96 per channel (1 - F0, F0=0.04)")
    print("  Grey diffuse:    ~0.48 per channel")
    print("  Mirror sphere:   ~1.0  per channel (reflects env directly)")
    print("  Roughness sweep: All should be ~0.96 (energy conserved at all roughness)")
    print("  Metal sweep:     All should be ~1.0 (energy conserved at all roughness)")
    print("                   Note: single-scatter GGX loses energy at high roughness")


def write_gltf(path, gltf_dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gltf_dict, f, indent=2)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate diagnostic test scenes")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "debug"),
        help="Output directory for generated files",
    )
    args = parser.parse_args()

    print("Generating diagnostic test scenes...")
    generate_all(args.output_dir)
