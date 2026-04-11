"""Generate golden reference data for GPU inference validation.

Produces binary files containing:
  v3 (golden_ref_v3.bin):
    - Weights (.denimodel format)
    - Input temporal+G-buffer (26 channels, FP16 packed per image)
    - Expected output (3 channels, FP32 — remodulated radiance, single-frame)

Both GPU shaders and PyTorch use zero-padding for Conv2d (padding=1).

SYNC REQUIREMENT: This script imports DeniTemporalResidualNet
directly from training/deni_train/models/. If the model architecture changes,
regenerate the golden reference:

    cd training
    python ../tests/generate_golden_reference.py

Usage:
    cd training
    python ../tests/generate_golden_reference.py
"""

import argparse
import os
import struct
import sys

import numpy as np
import torch
import torch.nn as nn

# Add training directory to path for model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from deni_train.models.temporal_unet import DeniTemporalResidualNet

# Test configuration -- small enough for fast CI, large enough for meaningful validation
WIDTH = 32
HEIGHT = 32
BASE_CHANNELS = 8


def make_deterministic_input(width: int, height: int) -> torch.Tensor:
    """Create a deterministic 19-channel input tensor in [0, 1] range.

    Channel layout matches the albedo-demodulated pipeline:
      0-2:   demodulated diffuse irradiance
      3-5:   demodulated specular irradiance
      6-8:   world normals
      9:     roughness
      10:    linear depth
      11-12: motion vectors
      13-15: diffuse albedo
      16-18: specular albedo
    """
    torch.manual_seed(42)
    # Use smooth gradients + small noise for realistic-ish G-buffer data
    x = torch.linspace(0, 1, width).unsqueeze(0).expand(height, width)
    y = torch.linspace(0, 1, height).unsqueeze(1).expand(height, width)

    channels = []
    for ch in range(19):
        # Each channel: smooth gradient + small deterministic variation
        base = (x * (ch + 1) * 0.3 + y * (19 - ch) * 0.2) % 1.0
        noise = torch.randn(height, width) * 0.02
        channels.append((base + noise).clamp(0.0, 1.0))

    return torch.stack(channels).unsqueeze(0)  # [1, 19, H, W]


def quantize_to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to FP16 precision (round-trip through half)."""
    return tensor.half().float()


def pack_gbuffer(input_19ch: torch.Tensor) -> dict:
    """Pack 19-channel input into G-buffer image arrays.

    Returns dict with keys matching DenoiserInput image names.
    All images use RGBA16F or RG16F format (uint16 arrays).

    Channel mapping (19 channels):
      noisy_diffuse:   channels 0-2 (RGB), A=1.0      -> RGBA16F (uint16)
      noisy_specular:  channels 3-5 (RGB), A=1.0      -> RGBA16F (uint16)
      world_normals:   channels 6-8 (XYZ), ch 9 (.w)  -> RGBA16F (uint16)
      linear_depth:    channel 10                      -> RG16F (uint16)
      motion_vectors:  channels 11-12                  -> RG16F (uint16)
      diffuse_albedo:  channels 13-15 (RGB), A=0.0     -> RGBA16F (uint16)
      specular_albedo: channels 16-18 (RGB), A=0.0     -> RGBA16F (uint16)
    """
    # input_19ch shape: [1, 19, H, W]
    data = input_19ch.squeeze(0)  # [19, H, W]

    def to_fp16_bytes(t: torch.Tensor) -> np.ndarray:
        return t.half().numpy().view(np.uint16)

    # RGBA16F images: interleave channels per pixel [H, W, 4]
    def make_rgba(r, g, b, a=None):
        if a is None:
            a = torch.ones_like(r)
        return torch.stack([r, g, b, a], dim=-1)  # [H, W, 4]

    diffuse = make_rgba(data[0], data[1], data[2])
    specular = make_rgba(data[3], data[4], data[5])
    normals = make_rgba(data[6], data[7], data[8], data[9])

    # RG16F for depth: .r = linear depth, .g = 0 (matches G-buffer format)
    depth = torch.stack([data[10], torch.zeros_like(data[10])], dim=-1)

    # RG16F: [H, W, 2]
    motion = torch.stack([data[11], data[12]], dim=-1)

    # RGBA16F albedo images: [H, W, 4] with A=0
    diff_albedo = make_rgba(data[13], data[14], data[15], torch.zeros_like(data[13]))
    spec_albedo = make_rgba(data[16], data[17], data[18], torch.zeros_like(data[16]))

    return {
        "diffuse": to_fp16_bytes(diffuse.contiguous()),
        "specular": to_fp16_bytes(specular.contiguous()),
        "normals": to_fp16_bytes(normals.contiguous()),
        "depth": to_fp16_bytes(depth.contiguous()),
        "motion": to_fp16_bytes(motion.contiguous()),
        "diffuse_albedo": to_fp16_bytes(diff_albedo.contiguous()),
        "specular_albedo": to_fp16_bytes(spec_albedo.contiguous()),
    }


def write_denimodel_bytes(state_dict: dict) -> bytes:
    """Write .denimodel binary format to bytes."""
    layers = []
    total_weight_bytes = 0
    for name, tensor in state_dict.items():
        data = tensor.detach().cpu().float().numpy()
        layers.append((name, data))
        total_weight_bytes += data.nbytes

    parts = []
    # Header
    parts.append(b"DENI")
    parts.append(struct.pack("<I", 1))  # version
    parts.append(struct.pack("<I", len(layers)))
    parts.append(struct.pack("<I", total_weight_bytes))

    for name, data in layers:
        name_bytes = name.encode("utf-8")
        parts.append(struct.pack("<I", len(name_bytes)))
        parts.append(name_bytes)
        parts.append(struct.pack("<I", data.ndim))
        for dim in data.shape:
            parts.append(struct.pack("<I", dim))
        parts.append(data.astype(np.float32).tobytes())

    return b"".join(parts)


V3_BASE_CHANNELS = 8
V3_BASE_CHANNELS_12 = 12


def write_golden_reference_v3(output_path: str, base_channels: int = V3_BASE_CHANNELS):
    """Generate and write v3 temporal golden reference binary file.

    For the first-frame test, temporal channels (reprojected diffuse/specular/
    disocclusion) are all zeros (no history available). The model should still
    produce reasonable output via the residual path (weight → 1.0 when
    disocclusion mask is zero).

    Binary format (version 2):
      [4 bytes]   magic: "GREF"
      [4 bytes]   version: 2
      [4 bytes]   width
      [4 bytes]   height
      [4 bytes]   base_channels
      [4 bytes]   denimodel_size (bytes)
      [N bytes]   denimodel data
      [4 bytes]   num_gbuffer_images (7)
      For each G-buffer image:
        [4 bytes]   name_length
        [M bytes]   name (UTF-8)
        [4 bytes]   data_size (bytes)
        [D bytes]   pixel data (FP16 binary)
      [4 bytes]   num_temporal_images (3)
      For each temporal image:
        [4 bytes]   name_length
        [M bytes]   name (UTF-8)
        [4 bytes]   data_size (bytes)
        [D bytes]   pixel data (FP16 binary)
      [4 bytes]   output_size (bytes)
      [O bytes]   expected output (FP32, [3][H][W] channel-major — remodulated radiance)
    """
    torch.manual_seed(42)

    model = DeniTemporalResidualNet(base_channels=base_channels)
    model.eval()

    # Create deterministic G-buffer input (same as v1)
    input_tensor = make_deterministic_input(WIDTH, HEIGHT)
    input_fp16 = quantize_to_fp16(input_tensor)

    DEMOD_EPS = 0.001

    # Build 26-channel model input
    # Temporal channels [0:7]: zeros for first-frame test
    temporal_channels = torch.zeros(1, 7, HEIGHT, WIDTH)
    temporal_fp16 = quantize_to_fp16(temporal_channels)

    # G-buffer channels [7:26]: same 19ch as v1, but demodulated
    model_input_gbuf = input_fp16.clone()
    albedo_d = input_fp16[:, 13:16].clamp(min=DEMOD_EPS)
    albedo_s = input_fp16[:, 16:19].clamp(min=DEMOD_EPS)
    model_input_gbuf[:, 0:3] = input_fp16[:, 0:3] / albedo_d
    model_input_gbuf[:, 3:6] = input_fp16[:, 3:6] / albedo_s

    model_input = torch.cat([temporal_fp16, model_input_gbuf], dim=1)  # [1, 26, H, W]

    # Run PyTorch inference → 6-channel demodulated irradiance
    with torch.no_grad():
        output, _ = model(model_input)  # [1, 6, H, W]

    denoised = output.squeeze(0)  # [6, H, W]

    # Remodulate
    albedo_d_t = input_fp16[0, 13:16]
    albedo_s_t = input_fp16[0, 16:19]
    diff_irrad = denoised[:3]
    spec_irrad = denoised[3:6]
    remod_diff = diff_irrad * torch.clamp(albedo_d_t, min=DEMOD_EPS)
    remod_spec = spec_irrad * torch.clamp(albedo_s_t, min=DEMOD_EPS)
    expected_output = remod_diff + remod_spec  # [3, H, W]

    # Pack data
    denimodel_bytes = write_denimodel_bytes(model.state_dict())
    gbuffer = pack_gbuffer(input_fp16)

    # Pack temporal images as zeroed FP16
    def to_fp16_bytes(t: torch.Tensor) -> np.ndarray:
        return t.half().numpy().view(np.uint16)

    # Temporal images: RGBA16F for reprojected diffuse/specular, R16F for disocclusion
    reprojected_diffuse = torch.zeros(HEIGHT, WIDTH, 4)   # RGBA16F
    reprojected_specular = torch.zeros(HEIGHT, WIDTH, 4)  # RGBA16F
    disocclusion_mask = torch.zeros(HEIGHT, WIDTH, 1)     # R16F

    temporal_images = {
        "reprojected_diffuse": to_fp16_bytes(reprojected_diffuse.contiguous()),
        "reprojected_specular": to_fp16_bytes(reprojected_specular.contiguous()),
        "disocclusion_mask": to_fp16_bytes(disocclusion_mask.contiguous()),
    }

    # Write binary file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b"GREF")
        f.write(struct.pack("<I", 2))  # version 2 for v3 temporal
        f.write(struct.pack("<I", WIDTH))
        f.write(struct.pack("<I", HEIGHT))
        f.write(struct.pack("<I", base_channels))

        f.write(struct.pack("<I", len(denimodel_bytes)))
        f.write(denimodel_bytes)

        # G-buffer images (same 7 as v1)
        image_names = ["diffuse", "specular", "normals", "depth", "motion",
                       "diffuse_albedo", "specular_albedo"]
        f.write(struct.pack("<I", len(image_names)))
        for name in image_names:
            data = gbuffer[name].tobytes()
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(data)))
            f.write(data)

        # Temporal images (3)
        temporal_names = ["reprojected_diffuse", "reprojected_specular",
                          "disocclusion_mask"]
        f.write(struct.pack("<I", len(temporal_names)))
        for name in temporal_names:
            data = temporal_images[name].tobytes()
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(data)))
            f.write(data)

        # Expected output
        output_data = expected_output.contiguous().numpy().astype(np.float32)
        f.write(struct.pack("<I", output_data.nbytes))
        f.write(output_data.tobytes())

    file_size = os.path.getsize(output_path)
    print(f"V3 golden reference written: {output_path} ({file_size:,} bytes)")
    print(f"  Resolution: {WIDTH}x{HEIGHT}, base_channels={base_channels}")
    print(f"  Weights: {len(denimodel_bytes):,} bytes")
    print(f"  Output range: [{expected_output.min():.4f}, {expected_output.max():.4f}]")
    print(f"  Output mean:  {expected_output.mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Generate golden reference for GPU validation")
    parser.add_argument("--output-v3", default=os.path.join(
        os.path.dirname(__file__), "data", "golden_ref_v3.bin"),
        help="Output binary file path (v3 temporal, base_channels=8)")
    parser.add_argument("--output-v3-12", default=os.path.join(
        os.path.dirname(__file__), "data", "golden_ref_v3_12ch.bin"),
        help="Output binary file path (v3 temporal, base_channels=12)")
    args = parser.parse_args()
    write_golden_reference_v3(args.output_v3, base_channels=V3_BASE_CHANNELS)
    write_golden_reference_v3(args.output_v3_12, base_channels=V3_BASE_CHANNELS_12)


if __name__ == "__main__":
    main()
