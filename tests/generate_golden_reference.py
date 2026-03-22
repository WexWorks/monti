"""Generate golden reference data for GPU inference validation.

Produces a binary file containing:
  - Weights (.denimodel format)
  - Input G-buffer (13 channels, FP16 packed per image)
  - Expected output (3 channels, FP32)

Both GPU shaders and PyTorch use zero-padding for Conv2d (padding=1).

SYNC REQUIREMENT: This script imports DeniUNet directly from
training/deni_train/models/unet.py. If the model architecture changes
(e.g. adding temporal anti-aliasing, changing channel counts, modifying
layer structure), regenerate the golden reference:

    cd training
    python ../tests/generate_golden_reference.py --output ../tests/data/golden_ref.bin

The C++ GPU tests (ml_inference_numerical_test.cpp) will then fail if the
GLSL compute shaders have not been updated to match the new architecture.
This is intentional -- the golden reference is the contract between the
training model and the GPU shaders.

Usage:
    cd training
    python ../tests/generate_golden_reference.py --output ../tests/data/golden_ref.bin
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

from deni_train.models.unet import DeniUNet

# Test configuration -- small enough for fast CI, large enough for meaningful validation
WIDTH = 32
HEIGHT = 32
BASE_CHANNELS = 8


def make_deterministic_input(width: int, height: int) -> torch.Tensor:
    """Create a deterministic 13-channel input tensor in [0, 1] range."""
    torch.manual_seed(42)
    # Use smooth gradients + small noise for realistic-ish G-buffer data
    x = torch.linspace(0, 1, width).unsqueeze(0).expand(height, width)
    y = torch.linspace(0, 1, height).unsqueeze(1).expand(height, width)

    channels = []
    for ch in range(13):
        # Each channel: smooth gradient + small deterministic variation
        base = (x * (ch + 1) * 0.3 + y * (13 - ch) * 0.2) % 1.0
        noise = torch.randn(height, width) * 0.02
        channels.append((base + noise).clamp(0.0, 1.0))

    return torch.stack(channels).unsqueeze(0)  # [1, 13, H, W]


def quantize_to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to FP16 precision (round-trip through half)."""
    return tensor.half().float()


def pack_gbuffer_fp16(input_13ch: torch.Tensor) -> dict:
    """Pack 13-channel input into G-buffer image arrays (FP16 binary).

    Returns dict with keys matching DenoiserInput image names,
    values are numpy arrays of uint16 (FP16 binary representation).

    Channel mapping:
      noisy_diffuse:  channels 0-2 (RGB), A=1.0  -> RGBA16F
      noisy_specular: channels 3-5 (RGB), A=1.0  -> RGBA16F
      world_normals:  channels 6-8 (XYZ), ch 9 (roughness, .w) -> RGBA16F
      linear_depth:   channel 10                  -> R16F
      motion_vectors: channels 11-12              -> RG16F
    """
    # input_13ch shape: [1, 13, H, W]
    data = input_13ch.squeeze(0)  # [13, H, W]
    h, w = data.shape[1], data.shape[2]

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

    # R16F: [H, W, 1]
    depth = data[10].unsqueeze(-1)

    # RG16F: [H, W, 2]
    motion = torch.stack([data[11], data[12]], dim=-1)

    return {
        "diffuse": to_fp16_bytes(diffuse.contiguous()),
        "specular": to_fp16_bytes(specular.contiguous()),
        "normals": to_fp16_bytes(normals.contiguous()),
        "depth": to_fp16_bytes(depth.contiguous()),
        "motion": to_fp16_bytes(motion.contiguous()),
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


def write_golden_reference(output_path: str):
    """Generate and write golden reference binary file.

    Binary format:
      [4 bytes]   magic: "GREF"
      [4 bytes]   version: 1
      [4 bytes]   width
      [4 bytes]   height
      [4 bytes]   base_channels
      [4 bytes]   denimodel_size (bytes)
      [N bytes]   denimodel data (complete .denimodel binary)
      [4 bytes]   num_gbuffer_images (5)
      For each G-buffer image:
        [4 bytes]   name_length
        [M bytes]   name (UTF-8)
        [4 bytes]   data_size (bytes)
        [D bytes]   FP16 pixel data
      [4 bytes]   output_size (bytes)
      [O bytes]   expected output (FP32, [3][H][W] channel-major)
    """
    torch.manual_seed(42)

    # Create model with small base_channels for fast testing
    model = DeniUNet(in_channels=13, out_channels=3, base_channels=BASE_CHANNELS)
    model.eval()

    # Create deterministic input
    input_tensor = make_deterministic_input(WIDTH, HEIGHT)
    # Quantize to FP16 to match what the GPU will read from G-buffer images
    input_fp16 = quantize_to_fp16(input_tensor)

    # Run PyTorch inference
    with torch.no_grad():
        output = model(input_fp16)  # [1, 3, H, W]

    # Output is what the GPU should produce (in FP32, before FP16 output quantization)
    expected_output = output.squeeze(0)  # [3, H, W]

    # Pack data
    denimodel_bytes = write_denimodel_bytes(model.state_dict())
    gbuffer = pack_gbuffer_fp16(input_fp16)

    # Write binary file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        # Header
        f.write(b"GREF")
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", WIDTH))
        f.write(struct.pack("<I", HEIGHT))
        f.write(struct.pack("<I", BASE_CHANNELS))

        # Denimodel weights
        f.write(struct.pack("<I", len(denimodel_bytes)))
        f.write(denimodel_bytes)

        # G-buffer images
        image_names = ["diffuse", "specular", "normals", "depth", "motion"]
        f.write(struct.pack("<I", len(image_names)))
        for name in image_names:
            data = gbuffer[name].tobytes()
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(data)))
            f.write(data)

        # Expected output: [3, H, W] as contiguous FP32
        output_data = expected_output.contiguous().numpy().astype(np.float32)
        f.write(struct.pack("<I", output_data.nbytes))
        f.write(output_data.tobytes())

    file_size = os.path.getsize(output_path)
    print(f"Golden reference written: {output_path} ({file_size:,} bytes)")
    print(f"  Resolution: {WIDTH}x{HEIGHT}, base_channels={BASE_CHANNELS}")
    print(f"  Weights: {len(denimodel_bytes):,} bytes")
    print(f"  Output range: [{expected_output.min():.4f}, {expected_output.max():.4f}]")
    print(f"  Output mean:  {expected_output.mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Generate golden reference for GPU validation")
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__), "data", "golden_ref.bin"),
        help="Output binary file path")
    args = parser.parse_args()
    write_golden_reference(args.output)


if __name__ == "__main__":
    main()
