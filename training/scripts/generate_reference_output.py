"""Generate reference U-Net inference output for validation against GLSL shaders.

Loads a DeniUNet checkpoint and runs inference on input EXR G-buffer slices,
then saves the 3-channel RGB output as an EXR for comparison with GPU results.

Usage:
    python scripts/generate_reference_output.py \
        --checkpoint model_best.pt \
        --diffuse  input_diffuse.exr \
        --specular input_specular.exr \
        --normals  input_normals.exr \
        --depth    input_depth.exr \
        --motion   input_motion.exr \
        --output   reference_output.exr
"""

import argparse
import os
import sys

import numpy as np
import torch

try:
    import OpenEXR
    import Imath

    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

# Add parent directory to path so we can import deni_train
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deni_train.models.unet import DeniUNet


def read_exr(path: str) -> np.ndarray:
    """Read an EXR file and return a float32 HWC array."""
    if not HAS_OPENEXR:
        raise RuntimeError("OpenEXR is required. Install via: pip install OpenEXR")
    exr = OpenEXR.InputFile(path)
    header = exr.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = list(header["channels"].keys())
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    data = {}
    for ch in channels:
        raw = exr.channel(ch, pt)
        data[ch] = np.frombuffer(raw, dtype=np.float32).reshape(height, width)

    # Stack available channels
    if "R" in data:
        arrays = []
        for ch in ["R", "G", "B", "A"]:
            if ch in data:
                arrays.append(data[ch])
        return np.stack(arrays, axis=-1)
    else:
        return np.stack(list(data.values()), axis=-1)


def write_exr(path: str, image: np.ndarray):
    """Write a float32 HWC array to an EXR file."""
    if not HAS_OPENEXR:
        raise RuntimeError("OpenEXR is required. Install via: pip install OpenEXR")
    height, width, channels = image.shape
    header = OpenEXR.Header(width, height)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    channel_names = ["R", "G", "B", "A"][:channels]
    header["channels"] = {ch: Imath.Channel(pt) for ch in channel_names}

    out = OpenEXR.OutputFile(path, header)
    channel_data = {}
    for i, ch in enumerate(channel_names):
        channel_data[ch] = image[:, :, i].astype(np.float32).tobytes()
    out.writePixels(channel_data)
    out.close()


def assemble_input(
    diffuse: np.ndarray,
    specular: np.ndarray,
    normals: np.ndarray,
    depth: np.ndarray,
    motion: np.ndarray,
) -> np.ndarray:
    """Assemble 13-channel input tensor from G-buffer components.

    Channel mapping (matches encoder_input_conv.comp):
      0-2:  noisy diffuse RGB
      3-5:  noisy specular RGB
      6-8:  world normals XYZ
      9:    roughness (normals.w)
      10:   linear depth
      11-12: motion vectors XY
    """
    channels = [
        diffuse[:, :, 0:3],                 # 0-2: diffuse RGB
        specular[:, :, 0:3],                 # 3-5: specular RGB
        normals[:, :, 0:4],                  # 6-9: normals XYZ + roughness W
        depth[:, :, 0:1],                    # 10: linear depth
        motion[:, :, 0:2],                   # 11-12: motion XY
    ]
    return np.concatenate(channels, axis=-1)  # HWC with 13 channels


def main():
    parser = argparse.ArgumentParser(description="Generate reference denoiser output")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--diffuse", required=True, help="Noisy diffuse EXR (RGBA16F)")
    parser.add_argument("--specular", required=True, help="Noisy specular EXR (RGBA16F)")
    parser.add_argument("--normals", required=True, help="World normals + roughness EXR")
    parser.add_argument("--depth", required=True, help="Linear depth EXR")
    parser.add_argument("--motion", required=True, help="Motion vectors EXR")
    parser.add_argument("--output", required=True, help="Output reference EXR (RGB)")
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    model_cfg = ckpt.get("model_config", {})
    if not model_cfg:
        base_ch = state_dict["down0.conv1.conv.weight"].shape[0]
        model_cfg = {"in_channels": 13, "out_channels": 3, "base_channels": base_ch}

    model = DeniUNet(
        in_channels=model_cfg.get("in_channels", 13),
        out_channels=model_cfg.get("out_channels", 3),
        base_channels=model_cfg.get("base_channels", 16),
    )
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: base_channels={model_cfg.get('base_channels', 16)}, params={total_params:,}")

    # Read G-buffer inputs
    print("Reading G-buffer inputs...")
    diffuse = read_exr(args.diffuse)
    specular = read_exr(args.specular)
    normals = read_exr(args.normals)
    depth = read_exr(args.depth)
    motion = read_exr(args.motion)

    height, width = diffuse.shape[:2]
    print(f"Resolution: {width}x{height}")

    # Assemble 13-channel input
    input_hwc = assemble_input(diffuse, specular, normals, depth, motion)
    assert input_hwc.shape == (height, width, 13), f"Expected 13 channels, got {input_hwc.shape}"

    # Convert to NCHW tensor
    input_tensor = torch.from_numpy(input_hwc).permute(2, 0, 1).unsqueeze(0).float()
    print(f"Input tensor shape: {input_tensor.shape}")

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output_tensor = model(input_tensor)

    print(f"Output tensor shape: {output_tensor.shape}")

    # Convert to HWC numpy
    output_hwc = output_tensor.squeeze(0).permute(1, 2, 0).numpy()

    # Write output EXR
    write_exr(args.output, output_hwc)
    print(f"Reference output saved to: {args.output}")

    # Print statistics
    print(f"Output range: [{output_hwc.min():.4f}, {output_hwc.max():.4f}]")
    print(f"Output mean: {output_hwc.mean():.4f}")


if __name__ == "__main__":
    main()
