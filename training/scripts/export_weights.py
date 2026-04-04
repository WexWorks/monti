"""Export trained DeniUNet / DeniTemporalResidualNet weights to .denimodel binary and ONNX formats.

CLI: python scripts/export_weights.py --checkpoint model_best.pt --output models/deni_v1.denimodel

.denimodel binary format:
    Header:
      [4 bytes]  magic: "DENI"
      [4 bytes]  version: 1
      [4 bytes]  num_layers
      [4 bytes]  total_weight_bytes
    Per layer (repeated num_layers times):
      [4 bytes]  name_length
      [N bytes]  name (UTF-8, not null-terminated)
      [4 bytes]  num_dims
      [4 bytes x num_dims]  shape
      [4 bytes x product(shape)]  float32 weight data (little-endian)
"""

import argparse
import os
import struct
import sys

import numpy as np
import torch

# Add parent directory to path so we can import deni_train
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deni_train.models.unet import DeniUNet
from deni_train.models.temporal_unet import DeniTemporalResidualNet


def write_denimodel(state_dict: dict[str, torch.Tensor], output_path: str):
    """Write model weights to .denimodel binary format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    layers = []
    total_weight_bytes = 0
    for name, tensor in state_dict.items():
        data = tensor.detach().cpu().float().numpy()
        layers.append((name, data))
        total_weight_bytes += data.nbytes

    with open(output_path, "wb") as f:
        # Header
        f.write(b"DENI")                                  # magic
        f.write(struct.pack("<I", 1))                     # version
        f.write(struct.pack("<I", len(layers)))           # num_layers
        f.write(struct.pack("<I", total_weight_bytes))    # total_weight_bytes

        # Per-layer data
        for name, data in layers:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))   # name_length
            f.write(name_bytes)                           # name
            f.write(struct.pack("<I", data.ndim))         # num_dims
            for dim in data.shape:
                f.write(struct.pack("<I", dim))           # shape dims
            f.write(data.astype(np.float32).tobytes())    # weight data (little-endian)


def export_onnx(model: torch.nn.Module, in_channels: int, output_path: str):
    """Export model to ONNX format with dynamic axes."""
    model.eval()
    dummy = torch.randn(1, in_channels, 256, 256)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
    )


def install_model(output_path: str):
    """Copy exported model to denoise/models/ for CMake pickup."""
    import shutil
    # Resolve denoise/models/ relative to the monti project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, "..", ".."))
    install_dir = os.path.join(project_root, "denoise", "models")
    os.makedirs(install_dir, exist_ok=True)
    dest = os.path.join(install_dir, os.path.basename(output_path))
    shutil.copy2(output_path, dest)
    print(f"Installed to denoiser library: {dest}")


def main():
    parser = argparse.ArgumentParser(description="Export DeniUNet weights")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", required=True, help="Output .denimodel path")
    parser.add_argument("--install", action="store_true",
                        help="Copy exported model to denoise/models/ for CMake pickup")
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Build model with hyperparameters from checkpoint
    model_cfg = ckpt.get("model_config")
    if not model_cfg:
        raise RuntimeError("Checkpoint missing 'model_config'; cannot determine model architecture")

    model_type = model_cfg.get("type", "DeniUNet")
    if model_type in ("DeniTemporalResidualNet", "temporal_residual"):
        model = DeniTemporalResidualNet(
            base_channels=model_cfg["base_channels"],
        )
        in_channels = 26  # fixed for temporal model
    else:
        model = DeniUNet(
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            base_channels=model_cfg["base_channels"],
        )
        in_channels = model_cfg["in_channels"]
    model.load_state_dict(state_dict)

    # Print layer summary
    print(f"{'Layer':<50} {'Shape':<25} {'Params':>10}")
    print("-" * 87)
    total_params = 0
    for name, tensor in state_dict.items():
        n = tensor.numel()
        total_params += n
        shape_str = "x".join(str(d) for d in tensor.shape)
        print(f"{name:<50} {shape_str:<25} {n:>10,}")
    print("-" * 87)
    print(f"{'Total':<50} {'':<25} {total_params:>10,}")

    # Write .denimodel
    write_denimodel(state_dict, args.output)
    file_size = os.path.getsize(args.output)
    print(f"\nExported .denimodel: {args.output} ({file_size:,} bytes)")

    # Write ONNX
    onnx_path = args.output.replace(".denimodel", ".onnx")
    export_onnx(model, in_channels, onnx_path)
    onnx_size = os.path.getsize(onnx_path)
    print(f"Exported ONNX: {onnx_path} ({onnx_size:,} bytes)")

    if args.install:
        install_model(args.output)


if __name__ == "__main__":
    main()
