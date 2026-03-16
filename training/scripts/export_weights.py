"""Export trained DeniUNet weights to .denimodel binary and ONNX formats.

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


def export_onnx(model: torch.nn.Module, output_path: str):
    """Export model to ONNX format with dynamic axes."""
    model.eval()
    dummy = torch.randn(1, 13, 256, 256)
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


def main():
    parser = argparse.ArgumentParser(description="Export DeniUNet weights")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", required=True, help="Output .denimodel path")
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Build model and load weights
    model = DeniUNet()
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
    export_onnx(model, onnx_path)
    onnx_size = os.path.getsize(onnx_path)
    print(f"Exported ONNX: {onnx_path} ({onnx_size:,} bytes)")


if __name__ == "__main__":
    main()
