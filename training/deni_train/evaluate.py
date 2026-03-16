"""Evaluation script for DeniUNet denoiser.

CLI: python -m deni_train.evaluate --checkpoint model_best.pt --data_dir ../training_data --output_dir results/
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from .data.exr_dataset import ExrDataset
from .models.unet import DeniUNet
from .utils.metrics import compute_psnr, compute_ssim
from .utils.tonemapping import aces_tonemap


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad spatial dims to next multiple. Returns padded tensor and (pad_h, pad_w)."""
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    # F.pad order: (left, right, top, bottom)
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


def _save_comparison_png(path: str, noisy: torch.Tensor, denoised: torch.Tensor,
                         target: torch.Tensor):
    """Save side-by-side comparison PNG: noisy | denoised | target (tonemapped)."""
    with torch.no_grad():
        noisy_tm = aces_tonemap(noisy.unsqueeze(0)).squeeze(0).clamp(0.0, 1.0)
        den_tm = aces_tonemap(denoised.unsqueeze(0)).squeeze(0).clamp(0.0, 1.0)
        tgt_tm = aces_tonemap(target.unsqueeze(0)).squeeze(0).clamp(0.0, 1.0)

        triplet = torch.cat([noisy_tm, den_tm, tgt_tm], dim=2)  # (3, H, W*3)
        arr = (triplet.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(path)


def evaluate(checkpoint_path: str, data_dir: str, output_dir: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    device = torch.device("cuda")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Infer model config from checkpoint keys (use default if not stored)
    model = DeniUNet()
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load dataset (no transforms -- full images)
    dataset = ExrDataset(data_dir)
    if len(dataset) == 0:
        print(f"No EXR pairs found in {data_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # U-Net has 2 levels of 2x MaxPool => spatial dims must be divisible by 4
    pad_multiple = 4

    results = []
    print(f"Evaluating {len(dataset)} images...")
    print(f"{'Image':<40} {'PSNR (dB)':>10} {'SSIM':>8}")
    print("-" * 60)

    with torch.no_grad():
        for i in range(len(dataset)):
            inp, tgt = dataset[i]
            inp = inp.to(device, dtype=torch.float32).unsqueeze(0)
            tgt = tgt.to(device, dtype=torch.float32).unsqueeze(0)

            # Pad for U-Net compatibility
            inp_padded, (pad_h, pad_w) = _pad_to_multiple(inp, pad_multiple)

            pred_padded = model(inp_padded)

            # Crop back to original size
            _, _, h_orig, w_orig = tgt.shape
            pred = pred_padded[:, :, :h_orig, :w_orig]

            psnr = compute_psnr(pred, tgt)
            ssim = compute_ssim(pred, tgt)

            # Noisy input visualization: diffuse + specular
            noisy_rgb = inp[0, :3] + inp[0, 3:6]

            # Save comparison PNG
            name = os.path.basename(dataset.pairs[i][0]).replace("_input.exr", "")
            png_path = os.path.join(output_dir, f"{name}_comparison.png")
            _save_comparison_png(png_path, noisy_rgb, pred.squeeze(0), tgt.squeeze(0))

            results.append({"name": name, "psnr": psnr, "ssim": ssim})
            print(f"{name:<40} {psnr:>10.2f} {ssim:>8.4f}")

    # Aggregate metrics
    mean_psnr = sum(r["psnr"] for r in results) / len(results)
    mean_ssim = sum(r["ssim"] for r in results) / len(results)
    print("-" * 60)
    print(f"{'Mean':<40} {mean_psnr:>10.2f} {mean_ssim:>8.4f}")
    print(f"\nComparison PNGs saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeniUNet denoiser")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", required=True, help="Path to EXR data directory")
    parser.add_argument("--output_dir", default="results/", help="Output directory for PNGs")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
