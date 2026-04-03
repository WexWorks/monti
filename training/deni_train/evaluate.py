"""Evaluation script for DeniUNet denoiser.

CLI: python -m deni_train.evaluate --checkpoint model_best.pt --data_dir ../training_data --output_dir results/
     python -m deni_train.evaluate --checkpoint model_best.pt --data_dir ../training_data --output_dir results/ --val-split --report results/v1_baseline/v1_baseline.md
     python -m deni_train.evaluate --checkpoint model_best.pt --data_dir ../training_data_st --output_dir results/ --data-format safetensors
"""

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image

from .data.exr_dataset import ExrDataset
from .data.safetensors_dataset import SafetensorsDataset
from .data.splits import (
    detect_data_format,
    scene_name_from_file,
    scene_name_from_pair,
    stratified_split,
    stratified_split_files,
)
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


def _get_val_indices(pairs: list[tuple[str, str]]) -> list[int]:
    """Return indices for the held-out validation split (~10% per scene).

    Uses the same stratified split logic as train.py.
    """
    _, val_indices = stratified_split(pairs)
    return val_indices


def _get_val_indices_from_files(files: list[str]) -> list[int]:
    """Return val indices for safetensors files (~10% per scene)."""
    _, val_indices = stratified_split_files(files)
    return val_indices


def _generate_report(
    report_path: str,
    checkpoint_path: str,
    results: list[dict],
    model: DeniUNet,
    val_split: bool,
    total_pairs: int,
    output_dir: str,
):
    """Auto-generate a Markdown quality assessment report."""
    report_dir = os.path.dirname(os.path.abspath(report_path))
    os.makedirs(report_dir, exist_ok=True)

    num_params = sum(p.numel() for p in model.parameters())
    mean_psnr = sum(r["psnr"] for r in results) / len(results)
    mean_ssim = sum(r["ssim"] for r in results) / len(results)
    mean_noisy_psnr = sum(r["noisy_psnr"] for r in results) / len(results)
    mean_noisy_ssim = sum(r["noisy_ssim"] for r in results) / len(results)
    mean_delta_psnr = mean_psnr - mean_noisy_psnr

    # Compute relative path from report to output_dir for image references
    abs_output = os.path.abspath(output_dir)
    try:
        rel_output = os.path.relpath(abs_output, report_dir)
    except ValueError:
        rel_output = abs_output

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    split_label = "per-scene stratified validation split (~10% per scene)" if val_split else "full dataset"
    lines = [
        "# DeniUNet Quality Baseline Report",
        "",
        "## Configuration",
        "",
        f"- **Checkpoint:** `{checkpoint_path}`",
        f"- **Model parameters:** {num_params:,}",
        f"- **Input channels:** {model.down0.conv1.conv.in_channels}",
        f"- **Output channels:** {model.out_conv.out_channels}",
        f"- **Base channels:** {model.down0.conv1.conv.out_channels}",
        f"- **Evaluation set:** {split_label} ({len(results)} of {total_pairs} total pairs)",
        f"- **Timestamp:** {timestamp}",
        "",
        "## Per-Image Metrics (ACES-tonemapped space)",
        "",
        "| Image | Noisy PSNR (dB) | Denoised PSNR (dB) | Delta PSNR (dB) | Noisy SSIM | Denoised SSIM |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for r in results:
        delta = r["psnr"] - r["noisy_psnr"]
        lines.append(
            f"| {r['name']} | {r['noisy_psnr']:.2f} | {r['psnr']:.2f} | "
            f"{delta:+.2f} | {r['noisy_ssim']:.4f} | {r['ssim']:.4f} |"
        )

    lines.extend([
        f"| **Mean** | **{mean_noisy_psnr:.2f}** | **{mean_psnr:.2f}** | "
        f"**{mean_delta_psnr:+.2f}** | **{mean_noisy_ssim:.4f}** | **{mean_ssim:.4f}** |",
        "",
        "## Per-Scene Summary",
        "",
        "| Scene | Frames | Mean Noisy PSNR | Mean Denoised PSNR | Mean Delta PSNR | Mean SSIM |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    # Group by scene for per-scene table
    from collections import defaultdict
    scene_groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        scene_groups[r.get("scene", "unknown")].append(r)

    for scene in sorted(scene_groups.keys()):
        sr = scene_groups[scene]
        n = len(sr)
        s_noisy = sum(r["noisy_psnr"] for r in sr) / n
        s_psnr = sum(r["psnr"] for r in sr) / n
        s_ssim = sum(r["ssim"] for r in sr) / n
        s_delta = s_psnr - s_noisy
        lines.append(
            f"| {scene} | {n} | {s_noisy:.2f} | {s_psnr:.2f} | "
            f"{s_delta:+.2f} | {s_ssim:.4f} |"
        )

    lines.extend([
        "",
        "## Summary",
        "",
        f"- **Mean noisy baseline PSNR:** {mean_noisy_psnr:.2f} dB",
        f"- **Mean denoised PSNR:** {mean_psnr:.2f} dB",
        f"- **Mean delta PSNR:** {mean_delta_psnr:+.2f} dB",
        f"- **Mean noisy baseline SSIM:** {mean_noisy_ssim:.4f}",
        f"- **Mean denoised SSIM:** {mean_ssim:.4f}",
        "",
        "## Comparison Images",
        "",
        "Each image shows: **Noisy** | **Denoised** | **Ground Truth** (ACES tonemapped)",
        "",
    ])

    for r in results:
        png_name = f"{r['name']}_comparison.png"
        png_rel = os.path.join(rel_output, png_name).replace("\\", "/")
        lines.append(f"### {r['name']}")
        lines.append(f"![{r['name']}]({png_rel})")
        lines.append("")

    lines.extend([
        "## TensorBoard Loss Curves",
        "",
        "*(Add screenshot manually: `tensorboard --logdir configs/runs/`)*",
        "",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report saved to {report_path}")


def evaluate(checkpoint_path: str, data_dir: str, output_dir: str,
             val_split: bool = False, report_path: str | None = None,
             data_format: str = "auto", max_per_scene: int | None = None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    device = torch.device("cuda")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Reconstruct model with hyperparameters from checkpoint
    model_cfg = ckpt.get("model_config", {})
    if not model_cfg:
        # Infer base_channels from first conv layer for older checkpoints
        base_ch = state_dict["down0.conv1.conv.weight"].shape[0]
        model_cfg = {"in_channels": 19, "out_channels": 6, "base_channels": base_ch}
    model = DeniUNet(
        in_channels=model_cfg.get("in_channels", 19),
        out_channels=model_cfg.get("out_channels", 6),
        base_channels=model_cfg.get("base_channels", 16),
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Resolve data format
    if data_format == "auto":
        data_format = detect_data_format(data_dir)
    use_safetensors = data_format == "safetensors"
    print(f"Data format: {data_format}")

    # Load dataset (no transforms -- full images)
    if use_safetensors:
        dataset = SafetensorsDataset(data_dir)
        total_count = len(dataset)
        if total_count == 0:
            print(f"No safetensors files found in {data_dir}")
            return
    else:
        dataset = ExrDataset(data_dir)
        total_count = len(dataset)
        if total_count == 0:
            print(f"No EXR pairs found in {data_dir}")
            return

    # Select evaluation indices
    if val_split:
        if use_safetensors:
            eval_indices = _get_val_indices_from_files(dataset.files)
        else:
            eval_indices = _get_val_indices(dataset.pairs)
        print(f"Evaluating validation split: {len(eval_indices)} of {total_count} samples")
    else:
        eval_indices = list(range(total_count))

    # Sub-sample N per scene for quick eval
    if max_per_scene is not None:
        from collections import defaultdict
        scene_buckets: dict[str, list[int]] = defaultdict(list)
        for idx in eval_indices:
            if use_safetensors:
                scene = scene_name_from_file(dataset.files[idx])
            else:
                scene = scene_name_from_pair(dataset.pairs[idx])
            scene_buckets[scene].append(idx)
        sampled: list[int] = []
        for scene in sorted(scene_buckets.keys()):
            indices = scene_buckets[scene]
            if len(indices) <= max_per_scene:
                sampled.extend(indices)
            else:
                # Evenly spaced selection
                step = len(indices) / max_per_scene
                sampled.extend(indices[int(i * step)] for i in range(max_per_scene))
        eval_indices = sampled
        print(f"Quick eval: {len(eval_indices)} samples ({max_per_scene} per scene)")

    os.makedirs(output_dir, exist_ok=True)

    # U-Net has 2 levels of 2x MaxPool => spatial dims must be divisible by 4
    pad_multiple = 4

    results = []
    print(f"Evaluating {len(eval_indices)} images...")
    print(f"{'Image':<40} {'Noisy PSNR':>11} {'PSNR (dB)':>10} {'Delta':>7} {'SSIM':>8}")
    print("-" * 78)

    with torch.no_grad():
        for i in eval_indices:
            inp, tgt, albedo_d, albedo_s = dataset[i]
            inp = inp.to(device, dtype=torch.float32).unsqueeze(0)
            tgt = tgt.to(device, dtype=torch.float32).unsqueeze(0)
            albedo_d = albedo_d.to(device, dtype=torch.float32).unsqueeze(0)
            albedo_s = albedo_s.to(device, dtype=torch.float32).unsqueeze(0)

            # Pad for U-Net compatibility
            inp_padded, (pad_h, pad_w) = _pad_to_multiple(inp, pad_multiple)

            pred_padded = model(inp_padded)

            # Crop back to original size
            _, _, h_orig, w_orig = tgt.shape
            pred = pred_padded[:, :, :h_orig, :w_orig]

            # Remodulate to radiance for metrics
            pred_radiance = pred[:, :3] * albedo_d + pred[:, 3:6] * albedo_s
            tgt_radiance = tgt[:, :3] * albedo_d + tgt[:, 3:6] * albedo_s

            psnr = compute_psnr(pred_radiance, tgt_radiance)
            ssim = compute_ssim(pred_radiance, tgt_radiance)

            # Noisy input: remodulate demodulated irradiance
            noisy_radiance = inp[:, :3] * albedo_d + inp[:, 3:6] * albedo_s
            noisy_psnr = compute_psnr(noisy_radiance, tgt_radiance)
            noisy_ssim = compute_ssim(noisy_radiance, tgt_radiance)
            delta_psnr = psnr - noisy_psnr

            # Save comparison PNG
            if use_safetensors:
                st_path = dataset.files[i]
                st_basename = os.path.basename(st_path)
                name = st_basename[:-len(".safetensors")]
                scene = scene_name_from_file(st_path)
            else:
                basename = os.path.basename(dataset.pairs[i][0])
                if basename == "input.exr":
                    name = os.path.basename(os.path.dirname(dataset.pairs[i][0]))
                else:
                    name = basename[:-len("_input.exr")]
                scene = scene_name_from_pair(dataset.pairs[i])
            png_path = os.path.join(output_dir, f"{name}_comparison.png")
            _save_comparison_png(png_path, noisy_radiance.squeeze(0),
                                pred_radiance.squeeze(0), tgt_radiance.squeeze(0))
            results.append({
                "name": name,
                "scene": scene,
                "psnr": psnr,
                "ssim": ssim,
                "noisy_psnr": noisy_psnr,
                "noisy_ssim": noisy_ssim,
            })
            print(f"{name:<40} {noisy_psnr:>11.2f} {psnr:>10.2f} {delta_psnr:>+7.2f} {ssim:>8.4f}")

    # Aggregate metrics
    mean_psnr = sum(r["psnr"] for r in results) / len(results)
    mean_ssim = sum(r["ssim"] for r in results) / len(results)
    mean_noisy_psnr = sum(r["noisy_psnr"] for r in results) / len(results)
    mean_delta = mean_psnr - mean_noisy_psnr
    print("-" * 78)
    print(f"{'Mean':<40} {mean_noisy_psnr:>11.2f} {mean_psnr:>10.2f} {mean_delta:>+7.2f} {mean_ssim:>8.4f}")
    print(f"\nComparison PNGs saved to {output_dir}")

    # Per-scene summary
    from collections import defaultdict
    scene_results: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        scene_results[r["scene"]].append(r)

    print(f"\nPer-Scene Summary:")
    print(f"{'Scene':<30} {'Frames':>6} {'Noisy PSNR':>11} {'PSNR (dB)':>10} {'Delta':>7} {'SSIM':>8}")
    print("-" * 78)
    for scene in sorted(scene_results.keys()):
        sr = scene_results[scene]
        n = len(sr)
        s_noisy = sum(r["noisy_psnr"] for r in sr) / n
        s_psnr = sum(r["psnr"] for r in sr) / n
        s_ssim = sum(r["ssim"] for r in sr) / n
        s_delta = s_psnr - s_noisy
        print(f"{scene:<30} {n:>6} {s_noisy:>11.2f} {s_psnr:>10.2f} {s_delta:>+7.2f} {s_ssim:>8.4f}")

    # Generate Markdown report
    if report_path:
        _generate_report(report_path, checkpoint_path, results, model, val_split,
                         total_count, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeniUNet denoiser")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", required=True, help="Path to data directory")
    parser.add_argument("--output_dir", default="results/", help="Output directory for PNGs")
    parser.add_argument("--data-format", default="auto", choices=["auto", "exr", "safetensors"],
                        help="Data format: auto-detect (default), exr, or safetensors")
    parser.add_argument("--val-split", action="store_true",
                        help="Evaluate only the held-out validation split (last 10%%)")
    parser.add_argument("--report", default=None, metavar="PATH",
                        help="Path to auto-generate a Markdown quality report")
    parser.add_argument("--max-per-scene", type=int, default=None, metavar="N",
                        help="Evaluate at most N samples per scene (quick eval)")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_dir, args.output_dir,
             val_split=args.val_split, report_path=args.report,
             data_format=args.data_format, max_per_scene=args.max_per_scene)


if __name__ == "__main__":
    main()
