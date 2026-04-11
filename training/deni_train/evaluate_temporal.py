"""Evaluation script for DeniTemporalResidualNet denoiser.

CLI:
    python -m deni_train.evaluate_temporal \\
        --checkpoint configs/checkpoints/model_best.pt \\
        --data_dir D:/training_data_temporal_st \\
        --output_dir results/temporal/

    python -m deni_train.evaluate_temporal \\
        --checkpoint configs/checkpoints/model_best.pt \\
        --data_dir D:/training_data_temporal_st \\
        --output_dir results/temporal/ \\
        --val-split --max-per-scene 3 \\
        --report results/temporal/report.md
"""

import argparse
import os
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data.splits import scene_name_from_file, stratified_split_files
from .data.temporal_safetensors_dataset import TemporalSafetensorsDataset
from .models.temporal_unet import DeniTemporalResidualNet
from .utils.metrics import compute_psnr, compute_ssim
from .utils.reproject import reproject
from .utils.tonemapping import aces_tonemap

# G-buffer channel layout within the 19-channel input (same as static model):
_CH_DEPTH = slice(10, 11)
_CH_MOTION = slice(11, 13)
_CH_ALBEDO_D = slice(13, 16)
_CH_ALBEDO_S = slice(16, 19)


def _build_temporal_input(
    gbuffer: torch.Tensor,
    prev_denoised: torch.Tensor | None,
    prev_depth: torch.Tensor | None,
) -> torch.Tensor:
    """Assemble 26-channel temporal input from G-buffer and reprojected history."""
    B, _, H, W = gbuffer.shape
    curr_depth = gbuffer[:, _CH_DEPTH]
    motion = gbuffer[:, _CH_MOTION]

    if prev_denoised is None:
        reprojected_d = torch.zeros(B, 3, H, W, device=gbuffer.device, dtype=gbuffer.dtype)
        reprojected_s = torch.zeros(B, 3, H, W, device=gbuffer.device, dtype=gbuffer.dtype)
        disocclusion = torch.zeros(B, 1, H, W, device=gbuffer.device, dtype=gbuffer.dtype)
    else:
        prev_d = prev_denoised[:, :3]
        prev_s = prev_denoised[:, 3:6]
        reprojected_d, valid_mask = reproject(prev_d, motion, prev_depth, curr_depth)
        reprojected_s, _ = reproject(prev_s, motion, prev_depth, curr_depth)
        disocclusion = valid_mask

    return torch.cat([reprojected_d, reprojected_s, disocclusion, gbuffer], dim=1)


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad spatial dims to next multiple. Returns padded tensor and (pad_h, pad_w)."""
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
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


def _generate_report(
    report_path: str,
    checkpoint_path: str,
    results: list[dict],
    model: DeniTemporalResidualNet,
    val_split: bool,
    total_sequences: int,
    output_dir: str,
):
    """Auto-generate a Markdown quality assessment report."""
    report_dir = os.path.dirname(os.path.abspath(report_path))
    os.makedirs(report_dir, exist_ok=True)

    num_params = sum(p.numel() for p in model.parameters())
    mean_psnr = sum(r["psnr"] for r in results) / len(results)
    mean_ssim = sum(r["ssim"] for r in results) / len(results)
    mean_noisy_psnr = sum(r["noisy_psnr"] for r in results) / len(results)
    mean_delta_psnr = mean_psnr - mean_noisy_psnr

    abs_output = os.path.abspath(output_dir)
    try:
        rel_output = os.path.relpath(abs_output, report_dir)
    except ValueError:
        rel_output = abs_output

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    split_label = "per-scene stratified validation split (~10% per scene)" if val_split else "full dataset"

    lines = [
        "# DeniTemporalResidualNet Quality Report",
        "",
        "## Configuration",
        "",
        f"- **Checkpoint:** `{checkpoint_path}`",
        f"- **Model parameters:** {num_params:,}",
        f"- **Evaluation set:** {split_label} ({len(results)} frames from {total_sequences} sequences)",
        f"- **Timestamp:** {timestamp}",
        "",
        "## Per-Frame Metrics (ACES-tonemapped space)",
        "",
        "| Frame | Noisy PSNR (dB) | Denoised PSNR (dB) | Delta PSNR (dB) | Noisy SSIM | Denoised SSIM |",
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
        f"**{mean_delta_psnr:+.2f}** | **{sum(r['noisy_ssim'] for r in results)/len(results):.4f}** | "
        f"**{mean_ssim:.4f}** |",
        "",
        "## Per-Scene Summary",
        "",
        "| Scene | Frames | Mean Noisy PSNR | Mean Denoised PSNR | Mean Delta PSNR | Mean SSIM |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    scene_groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        scene_groups[r["scene"]].append(r)

    for scene in sorted(scene_groups.keys()):
        sr = scene_groups[scene]
        n = len(sr)
        s_noisy = sum(r["noisy_psnr"] for r in sr) / n
        s_psnr = sum(r["psnr"] for r in sr) / n
        s_ssim = sum(r["ssim"] for r in sr) / n
        s_delta = s_psnr - s_noisy
        lines.append(
            f"| {scene} | {n} | {s_noisy:.2f} | {s_psnr:.2f} | {s_delta:+.2f} | {s_ssim:.4f} |"
        )

    lines.extend([
        "",
        "## Summary",
        "",
        f"- **Mean noisy baseline PSNR:** {mean_noisy_psnr:.2f} dB",
        f"- **Mean denoised PSNR:** {mean_psnr:.2f} dB",
        f"- **Mean delta PSNR:** {mean_delta_psnr:+.2f} dB",
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

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report saved to {report_path}")


def evaluate(checkpoint_path: str, data_dir: str, output_dir: str,
             val_split: bool = False, report_path: str | None = None,
             max_per_scene: int | None = None, warmup_frames: int = 0,
             eval_frame: int | None = None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    device = torch.device("cuda")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    model_cfg = ckpt.get("model_config", {})
    base_channels = model_cfg.get("base_channels", 12)

    model = DeniTemporalResidualNet(base_channels=base_channels)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    dataset = TemporalSafetensorsDataset(data_dir)
    total_count = len(dataset)
    if total_count == 0:
        print(f"No temporal safetensors files found in {data_dir}")
        return

    if val_split:
        _, eval_indices = stratified_split_files(dataset.files)
        print(f"Evaluating validation split: {len(eval_indices)} of {total_count} sequences")
    else:
        eval_indices = list(range(total_count))

    if max_per_scene is not None:
        scene_buckets: dict[str, list[int]] = defaultdict(list)
        for idx in eval_indices:
            scene = scene_name_from_file(dataset.files[idx])
            scene_buckets[scene].append(idx)
        sampled: list[int] = []
        for scene in sorted(scene_buckets.keys()):
            indices = scene_buckets[scene]
            if len(indices) <= max_per_scene:
                sampled.extend(indices)
            else:
                step = len(indices) / max_per_scene
                sampled.extend(indices[int(i * step)] for i in range(max_per_scene))
        eval_indices = sampled
        print(f"Quick eval: {len(eval_indices)} sequences ({max_per_scene} per scene)")

    os.makedirs(output_dir, exist_ok=True)

    # 2-level U-Net requires spatial dims divisible by 4
    pad_multiple = 4

    results = []
    if eval_frame is not None:
        print(f"Evaluating {len(eval_indices)} sequences (eval_frame={eval_frame}, 1 frame per sequence)...")
    else:
        print(f"Evaluating {len(eval_indices)} sequences (warmup_frames={warmup_frames})...")
    print(f"{'Frame':<50} {'Noisy PSNR':>11} {'PSNR (dB)':>10} {'Delta':>7} {'SSIM':>8}")
    print("-" * 88)

    with torch.no_grad():
        for seq_idx in eval_indices:
            sample = dataset[seq_idx]
            # inp_seq: (W, 19, H, Wspatial), tgt_seq: (W, 6, H, Wspatial)
            inp_seq = sample["input"].to(device, dtype=torch.float32)
            tgt_seq = sample["target"].to(device, dtype=torch.float32)
            num_frames = inp_seq.shape[0]

            seq_basename = os.path.splitext(os.path.basename(dataset.files[seq_idx]))[0]
            scene = scene_name_from_file(dataset.files[seq_idx])

            prev_denoised = None
            prev_depth = None

            for t in range(num_frames):
                gbuffer = inp_seq[t].unsqueeze(0)  # (1, 19, H, W)
                target = tgt_seq[t].unsqueeze(0)   # (1, 6, H, W)

                temporal_input = _build_temporal_input(gbuffer, prev_denoised, prev_depth)
                temporal_input_padded, (pad_h, pad_w) = _pad_to_multiple(temporal_input, pad_multiple)

                pred_padded, _ = model(temporal_input_padded)

                _, _, h_orig, w_orig = gbuffer.shape
                pred = pred_padded[:, :, :h_orig, :w_orig]  # (1, 6, H, W)

                albedo_d = gbuffer[:, _CH_ALBEDO_D]  # (1, 3, H, W)
                albedo_s = gbuffer[:, _CH_ALBEDO_S]  # (1, 3, H, W)

                pred_radiance = pred[:, :3] * albedo_d + pred[:, 3:6] * albedo_s
                tgt_radiance = target[:, :3] * albedo_d + target[:, 3:6] * albedo_s
                noisy_radiance = gbuffer[:, :3] * albedo_d + gbuffer[:, 3:6] * albedo_s

                prev_denoised = pred.detach()
                prev_depth = gbuffer[:, _CH_DEPTH].detach()

                # Skip frames outside the desired evaluation window.
                # --eval-frame F: output only frame F, use 0..F-1 as implicit warmup.
                # --warmup-frames N: skip first N frames, output N onward.
                if eval_frame is not None:
                    if t != eval_frame:
                        continue
                elif t < warmup_frames:
                    continue

                psnr = compute_psnr(pred_radiance, tgt_radiance)
                ssim = compute_ssim(pred_radiance, tgt_radiance)
                noisy_psnr = compute_psnr(noisy_radiance, tgt_radiance)
                noisy_ssim = compute_ssim(noisy_radiance, tgt_radiance)
                delta_psnr = psnr - noisy_psnr

                frame_name = f"{seq_basename}_f{t:02d}"
                png_path = os.path.join(output_dir, f"{frame_name}_comparison.png")
                _save_comparison_png(png_path, noisy_radiance.squeeze(0),
                                     pred_radiance.squeeze(0), tgt_radiance.squeeze(0))

                results.append({
                    "name": frame_name,
                    "scene": scene,
                    "psnr": psnr,
                    "ssim": ssim,
                    "noisy_psnr": noisy_psnr,
                    "noisy_ssim": noisy_ssim,
                })
                print(f"{frame_name:<50} {noisy_psnr:>11.2f} {psnr:>10.2f} {delta_psnr:>+7.2f} {ssim:>8.4f}")

    mean_psnr = sum(r["psnr"] for r in results) / len(results)
    mean_ssim = sum(r["ssim"] for r in results) / len(results)
    mean_noisy_psnr = sum(r["noisy_psnr"] for r in results) / len(results)
    mean_delta = mean_psnr - mean_noisy_psnr
    print("-" * 88)
    print(f"{'Mean':<50} {mean_noisy_psnr:>11.2f} {mean_psnr:>10.2f} {mean_delta:>+7.2f} {mean_ssim:>8.4f}")
    print(f"\nComparison PNGs saved to {output_dir}")

    scene_results: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        scene_results[r["scene"]].append(r)

    print(f"\nPer-Scene Summary:")
    print(f"{'Scene':<30} {'Frames':>6} {'Noisy PSNR':>11} {'PSNR (dB)':>10} {'Delta':>7} {'SSIM':>8}")
    print("-" * 74)
    for scene in sorted(scene_results.keys()):
        sr = scene_results[scene]
        n = len(sr)
        s_noisy = sum(r["noisy_psnr"] for r in sr) / n
        s_psnr = sum(r["psnr"] for r in sr) / n
        s_ssim = sum(r["ssim"] for r in sr) / n
        s_delta = s_psnr - s_noisy
        print(f"{scene:<30} {n:>6} {s_noisy:>11.2f} {s_psnr:>10.2f} {s_delta:>+7.2f} {s_ssim:>8.4f}")

    if report_path is not None:
        _generate_report(report_path, checkpoint_path, results, model,
                         val_split, len(eval_indices), output_dir)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeniTemporalResidualNet")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data_dir", required=True, help="Temporal safetensors directory")
    parser.add_argument("--output_dir", required=True, help="Directory for comparison PNGs")
    parser.add_argument("--val-split", action="store_true",
                        help="Evaluate only the held-out validation split (~10%% per scene)")
    parser.add_argument("--max-per-scene", type=int, default=None, metavar="N",
                        help="Evaluate at most N sequences per scene (quick eval)")
    parser.add_argument("--report", default=None, metavar="PATH",
                        help="Write a Markdown quality report to this path")
    parser.add_argument("--warmup-frames", type=int, default=0, metavar="N",
                        help="Run the first N frames autoregressively to build history "
                             "but exclude them from metrics and PNGs (default: 0). "
                             "Use 7 to output only the last frame of each 8-frame sequence.")
    parser.add_argument("--eval-frame", type=int, default=None, metavar="F",
                        help="Output only frame F (0-indexed) from each sequence; "
                             "frames 0..F-1 are run autoregressively as implicit warmup. "
                             "Mutually exclusive with --warmup-frames.")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        val_split=args.val_split,
        report_path=args.report,
        max_per_scene=args.max_per_scene,
        warmup_frames=args.warmup_frames,
        eval_frame=args.eval_frame,
    )


if __name__ == "__main__":
    main()
