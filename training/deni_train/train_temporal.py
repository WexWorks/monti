"""Temporal residual denoiser training script.

CLI: python -m deni_train.train_temporal --config configs/temporal.yaml [--resume ckpt.pt] [--weights-only]

Key differences from single-frame train.py:
- Autoregressive frame processing: frame 0 uses zero history, subsequent frames use
  reprojected model output from the previous frame.
- Temporal stability loss: penalizes flicker between consecutive outputs.
- TemporalSafetensorsDataset: loads pre-cropped 8-frame sequences.
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from .data.splits import stratified_split_files
from .data.temporal_safetensors_dataset import TemporalSafetensorsDataset
from .losses.denoiser_loss import DenoiserLoss
from .models.temporal_unet import DeniTemporalResidualNet
from .utils.reproject import reproject
from .utils.tonemapping import aces_tonemap

# G-buffer channel layout (19ch per frame, same as static model):
#   [0-2]   demod diffuse irradiance
#   [3-5]   demod specular irradiance
#   [6-8]   world normals XYZ
#   [9]     roughness
#   [10]    linear depth
#   [11-12] motion vectors XY
#   [13-15] diffuse albedo
#   [16-18] specular albedo

_CH_DIFFUSE = slice(0, 3)
_CH_SPECULAR = slice(3, 6)
_CH_DEPTH = slice(10, 11)
_CH_MOTION = slice(11, 13)
_CH_ALBEDO_D = slice(13, 16)
_CH_ALBEDO_S = slice(16, 19)

# Temporal input is 26ch: 7 temporal + 19 G-buffer
_TEMPORAL_CHANNELS = 7


class _Config:
    """Simple namespace wrapper for YAML config with dot access."""

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, _Config(v) if isinstance(v, dict) else v)


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_dataloaders(cfg: _Config):
    dataset = TemporalSafetensorsDataset(cfg.data.data_dir)
    n = len(dataset)
    if n == 0:
        raise RuntimeError(f"No temporal safetensors found in {cfg.data.data_dir}")

    train_indices, val_indices = stratified_split_files(dataset.files)
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=getattr(cfg.data, "num_workers", 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=getattr(cfg.data, "num_workers", 4) > 0,
        prefetch_factor=4 if getattr(cfg.data, "num_workers", 4) > 0 else None,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader


def _build_scheduler(optimizer, cfg: _Config, steps_per_epoch: int):
    """Cosine annealing with linear warmup."""
    warmup_steps = cfg.training.warmup_epochs * steps_per_epoch
    total_steps = cfg.training.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _build_temporal_input(
    gbuffer: torch.Tensor,
    prev_denoised: torch.Tensor | None,
    prev_depth: torch.Tensor | None,
) -> torch.Tensor:
    """Assemble 26-channel temporal model input from G-buffer and reprojected history.

    Args:
        gbuffer: (B, 19, H, W) current frame G-buffer.
        prev_denoised: (B, 6, H, W) previous frame denoised output, or None for first frame.
        prev_depth: (B, 1, H, W) previous frame depth, or None for first frame.

    Returns:
        (B, 26, H, W) temporal model input.
    """
    B, _, H, W = gbuffer.shape
    curr_depth = gbuffer[:, _CH_DEPTH]    # (B, 1, H, W)
    motion = gbuffer[:, _CH_MOTION]       # (B, 2, H, W)

    if prev_denoised is None:
        # First frame: zero reprojected, fully disoccluded
        reprojected_d = torch.zeros(B, 3, H, W, device=gbuffer.device, dtype=gbuffer.dtype)
        reprojected_s = torch.zeros(B, 3, H, W, device=gbuffer.device, dtype=gbuffer.dtype)
        disocclusion = torch.zeros(B, 1, H, W, device=gbuffer.device, dtype=gbuffer.dtype)
    else:
        prev_d = prev_denoised[:, :3]  # (B, 3, H, W)
        prev_s = prev_denoised[:, 3:6]  # (B, 3, H, W)

        reprojected_d, valid_mask = reproject(prev_d, motion, prev_depth, curr_depth)
        reprojected_s, _ = reproject(prev_s, motion, prev_depth, curr_depth)
        disocclusion = valid_mask  # 1=valid, 0=disoccluded

    # Concatenate: [reprojected_d(3), reprojected_s(3), disocclusion(1), gbuffer(19)]
    return torch.cat([reprojected_d, reprojected_s, disocclusion, gbuffer], dim=1)


def _process_sequence(
    model: nn.Module,
    inp_seq: torch.Tensor,
    tgt_seq: torch.Tensor,
    loss_fn: DenoiserLoss,
    lambda_temporal: float,
    lambda_blend_weight: float,
    blend_weight_threshold: float,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Process an 8-frame sequence autoregressively and compute combined loss.

    Args:
        model: DeniTemporalResidualNet.
        inp_seq: (B, W, 19, H, W_spatial) input sequence.
        tgt_seq: (B, W, 6, H, W_spatial) target sequence.
        loss_fn: DenoiserLoss (L1 + perceptual).
        lambda_temporal: Weight for temporal stability loss.
        lambda_blend_weight: Weight for supervised blend weight loss.
        blend_weight_threshold: Reprojection MAE above this (in tonemapped space) → force w=1.
        amp_dtype: Mixed precision dtype.

    Returns:
        total_loss: Scalar loss for backprop.
        outputs: List of per-frame denoised outputs for logging.
    """
    B, W, C_in, H, W_spatial = inp_seq.shape
    outputs = []
    frame_losses = []
    prev_denoised = None
    prev_depth = None

    for t in range(W):
        gbuffer = inp_seq[:, t]  # (B, 19, H, W_spatial)
        target = tgt_seq[:, t]   # (B, 6, H, W_spatial)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            temporal_input = _build_temporal_input(gbuffer, prev_denoised, prev_depth)
            denoised, predicted_weight = model(temporal_input)

        # Per-frame reconstruction loss
        albedo_d = gbuffer[:, _CH_ALBEDO_D]
        albedo_s = gbuffer[:, _CH_ALBEDO_S]

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            frame_loss = loss_fn(denoised, target, albedo_d, albedo_s)
        frame_losses.append(frame_loss)

        # Supervised blend weight loss: teach the network to fire w=1 wherever
        # reprojection was inaccurate, regardless of whether depth-based
        # disocclusion or velocity priors fired.
        if t > 0 and lambda_blend_weight > 0:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                reprojected_d = temporal_input[:, 0:3]
                reprojected_s = temporal_input[:, 3:6]
                repro_err_d = torch.abs(
                    aces_tonemap(reprojected_d) - aces_tonemap(target[:, :3])
                ).mean(dim=1, keepdim=True).detach()
                repro_err_s = torch.abs(
                    aces_tonemap(reprojected_s) - aces_tonemap(target[:, 3:6])
                ).mean(dim=1, keepdim=True).detach()
                gt_weight = torch.max(
                    (repro_err_d > blend_weight_threshold).float(),
                    (repro_err_s > blend_weight_threshold).float(),
                )
                blend_loss = F.binary_cross_entropy(predicted_weight, gt_weight)
                frame_losses.append(lambda_blend_weight * blend_loss)

        # Temporal stability loss: compare warped previous output with current output
        if t > 0 and lambda_temporal > 0:
            motion = gbuffer[:, _CH_MOTION]  # (B, 2, H, W_spatial)
            # Warp previous output into current frame's space using current frame's MVs
            # (current MVs point from current back to previous, exactly what reproject needs)
            warped_prev, warp_valid = reproject(
                outputs[-1].detach(), motion,
                prev_depth, gbuffer[:, _CH_DEPTH],
            )
            # Only penalize in valid (non-disoccluded) regions
            valid_count = warp_valid.sum().clamp(min=1.0)
            temporal_loss = (torch.abs(denoised - warped_prev) * warp_valid).sum() / valid_count
            frame_losses.append(lambda_temporal * temporal_loss)

        outputs.append(denoised)
        # Store for next frame's reprojection (detach to avoid backprop through
        # the full sequence — each frame only gets gradients from its own loss
        # plus the temporal loss connection to the next frame)
        prev_denoised = denoised.detach()
        prev_depth = gbuffer[:, _CH_DEPTH].detach()

    total_loss = sum(frame_losses) / W
    return total_loss, outputs


def _log_sample_images(writer, tag: str, gbuffer: torch.Tensor, pred: torch.Tensor,
                       tgt: torch.Tensor, global_step: int):
    """Log a triplet of tonemapped images to TensorBoard."""
    with torch.no_grad():
        alb_d = gbuffer[0, _CH_ALBEDO_D]  # (3, H, W)
        alb_s = gbuffer[0, _CH_ALBEDO_S]  # (3, H, W)

        inp_rgb = gbuffer[0, _CH_DIFFUSE] * alb_d + gbuffer[0, _CH_SPECULAR] * alb_s
        inp_tm = aces_tonemap(inp_rgb.unsqueeze(0)).squeeze(0).clamp(0.0, 1.0)

        pred_rgb = pred[0, :3] * alb_d + pred[0, 3:6] * alb_s
        pred_tm = aces_tonemap(pred_rgb.unsqueeze(0)).squeeze(0).clamp(0.0, 1.0)

        tgt_rgb = tgt[0, :3] * alb_d + tgt[0, 3:6] * alb_s
        tgt_tm = aces_tonemap(tgt_rgb.unsqueeze(0)).squeeze(0).clamp(0.0, 1.0)

        triplet = torch.cat([inp_tm, pred_tm, tgt_tm], dim=2)
        writer.add_image(tag, triplet, global_step)


def _validate(model, val_loader, loss_fn, lambda_temporal, lambda_blend_weight,
              blend_weight_threshold, device, amp_dtype):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            inp_seq = batch["input"].to(device, dtype=torch.float32)
            tgt_seq = batch["target"].to(device, dtype=torch.float32)
            loss, _ = _process_sequence(model, inp_seq, tgt_seq, loss_fn,
                                        lambda_temporal, lambda_blend_weight,
                                        blend_weight_threshold, amp_dtype)
            total_loss += loss.item() * inp_seq.size(0)
            count += inp_seq.size(0)
    model.train()
    if count == 0:
        return 0.0
    return total_loss / count


def train(config_path: str, resume_path: str | None = None, weights_only: bool = False):
    with open(config_path, "r") as f:
        cfg = _Config(yaml.safe_load(f))

    config_dir = os.path.dirname(os.path.abspath(config_path))
    if not os.path.isabs(cfg.data.data_dir):
        cfg.data.data_dir = os.path.normpath(os.path.join(config_dir, cfg.data.data_dir))

    _seed_everything(cfg.training.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training (mixed precision, GradScaler)")
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if cfg.training.mixed_precision else torch.float32

    # Data
    train_loader, val_loader = _build_dataloaders(cfg)
    print(f"Training samples: {len(train_loader.dataset)}, "
          f"Validation samples: {len(val_loader.dataset)}")

    # Model
    max_mv = getattr(cfg.model, "max_mv_for_weight", 0.05)
    model = DeniTemporalResidualNet(
        base_channels=cfg.model.base_channels,
        max_mv_for_weight=max_mv,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Loss
    lambda_temporal = getattr(cfg.loss, "lambda_temporal", 0.5)
    lambda_blend_weight = getattr(cfg.loss, "lambda_blend_weight", 0.0)
    blend_weight_threshold = getattr(cfg.loss, "blend_weight_threshold", 0.05)
    loss_fn = DenoiserLoss(
        lambda_l1=cfg.loss.lambda_l1,
        lambda_perceptual=cfg.loss.lambda_perceptual,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    steps_per_epoch = max(1, len(train_loader))
    scheduler = _build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.mixed_precision)

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0
    patience = getattr(cfg.training, "patience", 0)
    epochs_without_improvement = 0

    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if not weights_only:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            global_step = ckpt["global_step"]
            print(f"Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
        else:
            print(f"Loaded weights only (warm restart)")

    # TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    log_dir = os.path.join(config_dir, "runs",
                           os.path.splitext(os.path.basename(config_path))[0])
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard log dir: {log_dir}")

    # Checkpoints
    ckpt_dir = os.path.join(config_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    model.train()
    for epoch in range(start_epoch, cfg.training.epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        t0 = time.time()

        for batch in train_loader:
            inp_seq = batch["input"].to(device, dtype=torch.float32)
            tgt_seq = batch["target"].to(device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)

            loss, _ = _process_sequence(model, inp_seq, tgt_seq, loss_fn,
                                        lambda_temporal, lambda_blend_weight,
                                        blend_weight_threshold, amp_dtype)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_loss = loss.item()
            batch_size = inp_seq.size(0)
            epoch_loss += batch_loss * batch_size
            epoch_samples += batch_size
            global_step += 1

            if global_step % cfg.training.log_interval == 0:
                writer.add_scalar("train/loss", batch_loss, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        # Epoch summary
        avg_train_loss = epoch_loss / max(1, epoch_samples)
        val_loss = _validate(model, val_loader, loss_fn, lambda_temporal,
                             lambda_blend_weight, blend_weight_threshold,
                             device, amp_dtype)
        elapsed = time.time() - t0

        writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)

        patience_msg = ""
        if patience > 0:
            patience_msg = f" | patience={epochs_without_improvement}/{patience}"
        print(f"Epoch {epoch:03d}/{cfg.training.epochs} | "
              f"train_loss={avg_train_loss:.6f} | val_loss={val_loss:.6f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s{patience_msg}")

        # Sample images (last frame of first validation sequence)
        if (epoch + 1) % cfg.training.sample_interval == 0:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                sample_inp = sample_batch["input"].to(device, dtype=torch.float32)
                sample_tgt = sample_batch["target"].to(device, dtype=torch.float32)
                _, sample_outputs = _process_sequence(
                    model, sample_inp, sample_tgt, loss_fn, lambda_temporal,
                    lambda_blend_weight, blend_weight_threshold, amp_dtype)
                # Log last frame
                last_t = sample_inp.shape[1] - 1
                _log_sample_images(writer, "samples/val",
                                   sample_inp[:, last_t], sample_outputs[-1],
                                   sample_tgt[:, last_t], global_step)
            model.train()

        # Checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % cfg.training.checkpoint_interval == 0 or is_best:
            state = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config_path,
                "model_config": {
                    "type": "temporal_residual",
                    "base_channels": cfg.model.base_channels,
                    "max_mv_for_weight": max_mv,
                },
            }
            if (epoch + 1) % cfg.training.checkpoint_interval == 0:
                path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch:03d}.pt")
                torch.save(state, path)
                print(f"  Saved checkpoint: {path}")
            if is_best:
                path = os.path.join(ckpt_dir, "model_best.pt")
                torch.save(state, path)
                print(f"  Saved best model: {path} (val_loss={val_loss:.6f})")

        # Early stopping
        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping: no val_loss improvement for {patience} epochs")
            break

    writer.close()
    print(f"Training complete. Best val_loss={best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train temporal residual denoiser")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--weights-only", action="store_true",
                        help="Load only model weights (warm restart)")
    args = parser.parse_args()
    train(args.config, args.resume, args.weights_only)


if __name__ == "__main__":
    main()
