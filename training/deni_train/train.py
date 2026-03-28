"""Training script for DeniUNet denoiser.

CLI: python -m deni_train.train --config configs/default.yaml [--resume checkpoint.pt] [--weights-only]
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from .data.exr_dataset import ExrDataset
from .data.safetensors_dataset import SafetensorsDataset
from .data.splits import detect_data_format, stratified_split, stratified_split_files
from .data.transforms import Compose, RandomCrop, RandomRotation180
from .losses.denoiser_loss import DenoiserLoss
from .models.unet import DeniUNet
from .utils.tonemapping import aces_tonemap

# Input channel layout: [0-2] demod diffuse, [3-5] demod specular, [6-8] normals,
# [9] roughness, [10] depth, [11-12] motion, [13-15] albedo_d, [16-18] albedo_s


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
    transform = Compose([
        RandomCrop(cfg.data.crop_size),
        RandomRotation180(),
    ])

    data_format = getattr(cfg.data, "data_format", "auto")
    if data_format == "auto":
        data_format = detect_data_format(cfg.data.data_dir)
    print(f"Data format: {data_format}")

    if data_format == "safetensors":
        dataset = SafetensorsDataset(cfg.data.data_dir, transform=transform,
                                     crops_per_image=getattr(cfg.data, "crops_per_image", 1))
        n = len(dataset)
        if n == 0:
            raise RuntimeError(f"No safetensors files found in {cfg.data.data_dir}")
        train_indices, val_indices = stratified_split_files(dataset.files)
    else:
        dataset = ExrDataset(cfg.data.data_dir, transform=transform,
                             crops_per_image=getattr(cfg.data, "crops_per_image", 1))
        n = len(dataset)
        if n == 0:
            raise RuntimeError(f"No EXR pairs found in {cfg.data.data_dir}")
        train_indices, val_indices = stratified_split(dataset.pairs)
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    # On Windows, multi-process DataLoader workers can fail due to spawn
    # limitations with OpenEXR file handles. Fall back to num_workers=0.
    # Safetensors uses memory-mapped I/O and does not have this issue.
    num_workers = cfg.data.num_workers
    if sys.platform == "win32" and num_workers > 0 and data_format != "safetensors":
        num_workers = 0

    persistent = num_workers > 0
    prefetch = 4 if num_workers > 0 else 2
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
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


def _log_sample_images(writer, tag: str, inp: torch.Tensor, pred: torch.Tensor,
                       tgt: torch.Tensor, albedo_d: torch.Tensor,
                       albedo_s: torch.Tensor, global_step: int):
    """Log a triplet of tonemapped images to TensorBoard."""
    with torch.no_grad():
        # Take first sample in batch, remodulate for visualization
        alb_d = albedo_d[0]  # (3, H, W)
        alb_s = albedo_s[0]  # (3, H, W)

        # Remodulate input: demodulated irradiance * albedo
        inp_rgb = inp[0, :3] * alb_d + inp[0, 3:6] * alb_s
        inp_tm = aces_tonemap(inp_rgb.unsqueeze(0)).squeeze(0).clamp(0.0, 1.0)

        # Remodulate prediction and target
        pred_rgb = pred[0, :3] * alb_d + pred[0, 3:6] * alb_s
        pred_tm = aces_tonemap(pred_rgb.unsqueeze(0)).squeeze(0).clamp(0.0, 1.0)

        tgt_rgb = tgt[0, :3] * alb_d + tgt[0, 3:6] * alb_s
        tgt_tm = aces_tonemap(tgt_rgb.unsqueeze(0)).squeeze(0).clamp(0.0, 1.0)

        # Concatenate horizontally: input | predicted | target
        triplet = torch.cat([inp_tm, pred_tm, tgt_tm], dim=2)
        writer.add_image(tag, triplet, global_step)


def _validate(model, val_loader, loss_fn, device, amp_dtype):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for inp, tgt, albedo_d, albedo_s, hit_mask in val_loader:
            inp = inp.to(device, dtype=torch.float32)
            tgt = tgt.to(device, dtype=torch.float32)
            albedo_d = albedo_d.to(device, dtype=torch.float32)
            albedo_s = albedo_s.to(device, dtype=torch.float32)
            hit_mask = hit_mask.to(device, dtype=torch.float32)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                pred = model(inp)
                loss = loss_fn(pred, tgt, albedo_d, albedo_s, hit_mask)
            total_loss += loss.item() * inp.size(0)
            count += inp.size(0)
    model.train()
    if count == 0:
        return 0.0
    return total_loss / count


def train(config_path: str, resume_path: str | None = None, weights_only: bool = False):
    with open(config_path, "r") as f:
        cfg = _Config(yaml.safe_load(f))

    # Resolve data_dir relative to config file directory
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if not os.path.isabs(cfg.data.data_dir):
        cfg.data.data_dir = os.path.normpath(os.path.join(config_dir, cfg.data.data_dir))

    _seed_everything(cfg.training.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training (mixed precision, GradScaler)")
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if cfg.training.mixed_precision else torch.float32

    # Build data
    train_loader, val_loader = _build_dataloaders(cfg)
    print(f"Training samples: {len(train_loader.dataset)}, "
          f"Validation samples: {len(val_loader.dataset)}")

    # Build model
    model = DeniUNet(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        base_channels=cfg.model.base_channels,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Build loss, optimizer, scheduler
    loss_fn = DenoiserLoss(
        lambda_l1=cfg.loss.lambda_l1,
        lambda_perceptual=cfg.loss.lambda_perceptual,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    steps_per_epoch = max(1, len(train_loader))
    scheduler = _build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.mixed_precision)

    # Resume from checkpoint
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
            print(f"Loaded weights only (warm restart), best_val_loss={ckpt['best_val_loss']:.6f}")

    # TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    log_dir = os.path.join(os.path.dirname(config_path), "runs",
                           os.path.splitext(os.path.basename(config_path))[0])
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard log dir: {log_dir}")

    # Output directory for checkpoints
    ckpt_dir = os.path.join(os.path.dirname(config_path), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    model.train()
    for epoch in range(start_epoch, cfg.training.epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        t0 = time.time()

        for inp, tgt, albedo_d, albedo_s, hit_mask in train_loader:
            inp = inp.to(device, dtype=torch.float32)
            tgt = tgt.to(device, dtype=torch.float32)
            albedo_d = albedo_d.to(device, dtype=torch.float32)
            albedo_s = albedo_s.to(device, dtype=torch.float32)
            hit_mask = hit_mask.to(device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                pred = model(inp)
                loss = loss_fn(pred, tgt, albedo_d, albedo_s, hit_mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_loss = loss.item()
            batch_size = inp.size(0)
            epoch_loss += batch_loss * batch_size
            epoch_samples += batch_size
            global_step += 1

            if global_step % cfg.training.log_interval == 0:
                writer.add_scalar("train/loss", batch_loss, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        # Epoch summary
        avg_train_loss = epoch_loss / max(1, epoch_samples)
        val_loss = _validate(model, val_loader, loss_fn, device, amp_dtype)
        elapsed = time.time() - t0

        writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)

        patience_msg = ""
        if patience > 0:
            patience_msg = f" | patience={epochs_without_improvement}/{patience}"
        print(f"Epoch {epoch:03d}/{cfg.training.epochs} | "
              f"train_loss={avg_train_loss:.6f} | val_loss={val_loss:.6f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s{patience_msg}")

        # Sample images
        if (epoch + 1) % cfg.training.sample_interval == 0:
            model.eval()
            with torch.no_grad():
                sample_inp, sample_tgt, sample_alb_d, sample_alb_s, _ = next(iter(val_loader))
                sample_inp = sample_inp.to(device, dtype=torch.float32)
                sample_tgt = sample_tgt.to(device, dtype=torch.float32)
                sample_alb_d = sample_alb_d.to(device, dtype=torch.float32)
                sample_alb_s = sample_alb_s.to(device, dtype=torch.float32)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    sample_pred = model(sample_inp)
                _log_sample_images(writer, "samples/val", sample_inp, sample_pred,
                                   sample_tgt, sample_alb_d, sample_alb_s, global_step)
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
                    "in_channels": cfg.model.in_channels,
                    "out_channels": cfg.model.out_channels,
                    "base_channels": cfg.model.base_channels,
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
    parser = argparse.ArgumentParser(description="Train DeniUNet denoiser")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--weights-only", action="store_true",
                        help="Load only model weights from checkpoint (warm restart: fresh LR schedule)")
    args = parser.parse_args()
    train(args.config, args.resume, args.weights_only)


if __name__ == "__main__":
    main()
