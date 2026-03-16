"""Image quality metrics: PSNR and SSIM in ACES-tonemapped space."""

import torch
import torch.nn.functional as F

from .tonemapping import aces_tonemap


def compute_psnr(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between predicted and target in ACES-tonemapped space.

    Args:
        predicted: (B, 3, H, W) linear HDR tensor
        target: (B, 3, H, W) linear HDR tensor

    Returns:
        PSNR in dB (averaged over batch).
    """
    pred_tm = aces_tonemap(predicted)
    tgt_tm = aces_tonemap(target)
    mse = F.mse_loss(pred_tm, tgt_tm).item()
    if mse < 1e-10:
        return 100.0
    return -10.0 * torch.tensor(mse).log10().item()


def compute_ssim(
    predicted: torch.Tensor, target: torch.Tensor, window_size: int = 11
) -> float:
    """Compute SSIM between predicted and target in ACES-tonemapped space.

    Uses a uniform window (box filter) for simplicity. Operates per-channel
    then averages.

    Args:
        predicted: (B, 3, H, W) linear HDR tensor
        target: (B, 3, H, W) linear HDR tensor
        window_size: spatial window size for local statistics

    Returns:
        Mean SSIM (averaged over batch and channels).
    """
    pred_tm = aces_tonemap(predicted)
    tgt_tm = aces_tonemap(target)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    b, c, h, w = pred_tm.shape
    # Uniform window kernel
    kernel = torch.ones(c, 1, window_size, window_size, device=pred_tm.device, dtype=pred_tm.dtype)
    kernel = kernel / (window_size * window_size)
    pad = window_size // 2

    mu_pred = F.conv2d(pred_tm, kernel, padding=pad, groups=c)
    mu_tgt = F.conv2d(tgt_tm, kernel, padding=pad, groups=c)

    mu_pred_sq = mu_pred * mu_pred
    mu_tgt_sq = mu_tgt * mu_tgt
    mu_cross = mu_pred * mu_tgt

    sigma_pred_sq = F.conv2d(pred_tm * pred_tm, kernel, padding=pad, groups=c) - mu_pred_sq
    sigma_tgt_sq = F.conv2d(tgt_tm * tgt_tm, kernel, padding=pad, groups=c) - mu_tgt_sq
    sigma_cross = F.conv2d(pred_tm * tgt_tm, kernel, padding=pad, groups=c) - mu_cross

    numerator = (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_pred_sq + mu_tgt_sq + c1) * (sigma_pred_sq + sigma_tgt_sq + c2)

    ssim_map = numerator / denominator
    return ssim_map.mean().item()
