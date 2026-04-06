"""PyTorch reprojection utilities for temporal training.

Provides motion-vector-based warping and disocclusion detection that mirrors
the GPU reproject.comp shader logic, for use in the temporal training loop
(autoregressive reprojection and temporal stability loss).
"""

import torch
import torch.nn.functional as F


def reproject(
    image: torch.Tensor,
    motion_vectors: torch.Tensor,
    prev_depth: torch.Tensor,
    curr_depth: torch.Tensor,
    depth_threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp an image from the previous frame to the current frame using motion vectors.

    Mirrors reproject.comp: prev_pos = current_pos - mv * resolution.

    Args:
        image: (B, C, H, W) previous frame image to warp.
        motion_vectors: (B, 2, H, W) normalized [0,1] screen-space motion vectors
                        for the current frame (mv = screen_current - screen_prev).
        prev_depth: (B, 1, H, W) previous frame linear depth.
        curr_depth: (B, 1, H, W) current frame linear depth.
        depth_threshold: Relative depth difference threshold for disocclusion.

    Returns:
        warped: (B, C, H, W) warped image (zeros where disoccluded or out-of-bounds).
        valid_mask: (B, 1, H, W) validity mask (1=valid, 0=disoccluded/OOB).
    """
    B, C, H, W = image.shape

    # Build sampling grid: pixel centers in normalized [-1, 1] coordinates
    # grid_sample expects (B, H, W, 2) with (x, y) in [-1, 1]
    gy, gx = torch.meshgrid(
        torch.arange(H, dtype=image.dtype, device=image.device),
        torch.arange(W, dtype=image.dtype, device=image.device),
        indexing="ij",
    )
    # Pixel centers: (i + 0.5) / dim
    gx = (gx + 0.5) / W  # [0, 1]
    gy = (gy + 0.5) / H  # [0, 1]

    # Motion vectors: mv = screen_current - screen_prev
    # Previous position: prev = current - mv (in [0,1] space)
    mv_x = motion_vectors[:, 0:1]  # (B, 1, H, W)
    mv_y = motion_vectors[:, 1:2]  # (B, 1, H, W)

    prev_x = gx.unsqueeze(0).unsqueeze(0) - mv_x  # (B, 1, H, W)
    prev_y = gy.unsqueeze(0).unsqueeze(0) - mv_y  # (B, 1, H, W)

    # Convert to grid_sample's [-1, 1] range: x_norm = 2 * x_01 - 1
    grid_x = 2.0 * prev_x - 1.0
    grid_y = 2.0 * prev_y - 1.0
    grid = torch.cat([grid_x, grid_y], dim=1)  # (B, 2, H, W)
    grid = grid.permute(0, 2, 3, 1)  # (B, H, W, 2)

    # Sample from previous frame using bilinear interpolation
    warped = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros",
                           align_corners=False)

    # Out-of-bounds mask
    oob = (prev_x < 0) | (prev_x > 1) | (prev_y < 0) | (prev_y > 1)
    oob = oob.squeeze(1)  # (B, H, W)

    # Sample previous depth at warped positions
    warped_prev_depth = F.grid_sample(prev_depth, grid, mode="nearest",
                                       padding_mode="zeros", align_corners=False)

    # Depth-based disocclusion: |curr_z - prev_z| / max(prev_z, eps) > threshold
    depth_diff = torch.abs(curr_depth - warped_prev_depth)
    depth_ratio = depth_diff / torch.clamp(warped_prev_depth, min=1e-6)
    depth_valid = (depth_ratio < depth_threshold).squeeze(1)  # (B, H, W)

    # Combined validity
    valid = (~oob) & depth_valid  # (B, H, W)
    valid_mask = valid.unsqueeze(1).float()  # (B, 1, H, W)

    # Zero out invalid pixels
    warped = warped * valid_mask

    return warped, valid_mask


def warp_to_next_frame(
    image: torch.Tensor,
    next_motion_vectors: torch.Tensor,
) -> torch.Tensor:
    """Warp image from frame t into frame t+1's coordinate space.

    Used for temporal stability loss: compare warp(output_t, mv_{t+1}) vs output_{t+1}.

    Frame t+1's motion vectors satisfy: mv_{t+1} = screen_{t+1} - screen_t
    So: screen_t = screen_{t+1} - mv_{t+1}
    We want to sample image_t at positions corresponding to each pixel in frame t+1.

    Args:
        image: (B, C, H, W) frame t image.
        next_motion_vectors: (B, 2, H, W) frame t+1's normalized motion vectors.

    Returns:
        warped: (B, C, H, W) image warped into frame t+1's space (zeros at OOB).
    """
    B, C, H, W = image.shape

    gy, gx = torch.meshgrid(
        torch.arange(H, dtype=image.dtype, device=image.device),
        torch.arange(W, dtype=image.dtype, device=image.device),
        indexing="ij",
    )
    gx = (gx + 0.5) / W
    gy = (gy + 0.5) / H

    mv_x = next_motion_vectors[:, 0:1]
    mv_y = next_motion_vectors[:, 1:2]

    # For frame t+1: prev_pos (in frame t) = current_pos (t+1) - mv_{t+1}
    src_x = gx.unsqueeze(0).unsqueeze(0) - mv_x
    src_y = gy.unsqueeze(0).unsqueeze(0) - mv_y

    grid_x = 2.0 * src_x - 1.0
    grid_y = 2.0 * src_y - 1.0
    grid = torch.cat([grid_x, grid_y], dim=1).permute(0, 2, 3, 1)

    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False)
