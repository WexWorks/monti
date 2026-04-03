"""Tests for PyTorch reprojection utilities.

Validates motion warping and disocclusion detection against known reference cases.
"""

import pytest
import torch

from deni_train.utils.reproject import reproject, warp_to_next_frame


@pytest.fixture
def device():
    return torch.device("cpu")


class TestReprojectIdentity:
    """Zero motion vectors should return the original image."""

    def test_identity_warp(self, device):
        B, C, H, W = 1, 3, 16, 16
        image = torch.rand(B, C, H, W, device=device)
        motion = torch.zeros(B, 2, H, W, device=device)
        prev_depth = torch.ones(B, 1, H, W, device=device)
        curr_depth = torch.ones(B, 1, H, W, device=device)

        warped, valid = reproject(image, motion, prev_depth, curr_depth)

        assert valid.all(), "All pixels should be valid with zero motion and same depth"
        # Bilinear sampling at pixel centers with zero offset should reproduce input
        assert torch.allclose(warped, image, atol=1e-4)


class TestReprojectShift:
    """Known horizontal shift should translate the image."""

    def test_uniform_horizontal_shift(self, device):
        B, C, H, W = 1, 1, 8, 8
        image = torch.zeros(B, C, H, W, device=device)
        # Put a bright column at x=4
        image[:, :, :, 4] = 1.0

        # Motion = 2px rightward (in normalized coords: 2/8 = 0.25)
        # mv = screen_current - screen_prev, so prev = current - mv
        # If object moved right by 2px: at current pixel x, source is at x - 2
        motion = torch.zeros(B, 2, H, W, device=device)
        motion[:, 0] = 2.0 / W  # X motion, normalized

        prev_depth = torch.ones(B, 1, H, W, device=device)
        curr_depth = torch.ones(B, 1, H, W, device=device)

        warped, valid = reproject(image, motion, prev_depth, curr_depth)

        # The bright column at x=4 in prev frame should appear at x=6 after warp
        # (because at current x=6, prev_pos = 6 - 2 = 4, which is the bright column)
        assert warped[0, 0, 0, 6] > 0.5, f"Expected bright pixel at x=6, got {warped[0, 0, 0, 6]}"
        assert warped[0, 0, 0, 4] < 0.1, f"Original position should be dark after shift"


class TestReprojectDisocclusion:
    """Depth mismatch should produce disocclusion."""

    def test_depth_mismatch_disoccludes(self, device):
        B, C, H, W = 1, 3, 16, 16
        image = torch.rand(B, C, H, W, device=device)
        motion = torch.zeros(B, 2, H, W, device=device)
        prev_depth = torch.ones(B, 1, H, W, device=device)
        curr_depth = torch.ones(B, 1, H, W, device=device) * 2.0  # 100% depth change

        warped, valid = reproject(image, motion, prev_depth, curr_depth)

        assert not valid.any(), "All pixels should be disoccluded with 100% depth change"
        assert (warped == 0).all(), "Disoccluded pixels should be zeroed"

    def test_small_depth_change_stays_valid(self, device):
        B, C, H, W = 1, 3, 16, 16
        image = torch.rand(B, C, H, W, device=device)
        motion = torch.zeros(B, 2, H, W, device=device)
        prev_depth = torch.ones(B, 1, H, W, device=device)
        # 5% depth change (below 10% threshold)
        curr_depth = torch.ones(B, 1, H, W, device=device) * 1.05

        warped, valid = reproject(image, motion, prev_depth, curr_depth)

        assert valid.all(), "5% depth change should be within threshold"


class TestReprojectOutOfBounds:
    """Large motion pushing source outside frame should disocclude."""

    def test_large_motion_disoccludes(self, device):
        B, C, H, W = 1, 3, 8, 8
        image = torch.rand(B, C, H, W, device=device)
        # Giant motion: 2.0 normalized = 2× frame width
        motion = torch.full((B, 2, H, W), 2.0, device=device)
        prev_depth = torch.ones(B, 1, H, W, device=device)
        curr_depth = torch.ones(B, 1, H, W, device=device)

        warped, valid = reproject(image, motion, prev_depth, curr_depth)

        assert not valid.any(), "All pixels should be OOB"


class TestReprojectDualLobe:
    """Separate diffuse and specular warping should work independently."""

    def test_dual_lobe_consistency(self, device):
        B, H, W = 1, 16, 16
        diff = torch.rand(B, 3, H, W, device=device)
        spec = torch.rand(B, 3, H, W, device=device)
        motion = torch.zeros(B, 2, H, W, device=device)
        motion[:, 0] = 1.0 / W  # Small horizontal shift
        prev_depth = torch.ones(B, 1, H, W, device=device)
        curr_depth = torch.ones(B, 1, H, W, device=device)

        warped_d, valid_d = reproject(diff, motion, prev_depth, curr_depth)
        warped_s, valid_s = reproject(spec, motion, prev_depth, curr_depth)

        # Validity masks should be identical (same motion, same depth)
        assert torch.equal(valid_d, valid_s)


class TestWarpToNextFrame:
    """Test warp_to_next_frame for temporal stability loss."""

    def test_identity_warp(self, device):
        B, C, H, W = 1, 6, 16, 16
        image = torch.rand(B, C, H, W, device=device)
        next_mv = torch.zeros(B, 2, H, W, device=device)

        warped = warp_to_next_frame(image, next_mv)

        assert torch.allclose(warped, image, atol=1e-4)

    def test_shift_warp(self, device):
        B, C, H, W = 1, 1, 8, 8
        image = torch.zeros(B, C, H, W, device=device)
        image[:, :, :, 3] = 1.0  # Bright column at x=3

        # If object moved right by 1px from frame t to t+1:
        # mv_{t+1} = pos_{t+1} - pos_t, so source in t = pos_{t+1} - mv_{t+1}
        next_mv = torch.zeros(B, 2, H, W, device=device)
        next_mv[:, 0] = 1.0 / W  # Object moved right by 1px

        warped = warp_to_next_frame(image, next_mv)

        # At t+1 pixel x=4: source in t = 4 - 1 = 3, which is the bright column
        assert warped[0, 0, 0, 4] > 0.5
        # At t+1 pixel x=3: source in t = 3 - 1 = 2, which is dark
        assert warped[0, 0, 0, 3] < 0.1

    def test_output_shape(self, device):
        image = torch.rand(1, 6, 32, 32, device=device)
        next_mv = torch.rand(1, 2, 32, 32, device=device) * 0.01
        warped = warp_to_next_frame(image, next_mv)
        assert warped.shape == image.shape
