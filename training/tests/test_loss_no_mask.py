"""Tests for DenoiserLoss without hit_mask (Phase 5)."""

import pytest
import torch

from deni_train.losses.denoiser_loss import DenoiserLoss


@pytest.fixture
def loss_fn():
    """DenoiserLoss on CPU for testing."""
    return DenoiserLoss(lambda_l1=1.0, lambda_perceptual=0.1).eval()


class TestDenoiserLossNoMask:
    def test_zero_loss_on_identical_inputs(self, loss_fn):
        """L1 loss on uniform prediction is zero."""
        B, H, W = 1, 32, 32
        pred = torch.rand(B, 6, H, W)
        albedo_d = torch.rand(B, 3, H, W).clamp(min=0.1)
        albedo_s = torch.rand(B, 3, H, W).clamp(min=0.1)
        loss, _ = loss_fn(pred, pred.clone(), albedo_d, albedo_s)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_l1_correct_for_known_difference(self, loss_fn):
        """L1 loss is non-zero for different pred and target."""
        B, H, W = 1, 32, 32
        pred = torch.zeros(B, 6, H, W)
        tgt = torch.ones(B, 6, H, W)
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.ones(B, 3, H, W)
        loss, _ = loss_fn(pred, tgt, albedo_d, albedo_s)
        assert loss.item() > 0.0

    def test_vgg_uses_full_image(self, loss_fn):
        """VGG features are extracted from the full image including background pixels."""
        B, H, W = 1, 64, 64
        # Create pred/target with non-zero values everywhere
        pred = torch.rand(B, 6, H, W) * 0.5
        tgt = torch.rand(B, 6, H, W) * 0.5 + 0.5
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.zeros(B, 3, H, W)
        loss, _ = loss_fn(pred, tgt, albedo_d, albedo_s)
        assert loss.item() > 0.0

    def test_gradient_flows_to_all_pixels(self, loss_fn):
        """Gradients are non-zero for all pixel positions including background."""
        B, H, W = 1, 32, 32
        pred = torch.rand(B, 6, H, W, requires_grad=True)
        tgt = torch.rand(B, 6, H, W)
        # Use unit diffuse albedo (simulating background pixels)
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.zeros(B, 3, H, W)
        loss, _ = loss_fn(pred, tgt, albedo_d, albedo_s)
        loss.backward()
        assert pred.grad is not None
        # Every spatial position should receive gradient
        grad_spatial = pred.grad.abs().sum(dim=1)  # (B, H, W)
        assert (grad_spatial > 0).all(), "Some pixels received zero gradient"

    def test_forward_signature_no_hit_mask(self, loss_fn):
        """Loss forward() takes exactly 4 args (pred, tgt, albedo_d, albedo_s)."""
        import inspect
        sig = inspect.signature(loss_fn.forward)
        params = [p for p in sig.parameters if p != "self"]
        assert params == ["predicted", "target", "albedo_d", "albedo_s"]


class TestRadianceL1Loss:
    """Tests for the combined-radiance L1 loss component."""

    def test_radiance_l1_zero_on_identical(self):
        """Radiance L1 is zero when pred == target."""
        loss_fn = DenoiserLoss(lambda_l1=0.0, lambda_perceptual=0.0,
                               lambda_radiance_l1=1.0, lambda_hue=0.0).eval()
        B, H, W = 1, 32, 32
        pred = torch.rand(B, 6, H, W)
        albedo_d = torch.rand(B, 3, H, W).clamp(min=0.1)
        albedo_s = torch.rand(B, 3, H, W).clamp(min=0.1)
        loss, comps = loss_fn(pred, pred.clone(), albedo_d, albedo_s)
        assert comps["radiance_l1"].item() == pytest.approx(0.0, abs=1e-5)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_radiance_l1_nonzero_on_different(self):
        """Radiance L1 is positive when pred != target."""
        loss_fn = DenoiserLoss(lambda_l1=0.0, lambda_perceptual=0.0,
                               lambda_radiance_l1=1.0, lambda_hue=0.0).eval()
        B, H, W = 1, 32, 32
        pred = torch.zeros(B, 6, H, W)
        tgt = torch.ones(B, 6, H, W)
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.ones(B, 3, H, W)
        loss, comps = loss_fn(pred, tgt, albedo_d, albedo_s)
        assert comps["radiance_l1"].item() > 0.0

    def test_radiance_l1_remodulates_correctly(self):
        """Radiance L1 loss accounts for albedo — identical irradiance but different
        albedo-weighted radiance should produce non-zero radiance_l1."""
        loss_fn = DenoiserLoss(lambda_l1=0.0, lambda_perceptual=0.0,
                               lambda_radiance_l1=1.0, lambda_hue=0.0).eval()
        B, H, W = 1, 32, 32
        # Same irradiance, but pred=0.5 uniform, target differs only in specular
        pred = torch.full((B, 6, H, W), 0.5)
        tgt = pred.clone()
        tgt[:, 3:6] = 0.8  # different specular irradiance

        # With zero specular albedo, remodulated radiance should be the same
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s_zero = torch.zeros(B, 3, H, W)
        _, comps_zero_s = loss_fn(pred, tgt, albedo_d, albedo_s_zero)

        # With high specular albedo, the specular difference becomes visible
        albedo_s_high = torch.ones(B, 3, H, W)
        _, comps_high_s = loss_fn(pred, tgt, albedo_d, albedo_s_high)

        # The loss with high specular albedo should be larger
        assert comps_high_s["radiance_l1"].item() > comps_zero_s["radiance_l1"].item()

    def test_radiance_l1_gradient_flows(self):
        """Gradient flows through radiance L1 to all 6 output channels."""
        loss_fn = DenoiserLoss(lambda_l1=0.0, lambda_perceptual=0.0,
                               lambda_radiance_l1=1.0, lambda_hue=0.0).eval()
        B, H, W = 1, 32, 32
        pred = torch.rand(B, 6, H, W, requires_grad=True)
        tgt = torch.rand(B, 6, H, W)
        albedo_d = torch.ones(B, 3, H, W) * 0.5
        albedo_s = torch.ones(B, 3, H, W) * 0.5
        loss, _ = loss_fn(pred, tgt, albedo_d, albedo_s)
        loss.backward()
        # All 6 channels should have non-zero gradients
        for c in range(6):
            assert pred.grad[:, c].abs().sum() > 0, f"Channel {c} has zero gradient"


class TestHueLoss:
    """Tests for the cosine similarity hue loss component."""

    def test_hue_zero_on_identical(self):
        """Hue loss is zero when pred == target."""
        loss_fn = DenoiserLoss(lambda_l1=0.0, lambda_perceptual=0.0,
                               lambda_radiance_l1=0.0, lambda_hue=1.0).eval()
        B, H, W = 1, 32, 32
        pred = torch.rand(B, 6, H, W).clamp(min=0.01)
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.zeros(B, 3, H, W)
        loss, comps = loss_fn(pred, pred.clone(), albedo_d, albedo_s)
        assert comps["hue"].item() == pytest.approx(0.0, abs=1e-4)

    def test_hue_detects_color_rotation(self):
        """Hue loss is higher when colors are rotated (R→G shift) vs just scaled."""
        loss_fn = DenoiserLoss(lambda_l1=0.0, lambda_perceptual=0.0,
                               lambda_radiance_l1=0.0, lambda_hue=1.0).eval()
        B, H, W = 1, 32, 32
        # Target: strong red diffuse irradiance
        tgt = torch.zeros(B, 6, H, W)
        tgt[:, 0] = 1.0  # red diffuse channel
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.zeros(B, 3, H, W)

        # Pred A: same red but dimmer (luminance error, correct hue)
        pred_dim = torch.zeros(B, 6, H, W)
        pred_dim[:, 0] = 0.5
        _, comps_dim = loss_fn(pred_dim, tgt, albedo_d, albedo_s)

        # Pred B: green instead of red (hue error)
        pred_rotated = torch.zeros(B, 6, H, W)
        pred_rotated[:, 1] = 1.0  # green instead of red
        _, comps_rotated = loss_fn(pred_rotated, tgt, albedo_d, albedo_s)

        # Hue loss should be much higher for the color rotation
        assert comps_rotated["hue"].item() > comps_dim["hue"].item() * 2.0

    def test_hue_invariant_to_brightness(self):
        """Hue loss is similar for same-direction vectors at different magnitudes."""
        loss_fn = DenoiserLoss(lambda_l1=0.0, lambda_perceptual=0.0,
                               lambda_radiance_l1=0.0, lambda_hue=1.0).eval()
        B, H, W = 1, 32, 32
        tgt = torch.zeros(B, 6, H, W)
        tgt[:, 0] = 1.0  # red
        tgt[:, 1] = 0.3  # slight green
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.zeros(B, 3, H, W)

        # Same direction, half brightness
        pred_half = tgt.clone() * 0.5
        _, comps_half = loss_fn(pred_half, tgt, albedo_d, albedo_s)

        # Same direction, double brightness
        pred_double = tgt.clone() * 2.0
        _, comps_double = loss_fn(pred_double, tgt, albedo_d, albedo_s)

        # Both should have very low hue loss (same direction, ACES distortion aside)
        # Note: ACES tonemapping is nonlinear, so there IS some hue rotation at
        # different luminances. But it should be much smaller than a real R→G rotation.
        pred_rotated = torch.zeros(B, 6, H, W)
        pred_rotated[:, 1] = 1.0  # green
        _, comps_rotated = loss_fn(pred_rotated, tgt, albedo_d, albedo_s)

        assert comps_half["hue"].item() < comps_rotated["hue"].item() * 0.5
        assert comps_double["hue"].item() < comps_rotated["hue"].item() * 0.5

    def test_hue_gradient_flows(self):
        """Gradient from hue loss flows to prediction."""
        loss_fn = DenoiserLoss(lambda_l1=0.0, lambda_perceptual=0.0,
                               lambda_radiance_l1=0.0, lambda_hue=1.0).eval()
        B, H, W = 1, 32, 32
        pred = torch.rand(B, 6, H, W, requires_grad=True)
        tgt = torch.rand(B, 6, H, W)
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.zeros(B, 3, H, W)
        loss, _ = loss_fn(pred, tgt, albedo_d, albedo_s)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_hue_gated_on_dark_target(self):
        """Hue loss is zero when the target is below the brightness gate."""
        loss_fn = DenoiserLoss(lambda_l1=0.0, lambda_perceptual=0.0,
                               lambda_radiance_l1=0.0, lambda_hue=1.0).eval()
        B, H, W = 1, 32, 32
        # Very dark target: ACES(0.001) ≈ 0.0002, well below 0.05 gate
        tgt = torch.full((B, 6, H, W), 0.001)
        # Pred with different hue (would normally give high hue loss)
        pred = torch.zeros(B, 6, H, W)
        pred[:, 1] = 0.5  # green, different hue from target
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.zeros(B, 3, H, W)
        _, comps = loss_fn(pred, tgt, albedo_d, albedo_s)
        # Should be zero or near-zero due to brightness gating
        assert comps["hue"].item() < 0.01

    def test_hue_no_gradient_for_dark_target(self):
        """Brightness gating prevents gradient flow from dark-target pixels.

        This is the core mechanism: cosine similarity on near-black pixels
        produces a spurious gray hue target [0.577, 0.577, 0.577] regardless
        of actual pixel color. Without gating, ~26% of training pixels push
        the model toward gray/zero output. The gate must zero both the loss
        value AND the gradient for these pixels.
        """
        loss_fn = DenoiserLoss(lambda_l1=0.0, lambda_perceptual=0.0,
                               lambda_radiance_l1=0.0, lambda_hue=1.0).eval()
        B, H, W = 1, 32, 32
        # Dark target (below brightness gate)
        tgt = torch.full((B, 6, H, W), 0.001)
        # Colorful prediction (wrong hue relative to "gray" direction)
        pred = torch.zeros(B, 6, H, W, requires_grad=True)
        pred_data = pred.data
        pred_data[:, 0] = 1.0  # red
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.zeros(B, 3, H, W)
        loss, _ = loss_fn(pred, tgt, albedo_d, albedo_s)
        loss.backward()
        # Gradient must be zero — these pixels should not influence training
        assert pred.grad.abs().max().item() == 0.0

    def test_hue_active_on_bright_pixels_only(self):
        """Hue loss activates on bright pixels and ignores dark ones in same image.

        Verifies that a mixed image (half bright, half dark) only computes hue
        loss over the bright half.
        """
        loss_fn = DenoiserLoss(lambda_l1=0.0, lambda_perceptual=0.0,
                               lambda_radiance_l1=0.0, lambda_hue=1.0).eval()
        B, H, W = 1, 32, 32
        # Top half: bright red target. Bottom half: near-black.
        tgt = torch.zeros(B, 6, H, W)
        tgt[:, 0, :H // 2, :] = 1.0  # bright red on top half only
        tgt[:, :, H // 2:, :] = 0.001  # dark bottom half

        # Prediction: green everywhere (wrong hue for bright, irrelevant for dark)
        pred = torch.zeros(B, 6, H, W)
        pred[:, 1] = 1.0  # green

        # All bright: should give higher hue loss than mixed
        tgt_bright = torch.zeros(B, 6, H, W)
        tgt_bright[:, 0] = 1.0  # bright red everywhere
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.zeros(B, 3, H, W)

        _, comps_mixed = loss_fn(pred, tgt, albedo_d, albedo_s)
        _, comps_bright = loss_fn(pred, tgt_bright, albedo_d, albedo_s)

        # Both should report nonzero hue loss (bright region contributes)
        assert comps_mixed["hue"].item() > 0.1
        assert comps_bright["hue"].item() > 0.1
        # The mixed-image hue loss normalizes by bright pixel count, not total
        # pixel count, so the per-bright-pixel loss should be similar
        assert comps_mixed["hue"].item() == pytest.approx(
            comps_bright["hue"].item(), rel=0.3
        )

    def test_hue_gradient_bounded_near_zero(self):
        """Hue loss gradient is bounded even for very small predictions."""
        loss_fn = DenoiserLoss(lambda_l1=0.0, lambda_perceptual=0.0,
                               lambda_radiance_l1=0.0, lambda_hue=1.0).eval()
        B, H, W = 1, 32, 32
        pred = torch.full((B, 6, H, W), 0.001, requires_grad=True)
        tgt = torch.rand(B, 6, H, W).clamp(min=0.5)  # bright target
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.zeros(B, 3, H, W)
        loss, _ = loss_fn(pred, tgt, albedo_d, albedo_s)
        loss.backward()
        # With brightness gating and eps=1e-3, gradient should stay reasonable
        # (not explode to hundreds like with eps=1e-6)
        assert pred.grad.norm().item() < 50.0


class TestLossComponents:
    """Tests for the per-component loss reporting."""

    def test_all_components_returned(self):
        """forward() returns dict with all expected component keys."""
        loss_fn = DenoiserLoss(lambda_l1=1.0, lambda_perceptual=0.1,
                               lambda_radiance_l1=0.5, lambda_hue=0.25).eval()
        B, H, W = 1, 32, 32
        pred = torch.rand(B, 6, H, W)
        tgt = torch.rand(B, 6, H, W)
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.ones(B, 3, H, W)
        _, comps = loss_fn(pred, tgt, albedo_d, albedo_s)
        assert set(comps.keys()) == {"l1", "perceptual", "radiance_l1", "hue"}

    def test_components_are_unweighted(self):
        """Component values are raw (unweighted) — the total applies lambdas."""
        loss_fn = DenoiserLoss(lambda_l1=2.0, lambda_perceptual=0.0,
                               lambda_radiance_l1=0.0, lambda_hue=0.0).eval()
        B, H, W = 1, 32, 32
        pred = torch.zeros(B, 6, H, W)
        tgt = torch.ones(B, 6, H, W)
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.ones(B, 3, H, W)
        total, comps = loss_fn(pred, tgt, albedo_d, albedo_s)
        # With only L1 active (lambda=2.0), total should be 2 × l1 component
        assert total.item() == pytest.approx(2.0 * comps["l1"].item(), rel=1e-5)

    def test_total_is_weighted_sum(self):
        """Total loss equals weighted sum of components."""
        lam_l1, lam_p, lam_r, lam_h = 1.0, 0.1, 0.5, 0.25
        loss_fn = DenoiserLoss(lambda_l1=lam_l1, lambda_perceptual=lam_p,
                               lambda_radiance_l1=lam_r, lambda_hue=lam_h).eval()
        B, H, W = 1, 64, 64
        pred = torch.rand(B, 6, H, W)
        tgt = torch.rand(B, 6, H, W)
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.ones(B, 3, H, W)
        total, comps = loss_fn(pred, tgt, albedo_d, albedo_s)
        expected = (lam_l1 * comps["l1"] + lam_p * comps["perceptual"]
                    + lam_r * comps["radiance_l1"] + lam_h * comps["hue"])
        assert total.item() == pytest.approx(expected.item(), rel=1e-4)

    def test_disabled_components_dont_contribute(self):
        """Components with lambda=0 produce zero contribution to total."""
        loss_fn = DenoiserLoss(lambda_l1=0.0, lambda_perceptual=0.0,
                               lambda_radiance_l1=0.0, lambda_hue=0.0).eval()
        B, H, W = 1, 32, 32
        pred = torch.zeros(B, 6, H, W)
        tgt = torch.ones(B, 6, H, W)
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.ones(B, 3, H, W)
        total, comps = loss_fn(pred, tgt, albedo_d, albedo_s)
        assert total.item() == pytest.approx(0.0, abs=1e-6)
        # But the components themselves are still computed and nonzero
        assert comps["l1"].item() > 0.0
        assert comps["radiance_l1"].item() > 0.0
