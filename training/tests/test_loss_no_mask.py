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
        loss = loss_fn(pred, pred.clone(), albedo_d, albedo_s)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_l1_correct_for_known_difference(self, loss_fn):
        """L1 loss is non-zero for different pred and target."""
        B, H, W = 1, 32, 32
        pred = torch.zeros(B, 6, H, W)
        tgt = torch.ones(B, 6, H, W)
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.ones(B, 3, H, W)
        loss = loss_fn(pred, tgt, albedo_d, albedo_s)
        assert loss.item() > 0.0

    def test_vgg_uses_full_image(self, loss_fn):
        """VGG features are extracted from the full image including background pixels."""
        B, H, W = 1, 64, 64
        # Create pred/target with non-zero values everywhere
        pred = torch.rand(B, 6, H, W) * 0.5
        tgt = torch.rand(B, 6, H, W) * 0.5 + 0.5
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.zeros(B, 3, H, W)
        loss = loss_fn(pred, tgt, albedo_d, albedo_s)
        assert loss.item() > 0.0

    def test_gradient_flows_to_all_pixels(self, loss_fn):
        """Gradients are non-zero for all pixel positions including background."""
        B, H, W = 1, 32, 32
        pred = torch.rand(B, 6, H, W, requires_grad=True)
        tgt = torch.rand(B, 6, H, W)
        # Use unit diffuse albedo (simulating background pixels)
        albedo_d = torch.ones(B, 3, H, W)
        albedo_s = torch.zeros(B, 3, H, W)
        loss = loss_fn(pred, tgt, albedo_d, albedo_s)
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
