"""Tests for F9-2: U-Net architecture + loss functions."""

import torch
import torch.nn as nn
import pytest

from deni_train.models.unet import DeniUNet
from deni_train.models.blocks import ConvBlock, DownBlock, UpBlock
from deni_train.losses.denoiser_loss import DenoiserLoss
from deni_train.utils.tonemapping import aces_tonemap


# ---------------------------------------------------------------------------
# ConvBlock tests
# ---------------------------------------------------------------------------

class TestConvBlock:
    def test_output_shape(self):
        block = ConvBlock(13, 16)
        x = torch.randn(1, 13, 64, 64)
        y = block(x)
        assert y.shape == (1, 16, 64, 64)

    def test_kaiming_init(self):
        block = ConvBlock(3, 16)
        # Kaiming normal gives non-zero weights with specific fan_out variance
        assert block.conv.weight.abs().mean() > 0.0
        assert block.conv.bias.abs().max() == 0.0


# ---------------------------------------------------------------------------
# DownBlock tests
# ---------------------------------------------------------------------------

class TestDownBlock:
    def test_output_and_skip_shapes(self):
        block = DownBlock(16, 32)
        x = torch.randn(1, 16, 64, 64)
        pooled, skip = block(x)
        assert pooled.shape == (1, 32, 32, 32)
        assert skip.shape == (1, 32, 64, 64)


# ---------------------------------------------------------------------------
# UpBlock tests
# ---------------------------------------------------------------------------

class TestUpBlock:
    def test_output_shape(self):
        block = UpBlock(in_ch=64, skip_ch=32, out_ch=32)
        x = torch.randn(1, 64, 16, 16)
        skip = torch.randn(1, 32, 32, 32)
        y = block(x, skip)
        assert y.shape == (1, 32, 32, 32)


# ---------------------------------------------------------------------------
# DeniUNet tests
# ---------------------------------------------------------------------------

class TestDeniUNet:
    def test_forward_pass_shape(self):
        model = DeniUNet(in_channels=13, out_channels=3, base_channels=16)
        x = torch.randn(2, 13, 256, 256)
        y = model(x)
        assert y.shape == (2, 3, 256, 256)

    def test_forward_pass_small(self):
        model = DeniUNet(in_channels=13, out_channels=3, base_channels=16)
        x = torch.randn(1, 13, 64, 64)
        y = model(x)
        assert y.shape == (1, 3, 64, 64)

    def test_parameter_count_in_range(self):
        model = DeniUNet()
        num_params = sum(p.numel() for p in model.parameters())
        assert 100_000 <= num_params <= 200_000, (
            f"Parameter count {num_params:,} outside expected range 100K-200K"
        )

    def test_extra_repr_shows_params(self):
        model = DeniUNet()
        repr_str = repr(model)
        assert "params=" in repr_str

    def test_output_is_linear_hdr(self):
        """Output has no activation -- values can be negative or > 1."""
        model = DeniUNet()
        torch.manual_seed(42)
        x = torch.randn(1, 13, 64, 64) * 5.0
        y = model(x)
        # With random weights and large input, expect some values outside [0,1]
        assert y.min() < 0.0 or y.max() > 1.0


# ---------------------------------------------------------------------------
# ACES tonemapping tests
# ---------------------------------------------------------------------------

class TestAcesTonemap:
    def test_output_range(self):
        x = torch.rand(2, 3, 16, 16) * 10.0
        y = aces_tonemap(x)
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_black_maps_to_black(self):
        x = torch.zeros(1, 3, 4, 4)
        y = aces_tonemap(x)
        assert y.abs().max() < 1e-5

    def test_monotonic(self):
        """Brighter input should produce brighter output (per channel, on average)."""
        x_lo = torch.ones(1, 3, 4, 4) * 0.5
        x_hi = torch.ones(1, 3, 4, 4) * 2.0
        y_lo = aces_tonemap(x_lo)
        y_hi = aces_tonemap(x_hi)
        assert y_hi.mean() > y_lo.mean()

    def test_matches_glsl_reference(self):
        """Verify against manually computed GLSL reference values."""
        # Single pixel, white at intensity 1.0
        x = torch.ones(1, 3, 1, 1)
        y = aces_tonemap(x)
        # Each channel should be the same for uniform white input
        r, g, b = y[0, 0, 0, 0].item(), y[0, 1, 0, 0].item(), y[0, 2, 0, 0].item()
        # ACES(1,1,1) ~= (0.619, 0.619, 0.619) — verified against GLSL implementation
        assert abs(r - g) < 1e-4
        assert abs(g - b) < 1e-4
        assert 0.60 < r < 0.65, f"ACES(1,1,1) channel value {r} outside expected range"


# ---------------------------------------------------------------------------
# DenoiserLoss tests
# ---------------------------------------------------------------------------

class TestDenoiserLoss:
    @pytest.fixture(scope="class")
    def loss_fn(self):
        return DenoiserLoss(lambda_l1=1.0, lambda_perceptual=0.1)

    def test_computes_without_nan(self, loss_fn):
        pred = torch.rand(2, 3, 64, 64) * 2.0
        target = torch.rand(2, 3, 64, 64) * 2.0
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"
        assert loss.item() > 0.0

    def test_zero_loss_for_identical_inputs(self, loss_fn):
        x = torch.rand(1, 3, 64, 64) * 2.0
        loss = loss_fn(x, x.clone())
        assert loss.item() < 1e-5, f"Loss for identical inputs should be ~0, got {loss.item()}"

    def test_gradient_flows(self, loss_fn):
        pred = (torch.rand(1, 3, 64, 64) * 2.0).requires_grad_()
        target = torch.rand(1, 3, 64, 64) * 2.0
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0.0

    def test_loss_decreases_over_steps(self, loss_fn):
        """Sanity check: optimize predicted toward target for 10 steps."""
        torch.manual_seed(0)
        target = torch.rand(1, 3, 64, 64) * 2.0
        pred = torch.rand(1, 3, 64, 64) * 2.0
        pred = nn.Parameter(pred)
        optimizer = torch.optim.Adam([pred], lr=0.05)

        initial_loss = loss_fn(pred, target).item()
        for _ in range(10):
            optimizer.zero_grad()
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

        final_loss = loss_fn(pred, target).item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.6f} -> {final_loss:.6f}"
        )
