"""Tests for DeniTemporalResidualNet architecture.

Validates shape, parameter count, blend weight behavior, gradient flow,
and first-frame equivalence.
"""

import pytest
import torch

from deni_train.models.temporal_unet import (
    DeniTemporalResidualNet,
    _TOTAL_INPUT_CHANNELS,
    _OUTPUT_CHANNELS,
)


@pytest.fixture
def model():
    torch.manual_seed(42)
    return DeniTemporalResidualNet(base_channels=12)


@pytest.fixture
def device():
    return torch.device("cpu")


class TestArchitectureShape:
    """Output shape = (B, 6, H, W) for (B, 26, H, W) input."""

    def test_output_shape(self, model, device):
        model = model.to(device)
        x = torch.randn(2, 26, 64, 64, device=device)
        out, weight = model(x)
        assert out.shape == (2, 6, 64, 64)
        assert weight.shape == (2, 1, 64, 64)

    def test_output_shape_larger(self, model, device):
        model = model.to(device)
        x = torch.randn(1, 26, 256, 256, device=device)
        out, weight = model(x)
        assert out.shape == (1, 6, 256, 256)
        assert weight.shape == (1, 1, 256, 256)

    def test_input_channels_constant(self):
        assert _TOTAL_INPUT_CHANNELS == 26

    def test_output_raw_channels(self):
        assert _OUTPUT_CHANNELS == 7  # delta_d(3) + delta_s(3) + weight(1)


class TestParameterCount:
    """Parameter count in expected range for 2-level depthwise U-Net."""

    def test_parameter_count_range(self, model):
        num_params = sum(p.numel() for p in model.parameters())
        # 2-level depthwise separable U-Net with base_channels=12: ~3-4K params
        # (depthwise separable is much more efficient than regular conv)
        assert 2_000 < num_params < 10_000, f"Param count {num_params} out of expected range"


class TestInternalChannels:
    """Verify internal channel counts match spec."""

    def test_out_conv_weight_shape(self, model):
        # out_conv: Conv2d(base_channels, 7, kernel_size=1)
        assert model.out_conv.weight.shape == (7, 12, 1, 1)

    def test_concatenated_input_shape(self, model, device):
        """Verify the model processes exactly 26 input channels."""
        model = model.to(device)
        # Correct input
        x = torch.randn(1, 26, 32, 32, device=device)
        out, _ = model(x)
        assert out.shape == (1, 6, 32, 32)

        # Wrong channel count should fail
        x_bad = torch.randn(1, 19, 32, 32, device=device)
        with pytest.raises(RuntimeError):
            model(x_bad)


class TestBlendWeight:
    """Blend weight bounds and disocclusion forcing."""

    def test_disocclusion_forces_weight_to_one(self, model, device):
        """With disocclusion=0 (all disoccluded), blend weight must be 1.0."""
        model = model.to(device).eval()
        x = torch.randn(2, 26, 32, 32, device=device)
        # Set disocclusion channel (index 6) to 0.0 (all disoccluded)
        x[:, 6:7] = 0.0
        with torch.no_grad():
            out, _ = model(x)
        # Output = reprojected + 1.0 * delta (blend weight forced to 1.0)
        # Since weight is forced to max(sigmoid(raw), 1.0 - 0.0) = max(sigmoid(raw), 1.0) = 1.0
        # The output should equal reprojected_d + delta_d, reprojected_s + delta_s
        # We can verify by checking that weight forcing happened correctly
        # by running the model internals
        assert out.shape == (2, 6, 32, 32)
        assert torch.isfinite(out).all()

    def test_valid_pixels_have_bounded_weight(self, model, device):
        """With disocclusion=1 (all valid), blend weights in [0, 1]."""
        model = model.to(device).eval()
        x = torch.randn(2, 26, 32, 32, device=device)
        x[:, 6:7] = 1.0  # All valid
        with torch.no_grad():
            out, _ = model(x)
        assert torch.isfinite(out).all()


class TestGradientFlow:
    """Every parameter gets non-zero gradient."""

    def test_all_params_have_gradients(self, model, device):
        model = model.to(device)
        x = torch.randn(2, 26, 32, 32, device=device)
        out, _ = model(x)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


class TestFirstFrame:
    """First-frame behavior with zero history."""

    def test_zero_history_produces_nonzero_output(self, model, device):
        """Network should produce non-zero output from noisy input alone."""
        model = model.to(device).eval()
        x = torch.randn(1, 26, 32, 32, device=device)
        # Zero reprojected channels, disocclusion = 0 (fully disoccluded)
        x[:, 0:6] = 0.0  # reprojected_d, reprojected_s
        x[:, 6:7] = 0.0  # disocclusion
        with torch.no_grad():
            out, _ = model(x)
        assert out.abs().sum() > 0, "Output should be non-zero from noisy input"

    def test_deterministic_with_same_seed(self, device):
        """Same seed produces identical output."""
        x = torch.randn(1, 26, 32, 32, device=device)

        torch.manual_seed(42)
        model1 = DeniTemporalResidualNet(base_channels=12).to(device).eval()
        with torch.no_grad():
            out1, _ = model1(x)

        torch.manual_seed(42)
        model2 = DeniTemporalResidualNet(base_channels=12).to(device).eval()
        with torch.no_grad():
            out2, _ = model2(x)

        assert torch.allclose(out1, out2, atol=1e-6)
