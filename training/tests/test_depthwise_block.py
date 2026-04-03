"""Tests for DepthwiseSeparableConvBlock."""

import torch
import torch.nn as nn

from deni_train.models.blocks import (
    ConvBlock,
    DepthwiseSeparableConvBlock,
    DownBlock,
    UpBlock,
)


def test_output_shape():
    block = DepthwiseSeparableConvBlock(16, 32)
    x = torch.randn(2, 16, 64, 64)
    y = block(x)
    assert y.shape == (2, 32, 64, 64)


def test_output_finite():
    block = DepthwiseSeparableConvBlock(16, 32)
    x = torch.randn(2, 16, 64, 64)
    y = block(x)
    assert torch.isfinite(y).all()


def test_parameter_count():
    block = DepthwiseSeparableConvBlock(16, 32)
    count = sum(p.numel() for p in block.parameters())
    # depthwise weight: 16*1*3*3 = 144 (no bias)
    # pointwise weight: 32*16*1*1 = 512, bias: 32
    # norm weight: 32, bias: 32
    # total: 752
    assert count == 752


def test_fewer_params_than_convblock():
    ds = DepthwiseSeparableConvBlock(16, 32)
    conv = ConvBlock(16, 32)
    ds_count = sum(p.numel() for p in ds.parameters())
    conv_count = sum(p.numel() for p in conv.parameters())
    assert ds_count < conv_count


def test_same_in_out_channels():
    block = DepthwiseSeparableConvBlock(32, 32)
    x = torch.randn(1, 32, 16, 16)
    y = block(x)
    assert y.shape == (1, 32, 16, 16)


def test_determinism():
    torch.manual_seed(42)
    a = DepthwiseSeparableConvBlock(16, 16)
    torch.manual_seed(42)
    b = DepthwiseSeparableConvBlock(16, 16)
    x = torch.randn(2, 16, 32, 32)
    ya = a(x)
    yb = b(x)
    assert torch.equal(ya, yb)


def test_gradient_flow():
    block = DepthwiseSeparableConvBlock(16, 32)
    x = torch.randn(2, 16, 32, 32)
    y = block(x)
    loss = y.sum()
    loss.backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_dropin_downblock():
    """DepthwiseSeparableConvBlock produces same shapes as ConvBlock in DownBlock."""
    # Standard DownBlock
    std = DownBlock(16, 32)
    x = torch.randn(1, 16, 64, 64)
    std_pool, std_skip = std(x)

    # DownBlock with DepthwiseSeparableConvBlock substituted
    ds = DownBlock(16, 32)
    ds.conv1 = DepthwiseSeparableConvBlock(16, 32)
    ds.conv2 = DepthwiseSeparableConvBlock(32, 32)
    ds_pool, ds_skip = ds(x)

    assert ds_pool.shape == std_pool.shape
    assert ds_skip.shape == std_skip.shape


def test_dropin_upblock():
    """DepthwiseSeparableConvBlock produces same shapes as ConvBlock in UpBlock."""
    # Standard UpBlock
    std = UpBlock(64, 32, 32)
    x = torch.randn(1, 64, 16, 16)
    skip = torch.randn(1, 32, 32, 32)
    std_out = std(x, skip)

    # UpBlock with DepthwiseSeparableConvBlock substituted
    ds = UpBlock(64, 32, 32)
    ds.conv1 = DepthwiseSeparableConvBlock(64 + 32, 32)
    ds.conv2 = DepthwiseSeparableConvBlock(32, 32)
    ds_out = ds(x, skip)

    assert ds_out.shape == std_out.shape
