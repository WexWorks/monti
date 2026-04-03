"""U-Net building blocks: ConvBlock, DepthwiseSeparableConvBlock, DownBlock, UpBlock."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d(3x3, pad=1) + GroupNorm + LeakyReLU(0.01)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(
            self.conv.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu"
        )
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparableConvBlock(nn.Module):
    """Depthwise 3×3 + Pointwise 1×1 + GroupNorm + LeakyReLU(0.01)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(
            self.depthwise.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu"
        )
        nn.init.kaiming_normal_(
            self.pointwise.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu"
        )
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.norm(x))


class DownBlock(nn.Module):
    """Two ConvBlocks followed by MaxPool2d(2)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pooled_output, skip_connection)."""
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x


class UpBlock(nn.Module):
    """Bilinear upsample + skip concatenation + two ConvBlocks.

    in_ch is the channel count from the layer below (pre-concatenation).
    The first ConvBlock receives in_ch + skip_ch channels after concatenation.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvBlock(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)
