"""DeniUNet -- small 3-level U-Net for single-frame denoising."""

import torch
import torch.nn as nn

from .blocks import ConvBlock, DownBlock, UpBlock


class DeniUNet(nn.Module):
    """Small U-Net: 19-channel G-buffer input -> 6-channel denoised irradiance output.

    Input: 19 channels (demodulated diffuse/specular irradiance + normals + roughness
           + depth + motion + diffuse albedo + specular albedo).
    Output: 6 channels (denoised diffuse irradiance RGB + denoised specular irradiance RGB).

    Architecture:
        Encoder:  in->16 (skip_0) -> 16->32 (skip_1) -> MaxPool
        Bottleneck: 32->64->64 at H/4 x W/4
        Decoder:  64+32->32 -> 32+16->16
        Output:   Conv1x1(16->6), no activation (linear HDR)
    """

    def __init__(
        self, in_channels: int = 19, out_channels: int = 6, base_channels: int = 16
    ):
        super().__init__()
        c = base_channels

        # Encoder
        self.down0 = DownBlock(in_channels, c)
        self.down1 = DownBlock(c, c * 2)

        # Bottleneck (H/4 x W/4)
        self.bottleneck1 = ConvBlock(c * 2, c * 4)
        self.bottleneck2 = ConvBlock(c * 4, c * 4)

        # Decoder
        self.up1 = UpBlock(c * 4, c * 2, c * 2)
        self.up0 = UpBlock(c * 2, c, c)

        # Output projection -- no activation (linear HDR output)
        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)
        nn.init.kaiming_normal_(
            self.out_conv.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu"
        )
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x, skip_0 = self.down0(x)
        x, skip_1 = self.down1(x)

        # Bottleneck
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)

        # Decoder
        x = self.up1(x, skip_1)
        x = self.up0(x, skip_0)

        return self.out_conv(x)

    def extra_repr(self) -> str:
        num_params = sum(p.numel() for p in self.parameters())
        return f"params={num_params:,}"
