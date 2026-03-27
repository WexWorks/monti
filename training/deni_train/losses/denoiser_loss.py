"""Composite denoiser loss: L1 + VGG perceptual, in ACES-tonemapped space."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from ..utils.tonemapping import aces_tonemap


class VggFeatureExtractor(nn.Module):
    """Frozen VGG-16 feature extractor at relu1_2, relu2_2, relu3_3."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = vgg.features

        # VGG-16 features layout:
        #   relu1_2 = features[:4]   (Conv, ReLU, Conv, ReLU)
        #   relu2_2 = features[4:9]  (MaxPool, Conv, ReLU, Conv, ReLU)
        #   relu3_3 = features[9:16] (MaxPool, Conv, ReLU, Conv, ReLU, Conv, ReLU)
        self.slice1 = nn.Sequential(*list(features[:4]))
        self.slice2 = nn.Sequential(*list(features[4:9]))
        self.slice3 = nn.Sequential(*list(features[9:16]))

        # Freeze all VGG parameters (gradients still flow through inputs)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "VggFeatureExtractor":
        # Always stay in eval mode (no trainable params, no batch norm to worry about)
        return super().train(False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        return [h1, h2, h3]


class DenoiserLoss(nn.Module):
    """L1 + VGG perceptual loss in ACES-tonemapped space.

    Dual-space loss for demodulated denoising:
    - L1 loss on demodulated irradiance (direct supervision in network output space)
    - VGG perceptual loss on remodulated radiance (evaluates final visual quality)
    Both are computed in ACES-tonemapped space.
    VGG inputs are normalized with ImageNet statistics (standard LPIPS approach).
    """

    def __init__(self, lambda_l1: float = 1.0, lambda_perceptual: float = 0.1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.vgg = VggFeatureExtractor()

        # ImageNet normalization constants (registered as buffers for device transfer)
        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.imagenet_mean) / self.imagenet_std

    def forward(self, predicted: torch.Tensor, target: torch.Tensor,
                albedo_d: torch.Tensor, albedo_s: torch.Tensor,
                hit_mask: torch.Tensor) -> torch.Tensor:
        """Compute dual-space loss for demodulated denoising.

        Args:
            predicted: (B, 6, H, W) predicted demodulated irradiance
                       (channels 0-2: diffuse, 3-5: specular).
            target: (B, 6, H, W) ground truth demodulated irradiance.
            albedo_d: (B, 3, H, W) diffuse albedo.
            albedo_s: (B, 3, H, W) specular albedo.
            hit_mask: (B, 1, H, W) binary mask. 1 = valid geometry pixel,
                      0 = background/miss.
        """
        # L1 loss on demodulated irradiance (network output space)
        pred_tm = torch.cat([aces_tonemap(predicted[:, :3]),
                             aces_tonemap(predicted[:, 3:6])], dim=1)
        tgt_tm = torch.cat([aces_tonemap(target[:, :3]),
                            aces_tonemap(target[:, 3:6])], dim=1)

        diff = (pred_tm - tgt_tm).abs()
        valid_count = hit_mask.sum() * pred_tm.size(1)
        l1_loss = (diff * hit_mask).sum() / valid_count.clamp(min=1.0)

        # Remodulate to radiance for perceptual loss
        pred_radiance = predicted[:, :3] * albedo_d + predicted[:, 3:6] * albedo_s
        tgt_radiance = target[:, :3] * albedo_d + target[:, 3:6] * albedo_s

        pred_rad_tm = aces_tonemap(pred_radiance)
        tgt_rad_tm = aces_tonemap(tgt_radiance)

        # Zero out background for VGG so it contributes no perceptual signal
        pred_vgg = self._normalize_imagenet(pred_rad_tm * hit_mask)
        tgt_vgg = self._normalize_imagenet(tgt_rad_tm * hit_mask)

        with torch.no_grad():
            tgt_features = self.vgg(tgt_vgg)

        pred_features = self.vgg(pred_vgg)

        perceptual_loss = sum(
            F.l1_loss(pf, tf.detach()) for pf, tf in zip(pred_features, tgt_features)
        )

        return self.lambda_l1 * l1_loss + self.lambda_perceptual * perceptual_loss
