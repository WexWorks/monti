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

    Both predicted and target are tonemapped via ACES before computing losses.
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

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Tonemap to [0,1] perceptual space
        pred_tm = aces_tonemap(predicted)
        tgt_tm = aces_tonemap(target)

        # L1 loss in tonemapped space
        l1_loss = F.l1_loss(pred_tm, tgt_tm)

        # Perceptual loss with ImageNet normalization
        pred_vgg = self._normalize_imagenet(pred_tm)
        tgt_vgg = self._normalize_imagenet(tgt_tm)

        with torch.no_grad():
            tgt_features = self.vgg(tgt_vgg)

        pred_features = self.vgg(pred_vgg)

        perceptual_loss = sum(
            F.l1_loss(pf, tf.detach()) for pf, tf in zip(pred_features, tgt_features)
        )

        return self.lambda_l1 * l1_loss + self.lambda_perceptual * perceptual_loss
