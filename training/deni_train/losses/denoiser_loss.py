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
    """L1 + VGG perceptual + combined-radiance + hue loss in ACES-tonemapped space.

    Multi-space loss for demodulated denoising:
    - L1 loss on demodulated irradiance (direct supervision in network output space)
    - VGG perceptual loss on remodulated radiance (evaluates final visual quality)
    - Combined-radiance L1 loss on remodulated radiance (penalizes what the viewer sees)
    - Cosine similarity hue loss on remodulated radiance (penalizes hue rotation)
    All are computed in ACES-tonemapped space.
    VGG inputs are normalized with ImageNet statistics (standard LPIPS approach).
    """

    def __init__(self, lambda_l1: float = 1.0, lambda_perceptual: float = 0.1,
                 lambda_radiance_l1: float = 0.0, lambda_hue: float = 0.0,
                 lambda_log_l1: float = 0.0, lambda_log_radiance_l1: float = 0.0):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_radiance_l1 = lambda_radiance_l1
        self.lambda_hue = lambda_hue
        self.lambda_log_l1 = lambda_log_l1
        self.lambda_log_radiance_l1 = lambda_log_radiance_l1
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
                albedo_d: torch.Tensor, albedo_s: torch.Tensor
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute multi-space loss for demodulated denoising.

        Args:
            predicted: (B, 6, H, W) predicted demodulated irradiance
                       (channels 0-2: diffuse, 3-5: specular).
            target: (B, 6, H, W) ground truth demodulated irradiance.
            albedo_d: (B, 3, H, W) diffuse albedo.
            albedo_s: (B, 3, H, W) specular albedo.

        Returns:
            total_loss: Weighted sum of all loss components.
            components: Dict of unweighted per-component losses for logging.
        """
        # L1 loss on demodulated irradiance (network output space)
        _need_aces = (self.lambda_l1 > 0 or self.lambda_perceptual > 0
                      or self.lambda_radiance_l1 > 0 or self.lambda_hue > 0)

        # Remodulate to radiance (needed by both ACES and log1p radiance losses)
        _DEMOD_EPS = 1e-3
        albedo_d_clamped = torch.clamp(albedo_d, min=_DEMOD_EPS)
        albedo_s_clamped = torch.clamp(albedo_s, min=_DEMOD_EPS)
        pred_radiance = predicted[:, :3] * albedo_d_clamped + predicted[:, 3:6] * albedo_s_clamped
        tgt_radiance = target[:, :3] * albedo_d_clamped + target[:, 3:6] * albedo_s_clamped

        zero = torch.tensor(0.0, device=predicted.device)

        if _need_aces:
            pred_tm = torch.cat([aces_tonemap(predicted[:, :3]),
                                 aces_tonemap(predicted[:, 3:6])], dim=1)
            tgt_tm = torch.cat([aces_tonemap(target[:, :3]),
                                aces_tonemap(target[:, 3:6])], dim=1)
            l1_loss = F.l1_loss(pred_tm, tgt_tm)

            pred_rad_tm = aces_tonemap(pred_radiance)
            tgt_rad_tm = aces_tonemap(tgt_radiance)

            # VGG perceptual loss
            if self.lambda_perceptual > 0:
                pred_vgg = self._normalize_imagenet(pred_rad_tm)
                tgt_vgg = self._normalize_imagenet(tgt_rad_tm)
                with torch.no_grad():
                    tgt_features = self.vgg(tgt_vgg)
                pred_features = self.vgg(pred_vgg)
                perceptual_loss = sum(
                    F.l1_loss(pf, tf.detach()) for pf, tf in zip(pred_features, tgt_features)
                )
            else:
                perceptual_loss = zero

            radiance_l1_loss = F.l1_loss(pred_rad_tm, tgt_rad_tm)

            # Cosine similarity hue loss
            if self.lambda_hue > 0:
                _HUE_BRIGHTNESS_THRESHOLD = 0.05
                _HUE_EPS = 1e-3
                tgt_brightness = tgt_rad_tm.mean(dim=1, keepdim=True)
                bright_mask = (tgt_brightness > _HUE_BRIGHTNESS_THRESHOLD).float()
                bright_count = bright_mask.sum().clamp(min=1.0)
                cos_sim = F.cosine_similarity(
                    pred_rad_tm + _HUE_EPS, tgt_rad_tm + _HUE_EPS, dim=1
                )
                hue_loss = ((1.0 - cos_sim) * bright_mask.squeeze(1)).sum() / bright_count
            else:
                hue_loss = zero
        else:
            l1_loss = zero
            perceptual_loss = zero
            radiance_l1_loss = zero
            hue_loss = zero

        # Log-space L1 on demodulated irradiance
        if self.lambda_log_l1 > 0:
            log_pred = torch.log1p(predicted.clamp(min=0))
            log_tgt = torch.log1p(target.clamp(min=0))
            log_l1_loss = F.l1_loss(log_pred, log_tgt)
        else:
            log_l1_loss = zero

        # Log-space L1 on remodulated radiance (what the viewer sees, in log space)
        if self.lambda_log_radiance_l1 > 0:
            log_pred_rad = torch.log1p(pred_radiance.clamp(min=0))
            log_tgt_rad = torch.log1p(tgt_radiance.clamp(min=0))
            log_radiance_l1_loss = F.l1_loss(log_pred_rad, log_tgt_rad)
        else:
            log_radiance_l1_loss = zero

        components = {
            "l1": l1_loss.detach(),
            "perceptual": perceptual_loss.detach(),
            "radiance_l1": radiance_l1_loss.detach(),
            "hue": hue_loss.detach(),
            "log_l1": log_l1_loss.detach(),
            "log_radiance_l1": log_radiance_l1_loss.detach(),
        }

        total = (self.lambda_l1 * l1_loss
                 + self.lambda_perceptual * perceptual_loss
                 + self.lambda_radiance_l1 * radiance_l1_loss
                 + self.lambda_hue * hue_loss
                 + self.lambda_log_l1 * log_l1_loss
                 + self.lambda_log_radiance_l1 * log_radiance_l1_loss)
        return total, components
