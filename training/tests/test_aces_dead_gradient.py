"""Numerical proof that log1p loss fixes the ACES dead-gradient trap.

Reproduces the exact failure mode where the model outputs near-zero and proves:
1. ACES L1 gradient is literally zero at the model's operating point.
2. log1p L1 gradient is non-zero and correctly directed.
3. Adam optimizer escapes with log_l1, stays stuck without it.

Values below are from the real model at epoch 33 (probe_model.py output).
"""

import pytest
import torch
import torch.nn.functional as F

from deni_train.losses.denoiser_loss import DenoiserLoss
from deni_train.utils.tonemapping import aces_tonemap

# Real values from the stuck model at epoch 33:
_PRED_MEAN = 0.001     # pred demod mean
_TGT_MEAN = 13.5       # target demod mean
_ALBEDO_D = 0.087      # diffuse albedo mean
_ALBEDO_S = 0.043      # specular albedo mean


class TestAcesGradientDead:
    """ACES has zero gradient at the model's actual remodulated radiance level."""

    def test_aces_gradient_zero_at_pred_radiance(self):
        """The model's remodulated radiance (pred*albedo ≈ 0.0001) gets zero
        ACES gradient — this is why the model is stuck."""
        radiance = _PRED_MEAN * _ALBEDO_D  # ≈ 0.000087
        x = torch.full((1, 3, 4, 4), radiance, requires_grad=True)
        y = aces_tonemap(x)
        y.sum().backward()
        assert x.grad.abs().max().item() == 0.0, \
            f"Expected exactly zero ACES gradient at radiance={radiance}, " \
            f"got {x.grad.abs().max().item()}"

    def test_aces_gradient_zero_at_demod_level(self):
        """Even at the demodulated level (pred ≈ 0.001), ACES gradient is zero."""
        x = torch.full((1, 3, 4, 4), _PRED_MEAN, requires_grad=True)
        y = aces_tonemap(x)
        y.sum().backward()
        assert x.grad.abs().max().item() == 0.0

    def test_aces_gradient_alive_at_target_level(self):
        """Sanity: ACES does have gradient at the target's level."""
        x = torch.full((1, 3, 4, 4), _TGT_MEAN, requires_grad=True)
        y = aces_tonemap(x)
        y.sum().backward()
        assert x.grad.abs().max().item() > 0.001


class TestLog1pGradientAlive:
    """log1p has gradient everywhere, including at zero."""

    def test_log1p_gradient_at_zero(self):
        """log1p'(0) = 1/(1+0) = 1.0 — maximum gradient, never dead."""
        x = torch.tensor(0.0, requires_grad=True)
        torch.log1p(x).backward()
        assert x.grad.item() == pytest.approx(1.0, abs=1e-6)

    def test_log1p_gradient_at_pred_level(self):
        """log1p'(0.001) ≈ 1.0 — strong gradient at model operating point."""
        x = torch.tensor(_PRED_MEAN, requires_grad=True)
        torch.log1p(x).backward()
        assert x.grad.item() > 0.99


class TestPureL1GradientComparison:
    """Compare raw L1 gradients in ACES vs log1p space — no VGG, no hue,
    just the fundamental gradient difference."""

    def test_aces_l1_gradient_exactly_zero(self):
        """L1(aces(pred), aces(target)) has zero gradient when pred ≈ 0."""
        pred = torch.full((1, 3, 32, 32), _PRED_MEAN, requires_grad=True)
        target = torch.full((1, 3, 32, 32), _TGT_MEAN)
        loss = F.l1_loss(aces_tonemap(pred), aces_tonemap(target))
        loss.backward()
        assert pred.grad.abs().max().item() == 0.0

    def test_log_l1_gradient_nonzero(self):
        """L1(log1p(pred), log1p(target)) has non-zero gradient when pred ≈ 0."""
        pred = torch.full((1, 6, 32, 32), _PRED_MEAN, requires_grad=True)
        target = torch.full((1, 6, 32, 32), _TGT_MEAN)
        loss = F.l1_loss(torch.log1p(pred.clamp(min=0)), torch.log1p(target))
        loss.backward()
        # Gradient = -1/(1+x) / N, each element has |grad| ≈ 1/N ≈ 1/6144
        assert pred.grad.abs().max().item() > 1e-5

    def test_log_l1_gradient_direction(self):
        """Gradient points toward increasing pred (pred < target)."""
        pred = torch.full((1, 6, 32, 32), _PRED_MEAN, requires_grad=True)
        target = torch.full((1, 6, 32, 32), _TGT_MEAN)
        loss = F.l1_loss(torch.log1p(pred.clamp(min=0)), torch.log1p(target))
        loss.backward()
        # All gradients should be negative: descent direction = increase pred
        assert pred.grad.mean().item() < 0


class TestDenoiserLossGradient:
    """Full DenoiserLoss gradient comparison at the model's stuck state."""

    @staticmethod
    def _compute_gradient(lambda_log_l1: float, include_hue: bool = False):
        """Return gradient norm from DenoiserLoss with given config."""
        loss_fn = DenoiserLoss(
            lambda_l1=1.0,
            lambda_perceptual=0.0,   # skip VGG for speed + isolation
            lambda_radiance_l1=0.5,
            lambda_hue=0.25 if include_hue else 0.0,
            lambda_log_l1=lambda_log_l1,
        ).eval()
        B, H, W = 2, 32, 32
        # Use uniform values at the model's stuck operating point — avoids
        # random outliers that straddle the ACES activation threshold.
        pred = torch.full((B, 6, H, W), _PRED_MEAN)
        pred.requires_grad_(True)
        target = torch.full((B, 6, H, W), _TGT_MEAN)
        albedo_d = torch.full((B, 3, H, W), _ALBEDO_D)
        albedo_s = torch.full((B, 3, H, W), _ALBEDO_S)
        loss, _ = loss_fn(pred, target, albedo_d, albedo_s)
        loss.backward()
        return pred.grad.norm().item()

    def test_aces_only_no_hue_gradient_zero(self):
        """Without hue and without log_l1, gradient is exactly zero
        (pure ACES L1 + radiance_l1 in dead zone)."""
        grad_norm = self._compute_gradient(lambda_log_l1=0.0, include_hue=False)
        assert grad_norm < 1e-6, \
            f"Expected zero gradient from ACES-only (no hue), got {grad_norm:.8f}"

    def test_aces_with_hue_has_tiny_gradient(self):
        """Hue loss leaks a small gradient through its epsilon, but it's not
        useful for escaping the magnitude trap."""
        grad_norm = self._compute_gradient(lambda_log_l1=0.0, include_hue=True)
        # Hue gives a tiny gradient (direction: align hue, not increase magnitude)
        assert grad_norm < 0.01

    def test_log_l1_gradient_much_larger(self):
        """log_l1 provides much more gradient than hue leakage alone."""
        grad_aces = self._compute_gradient(lambda_log_l1=0.0, include_hue=True)
        grad_log = self._compute_gradient(lambda_log_l1=1.0, include_hue=True)
        assert grad_log > grad_aces * 3, \
            f"Expected log_l1 gradient >> ACES+hue gradient, " \
            f"got log={grad_log:.6f} vs aces={grad_aces:.6f}"


class TestAdamOptimization:
    """The definitive proof: Adam optimizer escapes dead zone with log_l1
    but stays stuck without it."""

    @staticmethod
    def _run_adam(use_log_l1: bool, steps: int = 200, lr: float = 0.01):
        """Optimize a prediction tensor with Adam. Returns (initial_mean, final_mean)."""
        pred = torch.nn.Parameter(
            torch.full((1, 6, 16, 16), _PRED_MEAN)
        )
        target = torch.full((1, 6, 16, 16), _TGT_MEAN)
        optimizer = torch.optim.Adam([pred], lr=lr)

        initial_mean = pred.data.mean().item()

        for _ in range(steps):
            optimizer.zero_grad()
            # ACES L1 on demodulated (same as DenoiserLoss.lambda_l1)
            pred_tm = torch.cat([aces_tonemap(pred[:, :3]),
                                 aces_tonemap(pred[:, 3:6])], dim=1)
            tgt_tm = torch.cat([aces_tonemap(target[:, :3]),
                                aces_tonemap(target[:, 3:6])], dim=1)
            loss = F.l1_loss(pred_tm, tgt_tm)

            if use_log_l1:
                log_loss = F.l1_loss(torch.log1p(pred.clamp(min=0)),
                                     torch.log1p(target))
                loss = loss + log_loss

            loss.backward()
            optimizer.step()

        return initial_mean, pred.data.mean().item()

    def test_adam_aces_only_stays_stuck(self):
        """Without log_l1, Adam cannot escape the dead zone in 200 steps."""
        initial, final = self._run_adam(use_log_l1=False, steps=200)
        movement = abs(final - initial)
        assert movement < 0.01, \
            f"Expected ACES-only to stay stuck, but moved {movement:.4f} " \
            f"({initial:.4f} → {final:.4f})"

    def test_adam_log_l1_escapes(self):
        """With log_l1, Adam escapes the dead zone within 200 steps."""
        initial, final = self._run_adam(use_log_l1=True, steps=200)
        assert final > 1.0, \
            f"Expected pred >> 1.0 with log_l1, got {final:.4f} (from {initial:.4f})"

    def test_adam_log_l1_approaches_target(self):
        """After 500 steps, log_l1 pushes pred well above 1.0 toward target.
        (Convergence slows as log1p gradient = 1/(1+x) shrinks with growing x.)"""
        _, final = self._run_adam(use_log_l1=True, steps=500)
        assert final > 2.0, \
            f"Expected pred > 2.0 after 500 Adam steps, got {final:.4f}"

    def test_escape_ratio(self):
        """log_l1 makes at least 100x more progress than ACES-only."""
        _, aces_final = self._run_adam(use_log_l1=False, steps=200)
        _, log_final = self._run_adam(use_log_l1=True, steps=200)

        aces_progress = max(abs(aces_final - _PRED_MEAN), 1e-10)
        log_progress = abs(log_final - _PRED_MEAN)

        ratio = log_progress / aces_progress
        assert ratio > 100, \
            f"Expected log_l1 to make >100x more progress, got {ratio:.0f}x " \
            f"(ACES: {aces_final:.4f}, log: {log_final:.4f})"
