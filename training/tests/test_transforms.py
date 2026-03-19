"""Tests for RandomRotation180 and ExposureJitter transforms."""

import torch
import pytest

from deni_train.data.transforms import ExposureJitter, RandomRotation180


class TestRandomRotation180:
    def test_motion_vectors_negated_on_rotation(self):
        """Motion vector X (ch 11) and Y (ch 12) are negated under 180° rotation."""
        inp = torch.ones(13, 4, 4, dtype=torch.float16)
        inp[11] = 2.0  # motion X
        inp[12] = 3.0  # motion Y
        tgt = torch.ones(3, 4, 4, dtype=torch.float16)

        rot = RandomRotation180(p=1.0)
        inp_r, _ = rot((inp, tgt))

        assert (inp_r[11] == -2.0).all()
        assert (inp_r[12] == -3.0).all()

    def test_normals_unchanged_on_rotation(self):
        """World-space normals (ch 6-8) are invariant under 180° rotation."""
        inp = torch.zeros(13, 4, 4, dtype=torch.float16)
        inp[6] = 0.5   # normal X
        inp[7] = -0.3  # normal Y
        inp[8] = 0.8   # normal Z
        tgt = torch.ones(3, 4, 4, dtype=torch.float16)

        rot = RandomRotation180(p=1.0)
        inp_r, _ = rot((inp, tgt))

        # Values are the same (just spatially rotated), not channel-negated
        torch.testing.assert_close(inp_r[6].sum(), inp[6].sum())
        torch.testing.assert_close(inp_r[7].sum(), inp[7].sum())
        torch.testing.assert_close(inp_r[8].sum(), inp[8].sum())

    def test_spatial_rotation(self):
        """Pixels are spatially rotated 180° (equivalent to flipping both axes)."""
        inp = torch.zeros(13, 4, 4, dtype=torch.float16)
        # Place a marker at top-left corner
        inp[0, 0, 0] = 42.0
        tgt = torch.zeros(3, 4, 4, dtype=torch.float16)
        tgt[0, 0, 0] = 99.0

        rot = RandomRotation180(p=1.0)
        inp_r, tgt_r = rot((inp, tgt))

        # After 180° rotation, top-left moves to bottom-right
        assert inp_r[0, 3, 3].item() == 42.0
        assert inp_r[0, 0, 0].item() == 0.0
        assert tgt_r[0, 3, 3].item() == 99.0
        assert tgt_r[0, 0, 0].item() == 0.0

    def test_no_rotation_preserves_data(self):
        inp = torch.ones(13, 4, 4, dtype=torch.float16) * 3.0
        tgt = torch.ones(3, 4, 4, dtype=torch.float16) * 5.0

        rot = RandomRotation180(p=0.0)
        inp_r, tgt_r = rot((inp, tgt))

        torch.testing.assert_close(inp_r, inp)
        torch.testing.assert_close(tgt_r, tgt)

    def test_guide_channels_unchanged(self):
        """Roughness (ch 9) and depth (ch 10) are scalar invariants."""
        inp = torch.ones(13, 4, 4, dtype=torch.float16)
        inp[9] = 0.7   # roughness
        inp[10] = 5.0  # depth
        tgt = torch.ones(3, 4, 4, dtype=torch.float16)

        rot = RandomRotation180(p=1.0)
        inp_r, _ = rot((inp, tgt))

        # Values are the same (spatially rotated, but all uniform so unchanged)
        torch.testing.assert_close(inp_r[9], inp[9])
        torch.testing.assert_close(inp_r[10], inp[10])


class TestExposureJitter:
    def test_radiance_scaled_guide_unchanged(self):
        """Input radiance and target radiance are scaled; guide channels are not."""
        inp = torch.ones(13, 4, 4, dtype=torch.float16)
        tgt = torch.ones(3, 4, 4, dtype=torch.float16)

        # Force jitter = 1.0 (scale = 2.0) by using a narrow range
        ej = ExposureJitter(range=(1.0, 1.0))
        inp_j, tgt_j = ej((inp, tgt))

        # Input diffuse RGB (ch 0-2) and specular RGB (ch 3-5) should be ~2.0
        torch.testing.assert_close(
            inp_j[:3], torch.full((3, 4, 4), 2.0, dtype=torch.float16), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            inp_j[3:6], torch.full((3, 4, 4), 2.0, dtype=torch.float16), atol=1e-2, rtol=1e-2)

        # Target should be ~2.0
        torch.testing.assert_close(
            tgt_j, torch.full((3, 4, 4), 2.0, dtype=torch.float16), atol=1e-2, rtol=1e-2)

        # Guide channels (ch 6-12) should be unchanged
        torch.testing.assert_close(inp_j[6:], inp[6:])

    def test_zero_jitter_preserves_data(self):
        """Jitter = 0 means scale = 1.0, so input and target are unchanged."""
        inp = torch.full((13, 4, 4), 3.0, dtype=torch.float16)
        tgt = torch.full((3, 4, 4), 5.0, dtype=torch.float16)

        ej = ExposureJitter(range=(0.0, 0.0))
        inp_j, tgt_j = ej((inp, tgt))

        torch.testing.assert_close(inp_j, inp)
        torch.testing.assert_close(tgt_j, tgt)

    def test_overflow_protection_no_inf(self):
        """Values near FP16 max with positive jitter produce no Inf."""
        inp = torch.zeros(13, 4, 4, dtype=torch.float16)
        inp[:6] = 60000.0  # near FP16 max (65504)
        tgt = torch.full((3, 4, 4), 60000.0, dtype=torch.float16)

        ej = ExposureJitter(range=(1.0, 1.0))  # scale = 2.0
        inp_j, tgt_j = ej((inp, tgt))

        assert torch.isfinite(inp_j).all(), "Inf found in input after ExposureJitter"
        assert torch.isfinite(tgt_j).all(), "Inf found in target after ExposureJitter"

    def test_overflow_preserves_chromaticity(self):
        """FP16 clamping preserves R/G/B ratios (chromaticity)."""
        inp = torch.zeros(13, 8, 8, dtype=torch.float16)
        # Set diffuse to known ratios: R=40000, G=20000, B=10000 (ratio 4:2:1)
        inp[0] = 40000.0
        inp[1] = 20000.0
        inp[2] = 10000.0
        tgt = torch.ones(3, 8, 8, dtype=torch.float16)

        ej = ExposureJitter(range=(1.0, 1.0))  # scale = 2.0
        inp_j, _ = ej((inp, tgt))

        # After clamping, R/G ratio should still be ~2.0 and R/B should be ~4.0
        r = inp_j[0].float()
        g = inp_j[1].float()
        b = inp_j[2].float()

        ratio_rg = (r / g).mean()
        ratio_rb = (r / b).mean()
        assert abs(ratio_rg.item() - 2.0) < 0.05, f"R/G ratio {ratio_rg} != 2.0"
        assert abs(ratio_rb.item() - 4.0) < 0.1, f"R/B ratio {ratio_rb} != 4.0"

    def test_negative_jitter_darkens(self):
        """Negative jitter (scale < 1) darkens radiance."""
        inp = torch.full((13, 4, 4), 4.0, dtype=torch.float16)
        tgt = torch.full((3, 4, 4), 4.0, dtype=torch.float16)

        ej = ExposureJitter(range=(-1.0, -1.0))  # scale = 0.5
        inp_j, tgt_j = ej((inp, tgt))

        torch.testing.assert_close(
            inp_j[:3], torch.full((3, 4, 4), 2.0, dtype=torch.float16), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            tgt_j, torch.full((3, 4, 4), 2.0, dtype=torch.float16), atol=1e-2, rtol=1e-2)

    def test_preserves_dtype(self):
        """Output dtype matches input dtype."""
        inp = torch.ones(13, 4, 4, dtype=torch.float16)
        tgt = torch.ones(3, 4, 4, dtype=torch.float16)

        ej = ExposureJitter(range=(0.5, 0.5))
        inp_j, tgt_j = ej((inp, tgt))

        assert inp_j.dtype == torch.float16
        assert tgt_j.dtype == torch.float16
