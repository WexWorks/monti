"""Tests for RandomRotation180 transform."""

import torch

from deni_train.data.transforms import RandomRotation180


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



