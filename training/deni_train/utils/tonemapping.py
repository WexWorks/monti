"""ACES filmic tone mapping -- PyTorch implementation matching monti's tonemap.comp.

Stephen Hill's fitted RRT+ODT curve. Applied to raw linear HDR (no exposure
multiplication). Used inside the loss function for gradient weighting in
perceptually uniform space.
"""

import torch

# Row-major matrices transposed from GLSL column-major layout in tonemap.comp.
_ACES_M1 = torch.tensor([
    [0.59719, 0.35458, 0.04823],
    [0.07600, 0.90834, 0.01566],
    [0.02840, 0.13383, 0.83777],
])

_ACES_M2 = torch.tensor([
    [ 1.60475, -0.53108, -0.07367],
    [-0.10208,  1.10813, -0.00605],
    [-0.00327, -0.07276,  1.07602],
])


def aces_tonemap(x: torch.Tensor) -> torch.Tensor:
    """ACES filmic tone mapping on (B, 3, H, W) linear HDR tensors.

    Matches monti's Stephen Hill fitted curve (tonemap.comp):
        v = m1 * hdr
        a = v * (v + 0.0245786) - 0.000090537
        b = v * (0.983729 * v + 0.4329510) + 0.238081
        result = clamp(m2 * (a / b), 0, 1)
    """
    m1 = _ACES_M1.to(x.device, x.dtype)
    m2 = _ACES_M2.to(x.device, x.dtype)

    v = torch.einsum("ij,bjhw->bihw", m1, x)
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    return torch.einsum("ij,bjhw->bihw", m2, a / b).clamp(0.0, 1.0)
