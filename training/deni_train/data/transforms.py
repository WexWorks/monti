"""Spatial transforms for paired input/target tensors."""

import torch


class RandomCrop:
    """Random spatial crop applied identically to input and target."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, pair: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        input_tensor, target_tensor = pair
        _, h, w = input_tensor.shape

        if h < self.size or w < self.size:
            raise ValueError(
                f"Image ({h}x{w}) smaller than crop size ({self.size}x{self.size})")

        top = torch.randint(0, h - self.size + 1, (1,)).item()
        left = torch.randint(0, w - self.size + 1, (1,)).item()

        input_crop = input_tensor[:, top:top + self.size, left:left + self.size]
        target_crop = target_tensor[:, top:top + self.size, left:left + self.size]
        return input_crop, target_crop


class RandomHorizontalFlip:
    """Random horizontal flip applied identically to input and target.

    Negates the motion vector X channel (index 11) when flipped.
    """

    MOTION_X_INDEX = 11

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, pair: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        input_tensor, target_tensor = pair

        if torch.rand(1).item() < self.p:
            input_tensor = input_tensor.flip(-1).clone()
            target_tensor = target_tensor.flip(-1).clone()
            # Negate motion vector X
            input_tensor[self.MOTION_X_INDEX] = -input_tensor[self.MOTION_X_INDEX]

        return input_tensor, target_tensor


class Compose:
    """Chain transforms sequentially."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, pair: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            pair = t(pair)
        return pair
