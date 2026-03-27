"""Safetensors dataset loader for pre-converted monti training data."""

import glob
import os

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset


class SafetensorsDataset(Dataset):
    """Dataset of pre-converted safetensors files from convert_to_safetensors.py.

    Each sample returns (input_tensor, target_tensor, albedo_d, albedo_s, hit_mask):
      - input_tensor: float16, shape (19, H, W) — demodulated irradiance + aux + albedo
      - target_tensor: float16, shape (6, H, W) — demodulated diffuse + specular irradiance
      - albedo_d: float16, shape (3, H, W) — diffuse albedo
      - albedo_s: float16, shape (3, H, W) — specular albedo
      - hit_mask: float16, shape (1, H, W) — geometry hit mask (1=hit, 0=miss)
    """

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.files: list[str] = sorted(
            glob.glob(os.path.join(data_dir, "**", "*.safetensors"), recursive=True)
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor, torch.Tensor, torch.Tensor]:
        tensors = load_file(self.files[idx])
        input_tensor = tensors["input"]    # (19, H, W)
        target_tensor = tensors["target"]  # (7, H, W) — 6ch irradiance + 1ch hit mask

        # Sanitize Inf/NaN that may exist in float16 data from demodulation overflow
        input_tensor = torch.nan_to_num(input_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        target_tensor = torch.nan_to_num(target_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        if self.transform is not None:
            input_tensor, target_tensor = self.transform((input_tensor, target_tensor))

        # Split target: 6ch irradiance + 1ch hit mask
        hit_mask = target_tensor[6:7]      # (1, H, W)
        target_tensor = target_tensor[:6]  # (6, H, W)

        # Extract albedo from input tensor
        albedo_d = input_tensor[13:16]     # (3, H, W)
        albedo_s = input_tensor[16:19]     # (3, H, W)

        return input_tensor, target_tensor, albedo_d, albedo_s, hit_mask
