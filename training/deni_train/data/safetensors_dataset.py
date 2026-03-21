"""Safetensors dataset loader for pre-converted monti training data."""

import glob
import os

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset


class SafetensorsDataset(Dataset):
    """Dataset of pre-converted safetensors files from convert_to_safetensors.py.

    Each sample returns (input_tensor, target_tensor) where:
      - input_tensor: float16, shape (13, H, W)
      - target_tensor: float16, shape (3, H, W)
    """

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.files: list[str] = sorted(
            glob.glob(os.path.join(data_dir, "**", "*.safetensors"), recursive=True)
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = load_file(self.files[idx])
        input_tensor = tensors["input"]
        target_tensor = tensors["target"]

        if self.transform is not None:
            input_tensor, target_tensor = self.transform((input_tensor, target_tensor))

        return input_tensor, target_tensor
