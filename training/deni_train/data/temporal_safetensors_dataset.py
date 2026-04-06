"""Temporal safetensors dataset loader.

Loads pre-cropped W-frame sequence safetensors produced by preprocess_temporal.py
in temporal mode (--window W). Each file is one training sample — windowing and
crop selection happen offline.
"""

import glob
import os

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset


class TemporalSafetensorsDataset(Dataset):
    """Dataset for pre-cropped temporal sequences.

    Each .safetensors file contains:
        input:  float16, (W, 19, H_crop, W_crop)  — W-frame sequence, 19 G-buffer channels
        target: float16, (W,  6, H_crop, W_crop)  — W-frame sequence, 6 target channels

    No windowing or crop logic at training time.
    """

    def __init__(self, data_dir: str):
        self.files = sorted(
            glob.glob(os.path.join(data_dir, "**", "*.safetensors"), recursive=True)
        )
        if not self.files:
            raise ValueError(f"No .safetensors files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        tensors = load_file(self.files[idx])
        inp = tensors["input"].float().clamp(-65504.0, 65504.0)
        tgt = tensors["target"].float().clamp(-65504.0, 65504.0)
        inp = torch.nan_to_num(inp, nan=0.0, posinf=0.0, neginf=0.0)
        tgt = torch.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0)
        return {"input": inp, "target": tgt}
