"""EXR dataset loader for monti training data pairs."""

import glob
import os
import re
import warnings

import numpy as np
import OpenEXR
import Imath
import torch
from torch.utils.data import Dataset


# Input EXR channel names (21 channels total, we load 13)
_INPUT_CHANNELS = [
    # Noisy diffuse RGB (discard A)
    ("diffuse.R", "diffuse.G", "diffuse.B"),
    # Noisy specular RGB (discard A)
    ("specular.R", "specular.G", "specular.B"),
    # World normals XYZ
    ("normal.X", "normal.Y", "normal.Z"),
    # Roughness (packed in normal.W)
    ("normal.W",),
    # Linear depth
    ("depth.Z",),
    # Motion vectors XY
    ("motion.X", "motion.Y"),
]

# Target EXR channel names (we load diffuse + specular RGB, sum them)
_TARGET_DIFFUSE = ("diffuse.R", "diffuse.G", "diffuse.B")
_TARGET_SPECULAR = ("specular.R", "specular.G", "specular.B")


def _read_exr_channels(path: str, channel_names: list[str]) -> dict[str, np.ndarray]:
    """Read specified channels from an EXR file as float32 numpy arrays."""
    exr = OpenEXR.InputFile(path)
    header = exr.header()

    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    result = {}
    pt_float = Imath.PixelType(Imath.PixelType.FLOAT)

    for name in channel_names:
        if name not in header["channels"]:
            raise KeyError(f"Channel '{name}' not found in {path}")
        raw = exr.channel(name, pt_float)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
        result[name] = arr

    exr.close()
    return result


class ExrDataset(Dataset):
    """Dataset of EXR input/target pairs from monti_datagen output.

    Each sample returns (input_tensor, target_tensor) where:
      - input_tensor: float16, shape (13, H, W)
      - target_tensor: float16, shape (3, H, W)
    """

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Find all input/target pairs
        input_files = sorted(glob.glob(os.path.join(data_dir, "**", "frame_*_input.exr"),
                                       recursive=True))

        self.pairs: list[tuple[str, str]] = []
        for input_path in input_files:
            target_path = re.sub(r"_input\.exr$", "_target.exr", input_path)
            if os.path.exists(target_path):
                self.pairs.append((input_path, target_path))
            else:
                warnings.warn(f"Missing target for {input_path}, skipping")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_path, target_path = self.pairs[idx]

        # Collect all input channel names
        input_channel_names = []
        for group in _INPUT_CHANNELS:
            input_channel_names.extend(group)

        # Read input channels
        input_data = _read_exr_channels(input_path, input_channel_names)

        # Stack into (13, H, W) tensor
        input_arrays = [input_data[name] for name in input_channel_names]
        input_tensor = torch.from_numpy(np.stack(input_arrays, axis=0)).to(torch.float16)

        # Read target channels (diffuse + specular, sum to combined radiance)
        target_channel_names = list(_TARGET_DIFFUSE) + list(_TARGET_SPECULAR)
        target_data = _read_exr_channels(target_path, target_channel_names)

        diff_r = target_data["diffuse.R"]
        diff_g = target_data["diffuse.G"]
        diff_b = target_data["diffuse.B"]
        spec_r = target_data["specular.R"]
        spec_g = target_data["specular.G"]
        spec_b = target_data["specular.B"]

        combined = np.stack([
            diff_r + spec_r,
            diff_g + spec_g,
            diff_b + spec_b,
        ], axis=0)
        target_tensor = torch.from_numpy(combined).to(torch.float16)

        if self.transform is not None:
            input_tensor, target_tensor = self.transform((input_tensor, target_tensor))

        return input_tensor, target_tensor
