"""EXR dataset loader for monti training data pairs."""

import glob
import os
import warnings

import numpy as np
import OpenEXR
import Imath
import torch
from torch.utils.data import Dataset


# Input EXR channel names (19 channels: demodulated irradiance + aux + albedo)
_INPUT_CHANNELS = [
    # Noisy diffuse RGB — demodulated to irradiance at load time (channels 0-2)
    ("diffuse.R", "diffuse.G", "diffuse.B"),
    # Noisy specular RGB — demodulated to irradiance at load time (channels 3-5)
    ("specular.R", "specular.G", "specular.B"),
    # World normals XYZ (channels 6-8)
    ("normal.X", "normal.Y", "normal.Z"),
    # Roughness (channel 9)
    ("normal.W",),
    # Linear depth (channel 10)
    ("depth.Z",),
    # Motion vectors XY (channels 11-12)
    ("motion.X", "motion.Y"),
    # Diffuse albedo RGB (channels 13-15)
    ("albedo_d.R", "albedo_d.G", "albedo_d.B"),
    # Specular albedo RGB (channels 16-18)
    ("albedo_s.R", "albedo_s.G", "albedo_s.B"),
]

# Flattened list of all input channel names (precomputed)
_INPUT_CHANNEL_NAMES = [name for group in _INPUT_CHANNELS for name in group]

# Target EXR channel names (separate diffuse + specular for demodulated output)
_TARGET_DIFFUSE = ("diffuse.R", "diffuse.G", "diffuse.B")
_TARGET_SPECULAR = ("specular.R", "specular.G", "specular.B")

# Demodulation epsilon (matches GPU DEMOD_EPS)
_DEMOD_EPS = 0.001


def _read_exr_channels(path: str, channel_names: list[str]) -> dict[str, np.ndarray]:
    """Read specified channels from an EXR file as float32 numpy arrays."""
    exr = OpenEXR.InputFile(path)
    try:
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
    finally:
        exr.close()
    return result


class ExrDataset(Dataset):
    """Dataset of EXR input/target pairs from monti_datagen output.

    Each sample returns (input_tensor, target_tensor, albedo_d, albedo_s):
      - input_tensor: float16, shape (19, H, W) — demodulated irradiance + aux + albedo
      - target_tensor: float16, shape (6, H, W) — demodulated diffuse + specular irradiance
      - albedo_d: float16, shape (3, H, W) — diffuse albedo
      - albedo_s: float16, shape (3, H, W) — specular albedo
    """

    def __init__(self, data_dir: str, transform=None, crops_per_image: int = 1):
        self.data_dir = data_dir
        self.transform = transform
        self.crops_per_image = max(1, crops_per_image)

        # Find all input/target pairs (directory-based and flat naming)
        dir_files = glob.glob(os.path.join(data_dir, "**", "input.exr"),
                              recursive=True)
        flat_files = glob.glob(os.path.join(data_dir, "**", "*_input.exr"),
                               recursive=True)
        input_files = sorted(set(dir_files + flat_files))

        self.pairs: list[tuple[str, str]] = []
        for input_path in input_files:
            basename = os.path.basename(input_path)
            if basename == "input.exr":
                target_path = os.path.join(os.path.dirname(input_path), "target.exr")
            else:
                target_path = input_path[:-len("_input.exr")] + "_target.exr"
            if os.path.exists(target_path):
                self.pairs.append((input_path, target_path))
            else:
                warnings.warn(f"Missing target for {input_path}, skipping")

    def __len__(self) -> int:
        return len(self.pairs) * self.crops_per_image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor, torch.Tensor]:
        input_path, target_path = self.pairs[idx // self.crops_per_image]

        # Read input channels
        input_data = _read_exr_channels(input_path, _INPUT_CHANNEL_NAMES)

        # Stack raw input channels into (19, H, W)
        input_arrays = np.stack([input_data[name] for name in _INPUT_CHANNEL_NAMES], axis=0)

        # Extract albedo arrays (channels 13-18)
        albedo_d = input_arrays[13:16]  # (3, H, W)
        albedo_s = input_arrays[16:19]  # (3, H, W)

        # Demodulate diffuse irradiance (channels 0-2)
        input_arrays[0:3] = input_arrays[0:3] / np.maximum(albedo_d, _DEMOD_EPS)

        # Demodulate specular irradiance (channels 3-5)
        input_arrays[3:6] = input_arrays[3:6] / np.maximum(albedo_s, _DEMOD_EPS)

        np.clip(input_arrays, -65504.0, 65504.0, out=input_arrays)
        input_tensor = torch.from_numpy(input_arrays).to(torch.float16)

        # Read target channels (separate diffuse + specular, demodulate)
        target_channel_names = list(_TARGET_DIFFUSE) + list(_TARGET_SPECULAR)
        target_data = _read_exr_channels(target_path, target_channel_names)

        target_d = np.stack([target_data[n] for n in _TARGET_DIFFUSE], axis=0)
        target_s = np.stack([target_data[n] for n in _TARGET_SPECULAR], axis=0)

        # Demodulate target using input-side albedo (albedo is a material property,
        # identical at any SPP)
        target_d = target_d / np.maximum(albedo_d, _DEMOD_EPS)
        target_s = target_s / np.maximum(albedo_s, _DEMOD_EPS)

        # Target: 6ch demodulated irradiance
        np.clip(target_d, -65504.0, 65504.0, out=target_d)
        np.clip(target_s, -65504.0, 65504.0, out=target_s)
        target_arrays = np.concatenate([target_d, target_s], axis=0)  # (6, H, W)
        target_tensor = torch.from_numpy(target_arrays).to(torch.float16)

        if self.transform is not None:
            # Transform applies same spatial op to input (19ch) and target (6ch).
            # Albedo is already in input channels 13-18.
            input_tensor, target_tensor = self.transform((input_tensor, target_tensor))

        # Extract albedo from (transformed) input tensor
        albedo_d_tensor = input_tensor[13:16]   # (3, H, W)
        albedo_s_tensor = input_tensor[16:19]   # (3, H, W)

        return input_tensor, target_tensor, albedo_d_tensor, albedo_s_tensor
