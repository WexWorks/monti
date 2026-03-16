"""Generate synthetic EXR pairs for pipeline validation.

Creates 10 input/target pairs with random data matching monti_datagen's
channel naming convention. Used only for testing — never for model training.
"""

import os
import sys

import numpy as np

try:
    import OpenEXR
    import Imath
except ImportError:
    print("Error: OpenEXR and Imath packages required. Install with:")
    print("  pip install OpenEXR Imath")
    sys.exit(1)


# Input EXR channels: name → (components, pixel_type)
_INPUT_CHANNEL_GROUPS = [
    ("diffuse",  ["R", "G", "B", "A"], "HALF"),
    ("specular", ["R", "G", "B", "A"], "HALF"),
    ("albedo_d", ["R", "G", "B"],      "HALF"),
    ("albedo_s", ["R", "G", "B"],      "HALF"),
    ("normal",   ["X", "Y", "Z", "W"], "HALF"),
    ("depth",    ["Z"],                 "FLOAT"),
    ("motion",   ["X", "Y"],           "HALF"),
]

# Target EXR channels
_TARGET_CHANNEL_GROUPS = [
    ("diffuse",  ["R", "G", "B", "A"], "FLOAT"),
    ("specular", ["R", "G", "B", "A"], "FLOAT"),
]


def _make_exr_header(width: int, height: int,
                     channel_groups: list[tuple[str, list[str], str]]) -> OpenEXR.Header:
    """Build an EXR header with specified channels."""
    header = OpenEXR.Header(width, height)
    channels = {}
    for prefix, suffixes, ptype in channel_groups:
        pt = Imath.PixelType(Imath.PixelType.FLOAT if ptype == "FLOAT"
                             else Imath.PixelType.HALF)
        for s in suffixes:
            channels[f"{prefix}.{s}"] = Imath.Channel(pt)
    header["channels"] = channels
    return header


def _generate_input_data(width: int, height: int,
                         rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate synthetic input channel data."""
    data = {}
    pixels = height * width

    # Noisy diffuse/specular: smooth gradient + noise
    for prefix in ("diffuse", "specular"):
        for suffix in ("R", "G", "B"):
            gradient = np.linspace(0.0, 1.0, pixels, dtype=np.float32).reshape(height, width)
            noise = rng.normal(0.0, 0.1, (height, width)).astype(np.float32)
            data[f"{prefix}.{suffix}"] = np.maximum(gradient + noise, 0.0)
        # Alpha = 1.0 for all pixels (geometry mask)
        data[f"{prefix}.A"] = np.ones((height, width), dtype=np.float32)

    # Albedo: uniform random per-pixel
    for prefix in ("albedo_d", "albedo_s"):
        for suffix in ("R", "G", "B"):
            data[f"{prefix}.{suffix}"] = rng.uniform(0.1, 0.9, (height, width)).astype(np.float32)

    # Normals: random unit vectors + roughness in W
    nx = rng.standard_normal((height, width)).astype(np.float32)
    ny = rng.standard_normal((height, width)).astype(np.float32)
    nz = rng.standard_normal((height, width)).astype(np.float32)
    length = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
    data["normal.X"] = nx / length
    data["normal.Y"] = ny / length
    data["normal.Z"] = nz / length
    data["normal.W"] = rng.uniform(0.0, 1.0, (height, width)).astype(np.float32)

    # Depth: smooth gradient
    data["depth.Z"] = np.linspace(0.1, 100.0, pixels, dtype=np.float32).reshape(height, width)

    # Motion vectors: small random values
    data["motion.X"] = rng.uniform(-2.0, 2.0, (height, width)).astype(np.float32)
    data["motion.Y"] = rng.uniform(-2.0, 2.0, (height, width)).astype(np.float32)

    return data


def _generate_target_data(input_data: dict[str, np.ndarray],
                          rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate synthetic target data: smoothed version of input radiance."""
    from scipy.ndimage import gaussian_filter

    data = {}
    sigma = 2.0

    for prefix in ("diffuse", "specular"):
        for suffix in ("R", "G", "B"):
            noisy = input_data[f"{prefix}.{suffix}"]
            data[f"{prefix}.{suffix}"] = gaussian_filter(noisy, sigma=sigma).astype(np.float32)
        data[f"{prefix}.A"] = np.ones_like(input_data[f"{prefix}.R"])

    return data


def _write_exr(path: str, width: int, height: int,
               channel_groups: list[tuple[str, list[str], str]],
               data: dict[str, np.ndarray]) -> None:
    """Write channel data to an EXR file."""
    header = _make_exr_header(width, height, channel_groups)

    channel_data = {}
    for prefix, suffixes, ptype in channel_groups:
        for s in suffixes:
            name = f"{prefix}.{s}"
            arr = data[name]
            if ptype == "HALF":
                channel_data[name] = arr.astype(np.float16).tobytes()
            else:
                channel_data[name] = arr.astype(np.float32).tobytes()

    out = OpenEXR.OutputFile(path, header)
    out.writePixels(channel_data)
    out.close()


def generate(output_dir: str, num_pairs: int = 10, width: int = 128,
             height: int = 96, seed: int = 42) -> None:
    """Generate synthetic EXR pairs."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    for i in range(num_pairs):
        input_data = _generate_input_data(width, height, rng)
        target_data = _generate_target_data(input_data, rng)

        input_path = os.path.join(output_dir, f"frame_{i:06d}_input.exr")
        target_path = os.path.join(output_dir, f"frame_{i:06d}_target.exr")

        _write_exr(input_path, width, height, _INPUT_CHANNEL_GROUPS, input_data)
        _write_exr(target_path, width, height, _TARGET_CHANNEL_GROUPS, target_data)

        print(f"  [{i + 1}/{num_pairs}] {input_path}")

    print(f"Generated {num_pairs} pairs in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic EXR training data")
    parser.add_argument("--output", default="training_data/synthetic",
                        help="Output directory")
    parser.add_argument("--num-pairs", type=int, default=10,
                        help="Number of input/target pairs to generate")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(args.output, args.num_pairs, args.width, args.height, args.seed)


if __name__ == "__main__":
    main()
