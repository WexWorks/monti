"""Validate training dataset EXR pairs and generate a thumbnail gallery.

Checks:
  - All expected channels present in input and target EXRs
  - No NaN/Inf values (fatal — indicates renderer bug)
  - Per-channel statistics: min, max, mean, std
  - Flags suspiciously low variance (possible black render)
  - Generates HTML gallery with ACES-tonemapped side-by-side comparisons

Usage:
    python scripts/validate_dataset.py --data_dir training_data/
"""

import argparse
import base64
import glob
import io
import os
import re
import sys

import numpy as np

try:
    import OpenEXR
    import Imath
except ImportError:
    print("Error: OpenEXR and Imath packages required.")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow required. Install with: pip install Pillow")
    sys.exit(1)


# Expected input channels (21 total)
_INPUT_CHANNELS = [
    "diffuse.R", "diffuse.G", "diffuse.B", "diffuse.A",
    "specular.R", "specular.G", "specular.B", "specular.A",
    "albedo_d.R", "albedo_d.G", "albedo_d.B",
    "albedo_s.R", "albedo_s.G", "albedo_s.B",
    "normal.X", "normal.Y", "normal.Z", "normal.W",
    "depth.Z",
    "motion.X", "motion.Y",
]

# Expected target channels (8 total)
_TARGET_CHANNELS = [
    "diffuse.R", "diffuse.G", "diffuse.B", "diffuse.A",
    "specular.R", "specular.G", "specular.B", "specular.A",
]

# Minimum per-channel std to flag as suspiciously low
_MIN_VARIANCE_THRESHOLD = 1e-6


def _read_all_channels(path: str) -> dict[str, np.ndarray]:
    """Read all channels from an EXR file as float32 numpy arrays."""
    exr = OpenEXR.InputFile(path)
    try:
        header = exr.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        pt_float = Imath.PixelType(Imath.PixelType.FLOAT)

        result = {}
        for name in header["channels"]:
            raw = exr.channel(name, pt_float)
            result[name] = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
    finally:
        exr.close()
    return result


def _aces_tonemap(rgb: np.ndarray) -> np.ndarray:
    """ACES filmic tonemapping matching deni_train/utils/tonemapping.py.

    Input: (3, H, W) float32 linear HDR.
    Output: (3, H, W) float32 [0, 1] tonemapped.
    """
    m1 = np.array([
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777],
    ], dtype=np.float32)

    m2 = np.array([
        [ 1.60475, -0.53108, -0.07367],
        [-0.10208,  1.10813, -0.00605],
        [-0.00327, -0.07276,  1.07602],
    ], dtype=np.float32)

    # (3, H, W) -> reshape for matmul
    c, h, w = rgb.shape
    flat = rgb.reshape(3, -1)  # (3, N)

    v = m1 @ flat  # (3, N)
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    result = m2 @ (a / b)
    return np.clip(result.reshape(3, h, w), 0.0, 1.0)


def _make_thumbnail(channels: dict[str, np.ndarray]) -> np.ndarray:
    """Create ACES-tonemapped RGB uint8 image from EXR channel data.

    Combines diffuse + specular RGB. Returns (H, W, 3) uint8.
    """
    r = channels.get("diffuse.R", np.zeros(1))
    g = channels.get("diffuse.G", np.zeros(1))
    b = channels.get("diffuse.B", np.zeros(1))

    sr = channels.get("specular.R", np.zeros_like(r))
    sg = channels.get("specular.G", np.zeros_like(g))
    sb = channels.get("specular.B", np.zeros_like(b))

    rgb = np.stack([r + sr, g + sg, b + sb], axis=0)  # (3, H, W)
    tonemapped = _aces_tonemap(rgb)

    # Convert to uint8 (H, W, 3) with sRGB gamma
    gamma = np.power(tonemapped, 1.0 / 2.2)
    img = (gamma * 255.0).clip(0, 255).astype(np.uint8)
    return img.transpose(1, 2, 0)  # (H, W, 3)


def _image_to_data_uri(img_array: np.ndarray) -> str:
    """Convert uint8 (H,W,3) array to a PNG data URI for HTML embedding."""
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class ValidationResult:
    def __init__(self):
        self.total_pairs = 0
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.channel_stats: list[dict] = []
        self.thumbnails: list[dict] = []


def validate_pair(input_path: str, target_path: str,
                  result: ValidationResult) -> bool:
    """Validate a single input/target EXR pair. Returns True if OK."""
    pair_name = os.path.basename(input_path).replace("_input.exr", "")
    parent_dir = os.path.basename(os.path.dirname(input_path))
    display_name = f"{parent_dir}/{pair_name}"
    ok = True

    # Read files
    try:
        input_data = _read_all_channels(input_path)
    except Exception as e:
        result.errors.append(f"{display_name}: Failed to read input: {e}")
        return False

    try:
        target_data = _read_all_channels(target_path)
    except Exception as e:
        result.errors.append(f"{display_name}: Failed to read target: {e}")
        return False

    # Check expected channels
    input_channels = set(input_data.keys())
    for ch in _INPUT_CHANNELS:
        if ch not in input_channels:
            result.errors.append(f"{display_name}: Missing input channel '{ch}'")
            ok = False

    target_channels = set(target_data.keys())
    for ch in _TARGET_CHANNELS:
        if ch not in target_channels:
            result.errors.append(f"{display_name}: Missing target channel '{ch}'")
            ok = False

    if not ok:
        return False

    # Check for NaN/Inf
    for prefix, data in [("input", input_data), ("target", target_data)]:
        for ch_name, arr in data.items():
            if np.any(np.isnan(arr)):
                result.errors.append(
                    f"{display_name}: NaN in {prefix} channel '{ch_name}'")
                ok = False
            if np.any(np.isinf(arr)):
                result.errors.append(
                    f"{display_name}: Inf in {prefix} channel '{ch_name}'")
                ok = False

    # Per-channel statistics
    for ch_name, arr in sorted(input_data.items()):
        stats = {
            "pair": display_name,
            "file": "input",
            "channel": ch_name,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }
        result.channel_stats.append(stats)
        if arr.std() < _MIN_VARIANCE_THRESHOLD and ch_name not in ("diffuse.A", "specular.A"):
            result.warnings.append(
                f"{display_name}: Suspiciously low variance in input '{ch_name}' "
                f"(std={arr.std():.2e})")

    for ch_name, arr in sorted(target_data.items()):
        stats = {
            "pair": display_name,
            "file": "target",
            "channel": ch_name,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }
        result.channel_stats.append(stats)

    # Generate thumbnails
    try:
        input_thumb = _make_thumbnail(input_data)
        target_thumb = _make_thumbnail(target_data)
        result.thumbnails.append({
            "name": display_name,
            "input": _image_to_data_uri(input_thumb),
            "target": _image_to_data_uri(target_thumb),
        })
    except Exception as e:
        result.warnings.append(f"{display_name}: Failed to generate thumbnails: {e}")

    return ok


def _generate_html(result: ValidationResult, output_path: str) -> None:
    """Generate an HTML gallery page from validation results."""
    html_parts = ["""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Training Dataset Validation Gallery</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }
h1 { color: #4fc3f7; }
h2 { color: #81c784; }
.pair { display: flex; gap: 10px; margin-bottom: 20px; align-items: flex-start; }
.pair img { max-width: 480px; border: 1px solid #444; }
.label { color: #aaa; font-size: 12px; text-align: center; }
.error { color: #ef5350; }
.warning { color: #ffa726; }
.ok { color: #66bb6a; }
table { border-collapse: collapse; margin: 10px 0; }
th, td { border: 1px solid #444; padding: 4px 8px; text-align: right; font-size: 12px; }
th { background: #333; }
</style>
</head>
<body>
<h1>Training Dataset Validation Gallery</h1>
"""]

    # Summary
    n_errors = len(result.errors)
    n_warnings = len(result.warnings)
    status_class = "ok" if n_errors == 0 else "error"
    html_parts.append(f'<p class="{status_class}">Total pairs: {result.total_pairs} | '
                      f'Errors: {n_errors} | Warnings: {n_warnings}</p>')

    if result.errors:
        html_parts.append("<h2>Errors</h2><ul>")
        for e in result.errors:
            html_parts.append(f'<li class="error">{e}</li>')
        html_parts.append("</ul>")

    if result.warnings:
        html_parts.append("<h2>Warnings</h2><ul>")
        for w in result.warnings:
            html_parts.append(f'<li class="warning">{w}</li>')
        html_parts.append("</ul>")

    # Thumbnails
    html_parts.append("<h2>Side-by-Side Comparisons (ACES tonemapped)</h2>")
    for thumb in result.thumbnails:
        html_parts.append(f'<h3>{thumb["name"]}</h3>')
        html_parts.append('<div class="pair">')
        html_parts.append(f'<div><div class="label">Noisy Input</div>'
                          f'<img src="{thumb["input"]}"></div>')
        html_parts.append(f'<div><div class="label">Reference Target</div>'
                          f'<img src="{thumb["target"]}"></div>')
        html_parts.append('</div>')

    # Channel statistics summary table (aggregated across all pairs)
    html_parts.append("<h2>Channel Statistics (per-pair)</h2>")
    if result.channel_stats:
        html_parts.append(
            "<table><tr><th>Pair</th><th>File</th><th>Channel</th>"
            "<th>Min</th><th>Max</th><th>Mean</th><th>Std</th></tr>")
        for s in result.channel_stats:
            html_parts.append(
                f'<tr><td style="text-align:left">{s["pair"]}</td>'
                f'<td>{s["file"]}</td>'
                f'<td>{s["channel"]}</td>'
                f'<td>{s["min"]:.4g}</td>'
                f'<td>{s["max"]:.4g}</td>'
                f'<td>{s["mean"]:.4g}</td>'
                f'<td>{s["std"]:.4g}</td></tr>')
        html_parts.append("</table>")

    html_parts.append("</body></html>")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"Gallery written: {output_path}")


def validate_dataset(data_dir: str, gallery_path: str | None = None) -> ValidationResult:
    """Validate all EXR pairs under data_dir."""
    result = ValidationResult()

    # Find all input EXRs recursively
    input_files = sorted(
        glob.glob(os.path.join(data_dir, "**", "frame_*_input.exr"), recursive=True))

    if not input_files:
        print(f"No EXR input files found in {data_dir}")
        return result

    pairs = []
    for input_path in input_files:
        target_path = re.sub(r"_input\.exr$", "_target.exr", input_path)
        if os.path.exists(target_path):
            pairs.append((input_path, target_path))
        else:
            result.warnings.append(
                f"Missing target for {os.path.basename(input_path)}")

    result.total_pairs = len(pairs)
    print(f"Found {result.total_pairs} EXR pairs in {data_dir}\n")

    errors_found = 0
    for i, (input_path, target_path) in enumerate(pairs):
        rel_input = os.path.relpath(input_path, data_dir)
        print(f"  [{i + 1}/{len(pairs)}] Validating {rel_input}...", end=" ")

        ok = validate_pair(input_path, target_path, result)
        if ok:
            print("OK")
        else:
            print("ERRORS")
            errors_found += 1

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Validation Summary")
    print(f"{'=' * 60}")
    print(f"  Total pairs:  {result.total_pairs}")
    print(f"  Errors:       {len(result.errors)}")
    print(f"  Warnings:     {len(result.warnings)}")

    if result.errors:
        print(f"\nERRORS:")
        for e in result.errors:
            print(f"  - {e}")

    if result.warnings:
        print(f"\nWARNINGS:")
        for w in result.warnings:
            print(f"  - {w}")

    if not result.errors:
        print(f"\nAll pairs validated successfully.")

    # Generate gallery
    if gallery_path is None:
        gallery_path = os.path.join(data_dir, "validation_gallery.html")
    if result.thumbnails:
        _generate_html(result, gallery_path)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate training dataset EXR pairs")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing EXR pairs (searched recursively)")
    parser.add_argument("--gallery", default=None,
                        help="Output HTML gallery path (default: <data_dir>/validation_gallery.html)")
    args = parser.parse_args()

    result = validate_dataset(args.data_dir, args.gallery)
    sys.exit(1 if result.errors else 0)


if __name__ == "__main__":
    main()
