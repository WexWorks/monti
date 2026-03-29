"""Validate training dataset pairs and generate a thumbnail gallery.

Supports both EXR pairs and pre-converted safetensors files.

Checks:
  - All expected channels/tensors present
  - No NaN/Inf values (fatal — indicates renderer bug)
  - Per-channel statistics: min, max, mean, std
  - Flags suspiciously low variance (possible black render)
  - Generates HTML gallery with ACES-tonemapped side-by-side comparisons

Usage:
    python scripts/validate_dataset.py --data_dir training_data/
    python scripts/validate_dataset.py --data_dir training_data_st/ --data-format safetensors
"""

import argparse
import base64
import glob
import io
import os
import random
import sys

import numpy as np

try:
    from safetensors.torch import load_file as _st_load_file
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False

try:
    import OpenEXR
    import Imath
    _HAS_OPENEXR = True
except ImportError:
    _HAS_OPENEXR = False

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

# Channels excluded from low-variance checks (expected to be near-zero in static scenes)
_VARIANCE_EXEMPT_CHANNELS = {"diffuse.A", "specular.A", "motion.X", "motion.Y"}

# Safetensors input tensor channel index → logical name mapping (13 channels)
_ST_INPUT_CHANNEL_NAMES = [
    "diffuse.R", "diffuse.G", "diffuse.B",
    "specular.R", "specular.G", "specular.B",
    "normal.X", "normal.Y", "normal.Z",
    "normal.W",
    "depth.Z",
    "motion.X", "motion.Y",
]

# Safetensors input channels exempt from low-variance checks
_ST_VARIANCE_EXEMPT_INDICES = {
    i for i, name in enumerate(_ST_INPUT_CHANNEL_NAMES)
    if name in _VARIANCE_EXEMPT_CHANNELS
}


def _read_all_channels(path: str) -> dict[str, np.ndarray]:
    """Read all channels from an EXR file as float32 numpy arrays."""
    if not _HAS_OPENEXR:
        raise RuntimeError("OpenEXR required. Install with: pip install OpenEXR")
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


def _rgb_to_thumbnail(rgb: np.ndarray) -> np.ndarray:
    """ACES-tonemap a (3, H, W) float32 linear HDR image to (H, W, 3) uint8."""
    tonemapped = _aces_tonemap(rgb)
    gamma = np.power(tonemapped, 1.0 / 2.2)
    return (gamma * 255.0).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)


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
    return _rgb_to_thumbnail(rgb)


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
                  result: ValidationResult,
                  data_dir: str | None = None) -> bool:
    """Validate a single input/target EXR pair. Returns True if OK."""
    basename = os.path.basename(input_path)
    if basename == "input.exr":
        if data_dir is not None:
            display_name = os.path.relpath(input_path, data_dir).replace(
                "/input.exr", "").replace("\\input.exr", "").replace("\\", "/")
        else:
            display_name = os.path.basename(os.path.dirname(input_path))
    else:
        display_name = basename[:-len("_input.exr")]
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

    # Per-channel statistics — only record channels with anomalies
    low_variance_channels = []
    for ch_name, arr in sorted(input_data.items()):
        std_val = float(arr.std())
        if std_val < _MIN_VARIANCE_THRESHOLD and ch_name not in _VARIANCE_EXEMPT_CHANNELS:
            low_variance_channels.append(ch_name)
            result.channel_stats.append({
                "pair": display_name,
                "file": "input",
                "channel": ch_name,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": std_val,
            })

    for ch_name, arr in sorted(target_data.items()):
        std_val = float(arr.std())
        if std_val < _MIN_VARIANCE_THRESHOLD and ch_name not in _VARIANCE_EXEMPT_CHANNELS:
            result.channel_stats.append({
                "pair": display_name,
                "file": "target",
                "channel": ch_name,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": std_val,
            })

    # Consolidate low-variance warnings
    non_exempt_input = [ch for ch in _INPUT_CHANNELS if ch not in _VARIANCE_EXEMPT_CHANNELS]
    if len(low_variance_channels) == len(non_exempt_input):
        result.warnings.append(
            f"{display_name}: All input channels have zero variance "
            f"(empty viewpoint — camera may not see the model)")
    elif low_variance_channels:
        result.warnings.append(
            f"{display_name}: Low variance in input channels: "
            f"{', '.join(low_variance_channels)}")

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


def _make_thumbnail_from_tensors(input_tensor: np.ndarray,
                                 target_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create ACES-tonemapped thumbnails from safetensors numpy arrays.

    input_tensor: (13, H, W) float32. Channels 0-2 = diffuse RGB, 3-5 = specular RGB.
    target_tensor: (3, H, W) float32. Already diffuse+specular sum.
    Returns (input_thumb, target_thumb) as (H, W, 3) uint8.
    """
    noisy_rgb = input_tensor[:3] + input_tensor[3:6]  # (3, H, W)
    return _rgb_to_thumbnail(noisy_rgb), _rgb_to_thumbnail(target_tensor)


def validate_safetensors_file(path: str, result: ValidationResult,
                              data_dir: str | None = None) -> bool:
    """Validate a single .safetensors file. Returns True if OK."""
    if data_dir is not None:
        display_name = os.path.relpath(path, data_dir).replace("\\", "/")
        display_name = display_name[:-len(".safetensors")]
    else:
        display_name = os.path.basename(path)[:-len(".safetensors")]

    ok = True

    try:
        import torch
        tensors = _st_load_file(path)
    except Exception as e:
        result.errors.append(f"{display_name}: Failed to load: {e}")
        return False

    # Check expected keys
    if "input" not in tensors:
        result.errors.append(f"{display_name}: Missing 'input' tensor")
        ok = False
    if "target" not in tensors:
        result.errors.append(f"{display_name}: Missing 'target' tensor")
        ok = False
    if not ok:
        return False

    inp = tensors["input"].float().numpy()
    tgt = tensors["target"].float().numpy()

    # Check shapes
    if inp.ndim != 3 or inp.shape[0] != 13:
        result.errors.append(
            f"{display_name}: Input shape {inp.shape}, expected (13, H, W)")
        ok = False
    if tgt.ndim != 3 or tgt.shape[0] != 3:
        result.errors.append(
            f"{display_name}: Target shape {tgt.shape}, expected (3, H, W)")
        ok = False
    if not ok:
        return False

    # Check for NaN/Inf
    if np.any(np.isnan(inp)):
        nan_channels = [_ST_INPUT_CHANNEL_NAMES[c] for c in range(13) if np.any(np.isnan(inp[c]))]
        result.errors.append(f"{display_name}: NaN in input channels: {', '.join(nan_channels)}")
        ok = False
    if np.any(np.isinf(inp)):
        inf_channels = [_ST_INPUT_CHANNEL_NAMES[c] for c in range(13) if np.any(np.isinf(inp[c]))]
        result.errors.append(f"{display_name}: Inf in input channels: {', '.join(inf_channels)}")
        ok = False
    if np.any(np.isnan(tgt)):
        result.errors.append(f"{display_name}: NaN in target tensor")
        ok = False
    if np.any(np.isinf(tgt)):
        result.errors.append(f"{display_name}: Inf in target tensor")
        ok = False

    # Per-channel variance checks
    low_variance_channels = []
    for c in range(13):
        ch_name = _ST_INPUT_CHANNEL_NAMES[c]
        std_val = float(inp[c].std())
        if std_val < _MIN_VARIANCE_THRESHOLD and c not in _ST_VARIANCE_EXEMPT_INDICES:
            low_variance_channels.append(ch_name)
            result.channel_stats.append({
                "pair": display_name,
                "file": "input",
                "channel": ch_name,
                "min": float(inp[c].min()),
                "max": float(inp[c].max()),
                "mean": float(inp[c].mean()),
                "std": std_val,
            })

    for c in range(3):
        ch_name = f"target.ch{c}"
        std_val = float(tgt[c].std())
        if std_val < _MIN_VARIANCE_THRESHOLD:
            result.channel_stats.append({
                "pair": display_name,
                "file": "target",
                "channel": ch_name,
                "min": float(tgt[c].min()),
                "max": float(tgt[c].max()),
                "mean": float(tgt[c].mean()),
                "std": std_val,
            })

    # Consolidate low-variance warnings
    non_exempt = [n for i, n in enumerate(_ST_INPUT_CHANNEL_NAMES)
                  if i not in _ST_VARIANCE_EXEMPT_INDICES]
    if len(low_variance_channels) == len(non_exempt):
        result.warnings.append(
            f"{display_name}: All input channels have zero variance "
            f"(empty viewpoint — camera may not see the model)")
    elif low_variance_channels:
        result.warnings.append(
            f"{display_name}: Low variance in input channels: "
            f"{', '.join(low_variance_channels)}")

    # Generate thumbnails
    try:
        input_thumb, target_thumb = _make_thumbnail_from_tensors(inp, tgt)
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
h2 { color: #81c784; border-bottom: 1px solid #444; padding-bottom: 4px; }
h3 { color: #aed581; margin-left: 10px; }
h4 { color: #90a4ae; margin-left: 20px; }
.pair { display: flex; gap: 10px; margin-bottom: 20px; margin-left: 30px;
        align-items: flex-start; }
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

    # Gallery: flat list of pairs with display_name as caption
    html_parts.append("<h2>Side-by-Side Comparisons (ACES tonemapped)</h2>")
    for thumb in result.thumbnails:
        html_parts.append('<div class="pair">')
        html_parts.append(
            f'<div><div class="label">{thumb["name"]} &middot; noisy</div>'
            f'<img src="{thumb["input"]}"></div>')
        html_parts.append(
            f'<div><div class="label">{thumb["name"]} &middot; reference</div>'
            f'<img src="{thumb["target"]}"></div>')
        html_parts.append('</div>')

    # Flagged channels table — only show channels with anomalies
    if result.channel_stats:
        html_parts.append("<h2>Flagged Channels (low variance)</h2>")
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


def _print_summary(result: ValidationResult) -> None:
    """Print validation summary to stdout."""
    print(f"\n{'=' * 60}")
    print(f"Validation Summary")
    print(f"{'=' * 60}")
    print(f"  Total files:  {result.total_pairs}")
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
        print(f"\nAll files validated successfully.")


def _detect_data_format(data_dir: str) -> str:
    """Return 'safetensors' if .safetensors files exist in data_dir, else 'exr'."""
    if glob.glob(os.path.join(data_dir, "**", "*.safetensors"), recursive=True):
        return "safetensors"
    return "exr"


def validate_dataset(data_dir: str, gallery_path: str | None = None,
                     max_variations: int | None = None,
                     max_viewpoints: int | None = 2,
                     data_format: str = "auto") -> ValidationResult:
    """Validate all data files under data_dir.

    Args:
        data_dir: Root directory to search for data files.
        gallery_path: Output HTML gallery path.
        max_variations: If set, only validate the first N files per scene.
        max_viewpoints: If set, randomly sample N viewpoints per scene.
        data_format: 'auto' (default), 'exr', or 'safetensors'.
    """
    result = ValidationResult()

    if data_format == "auto":
        data_format = _detect_data_format(data_dir)

    if data_format == "safetensors":
        return _validate_safetensors_dataset(data_dir, gallery_path, max_variations,
                                             max_viewpoints, result)
    return _validate_exr_dataset(data_dir, gallery_path, max_variations, max_viewpoints, result)


def _validate_safetensors_dataset(data_dir: str, gallery_path: str | None,
                                  max_variations: int | None,
                                  max_viewpoints: int | None,
                                  result: ValidationResult) -> ValidationResult:
    """Validate all .safetensors files under data_dir."""
    if not _HAS_SAFETENSORS:
        print("Error: safetensors package required. Install with: pip install safetensors")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.safetensors"), recursive=True))
    if not files:
        print(f"No .safetensors files found in {data_dir}")
        return result

    if max_variations is not None:
        scene_counts: dict[str, int] = {}
        filtered: list[str] = []
        for path in files:
            rel = os.path.relpath(path, data_dir)
            scene_key = rel.split(os.sep)[0] if os.sep in rel else rel
            count = scene_counts.get(scene_key, 0)
            if count < max_variations:
                filtered.append(path)
                scene_counts[scene_key] = count + 1
        print(f"Limiting to {max_variations} variation(s) per scene "
              f"({len(filtered)}/{len(files)} files)")
        files = filtered

    if max_viewpoints is not None:
        scene_files: dict[str, list[str]] = {}
        for path in files:
            rel = os.path.relpath(path, data_dir)
            scene_key = rel.split(os.sep)[0] if os.sep in rel else rel
            scene_files.setdefault(scene_key, []).append(path)
        sampled: list[str] = []
        for scene_key, scene_file_list in sorted(scene_files.items()):
            if len(scene_file_list) > max_viewpoints:
                chosen = random.Random(scene_key).sample(scene_file_list, max_viewpoints)
                sampled.extend(sorted(chosen))
            else:
                sampled.extend(scene_file_list)
        print(f"Limiting to {max_viewpoints} viewpoint(s) per scene "
              f"({len(sampled)}/{len(files)} files)")
        files = sampled

    result.total_pairs = len(files)
    print(f"Found {result.total_pairs} safetensors files in {data_dir}\n")

    for i, path in enumerate(files):
        rel = os.path.relpath(path, data_dir)
        print(f"  [{i + 1}/{len(files)}] Validating {rel}...", end=" ")

        ok = validate_safetensors_file(path, result, data_dir)
        print("OK" if ok else "ERRORS")

    _print_summary(result)

    if gallery_path is None:
        gallery_path = os.path.join(data_dir, "validation_gallery.html")
    if result.thumbnails:
        _generate_html(result, gallery_path)

    return result


def _validate_exr_dataset(data_dir: str, gallery_path: str | None,
                          max_variations: int | None,
                          max_viewpoints: int | None,
                          result: ValidationResult) -> ValidationResult:
    """Validate all EXR pairs under data_dir."""
    if not _HAS_OPENEXR:
        print("Error: OpenEXR and Imath packages required.")
        sys.exit(1)

    # Find all input EXRs recursively (directory-based and flat naming)
    dir_files = glob.glob(os.path.join(data_dir, "**", "input.exr"), recursive=True)
    flat_files = glob.glob(os.path.join(data_dir, "**", "*_input.exr"), recursive=True)
    input_files = sorted(set(dir_files + flat_files))

    if not input_files:
        print(f"No EXR input files found in {data_dir}")
        return result

    pairs = []
    for input_path in input_files:
        basename = os.path.basename(input_path)
        if basename == "input.exr":
            target_path = os.path.join(os.path.dirname(input_path), "target.exr")
        else:
            target_path = input_path[:-len("_input.exr")] + "_target.exr"
        if os.path.exists(target_path):
            pairs.append((input_path, target_path))
        else:
            result.warnings.append(
                f"Missing target for {os.path.relpath(input_path, data_dir)}")

    # Limit to first N pairs per scene if --max_variations is set
    if max_variations is not None:
        scene_counts: dict[str, int] = {}
        filtered_pairs = []
        for input_path, target_path in pairs:
            rel = os.path.relpath(input_path, data_dir)
            scene_key = rel.split(os.sep)[0] if os.sep in rel else rel
            count = scene_counts.get(scene_key, 0)
            if count < max_variations:
                filtered_pairs.append((input_path, target_path))
                scene_counts[scene_key] = count + 1
        print(f"Limiting to {max_variations} variation(s) per scene "
              f"({len(filtered_pairs)}/{len(pairs)} pairs)")
        pairs = filtered_pairs

    if max_viewpoints is not None:
        scene_pairs: dict[str, list[tuple[str, str]]] = {}
        for input_path, target_path in pairs:
            rel = os.path.relpath(input_path, data_dir)
            scene_key = rel.split(os.sep)[0] if os.sep in rel else rel
            scene_pairs.setdefault(scene_key, []).append((input_path, target_path))
        sampled_pairs: list[tuple[str, str]] = []
        for scene_key, scene_pair_list in sorted(scene_pairs.items()):
            if len(scene_pair_list) > max_viewpoints:
                chosen = random.Random(scene_key).sample(scene_pair_list, max_viewpoints)
                sampled_pairs.extend(sorted(chosen))
            else:
                sampled_pairs.extend(scene_pair_list)
        print(f"Limiting to {max_viewpoints} viewpoint(s) per scene "
              f"({len(sampled_pairs)}/{len(pairs)} pairs)")
        pairs = sampled_pairs

    result.total_pairs = len(pairs)
    print(f"Found {result.total_pairs} EXR pairs in {data_dir}\n")

    for i, (input_path, target_path) in enumerate(pairs):
        rel_input = os.path.relpath(input_path, data_dir)
        print(f"  [{i + 1}/{len(pairs)}] Validating {rel_input}...", end=" ")

        ok = validate_pair(input_path, target_path, result, data_dir)
        print("OK" if ok else "ERRORS")

    _print_summary(result)

    if gallery_path is None:
        gallery_path = os.path.join(data_dir, "validation_gallery.html")
    if result.thumbnails:
        _generate_html(result, gallery_path)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate training dataset (EXR pairs or safetensors)")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing data files (searched recursively)")
    parser.add_argument("--gallery", default=None,
                        help="Output HTML gallery path (default: <data_dir>/validation_gallery.html)")
    parser.add_argument("--max_variations", type=int, default=None,
                        help="Only validate the first N files per scene")
    parser.add_argument("--max-viewpoints", type=int, default=2,
                        help="Randomly sample N viewpoints per scene (default: 2; 0 = all)")
    parser.add_argument("--data-format", default="auto",
                        choices=["auto", "exr", "safetensors"],
                        help="Data format: auto-detect (default), exr, or safetensors")
    args = parser.parse_args()

    max_viewpoints = args.max_viewpoints if args.max_viewpoints != 0 else None
    result = validate_dataset(args.data_dir, args.gallery,
                              max_variations=args.max_variations,
                              max_viewpoints=max_viewpoints,
                              data_format=args.data_format)
    sys.exit(1 if result.errors else 0)


if __name__ == "__main__":
    main()
