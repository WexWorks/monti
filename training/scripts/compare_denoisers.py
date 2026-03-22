#!/usr/bin/env python3
"""Compare denoiser output quality against ground truth references.

Usage:
    python scripts/compare_denoisers.py --data-dir training_data/ --output results/comparison/
    python scripts/compare_denoisers.py --data-dir training_data/ --model models/deni_v1.onnx --output results/comparison/

When --model is provided, runs ONNX inference on each input to produce denoised output.
When --model is omitted, compares only noisy vs reference (baseline measurement).
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add parent so deni_train package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import OpenEXR
import Imath
import torch

from deni_train.utils.metrics import compute_psnr, compute_ssim


def read_exr_rgb(path: str, channel_prefix: str = "") -> np.ndarray:
    """Read RGB channels from an EXR file, returning (H, W, 3) float32 array."""
    exr = OpenEXR.InputFile(path)
    header = exr.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    channels = header["channels"]
    # Try prefixed channels first, fall back to R/G/B
    if f"{channel_prefix}R" in channels:
        names = [f"{channel_prefix}R", f"{channel_prefix}G", f"{channel_prefix}B"]
    elif "R" in channels:
        names = ["R", "G", "B"]
    else:
        raise ValueError(f"Cannot find RGB channels in {path}: {list(channels.keys())}")

    data = []
    for name in names:
        raw = exr.channel(name, pt)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
        data.append(arr)
    return np.stack(data, axis=-1)


def read_input_combined(path: str) -> np.ndarray:
    """Read noisy input EXR and combine diffuse + specular to RGB (H, W, 3)."""
    exr = OpenEXR.InputFile(path)
    header = exr.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    diff_names = ["diffuse.R", "diffuse.G", "diffuse.B"]
    spec_names = ["specular.R", "specular.G", "specular.B"]

    combined = np.zeros((height, width, 3), dtype=np.float32)
    for i, (d, s) in enumerate(zip(diff_names, spec_names)):
        diff = np.frombuffer(exr.channel(d, pt), dtype=np.float32).reshape(height, width)
        spec = np.frombuffer(exr.channel(s, pt), dtype=np.float32).reshape(height, width)
        combined[:, :, i] = diff + spec
    return combined


def read_target_combined(path: str) -> np.ndarray:
    """Read target (reference) EXR and combine diffuse + specular to RGB (H, W, 3)."""
    return read_input_combined(path)  # Same channel format


def find_pairs(data_dir: str) -> list[tuple[str, str, str]]:
    """Find input/target EXR pairs. Returns list of (name, input_path, target_path)."""
    pairs = []

    # Directory-based: <scene>/input.exr + <scene>/target.exr
    for input_path in sorted(glob.glob(os.path.join(data_dir, "**", "input.exr"), recursive=True)):
        target_path = os.path.join(os.path.dirname(input_path), "target.exr")
        if os.path.exists(target_path):
            name = os.path.relpath(os.path.dirname(input_path), data_dir)
            pairs.append((name, input_path, target_path))

    # Flat naming: <name>_input.exr + <name>_target.exr
    for input_path in sorted(glob.glob(os.path.join(data_dir, "**", "*_input.exr"), recursive=True)):
        target_path = input_path[: -len("_input.exr")] + "_target.exr"
        if os.path.exists(target_path):
            name = os.path.basename(input_path)[: -len("_input.exr")]
            pairs.append((name, input_path, target_path))

    return pairs


def np_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert (H, W, 3) numpy array to (1, 3, H, W) torch tensor."""
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()


def run_onnx_inference(model_path: str, input_path: str) -> np.ndarray:
    """Run ONNX model inference on an input EXR, returning (H, W, 3) output."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Install with: pip install onnxruntime", file=sys.stderr)
        sys.exit(1)

    exr = OpenEXR.InputFile(input_path)
    header = exr.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read all 13 input channels in the expected order
    channel_names = [
        "diffuse.R", "diffuse.G", "diffuse.B",
        "specular.R", "specular.G", "specular.B",
        "normal.X", "normal.Y", "normal.Z", "normal.W",
        "depth.Z",
        "motion.X", "motion.Y",
    ]

    channels = []
    for name in channel_names:
        raw = exr.channel(name, pt)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(1, 1, height, width)
        channels.append(arr)
    input_tensor = np.concatenate(channels, axis=1)  # (1, 13, H, W)

    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    result = sess.run(None, {input_name: input_tensor})[0]  # (1, 3, H, W)

    return result[0].transpose(1, 2, 0)  # (H, W, 3)


def write_png(img: np.ndarray, path: str) -> None:
    """Write (H, W, 3) float32 image as 8-bit PNG after simple tonemap."""
    from PIL import Image

    # Simple Reinhard tonemap for visualization
    tonemapped = img / (1.0 + img)
    srgb = np.clip(tonemapped ** (1.0 / 2.2), 0.0, 1.0)
    uint8 = (srgb * 255.0).astype(np.uint8)
    Image.fromarray(uint8).save(path)


def generate_html_gallery(results: list[dict], output_dir: str, has_denoised: bool) -> str:
    """Generate an HTML comparison gallery."""
    html_path = os.path.join(output_dir, "gallery.html")

    rows = ""
    for r in results:
        name = r["name"]
        noisy_src = os.path.relpath(r["noisy_png"], output_dir)
        ref_src = os.path.relpath(r["reference_png"], output_dir)

        denoised_cell = ""
        denoised_header = ""
        if has_denoised:
            denoised_header = "<th>Denoised</th>"
            if "denoised_png" in r:
                den_src = os.path.relpath(r["denoised_png"], output_dir)
                denoised_cell = f'<td><img src="{den_src}" style="max-width:100%"></td>'
            else:
                denoised_cell = "<td>N/A</td>"

        metrics_text = f"Noisy PSNR: {r['noisy_psnr']:.2f} dB, SSIM: {r['noisy_ssim']:.4f}"
        if "denoised_psnr" in r:
            metrics_text += f"<br>Denoised PSNR: {r['denoised_psnr']:.2f} dB, SSIM: {r['denoised_ssim']:.4f}"

        rows += f"""
        <tr>
            <td colspan="{'3' if has_denoised else '2'}" style="background:#222;color:#fff;padding:8px">
                <strong>{name}</strong> — {metrics_text}
            </td>
        </tr>
        <tr>
            <td><img src="{noisy_src}" style="max-width:100%"></td>
            {denoised_cell}
            <td><img src="{ref_src}" style="max-width:100%"></td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html>
<head><title>Denoiser Comparison</title>
<style>
    body {{ background: #111; color: #eee; font-family: sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #444; padding: 4px; text-align: center; }}
    img {{ display: block; margin: auto; }}
</style>
</head>
<body>
<h1>Denoiser Quality Comparison</h1>
<table>
    <tr>
        <th>Noisy</th>
        {denoised_header}
        <th>Reference</th>
    </tr>
    {rows}
</table>
</body>
</html>"""

    with open(html_path, "w") as f:
        f.write(html)
    return html_path


def main():
    parser = argparse.ArgumentParser(description="Compare denoiser output quality")
    parser.add_argument("--data-dir", required=True, help="Directory with input/target EXR pairs")
    parser.add_argument("--model", default=None, help="ONNX model for ML denoised output")
    parser.add_argument("--output", required=True, help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    images_dir = os.path.join(args.output, "images")
    os.makedirs(images_dir, exist_ok=True)

    pairs = find_pairs(args.data_dir)
    if not pairs:
        print(f"No input/target EXR pairs found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pairs)} scene pairs")

    results = []
    has_denoised = args.model is not None

    for name, input_path, target_path in pairs:
        print(f"Processing: {name}")

        noisy_rgb = read_input_combined(input_path)
        ref_rgb = read_target_combined(target_path)

        noisy_t = np_to_tensor(noisy_rgb)
        ref_t = np_to_tensor(ref_rgb)

        noisy_psnr = compute_psnr(noisy_t, ref_t)
        noisy_ssim = compute_ssim(noisy_t, ref_t)

        safe_name = name.replace(os.sep, "_").replace("/", "_")
        noisy_png = os.path.join(images_dir, f"{safe_name}_noisy.png")
        ref_png = os.path.join(images_dir, f"{safe_name}_reference.png")
        write_png(noisy_rgb, noisy_png)
        write_png(ref_rgb, ref_png)

        entry = {
            "name": name,
            "noisy_psnr": noisy_psnr,
            "noisy_ssim": noisy_ssim,
            "noisy_png": noisy_png,
            "reference_png": ref_png,
        }

        if args.model:
            denoised_rgb = run_onnx_inference(args.model, input_path)
            denoised_t = np_to_tensor(denoised_rgb)
            entry["denoised_psnr"] = compute_psnr(denoised_t, ref_t)
            entry["denoised_ssim"] = compute_ssim(denoised_t, ref_t)
            denoised_png = os.path.join(images_dir, f"{safe_name}_denoised.png")
            write_png(denoised_rgb, denoised_png)
            entry["denoised_png"] = denoised_png

        results.append(entry)
        print(f"  Noisy  PSNR={noisy_psnr:.2f} dB  SSIM={noisy_ssim:.4f}")
        if "denoised_psnr" in entry:
            print(f"  Denoised PSNR={entry['denoised_psnr']:.2f} dB  SSIM={entry['denoised_ssim']:.4f}")

    # Summary statistics
    noisy_psnrs = [r["noisy_psnr"] for r in results]
    noisy_ssims = [r["noisy_ssim"] for r in results]

    summary = {
        "scene_count": len(results),
        "noisy_psnr_mean": float(np.mean(noisy_psnrs)),
        "noisy_ssim_mean": float(np.mean(noisy_ssims)),
    }

    if has_denoised:
        den_psnrs = [r["denoised_psnr"] for r in results if "denoised_psnr" in r]
        den_ssims = [r["denoised_ssim"] for r in results if "denoised_ssim" in r]
        if den_psnrs:
            summary["denoised_psnr_mean"] = float(np.mean(den_psnrs))
            summary["denoised_ssim_mean"] = float(np.mean(den_ssims))

    # Serializable results (strip paths for JSON)
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if not k.endswith("_png")}
        json_results.append(jr)

    output_json = {
        "summary": summary,
        "scenes": json_results,
    }

    json_path = os.path.join(args.output, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)

    html_path = generate_html_gallery(results, args.output, has_denoised)

    print(f"\nSummary ({len(results)} scenes):")
    print(f"  Noisy  mean PSNR={summary['noisy_psnr_mean']:.2f} dB  SSIM={summary['noisy_ssim_mean']:.4f}")
    if "denoised_psnr_mean" in summary:
        print(f"  Denoised mean PSNR={summary['denoised_psnr_mean']:.2f} dB  SSIM={summary['denoised_ssim_mean']:.4f}")
    print(f"\nResults: {json_path}")
    print(f"Gallery: {html_path}")


if __name__ == "__main__":
    main()
