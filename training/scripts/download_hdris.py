"""Download diverse CC0 HDRIs from Poly Haven for training data lighting.

Downloads 5 HDRIs spanning indoor/outdoor, warm/cool, bright/dark, and
directional/diffuse lighting axes. Files are saved to training/environments/.

Usage:
    python scripts/download_hdris.py [--output environments/]
"""

import argparse
import os
import sys
import urllib.request

# EXR magic bytes: OpenEXR signature (little-endian 0x01312f76)
_EXR_MAGIC = b"\x76\x2f\x31\x01"

# HDRI names and descriptions
_HDRIS = [
    ("studio_small_09", "Indoor studio, neutral — even diffuse baseline"),
    ("kloppenheim_06", "Overcast outdoor, cool — soft shadows, low contrast"),
    ("sunset_fairway", "Warm directional sunset — strong shadows, warm tones"),
    ("moonlit_golf", "Dim nighttime — low-light noise, high dynamic range"),
    ("royal_esplanade", "Bright cloudy sky — high-key exterior, specular highlights"),
]

_BASE_URL = "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/2k"


def hdri_url(name: str) -> str:
    """Construct the Poly Haven download URL for a 2K HDRI."""
    return f"{_BASE_URL}/{name}_2k.exr"


def hdri_filename(name: str) -> str:
    """Construct the local filename for a downloaded HDRI."""
    return f"{name}_2k.exr"


def validate_exr_magic(path: str) -> bool:
    """Check that a file has valid OpenEXR magic bytes."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        return magic == _EXR_MAGIC
    except OSError:
        return False


def download_hdris(output_dir: str) -> list[str]:
    """Download all HDRIs to output_dir. Skip files that already exist.

    Returns list of downloaded (or already-present) file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for name, description in _HDRIS:
        filename = hdri_filename(name)
        filepath = os.path.join(output_dir, filename)

        if os.path.isfile(filepath):
            if validate_exr_magic(filepath):
                print(f"  [skip] {filename} (already exists)")
                paths.append(filepath)
                continue
            else:
                print(f"  [warn] {filename} exists but has invalid EXR magic — re-downloading")

        url = hdri_url(name)
        print(f"  [download] {filename} — {description}")
        print(f"             {url}")

        try:
            urllib.request.urlretrieve(url, filepath)
        except (urllib.error.URLError, OSError) as e:
            print(f"  [error] Failed to download {filename}: {e}", file=sys.stderr)
            continue

        if not validate_exr_magic(filepath):
            print(f"  [error] {filename} has invalid EXR magic bytes after download",
                  file=sys.stderr)
            os.remove(filepath)
            continue

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  [ok] {filename} ({size_mb:.1f} MB)")
        paths.append(filepath)

    return paths


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(script_dir, "..", "..", "scenes", "environments")

    parser = argparse.ArgumentParser(
        description="Download CC0 HDRIs from Poly Haven for training")
    parser.add_argument("--output", default=default_output,
                        help="Output directory (default: environments/)")
    args = parser.parse_args()

    print("=== HDRI Download ===")
    paths = download_hdris(args.output)
    print(f"\nDone: {len(paths)}/{len(_HDRIS)} HDRIs available in {args.output}")

    if len(paths) < len(_HDRIS):
        print("Warning: Some HDRIs failed to download. Re-run to retry.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
