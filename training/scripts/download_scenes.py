"""Download glTF sample models for training data generation.

Downloads from the Khronos glTF-Sample-Assets GitHub repository:
  - DamagedHelmet.glb — PBR textures, normal maps, emissive
  - DragonAttenuation.glb — Transmission, volume attenuation

Usage:
    python scripts/download_scenes.py [--output scenes/]
"""

import argparse
import os
import sys
import urllib.request
import urllib.error

# Base URL for Khronos glTF-Sample-Assets (main branch, GLB variants)
_BASE_URL = (
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models"
)

# Scene definitions: (filename, url_path, minimum expected file size in bytes)
_SCENES = [
    (
        "DamagedHelmet.glb",
        f"{_BASE_URL}/DamagedHelmet/glTF-Binary/DamagedHelmet.glb",
        3_000_000,  # ~3.7 MB
    ),
    (
        "DragonAttenuation.glb",
        f"{_BASE_URL}/DragonAttenuation/glTF-Binary/DragonAttenuation.glb",
        5_000_000,  # ~8 MB
    ),
]


def _validate_glb(path: str) -> bool:
    """Validate that a file is a valid GLB by checking the magic bytes."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            return magic == b"glTF"
    except OSError:
        return False


def _download_file(url: str, dest: str) -> None:
    """Download a file from URL to dest with progress reporting."""
    print(f"  Downloading: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "monti-training/1.0"})
        with urllib.request.urlopen(req) as response:
            total = response.headers.get("Content-Length")
            total = int(total) if total else None

            data = bytearray()
            block_size = 65536
            downloaded = 0

            while True:
                block = response.read(block_size)
                if not block:
                    break
                data.extend(block)
                downloaded += len(block)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r  Progress: {downloaded:,} / {total:,} bytes ({pct}%)",
                          end="", flush=True)

            print()

        with open(dest, "wb") as f:
            f.write(data)

    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP error {e.code} downloading {url}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error downloading {url}: {e.reason}") from e


def download_scenes(output_dir: str) -> None:
    """Download all training scenes to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    success = 0
    skipped = 0

    for filename, url, min_size in _SCENES:
        dest = os.path.join(output_dir, filename)

        # Skip if already downloaded and valid
        if os.path.exists(dest):
            file_size = os.path.getsize(dest)
            if file_size >= min_size and _validate_glb(dest):
                print(f"  {filename}: already exists ({file_size:,} bytes), skipping")
                skipped += 1
                continue
            else:
                print(f"  {filename}: exists but invalid (size={file_size:,}), re-downloading")

        _download_file(url, dest)

        # Validate
        file_size = os.path.getsize(dest)
        if file_size < min_size:
            print(f"  WARNING: {filename} is smaller than expected "
                  f"({file_size:,} < {min_size:,} bytes)")

        if not _validate_glb(dest):
            print(f"  ERROR: {filename} is not a valid GLB file!")
            os.remove(dest)
            continue

        print(f"  {filename}: OK ({file_size:,} bytes)")
        success += 1

    total = success + skipped
    print(f"\nDone: {total}/{len(_SCENES)} scenes available "
          f"({success} downloaded, {skipped} already present)")


def main():
    parser = argparse.ArgumentParser(
        description="Download glTF sample models for training data generation")
    parser.add_argument("--output", default="scenes/",
                        help="Output directory for .glb files (default: scenes/)")
    args = parser.parse_args()
    download_scenes(args.output)


if __name__ == "__main__":
    main()
