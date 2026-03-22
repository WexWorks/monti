"""Download glTF sample models for training data generation.

Downloads from the Khronos glTF-Sample-Assets GitHub repository:
  GLB models:
  - DamagedHelmet.glb — PBR textures, normal maps, emissive
  - DragonAttenuation.glb — Transmission, volume attenuation
  - WaterBottle.glb — PBR metal/roughness textures
  - AntiqueCamera.glb — Detailed PBR, small geometry
  - Lantern.glb — PBR wood/metal
  - ToyCar.glb — Clearcoat, transmission, sheen, texture transform
  - ABeautifulGame.glb — Chess set, transmission, volume
  - MosquitoInAmber.glb — Nested transmission, IOR, volume
  - GlassHurricaneCandleHolder.glb — Glass transmission, volume
  - BoomBox.glb — PBR, emissive front panel
  - SheenChair.glb — Sheen, texture transform, fabric

  Multi-file glTF (no GLB variant):
  - FlightHelmet — Multi-mesh PBR, leather/glass
  - Sponza (Crytek) — Large interior, many materials, core PBR

  Deferred (manual download):
  - Intel Sponza — detected and validated if present at scenes/IntelSponza/

Usage:
    python scripts/download_scenes.py [--output scenes/]
"""

import argparse
import json
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
    (
        "WaterBottle.glb",
        f"{_BASE_URL}/WaterBottle/glTF-Binary/WaterBottle.glb",
        1_000_000,
    ),
    (
        "AntiqueCamera.glb",
        f"{_BASE_URL}/AntiqueCamera/glTF-Binary/AntiqueCamera.glb",
        1_000_000,
    ),
    (
        "Lantern.glb",
        f"{_BASE_URL}/Lantern/glTF-Binary/Lantern.glb",
        1_000_000,
    ),
    (
        "ToyCar.glb",
        f"{_BASE_URL}/ToyCar/glTF-Binary/ToyCar.glb",
        1_000_000,
    ),
    (
        "ABeautifulGame.glb",
        f"{_BASE_URL}/ABeautifulGame/glTF-Binary/ABeautifulGame.glb",
        3_000_000,
    ),
    (
        "MosquitoInAmber.glb",
        f"{_BASE_URL}/MosquitoInAmber/glTF-Binary/MosquitoInAmber.glb",
        3_000_000,
    ),
    (
        "GlassHurricaneCandleHolder.glb",
        f"{_BASE_URL}/GlassHurricaneCandleHolder/glTF-Binary/GlassHurricaneCandleHolder.glb",
        1_000_000,
    ),
    (
        "BoomBox.glb",
        f"{_BASE_URL}/BoomBox/glTF-Binary/BoomBox.glb",
        5_000_000,
    ),
    (
        "SheenChair.glb",
        f"{_BASE_URL}/SheenChair/glTF-Binary/SheenChair.glb",
        1_000_000,
    ),
]

# Multi-file glTF scenes: (subdirectory, gltf_filename, base_url_path, min_total_bytes)
_GLTF_SCENES = [
    (
        "FlightHelmet",
        "FlightHelmet.gltf",
        f"{_BASE_URL}/FlightHelmet/glTF",
        4_000_000,  # ~5 MB total
    ),
    (
        "Sponza",
        "Sponza.gltf",
        f"{_BASE_URL}/Sponza/glTF",
        25_000_000,  # ~30 MB total
    ),
]

# Intel Sponza: manual download, detected if present
_INTEL_SPONZA_DIR = "IntelSponza"
_INTEL_SPONZA_GLTF = os.path.join("glTF", "NewSponza_Main_glTF_002.gltf")


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


def _download_gltf_scene(subdir: str, gltf_filename: str, base_url: str,
                          min_total_bytes: int, output_dir: str) -> bool:
    """Download a multi-file glTF scene (gltf + referenced buffers/textures).

    Returns True if the scene is available (downloaded or already present).
    """
    scene_dir = os.path.join(output_dir, subdir)
    gltf_path = os.path.join(scene_dir, gltf_filename)

    # Skip if already downloaded and valid
    if os.path.isfile(gltf_path):
        total_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(scene_dir)
            for f in fns
        )
        if total_size >= min_total_bytes:
            print(f"  {subdir}/: already exists ({total_size:,} bytes total), skipping")
            return True

    os.makedirs(scene_dir, exist_ok=True)

    # Download the .gltf file
    gltf_url = f"{base_url}/{gltf_filename}"
    _download_file(gltf_url, gltf_path)

    if not os.path.isfile(gltf_path):
        print(f"  ERROR: Failed to download {gltf_filename}")
        return False

    # Parse referenced buffers and images
    with open(gltf_path, "r", encoding="utf-8") as f:
        gltf_data = json.load(f)

    referenced_uris: list[str] = []
    for buf in gltf_data.get("buffers", []):
        uri = buf.get("uri", "")
        if uri and not uri.startswith("data:"):
            referenced_uris.append(uri)
    for img in gltf_data.get("images", []):
        uri = img.get("uri", "")
        if uri and not uri.startswith("data:"):
            referenced_uris.append(uri)

    print(f"  {subdir}/: {len(referenced_uris)} referenced files to download")

    for uri in referenced_uris:
        dest = os.path.join(scene_dir, uri)
        dest_dir = os.path.dirname(dest)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

        if os.path.isfile(dest):
            continue

        file_url = f"{base_url}/{uri}"
        try:
            _download_file(file_url, dest)
        except RuntimeError as e:
            print(f"  WARNING: Failed to download {uri}: {e}")

    # Validate total size
    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(scene_dir)
        for f in fns
    )
    if total_size < min_total_bytes:
        print(f"  WARNING: {subdir}/ total size {total_size:,} < "
              f"expected {min_total_bytes:,} bytes")

    print(f"  {subdir}/: OK ({total_size:,} bytes total)")
    return True


def _check_intel_sponza(output_dir: str) -> bool:
    """Check if Intel Sponza has been manually downloaded."""
    sponza_dir = os.path.join(output_dir, _INTEL_SPONZA_DIR)
    gltf_path = os.path.join(sponza_dir, _INTEL_SPONZA_GLTF)
    if os.path.isfile(gltf_path):
        print(f"  IntelSponza: detected at {sponza_dir}")
        return True
    print(f"  IntelSponza: not found (manual download required)")
    return False


def download_scenes(output_dir: str) -> None:
    """Download all training scenes to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    success = 0
    skipped = 0

    # GLB scenes
    print("=== GLB Scenes ===")
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

    # Multi-file glTF scenes
    print("\n=== Multi-file glTF Scenes ===")
    gltf_success = 0
    for subdir, gltf_filename, base_url, min_total in _GLTF_SCENES:
        if _download_gltf_scene(subdir, gltf_filename, base_url, min_total, output_dir):
            gltf_success += 1

    # Intel Sponza detection
    print("\n=== Deferred Scenes ===")
    intel_sponza = _check_intel_sponza(output_dir)

    glb_total = success + skipped
    print(f"\nDone: {glb_total}/{len(_SCENES)} GLB scenes, "
          f"{gltf_success}/{len(_GLTF_SCENES)} glTF scenes"
          + (", Intel Sponza detected" if intel_sponza else ""))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(script_dir, "..", "..", "scenes", "khronos")

    parser = argparse.ArgumentParser(
        description="Download glTF sample models for training data generation")
    parser.add_argument("--output", default=default_output,
                        help="Output directory (default: scenes/khronos/)")
    args = parser.parse_args()

    download_scenes(args.output)


if __name__ == "__main__":
    main()
