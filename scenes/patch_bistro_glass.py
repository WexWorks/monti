#!/usr/bin/env python3
"""Patch Cauldron-Media BistroInterior scene.gltf to add KHR_materials_transmission,
KHR_materials_volume, and KHR_materials_ior extensions to glass materials.

The original Cauldron-Media BistroInterior asset predates these glTF extensions,
so glass materials (wine glasses, bottles, windows, display cases) are authored as
fully opaque PBR.  This script injects physically-plausible transmission, IOR, and
volume-attenuation properties so a path tracer can render them as refractive glass.

Usage:
    python patch_bistro_glass.py                       # auto-detect scene path
    python patch_bistro_glass.py <path/to/scene.gltf>  # explicit path

The script is idempotent: re-running it on an already-patched file is a no-op.
A .bak backup is created only on the first run.
"""

import json
import os
import shutil
import sys

# ── Material patches ──────────────────────────────────────────────────────
# Each entry maps a material name to the transmission properties to inject.

GLASS_PATCHES = {
    # Clear architectural glass (windows, display cases, picture frames)
    "TransparentGlass.DoubleSided": {
        "transmission_factor": 0.95,
        "ior": 1.5,
        "attenuation_color": [0.98, 0.98, 0.98],
        "attenuation_distance": 0.5,
    },
    "MASTER_Glass_Exterior": {
        "transmission_factor": 0.95,
        "ior": 1.5,
        "attenuation_color": [0.95, 0.97, 0.95],
        "attenuation_distance": 0.5,
    },
    "Paris_Paintings_Glass": {
        "transmission_factor": 0.95,
        "ior": 1.5,
        "attenuation_color": [0.98, 0.98, 0.98],
        "attenuation_distance": 0.5,
    },
    "Paris_Cashregister_Glass": {
        "transmission_factor": 0.95,
        "ior": 1.5,
        "attenuation_color": [0.98, 0.98, 0.98],
        "attenuation_distance": 0.5,
    },
    # Frosted / textured glass
    "MASTER_Interior_01_Frozen_Glass": {
        "transmission_factor": 0.85,
        "ior": 1.5,
        "attenuation_color": [0.95, 0.95, 0.95],
        "attenuation_distance": 0.3,
    },
    # Wine glasses / drinkware
    "Paris_LiquorBottle_01_Glass_Wine": {
        "transmission_factor": 0.95,
        "ior": 1.5,
        "attenuation_color": [0.95, 0.98, 0.95],
        "attenuation_distance": 0.2,
    },
    # Liquor / wine bottles (slightly tinted glass)
    "Paris_LiquorBottle_01_Glass": {
        "transmission_factor": 0.90,
        "ior": 1.5,
        "attenuation_color": [0.85, 0.92, 0.85],
        "attenuation_distance": 0.1,
    },
    "Paris_LiquorBottle_02_Glass": {
        "transmission_factor": 0.90,
        "ior": 1.5,
        "attenuation_color": [0.85, 0.92, 0.85],
        "attenuation_distance": 0.1,
    },
    "Paris_LiquorBottle_03_Glass": {
        "transmission_factor": 0.90,
        "ior": 1.5,
        "attenuation_color": [0.85, 0.92, 0.85],
        "attenuation_distance": 0.1,
    },
}

REQUIRED_EXTENSIONS = [
    "KHR_materials_ior",
    "KHR_materials_transmission",
    "KHR_materials_volume",
]


def patch_material(mat, patch):
    """Add KHR_materials_transmission/ior/volume extensions to a material dict."""
    if "extensions" not in mat:
        mat["extensions"] = {}

    mat["extensions"]["KHR_materials_transmission"] = {
        "transmissionFactor": patch["transmission_factor"],
    }
    mat["extensions"]["KHR_materials_ior"] = {
        "ior": patch["ior"],
    }
    if "attenuation_color" in patch and "attenuation_distance" in patch:
        mat["extensions"]["KHR_materials_volume"] = {
            "attenuationColor": patch["attenuation_color"],
            "attenuationDistance": patch["attenuation_distance"],
            "thicknessFactor": 0.0,
        }


def is_already_patched(scene):
    """Return True if every target material already has a transmission extension."""
    mat_by_name = {m.get("name", ""): m for m in scene.get("materials", [])}
    for name in GLASS_PATCHES:
        mat = mat_by_name.get(name)
        if mat is None:
            continue
        exts = mat.get("extensions", {})
        if "KHR_materials_transmission" not in exts:
            return False
    return True


def find_scene_path():
    """Auto-detect the BistroInterior scene.gltf relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "scenes", "extended", "Cauldron-Media",
                     "BistroInterior", "scene.gltf"),
        os.path.join(script_dir, "..", "scenes", "extended", "Cauldron-Media",
                     "BistroInterior", "scene.gltf"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.normpath(c)
    return None


def main():
    # Resolve scene path
    if len(sys.argv) > 1:
        scene_path = sys.argv[1]
    else:
        scene_path = find_scene_path()
    if not scene_path or not os.path.isfile(scene_path):
        print(f"Error: scene.gltf not found (tried: {scene_path})", file=sys.stderr)
        sys.exit(1)

    # Load
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = json.load(f)

    # Idempotency check
    if is_already_patched(scene):
        print(f"Already patched: {scene_path}")
        return

    # Backup (only if no backup exists yet)
    backup_path = scene_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(scene_path, backup_path)
        print(f"Backed up original to {backup_path}")

    # Patch materials
    patched_count = 0
    for mat in scene.get("materials", []):
        name = mat.get("name", "")
        if name in GLASS_PATCHES:
            patch_material(mat, GLASS_PATCHES[name])
            patched_count += 1
            patch = GLASS_PATCHES[name]
            print(f"  Patched: {name} "
                  f"(transmission={patch['transmission_factor']}, ior={patch['ior']})")

    # Update extensionsUsed (don't add to extensionsRequired — these are optional)
    existing = set(scene.get("extensionsUsed", []))
    existing.update(REQUIRED_EXTENSIONS)
    scene["extensionsUsed"] = sorted(existing)

    # Write
    with open(scene_path, "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2)

    print(f"\nPatched {patched_count} materials in {scene_path}")
    print(f"extensionsUsed: {scene['extensionsUsed']}")


if __name__ == "__main__":
    main()
