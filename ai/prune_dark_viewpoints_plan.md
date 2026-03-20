# Plan: Remove Invalid Viewpoints from Training Data

## Summary

Create a pipeline to:
1. Add a unique 8-hex-char ID to each viewpoint definition for stable identification.
2. Simplify training file names to `<scene>_<id>_{input,target}.exr`.
3. Write a new script (`remove_invalid_viewpoints.py`) that scans rendered EXRs, detects invalid images (initially: near-black), moves the invalid EXR files to a sibling directory, removes the corresponding viewpoints from the viewpoint JSON files, and logs removed viewpoints for manual inspection and potential restoration.

### Context: Data Amplification Architecture

All data amplification — exposure levels, lighting rigs, environment maps — is being moved into the viewpoint definition itself (parallel effort). Each amplified variant gets its own viewpoint entry with its own unique ID. `generate_training_data.py` renders each viewpoint 1:1 with no additional amplification loops. This means every UUID maps to exactly one EXR pair, and removing a viewpoint removes exactly one rendered image.

The only remaining post-render amplification (e.g., image rotation) happens outside this pipeline.

---

## 1. Add Unique ID to Viewpoint Definitions

### 1a. `monti_view` (C++ — `app/view/main.cpp`)

The `SaveViewpoint()` function currently writes `{position, target, fov, exposure}`. It needs to also write an `"id"` field.

**Approach:** Generate an 8-character lowercase hex string using `<random>` with `std::mt19937` seeded from `std::random_device`, plus `std::uniform_int_distribution`. Add `entry["id"] = hex_str;` to the JSON serialisation.

**Format:** 8-character lowercase hex string (e.g., `"a3f1c0b2"`). 32 bits of entropy is more than sufficient for uniqueness within a single scene's viewpoint file (typically <1000 entries).

**Files changed:**
- `app/view/main.cpp` — `SaveViewpoint()`: generate and emit `"id"` field.

### 1b. `generate_viewpoints.py`

All viewpoint-generating functions (`compute_orbit_viewpoints`, `compute_hemisphere_viewpoints`, variation functions, etc.) produce dicts with `{position, target}` (plus optional `fov`, `exposure`, etc.). Each needs an `"id"` field.

**Approach:** Generate IDs centrally in `generate_all_viewpoints()` after the full viewpoint list for a scene is assembled, rather than inside each helper function. Use `uuid.uuid4().hex[:8]`.

**Collision handling:** After generating all IDs for a scene, verify uniqueness. On the extremely unlikely collision, regenerate the colliding ID.

**Files changed:**
- `training/scripts/generate_viewpoints.py` — add ID assignment after viewpoints are generated, before writing JSON.

No backfill script is needed. All viewpoints and training data will be regenerated from scratch after this update.

---

## 2. Simplify Training File Names

### Current naming

**Sampled mode** (`--max-variations`): `<scene>_ev<exposure>_vp<index>[_<rig>]_{input,target}.exr`
Example: `a_beautiful_game_ev+1.0_vp3_warm_input.exr`

**Full mode**: directory-based `<scene>/ev_<exposure>/vp_<index>/{input,target}.exr`

### New naming

All modes produce flat output: `<scene>_<id>_{input,target}.exr`

Examples:
- `a_beautiful_game_a3f1c0b2_input.exr`
- `a_beautiful_game_a3f1c0b2_target.exr`

The ID alone is sufficient to uniquely map back to the viewpoint definition. All render metadata (exposure, environment map, light rig, etc.) lives in the viewpoint JSON entry and is recoverable via ID lookup.

### Files changed

- `training/scripts/generate_training_data.py`:
  - Since all data amplification (exposure, rigs, envs) now lives in the viewpoint definitions, the generation script simplifies to iterating viewpoints 1:1. Each viewpoint has a unique ID; the output file is named `<scene>_<id>_{input,target}.exr`.
  - `monti_datagen` still writes to temp `vp_{i}/` subdirectories per invocation; the script renames files to the flat `<scene>_<id>` convention when moving to the final output directory.
  - The `_EXPOSURES` loop, `_format_exposure()`, sampled-mode vs full-mode distinction, and related exposure/rig amplification code can be removed (or simplified) since amplification now happens upstream in viewpoint generation.

- `training/deni_train/data/exr_dataset.py`: The `ExrDataset` already supports flat `*_input.exr` naming via glob. The new `<scene>_<id>_input.exr` → `<scene>_<id>_target.exr` pattern is compatible with the existing pair-matching logic. No changes required.

- `training/tests/test_generate_training_data.py`: Update tests that assert on the old naming format or the old exposure-loop structure.

---

## 3. New Script: `remove_invalid_viewpoints.py`

### Purpose

Scan a directory of rendered training EXR files, identify images that fail validation checks, and:
1. Move the invalid EXR files (both input and target) to a sibling directory for manual inspection.
2. Remove the corresponding viewpoints from the source viewpoint JSON files.
3. Log removed viewpoints (full original JSON) to a sibling directory for auditability and potential restoration.

The script is named generically (`remove_invalid_viewpoints.py`) because additional heuristic checks beyond near-black detection will be added in the future.

### Usage

```
python scripts/remove_invalid_viewpoints.py \
    --training-data <dir> \
    --viewpoints-dir <dir> \
    [--threshold <float>]  \
    [--dark-fraction <float>] \
    [--dry-run]
```

### Validation Check: Near-Black Detection

1. **Load the target EXR** (the high-SPP reference render). If the reference is near-black, the viewpoint is genuinely useless regardless of noise in the input.
2. **Read the combined radiance**: load `diffuse.{R,G,B}` + `specular.{R,G,B}`, sum them per-pixel to get total radiance `(R, G, B)`.
3. **Compute per-pixel luminance**: `L = 0.2126*R + 0.7152*G + 0.0722*B` (Rec. 709).
4. **Classify pixels as "dark"**: a pixel is dark if `L < threshold` (default: `0.001`, effectively black in linear HDR space).
5. **Classify image as "near-black"**: if the fraction of dark pixels exceeds `dark_fraction` (default: `0.98`, i.e. 98% of pixels are dark), the image fails validation.

**Rationale for defaults:** Noise and fireflies in path-traced images mean some pixels will have non-zero values even in genuinely dark renders. A 98% threshold accommodates scattered bright pixels while catching viewpoints that are pointing at nothing (facing away from the scene, inside geometry, at a completely unlit region).

### File-to-Viewpoint Mapping

With the naming convention `<scene>_<id>_{input,target}.exr`:
1. Parse the file name: extract `<scene>` and `<id>` from the stem (split on `_`, last segment before `_{input,target}` is the ID, everything before that is the scene name).
2. Load `<viewpoints-dir>/<scene>.json`.
3. Find the entry where `entry["id"] == <id>`.
4. If the image fails any validation check, mark for removal.

### Handling Invalid Files

**EXR files — move to sibling directory:**

If the training data is in `training_data/`, invalid files are moved to `invalid_training_data/` (sibling with `invalid_` prefix). The internal directory structure (if any) is preserved.

Example:
- `training_data/a_beautiful_game_84fd8ce2_input.exr` → `invalid_training_data/a_beautiful_game_84fd8ce2_input.exr`
- `training_data/a_beautiful_game_84fd8ce2_target.exr` → `invalid_training_data/a_beautiful_game_84fd8ce2_target.exr`

Both the input and target EXR are moved together, even though only the target was checked.

**Viewpoint entries — remove and log:**

Removed viewpoints are written to `invalid_viewpoints/<scene>.json` (sibling of the `viewpoints/` directory). Each file contains a JSON array of the full original viewpoint entries that were removed. This allows restoring individual viewpoints after manual inspection.

Example:
- Removed from: `viewpoints/antique_camera.json`
- Logged to: `invalid_viewpoints/antique_camera.json`

If `invalid_viewpoints/antique_camera.json` already exists from a previous run, the newly removed entries are **appended** to the existing array (avoiding loss of earlier removal logs).

### `--dry-run` Mode

Print which viewpoints would be removed and which files would be moved, without modifying anything.

### Files added

- `training/scripts/remove_invalid_viewpoints.py`
- `training/tests/test_remove_invalid_viewpoints.py`

---

## 4. Implementation Order

1. **Phase 1 — ID infrastructure**:
   - 1a: `monti_view` C++ change (add `"id"` to `SaveViewpoint`).
   - 1b: `generate_viewpoints.py` (add ID assignment).
2. **Phase 2 — File naming**: Update `generate_training_data.py` to use `<scene>_<id>` flat naming and remove the exposure/rig amplification loops (since amplification is now in viewpoint definitions).
3. **Phase 3 — Validation script**: Implement `remove_invalid_viewpoints.py` with near-black detection, file moving, and viewpoint logging.
4. **Phase 4 — Tests**: Add/update tests for all changed and new code.

---

## 5. Future Extensions

The `remove_invalid_viewpoints.py` script is designed to accommodate additional validation heuristics. Potential future checks:

- **Low-contrast detection**: images where max-min luminance range is extremely narrow (flat grey).
- **NaN/Inf detection**: images containing corrupted pixel values.
- **Geometric degeneracy**: camera inside geometry (could be detected by depth buffer analysis if the input EXR includes depth).
- **Saturation detection**: images where most pixels are fully saturated (blown highlights).

Each check would follow the same pattern: scan EXR → classify as invalid → move files + remove viewpoint + log.
