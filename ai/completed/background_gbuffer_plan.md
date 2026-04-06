# Background Pixel G-Buffer Fixes

> **Status: NOT STARTED**
>
> **Purpose:** Fix four related defects in the G-buffer outputs for background (miss)
> pixels. These defects cause problems for the temporal denoiser (T4/T5) and the
> crop-coverage filter in the training preprocessor. Geometry-hit pixels are
> **not affected** — all changes are scoped strictly to the miss-pixel write block
> in `raygen.rgen` and the coverage check in `preprocess_temporal.py`.

---

## Background and Motivation

When a primary ray misses all geometry, `raygen.rgen` writes sentinel values to
the G-buffer. The current sentinels have four problems:

| Channel | Current value | Problem |
|---|---|---|
| `img_motion_vectors` | `(0, 0)` | Wrong during camera rotation — reprojector sources the wrong env sample |
| `img_world_normals` | `(0, 0, 0, 0)` | No directional signal — model cannot learn per-pixel env map lookup |
| `img_specular_albedo` | `(0, 0, 0, 1)` | Should be dielectric F0 `(0.04, 0.04, 0.04)` matching spec for `diffuse_albedo = (1,1,1)` |
| `preprocess_temporal.py` coverage | `normal magnitude > 0.01` | Will always pass once normal carries ray direction (magnitude ≈ 1) |

The depth sentinel (`kSentinelDepth = 1e4`) is correct and unchanged.
The diffuse albedo `(1, 1, 1)` is correct and unchanged.

---

## Changes Required

### 1. `renderer/src/vulkan/shaders/raygen.rgen` — Miss-pixel G-buffer write block

The block starts at the comment `// Write sentinel G-buffer if nothing was hit (path 0 only)`.

#### 1a. Motion vectors — compute from camera rotation

Replace the hard-coded `(0, 0)` write with a proper screen-space motion vector,
computed identically to the hit-pixel calculation but projecting a point far along
the ray direction so that translation is negligible:

```glsl
// BEFORE:
imageStore(img_motion_vectors, pixel, vec4(0.0));

// AFTER:
vec2 screen_current = (vec2(pixel) + 0.5) / vec2(size);
vec4 clip_prev = frame.prev_view_proj * vec4(origin + direction * 1e6, 1.0);
vec2 screen_prev = clip_prev.xy / clip_prev.w * 0.5 + 0.5;
screen_prev.y = 1.0 - screen_prev.y;  // Flip Y: GLM projection is Y-up, image is Y-down
vec2 bg_motion = screen_current - screen_prev;
imageStore(img_motion_vectors, pixel, vec4(bg_motion, 0.0, 0.0));
```

`origin` and `direction` are already computed in the main body of `main()` before
the path loop and are in scope at this point. `size` is `ivec2(gl_LaunchSizeEXT.xy)`
from the top of `main()`.

The `1e6` multiplier makes the translation component of the view change negligible
relative to the rotation component — the standard technique for skybox motion
vectors. The formula is identical to the hit-pixel path (lines 341–346) so the
reprojection shader receives a consistent representation.

#### 1b. World normals — store normalized ray direction

Replace the zero-vector sentinel with the normalized primary ray direction.
The `.w` component carries roughness for geometry pixels; for background we store
`0.0` (no surface, no roughness):

```glsl
// BEFORE:
imageStore(img_world_normals, pixel, vec4(0.0));

// AFTER:
imageStore(img_world_normals, pixel, vec4(normalize(direction), 0.0));
```

This gives the network a unique, smoothly-varying directional signal for every
background pixel. The depth sentinel (`kSentinelDepth`) remains the canonical
indicator that a pixel is background — the normal channel is now signal, not mask.

#### 1c. Specular albedo — set to dielectric F0

The current value `(0, 0, 0)` is incorrect; it should be the standard dielectric
F0 `(0.04, 0.04, 0.04)` to be consistent with how the demodulation in
`encoder_input_conv.comp` and the remodulation in `output_conv.comp` treat
background pixels:

```glsl
// BEFORE:
imageStore(img_specular_albedo, pixel, vec4(0.0, 0.0, 0.0, 1.0));

// AFTER:
imageStore(img_specular_albedo, pixel, vec4(0.04, 0.04, 0.04, 1.0));
```

The diffuse albedo `vec4(1.0)` is already correct (yields `diffuse = (1,1,1,1)`)
and is unchanged.

#### 1d. Debug visualization — remove motion-vector override for background

The existing debug-clear block unconditionally zeroes `img_noisy_diffuse` for
the `kDebugModeMotionVectors` mode, which hides the background motion. Remove
`kDebugModeMotionVectors` from the condition so background motion is visible in
the debug view. The depth debug and other modes that still show no useful
background data can keep the black override:

```glsl
// BEFORE:
if (pc.debug_mode == kDebugModeDepth || pc.debug_mode == kDebugModeMotionVectors
    || pc.debug_mode >= kDebugModeTransmissionNdotV)
    imageStore(img_noisy_diffuse, pixel, vec4(0.0, 0.0, 0.0, 1.0));

// AFTER:
if (pc.debug_mode == kDebugModeDepth
    || pc.debug_mode >= kDebugModeTransmissionNdotV)
    imageStore(img_noisy_diffuse, pixel, vec4(0.0, 0.0, 0.0, 1.0));
```

The motion-vector debug path now falls through to the default (noisy diffuse =
blurred env color), which is a correct representation of the background.

---

### 2. `training/scripts/preprocess_temporal.py` — Coverage check

The coverage check currently uses `normal magnitude > 0.01` to detect geometry.
Once the normal channel stores the ray direction for background pixels (magnitude
≈ 1.0), all background crops will pass the check. Replace the normal-based test
with a depth-sentinel test at every occurrence.

The depth channel is index 10 in the 19-channel input tensor (`linear_depth`,
channel `depth.Z`). Background pixels have depth exactly `kSentinelDepth = 1e4`.
The threshold `5000.0` is safely below the sentinel and above any real scene
geometry depth.

There are **five** locations to update. All follow the same pattern:

#### Pattern A — PyTorch tensor path (4 occurrences in `_process_one` and `_process_temporal_window`)

```python
# BEFORE (single-frame, full-image path — _process_one, ~line 93):
normals = inp[6:9]  # world normals XYZ (channels 6-8)
coverage = (normals.norm(dim=0) > 0.01).float().mean().item()

# AFTER:
depth = inp[10]  # linear depth (channel 10); background pixels have kSentinelDepth = 1e4
coverage = (depth < 5000.0).float().mean().item()
```

```python
# BEFORE (single-frame, crop path — _process_one, ~line 123):
normals = crop_inp[6:9]  # world normals XYZ
coverage = (normals.norm(dim=0) > 0.01).float().mean().item()

# AFTER:
depth = crop_inp[10]  # linear depth channel
coverage = (depth < 5000.0).float().mean().item()
```

```python
# BEFORE (temporal, full-image path — _process_temporal_window, ~line 178):
normals = stacked_inp[0, 6:9]  # world normals XYZ
coverage = (normals.norm(dim=0) > 0.01).float().mean().item()

# AFTER:
depth = stacked_inp[0, 10]  # linear depth of first frame
coverage = (depth < 5000.0).float().mean().item()
```

```python
# BEFORE (temporal, crop path — _process_temporal_window, ~line 212):
normals = crop_inp[0, 6:9]  # world normals XYZ
coverage = (normals.norm(dim=0) > 0.01).float().mean().item()

# AFTER:
depth = crop_inp[0, 10]  # linear depth of first frame
coverage = (depth < 5000.0).float().mean().item()
```

#### Pattern B — NumPy path (1 occurrence in the verify block, ~line 494)

```python
# BEFORE:
region_normals = src_inp[6:9, cy:cy + crop_size, cx:cx + crop_size]
normal_mag = np.sqrt((region_normals ** 2).sum(axis=0))
coverage = float(np.mean(normal_mag > 0.01))

# AFTER:
region_depth = src_inp[10, cy:cy + crop_size, cx:cx + crop_size]
coverage = float(np.mean(region_depth < 5000.0))
```

#### Update the module-level comment

The docstring near `_MIN_COVERAGE` references the old detection method:

```python
# BEFORE:
# Geometry is detected by non-zero world normal magnitude (input channels 6-8).
# Crops below this threshold are mostly background/sky and add no useful
# denoising signal.

# AFTER:
# Geometry is detected by depth below the sentinel value (input channel 10,
# kSentinelDepth = 1e4). Crops below this threshold are mostly background/sky
# and add no useful denoising signal.
```

---

## Verification

After making these changes, re-run the existing test suite and perform a visual check:

1. **Build and run `monti_view`** with an outdoor scene (large sky coverage). Enable the
   motion-vector debug view (`kDebugModeMotionVectors`) and rotate the camera.
   - **Before fix:** sky pixels show black (zero motion).
   - **After fix:** sky pixels show the environment rotation motion, consistent with
     near-geometry pixels at the horizon.

2. **Check the normal debug view** for background pixels. The sky should display a
   smooth gradient corresponding to the view-space direction, not flat black.

3. **Run `preprocess_temporal.py`** on a scene with a large sky (e.g., Brutalism or
   AbandonedWarehouse). Verify that sky-only crops are still discarded and geometry
   crops are kept at the same rate as before.

4. **Run existing `[deni]` test suite** — no changes to the denoise pipeline itself,
   so all 26+ tests should pass unchanged.

---

## Files to Modify

| File | Changes |
|---|---|
| `renderer/src/vulkan/shaders/raygen.rgen` | Miss-pixel block: motion vectors (1a), world normals (1b), specular albedo (1c), debug clear (1d) |
| `training/scripts/preprocess_temporal.py` | Coverage check: 4× PyTorch tensor paths, 1× NumPy path, module comment |
