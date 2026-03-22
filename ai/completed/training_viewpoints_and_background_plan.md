# Training Data: Viewpoint Capture & Transparent Background Plan

> **Purpose:** Improve training data quality by (1) adding manual viewpoint capture to monti_view, (2) enhancing `generate_viewpoints.py` to accept seed viewpoints and generate variations, (3) changing monti_datagen to render background pixels as black/transparent by default with an opt-in flag to render the environment map background, and (4) fixing duplicate viewpoint sampling. Each phase is scoped for a single Copilot session.
>
> **Prerequisites:** F9-6a (viewpoints pipeline) and F9-6b (light rigs) ✅ complete. The `generate_viewpoints.py`, `generate_training_data.py`, and `monti_datagen` are functional. The `monti_view` interactive viewer has fly/orbit camera controls with ImGui panels.
>
> **Note on per-viewpoint exposure:** `monti_datagen` already supports per-viewpoint exposure via the `"exposure"` field in viewpoint JSON entries (`vp.exposure.value_or(config_.exposure)` in `GenerationSession.cpp`). The Python `generate_training_data.py` continues to use exposure as an external data-amplification axis (varying `--exposure` across invocations), which is the right approach for maximizing training variation without additional C++ complexity.
>
> **Motivation:** Current training data has several quality issues:
> - Most scenes use auto-generated orbit viewpoints that are too far from the object (BoomBox is 0.02 units across; orbit radius ≈ 0.5 puts camera 25× further than the object size).
> - Sponza is an interior scene but the orbit camera is placed outside, seeing only a gray cube.
> - Cornell Box orbits include rear views; the canonical view is always from the front.
> - Background pixels dominated by HDRI environment dominate many frames, potentially biasing training toward denoising environment samples rather than path-traced geometry.
> - When N=2 variations are sampled, duplicated viewpoints can occur since uniqueness is only enforced across the full (viewpoint, exposure, rig) tuple, not per-axis.
>
> All of the per-scene framing issues (Sponza interior, Cornell Box front-facing, ToyCar angles, BoomBox close-up) are addressed by manual viewpoint authoring in Phase 1 rather than special-case logic or embedded camera extraction. Two scenes (cornell_box.glb, ToyCar.glb) have embedded cameras in their glTF data, but manually capturing seed viewpoints in monti_view is simpler and more flexible.

---

## Phase 1: Manual Viewpoint Capture in monti_view

### Goal

Add a keybind and UI indicator to `monti_view` that saves the current camera position/target/FOV to a viewpoints JSON file, using the same format and naming convention as `generate_viewpoints.py`.

### File Naming Convention

The output viewpoint file uses the underscored scene name derived from the scene file path, matching the convention in `generate_viewpoints.py::_scene_name_from_path()`:
- `DamagedHelmet.glb` → `damaged_helmet.json`
- `FlightHelmet/FlightHelmet.gltf` → `flight_helmet.json`
- `cornell_box.glb` → `cornell_box.json`

The file is written to the current working directory (or a path specified by `--viewpoints-out`).

### Changes

#### `app/view/CameraController.h`

Add a public method to extract the current viewpoint as a serializable struct:

```cpp
struct SavedViewpoint {
    glm::vec3 position;
    glm::vec3 target;
    float fov_degrees;
};

SavedViewpoint CurrentViewpoint() const;
```

In fly mode, `target` is computed as `position_ + ForwardFromYawPitch(yaw_, pitch_) * orbit_distance_`, using the orbit distance (initialized from the auto-fit camera) to produce a meaningful target depth. In orbit mode, `target` is `orbit_target_`. Exposure is not part of the viewpoint struct — it is read from `PanelState::exposure_ev` by the `SaveViewpoint` function.

#### `app/view/Panels.h` / `PanelState`

Add fields to `PanelState`:

```cpp
// Viewpoint capture
int saved_viewpoint_count = 0;     // Display count in UI
bool viewpoint_just_saved = false; // Flash indicator
std::string viewpoints_out_path;   // Output file path
```

#### `app/view/Panels.cpp`

In the Camera collapsible section, add:
- Display: `"Saved viewpoints: N"` showing `saved_viewpoint_count`
- A brief flash message `"Saved!"` when `viewpoint_just_saved` is true (clear after 1–2 seconds)

#### `app/view/main.cpp`

1. **CLI option:** Add `--viewpoints-out <path>` (optional). If not provided, derive from scene path: extract scene filename, convert to snake_case, append `.json`. Write to current working directory.

2. **Scene name derivation:** Add a helper (or reuse logic matching `_scene_name_from_path`) that converts `DamagedHelmet.glb` → `damaged_helmet`, `FlightHelmet/FlightHelmet.gltf` → `flight_helmet`, etc. Use regex `([a-z0-9])([A-Z])` → `\1_\2` and `([A-Z]+)([A-Z][a-z])` → `\1_\2`, then lowercase.

3. **Keybind `P`:** In `CameraController::OnKeyDown`, add `SDLK_P` case. However, since the controller doesn't own the file I/O, signal via a flag or callback. Better approach: handle `SDLK_P` in the main event loop (before passing to the controller), calling a free function:

```cpp
void SaveViewpoint(const CameraController& controller,
                   float exposure_ev,
                   const std::string& viewpoints_path,
                   PanelState& panel_state);
```

This function:
- Calls `controller.CurrentViewpoint()` to get position/target/fov
- Reads `exposure_ev` from `panel_state.exposure_ev`
- Reads the existing JSON array from `viewpoints_path` (or creates `[]` if file doesn't exist)
- Appends a new entry: `{"position": [...], "target": [...], "fov": ..., "exposure": ...}`
- Writes the file back with `indent=2` formatting
- Increments `panel_state.saved_viewpoint_count`
- Sets `panel_state.viewpoint_just_saved = true`

The exposure is saved so that `monti_datagen` can use per-viewpoint exposure values (already supported via `vp.exposure.value_or(config_.exposure)` in `GenerationSession.cpp`). The background mode is **not** saved — `monti_datagen` always defaults to transparent black background.

4. **Startup:** If the viewpoints file already exists, read it and set `saved_viewpoint_count` to the array length, so the counter reflects previously saved viewpoints.

### Dependencies

- `nlohmann/json` (already used in datagen) — add to monti_view link target if not already linked.

### Testing

- Manual: Launch `monti_view scenes/DamagedHelmet.glb`, navigate to an interesting view, press `P`. Verify `damaged_helmet.json` is created/appended in CWD. Press `P` again from a different angle, verify 2 entries in the file.
- Load the saved viewpoints file in `monti_datagen --viewpoints damaged_helmet.json` and verify the renders match the saved views.

---

## Phase 2: Viewpoint Variation Generation from Seed Files

### Goal

Enhance `generate_viewpoints.py` to accept an optional seed viewpoints file (typically hand-authored via Phase 1) and generate random variations of those viewpoints, producing a richer viewpoint set for training.

### CLI Changes

```
python scripts/generate_viewpoints.py \
    --scenes scenes/ \
    --output viewpoints/ \
    --seeds viewpoints/manual/    # NEW: directory of seed viewpoint JSONs
    --variations-per-seed 4       # NEW: how many variations per seed viewpoint
    --seed-jitter 0.15            # NEW: position jitter as fraction of camera-to-target distance
```

- `--seeds <dir>`: Directory containing seed viewpoint JSON files (same format as output). Files are matched to scenes by name (e.g., `sponza.json` applies to the `sponza` scene).
- `--variations-per-seed <N>`: Target total viewpoints per seed (default: 4). The original seed is always included; additional random variations are generated until the target count is reached. For example, with 3 seeds and `--variations-per-seed 4`, the output contains 12 viewpoints: 3 originals + 9 variations.
- `--seed-jitter <frac>`: Maximum random offset applied to camera position, as a fraction of the camera-to-target distance (default: 0.15 = 15%).

### Variation Strategies

Each original seed viewpoint is included verbatim in the output. Additional variations are generated to reach the target count per seed, using a mix of:

1. **Position jitter:** Random offset to camera position within a sphere of radius `jitter * distance(position, target)`. Target stays fixed (same framing, slightly different angle).

2. **Target jitter:** Small random offset to the look-at target (±5% of camera-to-target distance). Camera position stays fixed (same location, slightly different framing).

3. **Interpolation between seeds:** If there are ≥2 seed viewpoints, generate interpolated views by blending position and target between randomly selected pairs. Use a random `t ∈ [0.2, 0.8]` to avoid duplicating endpoints.

4. **Orbit perturbation:** Convert the seed viewpoint to spherical coordinates around the target, then perturb azimuth (±15°), elevation (±10°), and distance (±10%). Convert back to Cartesian.

Each variation strategy is selected randomly (uniform) per generated viewpoint. The FOV from the seed is preserved (or jittered ±2°).

### Integration with Existing Pipeline

- When `--seeds` is provided, seed-derived viewpoints **replace** the auto-generated orbit viewpoints for scenes that have a matching seed file. Scenes without seed files still use the existing auto-generation.
- The auto-generated orbit viewpoints remain the default when `--seeds` is not specified, so the existing workflow is unchanged.
- Output format is identical: `viewpoints/<scene_name>.json` with `[{"position": [...], "target": [...], "fov": ...}, ...]`.

### Cornell Box Cleanup

The current `_CORNELL_BOX_CONFIG` uses `center: [0.0, 1.0, 0.0]` which is incorrect — the box AABB is `[0,0,0] → [1,1,1]`. When seed viewpoints are provided for `cornell_box`, they replace the hardcoded config entirely. The `_CORNELL_BOX_CONFIG` special case in `generate_viewpoints_for_scene()` should be removed or bypassed when seeds are available.

---

## Phase 3: Transparent Background for Training Data

### Goal

Change `monti_datagen` to render background (miss) pixels as black with zero alpha by default. Add `--env-background [blur]` opt-in flag to render the environment map as background with configurable blur level.

### Rationale

- Training crops that contain only background pixels can be detected (all alpha = 0) and skipped, ensuring the denoiser sees meaningful geometry.
- Eliminates potential bias from environment map samples dominating the training signal.
- The environment map still contributes to **lighting** (indirect illumination, specular reflections) — only the directly-visible background pixels change.
- The opt-in `--env-background` with blur level provides flexibility: blur level 0 = sharp environment, higher values = progressively blurred, matching the existing `sampleEnvironmentBlurred` 9-tap Gaussian approach but with wider kernels.

### Shader Changes (`shaders/raygen.rgen`)

#### New Uniform

Add `background_mode` to `FrameUniforms` (both C++ and GLSL):

```glsl
// In frame_uniforms.glsl — replaces one of the existing pad fields
uint  background_mode;  // 0 = transparent black, 1 = environment map
```

```cpp
// In FrameUniforms.h — replace pad0
uint32_t background_mode;  // 0 = transparent black, 1 = environment
```

The existing `skybox_mip_level` field already controls blur level and is currently hardcoded to `0.0f` in `Renderer.cpp`. We repurpose it: when `background_mode == 1`, `skybox_mip_level` controls how blurred the background appears.

#### Primary Miss Handling

In the raygen shader, modify the primary-ray miss block (~line 149–167):

```glsl
if (payload.missed) {
    if (bounce == 0 && transparent_count == 0) {
        if (frame.background_mode == 1u) {
            // Environment background enabled — sample with configured blur
            vec3 env_color = sampleEnvironmentBlurred(
                env_map, ray_dir, frame.skybox_mip_level, frame.env_rotation);
            path_radiance += throughput * env_color;
        }
        // else: background_mode == 0 → add nothing (black)
    } else {
        // Bounced miss — always sample environment for lighting correctness
        vec3 env_color = textureLod(
            env_map, directionToUVRotated(ray_dir, frame.env_rotation),
            kEnvMapBounceLod).rgb;
        path_radiance += throughput * env_color;
    }

    // Write sentinel G-buffer for background
    if (!wrote_primary && path == 0) {
        imageStore(img_motion_vectors, pixel, vec4(0.0));
        imageStore(img_linear_depth, pixel, vec4(kSentinelDepth, kSentinelDepth, 0.0, 0.0));
        imageStore(img_world_normals, pixel, vec4(0.0, 0.0, 1.0, 0.0));
        imageStore(img_diffuse_albedo, pixel, vec4(0.0));
        imageStore(img_specular_albedo, pixel, vec4(kDielectricF0, kDielectricF0, kDielectricF0, 1.0));
        if (pc.debug_mode == kDebugModeDepth || pc.debug_mode == kDebugModeMotionVectors)
            imageStore(img_noisy_diffuse, pixel, vec4(0.0, 0.0, 0.0, 1.0));
        wrote_primary = true;
    }
    break;
}
```

#### Alpha Channel

Change the final image store to write alpha based on whether the pixel hit geometry:

```glsl
// Track whether any path hit geometry (set during primary hit processing)
// Already tracked: wrote_primary is true when primary hit wrote G-buffer data
float alpha = wrote_primary ? 1.0 : 0.0;

// For background_mode == 1, background pixels still get alpha = 1.0
// (they have valid color data from the environment)
if (frame.background_mode == 1u)
    alpha = 1.0;

imageStore(img_noisy_diffuse, pixel, vec4(final_diffuse, alpha));
imageStore(img_noisy_specular, pixel, vec4(final_specular, alpha));
```

Note: `wrote_primary` currently indicates whether G-buffer was written. We need a separate `bool hit_geometry` flag that is set when the first opaque/transparent hit occurs (not on miss). This is distinct from `wrote_primary` which also gets set on primary miss (for sentinel G-buffer data). Concretely, add:

```glsl
bool hit_geometry = false;
```

Set `hit_geometry = true` at the first opaque hit (inside the main bounce loop, after the `if (first_opaque)` block). Use this to determine alpha:

```glsl
float alpha = hit_geometry ? 1.0 : 0.0;
if (frame.background_mode == 1u) alpha = 1.0;
```

### C++ Changes

#### `renderer/src/vulkan/FrameUniforms.h`

Replace `pad0` with `background_mode`:

```cpp
struct FrameUniforms {
    // ... existing fields ...
    uint32_t area_light_count;       // offset 224
    uint32_t background_mode;       // offset 228 (was pad0)
    uint32_t pad1;                   // offset 232
    uint32_t pad2;                   // offset 236
};
```

Size remains 240 bytes, alignment unchanged.

#### `renderer/src/vulkan/Renderer.cpp`

Where `FrameUniforms` is populated (~line 284):

```cpp
fu.skybox_mip_level = config_.skybox_blur_level;  // was 0.0f
fu.background_mode = config_.show_environment_background ? 1u : 0u;
```

#### Renderer Config

Add to `RendererDesc` or a separate runtime config struct:

```cpp
bool show_environment_background = false;  // Default: transparent black
float skybox_blur_level = 0.0f;            // Mip level for env background blur
```

#### `app/datagen/main.cpp`

Add CLI option:

```cpp
std::optional<float> env_background_blur;
app.add_option("--env-background", env_background_blur,
               "Render environment map as background with optional blur level (default: off/transparent)");
```

When `--env-background` is specified:
- Set `show_environment_background = true` on the renderer config
- Set `skybox_blur_level = env_background_blur.value_or(0.0f)` (no argument = sharp, numeric argument = blur mip level)

When `--env-background` is **not** specified (the default):
- `show_environment_background = false`
- Background pixels render as RGBA `(0, 0, 0, 0)` — black and fully transparent

#### `app/view/main.cpp`

In monti_view, **keep** the current behavior (environment background visible) for interactive use. Set `show_environment_background = true` and `skybox_blur_level = 0.0f` by default. Optionally expose the blur level as a slider in the Settings panel under the Render section.

### Environment Blur

The existing `sampleEnvironmentBlurred` function (`sampling.glsl:173`) provides a 9-tap Gaussian approximation at one mip level below the target, which is sufficient for the `--env-background` opt-in. The `skybox_mip_level` field in `FrameUniforms` (currently hardcoded to `0.0f`) controls the blur amount. No new blur implementation is needed — the existing function is reused as-is.

### EXR Writer Changes

The capture `Writer` currently writes `diffuse.A` and `specular.A` channels. Verify that:
- These channels faithfully capture the alpha value from `img_noisy_diffuse` / `img_noisy_specular`
- The EXR output preserves the 0.0 alpha for background pixels

No changes should be needed if the writer already passes through the RGBA16F data. The training script can use `diffuse.A == 0` (or `depth.Z == kSentinelDepth`) to identify background pixels in crops.

### Training Script Integration

After this phase, `validate_dataset.py` and the training pipeline can:
- Use alpha = 0 to compute a "geometry coverage" percentage per crop
- Skip or down-weight crops where geometry coverage is below a threshold (e.g., < 10%)
- Display geometry coverage in the validation gallery HTML

---

## Implementation Order

| Phase | Scope | Key Files |
|-------|-------|-----------|
| **1** | Viewpoint capture keybind in monti_view | `CameraController.h/cpp`, `Panels.h/cpp`, `app/view/main.cpp` |
| **2** | Seed viewpoints + variation generation | `scripts/generate_viewpoints.py`, `tests/test_viewpoints.py` |
| **3** | Transparent background default + env-background opt-in | `raygen.rgen`, `frame_uniforms.glsl`, `FrameUniforms.h`, `Renderer.cpp`, `datagen/main.cpp` |

Phases 1 and 2 are the highest priority — they directly unblock authoring good viewpoints for all 14 training scenes. Phase 3 improves training data quality.

Phases 1 and 3 are independent and can be implemented in any order. Phase 2 depends on Phase 1 (the seed viewpoints it consumes are authored via Phase 1).

> **Note:** Phase 4 (duplicate-free variation sampling) was rendered obsolete when the exposure/rig cross-product amplification was removed — viewpoints are now rendered 1:1.
