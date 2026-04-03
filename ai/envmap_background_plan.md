# Environment Map Background Pixels Plan

## Goal

Render the environment map into background (miss) pixels so that every pixel in the training data has a valid color. This eliminates the hit_mask from training, removes the zero-gradient dead zone for background pixels, and provides the VGG perceptual loss with coherent full-image features.

## Key Decisions

- **Remove `background_mode` entirely** from the shader and C++ API. The environment map is always sampled for miss pixels. When no environment map is loaded, the renderer's existing 1×1 black placeholder texture produces black background — consistent with a real all-black env map.
- **Keep all G-buffer formats as RGBA16F.** `VK_FORMAT_R16G16B16_SFLOAT` (3-channel, no alpha) has near-zero hardware support for storage images. Alpha stays in the format but is always set to 1.0 for all pixels (hit and miss). No format changes to avoid conflicts with the temporal denoiser work (T2–T5).
- **Remove `hit_mask` completely** from the training pipeline (datasets, loss, evaluation, safetensors conversion). The `hit > 0.5` conditionals in the denoiser shaders (`encoder_input_conv.comp`, `output_conv.comp`) are also removed — demodulation and remodulation become unconditional.
- **Add configurable environment blur** via `--env-blur` CLI argument (default 3.5) and UI slider in monti_view, and `--env-blur` CLI default + per-viewpoint `environmentBlur` JSON override in monti_datagen.
- **Auto-exposure**: Replace the `alpha < 0.5` background skip with skipping exactly-zero-luminance pixels. The existing `L < 1e-4` threshold in `luminance.comp` stays as supplementary noisy-pixel protection.
- **Training data is regenerated.** No backward compatibility with old 7-channel target format.

## Current State

**Shader** (`renderer/src/vulkan/shaders/raygen.rgen`):
- Primary miss with `background_mode == 0`: transparent black (nothing written to diffuse/specular)
- Primary miss with `background_mode == 1`: env map color added to `path_radiance`, routed to `total_diffuse` (since `is_specular_path` defaults to `false`)
- G-buffer sentinels for miss pixels (lines 196–209):
  - `img_diffuse_albedo`: `vec4(0.0)` — zero albedo
  - `img_specular_albedo`: `vec4(kDielectricF0, kDielectricF0, kDielectricF0, 1.0)`
  - `img_world_normals`: `vec4(0.0, 0.0, 1.0, 0.0)` — canonical up normal
  - `img_linear_depth`: `vec4(kSentinelDepth, kSentinelDepth, 0.0, 0.0)` — 1e4
  - `img_motion_vectors`: `vec4(0.0)`
- Alpha: `pixel_alpha` stays 0.0 for miss → `diffuse.A = 0.0` is the hit_mask source

**Datagen** (`app/datagen/main.cpp`):
- `background_mode` is set based on `environmentBlur` field in viewpoint JSON
- When absent, defaults to `background_mode = 0` (transparent black)

**Training pipeline**:
- `hit_mask` extracted from `diffuse.A` (>0.5 = hit)
- Demodulation skipped for miss pixels (`np.where(hit_bool, demod, raw)`)
- Loss masked by `hit_mask` — both L1 and VGG terms zero out background
- `convert_to_safetensors.py` stores hit_mask as channel 6 of the 7-channel target
- Evaluation metrics (PSNR, SSIM) computed on full image, unmasked

**Env map sampling** (`renderer/src/vulkan/shaders/include/sampling.glsl`):
- `sampleEnvironmentBlurred()` already implements 9-tap Gaussian using offset samples at `base_mip = max(0, mip_level - 1)` with cardinal (weight 0.125) and diagonal (weight 0.0625) taps around a center tap (weight 0.25)
- Controlled by `frame.skybox_mip_level` which maps to `Renderer::SetBackgroundMode(bool, float blur_level)`

**Denoiser shaders** (`denoise/src/vulkan/shaders/`):
- `encoder_input_conv.comp`: reads `noisy_diffuse.a` as hit mask, gates demodulation with `if (hit > 0.5)`
- `output_conv.comp`: reads `noisy_diffuse.a` as hit mask, gates remodulation with `if (hit > 0.5)`
- `passthrough_denoise.comp`: sums diffuse + specular, preserves alpha

**Temporal reprojection** (T3 ✅, `denoise/src/vulkan/shaders/reproject.comp`):
- Warps previous frame's denoised diffuse/specular to current frame using motion vectors
- Depth-based disocclusion detection (10% tolerance)
- Reads `prev_diffuse`, `prev_specular`, `prev_depth`, `curr_depth`, `motion_vectors` — **no alpha or hit_mask dependency**
- History images (`FrameHistory` in `MlInference.h`) store copies of denoised outputs after remodulation
- `DepthwiseSeparableConvBlock` added to PyTorch `blocks.py` (T2 ✅) — no data format dependency

**Auto-exposure** (`capture/src/Luminance.cpp`, `app/shaders/luminance.comp`):
- Both skip pixels with `alpha < 0.5` to exclude background from geometric mean

**No environment map loaded**: Renderer creates a 1×1 black placeholder RGBA16F texture. All env map samples return `(0,0,0,0)`.

## Design

### Background Pixel Values After Change

| Channel | Current Miss Value | New Miss Value |
|---------|-------------------|----------------|
| `img_noisy_diffuse` RGB | 0 or env×intensity | env×intensity (always) |
| `img_noisy_diffuse` A | 0.0 (miss flag) | 1.0 (always) |
| `img_noisy_specular` RGB | 0 | 0 (no specular for sky) |
| `img_noisy_specular` A | 0.0 | 1.0 |
| `img_diffuse_albedo` | (0,0,0,0) | **(1,1,1,1)** — unit albedo |
| `img_specular_albedo` | (0.04,0.04,0.04,1) | **(0,0,0,1)** — no specular |
| `img_world_normals` | (0,0,1,0) | **(0,0,0,0)** — zero normal |
| `img_linear_depth` | (1e4, 1e4, 0, 0) | (1e4, 1e4, 0, 0) — keep sentinel |
| `img_motion_vectors` | (0,0) | (0,0) — unchanged |

**Rationale**:
- `diffuse_albedo = (1,1,1)`: demodulation becomes `irradiance = radiance / 1.0 = radiance`, so the env map color passes through cleanly. No division-by-zero risk.
- `specular_albedo = (0,0,0)`: background has no specular surface. Zero albedo is safe because demodulation uses `max(a, DEMOD_EPS)` where `DEMOD_EPS = 0.001`, so division computes `0.0 / 0.001 = 0.0` — no NaN risk. Remodulation computes `~0 × 0.001 ≈ 0`, which also suppresses any small model prediction errors that leak into background specular.
- `diffuse.A = 1.0`: marks every pixel as "valid" → hit_mask is always 1.0 → no masking needed in loss or evaluation.
- `normals = (0,0,0,0)`: zero-length normal distinguishes sky from geometry (roughness in W = 0). The normal channels [6-8] in the 19-channel input become (0,0,0) for background, which the model can learn as "sky indicator."
- `depth = kSentinelDepth`: kept as-is since it's already distinct from any real geometry depth. The depth channel [10] in the input will have this sentinel, which is fine as a sky indicator.

### No Environment Map Loaded

When no environment map is specified, the renderer uses a 1×1 black placeholder texture. With this plan:
- Miss pixels get `env_color × intensity = (0,0,0) × intensity = (0,0,0)` — black background
- Alpha is still 1.0, albedo is still (1,1,1)
- Auto-exposure skips exactly-zero-luminance pixels, so black background doesn't drag exposure down
- The blur slider is disabled in the monti_view UI when no env map is loaded (blur of black is still black)

### Demodulation After Change

With `diffuse_albedo = (1,1,1)` for background, demodulation becomes unconditional everywhere:
```
demod_irradiance = env_radiance / max(albedo_d, eps) = env_radiance / 1.0 = env_radiance
```
No special-casing needed. The `hit_bool` conditional is removed from all code paths:
- `exr_dataset.py`, `convert_to_safetensors.py`: remove `np.where(hit_bool, demod, raw)`
- `encoder_input_conv.comp`: remove `if (hit > 0.5)` gate on demodulation
- `output_conv.comp`: remove `if (hit > 0.5)` gate on remodulation

### Training Loss After Change

With every pixel valid (`hit_mask = 1.0` everywhere):
- L1: `valid_count = total_pixels * channels` — standard mean L1
- VGG: `pred_vgg = normalize(pred_rad_tm)` — no masking, natural images throughout
- The `hit_mask` parameter is removed from `DenoiserLoss.forward()`

### Reference Target

For pure primary miss pixels, the noisy and reference (high-SPP) frames will produce the **same** env map color because the lookup is deterministic (same pixel → same ray direction → same texel). The noisy/target pair will be identical for background — the model learns identity for these trivial pixels, contributing near-zero loss after convergence.

---

## Phase 1: Shader — Remove `background_mode`, Always Write Env Map ✅ COMPLETE

**Scope**: Remove the `background_mode` uniform entirely. Always sample the environment map for miss pixels using a configurable blur level (`bg_env_mip_level`). Write unit diffuse albedo, zero specular albedo, zero normals, and alpha=1.0 for all pixels. Simplify denoiser shaders to always demodulate/remodulate.

### Files Modified

1. **`renderer/src/vulkan/shaders/raygen.rgen`** (lines 155–209, 878)
   - In the primary miss block (line 155–170): Remove the `if (frame.background_mode == 1u)` conditional. Always sample environment map for miss pixels using `bg_env_mip_level` uniform:
     ```glsl
     vec3 env_color;
     if (frame.bg_env_mip_level > 0.0) {
         env_color = sampleEnvironmentBlurred(
             env_map, ray_dir, frame.bg_env_mip_level, frame.env_rotation);
     } else {
         env_color = textureLod(
             env_map, directionToUVRotated(ray_dir, frame.env_rotation), 0.0).rgb;
     }
     vec3 bg_radiance = env_color * frame.env_intensity;
     ```
     Write `bg_radiance` to `total_diffuse`. (Already routed correctly since `is_specular_path` defaults to `false`.)
   - In the G-buffer sentinel block (lines 196–209):
     - `img_diffuse_albedo`: change from `vec4(0.0)` to `vec4(1.0, 1.0, 1.0, 1.0)`
     - `img_specular_albedo`: change from `vec4(kDielectricF0, kDielectricF0, kDielectricF0, 1.0)` to `vec4(0.0, 0.0, 0.0, 1.0)`
     - `img_world_normals`: change from `vec4(0.0, 0.0, 1.0, 0.0)` to `vec4(0.0, 0.0, 0.0, 0.0)`
     - `img_linear_depth`: keep `vec4(kSentinelDepth, kSentinelDepth, 0.0, 0.0)` (unchanged)
     - `img_motion_vectors`: keep `vec4(0.0)` (unchanged)
   - Final imageStore (~line 878): Always write `alpha = 1.0`. Remove the `(frame.background_mode == 1u) ? 1.0 : pixel_alpha` conditional. Remove `pixel_alpha` variable entirely.

2. **`renderer/src/vulkan/shaders/include/frame_uniforms.glsl`**
   - Remove `uint background_mode;` field.
   - Add `float bg_env_mip_level;` (replaces `_pad` slot or adjusts alignment). Must maintain std140 alignment.

3. **`renderer/src/vulkan/FrameUniforms.h`** (C++ side)
   - Remove `uint32_t background_mode;`.
   - Add `float bg_env_mip_level = 3.5f;` to match the GLSL struct. Default is 3.5 (not 0.0) so that background pixels are always blurred, even before Phases 2–3 wire up the CLI/slider override.

4. **`renderer/src/vulkan/Renderer.cpp`**
   - Replace `SetBackgroundMode(bool show_environment, float blur_level)` with `SetEnvironmentBlur(float mip_level)`.
   - Remove `show_environment_background` impl field. Rename `skybox_blur_level` to `env_blur_level`.
   - Update frame uniform mapping: remove `fu.background_mode = ...`, add `fu.bg_env_mip_level = impl_->env_blur_level;`.

5. **`renderer/include/monti/vulkan/Renderer.h`**
   - Replace `void SetBackgroundMode(bool show_environment, float blur_level = 0.0f);` with `void SetEnvironmentBlur(float mip_level);`.

6. **`renderer/src/vulkan/shaders/include/constants.glsl`**
   - No changes. `kSentinelDepth = 1e4` is retained.

7. **`denoise/src/vulkan/shaders/encoder_input_conv.comp`**
   - Remove hit mask read: delete `float hit = d.a;` and `float hit = s.a;` lines.
   - Remove conditional demodulation: change `return (hit > 0.5) ? raw / max(a, DEMOD_EPS) : raw;` to unconditional `return raw / max(a, DEMOD_EPS);` for both diffuse and specular channels.

8. **`denoise/src/vulkan/shaders/output_conv.comp`**
   - Remove hit mask read: delete `float hit = imageLoad(noisy_diffuse, ivec2(x, y)).a;`.
   - Remove conditional remodulation: change `if (hit > 0.5) { ... }` to unconditional remodulation:
     ```glsl
     diffuse_rad  *= max(albedo_d, vec3(DEMOD_EPS));
     specular_rad *= max(albedo_s, vec3(DEMOD_EPS));
     ```
   - Remove `noisy_diffuse` image binding (binding 3) if no longer needed for any other purpose.

### Tests

**C++ tests** (new test file: `tests/background_gbuffer_test.cpp`):

1. **"Background pixel writes unit diffuse albedo"**: Render a scene with known miss pixels (e.g., empty scene or camera points away from geometry). Read back `img_diffuse_albedo` for a miss pixel. Assert RGB ≈ (1.0, 1.0, 1.0).

2. **"Background pixel writes zero specular albedo"**: Same setup. Read back `img_specular_albedo`. Assert RGB ≈ (0.0, 0.0, 0.0).

3. **"Background pixel writes zero normal"**: Read back `img_world_normals`. Assert XYZW = (0.0, 0.0, 0.0, 0.0).

4. **"Background pixel alpha is 1.0"**: Read back `img_noisy_diffuse`. Assert A = 1.0 for miss pixel.

5. **"Background pixel contains env map color"**: Load a scene with a known solid-color environment map (e.g., uniform RGB = (0.5, 0.3, 0.1), intensity = 2.0). Render. Read back `img_noisy_diffuse` RGB for a miss pixel. Assert RGB ≈ (1.0, 0.6, 0.2) = env_color × env_intensity.

6. **"Background pixel depth is sentinel"**: Read back `img_linear_depth`. Assert RG = (kSentinelDepth, kSentinelDepth).

7. **"Background env blur 0.0 produces sharp env color"**: Render with `bg_env_mip_level = 0.0`. Read back miss pixel. Assert it matches `textureLod(env, uv, 0.0) * intensity`.

8. **"Background env blur produces smoothed env color"**: Render with `bg_env_mip_level = 4.0`. Read back miss pixel. Assert it differs from the sharp lookup (blurred value should be closer to the mean env color).

9. **"Frame uniform alignment is valid"**: Static assert that `offsetof(FrameUniforms, bg_env_mip_level)` matches the expected std140 offset. Verify total struct size is a multiple of 16 bytes.

10. **"Denoiser golden reference still passes"**: Re-run the existing `[deni][numerical][golden]` test to confirm the unconditional demodulation/remodulation in the denoiser shaders produces correct output. If the golden reference was generated with mixed hit/miss input, regenerate it with alpha=1.0 everywhere.

---

## Phase 2: monti_view — Add `--env-blur` CLI + UI Slider ✅ COMPLETE

**Scope**: Expose the environment blur setting in the viewer application via both a command-line argument and an interactive UI slider. The blur value is written to viewpoint JSON when recording camera paths.

### Files Modified

1. **`app/view/main.cpp`**
   - Add `--env-blur` CLI option (float, default 3.5). Initialize `panel_state.env_blur` from CLI value.
   - Call `renderer->SetEnvironmentBlur(panel_state.env_blur)` each frame in the main loop so slider changes take effect immediately.
   - Set `panel_state.has_env_map` based on whether a custom env map was loaded via `--env` (not the default grey placeholder).
   - Viewpoint JSON capture: write `"environmentBlur"` field from `panel_state.env_blur` when recording paths (alongside existing `environmentRotation` and `environmentIntensity`).

2. **`app/view/Panels.h`**
   - Add `float env_blur = 3.5f;` to `PanelState`.
   - Add `bool has_env_map = false;` to `PanelState`.

3. **`app/view/Panels.cpp`**
   - Add "Env Blur" slider (range 0.0–8.0) near the existing "Env Rotation" and "Env Intensity" sliders.
   - Disable (grey out) the slider when `panel_state.has_env_map == false`.

### Tests

Manual verification:
1. Start `monti_view` with `--env-blur 2.0 --env some.exr` → verify slider shows 2.0, background is blurred.
2. Move slider → background blur changes in real-time.
3. Start without `--env` → slider is disabled/greyed out, background is black.
4. Record a camera path (P key) → verify `"environmentBlur"` appears in the viewpoint JSON.

---

## Phase 3: monti_datagen — Per-Viewpoint Blur + CLI Default

**Scope**: Add `--env-blur` CLI argument to monti_datagen as a default blur level. Per-viewpoint `environmentBlur` in the JSON overrides the CLI default. Blur is applied per-frame (not globally from the first viewpoint as before).

### Files Modified

1. **`app/datagen/main.cpp`**
   - Add `--env-blur` CLI option (float, default 3.5). This sets the default blur for viewpoints that don't specify `environmentBlur` in their JSON entry.
   - In the per-viewpoint render loop: for each viewpoint, check `vp.environment_blur.has_value()`. If yes, use that value; otherwise use the CLI default. Call `renderer->SetEnvironmentBlur(blur_level)` before each render.
   - Remove the old `show_env_background` variable and `SetBackgroundMode()` calls entirely.

### Tests

1. **Per-viewpoint blur**: Create viewpoint JSON with two entries — one with `"environmentBlur": 0.0`, one with `"environmentBlur": 5.0`. Render both. Verify the first produces sharp background, the second produces blurred background.
2. **CLI default**: Run with `--env-blur 2.0` and a viewpoint JSON that has no `environmentBlur` field. Verify blur level 2.0 is applied.
3. **JSON override of CLI**: Run with `--env-blur 2.0` and a viewpoint with `"environmentBlur": 5.0`. Verify blur level 5.0 is applied for that viewpoint.

---

## Phase 4: Remove hit_mask from Auto-Exposure ✅ COMPLETE

**Scope**: Replace the alpha-based background pixel exclusion in luminance computation with a luminance-based exclusion. With alpha always 1.0, the old `alpha < 0.5` check is dead code. Skip exactly-zero-luminance pixels instead, which handles the case where no environment map is loaded (black background).

### Files Modified

1. **`capture/src/Luminance.cpp`**
   - Remove `float alpha = diffuse_f32[base + 3]; if (alpha < 0.5f) continue;` check.
   - Add `if (L == 0.0f) continue;` after computing luminance, to skip exactly-zero-luminance pixels.

2. **`app/shaders/luminance.comp`**
   - Remove `if (rgba.a < 0.5) return;` line.
   - The existing `if (L < 1e-4) return;` already handles near-zero pixels. This threshold subsumes exactly-zero luminance, so no additional check is strictly needed. However, `L < 1e-4` skips dim but non-zero pixels too — this is intentional existing behavior for noisy-pixel protection and remains unchanged.

3. **`tests/luminance_test.cpp`**
   - Update "background pixels skipped" test: background pixels now have alpha=1.0 and zero luminance (black env). They are excluded by the luminance threshold, not by alpha. Update assertions accordingly.

---

## Phase 5: Remove hit_mask from Training Pipeline

**Scope**: Remove the conditional demodulation (always demodulate), remove hit_mask from the loss function, datasets, and evaluation. The training data format changes from 7-channel target to 6-channel target (no hit_mask channel). Datasets return 4-tuples instead of 5-tuples.

### Files Modified

1. **`training/scripts/convert_to_safetensors.py`**
   - Remove `_HIT_MASK_CHANNEL` import and usage.
   - Remove `hit_bool = hit_mask > 0.5` and all `np.where(hit_bool, ...)` conditional demodulation. Demodulate unconditionally: `raw_d / np.maximum(albedo_d, _DEMOD_EPS)`.
   - Store target as 6 channels: remove `hit_mask[np.newaxis]` from `np.concatenate`. Target shape changes from (7, H, W) to (6, H, W).

2. **`training/deni_train/data/exr_dataset.py`**
   - Remove `_HIT_MASK_CHANNEL = "diffuse.A"` constant.
   - Remove `hit_bool = hit_mask > 0.5` computation.
   - Change demodulation from conditional `np.where(hit_bool, demod, raw)` to unconditional.
   - Remove hit_mask from target concatenation: target becomes (6, H, W).
   - Return 4-tuple: `(input, target, albedo_d, albedo_s)` instead of 5-tuple.

3. **`training/deni_train/data/safetensors_dataset.py`**
   - Remove hit_mask extraction from target channel 6.
   - Target is now (6, H, W) directly.
   - Return 4-tuple: `(input, target, albedo_d, albedo_s)`.

4. **`training/deni_train/losses/denoiser_loss.py`**
   - Remove `hit_mask` parameter from `forward()`.
   - L1 loss: standard `F.l1_loss(pred_tm, tgt_tm)` — no masking.
   - VGG loss: remove `* hit_mask` from `pred_vgg` and `tgt_vgg`. Simplify to `pred_vgg = self._normalize_imagenet(pred_rad_tm)`.

5. **`training/deni_train/train.py`**
   - Update dataloader unpacking: `for inp, tgt, albedo_d, albedo_s in train_loader:`.
   - Remove `hit_mask` from loss call: `loss = loss_fn(pred, tgt, albedo_d, albedo_s)`.
   - Update validation function similarly.

6. **`training/deni_train/evaluate.py`**
   - Update dataset unpacking to 4-tuple.
   - Metrics are already unmasked (no change needed to metrics.py).

7. **`training/deni_train/data/transforms.py`**
   - Verify transforms work with the new target shape (6 channels instead of 7). The `RandomRotation180` transform operates on the `(input, target)` tuple — confirm it handles the 6-channel target correctly.

### Backward Compatibility

The target tensor format changes from (7, H, W) to (6, H, W). Old safetensors files with 7-channel targets must be re-converted from the new background data, so backward compatibility with old files is not needed — the entire dataset is regenerated.

### Tests

**Python tests** (update existing + add new):

1. **Update `training/tests/test_dataset.py`**:
   - `test_target_tensor_shape`: assert `(6, H, W)`.
   - `test_input_tensor_shape`: unchanged (still 19 channels).
   - Remove hit_mask dtype and value assertions.
   - Add new test: **"demodulation is unconditional"**: Create synthetic data with background pixels (albedo=(1,1,1)). Verify that demodulated irradiance = radiance/albedo for all pixels including background.

2. **Update `training/tests/test_safetensors_dataset.py`**:
   - Mirror the ExrDataset test changes for SafetensorsDataset.
   - Assert 4-tuple return instead of 5-tuple.

3. **Update `training/tests/test_convert_to_safetensors.py`**:
   - Assert target has 6 channels.
   - Verify demodulation is applied to background pixels.

4. **New test: `training/tests/test_loss_no_mask.py`**:
   - **"L1 loss on uniform prediction is zero"**: `pred == target` → loss = 0.
   - **"L1 loss is correct for known difference"**: Construct pred and target with known pixel differences. Verify L1 matches expected value without masking.
   - **"VGG loss uses full image"**: Verify VGG features are extracted from the full image (no zero-masking).
   - **"Loss gradient flows to all pixels"**: Run forward+backward on a batch with background pixels. Verify gradients are non-zero for background pixel positions.

5. **Update `training/tests/test_train_safetensors.py`**:
   - Update any 5-tuple unpacking to 4-tuple.
   - Verify training loop runs without hit_mask.

6. **Update `training/tests/test_evaluate_safetensors.py`**:
   - Update evaluate call if it passes hit_mask-related arguments.

---

## Impact on Temporal Denoiser Plan (T2–T5)

### Implemented — verified zero impact:
- **T2 ✅** (depthwise separable PyTorch blocks): Zero impact — pure PyTorch `DepthwiseSeparableConvBlock` in `blocks.py`, no data format dependency.
- **T3 ✅** (motion reprojection): Zero impact — `reproject.comp` reads motion vectors and depth only, never reads alpha or hit_mask. History images (`prev_diffuse`, `prev_specular`, `prev_depth`) store post-remodulation outputs; with alpha=1.0 always, the `vec4` passthrough in reprojection is unaffected. `temporal_reproject_test.cpp` tests identity/shift/disocclusion — none depend on alpha.

### Not yet implemented — spec updates needed:
- **T4 training data**: Target becomes `(8, 6, H, W)` not `(8, 7, H, W)` — hit_mask channel removed. `preprocess_temporal.py` removes hit_mask from target concatenation.
- **T4 model architecture**: `DeniTemporalResidualNet` output stays 7ch (delta_d + delta_s + blend_weight) — **unaffected**. This "7" is the network output head, not the target data format.
- **T5 `temporal_encoder_input_conv.comp`**: Remove "gated by hit mask" demodulation — always demodulate (simplified).
- **T5 `temporal_output_conv.comp`**: Remove `hit > 0.5` conditional remodulation — always remodulate (simplified).
- **T5 golden reference tests**: Update channel count assertions for 6-channel targets.

T4 and T5 are not yet implemented, so these are spec updates only — no merge conflicts.

---

## Verification

1. **Shader**: Render scene with miss pixels → verify `diffuse.A = 1.0` everywhere, diffuse RGB = env_color × intensity for miss pixels, albedo = (1,1,1) for miss.
2. **Blur**: Render with blur=0 vs blur=4 → verify visual difference in miss pixel colors.
3. **No env map**: Render without env map → verify miss pixels are black, alpha=1.0, albedo=(1,1,1), blur slider disabled.
4. **UI**: Start monti_view with `--env-blur 2.0 --env some.exr` → verify slider shows 2.0, moving slider changes background sharpness in real-time.
5. **Datagen per-viewpoint**: Create viewpoint JSON with different `environmentBlur` values per entry → verify each render uses correct blur.
6. **Auto-exposure**: Scene with large black sky → verify auto-exposure doesn't underexpose (zero-luminance excluded).
7. **Denoiser**: Run `[deni][numerical][golden]` tests → unconditional demod/remod produces correct output.
8. **Training**: Regenerate small dataset with new shader → run `convert_to_safetensors.py` → verify 6-channel target → run 1 training epoch → no errors.
9. **Grep verification**: `grep -rn "hit_mask\|hit_bool\|background_mode" training/ capture/ app/ renderer/ denoise/` → no remaining references (except in comments/docs/plans).
10. **C++ tests**: `background_gbuffer_test.cpp` and updated `luminance_test.cpp` pass.
11. **Python tests**: All training tests pass with 4-tuple / 6-channel assertions.
