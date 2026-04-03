# Adaptive Sampling for Reference Rendering

## Overview

Add adaptive sampling to the datagen reference accumulation pipeline. Track per-pixel temporal variance using Welford's online algorithm in log-luminance space, with 3×3 spatial-max smoothing of the convergence decision. Converged pixels are masked out in both the raygen shader (skip tracing) and the accumulate shader (skip accumulation). Per-pixel sample counts enable correct normalization at finalization.

**All variance computation, convergence checking, and sample counting happens on the GPU.** The CPU only dispatches compute shaders and optionally reads back a single uint32 convergence counter for progress reporting.

---

## Current Pipeline

**Datagen reference rendering** (`GenerationSession::RenderReference()`):
1. For each of N reference frames: raygen dispatches all pixels at M SPP → writes noisy diffuse/specular (RGBA16F)
2. `accumulate.comp`: `accum += noisy × (1/N)` for all pixels (uniform weight)
3. After N frames: readback RGBA32F accumulators (already contain the final mean)

---

## New Pipeline

### Datagen Mode (Multi-Frame Accumulation)

For each reference frame:
1. **Raygen** reads convergence mask → skips converged pixels (early-out `return`) → traces M SPP for active pixels only
2. **`accumulate.comp`** reads mask → for active pixels: `accum += sample`, `sample_count++`; for converged: skip
3. **`variance_update.comp`** reads noisy output → for active pixels: computes log-luminance from diffuse+specular, updates Welford's mean/M2
4. **Every K frames** (default 4): **`convergence_check.comp`** evaluates relative error from Welford stats, takes 3×3 spatial max, writes convergence mask; atomically counts converged pixels for progress reporting
5. **After all frames**: **`finalize.comp`** divides `accum / sample_count` per pixel → readback normalized result

### Viewer Mode (Deferred — Future Phase F24-V)

Adaptive sampling in `monti_view` is **deferred** to a follow-up phase. The viewer's rendering model is fundamentally different from datagen: it renders single interactive frames and relies on the ML denoiser for temporal filtering. Adding adaptive sampling to the viewer requires a separate progressive accumulation infrastructure.

**Shared code:** The renderer library changes (raygen binding 17, `PushConstants.adaptive_mask`, `Renderer::SetConvergenceMask()`, `Renderer::SetAdaptiveSamplingEnabled()`, dummy 1×1 mask image) are shared between both apps and ship as part of F24. These ~60 LOC in the renderer are needed for datagen and will be available to the viewer at zero additional cost.

**Viewer-specific work (deferred):** Full viewer integration would require ~150–200 LOC of additional logic:
- Instantiate and manage a `GpuAccumulator` in the viewer
- Detect camera stationarity (delta-based, similar to path tracking)
- Route accumulated output to tonemap, bypassing the denoiser
- Reset accumulator on camera move, resize, scene reload, denoiser mode switch, SPP/bounce change
- Only accumulate when the denoiser is in **Passthrough** mode; disable accumulation and grey out the "Adaptive Sampling" checkbox when the ML denoiser is active (the ML denoiser handles its own temporal filtering)
- Display the converged accumulated result via tonemap, or fall back to the normal single-frame path when not enough frames have accumulated

**Why defer:** This is a cleanly separable feature with its own test surface. The datagen work is where the initial value is — speeding up reference image generation. The viewer benefits are visible only when the camera is stationary in Passthrough mode, which is a niche use case. Deferring keeps each session focused and testable.

---

## Variance Metric Design

### Domain: Log-Luminance

$$x = \ln(\max(L,\; 10^{-7}))$$

where $L = 0.2126R + 0.7152G + 0.0722B$ from combined diffuse+specular.

**Rationale:** Operating in log-luminance space ensures equal sensitivity across the dynamic range. A 1% noise fluctuation in a bright pixel (L=100) and a 1% noise fluctuation in a dark pixel (L=0.01) produce the same log-space delta. Without this, bright pixels would dominate the variance signal and converge too slowly while dark pixels converge prematurely (hiding shadow noise).

### Algorithm: Welford's Online Variance (Per-Pixel, GPU-Side)

```
n += 1
delta = x - mean
mean += delta / n
delta2 = x - mean
M2 += delta * delta2
```

Welford's algorithm is numerically stable for online computation — it avoids catastrophic cancellation that affects the naive `sum_of_squares - mean²` approach. Each pixel maintains three values: `n` (uint32), `mean` (float32), `M2` (float32).

### Convergence Criterion (Evaluated Every K Frames)

```
variance = M2 / max(n - 1, 1)
standard_error = sqrt(variance / n)
relative_error = standard_error / max(abs(mean), 0.01)
spatial_max = max(relative_error in 3×3 neighborhood)
converged = (spatial_max < threshold) AND (n >= min_frames)
```

### Bias Mitigations

| Concern | Mitigation |
|---------|------------|
| Premature convergence on insufficient data | **Minimum frame count** (default 16). No pixel can converge before accumulating enough samples for a stable variance estimate. |
| Bright-pixel bias | **Log-luminance domain.** Equal relative noise produces equal log deltas regardless of absolute luminance level. |
| Seam artifacts at convergence boundaries | **3×3 spatial max.** A pixel cannot converge unless all 8 neighbors also have low relative error. Prevents isolated frozen pixels adjacent to noisy ones. |
| Near-zero luminance instability | **Clamped denominator** (`max(abs(mean), 0.01)`). Prevents division-by-near-zero for very dark pixels (shadows, occluded areas). |
| Conservative threshold | Default 0.02 (2% relative standard error of the mean). Empirically tuned to be conservative — reference images should be visually indistinguishable from non-adaptive output. |

---

## Implementation Sessions

The implementation is split into 4 sessions, each designed to fit within a single Copilot context window. Every session produces testable, working code.

---

### Session 1: Accumulator Refactor — Raw-Sum + Finalize Normalization ✅ DONE

**Goal:** Change the accumulator from weighted-mean to raw-sum accumulation with a separate finalize normalization pass. This is a prerequisite for adaptive sampling (where different pixels have different sample counts) but ships as a backward-compatible refactor.

**Changes:**

1. **New shader: `capture/shaders/finalize.comp`**
   - 16×16 workgroups
   - Bindings: 0: `u_accum_diffuse` (read-write, rgba32f), 1: `u_accum_specular` (read-write, rgba32f), 2: `u_sample_count` (readonly, r32ui)
   - Logic: `accum /= sample_count` per pixel (handle count=0 → write black)

2. **New GPU image: `sample_count_` (R32UI) in `GpuAccumulator`**
   - Create alongside existing accum images
   - `Reset()` clears to zero
   - Exposed via new image view

3. **Modify `accumulate.comp`**
   - Remove `float weight` push constant
   - Add binding 4: `u_sample_count` (read-write, r32ui)
   - Change to raw sum: `accum += sample` (no weight multiplication)
   - Increment sample_count for every pixel

4. **Modify `GpuAccumulator` C++ class**
   - Add sample_count image creation/destruction
   - Add finalize compute pipeline
   - Extended descriptor set layout for new bindings
   - New method: `FinalizeNormalized(const ReadbackContext& ctx)` — dispatches finalize.comp, then reads back
   - `Accumulate(VkCommandBuffer cmd)` — no longer takes `float weight`
   - `Reset()` — also clears sample_count to zero

5. **Modify `GenerationSession::RenderReference()`**
   - Remove `weight` argument from `Accumulate()` calls
   - Call `FinalizeNormalized()` instead of `Finalize()`

6. **Update root `CMakeLists.txt`** — add `finalize.comp` to `CAPTURE_SHADER_SOURCES`

**Test: Non-Adaptive Backward Compatibility** (`tests/adaptive_sampling_test.cpp`)
- Render Cornell Box at 64×64, `ref_frames=4`, non-adaptive
- Verify output matches the old weighted-accumulation path within floating-point tolerance (<1e-5 per pixel)
- Verify all pixels have `sample_count == ref_frames`

**Files modified:** `capture/include/monti/capture/GpuAccumulator.h`, `capture/src/GpuAccumulator.cpp`, `capture/shaders/accumulate.comp`, `app/datagen/GenerationSession.cpp`, `CMakeLists.txt`
**Files created:** `capture/shaders/finalize.comp`, `tests/adaptive_sampling_test.cpp`

---

### Session 2: Raygen Early-Out + Variance Shaders

**Goal:** Add the convergence mask binding to the raygen shader (renderer library) and implement the two variance tracking compute shaders (capture library). The convergence mask is always all-zero at this point — no pixels are ever marked converged yet — so the raygen early-out has no effect, but the infrastructure is in place.

**Changes:**

1. **Add binding 17 to `RaytracePipeline`**
   - `VK_DESCRIPTOR_TYPE_STORAGE_IMAGE`, stage `VK_SHADER_STAGE_RAYGEN_BIT_KHR`
   - R8UI convergence mask (readonly in raygen)
   - Add `convergence_mask_view` to `DescriptorUpdateInfo`
   - Create 1×1 all-zero dummy mask image in the renderer (default when adaptive disabled)
   - Update descriptor set layout and descriptor writes

2. **Expand `PushConstants`** — add `uint32_t adaptive_mask` (0=disabled, 1=read mask). Size: 20 bytes.

3. **Modify `raygen.rgen`**
   - Add `layout(set = 0, binding = 17, r8ui) uniform readonly uimage2D img_convergence_mask;`
   - At top of `main()`: if `pc.adaptive_mask != 0u` and mask pixel != 0, `return` immediately

4. **Add `Renderer` public API**
   - `SetConvergenceMask(VkImageView view)` — sets mask for next descriptor update
   - `SetAdaptiveSamplingEnabled(bool enabled)` — controls push constant flag

5. **New shader: `capture/shaders/variance_update.comp`**
   - Reads noisy diffuse+specular, computes log-luminance, runs Welford's update
   - Skips converged pixels (reads convergence mask)

6. **New shader: `capture/shaders/convergence_check.comp`**
   - Evaluates relative error from Welford stats
   - 3×3 spatial max of relative error
   - Writes convergence mask=1 if below threshold AND count ≥ min_frames
   - Atomic counter for converged pixel count

7. **Add 3 new GPU images to `GpuAccumulator`**
   - `variance_mean_` (R32F), `variance_m2_` (R32F), `convergence_mask_` (R8UI)
   - Atomic counter buffer (`converged_count_`)
   - Create/destroy, clear in `Reset()`

8. **Add `GpuAccumulator` methods and pipelines**
   - `UpdateVariance(VkCommandBuffer cmd)` — dispatches variance_update.comp
   - `CheckConvergence(VkCommandBuffer cmd, uint32_t min_frames, float threshold, const ReadbackContext& ctx)` — dispatches convergence_check.comp, reads back atomic counter
   - `ConvergenceMaskView()` / `ConvergenceMaskImage()` getters
   - New descriptor sets and compute pipelines for the 2 new shaders

9. **Modify `accumulate.comp`** — add binding 5: `u_convergence_mask` (readonly, r8ui). Skip converged pixels.

10. **Add `GpuAccumulatorDesc` fields** — `bool adaptive_sampling = false`

11. **Update root `CMakeLists.txt`** — add `variance_update.comp`, `convergence_check.comp` to `CAPTURE_SHADER_SOURCES`

**Test: Raygen Early-Out + Variance Pipeline Smoke Test** (`tests/adaptive_sampling_test.cpp`)
- Create GpuAccumulator with adaptive enabled
- Render 4 frames with variance updates after each
- Verify variance_mean and variance_m2 are non-zero for non-trivial scene
- Verify convergence mask is still all-zero (min_frames not reached with only 4 frames)
- Verify final image output unchanged (early-out mask all-zero → no pixels skipped)

**Test: Welford Numerical Stability (CPU Unit Test)** (`tests/welford_test.cpp`)
- No GPU needed
- Constant sequence → variance = 0
- Linear sequence → variance matches std::
- Large-offset sequence (1e6 + small noise) → no catastrophic cancellation
- Single element → M2 = 0, mean = x

**Files modified:** `renderer/src/vulkan/RaytracePipeline.h`, `renderer/src/vulkan/RaytracePipeline.cpp`, `renderer/src/vulkan/shaders/raygen.rgen`, `renderer/include/monti/vulkan/Renderer.h`, `renderer/src/vulkan/Renderer.cpp`, `capture/include/monti/capture/GpuAccumulator.h`, `capture/src/GpuAccumulator.cpp`, `capture/shaders/accumulate.comp`, `CMakeLists.txt`
**Files created:** `capture/shaders/variance_update.comp`, `capture/shaders/convergence_check.comp`, `tests/welford_test.cpp`

---

### Session 3: Datagen Integration + CLI + E2E Tests

**Goal:** Wire adaptive sampling into the datagen pipeline end-to-end. Add CLI flags, modify the `RenderReference()` loop, add convergence reporting, and write the full integration tests.

**Changes:**

1. **Extend `GenerationConfig`**
   - `bool adaptive_sampling = false`
   - `uint32_t convergence_check_interval = 4`
   - `uint32_t min_convergence_frames = 16`
   - `float convergence_threshold = 0.02f`

2. **Modify `GenerationSession::RenderReference()`**
   - When `config_.adaptive_sampling && accumulator_`:
     - Set renderer convergence mask and enable adaptive mode
     - Per-frame loop: `RenderFrame()` → barrier → `Accumulate()` → barrier → `UpdateVariance()` → conditional `CheckConvergence()` every K frames
     - Print convergence progress: `[frame N/M] converged: X% (Y/Z pixels)`
     - Early termination if 100% converged
     - After loop: `FinalizeNormalized()` (dispatches normalize + readback)
     - Disable adaptive mode and clear mask
   - When adaptive disabled: existing flow (raw-sum accumulate + finalize — from Session 1)

3. **Add convergence reporting to viewpoint timing JSON**
   - `converged_pixel_fraction`, `adaptive_speedup`, `actual_pixel_frames`, `max_possible_pixel_frames`
   - Per-viewpoint summary line: `render reference: Xms (N frames, Y% converged, Z× speedup)`

4. **Add CLI flags to `monti_datagen` main.cpp**
   - `--adaptive` (bool flag)
   - `--convergence-threshold` (float, default 0.02)
   - `--convergence-interval` (uint, default 4)
   - `--min-convergence-frames` (uint, default 16)

**Test: Flat Color Scene Convergence** (`tests/adaptive_sampling_test.cpp`)
- Render flat-colored Cornell Box at 64×64, `ref_frames=32`, `min_convergence_frames=8`, `convergence_check_interval=4`
- Assert all pixels converge before frame 32
- Assert final image matches non-adaptive reference within <0.5% relative error per pixel
- Assert per-pixel sample counts ≥ `min_convergence_frames`

**Test: Early Termination** (`tests/adaptive_sampling_test.cpp`)
- Flat scene, `ref_frames=64`, `min_convergence_frames=8`, low threshold
- Assert loop terminates early (actual frames < ref_frames)
- Assert final image matches 64-frame non-adaptive reference within tolerance
- Assert `adaptive_speedup` > 2.0

**Test: High-Contrast Scene** (`tests/adaptive_sampling_test.cpp`)
- Cornell Box with bright emissive light + dark shadow, `ref_frames=48`
- Assert both bright and dark regions converge (log-luminance no-bias check)
- Assert final image vs non-adaptive reference: max per-pixel relative error < 1%

**Files modified:** `app/datagen/GenerationSession.h`, `app/datagen/GenerationSession.cpp`, `app/datagen/main.cpp`, `tests/adaptive_sampling_test.cpp`

---

### Session 4 (Future): Viewer Progressive Accumulation (F24-V)

**Goal:** Add progressive accumulation with adaptive sampling to `monti_view`. **Deferred** — the renderer API and raygen binding from Session 2 are already in place. This session adds the viewer-specific accumulation loop.

**Preconditions:** Only accumulate when ALL of:
- Camera is stationary (position + target unchanged for ≥1 frame)
- Adaptive sampling toggle is enabled in settings panel
- Denoiser mode is **Passthrough** (not ML)

The "Adaptive Sampling" checkbox in the settings panel is greyed out (`ImGui::BeginDisabled`) when the ML denoiser is active.

**Changes:**

1. **Add `PanelState` field** — `bool adaptive_sampling = false;`

2. **Add ImGui checkbox** in `DrawSettingsPanel()`
   - Checkbox "Adaptive Sampling" wrapped in `BeginDisabled(!state.denoiser_mode == kPassthrough)`
   - When ML denoiser is active: checkbox disabled and unchecked

3. **Add `--adaptive` CLI flag** to `monti_view`

4. **Add progressive accumulation to viewer render loop**
   - Create GpuAccumulator on demand (first time adaptive is used)
   - Detect camera stationarity: compare position/target/up to previous frame
   - When stationary + adaptive + passthrough:
     - Increment accumulator frame count
     - After RenderFrame: barrier → Accumulate → barrier → UpdateVariance
     - Every K frames: CheckConvergence
     - Display accumulated result via tonemap (bypass denoiser)
   - Reset triggers: camera move, resize, scene reload, denoiser mode switch, SPP change, bounce change, env parameter change
   - When not accumulating: normal single-frame path (RenderFrame → denoise → tonemap)

5. **Show convergence stats in top bar** — e.g., "Adaptive: 85% converged, 32 frames"

**Test: Viewer adaptive toggle**
- Verify checkbox is disabled when ML denoiser is active
- Verify checkbox is enabled when Passthrough denoiser is active
- Verify accumulation resets on camera move

**Files modified:** `app/view/Panels.h`, `app/view/Panels.cpp`, `app/view/main.cpp`

---

## Files Summary

### Modify (Sessions 1–3)

| File | Session | Changes |
|------|---------|---------|
| `capture/include/monti/capture/GpuAccumulator.h` | 1, 2 | Add images, atomic counter, new methods, expand `GpuAccumulatorDesc` |
| `capture/src/GpuAccumulator.cpp` | 1, 2 | Create/destroy images, compute pipelines, implement dispatch methods |
| `capture/shaders/accumulate.comp` | 1, 2 | Remove weight push constant, add sample_count binding (S1), add convergence mask binding (S2) |
| `renderer/src/vulkan/RaytracePipeline.h` | 2 | Add binding 17, expand `PushConstants` to 20 bytes |
| `renderer/src/vulkan/RaytracePipeline.cpp` | 2 | Add binding 17 to descriptor set layout and writes |
| `renderer/src/vulkan/shaders/raygen.rgen` | 2 | Add binding 17, early-out check |
| `renderer/include/monti/vulkan/Renderer.h` | 2 | Add `SetConvergenceMask()`, `SetAdaptiveSamplingEnabled()` |
| `renderer/src/vulkan/Renderer.cpp` | 2 | Implement new methods, dummy 1×1 mask image |
| `app/datagen/GenerationSession.h` | 3 | Add adaptive fields to `GenerationConfig` |
| `app/datagen/GenerationSession.cpp` | 1, 3 | Remove weight arg (S1), adaptive loop + reporting (S3) |
| `app/datagen/main.cpp` | 3 | Add adaptive CLI flags |
| `CMakeLists.txt` (root) | 1, 2 | Add `finalize.comp` (S1), `variance_update.comp`, `convergence_check.comp` (S2) to `CAPTURE_SHADER_SOURCES` |

### Create (Sessions 1–3)

| File | Session | Purpose |
|------|---------|---------|
| `capture/shaders/finalize.comp` | 1 | Per-pixel normalization (`accum / sample_count`) |
| `capture/shaders/variance_update.comp` | 2 | Welford's online variance update in log-luminance space |
| `capture/shaders/convergence_check.comp` | 2 | Convergence evaluation with 3×3 spatial max + atomic counter |
| `tests/adaptive_sampling_test.cpp` | 1, 2, 3 | Integration tests (progressive, each session adds tests) |
| `tests/welford_test.cpp` | 2 | CPU unit test for Welford algorithm |

### Deferred (Session 4 — F24-V)

| File | Changes |
|------|---------|
| `app/view/Panels.h` | Add `bool adaptive_sampling` to `PanelState` |
| `app/view/Panels.cpp` | Add checkbox with `BeginDisabled` when ML denoiser active |
| `app/view/main.cpp` | Add `--adaptive` CLI flag, progressive accumulation loop |

---

## Integration Tests Summary

Tests are defined inline within each session above. Cumulative test file: `tests/adaptive_sampling_test.cpp`.

| Test | Session | What It Verifies |
|------|---------|-----------------|
| Non-adaptive backward compat | 1 | Raw-sum + finalize = old weighted path |
| Raygen early-out smoke | 2 | Binding 17 + push constant + all-zero mask = no change |
| Variance pipeline smoke | 2 | Welford stats non-zero after 4 frames |
| Welford numerical stability (CPU) | 2 | Constant / linear / large-offset sequences |
| Flat color convergence | 3 | All pixels converge, output matches non-adaptive |
| Early termination | 3 | Loop breaks early, speedup > 2× |
| High-contrast scene | 3 | Log-luminance handles bright+dark, < 1% error |

---

## Decisions

| Decision | Rationale |
|----------|-----------|
| **Temporal variance (Welford's) + 3×3 spatial max** | Directly measures Monte Carlo noise without confusing signal edges with noise. Spatial variance alone would incorrectly flag edges as unconverged. |
| **Log-luminance domain** | Handles full HDR dynamic range without bright-pixel bias. |
| **Binary convergence mask (active/skip)** | Simpler than graduated SPP levels. Each pixel either traces M SPP or 0. Future extension could add 2-3 SPP tiers. |
| **Raw-sum accumulation + finalize normalization** | Cleanest semantics when pixels have different sample counts. Old weighted approach assumed uniform N for all pixels. |
| **Push constant flag for adaptive mask** | Zero cost when disabled — just a uint32 comparison in raygen. No specialization constants or shader variants needed. |
| **Dummy 1×1 all-zero image** for mask when disabled | Avoids null descriptor binding. Vulkan requires valid descriptors for all bindings at dispatch time. |
| **All variance computation on GPU** | Critical for performance. Per-pixel Welford updates, convergence checks, and normalization all run as compute shaders. CPU only reads back a single uint32 counter for progress reporting. |
| **Convergence mask is NOT used for the noisy frame** | Noisy frame (frame_index=0) always renders all pixels at `config_.spp`. Adaptive sampling only applies to reference accumulation frames (frame_index ≥ 1). |
| **accumulate.comp owns sample_count increment** | Single source of truth — variance_update reads the count but doesn't modify it. Avoids double-increment races. |

## Further Considerations

1. **Early termination:** When the atomic counter shows 100% convergence, the loop breaks immediately. This is the primary source of speedup for simple/well-lit scenes.

2. **Graduated SPP:** A future extension could define 2–3 SPP tiers (e.g., full/half/zero) instead of binary on/off. This would require a per-pixel SPP image (R8UI or R32UI) read by the raygen shader to set its inner loop count. The convergence check shader would reduce SPP in stages rather than immediately setting the mask to 1.

3. **Halton period interaction:** The Halton jitter sequence has a 16-frame period. Pixels that converge after e.g., 20 frames got 1 full period + 4 additional quasi-random samples. No special handling is needed — the samples remain well-distributed even with incomplete periods.

4. **Viewer progressive mode (F24-V):** Full progressive accumulation in `monti_view` is deferred to Session 4. Requires: (a) detecting camera stationarity, (b) creating a GpuAccumulator for the viewer, (c) resetting on camera movement / settings change, (d) displaying the accumulated result via tonemap (bypassing denoiser). Only active in Passthrough denoiser mode — checkbox greyed out when ML denoiser is active. The renderer API from Session 2 (`SetConvergenceMask`, `SetAdaptiveSamplingEnabled`) is shared with the viewer; only the ~150–200 LOC accumulation loop and UI are viewer-specific.

5. **Memory overhead:** 4 new images at 1920×1080:
   - `sample_count` (R32UI): 7.9 MB
   - `variance_mean` (R32F): 7.9 MB
   - `variance_m2` (R32F): 7.9 MB
   - `convergence_mask` (R8UI): 2.0 MB
   - Total: ~25.7 MB additional GPU memory. Negligible compared to the existing accumulator images (2× RGBA32F = 63.3 MB).
