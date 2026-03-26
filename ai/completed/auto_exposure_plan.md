# Auto-Exposure & Training Normalization Plan

> **Purpose:** Add optional auto-exposure to monti_view using GPU log-average luminance with temporal smoothing. For monti_datagen, compute log-average luminance on CPU, normalize training pairs to mid-gray (0.18), reject invalid viewpoints, and remove the `exposure` concept from the entire datagen/training pipeline. Add a symmetric exposure wedge in `generate_training_data.py` as the new data amplification strategy. Each phase is sized for a single Copilot Claude Opus 4.6 session and includes numerical validation tests.
>
> **Starting point:** monti_view has manual EV exposure control via `tonemap.comp` (ACES filmic). monti_datagen applies `2^exposure_ev100` multiplier to linear radiance. `generate_viewpoints.py` amplifies viewpoints across 5 EV levels via `_amplify_exposures()`. Training applies `ExposureJitter(±1 EV)` as batch-time augmentation.
>
> **End goal:**
> - monti_view: optional auto-exposure toggle with temporal smoothing, combinable with manual EV offset
> - monti_datagen: automatic mid-gray normalization, built-in validity checks (replacing `remove_invalid_viewpoints.py`), no exposure concept
> - Training pipeline: deterministic exposure wedge (default 5 steps) replaces both `_amplify_exposures()` and `ExposureJitter`
>
> **Session sizing:** Each phase is scoped to fit within a single Copilot Claude Opus 4.6 context session.

---

## Architecture Overview

```
Phase E1: GPU auto-exposure shaders + AutoExposure class (monti_view compute infrastructure)
Phase E2: Integrate auto-exposure into monti_view (tonemap, UI, main loop)
Phase E3: CPU luminance utility + datagen normalization + validation (monti_datagen)
Phase E4: Remove exposure from datagen/viewpoints pipeline
Phase E5: Exposure wedge in generate_training_data.py + remove ExposureJitter
```

### Algorithm: Log-Average Luminance

Both monti_view (GPU) and monti_datagen (CPU) use the same mathematical formula:

$$L_{\text{avg}} = \exp\!\left(\frac{1}{N}\sum_{i=1}^{N} \ln(L_i + \varepsilon)\right)$$

where $L_i = 0.2126 R_i + 0.7152 G_i + 0.0722 B_i$ (BT.709 luminance), $\varepsilon = 10^{-6}$.

The normalization multiplier maps $L_{\text{avg}}$ to mid-gray:

$$m = \frac{0.18}{L_{\text{avg}}}$$

### Why GPU for monti_view, CPU for monti_datagen

monti_view computes luminance every frame at 60fps on a full-resolution HDR image — GPU is essential. The shader reads a single combined RGBA16F image (denoiser output).

monti_datagen computes luminance once per viewpoint (seconds between invocations) on data that is already on CPU after `GpuAccumulator::Finalize()` reads back the reference frames. The data is two separate RGBA32F arrays (diffuse + specular). Using GPU would require: a different shader variant (two images instead of one), exposing `GpuAccumulator` internals, SSBO readback for the skip/no-skip decision, and a third compute shader to apply the normalization multiplier before readback. The CPU version is ~20 lines of identical math on data already in RAM.

Same algorithm, appropriate implementations. The CPU function is trivially unit-testable.

---

## Phase E1: GPU Auto-Exposure Shaders & AutoExposure Class ✅

**Goal:** Create the GPU compute infrastructure for log-average luminance with temporal smoothing. No integration with monti_view yet — just the shaders, the `AutoExposure` C++ class, and a CPU readback of the adapted luminance value.

### Tasks

#### 1. Create `luminance.comp` — log-average luminance accumulation

File: `app/shaders/luminance.comp`

```glsl
#version 460

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba16f) uniform readonly image2D input_image;

layout(set = 0, binding = 1) buffer LuminanceAccum {
    uint log_sum_fixed;   // Fixed-point sum of log(L + epsilon)
    uint pixel_count;     // Number of contributing pixels
};

// Encode float to a fixed-point uint for atomic addition.
// Range [-20, +20] mapped to [0, 2^24] — sufficient for log(luminance).
// 24-bit mantissa gives ~1e-7 precision per pixel, supporting up to ~16M
// pixels before overflow of uint32 (2^32 / 2^24 = 256 — but we use a wider
// window, so effective max is ~1B pixels at 1080p this is ~2M, no issue).
const float FIXED_SCALE = 65536.0;  // 2^16
const float FIXED_BIAS  = 20.0;     // shift log range to positive

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pixel.x >= size.x || pixel.y >= size.y) return;

    vec3 rgb = imageLoad(input_image, pixel).rgb;

    // BT.709 luminance
    float L = dot(rgb, vec3(0.2126, 0.7152, 0.0722));

    // Skip NaN/Inf pixels — they'd corrupt the sum
    if (isinf(L) || isnan(L)) return;

    float log_L = log(max(L, 1e-6));

    // Encode to fixed-point for atomicAdd
    uint fixed_val = uint((log_L + FIXED_BIAS) * FIXED_SCALE);

    atomicAdd(log_sum_fixed, fixed_val);
    atomicAdd(pixel_count, 1u);
}
```

#### 2. Create `luminance_resolve.comp` — finalization + temporal blend

File: `app/shaders/luminance_resolve.comp`

```glsl
#version 460

layout(local_size_x = 1) in;

layout(set = 0, binding = 0) buffer LuminanceAccum {
    uint log_sum_fixed;
    uint pixel_count;
};

layout(set = 0, binding = 1) buffer LuminanceResult {
    float adapted_luminance;
};

layout(push_constant) uniform PushConstants {
    float adaptation_speed;  // e.g. 5.0 — higher = faster convergence
    float delta_time;        // seconds since last frame
};

const float FIXED_SCALE = 65536.0;
const float FIXED_BIAS  = 20.0;

void main() {
    // Decode average log-luminance
    float count = float(pixel_count);
    float avg_log_L = (count > 0.0)
        ? (float(log_sum_fixed) / FIXED_SCALE - FIXED_BIAS * count) / count
        : log(0.18);  // fallback to mid-gray if no pixels

    float target_luminance = exp(avg_log_L);

    // Temporal smoothing: exponential blend toward target
    float blend = 1.0 - exp(-adaptation_speed * delta_time);
    float prev = adapted_luminance;

    // First frame: jump directly (prev is 0.0 from buffer init)
    if (prev <= 0.0)
        adapted_luminance = target_luminance;
    else
        adapted_luminance = prev + (target_luminance - prev) * blend;

    // Clear accumulators for next frame
    log_sum_fixed = 0u;
    pixel_count = 0u;
}
```

#### 3. Create `AutoExposure` class

File: `app/core/AutoExposure.h`

```cpp
class AutoExposure {
public:
    bool Create(VkDevice device, VmaAllocator allocator,
                std::string_view shader_dir,
                uint32_t width, uint32_t height,
                VkImageView hdr_input_view);
    void Destroy();
    bool Resize(uint32_t width, uint32_t height, VkImageView hdr_input_view);

    // Record luminance compute dispatches into cmd. Must be called before
    // ToneMapper::Apply() in the same command buffer.
    void Compute(VkCommandBuffer cmd, VkImage hdr_input, float delta_time);

    // CPU-side adapted luminance, updated after Compute() via host-visible buffer.
    // Returns 0.18 / adapted_luminance (the multiplier to apply to HDR).
    float ExposureMultiplier() const;

    // Read adapted luminance value directly (for UI display).
    float AdaptedLuminance() const;

    void SetAdaptationSpeed(float speed);
    float AdaptationSpeed() const;
};
```

GPU resources:
- **Accumulation SSBO** (8 bytes): `{uint log_sum_fixed, uint pixel_count}` — device-local
- **Result SSBO** (4 bytes): `{float adapted_luminance}` — host-visible (for CPU readback without staging transfer)
- **Two compute pipelines**: `luminance.comp` (16×16 workgroups) + `luminance_resolve.comp` (1×1×1)
- **One descriptor set layout** with 2 bindings (shared by both pipelines via compatible layout)

Command buffer flow:
1. Barrier: hdr_input (compute write → compute read)
2. Dispatch `luminance.comp`: `ceil(width/16) × ceil(height/16)` workgroups
3. Barrier: SSBO (compute write → compute read)
4. Dispatch `luminance_resolve.comp`: 1×1×1
5. (No barrier needed — result SSBO is host-visible, CPU reads previous frame's value)

The result SSBO uses `VMA_MEMORY_USAGE_AUTO` with `VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT` for persistent mapping. The CPU reads the adapted luminance from the mapped pointer, which is inherently one frame behind (N-1 latency). This is desirable — it avoids a GPU-CPU sync stall and is invisible due to temporal smoothing.

### Numerical Validation (E1)

1. **Fixed-point encoding round-trip test**: For a range of luminance values (0.001 to 1000), verify that `decode(encode(log(L)))` matches `log(L)` within ±0.001.

2. **Known-image luminance test**: Create a 4×4 test image where all pixels have RGB = (0.18, 0.18, 0.18). Expected log-average luminance = 0.18. Verify `ExposureMultiplier()` ≈ 1.0.

3. **Uniform bright image**: All pixels RGB = (1.0, 1.0, 1.0). Expected L_avg = 1.0, multiplier = 0.18.

4. **Mixed image**: Half pixels at (0.01, 0.01, 0.01), half at (1.0, 1.0, 1.0). Expected L_avg = exp(0.5 × log(0.01) + 0.5 × log(1.0)) ≈ 0.1. Verify multiplier ≈ 1.8.

5. **Temporal smoothing**: After convergence on a bright image, switch to a dark image. Verify adapted luminance moves toward the new value over several frames, not instantaneously. Verify speed parameter affects convergence rate.

6. **Build test**: Compile both shaders with `glslc --target-env=vulkan1.2`. Verify SPIR-V generation succeeds.

### Files Created/Modified

| File | Action |
|------|--------|
| `app/shaders/luminance.comp` | **Create** |
| `app/shaders/luminance_resolve.comp` | **Create** |
| `app/core/AutoExposure.h` | **Create** |
| `app/core/AutoExposure.cpp` | **Create** |
| `CMakeLists.txt` | Add `luminance.comp`, `luminance_resolve.comp` to `APP_SHADER_SOURCES`; add `AutoExposure.cpp` to `monti_view` sources |

---

## Phase E2: Integrate Auto-Exposure into monti_view ✅

**Goal:** Wire the `AutoExposure` class into the monti_view render loop, modify the tone mapper to accept the auto-exposure multiplier, and add a UI toggle.

### Tasks

#### 1. Modify `tonemap.comp` push constants

Current push constants:
```glsl
layout(push_constant) uniform PushConstants {
    float exposure;  // EV stops
} pc;
```

New push constants:
```glsl
layout(push_constant) uniform PushConstants {
    float exposure;                  // Manual EV stops (unchanged)
    float auto_exposure_multiplier;  // 1.0 when disabled, 0.18/L_avg when enabled
} pc;
```

Application: `vec3 exposed = hdr * pc.auto_exposure_multiplier * pow(2.0, pc.exposure);`

#### 2. Update `ToneMapper` C++ class

- Change push constant size from `sizeof(float)` to `2 * sizeof(float)`.
- Add `SetAutoExposureMultiplier(float)` method and `auto_exposure_multiplier_` field (default 1.0).
- Update `Apply()` to push both values.
- Update `CreatePipeline()` push constant range.

#### 3. Add `auto_exposure` to `PanelState`

In `Panels.h`:
```cpp
struct PanelState {
    // ... existing fields ...
    bool auto_exposure = false;
    float auto_exposure_luminance = 0.0f;  // Read-only display value
};
```

#### 4. Add UI toggle in `Panels.cpp`

In `DrawSettingsPanel()`, within the "Render" collapsing header, add before the Exposure EV slider:
```cpp
ImGui::Checkbox("Auto Exposure", &state.auto_exposure);
if (state.auto_exposure) {
    ImGui::Text("Avg Luminance: %.4f", state.auto_exposure_luminance);
}
```

The manual EV slider remains active — it acts as an offset from the auto-computed base.

#### 5. Wire in `main.cpp`

- Create `AutoExposure` alongside `ToneMapper` in initialization.
- In the render loop, after denoiser and before tone mapper:
  - If `panel_state.auto_exposure` is true:
    - Call `auto_exposure.Compute(cmd, hdr_input, delta_time)`
    - `tone_mapper.SetAutoExposureMultiplier(auto_exposure.ExposureMultiplier())`
    - `panel_state.auto_exposure_luminance = auto_exposure.AdaptedLuminance()`
  - Else:
    - `tone_mapper.SetAutoExposureMultiplier(1.0f)`
- Handle `Resize()` for `AutoExposure` alongside `ToneMapper`.

### Numerical Validation (E2)

1. **Toggle off**: Load a scene, verify rendering is bit-identical to before (multiplier = 1.0, no visual change). Compare screenshots.

2. **Toggle on — bright scene**: Load a bright outdoor scene (Sponza with sunlight environment). Toggle auto-exposure on. Verify the image converges to a reasonable mid-gray brightness within 1-2 seconds. The manual EV slider at 0 should show a well-exposed image.

3. **Toggle on — dark scene**: Load a dimly lit scene. Verify auto-exposure brightens it to similar apparent luminance as the bright scene.

4. **Manual EV offset**: With auto-exposure on, move the EV slider to +2. Verify the image brightens by ~4× from the auto base. Move to -2, verify it darkens.

5. **Temporal smoothing**: Rapidly pan the camera between a bright area and a dark corner. Verify smooth adaptation without flicker or popping.

6. **Debug modes**: Enable a debug visualization mode (normals, depth). Verify auto-exposure does not apply (debug bypass path unchanged).

### Files Created/Modified

| File | Action |
|------|--------|
| `app/shaders/tonemap.comp` | Modify push constants, add multiplier application |
| `app/core/ToneMapper.h` | Add `SetAutoExposureMultiplier()`, field |
| `app/core/ToneMapper.cpp` | Update push constant struct size, pass multiplier |
| `app/view/Panels.h` | Add `auto_exposure`, `auto_exposure_luminance` fields |
| `app/view/Panels.cpp` | Add checkbox + luminance display |
| `app/view/main.cpp` | Create `AutoExposure`, wire into render loop |

---

## Phase E3: CPU Luminance + Datagen Normalization + Validation ✅

**Goal:** Create a CPU-side log-average luminance utility, integrate it into `GenerationSession::WriteFrameFromJob()` to normalize training pairs to mid-gray, and reject invalid viewpoints (near-black, excessive NaN). Store the normalization multiplier in EXR metadata.

### Design

The CPU luminance function operates on the *reference* (high-SPP) data that is already on the CPU after `GpuAccumulator::Finalize()`. It produces:
- `log_average`: the geometric mean luminance of diffuse+specular combined
- `nan_count`: number of NaN/Inf pixels (for validation)
- `total_pixels`: for fraction computation

The normalization multiplier `0.18 / log_average` is applied to *all* radiance channels (noisy diffuse, noisy specular, reference diffuse, reference specular) — both input and target. Guide channels (normals, albedo, depth, motion vectors) are unchanged. This replaces the existing `2^exposure_ev100` multiplication in `WriteFrameFromJob()`.

Validation checks absorb `remove_invalid_viewpoints.py`:
- **Near-black**: `log_average < 0.001` → skip viewpoint (log warning)
- **Excessive NaN**: `nan_count / total_pixels > 0.001` → skip viewpoint (log warning)

### Tasks

#### 1. Create CPU luminance utility

Files: `capture/include/monti/capture/Luminance.h`, `capture/src/Luminance.cpp`

```cpp
namespace monti::capture {

struct LuminanceResult {
    float log_average;     // exp(mean(log(L + epsilon)))
    uint32_t nan_count;    // Number of NaN/Inf pixels
    uint32_t total_pixels; // Total pixel count
};

// Compute log-average luminance from separate diffuse and specular FP32 buffers.
// Each buffer has 4 floats per pixel (RGBA). Only RGB is used for luminance.
LuminanceResult ComputeLogAverageLuminance(
    const float* diffuse_f32,
    const float* specular_f32,
    uint32_t pixel_count);

}  // namespace monti::capture
```

Implementation (~20 lines):
```cpp
LuminanceResult ComputeLogAverageLuminance(
    const float* diffuse_f32, const float* specular_f32, uint32_t pixel_count) {
    constexpr float kEpsilon = 1e-6f;
    constexpr float kLumaR = 0.2126f;
    constexpr float kLumaG = 0.7152f;
    constexpr float kLumaB = 0.0722f;

    double log_sum = 0.0;
    uint32_t nan_count = 0;
    uint32_t valid_count = 0;

    for (uint32_t i = 0; i < pixel_count; ++i) {
        size_t base = static_cast<size_t>(i) * 4;
        float r = diffuse_f32[base + 0] + specular_f32[base + 0];
        float g = diffuse_f32[base + 1] + specular_f32[base + 1];
        float b = diffuse_f32[base + 2] + specular_f32[base + 2];
        float L = kLumaR * r + kLumaG * g + kLumaB * b;

        if (!std::isfinite(L)) { ++nan_count; continue; }

        log_sum += std::log(std::max(L, kEpsilon));
        ++valid_count;
    }

    float log_average = (valid_count > 0)
        ? static_cast<float>(std::exp(log_sum / valid_count))
        : 0.0f;

    return {log_average, nan_count, pixel_count};
}
```

#### 2. Add EXR metadata support to `Writer`

Modify `capture/src/Writer.cpp` to accept optional metadata and write it as custom EXR attributes via tinyexr's `EXRAttribute` API.

Add to `Writer::WriteFrameRaw()` (or the internal `WriteExr()` helper):
- Accept optional `std::span<std::pair<std::string, float>>` metadata parameter
- For each entry, create an `EXRAttribute` with type `"float"` and the given name/value
- Set `header.num_custom_attributes` and `header.custom_attributes` before `SaveEXRImageToFile()`

#### 3. Modify `WriteFrameFromJob()` — replace exposure with normalization

Replace the existing exposure multiplication block:
```cpp
// BEFORE:
float exposure_mul = std::pow(2.0f, job.exposure);
if (exposure_mul != 1.0f) { /* scale noisy + reference channels */ }

// AFTER:
auto lum_result = capture::ComputeLogAverageLuminance(
    job.ref_result.diffuse_f32.data(),
    job.ref_result.specular_f32.data(),
    job.width * job.height);

// Validation: reject near-black images
if (lum_result.log_average < 0.001f) {
    std::fprintf(stderr, "  [%s] SKIPPED: near-black (L_avg=%.6f)\n",
                 job.subdirectory.c_str(), lum_result.log_average);
    return WriteResult::kSkippedBlack;
}

// Validation: reject excessive NaN
float nan_fraction = static_cast<float>(lum_result.nan_count) / lum_result.total_pixels;
if (nan_fraction > 0.001f) {
    std::fprintf(stderr, "  [%s] SKIPPED: excessive NaN (%.2f%%)\n",
                 job.subdirectory.c_str(), nan_fraction * 100.0f);
    return WriteResult::kSkippedNaN;
}

// Normalize to mid-gray
float norm_mul = 0.18f / lum_result.log_average;
// Apply norm_mul to noisy diffuse/specular (FP16) and reference diffuse/specular (FP32)
// ... (same scaling loop as existing exposure code, using norm_mul instead of exposure_mul)
```

Pass `normalization_multiplier` and `log_average_luminance` as EXR metadata.

#### 4. Change `WriteFrameFromJob` return type

Replace `bool` with:
```cpp
enum class WriteResult { kSuccess, kSkippedBlack, kSkippedNaN, kError };
```

Update `Run()` to accumulate skip counts and include them in `viewpoint_timings_` JSON entries.

### Numerical Validation (E3)

1. **CPU luminance unit test — uniform mid-gray**: Create synthetic FP32 arrays where every pixel has R=G=B=0.18 for both diffuse and specular (combined luminance = 0.36). Verify `log_average ≈ 0.36` and `norm_mul ≈ 0.5`.

2. **CPU luminance unit test — uniform white**: All pixels (1.0, 1.0, 1.0) diffuse, (0,0,0) specular. Expected L_avg = 1.0, norm_mul = 0.18.

3. **CPU luminance unit test — mixed bright/dark**: Half pixels at 0.01, half at 1.0. Expected L_avg ≈ 0.1 (geometric mean). Verify norm_mul ≈ 1.8.

4. **CPU luminance unit test — NaN handling**: Insert NaN pixels. Verify they are excluded from the average and counted in `nan_count`. Verify 100% NaN returns `log_average = 0.0`.

5. **Near-black rejection**: Create synthetic all-black reference. Verify `WriteFrameFromJob()` returns `kSkippedBlack`.

6. **Normalization end-to-end**: Run `monti_datagen` on Sponza with a known viewpoint. Read the output EXR, compute luminance of diffuse+specular. Verify it is approximately 0.18. Verify `normalization_multiplier` is present in EXR metadata.

7. **EXR metadata round-trip**: Write an EXR with metadata, read it back with Python (`OpenEXR`), verify the custom attributes are present and correct.

### Files Created/Modified

| File | Action |
|------|--------|
| `capture/include/monti/capture/Luminance.h` | **Create** |
| `capture/src/Luminance.cpp` | **Create** |
| `capture/src/Writer.cpp` | Add metadata parameter to `WriteExr()` |
| `capture/include/monti/capture/Writer.h` | Add metadata to `WriteFrameRaw()` interface |
| `app/datagen/GenerationSession.h` | Add `WriteResult` enum, remove `WriteJob::exposure` |
| `app/datagen/GenerationSession.cpp` | Replace exposure with normalization + validation |
| `capture/CMakeLists.txt` (or top-level) | Add `Luminance.cpp` to capture library |
| `tests/` | Add unit test for `ComputeLogAverageLuminance()` |

---

## Phase E4: Remove Exposure from Datagen/Viewpoints Pipeline ✅

**Goal:** Remove the `exposure` concept entirely from monti_datagen, viewpoint JSON files, and `generate_viewpoints.py`. Exposure becomes a monti_view-only concept.

### Tasks

#### 1. Remove `--exposure` CLI from `app/datagen/main.cpp`

- Delete the `app.add_option("--exposure", ...)` line
- Remove the `float exposure = 0.0f;` variable
- Remove `config.exposure = exposure;` from config setup
- Remove exposure from the configuration print block
- When parsing viewpoint JSON entries, **ignore** the `"exposure"` field (remove the `if (entry.contains("exposure"))` block) — this provides backward compatibility with old viewpoint JSONs

#### 2. Remove `exposure` from `GenerationConfig` and `ViewpointEntry`

In `GenerationSession.h`:
- Remove `float exposure = 0.0f;` from `GenerationConfig`
- Remove `std::optional<float> exposure;` from `ViewpointEntry`
- Remove `float exposure;` from `WriteJob`

In `GenerationSession.cpp`:
- In `Run()`, change `camera.exposure_ev100 = vp.exposure.value_or(config_.exposure);` to `camera.exposure_ev100 = 0.0f;`
- In `Run()`, remove `.exposure = config_.exposure` from `WriteJob` initialization
- `WriteFrameFromJob()` no longer references `job.exposure` (already replaced by normalization in E3)

#### 3. Remove `_amplify_exposures()` from `generate_viewpoints.py`

- Delete the `_amplify_exposures()` function definition (lines 218-239)
- Delete the `_DEFAULT_EXPOSURES` constant (line 574)
- Remove the exposure amplification block in `generate_all_viewpoints()` (lines 793-796: the `if exposures:` block that strips and re-amplifies)
- Remove `exposures` parameter from `generate_all_viewpoints()` signature
- Remove `--exposures` and `--no-exposures` CLI arguments from `main()`
- Remove `exposures` from the `generate_all_viewpoints()` call in `main()`

#### 4. Remove exposure from seed viewpoint interpolation

In `generate_viewpoints.py`, search for `"exposure"` references in viewpoint interpolation/variation code and remove them. This includes the `_interpolate_seed_pair()` function where exposure is interpolated between seeds.

#### 5. Remove exposure from `generate_training_data.py`

Search for any `--exposure` forwarding to monti_datagen or exposure-related grouping. Remove if found.

### Numerical Validation (E4)

1. **monti_datagen runs without --exposure**: Invoke `monti_datagen scene.glb --output test/ --spp 1 --ref-frames 1`. Verify it runs successfully without the `--exposure` flag.

2. **Old viewpoint JSON compatibility**: Create a viewpoint JSON with `"exposure": 0.5` fields. Run monti_datagen with `--viewpoints`. Verify it ignores the exposure field without error.

3. **generate_viewpoints.py**: Run `python generate_viewpoints.py --scenes ../scenes/debug`. Verify output viewpoint JSON files contain no `"exposure"` fields. Verify `--exposures` flag is no longer accepted.

4. **Viewpoint count**: Before this change, 1 viewpoint × 5 exposures = 5 entries. After, 1 viewpoint = 1 entry. Verify output counts match expected (no amplification).

5. **generate_training_data.py**: Run on a small scene. Verify monti_datagen is invoked without `--exposure`.

### Files Created/Modified

| File | Action |
|------|--------|
| `app/datagen/main.cpp` | Remove `--exposure` CLI, ignore `exposure` in JSON |
| `app/datagen/GenerationSession.h` | Remove `exposure` from config, viewpoint, write job |
| `app/datagen/GenerationSession.cpp` | Set `exposure_ev100 = 0.0f`, remove `job.exposure` |
| `training/scripts/generate_viewpoints.py` | Remove `_amplify_exposures()`, `_DEFAULT_EXPOSURES`, CLI args, interpolation |
| `training/scripts/generate_training_data.py` | Remove any exposure forwarding |

---

## Phase E5: Exposure Wedge + Remove ExposureJitter ✅

**Goal:** Add a deterministic exposure wedge to `generate_training_data.py` as the new data amplification strategy. For each normalized EXR pair from monti_datagen, produce N copies at symmetric EV offsets (default 5 steps: -2, -1, 0, +1, +2). Remove `ExposureJitter` from training transforms. Deprecate `remove_invalid_viewpoints.py`.

### Tasks

#### 1. Add `--exposure-steps` CLI to `generate_training_data.py`

```python
parser.add_argument("--exposure-steps", type=int, default=5, choices=[3, 5, 7],
                    help="Number of exposure wedge steps (default: 5 → offsets -2..+2)")
```

Compute offsets: `offsets = list(range(-(steps // 2), steps // 2 + 1))`
- steps=3: [-1, 0, +1]
- steps=5: [-2, -1, 0, +1, +2]
- steps=7: [-3, -2, -1, 0, +1, +2, +3]

#### 2. Implement wedge generation function

```python
def apply_exposure_wedge(
    input_path: str, target_path: str,
    output_dir: str, base_name: str,
    offsets: list[int],
) -> list[str]:
    """Generate exposure-shifted copies of an input/target EXR pair.

    For each offset s in offsets, scales radiance channels by 2^s.
    The s=0 pair is copied without modification.

    Returns list of output file paths created.
    """
```

Key implementation details:
- Load all channels from input.exr and target.exr using OpenEXR/numpy
- For each offset `s`:
  - `scale = 2.0 ** s`
  - Scale radiance channels only: diffuse RGB, specular RGB (input); diffuse RGBA, specular RGBA (target)
  - For FP16 channels (input radiance): chromaticity-preserving clamp — if max(R,G,B) × scale > 65504, scale all by 65504/max
  - For FP32 channels (target): no clamping needed (float32 handles the range)
  - Guide channels (normals, albedo, depth, motion): unchanged
  - Write as `{base_name}_ev{s:+d}_input.exr` and `{base_name}_ev{s:+d}_target.exr`
  - Copy EXR metadata from source, add `"exposure_offset"` attribute
- s=0: just copy/rename (no scaling)

#### 3. Integrate wedge into the flatten/output step

Currently, `generate_training_data.py` flattens `vp_N/` subdirectories to `<scene>_<id>_{input,target}.exr`. Extend this to apply the wedge during flattening:

```python
for offset in offsets:
    # ... scale and write
    output_name = f"{scene}_{vp_id}_ev{offset:+d}"
    # write {output_name}_input.exr, {output_name}_target.exr
```

Total output pairs per viewpoint = `len(offsets)` (default 5).

#### 4. Remove `ExposureJitter` from training transforms

In `training/deni_train/data/transforms.py`:
- Remove the `ExposureJitter` class entirely
- Remove `_FP16_MAX` and `_INPUT_RADIANCE_SLICES`, `_TARGET_RADIANCE_SLICE` constants if only used by `ExposureJitter`

In `training/deni_train/train.py` (or wherever `Compose` is constructed):
- Remove `ExposureJitter(range=(-1.0, 1.0))` from the transform pipeline

#### 5. Deprecate `remove_invalid_viewpoints.py`

Add a comment at the top of the file:
```python
# DEPRECATED: Viewpoint validation is now integrated into monti_datagen (Phase E3).
# This script is no longer needed in the training pipeline. Retained for reference.
```

If `generate_training_data.py` invokes `remove_invalid_viewpoints.py`, remove that invocation.

#### 6. Update `ExrDataset` if file naming changes

The wedge adds `_ev+0`, `_ev+1`, etc. to filenames. Verify that `ExrDataset` discovers these files correctly. The current pattern likely matches `*_input.exr` / `*_target.exr` — the wedge naming `_ev+0_input.exr` should match. Verify and update the glob pattern if needed.

### Numerical Validation (E5)

1. **Wedge generation unit test**: Create a synthetic 4×4 EXR pair with known radiance values. Apply wedge with steps=5. Verify:
   - `_ev+0` pair has identical values to source
   - `_ev+1` pair has radiance values × 2
   - `_ev-2` pair has radiance values × 0.25
   - Guide channels unchanged in all copies

2. **FP16 overflow protection**: Create a synthetic input with a pixel at RGB = (60000, 60000, 60000) in FP16. Apply `_ev+2` (4× scale). Verify the pixel is clamped to ≤ 65504 with chromaticity preserved (all channels scaled equally).

3. **File naming**: Run `generate_training_data.py --exposure-steps 5` on a small scene. Verify output contains files named `{scene}_{id}_ev-2_input.exr`, `{scene}_{id}_ev-1_input.exr`, `{scene}_{id}_ev+0_input.exr`, etc.

4. **Pair count**: With 1 viewpoint and steps=5, verify 5 input/target pairs are produced. With steps=3, verify 3 pairs. With steps=7, verify 7 pairs.

5. **Training data loading**: Load the wedge-augmented dataset with `ExrDataset`. Verify all pairs load successfully, tensor shapes are correct, and no NaN/Inf in loaded data.

6. **Short training run**: Train for 100 steps on wedge-augmented data. Verify loss decreases and no NaN in gradients.

7. **Validate_dataset.py**: Run on wedge output. Verify HTML gallery shows the brightness progression across the 5 EV levels.

8. **Total disk space**: With steps=5, verify total dataset size is ~5× the single-pair size (minus compression gains on darker copies).

### Files Created/Modified

| File | Action |
|------|--------|
| `training/scripts/generate_training_data.py` | Add `--exposure-steps`, implement wedge generation |
| `training/deni_train/data/transforms.py` | Remove `ExposureJitter` class |
| `training/deni_train/train.py` | Remove `ExposureJitter` from `Compose` |
| `training/scripts/remove_invalid_viewpoints.py` | Add deprecation comment |
| `training/scripts/validate_dataset.py` | Verify compatibility with new naming |
| `training/deni_train/data/exr_dataset.py` | Update glob pattern if needed |

---

## Summary

### Phase Dependencies

```
E1 ──→ E2    (monti_view auto-exposure: shaders → integration)
E3 ──→ E4    (datagen: normalization → exposure removal)
E4 ──→ E5    (pipeline: removal → wedge replacement)
```

E1–E2 and E3–E4 can proceed in parallel (independent codepaths). E5 depends on E4.

### Total File Count

| Phase | New Files | Modified Files |
|-------|-----------|----------------|
| E1 | 4 (2 shaders, 2 C++) | 1 (CMakeLists.txt) |
| E2 | 0 | 6 (tonemap.comp, ToneMapper.h/cpp, Panels.h/cpp, main.cpp) |
| E3 | 2 (Luminance.h/cpp) | 5 (Writer.h/cpp, GenerationSession.h/cpp, CMake) + tests |
| E4 | 0 | 5 (main.cpp, GenSession.h/cpp, gen_viewpoints.py, gen_training_data.py) |
| E5 | 0 | 5-6 (gen_training_data.py, transforms.py, train.py, remove_invalid.py, dataset.py) |
| **Total** | **6 new** | **~20 modifications** |

### Risk Assessment

| Risk | Mitigation |
|------|------------|
| Fixed-point atomicAdd overflow at high resolution (8K+) | 16-bit mantissa × 33M pixels fits in uint32 with bias ≤ 20. For 8K (33.2M pixels), max log_sum = 33.2M × (20+20) × 65536 ≈ 8.7e13 — exceeds uint32. **If 4K+ support is needed, use uint64 atomicAdd (requires `GL_EXT_shader_atomic_int64`).** At 1080p (2M pixels) uint32 is safe. |
| Temporal smoothing lag on sudden scene changes | Adaptation speed default of 5.0 means ~90% convergence in ~0.5s. Acceptable for interactive use. User can increase speed via UI if needed. |
| Normalization creating very large multipliers on dim scenes | Near-black check (`L_avg < 0.001`) rejects pathological cases. For legitimate dim scenes, large multiplier amplifies noise — this is intentional training diversity. |
| Wedge FP16 overflow at +3 EV (8× multiplier) | Chromaticity-preserving clamp protects against overflow. At +3 bright pixels above 8191 would clip, but normalization to 0.18 mid-gray means most pixels are well below this. |
| Old viewpoint JSONs with `exposure` field | monti_datagen ignores unrecognized fields. generate_training_data.py passes through JSON unmodified. No breakage. |
| ExrDataset glob pattern mismatch with new naming | Validate in E5 and update pattern. New naming `_ev+0_input.exr` still ends with `_input.exr`. |
