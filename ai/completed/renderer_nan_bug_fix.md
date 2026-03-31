# Renderer NaN Bug Investigation & Fix

## Background

During training data generation, 531 viewpoints were rejected with reason
`excessive_nan` across many different scenes. This causes ~26% of expected
pairs to be lost (4062 expected → 2608 valid), costing roughly 1000–1500
additional training pairs.

The `excessive_nan` rejection fires in `training/scripts/generate_training_data.py`
when either the noisy or reference image contains more NaNs than a configured
threshold. The NaNs originate in the Vulkan path tracer.

Critically, the affected scenes are **not limited to glass/transmission scenes**:

| Scene               | Skipped viewpoints |
|---------------------|--------------------|
| Sponza              | 98                 |
| BistroInterior      | ~76                |
| ToyCar              | 34                 |
| BrutalistHall       | 32                 |
| DragonAttenuation   | 32                 |
| DamagedHelmet       | 28                 |
| AntiqueCamera       | 26                 |
| SheenChair          | 26                 |
| ClearCoatTest       | 25                 |
| FlightHelmet        | 23                 |
| GlassHurricane      | 21                 |
| ABeautifulGame      | 20                 |

The broad spread across PBR-metallic (DamagedHelmet, AntiqueCamera, FlightHelmet),
sheen (SheenChair), clearcoat (ClearCoatTest, ToyCar), and glass/volume
(DragonAttenuation, GlassHurricane) scenes points to an NaN source in a code
path shared by all of them: **GGX specular sampling**.

## Primary Suspect: `sampleGGX` in `sampling.glsl`

File: `renderer/src/vulkan/shaders/include/sampling.glsl`

```glsl
vec3 sampleGGX(vec2 xi, vec3 N, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    float phi = 2.0 * PI * xi.x;
    float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha2 - 1.0) * xi.y));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);   // ← SUSPECT
```

The formula for `cos_theta` is analytically bounded to `[0, 1]`, but due to
IEEE 754 float32 rounding — particularly at very low alpha2 values near
`kMinGGXAlpha2 = 4e-8` — `cos_theta` can be computed as a value infinitesimally
above `1.0`. The subsequent `sqrt(1.0 - cos_theta * cos_theta)` then receives
a tiny negative argument and returns `NaN`.

`NaN` in `sin_theta` → `NaN` in `H_tangent` → `sampleGGX` returns `NaN` vector.

In `raygen.rgen`, once `H` is NaN:
```glsl
vec3 H = sampleGGX(rands.xy, N, roughness);
L = reflect(-V, H);                         // L = NaN
float NdotL = dot(N, L);                    // NdotL = NaN
if (NdotL <= 0.0) break;                    // NaN <= 0.0 == false → doesn't break!
// continues with NaN L, NaN BRDF evaluation...
throughput *= (NaN BRDF / NaN PDF)          // throughput = NaN
path_radiance += throughput * ...           // path_radiance = NaN → written to G-buffer
```

The NaN comparison `NaN <= 0.0` evaluates to `false` in GLSL, so the loop
**does not break** and continues spreading NaN through `throughput` and
`path_radiance`.

### Fix

In `renderer/src/vulkan/shaders/include/sampling.glsl`, clamp the sqrt argument:

```glsl
float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
```

This is the standard defensive GLSL pattern. It costs one `max` and one float
comparison per GGX sample.

## Secondary Suspects to Audit

While fixing the primary suspect, audit these additional locations for similar
unguarded `sqrt` calls:

### 1. `cosineSampleHemisphere` — likely safe, check anyway

```glsl
float cos_theta = sqrt(1.0 - xi.y);   // xi.y ∈ [0,1], safe
float sin_theta = sqrt(xi.y);          // xi.y ∈ [0,1], safe
```

These use raw `xi.y` directly, so they are fine as long as the RNG returns
values in [0, 1] — which blue noise guarantees. Leave as-is but confirm.

### 2. `G_SmithG1GGX` in `brdf.glsl`

```glsl
float tan2 = (1.0 - cos2) / cos2;
return 2.0 / (1.0 + sqrt(1.0 + alpha2 * tan2));
```

`1.0 + alpha2 * tan2 >= 1.0` always, so the inner `sqrt` is safe.
The denominator `>= 2.0` so no division by zero. Confirm no NaN path.

### 3. Volume attenuation in `raygen.rgen`

```glsl
vec3 sigma = -log(max(atten_color, vec3(kMinCosTheta))) / atten_dist;
vec3 atten_factor = exp(-sigma * path_length);
```

Guarded by `if (atten_dist > 0.0 && ...)` and `max(atten_color, vec3(0.001))`.
Appears safe. If `atten_color` contains exactly `0.0` components in any scene,
verify the max clamp actually prevents `log(0)`.

### 4. Refraction path in `raygen.rgen`

```glsl
float eta = n1 / n2;
vec3 refracted = refract(-V, N, eta);
if (dot(refracted, refracted) < kTIRThreshold) {
    ray_dir = reflect(-V, N);
} else {
    ray_dir = normalize(refracted);
}
```

`refract()` returns `vec3(0)` on TIR, which is caught. However, if `n2 = 0.0`
(malformed IOR from a glTF file), `eta = inf`, `refract` may return NaN rather
than vec3(0). Check whether IOR is clamped at load time in `GpuScene.cpp` /
`GeometryManager.cpp`. If not, add a floor (e.g., `max(ior, 1.0)` or
`max(ior, 0.5)`) in the material buffer packing code.

### 5. `calculateAllPDFs` / `calculateMISWeight` in `mis.glsl`

Any place a PDF can be exactly zero that reaches a division. The guard
`if (chosen_pdf <= 0.0) break;` covers the outer loop, but the MIS weight
computation `calculateMISWeight` internally sums PDFs — make sure it cannot
produce `0/0`.

## Verification Plan

### Step 1: Apply the `sampleGGX` fix

Edit `renderer/src/vulkan/shaders/include/sampling.glsl`:
- Change `sqrt(1.0 - cos_theta * cos_theta)` to `sqrt(max(0.0, 1.0 - cos_theta * cos_theta))`

### Step 2: Rebuild shaders

```
cmake --build build --config Release --target monti_app
```

(Shaders are compiled at build time via glslc; no separate re-compile step.)

### Step 3: Run the existing NaN tests

```
.\build\Release\monti_tests.exe "[phase8h][phase8i][phase8c]" -r compact
```

These tests already check `nan_count == 0` on glass/transmission scenes and
should continue passing.

### Step 4: Targeted NaN sweep

Run a quick render of the known-bad scenes (start with DamagedHelmet and
AntiqueCamera since they are non-glass and should only be affected by the
GGX NaN path):

```
.\build\Release\monti_datagen.exe --scene scenes/khronos/DamagedHelmet --frames 1 --spp 4 ...
```

Inspect that the output EXR has zero NaN pixels.

### Step 5: Check IOR clamping

Search `GpuScene.cpp` and/or `GeometryManager.cpp` for where `ior` is packed
into the material buffer (slot `mat_base + 2`, component `.g`). Ensure:
```cpp
float ior = std::max(material.ior, 1.0f);  // IOR must be >= 1.0 for physical glass
```

### Step 6: Re-run full training data generation

Once the NaN fix is confirmed working, regenerate the training dataset:

```powershell
cd C:\Users\wex\src\WexWorks\monti\training
python scripts\generate_training_data.py `
    --monti-datagen ..\build\Release\monti_datagen.exe `
    --scenes ..\scenes\khronos ..\scenes\training ..\scenes\extended\Cauldron-Media `
    --viewpoints-dir viewpoints `
    --output training_data `
    --width 1920 --height 1080 `
    --spp 4 --ref-frames 256 --ref-spp 16 `
    --jobs 2 --exposure-steps 2
```

Expected improvement: recover ~500–1000 pairs from the previously-rejected
viewpoints, growing the dataset from ~2608 to ~3100–3600 valid pairs.

## Key Files

| File | Relevance |
|------|-----------|
| `renderer/src/vulkan/shaders/include/sampling.glsl` | Primary fix location |
| `renderer/src/vulkan/shaders/raygen.rgen` | NaN propagation path |
| `renderer/src/vulkan/shaders/include/brdf.glsl` | G_Smith / BRDF eval |
| `renderer/src/vulkan/shaders/include/mis.glsl` | PDF computation |
| `renderer/src/vulkan/GpuScene.cpp` | Material buffer packing, IOR |
| `renderer/src/vulkan/GeometryManager.cpp` | Geometry/material loading |
| `training/scripts/generate_training_data.py` | `excessive_nan` rejection logic |

## Expected Outcome

The single-line `max(0.0, ...)` fix in `sampleGGX` should eliminate NaN
production for all non-glass scenes that are failing. Glass/volume scenes
(DragonAttenuation, GlassHurricane) may also benefit if their NaN source is
the same GGX path; the volume/refraction secondary suspects should be audited
to handle any remaining rejections in those scenes.

---

## Session 2: Additional NaN Fixes (March 2026)

After running generate_training_data.py and convert_to_safetensors.py, a deeper
investigation of the remaining ~3,250 skipped viewpoints revealed four additional
NaN sources and a false-positive near-black issue.

### Data Accounting

| Category              | Viewpoints |
|-----------------------|------------|
| Converted to safetensors | 4,184   |
| Skipped: excessive_nan   | 2,877   |
| Skipped: near_black      | 375     |
| Lost to BistroInterior crash | 43  |
| Never generated (Brutalism) | ~590 |
| **Total planned**        | ~8,068  |

### Fix 1: Sheen NaN — `lambdaSheen()` in `sheen.glsl`

**File:** `renderer/src/vulkan/shaders/include/sheen.glsl`

**Root cause:** `lambdaSheen()` calls `pow(1.0 - cos_theta, exponent)` where
`cos_theta` can exceed 1.0 due to floating-point imprecision in TBN normal
reconstruction. `pow(negative, fractional)` returns NaN in GLSL.

**Affected scenes:** SheenChair, red velvet cloth in BistroInterior, any
material using sheen/fabric BRDF.

**Fix:** Clamp cos_theta input:
```glsl
cos_theta = clamp(cos_theta, kMinCosTheta, 1.0);
```

### Fix 2: Transmission Fresnel NaN in `raygen.rgen`

**File:** `renderer/src/vulkan/shaders/raygen.rgen`

**Root cause:** The transmission path computed Fresnel as
`pow(1.0 - max(dot(N, V), kMinCosTheta), 5.0)`. When `dot(N, V)` is negative
(back-facing geometry seen through transmission), `max` correctly clamps to
`kMinCosTheta` but the issue is that `dot(N, V)` can exceed 1.0 from FP
imprecision, making `1.0 - dot(N, V)` negative. The dedicated
`F_Schlick()` function already uses `clamp()` but this inline call did not.

**Fix:** Replace `max` with `clamp`:
```glsl
clamp(dot(N, V), kMinCosTheta, 1.0)
```

### Fix 3: FireflyClamp Inf → NaN in `sampling.glsl`

**File:** `renderer/src/vulkan/shaders/include/sampling.glsl`

**Root cause:** `FireflyClamp()` computes luminance from the input color. When
the color contains Inf components (from e.g. division by near-zero PDF),
`lum` is Inf. The scaling `color * (threshold / lum)` computes
`Inf * (threshold / Inf)` = `Inf * 0` = NaN.

**Fix:** Early return guard:
```glsl
if (isinf(lum) || isnan(lum)) return vec3(0.0);
```

### Fix 4: NaN-safe PDF check in `raygen.rgen`

**File:** `renderer/src/vulkan/shaders/raygen.rgen`

**Root cause:** The light sampling PDF check `if (ls.pdf <= 0.0) continue;`
does not catch NaN PDFs. IEEE 754 comparisons with NaN always return false,
so `NaN <= 0.0` is false, allowing NaN to propagate through the MIS weight.

**Fix:** Invert the comparison:
```glsl
if (!(ls.pdf > 0.0)) continue;
```

### Datagen CLI Improvements

Added three new command-line options to `monti_datagen`:

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--force-write` | bool | false | Write EXR even when skip checks fail (for debugging) |
| `--nan-threshold` | float | 0.001 | Max NaN pixel fraction before skip |
| `--black-threshold` | float | 0.00005 | Min log-average luminance before skip |

### Near-Black Threshold Tuning

The original hard-coded near-black threshold of 0.001 caused false positives on
BistroInterior — a dim interior scene lit primarily by emissive ceiling lights.
The raw HDR log-average luminance was ~0.00055, well below 0.001, but the images
looked correct after normalization (0.18 / L_avg).

The threshold was made configurable via `--black-threshold` and the default
lowered to 0.00005. At this threshold, the normalization multiplier would be
3600×, which is extreme enough that any image below it is likely a genuine
render failure.

### Validation

Per-scene NaN validation viewpoint files were created in
`training/training_data_debug/` (2 viewpoints each, extracted from the most
affected viewpoints per scene):

| File | Scene | Result |
|------|-------|--------|
| `toycar-nan-viewpoints.json` | ToyCar.glb | Pass |
| `abandoned-warehouse-nan-viewpoints.json` | AbandonedWarehouse.gltf | Pass |
| `waterbottle-nan-viewpoints.json` | WaterBottle.glb | Pass |
| `sheen-chair-nan-viewpoints.json` | SheenChair.glb | Pass |
| `sponza-nan-viewpoints.json` | Sponza.gltf | Pass |
| `bistro-interior-nan-viewpoints.json` | BistroInterior scene.gltf | Pass (after black threshold fix) |
| `flight-helmet-nan-viewpoints.json` | FlightHelmet.gltf | Pass |

### Files Changed

| File | Changes |
|------|---------|
| `renderer/src/vulkan/shaders/include/sheen.glsl` | Clamp cos_theta in lambdaSheen |
| `renderer/src/vulkan/shaders/raygen.rgen` | Clamp transmission Fresnel, NaN-safe PDF check |
| `renderer/src/vulkan/shaders/include/sampling.glsl` | isinf/isnan guard in FireflyClamp |
| `app/datagen/GenerationSession.h` | Added black_threshold, nan_threshold, force_write to GenerationConfig |
| `app/datagen/GenerationSession.cpp` | Use configurable thresholds, force-write bypass |
| `app/datagen/main.cpp` | Added --black-threshold, --nan-threshold, --force-write CLI args |
