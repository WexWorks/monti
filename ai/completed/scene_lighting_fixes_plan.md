# Scene Lighting Fixes Plan

## Overview

Three Cauldron-Media extended scenes are dark or incorrectly lit in Monti's path tracer.
Verified scene contents (confirmed by direct glTF inspection, March 2026):

| Scene | File | KHR_lights_punctual | Emissive materials |
|---|---|---|---|
| BistroInterior | `scene.gltf` | none | 4 lamp fixtures: `emissiveFactor=[1,1,1]` + texture, no `KHR_materials_emissive_strength` |
| AbandonedWarehouse | `AbandonedWarehouse.gltf` | none | none |
| BrutalistHall | `BrutalistHall.gltf` | 1 directional "Sun" (intensity=25) | none |

Two independent issues were found.

---

## Root Cause Summary

### Issue 1 — `ExtractEmissiveLights` never called in the app (Critical)
`vulkan::ExtractEmissiveLights` extracts emissive mesh triangles into the scene's triangle
light buffer for NEE (next-event estimation). It is implemented and unit-tested in
`phase8j_test.cpp`, but is **never called from `app/view/main.cpp`** or any app-level
code path. The renderer's `light_count` includes `TriangleLights().size()`, so if no
triangles are extracted, NEE never fires for emissive geometry. Emissive surfaces still
contribute via direct path hits, but without NEE the variance is high and scenes appear dark
at normal SPP settings.

Note: `app/datagen/main.cpp` already calls `ExtractEmissiveLights` (added separately).
Only `app/view/main.cpp` is missing the call.

**Fix location**: `app/view/main.cpp`, after `UploadAndRegisterMeshes` / `SubmitAndWait`.

### Issue 2 — BistroInterior emissive lamps have no strength multiplier (Moderate)
BistroInterior's 4 lamp materials (`MASTER_Interior_01_Paris_Lantern`, `Paris_Ceiling_Lamp`,
`Paris_CeilingFan`, `Paris_Wall_Light_Interior`) have `emissiveFactor=[1,1,1]` and an
emissive texture, but no `KHR_materials_emissive_strength` extension. Since `emissiveFactor`
is clamped to [0,1] per the glTF spec, the raw texture values pass through at face value —
which is very dim for interior lamps. A physically plausible interior lamp (500–1000 lm
over a small shade area) needs `emissive_strength` in the range of ~500–2000.

The original magnitude was never stored in the glTF; it would have been driven by a
rasterizer light rig that is not present in this export. The value must be estimated or
tuned artistically.

**Fix options**:
- One-time patch to `scene.gltf`: add `KHR_materials_emissive_strength` to those 4 materials.
- Post-load CLI override: `--emissive-strength-override "Paris_Ceiling_Lamp=1000"`.

### Issue 3 — BrutalistHall directional sun light silently dropped (Low / future work)
`GltfLoader.cpp` never reads `node->light`. BrutalistHall's single directional "Sun" light
(intensity=25) is silently ignored. A directional light cannot be converted to finite emissive
geometry — it requires a proper directional/sun light primitive in the path tracer, or an
equivalent environment HDR with a matching sun disk.

This is out of scope for the current plan. The scene can be lit with a suitable HDR environment
map passed via `--env` in the meantime.

### Issue 4 — Spec-gloss glass opacity bug (Moderate, general)
In `renderer/scene/src/gltf/GltfLoader.cpp`, `desc.opacity` is only written inside the
`if (gmat.has_pbr_metallic_roughness)` block. Spec-gloss materials
(`KHR_materials_pbrSpecularGlossiness`) skip that block, so their `desc.opacity` defaults
to `1.0` even when `alphaMode=BLEND` and `diffuse_factor[3] < 1.0`. Any scene using
spec-gloss with alpha-blend will render glass as opaque.

**Fix location**: `GltfLoader.cpp`, add opacity read in the `has_pbr_specular_glossiness` block.

---

## Phase 1 — Wire ExtractEmissiveLights into the App ✅ COMPLETE

**Goal**: Emissive geometry NEE works in the interactive viewer and datagen pipeline.

**Estimated session time**: ~30 minutes

### Changes

**`app/view/main.cpp`** — after `UploadAndRegisterMeshes` / `SubmitAndWait`, before the render loop:

```cpp
#include "../../renderer/src/vulkan/EmissiveLightExtractor.h"  // add to includes

// After mesh upload:
uint32_t emissive_count = vulkan::ExtractEmissiveLights(scene, load_result.mesh_data);
if (emissive_count > 0)
    std::printf("ExtractEmissiveLights: %u triangle lights extracted\n", emissive_count);
```

**`app/datagen/` pipeline** — same pattern wherever `LoadGltf` is called for offline rendering.

### Tests

**Existing test to run first** (already validates the extraction itself):
- `[phase8j]` — `Phase 8J: EmissiveMeshNEEReducesNoise`
  - Checks: `extracted == 2` triangles for the 0.3×0.3 m panel
  - Checks: variance with NEE < variance without NEE × 0.7
  - Checks: mean luminance with extraction > 0 (scene is lit)

**New regression test** — `tests/phase8j_app_integration_test.cpp` (or extend phase8j):

```
Test: BistroInteriorExtractsEmissiveLamps
- LoadGltf("scenes/extended/Cauldron-Media/BistroInterior/scene.gltf")
- Call ExtractEmissiveLights(scene, mesh_data)
- Assert: TriangleLights().size() >= 2  (at least some lamp geometry extracted)
- Assert: all extracted triangle lights have max(radiance) >= 0.01  (non-black)
```

Numerical assertion:
```
CHECK(scene.TriangleLights().size() >= 2);
for (const auto& tl : scene.TriangleLights())
    CHECK(std::max({tl.radiance.r, tl.radiance.g, tl.radiance.b}) >= 0.01f);
```

---

## Phase 2 — Fix Spec-Gloss Glass Opacity ✅ COMPLETE

**Goal**: Alpha-blended KHR_materials_pbrSpecularGlossiness materials use the correct opacity
value, allowing skylight and reflections through BistroInterior windows.

**Estimated session time**: ~45 minutes

### Changes

**`renderer/scene/src/gltf/GltfLoader.cpp`** inside the `has_pbr_specular_glossiness` branch,
after `desc.base_color` is set from `diffuse_factor`:

```cpp
// Spec-gloss: opacity comes from diffuse_factor[3], parallel to metallic-roughness.
if (gmat.has_pbr_specular_glossiness)
    desc.opacity = gmat.pbr_specular_glossiness.diffuse_factor[3];
```

Note: this mirrors the existing `desc.opacity = pbr.base_color_factor[3]` line in the
metallic-roughness block. No other changes needed — `alphaMode` is already parsed and
`desc.opacity < 1` already triggers the stochastic pass-through path in `raygen.rgen`.

### Tests

**New test** — extend `tests/` with a material loading unit test:

```
Test: SpecGlossMaterialOpacityIsRead
- Build a minimal in-memory cgltf document with:
    - one material: has_pbr_specular_glossiness=true, diffuse_factor=[1,1,1,0.3],
      alpha_mode=CGLTF_ALPHA_MODE_BLEND
- Call ExtractMaterials (or LoadGltf with inline gltf string)
- Assert: loaded_mat.opacity == Approx(0.3f).epsilon(1e-5f)
- Assert: loaded_mat.alpha_mode == AlphaMode::kBlend
```

Numerical assertion:
```
CHECK(desc.opacity == Approx(0.3f).epsilon(1e-5f));
```

If an inline gltf loader path is not available in tests, write a small JSON fixture
`tests/fixtures/spec_gloss_glass.gltf` and call `LoadGltf`.

---

## Phase 3 — BistroInterior Emissive Lamp Strength (Optional Polish) ⏸ DEFERRED

**Goal**: BistroInterior's 4 lamp fixtures glow at physically plausible brightness.
Currently they have `emissiveFactor=[1,1,1]` with an emissive texture but no
`KHR_materials_emissive_strength`, so texture values pass through at face value (dim).

**Estimated session time**: ~1 hour

### Approach

The original magnitude is not stored anywhere in the glTF. Estimate from first principles:
a typical interior pendant lamp emits ~800 lm over a small shade (area ≈ 0.05–0.1 m²).
Radiance ≈ 800 / (π × 0.07) ≈ 3600 nit → `emissive_strength ≈ 3600`.

Add a `LoadOptions` struct to `GltfLoader.h`:
```cpp
struct LoadOptions {
    // Map from material name substring to emissive_strength override applied post-load.
    std::unordered_map<std::string, float> emissive_strength_override;
};
```

Apply after material extraction in `LoadGltf`:
```cpp
for (auto& [name, strength] : opts.emissive_strength_override) {
    for (auto& mat : scene.Materials()) {
        if (mat.name.find(name) != std::string::npos)
            mat.emissive_strength = strength;
    }
}
```

CLI invocation example:
```
monti.exe scene.gltf --emissive-strength-override "Paris_Ceiling_Lamp=3600"
```

Alternatively, patch `scene.gltf` directly to add `KHR_materials_emissive_strength` on
the 4 lamp materials — a one-time change that removes the need for any app-level override.

### Tests
```
Test: EmissiveStrengthOverrideApplied
- Build a scene with a material named "lamp" and emissive_strength=1.0
- LoadGltf with options.emissive_strength_override = {{"lamp", 3600.0f}}
- Assert: loaded material emissive_strength == Approx(3600.0f)
```

---

## Execution Order

| Phase | File(s)                      | Test Tags                          | Risk   |
|-------|------------------------------|------------------------------------|--------|
| 1     | `app/view/main.cpp`          | `[phase8j]`, new smoke test        | Low    |
| 2     | `GltfLoader.cpp`             | new `[gltf_loader][spec_gloss]`    | Low    |
| 3     | `GltfLoader.h/cpp` or gltf patch | new unit test                  | Low    |

Phases 1 and 2 are fully independent. Phase 3 is optional polish; Phase 1 must be done
first to get NEE benefit from the lamp geometry.

---

## Verification

After Phase 1, render BistroInterior without an env map to confirm lamp NEE is working:
```
monti.exe scenes/extended/Cauldron-Media/BistroInterior/scene.gltf --spp 128
```
Expected: interior is lit by the 4 lamp fixtures. Mean luminance should be > 0 without env.

BrutalistHall requires a matching environment HDR (sun disk) for correct lighting until a
full directional light path-tracer primitive is implemented.
