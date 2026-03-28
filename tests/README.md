# Monti Test Suite

All tests use [Catch2](https://github.com/catchorg/Catch2) and are compiled into a single binary: `build/Release/monti_tests.exe`.

## Running Tests

Run all tests from the repo root:

```powershell
.\build\Release\monti_tests.exe
```

Expected output: **216 test cases, 0 failures, 0 skips.**

### Running a Subset by Tag

Catch2 supports tag filtering. Common tags:

| Tag | Description |
|-----|-------------|
| `[vulkan]` | Requires a Vulkan-capable GPU |
| `[integration]` | Full pipeline / end-to-end tests |
| `[unit]` | Pure CPU / logic tests |
| `[golden]` | FLIP comparison against stored reference PNGs |
| `[extended]` | Extended (Cauldron-Media) scenes |
| `[deni]` | ML denoiser tests |
| `[capture]` | Data capture / EXR writer tests |
| `[gltf]` | glTF loader tests |
| `[scene]` | Scene graph tests |
| `[auto_exposure]` | Auto-exposure compute shader tests |

```powershell
# Unit tests only (fast, no GPU required)
.\build\Release\monti_tests.exe "[unit]"

# All integration tests
.\build\Release\monti_tests.exe "[integration]"

# Only golden comparison tests
.\build\Release\monti_tests.exe "[golden]"

# Only ML denoiser tests
.\build\Release\monti_tests.exe "[deni]"
```

---

## Golden Reference Tests

Golden tests render a scene at **1024×1024**, **1024 SPP** (256 frames × 4 SPP/frame), apply ACES tone mapping, and compare the result against a stored reference PNG using [FLIP](https://github.com/NVlabs/flip). Tests fail if the mean FLIP error exceeds the threshold.

| Threshold constant | Value | Used for |
|--------------------|-------|----------|
| `kSimpleSceneFlipThreshold` | 0.05 | CornellBox (procedural) |
| `kComplexSceneFlipThreshold` | 0.08 | All glTF/extended scenes |

### Reference Images

Reference PNGs are stored in `tests/golden/`. Each corresponds to a test scene:

**Core / Khronos scenes** (committed to the repo):
- `CornellBox.png` — procedural Cornell box
- `DamagedHelmet.png`, `DragonAttenuation.png`, `ClearCoatTest.png`
- `ABeautifulGame.png`, `AntiqueCamera.png`, `BoomBox.png`, `FlightHelmet.png`
- `GlassHurricaneCandleHolder.png`, `Lantern.png`, `MaterialsVariantsShoe.png`
- `MosquitoInAmber.png`, `SheenChair.png`, `Sponza.png`, `ToyCar.png`, `WaterBottle.png`

**Extended / Cauldron-Media scenes** (gitignored — generated locally):
- `BistroInterior.png`, `AbandonedWarehouse.png`, `Brutalism.png`

Extended scene assets live under `scenes/extended/Cauldron-Media/`.

### Regenerating Reference Images

All golden generation tests are tagged `[golden_gen][.]`. The `[.]` tag means they are **hidden** — they do not run during a normal `monti_tests.exe` invocation. Run them explicitly to overwrite the stored PNGs:

```powershell
# Regenerate all reference images (core + extended)
.\build\Release\monti_tests.exe "[golden_gen]" --allow-running-no-tests

# Regenerate core khronos scenes only
.\build\Release\monti_tests.exe "[golden_gen]" ~"[extended]" --allow-running-no-tests

# Regenerate extended scenes only
.\build\Release\monti_tests.exe "[golden_gen][extended]" --allow-running-no-tests

# Regenerate a single scene
.\build\Release\monti_tests.exe "Generate golden: DamagedHelmet"
```

After regeneration, commit the updated PNGs in `tests/golden/` (core scenes only — extended PNGs are gitignored).

### Viewpoints

Each scene uses a fixed camera position copied from the first entry in the corresponding `training/viewpoints/manual/<SceneName>.json` file. The coordinates are hardcoded directly in `golden_test.cpp` so the test is not runtime-dependent on the JSON files.

To change a viewpoint: update the `camera.position` / `camera.target` values in both the `"Generate golden: ..."` and `"Golden test: ..."` test cases for that scene, then regenerate the reference PNG.

### Diagnostic Output

Every golden render (generation and comparison) writes a diagnostic PNG to `tests/output/golden_<name>.png` for visual inspection. This directory is gitignored.

---

## Test File Overview

| File | Area |
|------|------|
| `main_test.cpp` | Harness smoke tests — FLIP links, TypedId, Scene default-construct |
| `vulkan_context_test.cpp` | Headless Vulkan device creation and command submission |
| `scene_integration_test.cpp` | Scene graph: Cornell box structure, node/mesh/material accessors |
| `gltf_loader_test.cpp` | glTF/GLB loading: geometry, materials, samplers, tangent generation, normals |
| `geometry_manager_test.cpp` | BLAS/TLAS build, compaction, mesh removal, buffer address tables |
| `gpu_scene_test.cpp` | GPU-side mesh/material/texture upload and packed layout |
| `auto_exposure_test.cpp` | Auto-exposure compute shader: encoding, histogram, smoothing |
| `capture_writer_test.cpp` | EXR capture writer: validation, compression, subdirectory creation |
| `luminance_test.cpp` | Log-average luminance calculation (NaN/Inf handling, background masking) |
| `golden_test.cpp` | Golden reference generation and FLIP comparison for all 19 scenes |
| `extended_scene_test.cpp` | Extended scene load / 1 SPP / 64 SPP sanity checks |
| `ml_weight_loader_test.cpp` | `.denimodel` file parsing — valid round-trip and error cases |
| `ml_inference_test.cpp` | ML inference feature allocation, weight upload, resize, Denoiser::Create |
| `ml_inference_numerical_test.cpp` | GPU inference output matches PyTorch golden; determinism |
| `ml_denoiser_integration_test.cpp` | ML denoiser: non-zero output, mode switching, graceful fallback |
| `ml_e2e_quality_test.cpp` | End-to-end: denoised FLIP < noisy FLIP for 6 scenes |
| `deni_passthrough_test.cpp` | Passthrough denoiser: exact output match, resize, null-input rejection |
| `phase10a_test.cpp` | Tone mapping: LDR range, exposure response, ACES highlights, e2e golden |
| `phase11b_test.cpp` | Depth/RG16F extraction, FP16 EXR round-trip |
| `phase7a_test.cpp` | GBuffer images, environment map CDF/mipmaps, blue noise |
| `phase7b_test.cpp` | Raytrace pipeline: descriptor sets, SBT alignment, push constants, lights |
| `phase7c_test.cpp` | Barycentric raytrace render, environment map hits/misses, validation |
| `phase8b_test.cpp` | Multi-bounce path tracing: NaN/Inf, validation |
| `phase8c_test.cpp` | Transparency, motion vectors, sub-pixel jitter |
| `phase8d_test.cpp` | PBR materials: packed layout, emissive, normal maps, metallic-roughness |
| `phase8e_test.cpp` | Firefly clamping: hue preservation, passthrough, NaN/Inf edge cases |
| `phase8f_test.cpp` | Texture LOD: checkerboard, mip bands |
| `phase8g_test.cpp` | Area lights: sphere, triangle, quad, WRS, degenerate rejection |
| `phase8h_test.cpp` | Diffuse transmission: backlighting, color tinting, convergence, NaN |
| `phase8i_test.cpp` | Nested dielectrics: glass-in-glass, false intersection, stack overflow |
| `phase8j_test.cpp` | Emissive mesh NEE: noise reduction, FLIP improvement, extraction threshold |
| `phase8k_test.cpp` | WRS light selection: single/few lights unchanged, many-lights convergence |
| `phase8l_test.cpp` | Texture transforms: tiling, identity, rotation, NaN, high SPP |
| `phase8m_test.cpp` | Sheen BRDF: visibility, energy conservation, color tinting, NaN |
| `phase8n_test.cpp` | DDS textures: BC7/BC1/BC5, mip chains, non-DDS bypass, glTF integration |
| `phase9b_test.cpp` | Denoiser full-pipeline passthrough match, validation clean |

---

## Build Notes

The test binary picks up compiled SPIR-V from:
- `build/shaders/` — renderer shaders (`MONTI_SHADER_SPV_DIR`)
- `build/deni_shaders/` — denoiser shaders (`DENI_SHADER_SPV_DIR`)
- `build/app_shaders/` — post-processing shaders (`APP_SHADER_SPV_DIR`)

After editing any shader, rebuild the corresponding shader target before running tests:

```powershell
cmake --build build --config Release --target monti_vulkan_shaders
cmake --build build --config Release --target deni_vulkan_shaders
cmake --build build --config Release --target app_shaders
```

Or rebuild everything:

```powershell
cmake --build build --config Release
```

The ML denoiser tests require the trained model file. Its location is resolved at build time via `DENI_MODEL_DIR`. If the model is absent, `Denoiser::Create()` auto-discovers `deni_v1.denimodel` in that directory; affected tests fall back to passthrough mode and will still pass.
