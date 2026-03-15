# Monti & Deni — Master Implementation Plan

> **Purpose:** Incremental build plan for the Monti path tracer and Deni denoiser libraries. Each phase produces a verifiable deliverable. Phases are sequential; later phases build on earlier ones. The plan references the architecture in [monti_design_spec.md](monti_design_spec.md) and adapts code from the [rtx-chessboard](../../../rtx-chessboard/) Vulkan path tracer. The application executables (`monti_view`, `monti_datagen`) are specified in [app_specification.md](app_specification.md) and will be implemented after the core libraries are functional.
>
> **Session sizing:** Each phase (or sub-phase) is scoped to fit within a single Copilot Claude Opus 4.6 context session — roughly 2–3 new/modified source files referencing 3–5 existing files, producing one verifiable deliverable.
>
> **Platform:** Initial implementation targets MSVC on Windows. GCC/Clang cross-platform support will be added when needed.

---

## Testing Philosophy

- **GPU-side integration tests are the default.** Every rendering feature is tested by actually rendering images on the GPU — load a scene, configure the feature under test, render frames, read back pixels, and verify measurable properties. There is no CPU-side reimplementation of shader logic for testing purposes. If a feature runs on the GPU, it is tested on the GPU.
- **Each test targets a specific feature and detects regressions.** Tests are designed so that the tested feature produces a measurable, automated signal — not just "the image looks reasonable." The scene, materials, and camera are chosen to isolate the feature, and the pass/fail criterion is chosen so that disabling or breaking the feature causes the test to fail. Convergence tests (low vs high SPP FLIP comparison) are one tool, but not the only one — pixel property checks (e.g., channel ordering, value ranges, NaN/Inf absence), variance comparisons, and two-scene FLIP comparisons (different configurations that exercise the feature to different degrees) are equally valid.
- **Unit tests only for complex isolated CPU logic.** Reserve unit tests for non-trivial algorithms that can be tested independently on the CPU and where integration tests would be slow or unreliable (e.g., CDF computation, material packing math, EXR channel layout). Never reimplement GPU shader functions (BRDF evaluation, firefly clamping, tone mapping) on the CPU for testing — test them through rendered output instead.
- **Vulkan validation layers are always on** in debug builds. Zero validation errors is a pass/fail gate for every GPU phase.

### Automated Render Validation

**Tool:** [NVIDIA FLIP](https://github.com/NVlabs/flip) (BSD-3 license) — a perceptual image comparison metric designed specifically for rendered images. It models human contrast sensitivity and produces a per-pixel error map with a single mean error score. Fetched via `FetchContent` in CMake; the C++ library has no heavy dependencies.

**Three-tier validation strategy:**

1. **Feature-specific regression tests** — The primary automated gate. Each rendering feature has a dedicated test that renders a scene designed to exercise that feature, then checks a measurable property of the output that would change if the feature regressed. Examples: firefly clamping tested by comparing FLIP scores with extreme vs moderate emission (extreme emission produces visibly worse convergence if clamping breaks); hue preservation tested by checking that bright pixel channel ordering matches the input emission color ratios; texture LOD tested by comparing near-region vs far-region variance on a textured ground plane (ray cones should produce lower variance at distance). These tests fail when the feature breaks, not when unrelated rendering changes occur.

2. **Self-consistency (convergence) tests** — Render the same scene at low SPP (e.g., 4) and high SPP (e.g., 64+). Compute FLIP between the two. The score must be below a threshold (proves the renderer converges correctly without requiring stored reference images). These tests are resilient to intentional rendering changes.

3. **Golden reference regression tests** — A small curated set of high-SPP reference images stored in the repo. Compare each test render against its reference using FLIP. Threshold: mean FLIP < 0.05 (tuned during Phase 8A). When rendering changes are intentional, update the reference images.

Feature-specific regression tests and self-consistency tests are the primary automated gates. Golden reference tests catch regressions but require manual update when the renderer changes intentionally. All test types produce FLIP error maps and/or diagnostic PNGs as artifacts for debugging failures.

### Regression Testing via Scene Design (No Feature Toggles)

The renderer is an uber-shader — all features are always enabled in production code. We do **not** add compile-time or runtime feature toggles to the renderer for testing purposes. Feature toggles would add complexity, diverge the tested code path from production, and create a combinatorial explosion of configurations.

Instead, each test constructs a **scene that makes the feature's effect measurable**. The test designs a scenario where the feature produces a distinctive, quantifiable signal in the rendered output, and sets a pass/fail threshold on that signal. If the feature regresses, the signal changes and the test fails — without ever disabling the feature.

**Scene design strategies for isolating features:**

| Strategy | How it works | Example |
|---|---|---|
| **Stress input** | Push a parameter to extremes where the feature's effect dominates | Firefly clamp: emission=10000 produces extreme outliers that only converge well if clamped |
| **Channel ordering** | Use asymmetric inputs so the output preserves a known ordering | Hue preservation: colored emission R:G:B=10000:5000:1000 — if clamping is proportional, bright pixels maintain R>G>B |
| **Two-scene comparison** | Render two scenes that differ only in the feature's input, FLIP between them | Sphere light size: radius=0.3 vs radius=0.02 at same position — FLIP > threshold proves radius affects shadows |
| **Spatial property** | Measure a property that varies spatially in a predictable way | Ray cone LOD: near texels have higher variance than far texels (coarser mips smooth the far region) |
| **Threshold test** | Feature guarantees a bound; verify the bound holds | Firefly clamp: no output pixel exceeds the clamp threshold luminance |
| **Absence test** | Verify a pathological outcome does NOT occur | NaN/Inf checks: extreme inputs (zero emission, huge emission) produce no NaN/Inf |
| **Convergence improvement** | Feature should improve convergence; verify FLIP(low, high SPP) improves | NEE with emissive extraction: variance at 4 spp is lower than without extraction (scene with emissive geometry) |

When an A/B comparison is needed but the feature cannot be "turned off," the test instead constructs **two scenes where the feature has different impact** — e.g., a scene where the feature is irrelevant (dim emission, no glass, single light) vs a scene where it matters (extreme emission, nested glass, many lights). The delta between these scenes is the feature's signal.

### Real GPU Testing (No Mocking)

Integration tests use the **real platform GPU API** — no mocking, no software abstraction layer.

- **Vulkan tests** run on any machine with a Vulkan driver. For CI without a discrete GPU, [SwiftShader](https://github.com/aspect-build/aspect-docs) or [lavapipe](https://docs.mesa3d.org/drivers/lavapipe.html) (Mesa's software Vulkan ICD) provide `VK_PHYSICAL_DEVICE_TYPE_CPU` implementations. SwiftShader has partial ray tracing support; full ray tracing tests require a GPU runner.
- **Metal tests** (future) run on macOS with Metal support. No software fallback needed — all Apple Silicon and recent Intel Macs support Metal.
- **WebGPU tests** (future) use [Dawn](https://dawn.googlesource.com/dawn/) headless. Dawn's Null backend provides a fast path for pipeline plumbing tests; the Vulkan/Metal backends provide real GPU execution.
- **Reference images are per-platform.** Floating-point differences across GPU vendors and APIs make exact pixel match impossible. Each platform maintains its own golden references with FLIP comparison.

### Test Scenes

Automated render tests need deterministic scenes with known expected appearance. Scenes are organized by purpose:

**1. Cornell Box (programmatic — no external dependency)**
Built in code via the `monti::Scene` API: 5 quads (white floor/ceiling/back, red left, green right) + 2 boxes (short/tall) + area light on ceiling. Known analytical solution for global illumination validation. The programmatic scene also exercises the scene layer without requiring the glTF loader.

**2. Khronos glTF Sample Assets (downloaded at build time)**
The [glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets) repo provides canonical PBR test models. Fetched via `FetchContent` or a CMake download step (only the specific `.glb` files needed, not the full repo). Key models:

| Model | Tests | Notes |
|---|---|---|
| `Box.glb` | Basic geometry, single material | Simplest possible integration test |
| `DamagedHelmet.glb` | PBR textures (base color, normal, metallic-roughness, emissive, occlusion) | Standard PBR validation |
| `DragonAttenuation.glb` | Transmission, volume attenuation, IOR | Transparency + refraction |
| `MosquitoInAmber.glb` | Transmission + embedded geometry | Transparency with interior objects |
| `ClearCoatTest.glb` | Clear coat parameter sweep | Clear coat validation |
| `MaterialsVariantsShoe.glb` | Multiple material variants | Material system stress test |
| `MorphPrimitivesTest.glb` | Multi-primitive mesh (2 primitives, different materials) | Validates one-primitive-per-SceneNode extraction (morph targets silently ignored) |

**3. Heavy geometry (downloaded separately, not in default test suite)**
For performance and memory profiling, not part of the pass/fail test gate:

| Scene | Source | Triangles | Notes |
|---|---|---|---|
| Amazon Lumberyard Bistro | [NVIDIA ORCA](https://developer.nvidia.com/orca/amazon-lumberyard-bistro) | ~2.8M | Standard real-time RT benchmark |
| Intel Sponza | [Intel Graphics Research](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html) | ~262K | Classic radiosity test scene |
| San Miguel | [Morgan McGuire's Archive](https://casual-effects.com/data/) | ~7.8M | Stress test for BVH and memory |

Heavy scenes are downloaded by an opt-in CMake option (`MONTI_DOWNLOAD_BENCHMARK_SCENES=ON`) and excluded from CI.

### Feature Effect Reference

Each rendering feature has a specific visual effect. Understanding that effect is essential for designing a test scene whose pass/fail criterion breaks when the feature regresses. This table documents the visual signature of each feature introduced in Phases 8E–8K and the scene design that exposes it.

| Phase | Feature | Visual effect when working | What breaks visually if it regresses | Test scene & criterion |
|---|---|---|---|---|
| 8E | Firefly clamping | Extreme outlier fireflies suppressed; image converges faster at low SPP with bright emitters | Bright pixel speckle; slow convergence; single-pixel outliers dominate variance | Cornell box with emission=10000: FLIP(4spp, 64spp) stays below threshold. Colored emission (R>G>B): bright pixels preserve channel ordering |
| 8E | Hit distance output | `linear_depth.g` contains positive ray hit distances for denoiser temporal reprojection | Denoiser receives incorrect distances; temporal stability degrades | Cornell box 1 spp: hit pixels have `.g` in (0, scene_diagonal); miss pixels have `.g` = kSentinelDepth |
| 8F | Ray cone texture LOD | Distant textured surfaces appear smoother (higher mip); near surfaces sharp; reduced moiré | Distant surfaces show aliasing/shimmer; full-res texels fetched everywhere (cache thrashing) | Checkerboard ground plane: variance in far region < variance in near region × 0.8 |
| 8G | Sphere lights | Soft circular shadows; penumbra width scales with sphere radius | Hard point-light shadows regardless of radius; incorrect falloff | Two renders: large radius vs small radius — FLIP > threshold proves radius affects shadow softness |
| 8G | Triangle lights | Arbitrary emissive triangles illuminate nearby surfaces via NEE | Triangle emitters only contribute via random path hits; noisier indirect-only illumination | Triangle light on ceiling: floor has non-zero illumination; two-sided flag produces back-face emission |
| 8H | Diffuse transmission | Thin translucent surfaces (leaves, paper) scatter light through to the far side | Backlit surfaces are dark; no forward scattering through thin geometry | Backlit green quad with `diffuse_transmission_factor=0.8`: front face has green illumination. Same quad with factor=0.0 is dark — FLIP between the two > threshold |
| 8H | Thin-surface mode | No IOR refraction bending for thin surfaces; straight-through transmission | Background geometry appears distorted through thin panels (incorrect refraction applied) | Glass panel with `thin_surface=true` vs `false`: FLIP > threshold proves the flag changes refraction behavior |
| 8I | Nested dielectric priority | Correct IOR at boundaries between overlapping volumes (liquid in glass) | Inner sphere refracts as if surrounded by vacuum (IOR 1.0) instead of outer medium (IOR 1.33) | Sphere (IOR 1.5, priority 1) inside sphere (IOR 1.33, priority 2): FLIP vs inner sphere alone > threshold |
| 8J | Emissive mesh extraction | Emissive surfaces illuminate nearby objects via NEE shadow rays, not just random hits | Emissive objects are self-luminous but barely light their surroundings at low SPP (high variance) | Emissive object in dark room: pixel variance with extraction < variance without × 0.7 |
| 8K | WRS light selection | O(1) per-hit-point light sampling regardless of light count; correct at convergence | O(N) cost returns (frame time scales linearly with light count); or biased selection | 200-light scene: frame time ratio (200-light / 10-light) < 1.5; convergence FLIP at 64 spp below threshold |
| 10A | ACES tone mapping | HDR highlights compressed (not clipped); sRGB output in [0.0, 1.0] RGBA16F | Hard clipping at white; visible banding; incorrect gamma | Render with extreme emission: no large regions at 1.0/1.0/1.0; mean luminance < 0.78 |

### Test Infrastructure Consolidation

Test utility functions currently duplicated across test files should be consolidated into shared headers to reduce maintenance burden as new test phases are added. This is an ongoing refactoring task — move helpers to shared files when touching the relevant test code, not as a separate bulk refactoring pass.

**Shared test context** — `tests/test_context.h`: The `TestContext` struct (Vulkan instance + device + queue initialization for headless testing) is duplicated identically across 10 test files. Extract into a shared header.

**Shared render helpers** — `tests/render_test_helpers.h`: Functions used by 2+ test files for GPU rendering and readback:
- `ReadbackImage()` — copy GPU image to CPU staging buffer (4 files)
- `AnalyzeRGBA16F()` — NaN/Inf/nonzero/variation statistics (4 files)
- `TonemappedRGB()` — Reinhard tone-map for FLIP input (2 files)
- `ComputeMeanFlip()` — FLIP comparison wrapper (2 files)
- `WriteCombinedPNG()` — diffuse+specular combined diagnostic output (3 files)

**Shared scene builders** — `tests/test_scenes.h`: Programmatic scene construction helpers:
- `MakeQuad()`, `AddQuadToScene()` — parametric quad geometry (2 files)
- `MakeEnvMap()` — solid-color environment map (2 files)
- `MakeCheckerboard256()`, `MakeGroundPlane()` — textured geometry for LOD tests (1 file, will be needed by future phases)
- `RegionColorVariance()` — spatial variance measurement (1 file, will be needed by future A/B tests)

---

## Overview

| Phase | Deliverable | Verifiable Outcome |
|---|---|---|
| 1 ✅ | Project skeleton + build system | `cmake --build` succeeds, empty libraries link |
| 2 ✅ | Scene layer (`monti_scene`) | Integration test: build Cornell box, verify data round-trip + MeshData |
| 3 ✅ | glTF loader | Integration test: load glTF, verify mesh/material/texture counts + MeshData |
| 4 ✅ | Vulkan context + app scaffolding | `monti_view`: window opens, swapchain presents a cleared color. Headless context test passes. |
| 5 ✅ | GPU scene (`monti::vulkan::GpuScene`) | Integration test: register mesh buffers → verify bindings; pack materials → verify buffer |
| 6 ✅ | Acceleration structures (`GeometryManager`) | BLAS + TLAS built, compacted, device addresses valid |
| 7A ✅ | G-buffer images + environment map + blue noise | Environment map loaded, CDF buffers valid, G-buffer images allocated |
| 7B ✅ | Descriptor sets + pipeline + SBT | Ray tracing pipeline created, SBT populated, no validation errors |
| 7C ✅ | Raygen + miss + closesthit stub | Window shows environment map; glTF silhouettes visible (normals as color) |
| 8A ✅ | GLSL shader library + single-bounce PBR | Textured PBR scene renders with correct single-bounce shading |
| 8B ✅ | Multi-bounce MIS + clear coat | Multi-bounce reflections, MIS convergence, clear coat visible |
| 8C ✅ | Transparency + transmission + G-buffer aux + jitter | Fresnel refraction, volume attenuation, correct motion vectors, complete G-buffer |
| 8D ✅ | PBR texture sampling + normal mapping + emissive + MIS fix | Normal maps, metallic-roughness maps, emissive direct, named constants |
| 8E ✅ | Firefly filter + hit distance output | Luminance-based firefly clamping, RG16F linear depth + hit distance, `phase8e_test.cpp` passes |
| 8F ✅ | Ray cone texture LOD | Automatic mip selection via ray cone tracking, reduced texture aliasing |
| 8G | Spherical area lights + triangle light primitives | Sphere/triangle light types, unified PackedLight buffer |
| 8H | Diffuse transmission + thin-surface mode | Diffuse transmission BSDF lobe, thin-surface flag, 5-way MIS |
| 8I | Nested dielectric priority | IOR priority stack for overlapping transmissive volumes |
| 8J | Emissive mesh light extraction | Auto-extract emissive triangles for NEE, compute shader |
| 8K | Weighted reservoir sampling for NEE | O(1) WRS light selection replaces O(N) per-light loop |
| 9A ✅ | Standalone denoiser library (`deni_vulkan`) | Standalone unit test: diffuse + specular summed, output matches input sum |
| 9B ✅ | Denoiser integration test | Denoiser wired into render loop, end-to-end passthrough verified |
| 9C ✅ | Loader-agnostic Vulkan dispatch (`deni_vulkan`) | `deni_vulkan` compiles and links without volk; all Vulkan functions resolved via `get_device_proc_addr` |
| 9D ✅ | Loader-agnostic Vulkan dispatch (`monti_vulkan`) + app updates | `monti_vulkan` compiles and links without volk; apps pass `vkGetDeviceProcAddr` to both libraries |
| 10A ✅ | Tone map + present (end-to-end pipeline) | `monti_view`: complete render loop — trace → denoise → tonemap → present |
| 10A-2 | Extended scenes + golden test expansion | CMake scene download, golden reference tests for core + extended scenes |
| 10B ✅ | Interactive camera + ImGui overlay | `monti_view`: WASD/mouse fly+orbit camera, settings panel, debug G-buffer viz |
| 11A ✅ | Capture writer (`monti_capture`) | CPU-side EXR writer: write known data at two resolutions, reload and verify channels |
| 11B ✅ | GPU readback + headless datagen | `monti_datagen`: headless render at input resolution → GPU readback → high-SPP reference at target resolution → dual-file EXR output |

---

> **Completed phases 1-8F, 9A-9D, 10A, 10B, 11A-11B:** Detailed tasks and verification for all completed phases have been archived to [monti_implementation_plan_completed.md](monti_implementation_plan_completed.md).

---

## Phase 8G: Spherical Area Lights + Triangle Light Primitives

**Goal:** Extend the light system with two new light types: spherical area lights (analytic spheres with uniform emission) and triangle light primitives (for future emissive mesh decomposition). This expands light type coverage without requiring ReSTIR.

**Source:** RTXPT `PolymorphicLight.hlsli` (sphere lights, triangle lights), "Real-Time Polygonal-Light Shading with Linearly Transformed Cosines" (Heitz et al.)

### Design Decisions

- **`SphereLight` as a new scene-layer type.** A sphere light is defined by center (vec3), radius (float), and radiance (vec3). Sampling is straightforward: pick a visible point on the sphere via solid-angle uniform sampling, compute the PDF, trace a shadow ray. For small spheres viewed from a distance, the solid angle approaches a point light; the area integral ensures correct behavior at all distances.
- **`TriangleLight` as a new scene-layer type.** A triangle light is defined by three vertices (v0, v1, v2) and radiance (vec3, front-face emission). This is the fundamental primitive for decomposing emissive meshes into light sources in future phases. Sampling uses uniform random barycentric coordinates.
- **Unified `PackedLight` buffer replaces `PackedAreaLight`.** All light types are packed into a single storage buffer using a type discriminator. Each packed light is 64 bytes (4 × vec4): the `.w` of the first vec4 encodes the light type as a float-encoded uint (0 = quad, 1 = sphere, 2 = triangle). Shader branching is minimal (one switch per light per shadow ray). The existing `PackedAreaLight` struct and `area_light_count` push constant are replaced by `PackedLight` and `light_count`.
- **Quad `AreaLight` retained alongside `TriangleLight`.** The quad is not deprecated — it has a simpler uniform sampling formula (no barycentric coordinates), a direct solid-angle PDF, and is the natural primitive for rectangular emitters (ceiling panels, windows, screens). Two triangles could represent a quad, but the dedicated quad path is more efficient and ergonomic for the common case. Host applications continue to use `AddAreaLight()` for rectangular emitters and `AddTriangleLight()` for arbitrary emissive geometry.
- **`LightSample` struct and solid-angle PDF convention.** All `sample*Light()` functions return a `LightSample` containing the sampled position, geometric normal at the light surface, radiance, and a solid-angle PDF (units: sr⁻¹). The solid-angle PDF incorporates the `dist² / (area × cos_light)` conversion so that callers can use it directly without further geometric correction. The struct definition in `lights.glsl`:
  ```glsl
  struct LightSample {
      vec3  position;  // sampled point on light surface
      vec3  normal;    // outward geometric normal at sampled point
      vec3  radiance;  // emitted radiance
      float pdf;       // solid-angle PDF (sr⁻¹)
  };
  ```
- **GpuScene method renames.** The `PackedAreaLight` → `PackedLight` migration also renames the `GpuScene` accessors for consistency: `UpdateAreaLights()` → `UpdateLights()`, `AreaLightBuffer()` → `LightBuffer()`, `AreaLightBufferSize()` → `LightBufferSize()`. All call-sites in `Renderer` are updated to match.
- **Degenerate light validation.** `Scene::AddSphereLight()` validates `radius > 0` and `Scene::AddTriangleLight()` validates that the triangle has non-zero area (`length(cross(v1 - v0, v2 - v0)) > 0`). Degenerate lights are rejected with a logged warning rather than silently added.
- **No weighted selection yet.** NEE still iterates all lights per bounce (O(N) per hit). Weighted reservoir sampling (WRS) for O(1) selection is deferred to Phase 8J.

### Tasks

1. Add `SphereLight` and `TriangleLight` to `scene/include/monti/scene/Light.h`:
   ```cpp
   struct SphereLight {
       glm::vec3 center   = {0, 0, 0};
       float     radius   = 0.5f;
       glm::vec3 radiance = {1, 1, 1};
   };

   struct TriangleLight {
       glm::vec3 v0       = {0, 0, 0};
       glm::vec3 v1       = {1, 0, 0};
       glm::vec3 v2       = {0, 1, 0};
       glm::vec3 radiance = {1, 1, 1};
       bool      two_sided = false;
   };
   ```

2. Add `AddSphereLight()`, `AddTriangleLight()` and accessors to `Scene`:
   ```cpp
   void AddSphereLight(const SphereLight& light);
   void AddTriangleLight(const TriangleLight& light);
   const std::vector<SphereLight>& SphereLights() const;
   const std::vector<TriangleLight>& TriangleLights() const;
   ```
   `AddSphereLight()` validates `radius > 0`; `AddTriangleLight()` validates non-zero triangle area. Both log a warning and discard the light if validation fails.

3. Replace `PackedAreaLight` with `PackedLight` in `GpuScene`:
   ```cpp
   // renderer/src/vulkan/GpuScene.h
   enum class LightType : uint32_t { kQuad = 0, kSphere = 1, kTriangle = 2 };

   struct alignas(16) PackedLight {
       glm::vec4 data0;  // Quad: corner.xyz, type
                          // Sphere: center.xyz, type
                          // Triangle: v0.xyz, type
       glm::vec4 data1;  // Quad: edge_a.xyz, two_sided
                          // Sphere: radius, 0, 0, 0
                          // Triangle: v1.xyz, two_sided
       glm::vec4 data2;  // Quad: edge_b.xyz, 0
                          // Sphere: 0, 0, 0, 0
                          // Triangle: v2.xyz, 0
       glm::vec4 data3;  // All: radiance.xyz, 0
   };
   static_assert(sizeof(PackedLight) == 64);
   ```

4. Rename `GpuScene` methods: `UpdateAreaLights()` → `UpdateLights()`, `AreaLightBuffer()` → `LightBuffer()`, `AreaLightBufferSize()` → `LightBufferSize()`. Rename the internal buffer member `area_light_buffer_` → `light_buffer_`.

5. Update `GpuScene::UpdateLights()` — pack all three light types into `PackedLight[]`:
   - Iterate `AreaLights()` → pack as kQuad
   - Iterate `SphereLights()` → pack as kSphere
   - Iterate `TriangleLights()` → pack as kTriangle
   - Upload to storage buffer (binding 11)

6. Rename push constant `area_light_count` → `light_count` (total count across all types).

7. Update descriptor set writes in `Renderer`: replace `gpu_scene_.AreaLightBuffer()` / `gpu_scene_.AreaLightBufferSize()` references with `gpu_scene_.LightBuffer()` / `gpu_scene_.LightBufferSize()` when writing the binding 11 descriptor for the light storage buffer.

8. Create `shaders/include/lights.glsl` — define `LightSample` struct and light sampling functions. All sampling functions return a solid-angle PDF (sr⁻¹) that incorporates the `dist² / (area × cos_light)` conversion, so callers use the PDF directly without geometric correction:
   ```glsl
   struct LightSample {
       vec3  position;  // sampled point on light surface
       vec3  normal;    // outward geometric normal at sampled point
       vec3  radiance;  // emitted radiance
       float pdf;       // solid-angle PDF (sr⁻¹)
   };
   ```
   - `sampleQuadLight(PackedLight light, vec2 xi, vec3 shading_pos)` → `LightSample` — rewritten from scratch for the new `PackedLight` layout (not ported from the old inline code)
   - `sampleSphereLight(PackedLight light, vec2 xi, vec3 shading_pos)` → `LightSample` — visible-hemisphere solid-angle sampling
   - `sampleTriangleLight(PackedLight light, vec2 xi, vec3 shading_pos)` → `LightSample` — uniform barycentric sampling, solid-angle PDF
   - `sampleLight(PackedLight light, vec2 xi, vec3 shading_pos)` → dispatches by type via `uint(light.data0.w)` switch

9. Update `raygen.rgen` — remove the existing inline quad-sampling code entirely and replace the per-area-light loop with a per-light loop that calls `sampleLight()` from `lights.glsl`. The loop body simplifies to: sample light → trace shadow ray → accumulate `throughput × radiance × brdf × NdotL / pdf`.

### CornellBox Test Scene Changes

Remove the default area light from `BuildCornellBox()` so that it returns a scene with geometry and materials only (no lights). Update the `BuildCornellBox()` header comment to reflect this. Update existing tests that rely on the default light:
- `phase8b_test.cpp` — add the canonical ceiling area light explicitly via `scene.AddAreaLight(...)` before rendering.
- `phase8c_test.cpp` — same: add the ceiling area light explicitly.
- `scene_integration_test.cpp` — the test at line 155 that asserts `lights.size() == 1` must be updated to assert `lights.size() == 0` on a fresh `BuildCornellBox()`, then add a light and assert `lights.size() == 1`.
- `phase8e_test.cpp`, `gpu_scene_test.cpp` — no changes needed (these already add their own lights or don't depend on light presence).

### Verification

`tests/phase8g_test.cpp` — GPU integration tests that verify new light types through rendered output. All tests construct lights explicitly (no default light in `BuildCornellBox()`).

1. **`SphereLightIllumination`** (GPU integration) — Place a sphere light (radius=0.1, radiance=50) above the Cornell box floor via `scene.AddSphereLight(...)`. Render at 64 spp. Verify the floor directly beneath the light has higher mean luminance than a control render with no lights. This test regresses if sphere light sampling is broken.

2. **`SphereLightSoftShadow`** (GPU integration) — Place a large sphere light (radius=0.3) and a small sphere light (radius=0.02) at the same position via `scene.AddSphereLight(...)`. Render both at 64 spp. Compute FLIP between the two. Verify FLIP > 0.02 (the larger light produces visibly softer shadows). This test regresses if sphere radius is ignored in sampling.

3. **`TriangleLightIllumination`** (GPU integration) — Place a single triangle light on the Cornell box ceiling via `scene.AddTriangleLight(...)`. Render at 64 spp. Verify the floor has non-zero illumination and no NaN/Inf. Verify a render with `two_sided = true` illuminates both sides of a thin surface.

4. **`QuadLightBackwardCompatibility`** (GPU integration) — Add the canonical ceiling area light explicitly via `scene.AddAreaLight(...)`. Existing Cornell box area light tests pass unchanged after the `PackedAreaLight` → `PackedLight` migration. FLIP between pre- and post-migration renders at 64 spp < 0.01.

5. **`MixedLightConvergence`** (GPU integration, convergence) — Cornell box with one quad light + one sphere light + one triangle light (all added explicitly). FLIP(4spp, 64spp) below convergence threshold.

6. **`DegenerateLightRejection`** (unit test) — Verify `AddSphereLight()` rejects radius ≤ 0 and `AddTriangleLight()` rejects zero-area triangles. Confirm `SphereLights()` / `TriangleLights()` vectors remain empty after rejected adds.

- No NaN/Inf; no Vulkan validation errors.

---

## Phase 8H: Diffuse Transmission + Thin-Surface Mode

**Goal:** Add diffuse transmission (light passing through a thin surface with scattering, as in leaves and paper) and a thin-surface material flag that enables single-slab approximations without enter/exit tracking. These extend the BSDF with new lobes for translucent materials.

**Source:** RTXPT `ApplyDeltaLobes()` (diffuse transmission), glTF `KHR_materials_diffuse_transmission` extension, RTXPT nested dielectric priority model

### Design Decisions

- **Diffuse transmission as a new BSDF lobe.** When `diffuse_transmission_factor > 0`, a fraction of the incident light is transmitted diffusely through the surface (cosine-weighted hemisphere on the opposite side). This models thin translucent materials like leaves, paper, and fabric where light scatters forward through the medium. The factor controls the diffuse/transmission split: `(1 - diffuse_transmission_factor)` goes to regular diffuse reflection, `diffuse_transmission_factor` goes to diffuse transmission. The `diffuse_transmission_color` tints the transmitted light — it defaults to white `{1,1,1}` and the resulting transmitted BRDF uses `base_color * diffuse_transmission_color` as the albedo.

- **`thin_surface` material flag.** When true, the material uses single-slab approximations for all transmission effects: no IOR refraction (transmitted rays pass straight through without bending), thin-slab Beer-Lambert attenuation (existing Phase 8C behavior), and the surface is treated as infinitely thin (no enter/exit state tracking). When false, the existing Phase 8C Fresnel refraction + IOR behavior applies. Most real-world thin translucent materials (leaves, curtains, lamp shades) should set `thin_surface = true`.

- **Energy-correct delta/smooth split for specular transmission.** The current specular transmission code path handles ALL rays as delta events (specular reflect or refract) and always `continue`s, preventing smooth lobes (including diffuse transmission) from ever firing. Phase 8H restructures this into an energy-correct three-way split following RTXPT's `ApplyDeltaLobes()` pattern:
  1. **Specular reflection:** probability = `Fresnel`. Delta reflect, `continue`.
  2. **Specular transmission:** probability = `(1 - Fresnel) * transmission_factor`. Delta transmit (refract for thick; straight-through for thin), `continue`.
  3. **Smooth lobes (fall through to MIS):** probability = `(1 - Fresnel) * (1 - transmission_factor)`. The remaining energy enters the diffuse substrate and is handled by MIS — diffuse reflection, diffuse transmission, specular GGX, clearcoat, and environment strategies all participate.

  For materials with `transmission_factor = 0` the delta block is skipped entirely (existing behavior). For fully specular materials (`transmission_factor = 1.0`, `metallic = 0`), all non-Fresnel energy goes through specular transmission and the smooth lobes receive no energy — this is correct for glass. The throughput is not scaled by the choice probability because each branch's sampling probability equals its energy fraction (unbiased single-sample estimator).

  When `thin_surface = true`, specular transmission uses the incident ray direction unchanged (no `refract()` bend). Thin-slab attenuation (Beer-Lambert) still applies if `attenuation_distance > 0` and `thickness_factor > 0`.

- **PackedMaterial extended to 8 vec4 (128 bytes).** A new 8th vec4 stores diffuse transmission data with half-float packing for the color:
  - `.r` = `diffuse_transmission_factor`
  - `.g` = `thin_surface` (0.0/1.0)
  - `.b` = `packHalf2x16(diffuse_transmission_color.rg)` — two half-floats encoding the red and green color channels
  - `.a` = `packHalf2x16(vec2(diffuse_transmission_color.b, 0.0))` — blue channel in upper half, lower half reserved for Phase 8I `nested_priority`

  C++ packing uses `glm::packHalf2x16()` → `std::bit_cast<float>()`. GLSL unpacking uses `unpackHalf2x16(floatBitsToUint(val))`.

- **MIS update: 5-way strategy with energy-conserving diffuse split.** The diffuse transmission lobe adds a fifth sampling strategy. `SamplingProbabilities` gains a `diffuse_transmission` field. The existing diffuse probability is split by `diffuse_transmission_factor` to maintain energy conservation:
  - `prob_diffuse_reflect = base_diffuse * (1 - diffuse_transmission_factor)` — standard Lambertian reflection
  - `prob_diffuse_transmit = base_diffuse * diffuse_transmission_factor` — Lambertian transmission into back hemisphere

  Where `base_diffuse = (1 - metallic) * (1 - Fresnel)` as before. The two diffuse probabilities sum to the original `base_diffuse`, preserving total energy. Metals cannot transmit diffusely (`metallic → 1` drives `base_diffuse → 0`). The minimum strategy probability floor (`kMinStrategyProb`) applies to the diffuse transmission strategy only when `diffuse_transmission_factor > 0`.

- **Diffuse transmission PDF is zero on the front hemisphere (and vice versa).** `evaluateDiffuseTransmission()` uses `max(-NdotL, 0.0)` — only nonzero for back-hemisphere directions. The standard diffuse PDF `max(NdotL, 0.0)` is only nonzero for front-hemisphere directions. This means there is zero cross-contribution between the two diffuse strategies for any given sample direction, which is correct: a direction in the front hemisphere has zero diffuse-transmission PDF, and a direction in the back hemisphere has zero diffuse-reflection PDF.

- **`double_sided` interaction with diffuse transmission.** When the camera hits the back face of a double-sided material, the shading normal `N` is flipped to face the camera. Diffuse transmission then transmits into `-N` (away from the camera), which is the geometric "front" of the surface. This is physically correct for a thin material: regardless of which face you view, light transmits through to the opposite side.

- **cgltf v1.14 does not natively support `KHR_materials_diffuse_transmission`.** The extension must be parsed via cgltf's generic `extensions[]` array on each material. Each unrecognized extension is stored as `{name, data}` where `data` is a raw JSON string. A minimal JSON parser extracts `diffuseTransmissionFactor` (float), `diffuseTransmissionColorFactor` (float[3]), and optional `diffuseTransmissionTexture` (deferred — texture not parsed in v1). If the extension is present, `thin_surface` is set to `true` (glTF diffuse transmission implies thin-surface geometry).

### Tasks

1. Add material fields to `MaterialDesc`:
   ```cpp
   float     diffuse_transmission_factor = 0.0f;
   glm::vec3 diffuse_transmission_color  = {1, 1, 1};
   bool      thin_surface                = false;
   ```

2. Extend `PackedMaterial` to 8 vec4 (128 bytes):
   ```cpp
   glm::vec4 transmission_ext;  // .r = diffuse_transmission_factor,
                                 // .g = thin_surface (0.0/1.0),
                                 // .b = packHalf2x16(dt_color.rg) as float,
                                 // .a = packHalf2x16(vec2(dt_color.b, 0.0)) as float
   ```
   Update `static_assert(sizeof(PackedMaterial) == 128)`.

3. Update `GpuScene::UpdateMaterials()` to pack new fields:
   - Pack `diffuse_transmission_factor` into `.r`
   - Pack `thin_surface` as `0.0f`/`1.0f` into `.g`
   - Pack `diffuse_transmission_color.rg` via `glm::packHalf2x16()` → `std::bit_cast<float>()` into `.b`
   - Pack `diffuse_transmission_color.b` and `0.0f` (reserved for Phase 8I `nested_priority`) via `glm::packHalf2x16()` → `std::bit_cast<float>()` into `.a`

4. Update `shaders/include/constants.glsl`:
   - `kMaterialStride = 8u` (was 7)

5. Update `shaders/include/mis.glsl` — add diffuse transmission strategy:
   - Add `STRATEGY_DIFFUSE_TRANSMISSION = 4` constant, `NUM_STRATEGIES = 5`
   - Add `diffuse_transmission` field to `SamplingProbabilities` and `AllPDFs`
   - Update `calculateSamplingProbabilities()` — add `float diffuse_transmission_factor` parameter. Compute `prob_diffuse_reflect = base_diffuse * (1.0 - diffuse_transmission_factor)` and `prob_diffuse_transmit = base_diffuse * diffuse_transmission_factor`. Apply `kMinStrategyProb` floor only when `diffuse_transmission_factor > 0`. Normalize all 5 strategies.
   - Add `calculateDiffuseTransmissionPDF(N, L)` — returns `max(-dot(N, L), 0.0) / PI` (cosine PDF for back hemisphere)
   - Update `calculateAllPDFs()` — compute `pdfs.diffuse_transmission` via `calculateDiffuseTransmissionPDF()`
   - Update `chooseStrategy()` — extend CDF for 5 strategies
   - Update `calculateMISWeight()` — add `w5 = probs.diffuse_transmission * pdfs.diffuse_transmission` to power heuristic sum

6. Add `evaluateDiffuseTransmission()` to `shaders/include/brdf.glsl`:
   ```glsl
   vec3 evaluateDiffuseTransmission(vec3 albedo, vec3 dt_color,
                                     float NdotL_back,
                                     float diffuse_transmission_factor) {
       return albedo * dt_color * diffuse_transmission_factor / PI
              * max(-NdotL_back, 0.0);
   }
   ```

7. Restructure `raygen.rgen` specular transmission block and integrate diffuse transmission:

   **a. Restructure the specular transmission block** for energy-correct delta/smooth split. Replace the current `if (transmission > 0.0) { ... continue; }` block with:
   ```glsl
   if (transmission > 0.0) {
       // ... compute Fresnel as before ...
       float p_reflect = fresnel;
       float p_transmit = (1.0 - fresnel) * transmission;
       // p_smooth = (1.0 - fresnel) * (1.0 - transmission) → falls through to MIS

       float delta_rand = trans_rands.x;
       if (delta_rand < p_reflect) {
           // Specular reflection (delta)
           ray_dir = reflect(-V, N);
           transparent_count++;
           ray_origin = payload.hit_pos + ray_dir * kSurfaceBias;
           continue;
       } else if (delta_rand < p_reflect + p_transmit) {
           // Specular transmission (delta)
           if (thin_surface) {
               ray_dir = -V;  // Straight through, no IOR bend
           } else {
               // Existing refract() code with TIR fallback
           }
           // Thin-slab attenuation if applicable
           transparent_count++;
           ray_origin = payload.hit_pos + ray_dir * kSurfaceBias;
           continue;
       }
       // else: fall through to MIS block for smooth lobes
   }
   ```

   **b. Read diffuse transmission fields from material** (after the existing material unpacking):
   - `float dt_factor = transmission_ext.r;`
   - `float thin_surface = transmission_ext.g;`
   - `vec2 dt_color_rg = unpackHalf2x16(floatBitsToUint(transmission_ext.b));`
   - `vec2 dt_color_ba = unpackHalf2x16(floatBitsToUint(transmission_ext.a));`
   - `vec3 dt_color = vec3(dt_color_rg, dt_color_ba.x);`

   **c. Pass `dt_factor` to `calculateSamplingProbabilities()`** to compute the 5-way split.

   **d. Add `STRATEGY_DIFFUSE_TRANSMISSION` sampling branch:** Sample cosine hemisphere on the **opposite** side of the normal (`-N`): `L = -cosineSampleHemisphere(rands.xy, N);` (negate the result of the standard cosine sample).

   **e. Add `STRATEGY_DIFFUSE_TRANSMISSION` evaluation:** Call `evaluateDiffuseTransmission(albedo, dt_color, dot(N, L), dt_factor)`.

   **f. Mark diffuse transmission paths as `is_specular_path = false`** (they are diffuse).

8. Update glTF loader to parse `KHR_materials_diffuse_transmission`:
   - cgltf v1.14 does not natively support this extension. Iterate `gmat.extensions[0..extensions_count-1]` and match `name == "KHR_materials_diffuse_transmission"`.
   - Parse the raw JSON `data` string (a simple key-value object) for:
     - `"diffuseTransmissionFactor"` → `float diffuse_transmission_factor`
     - `"diffuseTransmissionColorFactor"` → `glm::vec3 diffuse_transmission_color`
     - `"diffuseTransmissionTexture"` → deferred (log warning, do not parse texture reference in v1)
   - Set `thin_surface = true` when the extension is present (glTF diffuse transmission implies thin-surface geometry).
   - Add a minimal JSON number/array parser (or use an existing utility) to extract float values from the raw JSON string. The JSON is a flat object — no nested structures. Example format: `{"diffuseTransmissionFactor":0.8,"diffuseTransmissionColorFactor":[0.2,0.8,0.1]}`.

### Verification

`tests/phase8h_test.cpp` — GPU integration tests (Catch2 `TEST_CASE`) that verify diffuse transmission and thin-surface mode through rendered output. Uses `TestContext`, `MakeQuad()`, `WriteCombinedPNG()`, `AnalyzeRGBA16F()`, and `ComputeMeanFlip()` from the existing test infrastructure.

1. **`DiffuseTransmissionBacklitLeaf`** (GPU integration) — Build a thin quad with `diffuse_transmission_factor = 0.8`, `diffuse_transmission_color = {0.2, 0.8, 0.1}` (green), and white base color, lit from behind by an area light. Render at 64 spp. Verify the front face (camera side, opposite the light) has non-zero green illumination. Render the same scene with `diffuse_transmission_factor = 0.0` and verify the front face is significantly darker. FLIP between the two > 0.05. This test regresses if diffuse transmission is not implemented or broken.

2. **`ThinSurfaceNoRefraction`** (GPU integration, two-config comparison) — Render a glass panel (`transmission_factor = 1.0`, IOR = 1.5) with `thin_surface = true` and with `thin_surface = false`. Place a colored Cornell box behind the panel. With `thin_surface = true`, geometry behind the panel should appear undistorted (straight-through). With `thin_surface = false`, refraction should shift the geometry. FLIP between the two > 0.02 confirms the thin-surface flag changes behavior.

3. **`DiffuseTransmissionColorTinting`** (GPU integration) — Build a quad with `diffuse_transmission_factor = 0.8` and `diffuse_transmission_color = {1.0, 0.0, 0.0}` (red tint), base color white, lit from behind by a white area light. Render at 64 spp. Verify the transmitted light on the front face is red-dominant (R channel significantly higher than G and B). Confirms the color tinting pipeline from `MaterialDesc` → `PackedMaterial` (half-float packing) → shader unpacking → BRDF evaluation.

4. **`DiffuseTransmissionConvergence`** (GPU integration, convergence) — Scene with translucent materials at 4 spp vs 64 spp. FLIP below convergence threshold.

5. **`DiffuseTransmissionNoNaN`** (GPU integration) — Render with `diffuse_transmission_factor = 1.0` (all light transmitted, no reflection). Verify no NaN/Inf in output.

6. **`SpecularPlusDiffuseTransmission`** (GPU integration, coexistence) — Build a thin panel with both `transmission_factor = 0.5` (half specular transmission) and `diffuse_transmission_factor = 0.6`, `thin_surface = true`. Render at 64 spp. Verify no NaN/Inf and that the output differs from a panel with only specular transmission (`diffuse_transmission_factor = 0.0`). FLIP > 0.02 confirms both lobes contribute independently.

- No NaN/Inf; no Vulkan validation errors.

---

## Phase 8I: Nested Dielectric Priority

**Goal:** Implement a material priority system for correctly handling overlapping dielectric volumes (e.g., liquid inside glass, coated objects). Without priority, the renderer cannot determine which IOR to use when exiting one volume and entering another simultaneously.

**Source:** RTXPT `InteriorList.hlsli` and `PathTracerNestedDielectrics.hlsli` (sorted interior list model), "Simple Nested Dielectrics in Ray Traced Images" (Schmidt & Budge, JGT 2002)

### Design Decisions

- **Sorted interior list (matching RTXPT).** Each transmissive material has a 4-bit integer `nested_priority` (0–14 usable; 0 is remapped internally to `kMaxNestedPriority = 15`, making it the highest priority). When a ray hits a dielectric surface, the interior list determines which medium's IOR governs the interface. A higher-priority medium "wins" at shared boundaries: lower-priority surface intersections inside a higher-priority volume are *rejected* (the ray passes through without shading), matching RTXPT's `isTrueIntersection()` behavior.
- **Slot-based interior list.** The interior list uses 2 slots (matching RTXPT's default `INTERIOR_LIST_SLOT_COUNT = 2`). Each slot packs a `materialID` (28 bits) and `nestedPriority` (4 bits) into a single `uint`, with priority in the high bits so a simple integer sort keeps the list ordered highest-priority-first. Empty slots are 0. The list is kept sorted after every insert/remove via a 2-element sorting network (`CSWAP`).
- **Local variable in bounce loop.** The interior list is a local variable in the raygen shader's bounce loop, initialized empty at path start. This works because monti uses iterative path tracing in a single raygen invocation (not recursive `traceRay` calls). This matches the conceptual role of RTXPT's `PathState.interiorList`, which persists across bounces in their payload.
- **IOR lookup via material buffer.** When computing the outside IOR, the interior list provides a `materialID`; the shader reads that material's IOR from the material storage buffer (at `materials.data[materialID * kMaterialStride + 2].g`, the `opacity_ior.g` field). This avoids storing IOR redundantly in the stack and matches RTXPT's `Bridge::loadIoR(materialID)` pattern.
- **Outside IOR computation (matching RTXPT `ComputeOutsideIoR`).** On a hit:
  - If entering: the outside medium is the current top-of-stack material. `n1 = loadIOR(top material)` (or 1.0 if stack is empty).
  - If exiting: if the exiting material *is* the top of the stack, `n1 = loadIOR(next material)` (or 1.0 if only one entry). Otherwise `n1 = loadIOR(top material)`.
  - `n2` is always the hit material's own IOR.
  - Fresnel and refraction use `eta = n1 / n2` when entering, `eta = n2 / n1` when exiting.
- **False intersection rejection (matching RTXPT `HandleNestedDielectrics`).** Before shading a transmissive hit, check `isTrueIntersection(nestedPriority)` — true if the hit's priority ≥ the stack's top priority. On false intersection: call `handleIntersection()` to update the list, offset the ray origin to the opposite side of the surface, skip shading, and continue the bounce loop (decrement bounce counter so the false hit doesn't consume a bounce). A maximum of 4 rejected hits per path prevents infinite loops in pathological cases; if exceeded, the path is terminated.
- **Thin surfaces skip the interior list (matching RTXPT).** If a surface has `thin_surface = true`, the interior list is not modified (no push/pop). Thin surfaces transmit without entering a volume. Diffuse transmission also skips the interior list — it is a surface-only scattering event, not a volume boundary crossing.
- **Priority packing in `PackedMaterial`.** Priority is stored in the second half-float of `transmission_ext.a`, which Phase 8H packs as `packHalf2x16(vec2(diffuse_transmission_color.b, 0.0))`. Phase 8I replaces the `0.0` with `float(nested_priority)`, decoded in the shader as `uint(unpackHalf2x16(...).y + 0.5)`. The value is then remapped: 0 → `kMaxSlotPriority` (15), nonzero → `min(value, kMaxSlotPriority)`.
- **No custom glTF extensions.** There is no standard glTF extension for nested dielectric priority. The `nested_priority` field defaults to 0. Priority values are set programmatically by the application or test code. The glTF loader does not parse a priority property.

### Tasks

1. Add `nested_priority` to `MaterialDesc`:
   ```cpp
   uint8_t nested_priority = 0;  // 0-14: nesting priority (0 = default/highest after remap)
   ```

2. Pack into `PackedMaterial::transmission_ext.a` (second half-float): update the Phase 8H packing from `packHalf2x16(vec2(dt_color.b, 0.0))` to `packHalf2x16(vec2(dt_color.b, float(nested_priority)))`.

3. Add `MakeIcosphere` to `tests/scenes/Primitives.h/.cpp` — a shared test utility that generates an icosphere `MeshData` by recursive subdivision of an icosahedron:
   ```cpp
   // center: world-space center; radius: sphere radius;
   // subdivisions: recursion depth (0 = icosahedron, 1 = 42 verts, 2 = 162 verts, etc.)
   MeshData MakeIcosphere(const glm::vec3& center, float radius, uint32_t subdivisions = 2);
   ```

4. Implement `InteriorList` in `shaders/include/interior_list.glsl`:
   ```glsl
   const uint kInteriorListSlots = 2u;
   const uint kMaterialBits      = 28u;
   const uint kPriorityBits      = 4u;
   const uint kMaterialMask      = (1u << kMaterialBits) - 1u;
   const uint kPriorityOffset    = kMaterialBits;
   const uint kMaxSlotPriority   = (1u << kPriorityBits) - 1u;  // 15
   const uint kNoMaterial        = 0xFFFFFFFFu;

   struct InteriorList {
       uint slots[kInteriorListSlots];  // each: priority[31:28] | materialID[27:0]; 0 = empty
   };

   uint  makeSlot(uint material_id, uint priority);
   bool  isSlotActive(uint slot);
   uint  getSlotPriority(uint slot);
   uint  getSlotMaterialID(uint slot);
   void  sortSlots(inout InteriorList list);       // 2-element sorting network
   bool  isEmpty(InteriorList list);
   uint  getTopPriority(InteriorList list);
   uint  getTopMaterialID(InteriorList list);      // returns kNoMaterial if empty
   uint  getNextMaterialID(InteriorList list);     // returns kNoMaterial if < 2 entries
   bool  isTrueIntersection(InteriorList list, uint priority);  // priority >= top priority
   void  handleIntersection(inout InteriorList list, uint material_id,
                            uint priority, bool entering);
   ```

5. Implement `computeOutsideIOR` in `shaders/include/interior_list.glsl`:
   ```glsl
   // Returns the IOR of the medium on the outside of the current interface.
   // material_id: the material being hit; entering: front-face hit.
   // Reads IOR from material buffer: materials.data[id * kMaterialStride + 2].g
   float computeOutsideIOR(InteriorList list, uint material_id, bool entering);
   ```

6. Update `raygen.rgen` transmission code path:
   - Declare `InteriorList interior_list` before the bounce loop, initialized to `{uint[kInteriorListSlots](0, 0)}`.
   - Before shading a transmissive hit (where `transmission > 0.0` and `thin_surface < 0.5`): unpack `nested_priority` from `transmission_ext.a`, remap (0 → `kMaxSlotPriority`, nonzero → `min(value, kMaxSlotPriority)`), call `isTrueIntersection()`. If false: call `handleIntersection()`, offset ray origin past the surface (`computeRayOrigin` with flipped normal), decrement bounce counter, `continue`. Track rejected hit count; terminate path after 4 rejections.
   - If true intersection: call `computeOutsideIOR()` to get `n1`. Use the hit material's IOR as `n2`. Replace the current hardcoded `n1 = entering ? 1.0 : ior; n2 = entering ? ior : 1.0;` with the interior-list-derived values.
   - After a transmission scatter (ray refracts through the surface): call `handleIntersection(interior_list, material_id, priority, entering)` to update the list. Track `inside_dielectric_volume = !isEmpty(interior_list)` for future use.
   - Skip all interior list operations for thin surfaces and diffuse transmission bounces.

### Verification

`tests/phase8i_test.cpp` — GPU integration tests that verify nested dielectric priority through rendered output.

**Prerequisites:** `MakeIcosphere` from `tests/scenes/Primitives.h` for building spherical geometry.

1. **`NestedDielectricGlassInGlass`** (GPU integration, two-scene comparison) — Build a scene with an inner icosphere (IOR 1.5, `nested_priority = 1`) inside a larger icosphere (IOR 1.33, `nested_priority = 2`), with a colored Cornell box background. Also build a second scene with the inner sphere alone (no outer sphere, same camera). Render both at 64 spp. FLIP between the two renders > 0.02, confirming the outer medium's IOR affects the inner sphere's refraction. This test regresses if the interior list is broken (inner sphere would render identically in both cases because `n1` would default to 1.0 in both).

2. **`NestedDielectricFalseIntersection`** (GPU integration) — Build a scene with a high-priority outer icosphere (`nested_priority = 3`) and a low-priority inner icosphere (`nested_priority = 1`) overlapping it. Render at 64 spp. The low-priority inner surface should be invisible (false intersection rejected). Compare against a scene with only the outer sphere. FLIP < 0.01, confirming false intersection rejection works — the inner sphere's boundary is skipped.

3. **`NestedDielectricStackOverflow`** (GPU integration, edge case) — Build a scene with 8 concentric transmissive icospheres (exceeding `kInteriorListSlots = 2`). Render at 4 spp. Verify no NaN/Inf and no GPU hang — confirms graceful behavior when the interior list is full (excess insertions are silently dropped, matching RTXPT).

4. **`NestedDielectricThinSurfaceBypass`** (GPU integration) — Build two scenes: (A) a transmissive icosphere with `thin_surface = true` and `nested_priority = 5` inside another transmissive icosphere, and (B) the same scene with `thin_surface = false` on the inner sphere. Render both at 64 spp. In scene A, the interior list should be unaffected by the thin inner sphere. FLIP between A and B > 0.02, confirming thin surfaces skip the interior list.

- No NaN/Inf; no Vulkan validation errors.

---

## Phase 8J: Emissive Mesh Light Extraction

**Goal:** Automatically extract emissive mesh surfaces into triangle light primitives so they contribute to NEE (next-event estimation) via shadow rays. Currently, emissive surfaces only contribute when a path randomly bounces into them; this phase enables explicit light sampling of emissive geometry.

**Source:** RTXPT emissive triangle extraction (compute shader), "Practical Path Guiding for Efficient Light-Transport Simulation" (Müller et al.)

### Design Decisions

- **Compute shader extraction at scene load time.** A Vulkan compute shader scans all materials for `emissive_strength > 0`, reads the corresponding mesh triangles from the buffer address table, and writes `TriangleLight` entries into the light buffer. This runs once at scene load (or when the scene changes), not per frame.
- **Per-triangle emissive radiance.** Each extracted triangle inherits `emissive_factor * emissive_strength` from its material. If an emissive texture is present, the average texel luminance over the triangle's UV region is used (approximated by sampling the texture at the triangle's centroid UV, which is fast and sufficient for uniform-ish emissive textures). Per-texel emissive variation is captured naturally by the path tracer's BSDF sampling — the extracted lights provide NEE importance.
- **De-duplication with explicit lights.** Extracted emissive triangle lights are appended to the same `PackedLight` buffer as explicit area/sphere/triangle lights from Phase 8G. The push constant `light_count` includes all light types.
- **Threshold for extraction.** Only triangles with `emissive_strength * max(emissive_factor) > kMinEmissiveLuminance` are extracted. Dim emitters below the threshold rely on random path hits only (minimal contribution, not worth shadow rays).

### Tasks

1. Add `kMinEmissiveLuminance` to `constants.glsl` (default: `0.01`).

2. Implement `EmissiveLightExtractor` in `renderer/src/vulkan/EmissiveLightExtractor.h/.cpp`:
   - Scan `Scene::Materials()` for emissive materials
   - For each emissive material, find all `SceneNode` referencing it
   - Read triangle vertices from the mesh address table (CPU-side, from `GpuScene` bindings)
   - Transform triangle vertices to world space using node transforms
   - Compute per-triangle emissive radiance
   - Produce `std::vector<TriangleLight>` for `GpuScene::UpdateLights()`

3. Wire extraction into `Renderer::RenderFrame()`:
   - Call `EmissiveLightExtractor` after scene setup, before light buffer upload
   - Add extracted triangle lights to `Scene::TriangleLights()` (or pass separately to `GpuScene::UpdateLights()`)

4. Update light buffer to accommodate extracted lights (may need larger allocation).

### Verification

`tests/phase8j_test.cpp` — GPU integration tests that verify emissive mesh light extraction through rendered output.

1. **`EmissiveMeshNEEReducesNoise`** (GPU integration, two-scene comparison) — Build two scenes: (A) a glTF object with emissive material (emissive_strength=100) in a dark Cornell box (emissive triangles extracted into light buffer for NEE), and (B) the same scene but with emissive_strength set below `kMinEmissiveLuminance` (so the mesh emits via random path hits only, no extracted triangle lights). Render both at 4 spp. The scene with extraction (A) should have lower pixel variance on surfaces near the emissive object. Verify `variance_A < variance_B * 0.7`. This test regresses if extraction fails to add triangle lights to the light buffer.

2. **`EmissiveMeshFLIPImprovement`** (GPU integration, convergence) — Same scene as above. Render at 4 spp and 64 spp with extraction enabled. FLIP(4spp, 64spp) below convergence threshold, confirming extraction improves convergence.

3. **`EmissiveMeshExtractionThreshold`** (GPU integration) — Set emission below `kMinEmissiveLuminance`. Verify the light count in the light buffer does not include the dim emissive triangles (confirm via a diagnostic counter or by verifying dim emitters produce no NEE shadow rays — equivalent to the no-extraction case).

4. **`EmissiveMeshNoNaN`** (GPU integration) — Render a scene with both extracted and explicit lights. Verify no NaN/Inf in output.

- No Vulkan validation errors.

---

## Phase 8K: Weighted Reservoir Sampling for NEE

**Goal:** Replace the O(N) per-light NEE loop with O(1) weighted reservoir sampling (WRS). When scenes contain many lights (dozens of explicit lights + hundreds of emissive triangles from Phase 8J), iterating all lights per hit point is prohibitively expensive. WRS selects a single light with probability proportional to its estimated contribution.

**Source:** RTXPT `LightSampling.hlsli` (WRS), "Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting" (Bitterli et al., 2020)

### Design Decisions

- **Single-pass streaming WRS.** For each hit point, iterate all lights in a single pass maintaining a reservoir of size 1. Each light's selection weight is its estimated contribution: `weight_i = luminance(radiance_i) * geometric_factor_i / distance²_i` (a cheap unshadowed estimate). The selected light gets a full shadow ray trace and BRDF evaluation. The MIS weight accounts for the WRS selection probability.
- **No temporal/spatial resampling.** This phase implements basic WRS only — a foundation for future ReSTIR DI (Phase F2) which adds temporal and spatial reservoir resampling for vastly improved quality. Basic WRS alone converts O(N) per-light cost to O(1) with correct (if noisier) results.
- **WRS replaces the light iteration loop.** The existing per-light shadow ray loop from Phase 8B/8G is replaced entirely. With WRS, exactly one light is sampled per hit per bounce, regardless of light count.

### Tasks

1. Create `shaders/include/wrs.glsl` — reservoir data structure and sampling:
   ```glsl
   struct Reservoir {
       uint  selected_light;
       float selected_weight;
       float weight_sum;
       uint  sample_count;
   };

   void initReservoir(out Reservoir r);
   void updateReservoir(inout Reservoir r, uint light_index,
                        float weight, float random);
   float getReservoirPdf(Reservoir r);
   ```

2. Create `shaders/include/light_sampling.glsl` — WRS-based light selection:
   ```glsl
   Reservoir selectLight(vec3 shading_pos, vec3 N,
                         uint light_count, float random_seed) {
       Reservoir r;
       initReservoir(r);
       for (uint i = 0; i < light_count; ++i) {
           PackedLight light = lights.data[i];
           float weight = estimateLightContribution(light, shading_pos, N);
           float rand_i = fract(random_seed + float(i) * 0.618033988749895);
           updateReservoir(r, i, weight, rand_i);
       }
       return r;
   }
   ```

3. Update `raygen.rgen` — replace per-light loop with WRS:
   - At each bounce, call `selectLight()` to pick one light
   - Sample the selected light via `sampleLight()` from Phase 8G
   - Trace a single shadow ray
   - MIS weight: account for WRS selection probability in the estimator

4. Update `shaders/include/lights.glsl` — add `estimateLightContribution()`:
   - Cheap unshadowed estimate: `luminance(radiance) * solid_angle_estimate`
   - Different geometric factor per light type (quad, sphere, triangle)

### Verification

`tests/phase8k_test.cpp` — GPU integration tests that verify WRS light selection through rendered output.

1. **`WRSSingleLightMatchesO_N`** (GPU integration, backward compatibility) — Render Cornell box with exactly 1 area light using WRS. FLIP against the pre-WRS O(N) render at 64 spp < 0.01. Confirms WRS trivially selects the only light and produces identical results.

2. **`WRSManyLightsConvergence`** (GPU integration, convergence) — Build a scene with 100+ mixed lights (quad + sphere + triangle). Render at 4 spp and 64 spp with WRS. FLIP(4spp, 64spp) below convergence threshold. Confirms WRS produces unbiased results at high SPP.

3. **`WRSManyLightsConstantTime`** (GPU integration, performance) — Render the same scene with 10 lights and 200 lights at identical SPP. Verify frame time ratio (200-light / 10-light) is < 1.5 (approximately O(1) vs the O(N) expectation of ~20×). This test regresses if WRS is accidentally disabled and the O(N) loop returns.

4. **`WRSManyLightsNoNaN`** (GPU integration) — Render the 100+ light scene at 1 spp. Verify no NaN/Inf.

- No NaN/Inf; no Vulkan validation errors.


---

## Phase 10A-2: Extended Scene Acquisition + Golden Test Expansion

**Goal:** Add CMake infrastructure for downloading large test scenes, establish golden reference tests for all core and extended scenes.

**Prerequisite:** Phase 10A (end-to-end pipeline working).

### Design Decisions

- **Two-tier scene assets.** Core scenes (small, already committed to `tests/assets/`) are always available. Extended scenes (multi-GB) are downloaded on demand via a CMake option.

- **Default camera for all scenes.** All test scenes use the same auto-fit camera placement as `monti_view`: positioned on −Z axis at a distance that fits the scene AABB within a 60° FOV. No per-scene camera config files.

- **Golden references generated by test infrastructure.** Golden reference PNGs are rendered at high SPP by the test executable itself (not by `monti_datagen`, which is Phase 11B). A dedicated test or script renders each scene through the full pipeline at 256+ SPP, tone-maps, and writes the result as a PNG to `tests/golden/`.

### Tasks

1. Extended scene download infrastructure:
   - Add CMake option `MONTI_DOWNLOAD_EXTENDED_SCENES` (default OFF) that fetches large glTF scenes at configure time
   - Download targets (verified URLs):
     - Amazon Lumberyard Bistro: from [NVIDIA RTXPT-Assets](https://github.com/NVIDIA-RTX/RTXPT-Assets)
     - NVIDIA Emerald Square: from [NVIDIA RTXPT-Assets](https://github.com/NVIDIA-RTX/RTXPT-Assets)
     - Intel Sponza: from [Intel Graphics Research Samples](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html)
     - Khronos ToyCar: from [glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets)
     - Khronos FlightHelmet: from [glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets)
   - Downloaded to `tests/assets/extended/` (gitignored)

2. Golden reference generation and storage:
   - Add `tests/golden/` directory structure (gitignored for large files)
   - Create a test or helper that renders each scene at 256 spp → tonemap → write PNG to `tests/golden/`
   - Golden references are generated locally and committed selectively (core scenes only)

3. Golden test expansion:
   - Create `tests/golden_test.cpp` with parameterized test cases for each core scene
   - Each test: load scene → auto-fit camera → render 256 SPP → tonemap → FLIP compare against stored golden reference
   - FLIP thresholds: mean < 0.05 for simple scenes (Cornell, Box), mean < 0.08 for complex scenes (Bistro, Sponza)
   - Core scenes (always run in CI): Box.glb, DamagedHelmet.glb, DragonAttenuation.glb, ClearCoatTest.glb, MorphPrimitivesTest.glb, programmatic CornellBox
   - Extended scenes (CI nightly/manual only, guarded by `MONTI_DOWNLOAD_EXTENDED_SCENES`): Bistro, Sponza, ToyCar, FlightHelmet

### Verification
- All core golden tests pass (FLIP mean < 0.05)
- Extended golden tests pass when `MONTI_DOWNLOAD_EXTENDED_SCENES=ON` (FLIP mean < 0.08)
- Scene download is idempotent (re-running configure does not re-download)
- Golden reference images are visually inspectable as diagnostic artifacts

### RTXPT Reference
- [RTXPT-Assets](https://github.com/NVIDIA-RTX/RTXPT-Assets): Bistro, Kitchen, A Beautiful Game, Emerald Square scene data


---

## Future Phases (Not in Initial Plan)

These are documented for roadmap visibility but not scheduled. See [roadmap.md](roadmap.md) for detailed breakdowns.

| Future Phase | Description | Prerequisite |
|---|---|---|
| F1 | DLSS-RR in `monti_view` (NVIDIA-only, app-level quality reference) + denoiser selection UI | Phase 10B complete (interactive viewer with ImGui); add Passthrough / DLSS-RR toggle to settings panel |
| F2 | ReSTIR Direct Illumination | Phase 8K complete (WRS foundation) |
| F3 | Emissive mesh importance sampling via ReSTIR | Phase 8J + F2 |
| F4 | Volume enhancements (homogeneous scattering, heterogeneous media) | Phase 8I complete (nested dielectrics) |
| F6 | Mobile Vulkan renderer (`monti_vulkan_mobile`) | Phase 8K complete (shared GpuScene/GeometryManager); hybrid rasterization (default) + ray query pipeline; projection-matrix jitter for TAA; format-agnostic G-buffer via `shaderStorageImageReadWithoutFormat` |
| F7 | Metal renderer (C API) | Phase 8K design patterns established |
| F8 | WebGPU renderer (C API → WASM) | Phase 8K design patterns established |
| F9 | ML denoiser training pipeline | Phase 11B complete (training data capture working) |
| F10 | Shader permutation cache | Phase 8K complete |
| F11 | ML denoiser deployment in Deni (desktop + mobile) | F9 complete (trained model weights available) |
| F12 | Super-resolution in ML denoiser | F11 complete; uses `ScaleMode` enum in `DenoiserInput` (kQuality 1.5×, kPerformance 2×) |
| F13 | Fragment shader denoiser (mobile) | F6 + F11 complete; denoise → tonemap → present as render pass subpasses; `Denoiser` auto-selects compute vs fragment based on device |
| F14 | GPU skinning + morph targets | Phase 6 complete (BLAS refit hooks); compute shader pipeline for joint transforms + morph weight blending, BLAS refit on deformed vertices |
| F16 | NRD ReLAX denoiser in Deni (cross-vendor, open-source) | F11 complete (only needed if cross-vendor denoising is required before ML denoiser quality is sufficient) |

---

## Dependency Graph

```
Phase 1 (skeleton)
  ├─→ Phase 2 (scene)
  │     └─→ Phase 3 (glTF loader)
  │           └─→ Phase 5 (GpuScene) ─→ Phase 6 (accel structs)
  │                                       └─→ Phase 7A (G-buffer + env map)
  │                                             └─→ Phase 7B (pipeline + SBT)
  │                                                   └─→ Phase 7C (shaders + RenderFrame)
  │                                                         └─→ Phase 8A (GLSL lib + single-bounce)
  │                                                               └─→ Phase 8B (multi-bounce MIS)
  │                                                                     └─→ Phase 8C (transparency + transmission)
  │                                                                           └─→ Phase 8D (PBR textures + normal map + emissive)
  │                                                                                 ├─→ Phase 8E (firefly + hit distance)
  │                                                                                 ├─→ Phase 8F (ray cone LOD)
  │                                                                                 ├─→ Phase 8G (sphere + triangle lights)
  │                                                                                 │     └─→ Phase 8J (emissive mesh extraction)
  │                                                                                 │           └─→ Phase 8K (WRS for NEE)
  │                                                                                 ├─→ Phase 8H (diffuse transmission + thin-surface)
  │                                                                                 └─→ Phase 8I (nested dielectric priority)
  ├─→ Phase 4 (Vulkan context + app scaffolding)
  │     └─→ Phase 5 ─→ ... ─→ Phase 8D
  │                                ├─→ Phase 9B (denoiser integration) ─→ Phase 10A (monti_view: tonemap + present)
  │                                │                                          ├─→ Phase 10A-2 (extended scenes + golden tests)
  │                                │                                          ├─→ Phase 10B (monti_view: interactive camera + ImGui)
  │                                │                                          │     └─→ F1 (DLSS-RR + denoiser selection UI)
  │                                │                                          └─→ Phase 11B (monti_datagen: readback + headless)
  │                                └─→ Phase 10A (monti_view: tonemap + present)
  ├─→ Phase 9A (standalone denoiser)
  │     └─→ Phase 9C (deni loader-agnostic dispatch) ─→ Phase 9B
  │     └─→ Phase 9B (denoiser integration)
  ├─→ Phase 9D (monti_vulkan loader-agnostic dispatch + app updates)
  │     Requires: Phase 9C (proven pattern) + Phase 7C+ (monti_vulkan implementation)
  └─→ Phase 11A (capture writer — CPU-only)        ─→ Phase 11B
```

Phases 2 and 4 can be developed in parallel. Phase 9A (standalone denoiser library) can be developed in parallel with Phases 2–8 since it has no Monti dependencies. Phase 9C (deni loader-agnostic dispatch) follows 9A and should be completed before 9B so the integration test uses the final loader-agnostic API. Phase 9D (monti_vulkan loader-agnostic dispatch) requires both 9C (proven pattern) and sufficient monti_vulkan implementation (Phase 7C+). Phase 11A (capture writer) can also be developed in parallel with Phases 2–10 since it is CPU-only with no GPU dependency. Phase 9B requires both 8D and 9A (or 9C). Phase 10A (`monti_view` tonemap + present) can start after 8D + 9B. Phase 10A-2 (extended scenes + golden tests) depends on 10A. Phase 10B (`monti_view` interactive camera + ImGui) depends on 10A. Phase 11B (`monti_datagen` headless data generator) depends on 10A + 11A. Phases 8E–8K can be developed in any order after 8D, except: 8J requires 8G (triangle light type), 8K requires 8G+8J (light buffer with all types).

**Denoiser strategy:** Deni ships with a passthrough denoiser only (Phases 9A/9B), made loader-agnostic in Phase 9C. DLSS-RR is integrated at the app level in `monti_view` (F1) as the quality reference denoiser during development — this leverages the existing rtx-chessboard DLSS-RR + Volk integration. The ML denoiser is trained (F9) and deployed in Deni (F11) using DLSS-RR output as the quality ceiling comparison. NRD ReLAX (F16) is deferred until cross-vendor denoising is needed. ReBLUR is not planned.

**Vulkan loader strategy:** volk is confined to the application layer (`monti_view`, `monti_datagen`, test executables). Both product libraries (`deni_vulkan`, `monti_vulkan`) resolve Vulkan functions via `PFN_vkGetDeviceProcAddr` passed by the host — no build-time loader dependency. This enables any Vulkan application to integrate Deni regardless of their Vulkan loading strategy.
