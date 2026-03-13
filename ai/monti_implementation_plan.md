# Monti & Deni — Master Implementation Plan

> **Purpose:** Incremental build plan for the Monti path tracer and Deni denoiser libraries. Each phase produces a verifiable deliverable. Phases are sequential; later phases build on earlier ones. The plan references the architecture in [monti_design_spec.md](monti_design_spec.md) and adapts code from the [rtx-chessboard](../../../rtx-chessboard/) Vulkan path tracer. The application executables (`monti_view`, `monti_datagen`) are specified in [app_specification.md](app_specification.md) and will be implemented after the core libraries are functional.
>
> **Session sizing:** Each phase (or sub-phase) is scoped to fit within a single Copilot Claude Opus 4.6 context session — roughly 2–3 new/modified source files referencing 3–5 existing files, producing one verifiable deliverable.
>
> **Platform:** Initial implementation targets MSVC on Windows. GCC/Clang cross-platform support will be added when needed.

---

## Testing Philosophy

- **Prefer integration tests.** Every phase ends with a test that exercises the full code path from input to output — load a scene, upload to GPU, render a frame, verify pixels. Automated perceptual comparison replaces manual visual inspection.
- **Unit tests only for complex isolated logic.** Reserve unit tests for non-trivial algorithms that can be tested independently and where integration tests would be slow or unreliable (e.g., CDF computation, material packing math). Simple data containers and ID types get compile-time verification, not dedicated tests.
- **Vulkan validation layers are always on** in debug builds. Zero validation errors is a pass/fail gate for every GPU phase.

### Automated Render Validation

**Tool:** [NVIDIA FLIP](https://github.com/NVlabs/flip) (BSD-3 license) — a perceptual image comparison metric designed specifically for rendered images. It models human contrast sensitivity and produces a per-pixel error map with a single mean error score. Fetched via `FetchContent` in CMake; the C++ library has no heavy dependencies.

**Two-tier validation strategy:**

1. **Self-consistency (convergence) tests** — Render the same scene at low SPP (e.g., 4) and high SPP (e.g., 256). Compute FLIP between the two. The score must be below a threshold (proves the renderer converges correctly without requiring stored reference images). These tests are resilient to intentional rendering changes.

2. **Golden reference regression tests** — A small curated set of high-SPP reference images stored in the repo. Compare each test render against its reference using FLIP. Threshold: mean FLIP < 0.05 (tuned during Phase 8A). When rendering changes are intentional, update the reference images.

Self-consistency tests are the primary automated gate. Golden reference tests catch regressions but require manual update when the renderer changes intentionally. Both test types produce FLIP error maps as artifacts for debugging failures.

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

---

## Overview

| Phase | Deliverable | Verifiable Outcome |
|---|---|---|
| 1 | Project skeleton + build system | `cmake --build` succeeds, empty libraries link |
| 2 | Scene layer (`monti_scene`) | Integration test: build Cornell box, verify data round-trip + MeshData |
| 3 | glTF loader | Integration test: load glTF, verify mesh/material/texture counts + MeshData |
| 4 | Vulkan context + app scaffolding | `monti_view`: window opens, swapchain presents a cleared color. Headless context test passes. |
| 5 | GPU scene (`monti::vulkan::GpuScene`) | Integration test: register mesh buffers → verify bindings; pack materials → verify buffer |
| 6 | Acceleration structures (`GeometryManager`) | BLAS + TLAS built, compacted, device addresses valid |
| 7A | G-buffer images + environment map + blue noise | Environment map loaded, CDF buffers valid, G-buffer images allocated |
| 7B | Descriptor sets + pipeline + SBT | Ray tracing pipeline created, SBT populated, no validation errors |
| 7C | Raygen + miss + closesthit stub | Window shows environment map; glTF silhouettes visible (normals as color) |
| 8A | GLSL shader library + single-bounce PBR | Textured PBR scene renders with correct single-bounce shading |
| 8B | Multi-bounce MIS + clear coat | Multi-bounce reflections, MIS convergence, clear coat visible |
| 8C | Transparency + transmission + G-buffer aux + jitter | Fresnel refraction, volume attenuation, correct motion vectors, complete G-buffer |
| 8D | PBR texture sampling + normal mapping + emissive + MIS fix | Normal maps, metallic-roughness maps, emissive direct, named constants |
| 8E | Firefly filter + hit distance output | Luminance-based firefly clamping, RG16F linear depth + hit distance, `phase8e_test.cpp` passes |
| 8F | Ray cone texture LOD | Automatic mip selection via ray cone tracking, reduced texture aliasing |
| 8G | Spherical area lights + triangle light primitives | Sphere/triangle light types, unified PackedLight buffer |
| 8H | Diffuse transmission + thin-surface mode | Diffuse transmission BSDF lobe, thin-surface flag, 5-way MIS |
| 8I | Nested dielectric priority | IOR priority stack for overlapping transmissive volumes |
| 8J | Emissive mesh light extraction | Auto-extract emissive triangles for NEE, compute shader |
| 8K | Weighted reservoir sampling for NEE | O(1) WRS light selection replaces O(N) per-light loop |
| 9A | Standalone denoiser library (`deni_vulkan`) | Standalone unit test: diffuse + specular summed, output matches input sum |
| 9B | Denoiser integration test | Denoiser wired into render loop, end-to-end passthrough verified |
| 10A | Tone map + present (end-to-end pipeline) | `monti_view`: complete render loop — trace → denoise → tonemap → present |
| 10B | Interactive camera + ImGui overlay | `monti_view`: WASD/mouse camera, settings panel, frame timing |
| 11A | Capture writer (`monti_capture`) | CPU-side EXR writer: write known data at two resolutions, reload and verify channels |
| 11B | GPU readback + headless datagen | `monti_datagen`: headless render at input resolution → GPU readback → high-SPP reference at target resolution → dual-file EXR output |

---

## Phase 1: Project Skeleton + Build System

**Goal:** Establish repository structure, CMake build, dependencies, shader compilation pipeline, and `.gitignore`.

**Source:** rtx-chessboard `CMakeLists.txt` for dependency fetching (`FetchContent`), compiler flags, and shader compilation patterns.

### Tasks

1. Create directory structure per §3 of the design doc and §8 of the app spec:
   ```
   denoise/include/deni/vulkan/
   denoise/src/vulkan/
   denoise/src/vulkan/shaders/
   scene/include/monti/scene/
   scene/src/
   scene/src/gltf/
   renderer/include/monti/vulkan/
   renderer/src/vulkan/
   renderer/src/vulkan/shaders/
   capture/include/monti/capture/
   capture/src/
   tests/
   tests/scenes/
   tests/assets/           # Downloaded glTF test models (gitignored)
   tests/references/        # Golden reference images (checked in)
   tests/references/vulkan/ # Per-platform golden references
   app/
   app/core/               # Shared by monti_view and monti_datagen
   app/view/               # monti_view only (windowed, interactive)
   app/datagen/            # monti_datagen only (headless, batch)
   app/shaders/
   app/assets/fonts/
   app/assets/env/
   ```

2. Create `.gitignore` for the full project lifecycle:
   - Build outputs: `build/`, `out/`, `cmake-build-*/`
   - IDE files: `.vs/`, `*.vcxproj.user`, `.idea/`
   - Compiled shaders: `*.spv`
   - Test assets: `tests/assets/`
   - Capture output: `capture/`
   - OS files: `Thumbs.db`, `.DS_Store`
   - Dependencies fetched by CMake: `_deps/`

3. Create root `CMakeLists.txt`:
   - C++20, strict warnings (`/W4 /WX` on MSVC)
   - `FetchContent` for: Vulkan SDK headers, volk, VMA, GLM, cgltf, tinyexr, stb, **MikkTSpace** (tangent generation for glTF meshes missing tangent attributes), **NVIDIA FLIP** (C++ library, built from source via FetchContent), **Catch2** (test framework), **CLI11** (argument parsing — used by test runner and later by apps)
   - Shader compilation: find `glslc`, custom command for `.rgen`/`.rchit`/`.rmiss`/`.comp` → `.spv` (actual shader sources and compilation rules added in Phase 7B when skeleton shaders are created)
   - Library targets: `deni_vulkan`, `monti_scene`, `monti_vulkan`, `monti_capture`
   - **Test target:** `monti_tests` (links Catch2 + FLIP + relevant libraries)
   - CMake option: `MONTI_BUILD_APPS=OFF` — app executables (`monti_view`, `monti_datagen`) are not built until libraries are functional (see [app_specification.md](app_specification.md))
   - CMake option: `MONTI_DOWNLOAD_TEST_ASSETS=ON` — fetches Khronos glTF sample models at configure time
   - CMake option: `MONTI_DOWNLOAD_BENCHMARK_SCENES=OFF` — opt-in download of heavy benchmark scenes
   - SDL3, ImGui, FreeType, nlohmann/json fetched only when `MONTI_BUILD_APPS=ON` (app dependencies)

4. Create **public API headers** with the types and class declarations from the design spec:
   - `denoise/include/deni/vulkan/Denoiser.h` — full Denoiser class, DenoiserDesc, DenoiserInput, DenoiserOutput, ScaleMode per §4.1 (no GLM dependency — Vulkan-native and scalar types only)
   - `scene/include/monti/scene/Types.h` — TypedId, Transform, Vertex, PixelFormat, SamplerWrap, SamplerFilter per §5.1
   - `scene/include/monti/scene/Material.h` — Mesh, MeshData, TextureDesc, MaterialDesc per §5.3
   - `scene/include/monti/scene/Light.h` — EnvironmentLight, AreaLight per §5.4
   - `scene/include/monti/scene/Camera.h` — CameraParams per §5.5
   - `scene/include/monti/scene/Scene.h` — Scene class, SceneNode per §5.2
   - `renderer/include/monti/vulkan/Renderer.h` — Renderer class, RendererDesc (including `get_device_proc_addr`), GBuffer per §6.3
   - `renderer/include/monti/vulkan/GpuBufferUtils.h` — GpuBuffer, upload helpers per §6.1.1
   - `capture/include/monti/capture/Writer.h` — Writer class, WriterDesc, InputFrame, TargetFrame per §8
   - **Internal headers** (GpuScene.h, GeometryManager.h, EnvironmentMap.h, BlueNoise.h, RtPipeline.h) are deferred to their respective implementation phases.

5. Create stub source files for each library (empty implementations, just enough for linking):
   - `denoise/src/vulkan/Denoiser.cpp` — stub `Create()`, `Denoise()`, `Resize()`
   - `scene/src/Scene.cpp` — stub scene methods
   - `scene/src/gltf/GltfLoader.cpp` — stub `LoadGltf()`
   - `renderer/src/vulkan/Renderer.cpp` — stub renderer
   - `capture/src/Writer.cpp` — stub writer

6. Create test entry point (`tests/main_test.cpp`):
   - Catch2 `TEST_CASE` that verifies the test harness works (e.g., `REQUIRE(true)`)
   - Include FLIP header to confirm linkage
   - Validates that the test target compiles, links Catch2, links FLIP, and links all library targets

### Verification
- `cmake -B build -S .` configures without errors
- `cmake --build build` compiles and links all targets (libraries + `monti_tests`)
- `monti_tests` runs and Catch2 reports all tests passed
- FLIP library links successfully into test target (verified by including FLIP header in test)
- Catch2 and CLI11 link successfully via FetchContent
- No compiler warnings (treated as errors via `/W4 /WX`)
- `.gitignore` correctly excludes build artifacts

### rtx-chessboard Reference
- [CMakeLists.txt](../../rtx-chessboard/CMakeLists.txt): FetchContent patterns, shader compilation commands, compiler flags
- [FetchDependencies.cmake](../../rtx-chessboard/cmake/FetchDependencies.cmake): dependency versions and options
- Shader compilation: `glslc --target-env=vulkan1.2 -I shaders/` pattern

---

## Phase 2: Scene Layer (`monti_scene`)

**Goal:** Implement the platform-agnostic scene data model with typed IDs, meshes, materials, textures, nodes, lights, and cameras.

**Source:** rtx-chessboard `scene/scene.h`, `scene/scene_types.h`, `scene/material.h`, `scene/mesh.h`, `scene/scene_node.h`

### Tasks

1. Implement `scene/include/monti/scene/Types.h`:
   - `TypedId<Tag>` template with `operator<=>`, `operator==`, explicit `bool`, `kInvalid`
   - `std::hash` specialization
   - `Transform` with `ToMatrix()` (glm TRS composition)
   - `Vertex` struct (position, normal, tangent, tex_coord_0, tex_coord_1)
   - `PixelFormat` enum class
   - `SamplerWrap` enum class (`kRepeat`, `kClampToEdge`, `kMirroredRepeat`)
   - `SamplerFilter` enum class (`kLinear`, `kNearest`)

2. Implement `scene/include/monti/scene/Material.h`:
   - `Mesh` struct (id, name, vertex_count, index_count, vertex_stride, bbox) — metadata only, no vertex/index data
   - `MeshData` struct (mesh_id, vertices, indices) — transient data returned by loaders for host-driven GPU upload
   - `TextureDesc` struct (id, name, dimensions, format, pixel data, sampler parameters: wrap_s, wrap_t, mag_filter, min_filter)
   - `MaterialDesc` struct (full PBR per §5.3; transmission/volume fields implemented, not deferred; emissive fields included but noted as deferred pending ReSTIR (desktop initially; mobile with HW RT later); sheen deferred — not in v1)

3. Implement `scene/include/monti/scene/Light.h`:
   - `EnvironmentLight` (HDR equirectangular map)
   - `AreaLight` (emissive quad: corner, edge_a, edge_b, radiance, two_sided) per §5.4
   - Point, spot, and directional lights are intentionally omitted — area lights and environment lights cover all practical physically-based lighting scenarios

4. Implement `scene/include/monti/scene/Camera.h`:
   - `CameraParams` struct

5. Implement `scene/include/monti/scene/Scene.h` and `scene/src/Scene.cpp`:
   - `SceneNode` struct (id, mesh_id, material_id, transform, prev_transform, visible, name)
   - `Scene` class: `AddMesh()`, `AddMaterial()`, `AddTexture()`, `AddNode()`, `RemoveNode()`, `RemoveMesh()`, accessors
   - `RemoveNode(NodeId)`: removes a scene node; does not affect the mesh or material it referenced
   - `RemoveMesh(MeshId)`: removes a mesh only when no nodes reference it; returns `bool` (`false` if references exist)
   - `SetEnvironmentLight()`, `GetEnvironmentLight()`
   - `AddAreaLight()`, `AreaLights()`
   - `SetActiveCamera()`, `GetActiveCamera()`
   - Auto-incrementing IDs

6. Implement programmatic Cornell box builder (`tests/scenes/CornellBox.h/.cpp`):
   - Build the classic Cornell box via the `monti::Scene` API using **unit scale** (room from 0 to 1 on all axes): 5 quads (white floor/ceiling/back wall, red left wall, green right wall), 2 boxes (short/tall), area light quad on ceiling
   - Materials: white diffuse (base_color={0.73, 0.73, 0.73}), red diffuse ({0.65, 0.05, 0.05}), green diffuse ({0.12, 0.45, 0.15}), light emissive (base_color white, emissive_factor set for reference)
   - An `AreaLight` quad positioned on the ceiling, emitting downward
   - Camera positioned at canonical Cornell box viewpoint (centered, looking into the box along -Z)
   - Returns a `CornellBoxResult` (defined in `CornellBox.h`) containing both a fully populated `monti::Scene` and a `std::vector<MeshData>` with the vertex/index data for all 7 meshes, following the same pattern as the glTF `LoadResult`

7. Write integration test (`tests/scene_integration_test.cpp`):
   - Build the Cornell box scene via the programmatic builder
   - Verify mesh count (7: 5 walls + 2 boxes), material count (4: white, red, green, light), node count (7)
   - Verify `CornellBoxResult::mesh_data` contains 7 entries with matching mesh IDs and non-empty vertex/index data
   - Verify all data round-trips through `Scene` accessors (mesh metadata, material properties, node transforms)
   - Verify ID lookup correctness: `GetMesh(id)`, `GetMaterial(id)`, `GetNode(id)` return the right data
   - **TypedId compile-time safety** is inherently enforced by the `TypedId<Tag>` template — distinct tag types are not implicitly convertible. No runtime test needed; this is documented and verified by the type system.
   - Verify camera is set correctly (position, target, up, FOV)
   - Verify area light is set correctly (corner, edges, radiance)
   - Verify `RemoveNode()` removes a node and decrements node count
   - Verify `RemoveMesh()` returns `false` when nodes still reference the mesh, returns `true` and removes when no references remain

### Verification
- Integration test passes: Cornell box scene builds correctly with expected counts, data round-trip, and MeshData contents
- TypedId compile-time safety is inherently enforced by the type system (documented, not tested at runtime)
- Scene handles empty state gracefully (no crashes on empty accessors)
- `RemoveNode()` and `RemoveMesh()` work correctly; `RemoveMesh()` returns `false` when references exist, `true` on success
- Area light stored and retrieved correctly

### rtx-chessboard Reference
- [scene_types.h](../../rtx-chessboard/src/scene/scene_types.h): `TypedId<Tag>` pattern
- [scene.h](../../rtx-chessboard/src/scene/scene.h): `Scene` class with flat vectors and `AddMesh()`/`AddMaterial()`/`AddNode()`
- [material.h](../../rtx-chessboard/src/scene/material.h): `Material` struct
- [mesh.h](../../rtx-chessboard/src/scene/mesh.h): `Mesh` struct with vertices/indices/bbox

---

## Phase 3: glTF Loader

**Goal:** Load `.glb`/`.gltf` files into the Scene, populating meshes, materials, textures, and nodes. Each glTF primitive becomes a separate `Mesh` + `SceneNode`. Node hierarchy is flattened into world transforms.

**Source:** rtx-chessboard `loaders/gltf_loader.cpp`

### Design Decisions

- **One primitive = one Mesh + SceneNode.** A single glTF mesh may contain multiple primitives with different materials. Each primitive is extracted as its own `Mesh` (with independent vertex/index data and bounding box) and gets its own `SceneNode` with the appropriate material reference. This simplifies the GPU scene — every scene node maps 1:1 to a BLAS instance with a single material.
- **Flattened hierarchy.** The glTF node tree is walked recursively, concatenating parent transforms. Each `SceneNode` stores the final **world transform** (`SceneNode::transform` is set from the composed glTF world matrix). No parent-child relationships are retained in `Scene`.
- **MikkTSpace tangent generation.** When TANGENT attributes are missing from a primitive, tangents are generated using the MikkTSpace algorithm (the glTF-recommended approach). This is critical for correct normal mapping.
- **Face-weighted normal generation.** When NORMAL attributes are missing, face-weighted normals are computed from triangle geometry.
- **Texture format mapping.** All color textures (base color, emissive) are decoded to `kRGBA8_UNORM` via stb_image (4-channel, 8-bit). Normal maps are decoded to `kRGBA8_UNORM` (stb_image doesn't natively decode to SNORM; GPU-side conversion is handled in shaders). Metallic-roughness textures are decoded to `kRGBA8_UNORM`. Single-channel textures (transmission) are decoded to `kR8_UNORM`. KTX2/Basis compressed textures are out of scope.
- **Sampler extraction.** Each glTF texture references a sampler with wrap and filter modes. These are mapped to `TextureDesc::wrap_s`, `wrap_t`, `mag_filter`, `min_filter`. Defaults follow glTF 2.0: `kRepeat` wrap, `kLinear` filter.
- **Occlusion map deferred.** The glTF occlusion texture (R channel of the metallic-roughness image in ORM packing) is not extracted. The rtx-chessboard renderer does not use occlusion, and none of the initial test scenes require it. Can be added later by reading `occlusionTexture` and adding an `occlusion_map` field to `MaterialDesc`.
- **Camera extraction skipped.** glTF cameras are ignored. The host always sets the camera via `scene.SetActiveCamera()`.
- **Skin, animation, morph targets silently ignored.** These features are out of scope for a static-scene path tracer. The loader does not crash on assets containing them — it simply skips the data.
- **Explicit failure on missing required data.** If a glTF file is valid but a required texture file is missing (e.g., external `.png` not found), the loader fails with `success = false` and a descriptive `error_message` rather than substituting fallback data. Partial-load recovery can be added later as test coverage expands.

### Tasks

1. Create `scene/src/gltf/GltfLoader.h` (public header for the glTF loader — types are currently defined in the `.cpp` stub; move them to a proper header):
   - `LoadResult` struct: `success`, `error_message`, `nodes`, `mesh_data`
   - `LoadOptions` struct: `generate_missing_normals`, `generate_missing_tangents`
   - `LoadGltf()` function declaration

2. Implement `scene/src/gltf/GltfLoader.cpp`:
   - **Parse glTF** via cgltf (`cgltf_parse_file` + `cgltf_load_buffers` + `cgltf_validate`)
   - **Extract textures** first (textures are referenced by materials):
     - Iterate `data->images`, decode each via stb_image (`stbi_load_from_memory` using cgltf buffer view data)
     - Map format: color textures → `kRGBA8_UNORM` (request 4 channels from stb), single-channel → `kR8_UNORM`
     - Build a cgltf image pointer → `TextureId` lookup map
     - Extract sampler wrap/filter modes from `data->samplers`; map cgltf sampler constants to `SamplerWrap`/`SamplerFilter` enums
     - `scene.AddTexture()` for each decoded image
   - **Extract materials** (materials reference textures by index):
     - Iterate `data->materials`, populate `MaterialDesc` fields:
       - PBR metallic-roughness: `base_color`, `roughness`, `metallic` from `pbr_metallic_roughness`
       - Texture maps: `base_color_map`, `normal_map`, `metallic_roughness_map`, `emissive_map`, `transmission_map` — use the image→TextureId map to resolve
       - `normal_scale` from `normal_texture.scale`
       - Alpha: `alpha_mode`, `alpha_cutoff`, `double_sided`
       - Clear coat: from `KHR_materials_clearcoat` extension
       - Transmission/volume: from `KHR_materials_transmission` and `KHR_materials_volume` extensions
       - Emissive: `emissive_factor`, `emissive_map`, `emissive_strength` (from `KHR_materials_emissive_strength`)
       - `opacity` from `base_color.a` (for blended materials) or 1.0 for opaque
       - `ior` from `KHR_materials_ior` extension (default 1.5)
     - `scene.AddMaterial()` for each material
     - Build a cgltf material pointer → `MaterialId` lookup map
   - **Extract meshes and nodes** (walk glTF node hierarchy):
     - Recursively walk `data->scenes[0]` node tree, accumulating the parent→child transform chain
     - For each node with a mesh: iterate its primitives. For each primitive:
       - Read POSITION, NORMAL, TANGENT, TEXCOORD_0, TEXCOORD_1 attributes via cgltf accessor helpers
       - If NORMAL missing and `options.generate_missing_normals`: compute face-weighted normals from triangle geometry
       - If TANGENT missing and `options.generate_missing_tangents`: generate via MikkTSpace (requires positions, normals, UVs)
       - If TEXCOORD_0 missing: fill with `(0, 0)`
       - If TEXCOORD_1 missing: fill with `(0, 0)`
       - Read indices (cgltf handles uint8/uint16/uint32 conversion)
       - Populate `Vertex` array and `uint32_t` index array → `MeshData`
       - Compute bounding box from positions
       - `scene.AddMesh()` with metadata (vertex_count, index_count, bbox)
       - `scene.AddNode()` with the primitive's material and the composed **world transform** (decomposed back to TRS via `glm::decompose` for `Transform` struct)
       - Append `MeshData` to `LoadResult::mesh_data`
       - Append `NodeId` to `LoadResult::nodes`
   - **Handle default material**: if a primitive has no material assigned, create a default white PBR material (base_color white, roughness 0.5, metallic 0.0) and reuse it for all unassigned primitives

3. Write integration tests (`tests/gltf_loader_test.cpp`):
   - **Box.glb test:** Load `Box.glb` (simplest Khronos sample):
     - Verify 1 mesh, 1 material, 1 node
     - Verify expected vertex count (24) and index count (36) for a box
     - Verify `LoadResult::mesh_data` has 1 entry with matching mesh ID and non-empty vertex/index data
     - Verify bounding box is non-degenerate
   - **DamagedHelmet.glb test:** Load `DamagedHelmet.glb`:
     - Verify mesh count, material count, texture count
     - Verify material has PBR textures assigned: `base_color_map`, `normal_map`, `metallic_roughness_map`, `emissive_map` are set
     - Verify material properties: metallic = 1.0, roughness = 1.0 (DamagedHelmet uses texture-driven values with factor defaults of 1.0)
   - **Multi-primitive test:** Load `MorphPrimitivesTest.glb` (1 glTF mesh with 2 primitives using different materials — red and green). Verify that loading produces 2 meshes, 2 materials, and 2 nodes. Verify that each primitive's material assignment is correct (distinct `MaterialId` per node). Morph target data in this asset is silently ignored.
   - **Error test:** Verify `LoadGltf()` returns `success = false` with a descriptive `error_message` for a non-existent file path
   - **Sampler test:** Verify that texture sampler wrap/filter modes are correctly extracted (check at least one texture with non-default sampler settings, or verify defaults match glTF 2.0 spec)

### Verification
- `LoadGltf()` returns `success = true` for valid glTF files
- Scene contains expected meshes, materials, textures per the glTF content
- Each glTF primitive produces a separate `Mesh` + `SceneNode`
- World transforms are correctly computed (hierarchy is flattened)
- Missing normals are generated from face geometry
- Missing tangents are generated via MikkTSpace
- Texture sampler modes are correctly mapped
- `LoadGltf()` returns `success = false` with error message for invalid/missing files
- Skin, animation, and morph target data do not cause errors (silently ignored)

### rtx-chessboard Reference
- [gltf_loader.cpp](../../rtx-chessboard/src/loaders/gltf_loader.cpp): cgltf parsing, vertex attribute extraction, multi-primitive handling, world transform baking
- [chess_scene_builder.cpp](../../rtx-chessboard/src/scene/chess_scene_builder.cpp): how the loader is called and nodes are set up

---

## Phase 4: Vulkan Context + App Scaffolding

**Goal:** Create the shared Vulkan context (supporting both windowed and headless modes), the `monti_view` windowed frame loop, and the `app/` directory structure per [app_specification.md](app_specification.md) §8. At the end of this phase, `monti_view` opens a window presenting a cleared color, and an automated headless context test validates device creation and command submission without a window.

**Source:** rtx-chessboard `core/vulkan_context.h`, `core/swapchain.h`, `core/command_pool.h`, `core/sync_objects.h`, `main.cpp`

### Design Decisions

- **Two-app architecture.** The `app/` directory follows the `core/` + `view/` + `datagen/` split from [app_specification.md](app_specification.md) §8. Shared Vulkan initialization, frame resources, and G-buffer allocation live in `app/core/`. The `monti_view` windowed entry point and swapchain live in `app/view/`. The `monti_datagen` headless entry point lives in `app/datagen/` (stubbed in this phase, implemented in Phase 11B).
- **Minimum Vulkan 1.3.** Requiring Vulkan 1.3 promotes `VK_KHR_synchronization2`, `VK_KHR_buffer_device_address`, `VK_EXT_descriptor_indexing`, and `VK_KHR_dynamic_rendering` to core, simplifying extension management. The only extensions that remain explicit are `VK_KHR_ray_tracing_pipeline`, `VK_KHR_acceleration_structure`, `VK_KHR_deferred_host_operations`, and `VK_KHR_swapchain` (for `monti_view`).
- **Volk for function loading.** The app uses volk (`volkInitialize()` → create instance → `volkLoadInstance()` → create device → `volkLoadDevice()`) so no linking to `vulkan-1.lib` is needed. The host passes `vkGetDeviceProcAddr` (from volk) via `RendererDesc::get_device_proc_addr` and `DenoiserDesc::get_device_proc_addr` so the libraries remain loader-agnostic.
- **Validation layers in debug builds.** `VK_EXT_debug_utils` and `VK_LAYER_KHRONOS_validation` are enabled when `CMAKE_BUILD_TYPE` is `Debug` (or MSVC `_DEBUG`). The debug messenger prints to stderr. No additional debugging scaffolding is added until needed.
- **VulkanContext accepts a `VkSurfaceKHR`, not `SDL_Window*`.** Since `vulkan_context.cpp` lives in `CORE_SOURCES` (compiled into both executables) and `monti_datagen` does not link SDL3, the VulkanContext must not depend on SDL3 headers. Instance creation accepts a list of required instance extensions (so `monti_view` can pass SDL's required extensions). After instance creation, the caller creates a `VkSurfaceKHR` externally (`app/view/main.cpp` calls `SDL_Vulkan_CreateSurface()`), then passes it to VulkanContext for device creation. When no surface is provided (headless), swapchain-related setup is skipped and queue family selection requires only graphics capability. This keeps `app/core/` free of SDL3 dependencies.

### Tasks

1. Update `CMakeLists.txt` for app targets (behind `MONTI_BUILD_APPS=ON`):
   - Add `monti_view` executable: `CORE_SOURCES` + `VIEW_SOURCES` per [app_specification.md](app_specification.md) §9
   - Add `monti_datagen` executable: `CORE_SOURCES` + `DATAGEN_SOURCES` (stub `datagen/main.cpp` only — full implementation in Phase 11B)
   - `monti_view` links `SDL3::SDL3-static`, `freetype`; `monti_datagen` links `monti_capture`
   - Fetch SDL3, ImGui, FreeType, nlohmann/json when `MONTI_BUILD_APPS=ON` (already specified in Phase 1)
   - Define `VK_NO_PROTOTYPES`, `GLM_FORCE_DEPTH_ZERO_TO_ONE`, `_CRT_SECURE_NO_WARNINGS` for both targets

2. Implement `app/core/vulkan_context.h` and `vulkan_context.cpp`:
   - Two-step initialization: (1) `CreateInstance(extra_instance_extensions)` → creates instance, loads volk, sets up debug messenger; (2) `CreateDevice(optional_surface)` → selects physical device, creates logical device and VMA allocator
   - `volkInitialize()` → Vulkan 1.3 instance creation (with caller-provided extra instance extensions) → `volkLoadInstance()`
   - Validation layers + `VK_EXT_debug_utils` debug messenger (debug builds only)
   - Physical device selection: require `VK_KHR_ray_tracing_pipeline` + `VK_KHR_acceleration_structure` support
   - Logical device creation via `volkLoadDevice()` with:
     - Vulkan 1.3 core features: `synchronization2`, `bufferDeviceAddress`, `descriptorIndexing`, `dynamicRendering`, `maintenance4`
     - Additional Vulkan 1.3 features: `shaderStorageImageReadWithoutFormat`, `shaderStorageImageWriteWithoutFormat` (required for format-agnostic G-buffer storage images)
     - Extensions: `VK_KHR_ray_tracing_pipeline`, `VK_KHR_acceleration_structure`, `VK_KHR_deferred_host_operations`, and `VK_KHR_swapchain` (only when surface is present)
     - Ray tracing pipeline features + acceleration structure features
   - Queue family selection: graphics queue (+ present capability when surface is present)
   - VMA allocator creation (using volk function pointers)
   - Public accessors: `Device()`, `PhysicalDevice()`, `GraphicsQueue()`, `QueueFamilyIndex()`, `Allocator()`, `Instance()`, `RtPipelineProperties()`, `AccelStructProperties()`
   - `BeginOneShot()` / `SubmitAndWait()` convenience for one-shot command submission
   - `WaitIdle()` for clean teardown

3. Implement `app/core/frame_resources.h` and `frame_resources.cpp`:
   - `FrameResources` class: per-frame command pool + command buffer + fence + semaphores
   - Triple-buffered: `kFramesInFlight = 3`
   - `WaitForFence()`, `ResetCommandBuffer()`, `SubmitAndPresent()` (present is a no-op in headless mode)
   - Image-available semaphore (per frame) and render-finished semaphore (per swapchain image)

4. Implement `app/view/swapchain.h` and `swapchain.cpp`:
   - `Swapchain` class: create/recreate swapchain
   - `VK_FORMAT_B8G8R8A8_SRGB` preferred, `VK_PRESENT_MODE_MAILBOX_KHR` preferred
   - Acquire image / present flow
   - Handles `VK_ERROR_OUT_OF_DATE_KHR` → triggers recreate

5. Implement `app/view/main.cpp`:
   - SDL3 initialization (`SDL_Init(SDL_INIT_VIDEO)`)
   - SDL3 window creation (1280×720 default, resizable, `SDL_WINDOW_VULKAN`)
   - VulkanContext two-step init: `CreateInstance(SDL_Vulkan_GetInstanceExtensions())` → `SDL_Vulkan_CreateSurface()` → `CreateDevice(surface)`
   - Swapchain creation
   - FrameResources creation
   - Frame loop: acquire → wait fence → reset cmd → begin cmd → clear color → end cmd → submit + present
   - Handle `SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED` via `SDL_AddEventWatch` for resize-during-drag (recreate swapchain immediately, not deferred to next frame)
   - Handle `VK_ERROR_OUT_OF_DATE_KHR` from present → recreate swapchain
   - Clean shutdown: `WaitIdle()`, destroy in reverse order

6. Create `app/datagen/main.cpp` (stub):
   - Minimal entry point: `CreateInstance({})` → `CreateDevice(std::nullopt)` → prints device name → exits
   - Full implementation deferred to Phase 11B

### Verification
- **`monti_view` (manual):** Application opens a window showing a solid color (cornflower blue or similar). Window resizes correctly during drag without crashes or validation errors. Clean shutdown with zero validation errors.
- **Headless context test (automated, `tests/vulkan_context_test.cpp`):** Create a headless VulkanContext (no window). Verify device is not null. Verify ray tracing pipeline and acceleration structure properties are populated. Record a trivial command buffer (pipeline barrier or no-op), submit, and wait. Destroy cleanly with zero validation errors. This test can run in CI without a display server.
- **`monti_datagen` stub (manual/CI):** Runs, prints device name, and exits with code 0 without validation errors.

### rtx-chessboard Reference
- [vulkan_context.h/.cpp](../../rtx-chessboard/src/core/vulkan_context.h): instance/device creation, VMA, queue selection
- [swapchain.h/.cpp](../../rtx-chessboard/src/core/swapchain.h): swapchain creation/recreation
- [command_pool.h/.cpp](../../rtx-chessboard/src/core/command_pool.h): per-frame command buffers
- [sync_objects.h/.cpp](../../rtx-chessboard/src/core/sync_objects.h): fences + semaphores
- [main.cpp](../../rtx-chessboard/src/main.cpp): frame loop structure, SDL window creation, resize-during-drag via `SDL_AddEventWatch`

---

## Phase 5: GPU Scene (`monti::vulkan::GpuScene` — Internal)

**Goal:** Create the internal `GpuScene` that registers host-provided geometry buffers, packs materials, and uploads textures for ray tracing. `GpuScene` is internal to the renderer — the host interacts through `Renderer::RegisterMeshBuffers()`.

**Source:** rtx-chessboard `render/gpu_scene.h/.cpp`, `core/buffer.h/.cpp`, `core/image.h/.cpp`, `core/upload.h/.cpp`

### Design Decisions

- **GpuScene is internal.** `GpuScene` lives in `renderer/src/vulkan/` (not in public `include/`). The host registers geometry via `Renderer::RegisterMeshBuffers()`, which delegates to the internal `GpuScene`. Material and texture uploads are triggered internally by the renderer (on first `RenderFrame()` after `SetScene()`, and when the scene changes), not by direct host calls.
- **Constructor injection for allocator/device.** `GpuScene` receives `VmaAllocator` and `VkDevice` in its constructor. These are stored and reused for all internal allocations. This is simpler than passing them on every method call and sufficient for the expected use case (single allocator per renderer).
- **Host-visible material buffer.** The material storage buffer uses `VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT` + `VMA_MEMORY_USAGE_AUTO`, allowing direct `memcpy` without staging. Material arrays are small (hundreds of materials × 80 bytes ≈ 16 KB) and updated infrequently, so the negligible PCIe overhead on discrete GPUs is acceptable. This avoids staging buffer management and command buffer submission for material updates.
- **Device-local textures with staging.** Texture images are device-local for optimal GPU read bandwidth. Upload uses a staging buffer + `vkCmdCopyBufferToImage`. The staging buffer is freed after the copy commands are recorded; the caller must ensure the command buffer completes (fence signal) before the staging memory is reused.
- **All texture indices packed now.** `PackedMaterial` includes slots for all 5 texture map indices (`base_color_map`, `normal_map`, `metallic_roughness_map`, `emissive_map`, `transmission_map`). Absent textures use `UINT32_MAX` as a sentinel value (encoded as `std::bit_cast<float>(UINT32_MAX)`; shader checks `floatBitsToUint(index) == 0xFFFFFFFFu`).
- **Per-texture VkSampler.** Each uploaded texture gets its own `VkSampler` created from the `TextureDesc` sampler parameters (`wrap_s`, `wrap_t`, `mag_filter`, `min_filter`). This is standard practice for `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` bindless arrays and avoids separate sampler management.
- **Mip generation optional per texture.** Controlled by `TextureDesc::mip_levels`: when `mip_levels > 1`, mip chain is generated via `vkCmdBlitImage` during upload. When `mip_levels == 1`, only the base level is uploaded.
- **Full texture upload initially.** All textures are uploaded when `UploadTextures()` is first called. Individual texture updates are deferred to a future phase for streaming/LOD support.

### Tasks

1. Implement internal RAII buffer and image helpers in `renderer/src/vulkan/`:
   - `Buffer` class (`Buffer.h/.cpp`): VMA-allocated, supports host-visible (mapped) and device-local modes
   - `Image` class (`Image.h/.cpp`): VMA-allocated, supports mip generation via `vkCmdBlitImage`, creates `VkImageView` and per-texture `VkSampler`
   - `Upload` class (`Upload.h/.cpp`): staging buffer management for CPU→GPU transfers (used by texture upload)
   - These are **internal** to the renderer — not exposed in public headers

2. Implement `renderer/src/vulkan/GpuScene.h` and `GpuScene.cpp`:
   - **Constructor:** `GpuScene(VmaAllocator allocator, VkDevice device)`
   - `RegisterMeshBuffers(mesh_id, binding)`:
     - Store `MeshBufferBinding` (VkBuffer, VkDeviceAddress for vertex+index, counts, stride)
     - Host is responsible for uploading vertex/index data to GPU buffers before calling this
     - Device addresses used later for BLAS building and shader `buffer_reference` access
   - `UpdateMaterials(scene)`:
     - Pack `MaterialDesc` → `PackedMaterial` array → host-visible storage buffer
     - All 5 texture indices encoded: `base_color_map`, `normal_map`, `metallic_roughness_map`, `emissive_map`, `transmission_map`
     - Absent (unset `std::optional`) textures encoded as `std::bit_cast<float>(UINT32_MAX)`
     - Allocate buffer on first call; reallocate if material count grows
     - Map buffer → `memcpy` packed array → no staging needed
   - `UploadTextures(scene, cmd)`:
     - Upload `TextureDesc::data` to device-local `VkImage` per texture via staging buffer + `vkCmdCopyBufferToImage`
     - Explicit layout transitions:
       1. `VK_IMAGE_LAYOUT_UNDEFINED` → `VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL` (pipeline barrier before copy)
       2. Copy staging → image base level
       3. If `TextureDesc::mip_levels > 1`: generate mip chain via `vkCmdBlitImage` (transition each level `TRANSFER_DST` → `TRANSFER_SRC` for source, next level stays `TRANSFER_DST`)
       4. Final transition → `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` (all levels)
     - Create `VkImageView` per texture
     - Create `VkSampler` per texture from `TextureDesc` sampler parameters (`wrap_s`/`wrap_t` → `VkSamplerAddressMode`, `mag_filter`/`min_filter` → `VkFilter`)
     - Populate `texture_images_` vector and `image_id_to_index_` map for bindless descriptor binding
   - ID → index lookup maps for meshes, materials, and textures
   - `PackedMaterial` layout (5 × `vec4`, 80 bytes, `alignas(16)`):
     ```
     base_color_roughness:   .rgb = base_color, .a = roughness
     metallic_clearcoat:     .r = metallic, .g = clear_coat,
                             .b = clear_coat_roughness,
                             .a = base_color_map index
     opacity_ior:            .r = opacity, .g = ior,
                             .b = normal_map index,
                             .a = metallic_roughness_map index
     transmission_volume:    .r = transmission_factor, .g = thickness,
                             .b = attenuation_distance,
                             .a = transmission_map index
     attenuation_color_pad:  .rgb = attenuation_color,
                             .a = emissive_map index
     ```
     All texture indices are `float`-encoded `uint32_t` via `std::bit_cast<float>()`. `UINT32_MAX` = no texture. Additional material properties (emissive factor, alpha mode, normal scale) will expand `PackedMaterial` when the corresponding shader features are implemented.
   - `UpdateAreaLights(scene)`:
     - Pack `AreaLight` → `PackedAreaLight` array (4 × `vec4`, 64 bytes per light, `alignas(16)`) → host-visible storage buffer
     - `PackedAreaLight` layout: `.corner_two_sided` (.xyz = corner, .w = two_sided as 1.0/0.0), `.edge_a` (.xyz), `.edge_b` (.xyz), `.radiance` (.xyz)
     - Allocate buffer on first call (minimum 1 element placeholder); reallocate if light count grows
     - Map buffer → `memcpy` packed array → no staging needed (same pattern as material buffer)
     - Returns `uint32_t` area light count for push constant `area_light_count`
     - When the scene has no area lights, the 1-element placeholder buffer stays bound — `area_light_count = 0` prevents shader iteration

3. Implement GPU buffer upload helpers in `renderer/src/vulkan/GpuBufferUtils.cpp` (implementation for the existing public header `renderer/include/monti/vulkan/GpuBufferUtils.h`):
   - `UploadMeshToGpu(allocator, device, cmd, mesh_data)` → allocates VMA staging buffer, records `vkCmdCopyBuffer` into `cmd`, returns `{vertex_buffer, index_buffer}` pair
   - `CreateVertexBuffer()`, `CreateIndexBuffer()` → individual buffer creation with staging copy
   - `DestroyGpuBuffer()` → frees VMA allocation and buffer
   - Buffer usage flags: `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`
   - **Staging buffer lifespan:** staging allocations are freed internally after copy commands are recorded into `cmd`. The returned `GpuBuffer` is owned by the caller and must be kept alive for the renderer's lifetime (until `DestroyGpuBuffer()` is called after the mesh is removed). The device-local buffer is usable once the command buffer completes (fence signal). This is the standard Vulkan staging pattern and is documented in the header.
   - These are optional convenience helpers in the `monti_vulkan` library — hosts with their own buffer management skip these and call `Renderer::RegisterMeshBuffers()` directly
   - `UploadAndRegisterMeshes(renderer, allocator, device, cmd, mesh_data)` → convenience wrapper that calls `UploadMeshToGpu` + `Renderer::RegisterMeshBuffers` per mesh, returns `vector<GpuBuffer>` the host must keep alive

4. Add `Renderer::RegisterMeshBuffers()` to public API:
   - Move `MeshBufferBinding` struct to `renderer/include/monti/vulkan/Renderer.h` (public type)
   - Add `void Renderer::RegisterMeshBuffers(MeshId mesh, const MeshBufferBinding& binding)` — internally delegates to `GpuScene::RegisterMeshBuffers()`
   - This keeps `GpuScene` fully internal while allowing the host to register geometry buffers through the `Renderer` interface

### Verification (`tests/gpu_scene_test.cpp`)
- Integration test: create headless VulkanContext, build a simple scene (Cornell box or programmatic), upload mesh data via `UploadMeshToGpu`, register via `Renderer::RegisterMeshBuffers()`, verify bindings stored with correct device addresses
- Material buffer: call internal `UpdateMaterials` (via Renderer internals in the test), read back host-visible buffer, compare packed values against original `MaterialDesc` fields — including all 5 texture indices (present and absent, verifying `UINT32_MAX` sentinel for missing textures) and transmission/volume fields
- Texture images: verify images created with correct dimensions, formats, and mip levels
- Sampler creation: verify `VkSampler` created per texture with correct wrap/filter modes
- Area light buffer: add 2 area lights to scene, call `UpdateAreaLights()`, read back buffer and verify `PackedAreaLight` values match source `AreaLight` data. Verify `area_light_count` returned correctly. Verify empty scene binds placeholder buffer with count 0.
- No VMA allocation failures or validation errors
- Host helper: `UploadMeshToGpu` uploads glTF `MeshData` and registers through `Renderer` successfully

### rtx-chessboard Reference
- [gpu_scene.h/.cpp](../../rtx-chessboard/src/render/gpu_scene.h): merged buffer approach, `PackedMaterial`, texture upload
- [buffer.h/.cpp](../../rtx-chessboard/src/core/buffer.h): VMA buffer wrapper
- [image.h/.cpp](../../rtx-chessboard/src/core/image.h): VMA image wrapper, mip generation
- [upload.h/.cpp](../../rtx-chessboard/src/core/upload.h): staging buffer management

---

## Phase 6: Acceleration Structures (`GeometryManager` — Internal)

**Goal:** Build BLAS per mesh and a single TLAS for all scene nodes. `GeometryManager` is internal to the renderer (§6.2 of the design doc) — it is not exposed in public headers. `RenderFrame()` calls it automatically. Tests in this phase exercise the class through its internal interface.

**Source:** rtx-chessboard `render/hw/hw_path_tracer.cpp` (BLAS/TLAS building functions)

### Architecture: Separate Per-Mesh Buffers with Buffer Address Table

Unlike rtx-chessboard (which merges all mesh vertex/index data into a single buffer pair with a `MeshGPURange` offset table), monti keeps **per-mesh separate buffers** owned by the host. This means shaders cannot index into a single merged buffer by offset. Instead, shaders use GLSL `buffer_reference` to access per-mesh vertex/index data via device addresses looked up from a **buffer address table**.

**Buffer address table (storage buffer):** An array of `MeshAddressEntry` uploaded to a storage buffer by `GpuScene`, one entry per registered mesh:

```glsl
// GPU-side layout (std430)
struct MeshAddressEntry {
    uint64_t vertex_address;  // VkDeviceAddress of the vertex buffer
    uint64_t index_address;   // VkDeviceAddress of the index buffer
    uint     vertex_count;
    uint     index_count;
};
```

The C++ mirror struct lives in `GpuScene.h`. `GpuScene` maintains a `std::vector<MeshAddressEntry>` and a `mesh_id_to_address_index_` map, populated each time `RegisterMeshBuffers()` is called. The storage buffer is uploaded (or re-uploaded) lazily when new meshes are registered — either in `RenderFrame()` or via an explicit `UploadMeshAddressTable(cmd)` method.

**Instance custom index encoding (24 bits):**

| Bits 0–11 (lower 12) | Bits 12–23 (upper 12) |
|---|---|
| `mesh_address_index` — index into the buffer address table | `material_index` — index into the packed material buffer |

The shader extracts these from `gl_InstanceCustomIndexEXT`:
```glsl
uint mesh_addr_idx = gl_InstanceCustomIndexEXT & 0xFFFu;
uint material_idx  = gl_InstanceCustomIndexEXT >> 12u;
MeshAddressEntry entry = mesh_address_table.entries[mesh_addr_idx];
// Use buffer_reference with entry.vertex_address / entry.index_address
```

This supports up to 4096 unique meshes and 4096 unique materials per scene.

**Each `SceneNode` maps to exactly one mesh primitive and one material.** Multi-primitive glTF meshes are split into one `SceneNode` per primitive during loading (Phase 3's "one-primitive-per-SceneNode extraction"). There is no concept of multi-material meshes at the renderer level.

### Prerequisite: Scene Generation Counter for TLAS Dirty Tracking

`Scene` currently has no mechanism to signal that transforms or structure have changed. Add a `uint64_t tlas_generation_` counter (initially 0) to `Scene`, incremented by:
- `AddNode()`, `RemoveNode()` — structural changes
- `SetNodeTransform()` — transform changes
- Any visibility toggle (if added later)

Expose via `uint64_t TlasGeneration() const`. `GeometryManager` caches the last-seen generation and skips the TLAS rebuild entirely when it matches. This avoids rebuilding the TLAS every frame for static scenes while keeping the implementation trivial.

### Tasks

1. **Add `tlas_generation_` to `Scene`** (`scene/include/monti/scene/Scene.h`, `scene/src/Scene.cpp`):
   - Add `uint64_t tlas_generation_ = 0;` private member
   - Increment in `AddNode()`, `RemoveNode()`, `SetNodeTransform()`
   - Add `uint64_t TlasGeneration() const;` public accessor

2. **Add buffer address table to `GpuScene`** (`renderer/src/vulkan/GpuScene.h`, `GpuScene.cpp`):
   - Add `MeshAddressEntry` C++ struct (matches GPU layout, `alignas(16)`)
   - Maintain `std::vector<MeshAddressEntry> mesh_address_entries_` and `std::unordered_map<MeshId, uint32_t> mesh_id_to_address_index_`
   - Populate on each `RegisterMeshBuffers()` call
   - Add `UploadMeshAddressTable()` — creates/re-creates host-visible storage buffer from entries via `memcpy` through persistent mapping (no command buffer needed — the buffer is host-visible)
   - Add `uint32_t GetMeshAddressIndex(MeshId id) const` accessor
   - Add `VkBuffer MeshAddressBuffer() const` and `VkDeviceSize MeshAddressBufferSize() const` accessors

3. **Implement `renderer/src/vulkan/GeometryManager.h` and `GeometryManager.cpp`:**

   - **Constructor:** Takes `VmaAllocator`, `VkDevice`. Creates a `VkQueryPool` for `VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR` (owned by `GeometryManager`, created once, reused for all compaction queries). Initial pool size: 64 queries; grow if needed.

   - **`BuildDirtyBlas(cmd, gpu_scene)`:**
     - One `VkAccelerationStructureKHR` per registered mesh
     - Only builds BLAS for meshes not yet in the internal BLAS map (new meshes) or flagged for rebuild
     - Refits BLAS for meshes flagged as deformed with `topology_changed = false`
     - Triangle geometry referencing host-provided vertex/index buffers via device addresses from `MeshBufferBinding`
     - Vertex format: `VK_FORMAT_R32G32B32_SFLOAT` (position only)
     - Stride: from `MeshBufferBinding::vertex_stride`
     - Build flags: `VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR`
     - **Deferred compaction (across frames):** `vkGetQueryPoolResults` requires the command buffer to have been submitted and completed — a pipeline barrier alone is insufficient for host readback. To avoid a mid-frame submission stall, compaction is deferred:
       - **Frame N:** Build uncompacted BLAS (fully functional for rendering), write compaction size query
       - **Frame N+1:** After frame N's fence has signaled, read query results via `vkGetQueryPoolResults`, allocate compacted BLAS, record `vkCmdCopyAccelerationStructureKHR` (uncompacted → compacted), update device address cache, queue uncompacted BLAS for destruction
       - `GeometryManager` tracks BLAS entries in a `kPendingCompaction` state between build and compaction. Rendering uses the uncompacted BLAS during this window — it is fully valid, just uses more memory temporarily.
       - After compaction, BLAS device addresses change. Set an internal `tlas_force_rebuild_` flag so `BuildTlas` rebuilds the TLAS with the new addresses on the same frame, even if `TlasGeneration()` hasn't advanced.
     - Cache device addresses per BLAS (updated when compacted BLAS replaces uncompacted)
     - **Scratch buffer:** Single allocation sized for the largest individual BLAS build. Reused across sequential builds (each BLAS build is separated by a pipeline barrier). Also reused for TLAS build if large enough; grow if TLAS scratch exceeds BLAS scratch.

   - **`BuildTlas(cmd, scene, gpu_scene)`:**
     - Checks `scene.TlasGeneration()` against cached value **and** `tlas_force_rebuild_` flag; skips entirely if generation unchanged and flag is clear
     - One `VkAccelerationStructureInstanceKHR` per visible `SceneNode`
     - Transform conversion: free function `ToVkTransformMatrix(const glm::mat4&)` in `GeometryManager.cpp` — transposes column-major `glm::mat4` to row-major 3×4 `VkTransformMatrixKHR` (strips the last row)
     - Instance custom index: `EncodeCustomIndex(gpu_scene.GetMeshAddressIndex(node.mesh_id), gpu_scene.GetMaterialIndex(node.material_id))`
     - Instance mask `0xFF`
     - Instance shader binding table offset: `0` (single hit group)
     - Instance flags: `VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR` (let shader handle culling)
     - TLAS build flags: `VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR`
     - Updates cached generation on success

   - **TLAS instance buffer:** Host-visible device-local via VMA (`VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT`). Persistently mapped. Re-allocated if instance count grows. Writes instance data directly via `memcpy` each frame the TLAS is rebuilt — no staging buffer needed.

   - **`NotifyMeshDeformed(MeshId, bool topology_changed)`:**
     - Mark BLAS as needing refit (not full rebuild) when `topology_changed = false`
     - Mark BLAS as needing full rebuild when `topology_changed = true`

   - **`CleanupRemovedMeshes(const Scene& scene)`:**
     - Compare internal BLAS map keys against `scene.Meshes()`
     - Destroy BLAS for any mesh no longer in the scene
     - Called by `RenderFrame()` before building

   - **`Tlas() const`:** Returns the current `VkAccelerationStructureKHR` for descriptor binding.

4. **Wire `GeometryManager` into `Renderer::Impl`** (`renderer/src/vulkan/Renderer.cpp`):
   - Add `std::unique_ptr<GeometryManager> geometry_manager_` member, created in `Init()`
   - In `RenderFrame()`, after material/texture upload:
     1. `geometry_manager_->CleanupRemovedMeshes(*scene)`
     2. `geometry_manager_->BuildDirtyBlas(cmd, *gpu_scene)`
     3. `geometry_manager_->BuildTlas(cmd, *scene, *gpu_scene)`
   - Forward `NotifyMeshDeformed()` to `geometry_manager_->NotifyMeshDeformed()`
   - Invoke mesh cleanup callback when BLAS is destroyed for a removed mesh

### Verification

Test file: `tests/geometry_manager_test.cpp` (new file — no significant code sharing with `gpu_scene_test.cpp`). Uses the same `TestContext` pattern from existing GPU tests (headless `VulkanContext`).

- BLAS build completes without validation errors
- BLAS compaction reduces memory (log before/after sizes) — requires two `RenderFrame()` calls: first builds uncompacted, second compacts after fence signals
- TLAS build completes with correct instance count matching visible `SceneNode` count
- Device addresses are non-zero for all BLAS entries and are updated after compaction
- Buffer address table contains correct device addresses for all registered meshes
- Modifying a scene node transform increments `TlasGeneration()` and calling `RenderFrame()` rebuilds the TLAS
- Calling `RenderFrame()` again without changes does **not** rebuild the TLAS (generation unchanged)
- Removing a node, then removing the unreferenced mesh via `Scene::RemoveMesh()`, causes BLAS cleanup on the next `RenderFrame()`

### rtx-chessboard Reference
- [hw_path_tracer.cpp](../../rtx-chessboard/src/render/hw/hw_path_tracer.cpp): `BuildAllBLAS()` (~line 274), `BuildTLASImpl()` (~line 533), compaction logic, instance custom index encoding

---

## Phase 7A: G-Buffer Images + Environment Map + Blue Noise

**Goal:** Create the supporting resources needed by the ray tracing pipeline: G-buffer output images, HDR environment map with importance sampling CDFs, and blue noise table.

**Source:** rtx-chessboard `render/hw/hw_path_tracer.cpp` (image creation), `loaders/environment_loader.cpp`, `render/blue_noise.h`

### Tasks

1. Create G-buffer images (host-owned in `app/core/gbuffer_images.h/.cpp`):
   - `noisy_diffuse`: RGBA16F, `VK_IMAGE_USAGE_STORAGE_BIT`
   - `noisy_specular`: RGBA16F, `VK_IMAGE_USAGE_STORAGE_BIT`
   - `motion_vectors`: RG16F (recommended) or RGBA16F, `VK_IMAGE_USAGE_STORAGE_BIT`
   - `linear_depth`: RG16F (recommended) or RGBA16F, `VK_IMAGE_USAGE_STORAGE_BIT` — `.r` = view-space depth, `.g` = hit distance (Phase 8E)
   - `world_normals`: RGBA16F, `VK_IMAGE_USAGE_STORAGE_BIT`
   - `diffuse_albedo`: R11G11B10F (recommended) or RGBA16F, `VK_IMAGE_USAGE_STORAGE_BIT`
   - `specular_albedo`: R11G11B10F (recommended) or RGBA16F, `VK_IMAGE_USAGE_STORAGE_BIT`
   - A single set of G-buffer images is sufficient — the renderer and denoiser are sequential within the same command buffer (render completes, then denoise reads). Temporal denoisers (ReLAX, ML) maintain their own internal history; the host does not provide previous-frame G-buffer images.
   - The `monti_view` app creates images with `VK_IMAGE_USAGE_STORAGE_BIT` only. The `monti_datagen` app adds `VK_IMAGE_USAGE_TRANSFER_SRC_BIT` for GPU→CPU readback via staging buffers (see Phase 11B). This keeps datagen concerns out of the core G-buffer allocation.
   - Support resize/recreate on window resize
   - Transition images from `VK_IMAGE_LAYOUT_UNDEFINED` → `VK_IMAGE_LAYOUT_GENERAL` on creation (required for storage image access by the renderer and denoiser)
   - Require `shaderStorageImageReadWithoutFormat` + `shaderStorageImageWriteWithoutFormat` at device creation (see design decision #21) — enables format-agnostic storage image access, so shaders work with either compact or RGBA16F formats without permutations

2. Implement environment map GPU resources (`renderer/src/vulkan/EnvironmentMap.h/.cpp`):
   - **Loading flow follows the existing Scene texture pattern:**
     1. App loads EXR pixels via tinyexr → `float*` RGBA data (tinyexr stays in the app layer, not in `monti_vulkan`)
     2. App wraps the pixel data in a `TextureDesc` with `PixelFormat::kRGBA16F` and calls `scene.AddTexture()` → gets `TextureId`
     3. App calls `scene.SetEnvironmentLight({.hdr_lat_long = texture_id, .intensity = 1.0f, .rotation = 0.0f})`
     4. Renderer discovers the environment light via `Scene::GetEnvironmentLight()` during `RenderFrame()` (or `SetScene()`), and the internal `EnvironmentMap` class computes CDFs, generates mipmaps, and uploads GPU resources
   - If no environment map is specified (neither in the glTF scene nor via `--env`), the renderer operates without one — miss rays return black. The renderer creates 1×1 black placeholder textures (env map + CDF images) at initialization and binds them by default, so the shader code paths and descriptor sets remain valid without branching or null descriptors. When an environment light is later set on the Scene, the real textures replace the placeholders.
   - The `EnvironmentLight` struct in Scene controls `intensity` and `rotation` — these parameters are read by the renderer each frame (no GPU re-upload needed for parameter changes, only push constant/uniform updates)
   - Create `VkImage` + `VkImageView` in `VK_FORMAT_R16G16B16A16_SFLOAT` (RGBA16F — sufficient HDR range, half the memory of RGBA32F)
   - Generate full mipmap chain via `vkCmdBlitImage` cascade (required for multi-tap background blur sampling in shaders). Image usage flags: `VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT`
   - Create `VkSampler` matching rtx-chessboard: `magFilter = LINEAR`, `minFilter = LINEAR`, `mipmapMode = LINEAR`, `addressModeU = REPEAT` (equirectangular wraps horizontally), `addressModeV = CLAMP_TO_EDGE` (clamp at poles), `maxLod = VK_LOD_CLAMP_NONE`
   - Pre-compute marginal/conditional CDFs for importance sampling as **sampled images** (matching rtx-chessboard). Marginal CDF: 1D image (height × 1, `VK_FORMAT_R32_SFLOAT`, `VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT`). Conditional CDF: 2D image (width × height, `VK_FORMAT_R32_SFLOAT`, same usage flags). Both use a `VK_FILTER_NEAREST` sampler (accessed via `texelFetch` in the binary search — no filtering needed). Using sampled images matches the rtx-chessboard shader code directly and avoids rewriting the `binarySearchCDF1D`/`binarySearchCDF2D` GLSL functions.
   - Follow rtx-chessboard's CDF computation: luminance compression (`CompressLuminance` — linear below 1.0, logarithmic above), cos(θ) weighting for equirectangular solid angles, marginal PDF (row sums), marginal CDF, per-row conditional CDF. Track `EnvironmentStatistics` (average/max/variance/solid-angle-weighted luminance).

3. Implement blue noise table (`renderer/src/vulkan/BlueNoise.h/.cpp`):
   - Generate blue noise data on the CPU using Sobol sequence with Owen scrambling (matching rtx-chessboard's `BlueNoise` class)
   - Table size: 128×128 tile (`kTableSize = 16384`), 4 components per entry (`kComponentsPerEntry = 4` for 4 bounces)
   - Pack 4 random bytes per bounce: `uint32_t = (r0) | (r1 << 8) | (r2 << 16) | (r3 << 24)`
   - Owen scrambling: XOR each tile entry with tile-specific MurmurHash3 hash
   - Buffer size: 16384 × 4 × 4 = 256 KB, uploaded to `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`
   - Upload via staging buffer to device-local memory

### Verification
- Integration test: load an HDR environment map, verify CDF images are non-zero and marginal CDF last entry ≈ 1.0 (readback marginal CDF last texel)
- Environment map image has correct mipmap chain (verify `mip_levels` = floor(log2(max(w,h))) + 1)
- G-buffer images created at correct resolution with correct formats
- Blue noise buffer allocated and populated (256 KB)
- No VMA allocation failures or validation errors
- Window resize recreates all G-buffer images without leaks (environment map and blue noise are resolution-independent)

### rtx-chessboard Reference
- [hw_path_tracer.cpp](../../rtx-chessboard/src/render/hw/hw_path_tracer.cpp): output image creation, env map sampler (line ~1164), CDF sampler (line ~1200)
- [environment_loader.cpp](../../rtx-chessboard/src/loaders/environment_loader.cpp): HDR loading, CDF computation, mipmap generation
- [blue_noise.h/.cpp](../../rtx-chessboard/src/render/blue_noise.h): Sobol + Owen scrambling blue noise generation

---

## Phase 7B: Descriptor Sets + Push Constants + Ray Tracing Pipeline

**Goal:** Create the Vulkan ray tracing pipeline object, descriptor set layout, descriptor pool, descriptor sets, push constant layout, and shader binding table. Skeleton shaders declare the full descriptor layout and push constants but contain no real logic — this is the pipeline plumbing. Real shader logic begins in Phase 7C.

**Source:** rtx-chessboard `render/hw/hw_path_tracer.cpp` (pipeline creation ~line 59, SBT setup, descriptor layout, push constants)

**New files:** `renderer/src/vulkan/RtPipeline.h` and `RtPipeline.cpp` — encapsulates descriptor set layout, descriptor pool, descriptor sets, pipeline layout, ray tracing pipeline, and SBT. Owned by `Renderer::Impl`. Does not duplicate code from `Renderer.cpp` (which remains responsible for `RenderFrame()` orchestration, scene management, and public API).

### Design Decisions

- **`maxPipelineRayRecursionDepth = 1`.** This controls how deeply `traceRayEXT` calls can *nest* (a closest-hit shader calling `traceRayEXT` which invokes another closest-hit, etc.) — it does **not** limit the number of bounces. The path tracer uses an iterative bounce loop in the raygen shader: trace a ray, get hit data back in the payload, evaluate the BRDF, pick a new direction, trace again — all from raygen. Shadow rays are also traced from raygen after the closest-hit returns. No shader ever calls `traceRayEXT` from within another `traceRayEXT` invocation, so depth 1 (raygen → closest-hit/miss and back) is sufficient. The bounce count (4 opaque + 8 transparency) is controlled by the loop counter and `max_bounces` push constant. Setting recursion depth higher would waste GPU stack memory per ray for no benefit. Matches rtx-chessboard. Defined as `constexpr uint32_t kMaxRayRecursionDepth = 1` in `RtPipeline.h`.
- **Push constants defined fully up front.** All fields needed through Phase 8C are included from the start to avoid pipeline layout recreation in later phases. Fields unused by the skeleton shaders are zero-initialized and ignored until the relevant phase activates them.
- **Descriptor update-after-bind for bindless textures.** The bindless texture array (binding 10) uses `VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT` and `VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT` with a pre-allocated maximum of `kMaxBindlessTextures = 1024` entries. This is the standard Vulkan 1.2 approach for variable-size texture arrays: descriptors can be updated after the set is bound, and unwritten slots are valid as long as they aren't accessed by shaders. All other bindings are updated normally via `vkUpdateDescriptorSets`. Requires `descriptorBindingPartiallyBound` and `descriptorBindingUpdateAfterBind` features (Vulkan 1.2 core).
- **Area light buffer pre-allocated.** The area light storage buffer (binding 11) is allocated as a small placeholder (1 element) at initialization. When the scene has area lights, `GpuScene` re-uploads the buffer contents. `area_light_count` in push constants controls shader iteration. This keeps the descriptor set valid at all times without null descriptors or branching.

### Tasks

1. Create descriptor set layout (`RtPipeline::CreateDescriptorSetLayout()`):

   | Binding | Descriptor Type | Count | Stage Flags | Description |
   |---------|-----------------|-------|-------------|-------------|
   | 0 | `ACCELERATION_STRUCTURE_KHR` | 1 | `RAYGEN` | TLAS |
   | 1 | `STORAGE_IMAGE` | 1 | `RAYGEN` | noisy_diffuse output |
   | 2 | `STORAGE_IMAGE` | 1 | `RAYGEN` | noisy_specular output |
   | 3 | `STORAGE_IMAGE` | 1 | `RAYGEN` | motion_vectors output |
   | 4 | `STORAGE_IMAGE` | 1 | `RAYGEN` | linear_depth output |
   | 5 | `STORAGE_IMAGE` | 1 | `RAYGEN` | world_normals output |
   | 6 | `STORAGE_IMAGE` | 1 | `RAYGEN` | diffuse_albedo output |
   | 7 | `STORAGE_IMAGE` | 1 | `RAYGEN` | specular_albedo output |
   | 8 | `STORAGE_BUFFER` | 1 | `CLOSEST_HIT` | Mesh address table (`MeshAddressEntry[]`) |
   | 9 | `STORAGE_BUFFER` | 1 | `RAYGEN \| CLOSEST_HIT` | Material buffer (`PackedMaterial[]`) |
   | 10 | `COMBINED_IMAGE_SAMPLER` | `kMaxBindlessTextures` | `RAYGEN \| CLOSEST_HIT` | Bindless texture array (variable count, update-after-bind, partially bound) |
   | 11 | `STORAGE_BUFFER` | 1 | `RAYGEN` | Area light buffer (`PackedAreaLight[]`) |
   | 12 | `STORAGE_BUFFER` | 1 | `RAYGEN \| CLOSEST_HIT` | Blue noise table |
   | 13 | `COMBINED_IMAGE_SAMPLER` | 1 | `RAYGEN \| CLOSEST_HIT \| MISS` | Environment map (linear sampler) |
   | 14 | `COMBINED_IMAGE_SAMPLER` | 1 | `RAYGEN \| CLOSEST_HIT` | Marginal CDF (nearest sampler) |
   | 15 | `COMBINED_IMAGE_SAMPLER` | 1 | `RAYGEN \| CLOSEST_HIT` | Conditional CDF (nearest sampler) |

   > Note: Per-mesh vertex/index data is accessed via GLSL `buffer_reference` using device addresses from the mesh address table (binding 8) — no separate vertex/index descriptor bindings needed. Binding 10 requires `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT` on the layout and `VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT` on the pool.

2. Create descriptor pool and allocate descriptor set (`RtPipeline::CreateDescriptorPool()`):
   - Pool sizes:
     - Acceleration structures: 1
     - Storage images: 7 (bindings 1–7)
     - Storage buffers: 4 (bindings 8, 9, 11, 12)
     - Combined image samplers: `kMaxBindlessTextures` + 3 (bindings 10, 13, 14, 15)
   - Pool flags: `VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT` (required for binding 10)
   - `maxSets = 1`
   - Allocate single descriptor set from pool using the layout from Task 1
   - Use `VkDescriptorSetVariableDescriptorCountAllocateInfo` to specify the actual texture count for binding 10 (initially 0 or 1 placeholder; updated as textures are uploaded)

3. Create push constant layout and struct:

   ```cpp
   // renderer/src/vulkan/RtPipeline.h
   struct PushConstants {
       // ── Camera (192 bytes) ───────────────────────────────────────
       glm::mat4 inv_view;              // 64 bytes, offset 0
       glm::mat4 inv_proj;              // 64 bytes, offset 64
       glm::mat4 prev_view_proj;        // 64 bytes, offset 128

       // ── Render parameters (16 bytes) ─────────────────────────────
       uint32_t frame_index;            // 4 bytes, offset 192
       uint32_t paths_per_pixel;        // 4 bytes, offset 196
       uint32_t max_bounces;            // 4 bytes, offset 200
       uint32_t area_light_count;       // 4 bytes, offset 204

       // ── Scene globals (16 bytes) ─────────────────────────────────
       uint32_t env_width;              // 4 bytes, offset 208
       uint32_t env_height;             // 4 bytes, offset 212
       float    env_avg_luminance;      // 4 bytes, offset 216
       float    env_max_luminance;      // 4 bytes, offset 220

       // ── Scene globals continued (16 bytes) ───────────────────────
       float    env_rotation;           // 4 bytes, offset 224 (radians)
       float    skybox_mip_level;       // 4 bytes, offset 228
       float    jitter_x;              // 4 bytes, offset 232
       float    jitter_y;              // 4 bytes, offset 236

       // ── Debug (8 bytes + 8 padding → 16 bytes) ───────────────────
       uint32_t debug_mode;             // 4 bytes, offset 240
       uint32_t pad0;                   // 4 bytes, offset 244 (pad to 248)
       // Total: 248 bytes (within 256-byte guaranteed minimum)
   };
   static_assert(sizeof(PushConstants) == 248);
   ```

   - Push constant range: offset 0, size 248, stage flags `VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR`
   - `prev_view_proj` is used for motion vectors (Phase 8C) — zero-initialized until then
   - `area_light_count` is 0 when no area lights are present
   - `jitter_x`, `jitter_y` are 0.0 until sub-pixel jitter (Phase 8C)
   - `debug_mode` is 0 (disabled) by default

4. Create ray tracing pipeline (`RtPipeline::CreatePipeline()`):
   - **SPIR-V compilation pipeline** (deferred from Phase 1):
     - CMake custom command: `glslc --target-env=vulkan1.2 -I ${SHADER_DIR} -O -o <output>.spv <input>`
     - Shader sources: `shaders/raygen.rgen`, `shaders/miss.rmiss`, `shaders/closesthit.rchit`
     - Generates `.spv` files in the build directory; loaded at runtime via `LoadShaderFile()` as `std::vector<uint32_t>`
     - Shader include directory: `shaders/` (for shared GLSL includes added in Phase 8A)
   - **Skeleton shaders** (minimal stubs that declare the full descriptor/push-constant layout for pipeline validation — no real logic):
     - `shaders/raygen.rgen`: declares all 16 descriptor bindings, push constant block, and ray payload struct. Entry point calls `traceRayEXT()` with a fixed direction and writes a solid color to `noisy_diffuse`. All other output images written to zero.
     - `shaders/miss.rmiss`: declares push constants and ray payload. Writes a solid background color to payload.
     - `shaders/closesthit.rchit`: declares descriptor bindings for mesh address table and materials, push constants, and ray payload. Writes a solid color to payload.
   - Create `VkShaderModule` for each loaded SPIR-V blob
   - 3 shader stages: raygen (stage 0), miss (stage 1), closest-hit (stage 2)
   - 3 shader groups:
     - Group 0: `VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR` — raygen (stage 0)
     - Group 1: `VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR` — miss (stage 1)
     - Group 2: `VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR` — closest-hit (stage 2)
   - `maxPipelineRayRecursionDepth = kMaxRayRecursionDepth` (= 1)
   - Pass `RendererDesc::pipeline_cache` to `vkCreateRayTracingPipelinesKHR` for faster subsequent creation (may be `VK_NULL_HANDLE` if the host didn't provide one)
   - Destroy shader modules after pipeline creation (they are no longer needed)

5. Create shader binding table (`RtPipeline::CreateSbt()`):
   - Query `VkPhysicalDeviceRayTracingPipelinePropertiesKHR` for `shaderGroupHandleSize`, `shaderGroupHandleAlignment`, `shaderGroupBaseAlignment`
   - Compute aligned handle size: `AlignUp(shaderGroupHandleSize, shaderGroupHandleAlignment)`
   - Retrieve shader group handles via `vkGetRayTracingShaderGroupHandlesKHR`
   - Compute SBT region sizes (each aligned to `shaderGroupBaseAlignment`):
     - Raygen region: 1 handle
     - Miss region: 1 handle
     - Hit region: 1 handle
   - Allocate SBT buffer: `VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT`, `VMA_MEMORY_USAGE_CPU_TO_GPU`
   - Copy handles into buffer at correct aligned offsets
   - Store `VkStridedDeviceAddressRegionKHR` for raygen, miss, hit, and callable (empty) regions — used by `vkCmdTraceRaysKHR` in `RenderFrame()`

### Verification
- Pipeline creation succeeds without validation errors
- SBT buffer allocated with correct size and alignment (verify `shaderGroupBaseAlignment` from `VkPhysicalDeviceRayTracingPipelinePropertiesKHR`)
- Descriptor set layout creation succeeds with `UPDATE_AFTER_BIND` flags
- Descriptor pool creation succeeds with correct pool sizes
- Descriptor set allocation succeeds (including variable descriptor count for binding 10)
- Descriptor sets update without validation errors when bound to the resources from Phase 7A and Phases 5–6
- Push constant struct size (248 bytes) fits within device `maxPushConstantsSize` limit (guaranteed ≥ 128, desktop GPUs typically 256)
- Skeleton shaders compile via `glslc` without errors
- Skeleton shaders declare the full descriptor/push-constant layout matching the C++ structs (validated by pipeline creation)

### rtx-chessboard Reference
- [hw_path_tracer.cpp](../../rtx-chessboard/src/render/hw/hw_path_tracer.cpp): pipeline creation (~line 673), SBT setup (~line 1219), descriptor layout (~line 790), push constant struct (~line 37), descriptor pool (~line 851)

---

## Phase 7C: Raygen + Miss + Closesthit Stub + RenderFrame

**Goal:** Replace the Phase 7B skeleton shaders with initial functional shaders and complete `Renderer::RenderFrame()` trace command recording so that primary rays are cast, the environment map is visible via miss rays, and geometry produces a placeholder color on hit. The descriptor layout, push constants, pipeline structure, and SBT from Phase 7B are unchanged — only the shader source files and Renderer command recording are updated.

**Source:** rtx-chessboard `shaders/raygen.rgen`, `shaders/miss.rmiss`, `shaders/closesthit.rchit`, `shaders/include/sampling.glsl`

### Design Decisions

- **Camera matrices computed from `CameraParams`.** The `CameraParams` struct in `Scene` defines the camera in decomposed form (position, target, fov, etc.). `CameraParams` provides `ViewMatrix()` and `ProjectionMatrix(float aspect)` methods that compute the `glm::mat4` matrices. The renderer computes `inv_view` and `inv_proj` by inverting these matrices each frame, and caches the previous frame's view-projection matrix for `prev_view_proj`. This keeps matrix computation in the `Camera`/`CameraParams` layer (reusable) and inverse computation in the renderer (where it's consumed).
- **Miss shader sets `payload.missed = true`; raygen handles environment sampling.** This matches rtx-chessboard's pattern and is needed for Phase 8A+ where the bounce number determines the sampling strategy (multi-tap blur for primary misses, sharp for secondary). No environment sampling in the miss shader.
- **Closesthit returns barycentric coordinates as color.** No vertex buffer fetch or normal computation — just `vec3(1-u-v, u, v)` from `hitAttributeEXT`. Full vertex interpolation and material shading are deferred to Phase 8A.
- **Unused G-buffer images written to `vec4(0.0)`.** Only `noisy_diffuse` carries real data in this phase. All other G-buffer outputs (`noisy_specular`, `motion_vectors`, `linear_depth`, `world_normals`, `diffuse_albedo`, `specular_albedo`) are written to zero. Real data fills in during Phases 8A–8C.
- **Per-frame `UNDEFINED → GENERAL` transitions.** All G-buffer storage images are transitioned from `VK_IMAGE_LAYOUT_UNDEFINED` to `VK_IMAGE_LAYOUT_GENERAL` every frame before tracing. This discards previous contents (safe — each frame overwrites all pixels) and matches rtx-chessboard's pattern.

### Prerequisites

- **Add `ViewMatrix()` and `ProjectionMatrix(float aspect)` to `CameraParams`:** These are free functions or methods on `CameraParams` that compute `glm::mat4` from the decomposed camera parameters. `ViewMatrix()` uses `glm::lookAt(position, target, up)`. `ProjectionMatrix(aspect)` uses `glm::perspective(vertical_fov_radians, aspect, near_plane, far_plane)`. These belong in the scene layer since camera matrices are needed beyond just the renderer. Added inline in `Camera.h`.

### Tasks

1. Create `shaders/include/common.glsl` (early shader library — expanded in Phase 8A):
   - `const float PI = 3.14159265358979323846;`
   - `vec3 rotateDirectionY(vec3 dir, float rotation)` — rotate direction around Y by radians
   - `vec2 directionToUV(vec3 dir)` — convert world-space direction to equirectangular UV
   - `vec2 directionToUVRotated(vec3 dir, float rotation)` — `directionToUV` with azimuthal rotation
   - `vec3 sampleEnvironmentBlurred(sampler2D env_map, vec3 direction, float mip_level, float rotation)` — 9-tap Gaussian blur for environment map background (center weight 0.25, cardinal 0.125×4, diagonal 0.0625×4)
   - Ported from rtx-chessboard `shaders/include/sampling.glsl` (only the functions listed above)

2. Update `shaders/raygen.rgen` (replace skeleton logic):
   - `#include "include/common.glsl"`
   - Compute ray origin and direction from pixel coordinates + camera inverse matrices (`inv_view`, `inv_proj` from push constants):
     ```glsl
     vec2 ndc = (vec2(pixel) + 0.5) / vec2(size) * 2.0 - 1.0;
     vec4 target = pc.inv_proj * vec4(ndc, 1.0, 1.0);
     vec3 direction = normalize((pc.inv_view * vec4(normalize(target.xyz / target.w), 0.0)).xyz);
     vec3 origin = (pc.inv_view * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
     ```
   - Single sample per pixel (no MIS, no bounce loop, `paths_per_pixel` ignored this phase)
   - `traceRayEXT()` with `gl_RayFlagsOpaqueEXT`, `tMin = 0.001`, `tMax = 10000.0`
   - After trace, check `payload.missed`:
     - **If missed:** sample environment map using `sampleEnvironmentBlurred(env_map, direction, pc.skybox_mip_level, pc.env_rotation)` and write result to `noisy_diffuse`
     - **If hit:** write `payload.color` (barycentric coordinates from closesthit) to `noisy_diffuse`
   - Write sentinel values for primary miss G-buffer outputs (matching rtx-chessboard):
     - `motion_vectors`: `vec4(0.0)`
     - `linear_depth`: `vec4(1e4, 0.0, 0.0, 0.0)` (large depth signals sky)
     - `world_normals`: `vec4(0.0, 0.0, 1.0, 0.0)` (forward-facing placeholder)
     - `diffuse_albedo`: `vec4(0.0)`
     - `specular_albedo`: `vec4(0.04, 0.04, 0.04, 1.0)` (default F0)
   - Write `vec4(0.0)` to all other unused G-buffer images (`noisy_specular`, and for hit pixels: `motion_vectors`, `linear_depth`, `world_normals`, `diffuse_albedo`, `specular_albedo`)

3. Update `shaders/miss.rmiss` (replace skeleton logic):
   - Set `payload.missed = true` (matching rtx-chessboard)
   - No environment map sampling in the miss shader — that is handled in raygen

4. Update `shaders/closesthit.rchit` (replace skeleton logic):
   - Compute barycentric coordinates: `vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y)`
   - Write barycentrics to `payload.color`
   - Set `payload.missed = false`
   - Set `payload.hit_t = gl_HitTEXT`
   - No vertex buffer fetch, no material fetch, no normal computation — deferred to Phase 8A

5. Update `RayPayload` struct (shared across all three shaders):
   ```glsl
   struct RayPayload {
       vec3 color;
       float hit_t;
       bool missed;
   };
   ```

6. Complete `Renderer::RenderFrame()` trace command recording (append to existing method after descriptor update):
   - **Populate `PushConstants` from scene state:**
     - `inv_view = glm::inverse(camera.ViewMatrix())`
     - `inv_proj = glm::inverse(camera.ProjectionMatrix(aspect))` where `aspect = width / height` from `RendererDesc`
     - `prev_view_proj` = cached previous frame's `projection * view` matrix (zero on first frame)
     - `frame_index` from the method parameter (currently unused placeholder — wire it through)
     - `paths_per_pixel = samples_per_pixel_` (from `SetSamplesPerPixel()`)
     - `max_bounces = 0` (no bounces this phase)
     - `area_light_count = scene->AreaLights().size()`
     - `env_width`, `env_height`, `env_avg_luminance`, `env_max_luminance` from `EnvironmentMap`
     - `env_rotation` from `scene->GetEnvironmentLight()->rotation` (0.0 if no env light)
     - `skybox_mip_level` = 0.0 (default, no blur control UI yet)
     - `jitter_x = 0.0`, `jitter_y = 0.0` (sub-pixel jitter deferred to Phase 8C)
     - `debug_mode = 0`
   - Cache current frame's view-projection matrix as `prev_view_proj_` for next frame
   - **Transition G-buffer images:** `VK_IMAGE_LAYOUT_UNDEFINED → VK_IMAGE_LAYOUT_GENERAL` for all 7 G-buffer images using `VkImageMemoryBarrier2` with:
     - `srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT`, `srcAccessMask = 0`
     - `dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR`, `dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT`
   - **Bind pipeline:** `vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, raytrace_pipeline.Pipeline())`
   - **Bind descriptor set:** `vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, layout, 0, 1, &set, 0, nullptr)`
   - **Push constants:** `vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR, 0, sizeof(PushConstants), &pc)`
   - **Dispatch trace:** `vkCmdTraceRaysKHR(cmd, &raygen_region, &miss_region, &hit_region, &callable_region, width, height, 1)`

### Verification
- **Integration test (automated, offscreen):** Load Cornell box via `test::BuildCornellBox()`, set an environment map on the scene, call `RenderFrame()` to render into G-buffer images, read back `noisy_diffuse` pixels via staging buffer:
  - Pixels outside geometry (miss rays) are non-zero and match environment map colors (not solid black or solid blue)
  - Pixels inside geometry (hit rays) show barycentric color variation (not a single solid color)
  - At least some pixels differ between miss and hit regions
  - No NaN or Inf values in the output
- Zero Vulkan validation errors during the entire render pass
- Push constants struct size (248 bytes) verified against `VkPhysicalDeviceLimits::maxPushConstantsSize`
- FLIP-based perceptual comparison tests are **not** added in this phase; they begin in Phase 8A when the renderer produces meaningful shaded output
- Interactive visual verification via `monti_view` is deferred to Phase 10B; this phase tests via automated offscreen rendering only

### rtx-chessboard Reference
- [raygen.rgen](../../rtx-chessboard/shaders/raygen.rgen): camera ray generation, `sampleEnvironmentBlurred` usage on primary miss
- [miss.rmiss](../../rtx-chessboard/shaders/miss.rmiss): `payload.missed = true` pattern
- [closesthit.rchit](../../rtx-chessboard/shaders/closesthit.rchit): barycentric coordinates, hit payload
- [sampling.glsl](../../rtx-chessboard/shaders/include/sampling.glsl): `directionToUV`, `rotateDirectionY`, `sampleEnvironmentBlurred`
- [hw_path_tracer.cpp](../../rtx-chessboard/src/render/hw/hw_path_tracer.cpp): `Trace()` method, image transitions, command recording

---

## Phase 8A: GLSL Shader Library + Single-Bounce PBR

**Goal:** Port the GLSL utility library from rtx-chessboard and implement full material shading in the closest-hit shader. The raygen uses a single bounce (primary ray + direct lighting via next-event estimation) to validate correct PBR output before adding the full bounce loop in Phase 8B.

**Source:** rtx-chessboard `shaders/include/*.glsl`, `shaders/closesthit.rchit`

### Design Decisions

- **`HitPayload` struct matches rtx-chessboard.** The Phase 7C `RayPayload` is replaced with `HitPayload` carrying: `hit_pos` (vec3), `hit_t` (float), `normal` (vec3), `material_index` (uint), `uv` (vec2), `missed` (bool), `_pad` (float). The raygen reads material data from the material buffer using `material_index`; closesthit is geometry-only (no material fetch, no texture sampling). This keeps closesthit fast and moves all shading decisions to raygen.

- **Vertex access via `buffer_reference`.** Unlike rtx-chessboard's merged vertex/index buffers, Monti uses per-mesh separate buffers with device addresses. Closesthit decodes `gl_InstanceCustomIndexEXT` to get the mesh address index (lower 12 bits), looks up `MeshAddressEntry` from binding 8, then uses `GL_EXT_buffer_reference2` to fetch vertex/index data at the device address. A new shared include (`shaders/include/vertex.glsl`) defines the GLSL `Vertex` struct (matching the 56-byte C++ `Vertex`: position vec3, normal vec3, tangent vec4, tex_coord_0 vec2, tex_coord_1 vec2), `MeshAddressEntry` struct, and `buffer_reference` layout declarations.

- **`common.glsl` refactored — functions move to `sampling.glsl`.** The coordinate conversion functions (`directionToUV`, `rotateDirectionY`, `directionToUVRotated`) and `sampleEnvironmentBlurred` are moved from `common.glsl` into `sampling.glsl` (matching rtx-chessboard's file organization). `PI` moves to `brdf.glsl`. `common.glsl` becomes a thin umbrella include that `#include`s `brdf.glsl` and `sampling.glsl`, preserving backward compatibility for shaders that `#include "include/common.glsl"`.

- **`mis.glsl` ported with clearcoat stubbed to zero.** The full `SamplingProbabilities`, `AllPDFs`, `chooseStrategy`, `calculateAllPDFs`, and `calculateMISWeight` functions are ported from rtx-chessboard. In place of `#include "clearcoat.glsl"`, Phase 8A defines a `CLEAR_COAT_F0` constant (`vec3(0.04)`) and a stub `calculateClearCoatAttenuation` that returns `1.0` (no attenuation), inline within `mis.glsl`. The `calculateSamplingProbabilities` function forces `cc_strength = 0.0` so clearcoat probability is always zero and the strategy selection is effectively 3-way (diffuse, specular, environment). Phase 8B replaces the stub with the real `#include "clearcoat.glsl"` and removes the forced zero.

- **Single-bounce = direct lighting only.** The raygen traces the primary ray. On hit, it evaluates direct lighting via next-event estimation: one BRDF-sampled direction (3-way MIS: diffuse/specular/environment) and per-area-light shadow rays. No recursive bounce loop — indirect illumination is deferred to Phase 8B. All radiance goes to `noisy_diffuse`; `noisy_specular` is written as zero (diffuse/specular split requires the bounce loop classification from Phase 8B).

- **Normal mapping deferred.** Phase 8A interpolates the geometric normal only (no tangent interpolation, no TBN construction, no normal map sampling). The `tangent` field in the Vertex struct is available via `buffer_reference` but unused this phase. Normal mapping is added in a later phase alongside TBN construction in closesthit.

- **BRDF evaluation uses `evaluatePBR` directly (no clearcoat layer).** Since clearcoat is deferred to Phase 8B, raygen calls `evaluatePBR()` from `brdf.glsl` directly, not `evaluateMultilayerBRDF()`. Phase 8B introduces `clearcoat.glsl` and switches to the multilayer evaluation.

- **Multi-sample averaging.** When `paths_per_pixel > 1`, each path is traced independently and the results are summed, then divided by `paths_per_pixel` (simple 1/N average). Each path gets distinct blue noise via XOR scrambling with a golden-ratio-derived constant per path index.

  > **Future note:** Weighted averaging (e.g., inverse-variance or MIS-weighted across paths) is not a standard improvement for uniform-weight multi-sample path tracing. The simple average is unbiased and correct. If adaptive sampling is added later (varying SPP per pixel based on variance), the weighting would change at the adaptive sampling level, not the per-path accumulation level.

- **Texture index sentinel value.** `PackedMaterial` encodes texture indices as `float`-encoded `uint32_t` via `std::bit_cast<float>()`. In GLSL, decode with `floatBitsToUint(field)`. The sentinel for "no texture" is `0xFFFFFFFFu`. Shaders must check: `uint tex_idx = floatBitsToUint(mat.metallic_clearcoat.a); if (tex_idx != 0xFFFFFFFFu) { albedo *= texture(bindless_textures[nonuniformEXT(tex_idx)], uv).rgb; }`. This pattern applies to all texture index fields: `metallic_clearcoat.a` (base color map), `opacity_ior.b` (normal map), `opacity_ior.a` (metallic-roughness map), `transmission_volume.a` (transmission map), `attenuation_color_pad.a` (emissive map).

### Tasks

1. Port GLSL shader includes:

   a. **`shaders/include/brdf.glsl`**: Port from rtx-chessboard. Contains `PI` constant (moved from `common.glsl`), `F_Schlick`, `D_GGX`, `G_SmithG1GGX`, `G_SmithGGX`, `evaluatePBR`. Uses `#ifndef BRDF_GLSL` / `#define BRDF_GLSL` / `#endif` guard.

   b. **`shaders/include/sampling.glsl`**: Port from rtx-chessboard. `#include "brdf.glsl"` (for `PI`). Move `directionToUV`, `rotateDirectionY`, `directionToUVRotated`, `sampleEnvironmentBlurred` here from `common.glsl`. Add new functions: `cosineSampleHemisphere(xi, N)`, `sampleGGX(xi, N, roughness)`, `buildONB(N, out T, out B)`, environment CDF sampling (`binarySearchCDF1D`, `binarySearchCDF2D`, `environmentCDFSample`, `environmentCDFPdf`), `uvToDirection`.

   c. **`shaders/include/bluenoise.glsl`**: Port from rtx-chessboard. Constants: `kBlueNoiseTileSize = 128u`, `kBlueNoiseTableSize = 16384u`. Functions: `getSpatialHashTemporal(pixel, frameIndex)`, `extractBounceRandoms(packed)`, `getBlueNoiseRandom(packed, bounce)`. No external include dependencies.

   d. **`shaders/include/mis.glsl`**: Port from rtx-chessboard with clearcoat stubbed. `#include "brdf.glsl"`. Inline stub: `const vec3 CLEAR_COAT_F0 = vec3(0.04); float calculateClearCoatAttenuation(float VdotH, float cc_strength) { return 1.0; }`. In `calculateSamplingProbabilities`, force `cc_strength = 0.0` so clearcoat probability is always zero. All other functions (`calculateDiffusePDF`, `calculateGGXPDF`, `chooseStrategy`, `calculateAllPDFs`, `calculateMISWeight`) ported verbatim. Phase 8B replaces the inline stub with `#include "clearcoat.glsl"` and removes the forced zero.

   e. **Refactor `shaders/include/common.glsl`**: Remove `PI`, `directionToUV`, `rotateDirectionY`, `directionToUVRotated`, `sampleEnvironmentBlurred`. Replace body with: `#include "brdf.glsl"` and `#include "sampling.glsl"`. Keeps backward compatibility for existing `#include "include/common.glsl"` in shaders.

2. Create `shaders/include/vertex.glsl` — buffer reference vertex/index access:

   This is a new shared include (no rtx-chessboard equivalent) required because Monti uses per-mesh separate buffers instead of merged buffers.

   - Enable `GL_EXT_buffer_reference2` and `GL_EXT_scalar_block_layout` extensions
   - Define `buffer_reference` layouts for vertex and index data:
     ```glsl
     layout(buffer_reference, scalar) readonly buffer VertexBufferRef {
         Vertex vertices[];
     };
     layout(buffer_reference, scalar) readonly buffer IndexBufferRef {
         uint indices[];
     };
     ```
   - Define `Vertex` struct matching the C++ 56-byte layout:
     ```glsl
     struct Vertex {
         vec3 position;     // offset 0
         vec3 normal;       // offset 12
         vec4 tangent;      // offset 24 (xyz = direction, w = bitangent sign)
         vec2 tex_coord_0;  // offset 40
         vec2 tex_coord_1;  // offset 48
     };
     ```
   - Define `MeshAddressEntry` matching the C++ 32-byte layout:
     ```glsl
     struct MeshAddressEntry {
         uint64_t vertex_address;
         uint64_t index_address;
         uint vertex_count;
         uint index_count;
         uint pad_0;
         uint pad_1;
     };
     ```
     Requires `GL_EXT_shader_explicit_arithmetic_types_int64` for `uint64_t`.
   - Provide a helper function:
     ```glsl
     void fetchTriangleVertices(MeshAddressEntry entry, uint primitive_id,
                                out Vertex v0, out Vertex v1, out Vertex v2) {
         IndexBufferRef ib = IndexBufferRef(entry.index_address);
         VertexBufferRef vb = VertexBufferRef(entry.vertex_address);
         uint base = primitive_id * 3;
         v0 = vb.vertices[ib.indices[base + 0]];
         v1 = vb.vertices[ib.indices[base + 1]];
         v2 = vb.vertices[ib.indices[base + 2]];
     }
     ```

3. Implement full `closesthit.rchit`:

   - Add extensions: `GL_EXT_buffer_reference2`, `GL_EXT_scalar_block_layout`, `GL_EXT_shader_explicit_arithmetic_types_int64`, `GL_GOOGLE_include_directive`, `GL_EXT_nonuniform_qualifier`
   - `#include "include/vertex.glsl"`
   - Replace `RayPayload` with `HitPayload`:
     ```glsl
     struct HitPayload {
         vec3 hit_pos;
         float hit_t;
         vec3 normal;
         uint material_index;
         vec2 uv;
         bool missed;
         float _pad;
     };
     ```
   - Decode `gl_InstanceCustomIndexEXT`:
     ```glsl
     uint custom_index = gl_InstanceCustomIndexEXT;
     uint mesh_addr_index = custom_index & 0xFFFu;
     uint material_index = (custom_index >> 12u) & 0xFFFu;
     ```
   - Look up `MeshAddressEntry` from binding 8 (mesh address table):
     ```glsl
     MeshAddressEntry entry = MeshAddressEntry(
         mesh_address_table.entries[mesh_addr_index * 2].xy,  // vertex_address
         mesh_address_table.entries[mesh_addr_index * 2].zw,  // index_address
         ...);  // or equivalent unpacking from uvec4[]
     ```
     > **Implementation note:** The mesh address table is declared as `uvec4 entries[]` in the current shader layout (matching Phase 7B). Each `MeshAddressEntry` is 32 bytes = 2 × uvec4. Unpack the two `uint64_t` device addresses from the first uvec4 (8 bytes each), and vertex/index counts from the second. Alternatively, redeclare the SSBO with the `MeshAddressEntry` struct directly using `scalar` layout — this is cleaner and matches the C++ side. Coordinate with the binding 8 declaration in raygen (which currently uses `uvec4 entries[]` as a placeholder).
   - Fetch triangle vertices via `fetchTriangleVertices(entry, gl_PrimitiveID, v0, v1, v2)`
   - Barycentric interpolation:
     ```glsl
     vec3 bary = vec3(1.0 - hit_attribs.x - hit_attribs.y, hit_attribs.x, hit_attribs.y);
     vec3 object_normal = normalize(v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z);
     vec2 uv = v0.tex_coord_0 * bary.x + v1.tex_coord_0 * bary.y + v2.tex_coord_0 * bary.z;
     ```
   - Transform normal to world space: `vec3 world_normal = normalize(gl_ObjectToWorldEXT * vec4(object_normal, 0.0))`
   - Compute hit position: `vec3 hit_pos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT`
   - Populate `HitPayload`: `hit_pos`, `hit_t`, `normal` (world-space), `material_index`, `uv`, `missed = false`
   - **No tangent interpolation, no normal map sampling, no material fetch, no texture sampling** — all deferred or handled in raygen

4. Update `raygen.rgen` for single-bounce shading:

   - Replace `RayPayload` with `HitPayload` (same struct as closesthit)
   - Add includes: `#include "include/bluenoise.glsl"`, `#include "include/mis.glsl"` (which pulls in `brdf.glsl`). `common.glsl` include retained for `sampling.glsl` functions.
   - **Per-pixel multi-sample loop:**
     ```glsl
     vec3 total_radiance = vec3(0.0);
     for (int path = 0; path < pc.paths_per_pixel; ++path) {
         // Per-path blue noise scrambling
         uvec4 bn_packed = blue_noise.entries[getSpatialHashTemporal(uvec2(pixel), pc.frame_index)];
         if (path > 0) {
             uint scramble = uint(path) * 0x9E3779B9u;
             bn_packed ^= uvec4(scramble, scramble * 2654435761u,
                                scramble * 2246822519u, scramble * 3266489917u);
         }
         // ... trace primary ray, shade, accumulate ...
         total_radiance += path_radiance;
     }
     vec3 final_color = total_radiance / float(pc.paths_per_pixel);
     ```
   - **Primary ray:** Same as Phase 7C — compute ray from pixel + inverse camera matrices. No sub-pixel jitter (deferred to Phase 8C); all samples per pixel use the same ray direction (blue noise only affects shading strategy selection and light sampling).
   - **On miss:** Same environment sampling as Phase 7C (`sampleEnvironmentBlurred`).
   - **On hit — material fetch in raygen:**
     - Unpack `PackedMaterial` from binding 9 using `payload.material_index`. Material is 5 × vec4 = 80 bytes, so material starts at `materials.data[payload.material_index * 5]`.
     - Extract: `albedo = base_color_roughness.rgb`, `roughness = max(base_color_roughness.a, 0.04)`, `metallic = metallic_clearcoat.r`, `opacity = opacity_ior.r`, `ior = opacity_ior.g`
     - **Texture index decoding:** Base color map index is `floatBitsToUint(metallic_clearcoat.a)`. If not `0xFFFFFFFFu` (sentinel for "no texture"), sample bindless texture: `albedo *= texture(bindless_textures[nonuniformEXT(tex_idx)], payload.uv).rgb`
     - Compute `F0 = mix(vec3(0.04), albedo, metallic)` for Fresnel reflectance at normal incidence
   - **Direct lighting via 3-way MIS (single sample):**
     1. Compute shading vectors: `N = payload.normal`, `V = normalize(-gl_WorldRayDirectionEXT)`, `NdotV = max(dot(N, V), 0.001)`
     2. Get blue noise randoms for bounce 0: `vec4 rands = getBlueNoiseRandom(bn_packed, 0)`
     3. Calculate sampling probabilities: `calculateSamplingProbabilities(N, V, F0, metallic, roughness, 0.0, 0.04, pc.env_avg_luminance, pc.env_max_luminance)` — clearcoat args are zero/default, resulting in 3-way split
     4. Choose strategy via `chooseStrategy(probs, rands.z)`
     5. Sample direction `L` based on strategy:
        - `STRATEGY_DIFFUSE`: `L = cosineSampleHemisphere(rands.xy, N)`
        - `STRATEGY_SPECULAR`: `H = sampleGGX(rands.xy, N, roughness); L = reflect(-V, H)`
        - `STRATEGY_ENVIRONMENT`: `env_uv = environmentCDFSample(rands.xy, marginal_cdf, conditional_cdf, env_size); L = rotateDirectionY(uvToDirection(env_uv), -pc.env_rotation)`
     6. Evaluate `NdotL = dot(N, L)`. If `NdotL <= 0`, contribution is zero (below horizon).
     7. Trace shadow ray toward `L` from `payload.hit_pos + N * 0.001`:
        - If miss (reached environment): `env_color = textureLod(env_map, directionToUVRotated(L, pc.env_rotation), 0.5).rgb`
        - Evaluate `brdf = evaluatePBR(albedo, roughness, metallic, F0, NdotV, NdotL, NdotH, VdotH)`
        - Compute all PDFs: `pdfs = calculateAllPDFs(N, V, L, roughness, 0.04)` and fill `pdfs.environment = environmentCDFPdf(env_uv, ...)`
        - MIS weight: `mis_w = calculateMISWeight(strategy, pdfs, probs)`
        - Accumulate: `path_radiance += env_color * brdf * NdotL * mis_w / chosen_pdf`
        - If hit (occluded): zero contribution from this sample
   - **Per-area-light shadow rays (separate from MIS):**
     - For each area light `i` in `[0, pc.area_light_count)`:
       - Unpack `PackedAreaLight` from binding 11: corner, edge_a, edge_b, radiance, two_sided
       - Sample uniform random point on quad: `p = corner + rands_light.x * edge_a + rands_light.y * edge_b` (use blue noise from bounce 1+ or derive from path-scrambled noise)
       - Compute light direction `L_light = p - payload.hit_pos`, distance, solid-angle PDF
       - Trace shadow ray; if unoccluded: evaluate BRDF at `L_light`, accumulate `radiance * brdf * NdotL / pdf`
   - **Write results:**
     - `imageStore(img_noisy_diffuse, pixel, vec4(final_color, 1.0))` — all radiance to diffuse this phase
     - `imageStore(img_noisy_specular, pixel, vec4(0.0))` — zero (split deferred to Phase 8B)
     - G-buffer auxiliary outputs: same as Phase 7C for now (zeroed for hit pixels), except:
       - `img_world_normals`: write `vec4(payload.normal, roughness)` for hit pixels (useful for future denoiser even before Phase 8C fills all G-buffer channels)
       - `img_diffuse_albedo`: write `vec4(albedo * (1.0 - metallic), 1.0)` for hit pixels
       - `img_specular_albedo`: write `vec4(F0, 1.0)` for hit pixels
     - Miss pixel G-buffer sentinels: same as Phase 7C

5. Update `raygen.rgen` and `miss.rmiss` payload struct:
   - Both shaders update from `RayPayload` to `HitPayload` (same struct definition)
   - `miss.rmiss` sets `payload.missed = true` as before; other fields are ignored on miss
   - Ensure `layout(location = 0)` payload declaration is consistent across all three shaders

### Verification
- **Integration test:** render `DamagedHelmet.glb` at 4 spp and 256 spp, compute FLIP score — must be below convergence threshold
- **Golden reference test:** render Cornell box at 256 spp, compare against stored reference with FLIP (mean < 0.05)
- Metallic surfaces show environment reflections
- Texture sampling works (base color maps applied correctly)
- Area lights produce visible illumination on the Cornell box (ceiling-mounted quad light)
- MIS reduces variance compared to single-strategy sampling (visual check: less noise at same SPP)
- No validation errors; no NaN/Inf in output

### rtx-chessboard Reference
- [closesthit.rchit](../../rtx-chessboard/shaders/closesthit.rchit): barycentric interpolation, instance custom index decoding, HitPayload structure
- [brdf.glsl](../../rtx-chessboard/shaders/include/brdf.glsl): Cook-Torrance implementation, PI constant
- [sampling.glsl](../../rtx-chessboard/shaders/include/sampling.glsl): importance sampling, CDF sampling, coordinate conversion functions
- [bluenoise.glsl](../../rtx-chessboard/shaders/include/bluenoise.glsl): blue noise hashing
- [mis.glsl](../../rtx-chessboard/shaders/include/mis.glsl): MIS weight computation — note clearcoat dependency stubbed in Phase 8A
- [raygen.rgen](../../rtx-chessboard/shaders/raygen.rgen): material fetch in raygen, texture index decoding, blue noise per-path scrambling, multi-sample loop

---

## Phase 8B: Multi-Bounce MIS + Clear Coat

**Goal:** Extend the raygen shader from Phase 8A's single-bounce direct lighting into a full multi-bounce path tracer with 4-way MIS, Russian roulette, diffuse/specular classification, and clear coat support. Transparency/transmission is deferred to Phase 8C — this phase handles opaque bounces only.

**Source:** rtx-chessboard `shaders/raygen.rgen`, `shaders/include/clearcoat.glsl`, `shaders/include/mis.glsl`

### Design Decisions

- **Closesthit remains geometry-only.** The Phase 8A architecture is unchanged: closesthit populates `HitPayload` (position, normal, UV, material_index) and raygen performs all material fetch, texture sampling, and BRDF evaluation at every bounce. Clearcoat parameters (`clear_coat`, `clear_coat_roughness`) come from `PackedMaterial` fields `metallic_clearcoat.g` and `metallic_clearcoat.b`, read in raygen.

- **Multi-bounce shading in raygen.** Phase 8A evaluated direct lighting at the primary hit only. Phase 8B replaces this with an iterative bounce loop: at each bounce, raygen traces a ray, reads `HitPayload`, fetches the material, evaluates the multilayer BRDF (clearcoat + base), samples a new direction via 4-way MIS, and accumulates throughput. The loop continues until a miss (environment hit), Russian roulette termination, or `max_bounces` is reached. This matches the rtx-chessboard raygen architecture exactly.

- **4-way MIS replaces 3-way.** Phase 8A forced `cc_strength = 0.0` for a 3-way split (diffuse/specular/environment). Phase 8B removes this override, enabling the clearcoat strategy. `calculateSamplingProbabilities` now receives the real `clear_coat` and `clear_coat_roughness` from the material, producing 4-way probabilities (diffuse/specular/clearcoat/environment).

- **Diffuse/specular classification at first opaque bounce.** The first opaque hit's MIS strategy determines the entire path's classification: `STRATEGY_SPECULAR` or `STRATEGY_CLEAR_COAT` → specular path; `STRATEGY_DIFFUSE` or `STRATEGY_ENVIRONMENT` → diffuse path. All subsequent bounce contributions (including environment hits at later bounces) accumulate into the chosen bucket. This matches rtx-chessboard and enables demodulated denoising.

- **G-buffer outputs remain at primary hit only.** G-buffer auxiliary data (depth, normals, albedo, motion vectors) is written once at the first intersection of the first path (bounce 0, path 0), same as Phase 8A. Only the radiance outputs (`noisy_diffuse`, `noisy_specular`) change from Phase 8A's all-to-diffuse to the proper first-bounce-classified split.

- **No transparency this phase.** All surfaces are treated as opaque (`opacity` is ignored). The transparency/refraction loop, alpha masking, and physical transmission are added in Phase 8C. The outer loop bound is simply `max_bounces` (no `+ 8` transparency headroom).

- **Blue noise 4-bounce coverage.** `getBlueNoiseRandom(packed, bounce)` maps bounce indices 0–3 to `uvec4.xyzw`, providing 4 independent random vectors via byte-unpacking. With the default `max_bounces = 4`, bounces 0, 1, 2, 3 are fully covered by distinct blue noise channels. For `bounce >= 4`, `getBlueNoiseRandom` falls back internally to a Wang hash (`wangHashBounceRandoms`), producing decorrelated pseudo-random `vec4` values from a seed derived from `bn_packed.x` and the bounce index. This supports arbitrary `max_bounces` values without silent correlation, at the cost of losing blue noise's spatial stratification for deep bounces (acceptable since Russian roulette terminates most paths before bounce 4). Area light sampling at each bounce also uses `wangHashBounceRandoms` with a seed incorporating `bn_packed.x`, the bounce index, and the light index. Phase 8C's transparency bounces will use the same Wang hash fallback for transparent bounce indices.

### Tasks

1. Port `shaders/include/clearcoat.glsl` from rtx-chessboard:

   - Define `CLEAR_COAT_F0 = vec3(0.04)` (dielectric IOR 1.5: `((1.5−1)/(1.5+1))² = 0.04`)
   - Port `calculateClearCoatAttenuation(float VdotH, float clear_coat_strength)`:
     ```glsl
     float clearcoat_F = F_Schlick(VdotH, CLEAR_COAT_F0).r;
     float transmission = 1.0 - clearcoat_F;
     float transmission2 = transmission * transmission;  // Double pass: entering + exiting
     return mix(1.0, transmission2, clear_coat_strength);
     ```
   - Port `evaluateMultilayerBRDF(albedo, roughness, metallic, F0, clear_coat_strength, clear_coat_roughness, NdotV, NdotL, NdotH, VdotH)`:
     1. Evaluate base PBR: `base_brdf = evaluatePBR(albedo, roughness, metallic, F0, NdotV, NdotL, NdotH, VdotH)`
     2. Early-out if `clear_coat_strength <= 0.0`: return `base_brdf`
     3. Clearcoat specular lobe (Cook-Torrance GGX with `CLEAR_COAT_F0`):
        ```glsl
        float cc_alpha = clear_coat_roughness * clear_coat_roughness;
        float cc_D = D_GGX(NdotH, cc_alpha * cc_alpha);
        float cc_G = G_SmithGGX(NdotV, NdotL, clear_coat_roughness);
        vec3 cc_F = F_Schlick(VdotH, CLEAR_COAT_F0);
        vec3 cc_brdf = (cc_D * cc_G * cc_F) / max(4.0 * NdotV * NdotL, 0.001);
        ```
     4. Energy-conserving combination: `return cc_brdf * clear_coat_strength + base_brdf * calculateClearCoatAttenuation(VdotH, clear_coat_strength)`
   - Use `#ifndef CLEARCOAT_GLSL` / `#define CLEARCOAT_GLSL` / `#endif` guard. `#include "brdf.glsl"`.

2. Update `shaders/include/mis.glsl` — replace Phase 8A clearcoat stub:

   - Replace the inline `CLEAR_COAT_F0` constant and `calculateClearCoatAttenuation` stub with `#include "clearcoat.glsl"`
   - Remove the `cc_strength = 0.0` override in `calculateSamplingProbabilities` so the real `clear_coat` value participates in 4-way probability calculation
   - Add `clear_coat_roughness` parameter to `calculateAllPDFs`:
     ```glsl
     AllPDFs calculateAllPDFs(vec3 N, vec3 V, vec3 L, float roughness, float clear_coat_roughness) {
         AllPDFs pdfs;
         pdfs.diffuse = calculateDiffusePDF(N, L);
         pdfs.specular = calculateGGXPDF(N, V, L, roughness);
         pdfs.clear_coat = calculateGGXPDF(N, V, L, clear_coat_roughness);
         pdfs.environment = 0.0;  // Filled in by caller via environmentCDFPdf
         return pdfs;
     }
     ```
   - Verify the `AllPDFs` struct has the `clear_coat` field (add if Phase 8A omitted it):
     ```glsl
     struct AllPDFs {
         float diffuse;
         float specular;
         float clear_coat;
         float environment;
     };
     ```
   - `calculateMISWeight` already handles 4 strategies via weighted power heuristic — no change needed

3. Implement full bounce loop in `raygen.rgen`:

   Replace the Phase 8A single-bounce shading block with the iterative multi-bounce loop. The outer multi-sample loop and ray generation remain unchanged. The per-path body becomes:

   ```glsl
   vec3 throughput = vec3(1.0);
   vec3 ray_dir = direction;
   vec3 ray_origin = origin;

   int bounce = 0;
   bool first_opaque = true;
   bool is_specular_path = false;

   for (int i = 0; i < max_bounces; ++i) {
       payload.missed = true;

       traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF,
                   0, 0, 0,
                   ray_origin, 0.001, ray_dir, 10000.0, 0);

       // ── Miss: accumulate environment and break ──
       if (payload.missed) {
           vec3 env_color;
           if (bounce == 0)
               env_color = sampleEnvironmentBlurred(
                   env_map, ray_dir, pc.skybox_mip_level, pc.env_rotation);
           else
               env_color = textureLod(
                   env_map, directionToUVRotated(ray_dir, pc.env_rotation), 0.5).rgb;
           path_radiance += throughput * env_color;

           // Write sentinel G-buffer if primary ray missed
           if (!wrote_primary && path == 0) {
               imageStore(img_motion_vectors, pixel, vec4(0.0));
               imageStore(img_linear_depth, pixel, vec4(1e4, 0.0, 0.0, 0.0));
               imageStore(img_world_normals, pixel, vec4(0.0, 0.0, 1.0, 0.0));
               imageStore(img_diffuse_albedo, pixel, vec4(0.0));
               imageStore(img_specular_albedo, pixel, vec4(0.04, 0.04, 0.04, 1.0));
               wrote_primary = true;
           }
           break;
       }

       // ── Hit: fetch material (raw vec4 pattern, matching Phase 8A) ──
       uint mat_base = payload.material_index * 5;
       vec4 base_color_roughness = materials.data[mat_base + 0];
       vec4 metallic_clearcoat   = materials.data[mat_base + 1];

       vec3 albedo  = base_color_roughness.rgb;
       float roughness = max(base_color_roughness.a, 0.04);
       float metallic  = metallic_clearcoat.r;
       float clear_coat = metallic_clearcoat.g;
       float clear_coat_roughness = max(metallic_clearcoat.b, 0.04);

       // Sample base color texture if present
       uint base_color_tex_idx = floatBitsToUint(metallic_clearcoat.a);
       if (base_color_tex_idx != 0xFFFFFFFFu)
           albedo *= texture(bindless_textures[nonuniformEXT(base_color_tex_idx)], payload.uv).rgb;

       // ── Write G-buffer at primary hit (bounce 0, path 0) ──
       if (!wrote_primary && path == 0) {
           // linear depth, world normals + roughness, motion vectors,
           // diffuse albedo, specular albedo — same as Phase 8A
           wrote_primary = true;
       }

       // ── Opaque surface: 4-way MIS ──
       vec3 N = payload.normal;
       vec3 V = -ray_dir;
       bool entering = dot(N, V) > 0.0;
       if (!entering) N = -N;

       float NdotV = max(dot(N, V), 0.001);
       vec3 F0 = mix(vec3(0.04), albedo, metallic);

       // Blue noise randoms for this bounce (Wang hash fallback for bounce >= 4)
       vec4 rands = getBlueNoiseRandom(bn_packed, bounce);

       // ── Calculate 4-way sampling probabilities ──
       SamplingProbabilities probs = calculateSamplingProbabilities(
           N, V, F0, metallic, roughness,
           clear_coat, clear_coat_roughness,       // real values (was 0.0, 0.04 in Phase 8A)
           pc.env_avg_luminance, pc.env_max_luminance);

       int strategy = chooseStrategy(probs, rands.z);

       // ── First-bounce diffuse/specular classification ──
       if (first_opaque) {
           is_specular_path = (strategy == STRATEGY_SPECULAR ||
                               strategy == STRATEGY_CLEAR_COAT);
           first_opaque = false;
       }

       // ── Sample direction based on chosen strategy ──
       vec3 L;
       if (strategy == STRATEGY_DIFFUSE) {
           L = cosineSampleHemisphere(rands.xy, N);
       } else if (strategy == STRATEGY_SPECULAR) {
           vec3 H = sampleGGX(rands.xy, N, roughness);
           L = reflect(-V, H);
       } else if (strategy == STRATEGY_CLEAR_COAT) {
           vec3 H = sampleGGX(rands.xy, N, clear_coat_roughness);
           L = reflect(-V, H);
       } else { // STRATEGY_ENVIRONMENT
           vec2 env_sample_uv = environmentCDFSample(
               rands.xy, marginal_cdf, conditional_cdf, env_size);
           L = rotateDirectionY(uvToDirection(env_sample_uv), -pc.env_rotation);
       }

       float NdotL = max(dot(N, L), 0.0);
       if (NdotL <= 0.0) break;  // Below horizon

       vec3 H = normalize(V + L);
       float NdotH = max(dot(N, H), 0.001);
       float VdotH = max(dot(V, H), 0.001);

       // ── Multilayer BRDF evaluation (clearcoat + base) ──
       vec3 brdf = evaluateMultilayerBRDF(albedo, roughness, metallic, F0,
           clear_coat, clear_coat_roughness,
           NdotV, NdotL, NdotH, VdotH);

       // ── MIS weight computation ──
       AllPDFs pdfs = calculateAllPDFs(N, V, L, roughness, clear_coat_roughness);
       vec2 L_env_uv = directionToUVRotated(L, pc.env_rotation);
       pdfs.environment = environmentCDFPdf(
           L_env_uv, marginal_cdf, conditional_cdf, env_size);

       float chosen_pdf;
       if (strategy == STRATEGY_DIFFUSE) chosen_pdf = pdfs.diffuse;
       else if (strategy == STRATEGY_SPECULAR) chosen_pdf = pdfs.specular;
       else if (strategy == STRATEGY_CLEAR_COAT) chosen_pdf = pdfs.clear_coat;
       else chosen_pdf = pdfs.environment;

       if (chosen_pdf <= 0.0) break;

       float mis_weight = calculateMISWeight(strategy, pdfs, probs);

       // ── Throughput update ──
       throughput *= brdf * NdotL * mis_weight / chosen_pdf;

       // ── Russian roulette after bounce 3 ──
       if (bounce >= 3) {
           float p = max(max(throughput.r, throughput.g), throughput.b);
           if (p < 0.01) break;           // Hard cutoff: negligible contribution
           p = min(p, 0.95);              // Cap survival to prevent fireflies
           if (rands.w >= p) break;        // Stochastic termination
           throughput /= p;               // Unbiased compensation
       }

       // ── Set up next bounce ──
       ray_origin = payload.hit_pos + N * 0.001;
       ray_dir = L;
       bounce++;
   }
   ```

   **Per-area-light shadow rays:** Same structure as Phase 8A — for each area light `i` in `[0, pc.area_light_count)`, sample a point on the quad, trace a shadow ray from `payload.hit_pos`, and accumulate `light_radiance * brdf * NdotL / pdf` if unoccluded. Area light sampling occurs at **every bounce** within the loop, immediately after the MIS direction sample. BRDF evaluation uses `evaluateMultilayerBRDF` (replacing Phase 8A's `evaluatePBR`), so area lights benefit from clearcoat. Area light random values use `wangHashBounceRandoms(bn_packed.x ^ uint(bounce * 16 + li + 4))`, providing decorrelated randoms per bounce and per light without consuming blue noise channels. The contribution is added to `path_radiance` directly (scaled by `throughput`), not folded into `throughput`.

   **Radiance accumulation after the bounce loop:**
   ```glsl
   // Accumulate into diffuse or specular based on first-bounce classification
   if (is_specular_path)
       total_specular += path_radiance;
   else
       total_diffuse += path_radiance;
   ```

   **Final output (after all paths):**
   ```glsl
   float inv_paths = 1.0 / float(paths_per_pixel);
   vec3 final_diffuse = total_diffuse * inv_paths;
   vec3 final_specular = total_specular * inv_paths;

   imageStore(img_noisy_diffuse, pixel, vec4(final_diffuse, 1.0));
   imageStore(img_noisy_specular, pixel, vec4(final_specular, 1.0));
   ```

4. Update `bluenoise.glsl` — add Wang hash fallback for arbitrary bounce counts:

   - Add `wangHash(uint seed)` and `wangHashBounceRandoms(uint seed)` functions that produce decorrelated pseudo-random values using a Wang hash, for use when `bounce >= 4` exceeds the 4 blue noise channels and for area light sampling:
     ```glsl
     uint wangHash(uint seed) {
         seed = (seed ^ 61u) ^ (seed >> 16u);
         seed *= 9u;
         seed ^= seed >> 4u;
         seed *= 0x27d4eb2du;
         seed ^= seed >> 15u;
         return seed;
     }

     vec4 wangHashBounceRandoms(uint seed) {
         uint h0 = wangHash(seed);
         uint h1 = wangHash(h0);
         uint h2 = wangHash(h1);
         uint h3 = wangHash(h2);
         return vec4(float(h0), float(h1), float(h2), float(h3)) / 4294967295.0;
     }
     ```
   - Update `getBlueNoiseRandom` bounce >= 4 fallback: replace the current XOR-mixing approach (`packed.x ^ packed.y ^ packed.z ^ mixHash`) with `wangHashBounceRandoms(packed.x ^ uint(bounce) * 0x9E3779B9u)`. The existing `extractBounceRandoms` byte-unpacker remains unchanged for bounces 0–3.
   - Area light sampling at each bounce uses `wangHashBounceRandoms(bn_packed.x ^ uint(bounce * 16 + li + 4))` — the `* 16` spread and `+ 4` offset ensure decorrelation from MIS direction randoms.
   - > **Known limitation:** For `bounce >= 4` and all area light samples, random values lose blue noise spatial stratification and degrade to white noise. This is acceptable because Russian roulette terminates most paths before bounce 4 (survival probability drops per bounce), and the throughput of surviving deep paths is low — white noise variance is negligible relative to the path's diminished energy contribution. If a future phase raises `max_bounces` significantly (e.g., for caustics), a larger blue noise table (8+ channels) or a Cranley-Patterson rotation scheme should be evaluated.

5. No changes to `closesthit.rchit`:

   Closesthit remains geometry-only as established in Phase 8A. All material fetch (including clearcoat parameters) and BRDF evaluation happen in raygen at every bounce. The `HitPayload` struct is unchanged.

### Verification
- **Convergence test:** Cornell box at 4 spp vs 256 spp, FLIP score below threshold (validates multi-bounce GI convergence — color bleeding from red/green walls requires 2+ bounces)
- **Golden reference test:** Cornell box at 256 spp matches stored reference (FLIP mean < 0.05)
- Metallic surfaces show recursive environment reflections (multiple bounces visible — confirms bounce loop works)
- Clear coat shows dual-layer effect (`ClearCoatTest.glb` renders correctly — glossy clearcoat over matte base)
- Diffuse/specular split: `noisy_diffuse` contains diffuse-classified paths, `noisy_specular` contains specular/clearcoat-classified paths, sum equals total radiance
- Russian roulette terminates paths without visible bias (compare RR-enabled vs disabled at high SPP — FLIP < threshold)
- Area lights produce correct multi-bounce illumination (light bouncing off walls illuminates ceiling)
- No validation errors; no NaN/Inf in output
- `max_bounces > 4` works correctly (hash fallback produces valid random values without NaN)

### rtx-chessboard Reference
- [raygen.rgen](../../rtx-chessboard/shaders/raygen.rgen): full path tracing loop with iterative bounce, 4-way MIS, Russian roulette, diffuse/specular classification, G-buffer writes at primary hit
- [clearcoat.glsl](../../rtx-chessboard/shaders/include/clearcoat.glsl): `CLEAR_COAT_F0`, `calculateClearCoatAttenuation`, `evaluateMultilayerBRDF`
- [mis.glsl](../../rtx-chessboard/shaders/include/mis.glsl): `calculateSamplingProbabilities` (4-way), `calculateAllPDFs` (with `clear_coat_roughness` param), `calculateMISWeight` (power heuristic β=2), `chooseStrategy` (CDF selection)

---

## Phase 8C: Transparency + Transmission + G-Buffer Completion + Sub-pixel Jitter

**Goal:** Add alpha masking (via any-hit shader), alpha-blend transparency, physical Fresnel transmission with IOR, thin-slab volume attenuation, complete all G-buffer auxiliary writes with explicit formulas, and implement per-frame sub-pixel jitter via projection-matrix perturbation.

**Source:** rtx-chessboard `shaders/raygen.rgen` (transparency loop, G-buffer writes, jitter application), `shaders/closesthit.rchit` (geometry-only payload), `src/render/camera.cpp` (Halton sequence, jittered projection), `src/render/hw_path_tracer.cpp` (G-buffer images, push constants population)

### Design Decisions

- **PackedMaterial extended to 6 vec4 (96 bytes).** `alpha_mode` (float-encoded `uint32_t`: 0 = kOpaque, 1 = kMask, 2 = kBlend) and `alpha_cutoff` (float) are added in a new 6th vec4. Shader material addressing changes from `material_index * 5` to `material_index * 6`. This adds 16 bytes per material but keeps the encoding clear and avoids bit-packing into existing fields. The new vec4's `.b` and `.a` are reserved padding.

- **Any-hit shader for `AlphaMode::kMask`.** A new `anyhit.rahit` shader handles alpha masking during hardware ray traversal. For `kMask` materials, the any-hit shader fetches the material's `alpha_cutoff` and base color texture alpha, calling `ignoreIntersectionEXT()` for fragments below the cutoff. This ensures the acceleration structure correctly finds the first opaque surface behind masked geometry (foliage, fences). The SBT hit group is updated to include the any-hit shader. To preserve Phase 8B performance for fully opaque geometry, `VK_GEOMETRY_OPAQUE_BIT` is set on BLAS geometry instances whose material `alpha_mode == kOpaque`, telling the hardware to skip any-hit invocation for those triangles.

- **Alpha blending (`AlphaMode::kBlend`) handled in raygen.** Unlike `kMask` (which filters during traversal), alpha-blend transparency is resolved in the raygen shader after closest-hit returns. A stochastic alpha test (`random > opacity`) determines whether the ray passes through or treats the surface as opaque. Pass-through rays continue in the same direction without refraction and attenuate throughput by `albedo`. This avoids the complexity of ordered transparency and produces unbiased results with sufficient SPP.

- **Physical Fresnel transmission separate from alpha blending.** Materials with `transmission_factor > 0.0` undergo Fresnel reflection/refraction using proper IOR via Schlick's approximation and GLSL `refract()`. This is a distinct code path from alpha blending: alpha blend is for fade/ghost effects (no refraction), while transmission is for glass/water (refraction with IOR). A material with both `opacity < 1.0` AND `transmission_factor > 0.0` tests alpha blend first; if the ray passes through, it then applies Fresnel transmission.

- **Thin-slab volume attenuation (Beer-Lambert).** For transmitted rays, attenuation is computed using the glTF `KHR_materials_volume` thin-slab approximation: `path_length = thickness_factor / max(|N·V|, 0.001)`, then `throughput *= exp(-path_length * sigma)` where `sigma = -log(attenuation_color) / attenuation_distance`. This provides view-angle-dependent absorption without tracking medium enter/exit state. Full volumetric tracking (enter/exit state across bounces, distance accumulation through nested media) is deferred to a future phase — see "Deferred to Future Phases" below.

- **Transparency bounces do not consume the opaque bounce counter.** The outer loop runs for `max_bounces + 8` iterations, but `bounce` only increments on opaque hits. `if (bounce >= max_bounces) break` respects the opaque limit. Transparent surfaces `continue` without incrementing `bounce`, and a separate `transparent_count` tracks transparency iterations. Blue noise for transparent bounces uses index `max_bounces + transparent_count` to avoid conflicts with opaque bounce random channels.

- **G-buffer written at first non-fully-transparent hit.** Unless the primary surface has `opacity == 0.0` (fully transparent), it is the first-hit surface for G-buffer output. Partially transparent surfaces (`0.0 < opacity < 1.0`) and specular transmission surfaces still write G-buffer data (depth, normals, motion vectors, albedo) from their first intersection, providing stable data for the denoiser. For `kMask` materials, the any-hit shader guarantees the first closest-hit result is above the alpha cutoff, so G-buffer data is always valid.

- **Sub-pixel jitter via projection-matrix perturbation.** Halton sequence (base 2, 3) with a 16-frame period generates per-frame sub-pixel offsets in [-0.5, 0.5] pixel space. The C++ side applies the offset to the projection matrix (`proj[2][0]` and `proj[2][1]`), then passes `glm::inverse(jittered_proj)` as `inv_proj` in push constants. The raygen shader generates rays through the jittered inverse projection — no shader-side changes are needed for ray generation. `prev_view_proj` stores the **non-jittered** previous-frame view-projection matrix for clean motion vectors. `jitter_x` and `jitter_y` in push constants are populated with the raw pixel-space offsets; they are not consumed by ray generation (inv_proj embeds the jitter) or motion vectors (prev_view_proj is non-jittered), but are available for debug visualization and will be forwarded to the denoiser API when an ML denoiser is integrated in a future phase.

- **`double_sided` handling deferred.** The `double_sided` flag in `MaterialDesc` is not packed into `PackedMaterial` or consumed by shaders in this phase. Currently, all surfaces flip normals to face the ray (`dot(N, V) > 0.0`). Proper single-sided culling (rejecting backface hits for non-double-sided surfaces) is deferred to a future phase.

- **Per-ray sub-pixel offset deferred.** Phase 8C implements per-frame projection jitter only (all pixels shift by the same sub-pixel offset). Per-ray sub-pixel AA offsets (each path within a pixel gets an independent sub-pixel offset from blue noise) can be added in a future phase after the denoiser is verified to handle the additional variance correctly.

### Tasks

1. Extend `PackedMaterial` to 6 vec4 (96 bytes):

   Add a 6th vec4 for alpha parameters:
   ```cpp
   struct alignas(16) PackedMaterial {
       glm::vec4 base_color_roughness;   // .rgb = base_color, .a = roughness
       glm::vec4 metallic_clearcoat;     // .r = metallic, .g = clear_coat,
                                         // .b = clear_coat_roughness,
                                         // .a = base_color_map index
       glm::vec4 opacity_ior;            // .r = opacity, .g = ior,
                                         // .b = normal_map index,
                                         // .a = metallic_roughness_map index
       glm::vec4 transmission_volume;    // .r = transmission_factor, .g = thickness,
                                         // .b = attenuation_distance,
                                         // .a = transmission_map index
       glm::vec4 attenuation_color_pad;  // .rgb = attenuation_color,
                                         // .a = emissive_map index
       glm::vec4 alpha_mode_misc;        // .r = alpha_mode (float-encoded uint: 0/1/2),
                                         // .g = alpha_cutoff,
                                         // .b = reserved,
                                         // .a = reserved
   };

   static_assert(sizeof(PackedMaterial) == 96);
   ```

   Update `GpuScene::PackMaterial()` to populate the new vec4 from `MaterialDesc::alpha_mode` and `MaterialDesc::alpha_cutoff`. Update all shader material fetch code: `material_index * 5` → `material_index * 6`.

2. Implement any-hit shader (`shaders/anyhit.rahit`):

   ```glsl
   #version 460
   #extension GL_EXT_ray_tracing : require
   #extension GL_EXT_nonuniform_qualifier : require
   #extension GL_EXT_buffer_reference2 : require
   #extension GL_EXT_scalar_block_layout : require
   #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
   #extension GL_GOOGLE_include_directive : enable

   #include "include/vertex.glsl"

   // ── Descriptor bindings (same indices as raygen/closesthit) ──
   layout(set = 0, binding = 8, scalar) readonly buffer MeshAddressTable {
       MeshAddressEntry entries[];
   } mesh_address_table;
   layout(set = 0, binding = 9, std430) readonly buffer MaterialBuffer { vec4 data[]; } materials;
   layout(set = 0, binding = 10) uniform sampler2D bindless_textures[];

   hitAttributeEXT vec2 hit_attribs;

   const uint kMeshAddrIndexBits = 12u;
   const uint kMeshAddrIndexMask = (1u << kMeshAddrIndexBits) - 1u;

   void main() {
       uint custom_index = gl_InstanceCustomIndexEXT;
       uint mesh_addr_index = custom_index & kMeshAddrIndexMask;
       uint material_index = (custom_index >> kMeshAddrIndexBits) & kMeshAddrIndexMask;

       uint mat_base = material_index * 6;
       uint alpha_mode = floatBitsToUint(materials.data[mat_base + 5].r);
       if (alpha_mode != 1u) return;  // Only process kMask (1)

       float alpha_cutoff = materials.data[mat_base + 5].g;

       // Check for base color texture
       uint base_color_tex_idx = floatBitsToUint(materials.data[mat_base + 1].a);
       if (base_color_tex_idx == 0xFFFFFFFFu) return;  // No texture → accept hit

       // Interpolate UV (same pattern as closesthit)
       MeshAddressEntry entry = mesh_address_table.entries[mesh_addr_index];
       Vertex v0, v1, v2;
       fetchTriangleVertices(entry, gl_PrimitiveID, v0, v1, v2);
       vec3 bary = vec3(1.0 - hit_attribs.x - hit_attribs.y,
                        hit_attribs.x, hit_attribs.y);
       vec2 uv = v0.tex_coord_0 * bary.x + v1.tex_coord_0 * bary.y
               + v2.tex_coord_0 * bary.z;

       float alpha = texture(bindless_textures[nonuniformEXT(base_color_tex_idx)], uv).a;
       if (alpha < alpha_cutoff)
           ignoreIntersectionEXT();
   }
   ```

   **SBT update:** The ray tracing pipeline's hit group changes from (closesthit-only) to (closesthit + anyhit). Update `RaytracePipeline` to include the any-hit shader stage in `VkRayTracingShaderGroupCreateInfoKHR`.

   **Ray flags update:** Phase 8B uses `gl_RayFlagsOpaqueEXT` which skips any-hit invocation. Phase 8C changes this to `gl_RayFlagsNoneEXT` so the any-hit shader runs. `VK_GEOMETRY_OPAQUE_BIT` on opaque BLAS instances still skips the any-hit shader for opaque geometry, preserving Phase 8B performance.

3. Restructure the bounce loop for transparency in `raygen.rgen`:

   Replace the Phase 8B bounce loop with the following structure. Changes from Phase 8B are marked with `// ← 8C`.

   ```glsl
   vec3 throughput = vec3(1.0);
   vec3 path_radiance = vec3(0.0);
   vec3 ray_dir = direction;
   vec3 ray_origin = origin;

   int bounce = 0;
   int transparent_count = 0;                              // ← 8C
   bool first_opaque = true;
   bool is_specular_path = false;

   for (int i = 0; i < max_bounces + 8; ++i) {            // ← 8C: +8 transparency headroom
       if (bounce >= max_bounces) break;                   // ← 8C: opaque limit check

       payload.missed = true;
       traceRayEXT(tlas, gl_RayFlagsNoneEXT, 0xFF,        // ← 8C: NoneEXT (was OpaqueEXT)
                   0, 0, 0,
                   ray_origin, 0.001, ray_dir, 10000.0, 0);

       // ── Miss: accumulate environment and break ──
       if (payload.missed) {
           vec3 env_color;
           if (bounce == 0 && transparent_count == 0)      // ← 8C: check transparent_count
               env_color = sampleEnvironmentBlurred(
                   env_map, ray_dir, pc.skybox_mip_level, pc.env_rotation);
           else
               env_color = textureLod(
                   env_map, directionToUVRotated(ray_dir, pc.env_rotation), 0.5).rgb;
           path_radiance += throughput * env_color;

           // Write sentinel G-buffer if nothing was hit (path 0 only)
           if (!wrote_primary && path == 0) {
               imageStore(img_motion_vectors, pixel, vec4(0.0));
               imageStore(img_linear_depth, pixel, vec4(1e4, 0.0, 0.0, 0.0));
               imageStore(img_world_normals, pixel, vec4(0.0, 0.0, 1.0, 0.0));
               imageStore(img_diffuse_albedo, pixel, vec4(0.0));
               imageStore(img_specular_albedo, pixel, vec4(0.04, 0.04, 0.04, 1.0));
               wrote_primary = true;
           }
           break;
       }

       // ── Hit: fetch material (6 vec4 per material) ──           // ← 8C: was 5
       uint mat_base = payload.material_index * 6;
       vec4 base_color_roughness = materials.data[mat_base + 0];
       vec4 metallic_clearcoat   = materials.data[mat_base + 1];
       vec4 opacity_ior_tex      = materials.data[mat_base + 2];
       vec4 transmission_volume  = materials.data[mat_base + 3];          // ← 8C
       vec4 attenuation_color_p  = materials.data[mat_base + 4];          // ← 8C

       vec3 albedo     = base_color_roughness.rgb;
       float roughness = max(base_color_roughness.a, 0.04);
       float metallic  = metallic_clearcoat.r;
       float clear_coat = metallic_clearcoat.g;
       float clear_coat_roughness = max(metallic_clearcoat.b, 0.04);

       float opacity      = opacity_ior_tex.r;                            // ← 8C
       float ior          = opacity_ior_tex.g;                            // ← 8C
       float transmission = transmission_volume.r;                        // ← 8C
       float thickness    = transmission_volume.g;                        // ← 8C
       float atten_dist   = transmission_volume.b;                        // ← 8C
       vec3 atten_color   = attenuation_color_p.rgb;                      // ← 8C

       // Sample base color texture if present
       uint base_color_tex_idx = floatBitsToUint(metallic_clearcoat.a);
       vec4 tex_sample = vec4(1.0);
       if (base_color_tex_idx != 0xFFFFFFFFu)
           tex_sample = texture(
               bindless_textures[nonuniformEXT(base_color_tex_idx)], payload.uv);
       albedo *= tex_sample.rgb;

       vec3 N = payload.normal;
       vec3 V = -ray_dir;
       bool entering = dot(N, V) > 0.0;
       if (!entering) N = -N;
       float NdotV = max(dot(N, V), 0.001);

       // ── Write G-buffer at first non-fully-transparent hit ──    // ← 8C
       // Partially transparent and transmission surfaces still write G-buffer.
       // Only fully transparent (opacity == 0.0) surfaces are skipped.
       // For kMask, the any-hit shader guarantees above-cutoff hits only.
       if (!wrote_primary && path == 0 && opacity > 0.0) {
           // Linear depth: signed view-space distance
           vec3 cam_forward = normalize(
               (pc.inv_view * vec4(0.0, 0.0, -1.0, 0.0)).xyz);
           float linear_depth = dot(cam_forward,
               payload.hit_pos - primary_origin);
           imageStore(img_linear_depth, pixel,
               vec4(linear_depth, 0.0, 0.0, 0.0));

           // World normals (.xyz) + roughness (.w)
           imageStore(img_world_normals, pixel,
               vec4(payload.normal, roughness));

           // Motion vectors: current screen pos minus reprojected previous pos
           vec2 screen_current = (vec2(pixel) + 0.5) / vec2(size);
           vec4 clip_prev = pc.prev_view_proj * vec4(payload.hit_pos, 1.0);
           vec2 screen_prev = clip_prev.xy / clip_prev.w * 0.5 + 0.5;
           vec2 motion = screen_current - screen_prev;
           imageStore(img_motion_vectors, pixel,
               vec4(motion, 0.0, 0.0));

           // Diffuse albedo: non-metallic portion of base color
           imageStore(img_diffuse_albedo, pixel,
               vec4(albedo * (1.0 - metallic), 1.0));

           // Specular albedo: Fresnel F0
           imageStore(img_specular_albedo, pixel,
               vec4(mix(vec3(0.04), albedo, metallic), 1.0));

           wrote_primary = true;
       }

       // ── Alpha-blend pass-through (kBlend, opacity < 1.0) ──    // ← 8C
       // Stochastic alpha: randomly pass through based on opacity.
       // Pass-through rays continue without refraction, attenuated by albedo.
       if (opacity < 1.0 && opacity > 0.0) {
           vec4 rands = getBlueNoiseRandom(
               bn_path, max_bounces + transparent_count);
           if (rands.x > opacity) {
               throughput *= albedo;
               transparent_count++;
               ray_origin = payload.hit_pos + ray_dir * 0.001;
               continue;
           }
           // else: treat as opaque, fall through to MIS
       }

       // ── Specular transmission (transmission_factor > 0.0) ──   // ← 8C
       // Fresnel reflection/refraction with IOR, plus thin-slab attenuation.
       if (transmission > 0.0) {
           vec4 rands = getBlueNoiseRandom(
               bn_path, max_bounces + transparent_count);
           float cos_i = max(dot(N, V), 0.001);

           // Dielectric Fresnel: F0 = ((n1-n2)/(n1+n2))²
           float n1 = entering ? 1.0 : ior;
           float n2 = entering ? ior : 1.0;
           float f0 = (n1 - n2) / (n1 + n2);
           f0 = f0 * f0;
           float fresnel = f0 + (1.0 - f0) * pow(1.0 - cos_i, 5.0);

           // Modulate reflection probability by transmission factor
           float reflect_prob = mix(1.0, fresnel, transmission);

           if (rands.x < reflect_prob) {
               // Reflect
               ray_dir = reflect(-V, N);
           } else {
               // Refract (Snell's law)
               float eta = n1 / n2;
               vec3 refracted = refract(-V, N, eta);
               if (length(refracted) < 0.001) {
                   ray_dir = reflect(-V, N);  // Total internal reflection
               } else {
                   ray_dir = normalize(refracted);

                   // Thin-slab volume attenuation (Beer-Lambert)
                   if (atten_dist > 0.0 && thickness > 0.0) {
                       float path_length = thickness
                           / max(abs(dot(N, V)), 0.001);
                       vec3 sigma = -log(max(atten_color, vec3(0.001)))
                           / atten_dist;
                       throughput *= exp(-sigma * path_length);
                   }
               }
           }

           transparent_count++;
           ray_origin = payload.hit_pos + ray_dir * 0.001;
           continue;
       }

       // ── Fully transparent surface (opacity == 0.0): pass through ──
       if (opacity <= 0.0) {
           transparent_count++;
           ray_origin = payload.hit_pos + ray_dir * 0.001;
           continue;
       }

       // ── Opaque surface: 4-way MIS (unchanged from Phase 8B) ──
       vec3 F0 = mix(vec3(0.04), albedo, metallic);
       vec4 rands = getBlueNoiseRandom(bn_packed, bounce);

       // ... (Phase 8B: calculateSamplingProbabilities, chooseStrategy,
       //      first-bounce classification, direction sampling,
       //      evaluateMultilayerBRDF, calculateAllPDFs, calculateMISWeight,
       //      throughput update, Russian roulette, area light sampling) ...

       ray_origin = payload.hit_pos + N * 0.001;
       ray_dir = L;
       bounce++;                          // Only opaque bounces increment
   }
   ```

   **Radiance accumulation after the bounce loop** is unchanged from Phase 8B:
   ```glsl
   if (is_specular_path)
       total_specular += path_radiance;
   else
       total_diffuse += path_radiance;
   ```

4. Implement sub-pixel jitter on the C++ side:

   Add a Halton sequence utility (van der Corput base-2, 3) matching rtx-chessboard's `Camera::JitterOffset`:

   ```cpp
   glm::vec2 HaltonJitter(uint32_t frame_index) {
       auto van_der_corput = [](uint32_t index, uint32_t base) -> float {
           float inv_base = 1.0f / static_cast<float>(base);
           float result = 0.0f;
           float fraction = inv_base;
           uint32_t n = index;
           while (n > 0) {
               result += static_cast<float>(n % base) * fraction;
               n /= base;
               fraction *= inv_base;
           }
           return result;
       };
       auto h = glm::vec2(
           van_der_corput((frame_index % 16) + 1, 2),
           van_der_corput((frame_index % 16) + 1, 3));
       return h - glm::vec2(0.5f);  // Center to [-0.5, 0.5] pixel space
   }
   ```

   Apply jitter to the projection matrix before inverting, and store the non-jittered VP for the next frame:

   ```cpp
   glm::vec2 jitter = HaltonJitter(frame_index);

   // Jittered projection → inv_proj for ray generation
   glm::mat4 proj = camera.ProjectionMatrix(aspect);
   proj[2][0] += jitter.x * 2.0f / static_cast<float>(width);
   proj[2][1] += jitter.y * 2.0f / static_cast<float>(height);

   PushConstants pc{};
   pc.inv_view = glm::inverse(view);
   pc.inv_proj = glm::inverse(proj);               // Jittered inverse
   pc.prev_view_proj = previous_non_jittered_vp_;  // Non-jittered for motion vectors
   pc.jitter_x = jitter.x;
   pc.jitter_y = jitter.y;

   // Store non-jittered VP for next frame's prev_view_proj
   previous_non_jittered_vp_ = camera.ProjectionMatrix(aspect) * camera.ViewMatrix();
   ```

   The raygen shader requires **no changes** for jitter — it already generates rays via `pc.inv_proj * vec4(ndc, 1.0, 1.0)`, which now produces jittered ray directions automatically. Motion vectors use `pc.prev_view_proj` (non-jittered), so they represent true 3D surface motion without per-frame jitter artifacts.

   > `jitter_x` and `jitter_y` are populated for debug visualization and future ML denoiser consumption. They are not used by ray generation (inv_proj embeds the jitter) or motion vector computation (prev_view_proj is non-jittered). When an ML denoiser is integrated, these values will be forwarded through the `DenoiserInput` API.

5. Update `RaytracePipeline` for any-hit shader integration:

   - Compile `shaders/anyhit.rahit` to SPIR-V (add to CMake shader compilation list)
   - Add `VK_SHADER_STAGE_ANY_HIT_BIT_KHR` shader stage to the hit group in `VkRayTracingShaderGroupCreateInfoKHR`
   - The any-hit shader shares descriptor bindings with raygen/closesthit: mesh address table (binding 8), material buffer (binding 9), bindless textures (binding 10)
   - Update `stageFlags` in `RaytracePipeline.cpp` for bindings 8, 9, and 10 to include `VK_SHADER_STAGE_ANY_HIT_BIT_KHR` (they already include `CLOSEST_HIT_BIT` and/or `RAYGEN_BIT`)

6. Set `VK_GEOMETRY_OPAQUE_BIT` on opaque BLAS geometry (performance):

   During BLAS build, set `VK_GEOMETRY_OPAQUE_BIT` on geometry instances whose material `alpha_mode == kOpaque`. This tells the hardware to skip any-hit shader invocation for these triangles, preserving Phase 8B performance for fully opaque geometry. Only `kMask` materials invoke the any-hit shader.

7. No changes to `closesthit.rchit`:

   Closesthit remains geometry-only as established in Phase 8A. All material fetch, transparency decisions, and BRDF evaluation continue to happen in raygen. The `HitPayload` struct is unchanged.

### Verification
- **Integration test:** render `DragonAttenuation.glb` — transparent dragon shows correct Fresnel refraction and IOR bending. Thin-slab attenuation produces view-angle-dependent color shifts (note: fully accurate thickness-dependent absorption through complex enclosed geometry requires full volumetric tracking, deferred to a future phase)
- Alpha-masked materials (foliage, fences) show correct cutout holes with no false occlusion behind them
- Alpha-blend materials fade correctly with stochastic pass-through (no hard edges at high SPP)
- `AlphaMode::kOpaque` geometry renders identically to Phase 8B (any-hit not invoked due to `VK_GEOMETRY_OPAQUE_BIT`)
- **Convergence test:** Cornell box with transparency at 4 spp vs 256 spp, FLIP below threshold
- **Motion vectors test:** camera dolly sequence, verify pixel deltas match expected 3D point movement (non-jittered `prev_view_proj` produces clean vectors without jitter artifacts)
- Linear depth via `dot(cam_forward, hit_pos - origin)` where `cam_forward = normalize((inv_view * vec4(0, 0, -1, 0)).xyz)` — monotonically increases with distance from camera
- Diffuse albedo = `albedo * (1.0 - metallic)` — pure metals produce black diffuse albedo
- Specular albedo = `mix(vec3(0.04), albedo, metallic)` — dielectrics show 0.04, metals show base color
- World normals `.xyz` match expected surface orientation, `.w` stores roughness
- Sub-pixel jitter: edges smooth over 16-frame accumulation; Halton pattern visible in frame-by-frame jitter offset visualization
- G-buffer stability: partially transparent surfaces write G-buffer from first hit; fully transparent surfaces (opacity = 0) defer to next hit
- Specular transmission: glass/water surfaces refract correctly; total internal reflection at grazing angles for high-IOR materials
- No NaN/Inf values in any G-buffer channel or radiance output
- No Vulkan validation errors

### Deferred to Future Phases
- **Full volumetric tracking:** Enter/exit medium state tracking across bounces, distance accumulation through nested media, correct absorption for complex enclosed geometry (e.g., `MosquitoInAmber.glb`-style nested transmission). Current phase uses thin-slab approximation only. See Phase 8I (nested dielectric priority) and Phase F4 (volume enhancements).
- **`double_sided` surface handling:** Proper single-sided culling (rejecting backface hits for non-double-sided materials). Currently all surfaces flip normals to face the ray.
- **Per-ray sub-pixel offset:** Each path within a pixel gets an independent sub-pixel offset from blue noise, providing intra-frame AA. Deferred until denoiser compatibility is verified.
- **Opacity micromap (`VK_EXT_opacity_micromap`):** Hardware-accelerated alpha masking. The current any-hit approach is correct but slower for dense foliage.
- **Emissive importance sampling:** Direct emissive surface rendering is implemented in Phase 8D. Emissive mesh extraction for NEE sampling is Phase 8J. Full ReSTIR-based importance sampling is deferred to future phase F3.

### rtx-chessboard Reference
- [raygen.rgen](../../rtx-chessboard/shaders/raygen.rgen): transparency loop (lines 204–236), G-buffer writes (lines 130–188), Halton jitter application (lines 76–81)
- [closesthit.rchit](../../rtx-chessboard/shaders/closesthit.rchit): geometry-only payload (no alpha handling in closest-hit)
- [camera.cpp](../../rtx-chessboard/src/render/camera.cpp): Halton sequence (lines 110–149), jittered projection matrix, non-jittered prev_view_proj storage
- [hw_path_tracer.cpp](../../rtx-chessboard/src/render/hw_path_tracer.cpp): G-buffer image creation (lines 1097–1126), push constants population with jitter values

---

## Phase 8D: PBR Texture Sampling + Normal Mapping + Emissive + MIS Fix

**Goal:** Complete PBR material fidelity by sampling all texture-mapped material channels (metallic-roughness, normal, transmission, emissive), constructing TBN matrices for normal mapping, rendering direct emissive surface contribution, fixing the MIS weight estimator, and replacing all magic numbers with named constants.

**Source:** rtx-chessboard `shaders/raygen.rgen` (texture sampling), `shaders/closesthit.rchit` (TBN interpolation), glTF 2.0 specification (PBR texture channels)

### Design Decisions

- **Named constants extracted to `constants.glsl`.** All magic numbers across the shader library are consolidated into a shared include file. This improves readability and ensures consistency when the same value (e.g., the BRDF denominator floor of 0.001) is used in multiple files. Constants use `kPascalCase` naming per project convention.

- **PackedMaterial extended to 7 vec4 (112 bytes).** A new 7th vec4 stores `emissive_factor` (.rgb) and `emissive_strength` (.a). The existing `alpha_mode_misc.b` (previously reserved) now stores `normal_scale`. Material stride in shaders changes from 6 to 7 (`kMaterialStride`).

- **MIS weight fix: divide by strategy selection probability.** The correct multi-strategy one-sample MIS estimator is `f(x) * w_j(x) / (c_j * p_j(x))`, where `c_j` is the probability of selecting strategy `j` and `p_j(x)` is the PDF of that strategy. The prior code divided only by `p_j(x)` (`chosen_pdf`), omitting `c_j` (`chosen_prob`). The fix adds `chosen_prob` to the throughput divisor. The `calculateMISWeight` function itself is correct — it already weights by `c_j * p_j` in the power heuristic numerator and denominator. Note: rtx-chessboard has the same theoretical issue.

- **Metallic-roughness map sampling.** The glTF `metallicRoughnessTexture` stores roughness in the green channel and metallic in the blue channel. The sampled values multiply the base `roughness` and `metallic` material factors. Roughness is clamped to `kMinRoughness` after texture application to avoid GGX singularities.

- **Normal map via TBN from closesthit.** Closesthit now interpolates vertex tangents barycentrically and transforms them to world space, passing `tangent` (vec3) and `tangent_w` (float, bitangent sign) through `HitPayload`. The raygen shader constructs the full TBN matrix using Gram-Schmidt re-orthogonalization, decodes the tangent-space normal from the normal map texture, scales .xy by `normal_scale`, and transforms to world space. This perturbed normal is used for all subsequent BRDF evaluation and lighting.

- **Transmission map sampling.** The `KHR_materials_transmission` texture (.r channel) multiplies the base `transmission_factor`, enabling spatial variation of transmission across a surface.

- **Emissive: direct contribution only.** Emissive surfaces add `emissive_factor * emissive_strength * emissive_texture` to path radiance at each hit, scaled by throughput. This renders self-luminous surfaces correctly when viewed directly or via bounces. Importance sampling of emissive geometry as light sources (required for efficient indirect illumination from arbitrary emissive meshes) is deferred to ReSTIR (Phase F3).

- **Deferred:** Double-sided material handling, second UV set (`tex_coord_1`), `KHR_materials_specular/sheen/anisotropy/iridescence/unlit`, occlusion texture.

### Tasks

1. Create `shaders/include/constants.glsl` — shared named constants:

   All magic numbers from the shader library consolidated into one include file:
   - Ray tracing: `kRayTMin` (0.001), `kRayTMax` (10000.0), `kSurfaceBias` (0.001), `kShadowRayBias` (0.002)
   - BRDF: `kMinCosTheta` (0.001), `kMinRoughness` (0.04), `kMinGGXAlpha2` (0.0002²), `kDielectricF0` (0.04), `kBRDFDenomFloor` (0.001)
   - Russian roulette: `kRussianRouletteStartBounce` (3), `kRussianRouletteMinThroughput` (0.01), `kRussianRouletteMaxSurvival` (0.95)
   - Transparency: `kMaxTransparencyBounces` (8), `kTIRThreshold` (1e-6)
   - MIS: `kMinStrategyProb` (0.03), `kMaxRoughnessBoost` (0.6), `kEnvRoughnessBoostFactor` (2.0), `kEnvFresnelBoostFactor` (0.5), `kEnvDynamicRangeScale` (10.0), `kMinDynamicRangeBoost` (0.1), `kMaxDynamicRangeBoost` (1.0)
   - Texture: `kNoTexture` (0xFFFFFFFFu)
   - Layout: `kMaterialStride` (7u), `kAreaLightStride` (4u)
   - Environment: `kEnvMapBounceLod` (0.5), `kSentinelDepth` (1e4)
   - Sampling: `kONBUpThreshold` (0.999), `kDiagonalSpread` (0.707)

2. Update all shader includes to use named constants:
   - `brdf.glsl`: `#include "constants.glsl"`, use `kMinGGXAlpha2`, `kMinCosTheta`, `kBRDFDenomFloor`
   - `mis.glsl`: use `kMinCosTheta`, `kMinStrategyProb`, `kMaxRoughnessBoost`, `kEnvRoughnessBoostFactor`, `kEnvFresnelBoostFactor`, `kEnvDynamicRangeScale`, `kMinDynamicRangeBoost`, `kMaxDynamicRangeBoost`
   - `sampling.glsl`: `#include "constants.glsl"`, use `kONBUpThreshold`, `kMinCosTheta`, `kDiagonalSpread`
   - `clearcoat.glsl`: use `kBRDFDenomFloor`
   - `anyhit.rahit`: `#include "include/constants.glsl"`, use `kMaterialStride`, `kNoTexture`

3. Extend `PackedMaterial` to 7 vec4 (112 bytes):
   - Add `glm::vec4 emissive` (.rgb = emissive_factor, .a = emissive_strength)
   - Change `alpha_mode_misc.b` from reserved to `normal_scale`
   - Update `static_assert(sizeof(PackedMaterial) == 112)`
   - Update `GpuScene::UpdateMaterials` to pack `mat.normal_scale` into `alpha_mode_misc.b` and `mat.emissive_factor`/`mat.emissive_strength` into the new vec4

4. Add tangent to `HitPayload` and interpolate in closesthit:
   - Extend `HitPayload`: add `vec3 tangent` and `float tangent_w`
   - In `closesthit.rchit`: barycentric interpolation of `v0.tangent.xyz`, `v1.tangent.xyz`, `v2.tangent.xyz`; transform to world space via `gl_ObjectToWorldEXT`; store in `payload.tangent` and `payload.tangent_w = v0.tangent.w`

5. Update `raygen.rgen` — material stride, texture sampling, normal mapping, MIS fix, emissive:
   - Material stride: `material_index * 6` → `material_index * kMaterialStride`
   - **Metallic-roughness map:** Decode index from `opacity_ior_tex.a`; if present, multiply base roughness by `.g` channel, base metallic by `.b` channel
   - **Normal map:** Decode index from `opacity_ior_tex.b`; if present, construct TBN from `payload.tangent`/`payload.tangent_w` with Gram-Schmidt re-orthogonalization, decode tangent-space normal, scale .xy by `normal_scale` from `alpha_mode_misc.b`, transform to world space
   - **Transmission map:** Decode index from `transmission_volume.a`; if present, multiply base transmission factor by `.r` channel
   - **Emissive:** Read emissive_factor and emissive_strength from material vec4 #6; if emissive_strength > 0, add `throughput * emissive_factor * emissive_strength * emissive_texture` to path_radiance
   - **MIS fix:** Track `chosen_prob` alongside `chosen_pdf` (from `SamplingProbabilities`); change throughput update to `brdf * NdotL * mis_weight / (chosen_prob * chosen_pdf)`
   - Replace all remaining magic numbers with named constants

6. Update `Renderer.cpp` — named constants:
   - Add `constexpr uint32_t kHaltonPeriod = 16` and `constexpr uint32_t kDefaultMaxBounces = 4`
   - Replace `frame_index % 16` with `frame_index % kHaltonPeriod`
   - Replace `pc.max_bounces = 4` with `pc.max_bounces = kDefaultMaxBounces`

### Verification
- **Normal mapping:** DamagedHelmet.glb shows surface detail from normal maps (not flat-shaded)
- **Metallic-roughness map:** MetalRoughSpheres.glb renders with correct per-texel roughness/metallic variation
- **Transmission map:** Transmission-textured surfaces show spatially varying transparency
- **Emissive:** Emissive surfaces glow when viewed directly; emissive contribution visible through bounced paths
- **MIS fix:** Energy convergence at high SPP — compare total radiance sum before/after fix; no visible brightness change at equal probability (fix only matters when strategy probabilities are unequal)
- **Named constants:** Shader compilation succeeds; rendering output matches pre-refactor (pixel-exact, since values are unchanged)
- **G-buffer:** Normal map-perturbed normals do NOT affect G-buffer world_normals (which uses geometric normal for denoiser stability)
- No NaN/Inf in output; no Vulkan validation errors

### Deferred
- **Emissive importance sampling (ReSTIR):** Emissive mesh extraction for direct NEE is implemented in Phase 8J; full ReSTIR-based importance sampling of emissive geometry is deferred to Phase F3
- **Double-sided materials:** Proper single-sided culling for non-double-sided surfaces
- **Second UV set:** `tex_coord_1` support for materials that reference it
- **Additional PBR extensions:** `KHR_materials_specular`, `KHR_materials_sheen`, `KHR_materials_anisotropy`, `KHR_materials_iridescence`, `KHR_materials_unlit`
- **Occlusion texture:** Ambient occlusion map sampling

---

## Phase 8E: Firefly Filter + Hit Distance Output

**Goal:** Add a post-trace firefly suppression filter and output hit distance in the G-buffer. Both are foundational for future denoiser quality (NRD/ReLAX require hit distance; firefly clamping prevents denoiser ghosting from extreme outlier samples).

**Source:** RTXPT `PathTracer.hlsl` (firefly clamping), NRD SDK documentation (hit distance requirements)

### Design Decisions

- **Luminance-based firefly clamp with separate diffuse/specular thresholds.** After the bounce loop completes, each path's radiance is clamped using its luminance to preserve hue. The luminance is computed as `lum = dot(path_radiance, vec3(0.2126, 0.7152, 0.0722))`, and if it exceeds the threshold the entire vector is scaled down proportionally: `path_radiance *= threshold / lum`. This avoids the hue shift that per-component `min()` clamping introduces when one channel is an extreme outlier. Separate thresholds are used for diffuse and specular paths: `kFireflyClampDiffuse = 20.0` and `kFireflyClampSpecular = 80.0`. Specular paths have a higher threshold because bright mirror reflections of light sources are physically expected and should not be aggressively suppressed. Both are named constants in `constants.glsl`. This introduces a small amount of energy loss but eliminates the extreme outlier samples that cause bright speckles (fireflies) in low-SPP renders.
- **Hit distance stored in G-buffer `linear_depth.g` channel.** The `linear_depth` G-buffer image format is widened from `R16F` to `RG16F`. The `.r` channel stores signed view-space linear depth (unchanged). The `.g` channel now stores the primary ray raw hit distance (`payload.hit_t`). NRD's ReLAX and ReBLUR consume hit distance for adaptive spatial filtering — larger hit distances allow wider filter kernels. For miss rays, hit distance is set to `kSentinelDepth` (1e4). Raw hit distance is stored now; any normalization required by a specific denoiser (e.g., `REBLUR_FrontEnd_GetNormHitDist()`) is deferred to the denoiser integration phase.
- **No new G-buffer images.** Hit distance is packed into the widened `linear_depth` channel to avoid adding a new descriptor binding or image allocation.
- **C++ and GLSL changes for RG16F.** The format change touches: `gbuffer_images.cpp` (format table entry), `Renderer.h` (comment), and `raygen.rgen` (layout qualifier `r16f` → `rg16f`).

### Tasks

1. **Add firefly constants to `shaders/include/constants.glsl`:**
   ```glsl
   const float kFireflyClampDiffuse = 20.0;
   const float kFireflyClampSpecular = 80.0;
   ```

2. **Add `FireflyClamp()` helper to `shaders/include/sampling.glsl`:**
   ```glsl
   vec3 FireflyClamp(vec3 radiance, float threshold) {
       float lum = dot(radiance, vec3(0.2126, 0.7152, 0.0722));
       return (lum > threshold) ? radiance * (threshold / lum) : radiance;
   }
   ```

3. **Update `raygen.rgen` — firefly clamp after bounce loop:**
   Apply the appropriate threshold based on path type, before accumulating into `total_diffuse`/`total_specular`:
   ```glsl
   if (is_specular_path)
       path_radiance = FireflyClamp(path_radiance, kFireflyClampSpecular);
   else
       path_radiance = FireflyClamp(path_radiance, kFireflyClampDiffuse);
   ```

4. **Widen `linear_depth` G-buffer from R16F to RG16F:**
   - `app/core/gbuffer_images.cpp`: change `VK_FORMAT_R16_SFLOAT` → `VK_FORMAT_R16G16_SFLOAT` for `kLinearDepth`.
   - `renderer/include/monti/vulkan/Renderer.h`: update comment `// R16F` → `// RG16F`.
   - `raygen.rgen`: change layout qualifier from `r16f` to `rg16f` on `img_linear_depth`.

5. **Update `raygen.rgen` — hit distance output:**
   - At the primary hit G-buffer write (the `!wrote_primary && path == 0` block), write `payload.hit_t` to `linear_depth.g`:
     ```glsl
     imageStore(img_linear_depth, pixel,
         vec4(linear_depth, payload.hit_t, 0.0, 0.0));
     ```
   - For miss rays, the sentinel write becomes:
     ```glsl
     imageStore(img_linear_depth, pixel, vec4(kSentinelDepth, kSentinelDepth, 0.0, 0.0));
     ```

6. **Create `tests/phase8e_test.cpp`** — see Verification section below.

### Verification

Create `tests/phase8e_test.cpp` with the following test cases:

1. **`FireflyClampPreservesHue`** — Construct synthetic `vec3` radiance values with high luminance and verify that `FireflyClamp()` scales the vector proportionally (output hue matches input hue within tolerance). Compare against per-component `min()` to demonstrate hue preservation.

2. **`FireflyClampBelowThreshold`** — Verify that radiance vectors with luminance below both `kFireflyClampDiffuse` and `kFireflyClampSpecular` pass through unmodified.

3. **`FireflyClampDiffuseThreshold`** — Verify that a radiance vector with luminance exceeding `kFireflyClampDiffuse` (but below `kFireflyClampSpecular`) is clamped when treated as diffuse, and passes through unmodified when treated as specular.

4. **`FireflyClampZeroAndNegative`** — Verify that zero radiance and edge cases (near-zero luminance) do not produce NaN or Inf.

5. **`LinearDepthFormatIsRG16F`** — Verify `GBufferImages` allocates `kLinearDepth` with `VK_FORMAT_R16G16_SFLOAT`.

6. **`HitDistanceOutput`** (integration, requires headless Vulkan context) — Render Cornell box at 1 spp, read back the `linear_depth` image:
   - Verify `.r` channel contains signed view-space linear depth for hit pixels.
   - Verify `.g` channel contains positive raw hit distances (0 < hit_t < scene_diagonal) for hit pixels.
   - Verify miss pixels have both `.r` and `.g` equal to `kSentinelDepth`.
   - Verify no NaN/Inf in either channel.

**Additional manual verification:**
- **Energy conservation note:** The firefly clamp is biased (removes energy). At high SPP (256+), verify the clamped render's mean luminance is within 5% of unclamped. If not, increase the threshold constants.
- No Vulkan validation errors.

---

## Phase 8F: Ray Cone Texture LOD

**Goal:** Implement ray cone tracking for automatic texture mip level selection. Ray cones estimate the footprint of each ray on a surface, enabling the shader to select the appropriate mip level for texture sampling. This improves image quality (reduces aliasing on distant surfaces) and GPU texture cache performance (avoids fetching full-resolution texels when coarser mips suffice).

**Source:** RTXPT `PathTracerShared.h` (ray cone state), `TexLODHelpers.hlsli` (LOD computation), `PathTracer.hlsli` (cone propagation + PDF-based spread expansion), "Improved Shader and Texture Level of Detail Using Ray Cones" (Akenine-Möller et al., JCGT 2021)

### Design Decisions

- **Ray cone state: 2 × float per ray (8 bytes).** Each ray tracks `cone_width` (spread at current distance) and `cone_spread_angle` (angular rate of change). Both are stored as full `float` fields in `HitPayload` for simplicity. The initial cone for primary rays is computed from the pixel footprint: `spread_angle = atan(2.0 * tan(fov/2) / screen_height)`, `width = 0.0` (ray starts at camera). FOV is recovered from `inv_proj[1][1]`: for a standard symmetric perspective matrix, `tan(fov/2) = 1.0 / inv_proj[1][1]`; since `inv_proj` is the *inverse* projection, the relationship is `tan(fov_y/2) = inv_proj[1][1]`. Push constants remain at 248 bytes (within the 256-byte limit assumed for desktop GPUs); no new push constant is needed.
- **Pre-computed triangle LOD constant (log-domain).** Closesthit computes `tri_lod_constant = 0.5 * safeLog2(uv_area / world_area)` — a single float in log-domain that is independent of texture dimensions. This matches RTXPT's `computeRayConeTriangleLODValue()` pattern. The full LOD formula is: `lod = tri_lod_constant + safeLog2(cone_width / normalTerm)` where `normalTerm = sqrt(abs(dot(ray_dir, normal)))` (sqrt provides more detail on grazing angles). Per-texture dimensions are added at sample time: `final_lod = 0.5 * baseLOD + lod` where `baseLOD = log2(tex_width * tex_height)`. This avoids computing a log of a potentially extreme ratio at sample time.
- **Mip level computed in raygen at texture sample sites.** After a hit, the raygen shader computes the texture-independent LOD from the ray cone width and the triangle's pre-computed LOD constant. This per-texture-independent `lod` is computed once per hit, then each `textureLod()` call adds the per-texture size term (`0.5 * log2(tex_width * tex_height)`). All material textures (base color, metallic-roughness, normal map, emission) use the same LOD — no special treatment for normal maps, matching RTXPT behavior.
- **PDF-based cone spread expansion on scatter (from RTXPT).** On each non-delta bounce, the cone spread angle is widened based on the BSDF sampling PDF: `spread_expansion = 0.3 * 2.0 * acos(max(-1.0, 1.0 - 1.0 / pdf / (2.0 * PI)))`. The factor `0.3` is a conservative underestimate (per JCGT 2021, Chapter 3) since stochastic supersampling handles antialiasing; the main goal is avoiding overblur. For delta events (perfect mirrors, glass at exact IOR), spread angle is unchanged. The cone spread is clamped to `2π` to prevent runaway growth.
- **Alpha-tested textures sample at LOD 0.** The existing `anyhit.rahit` shader samples base color alpha via `texture()` (implicit LOD). This is changed to `textureLod(..., 0.0)` to hard-code LOD 0, matching RTXPT's approach. Coarser mips would change alpha cutoff silhouettes, so alpha testing always uses full resolution.
- **No hard fallback at bounce >= 2.** Unlike an earlier design, there is no `max(lod, 1.0)` clamp. The PDF-based spread expansion naturally increases the LOD on deep bounces as scatter PDFs widen the cone. This matches RTXPT's behavior, which tracks ray cones at all bounces without a hard cutoff. The result is that close-up sharp reflections at bounce 2+ can still sample fine mips when warranted.
- **No impact on existing convergence tests.** Texture LOD affects aliasing quality, not convergence or energy conservation. Existing FLIP tests remain valid; new tests compare quality metrics.

### Tasks

1. **Add `kMinCosTheta` constant** to `shaders/include/constants.glsl`:
   ```glsl
   const float kMinCosTheta = 1e-5;
   ```

2. **Add `safeLog2()` helper** to `shaders/include/sampling.glsl`:
   ```glsl
   // log2 clamped to valid domain. Returns values in [-126, 127].
   float safeLog2(float x) {
       return log2(clamp(x, 1.175494e-38, 3.402823e+38));  // [FLT_MIN, FLT_MAX]
   }
   ```

3. **Extend `HitPayload`** in `shaders/include/payload.glsl` with ray cone data:
   ```glsl
   struct HitPayload {
       // ... existing fields ...
       float tri_lod_constant;  // 0.5 * log2(uv_area / world_area), precomputed per triangle
       // Ray cone state (passed through payload for multi-bounce tracking)
       float cone_width;
       float cone_spread_angle;
   };
   ```

4. **Update `closesthit.rchit`** — compute triangle LOD constant in log-domain:
   ```glsl
   // World-space triangle edges (already-fetched vertices, transformed)
   vec3 e1_world = (gl_ObjectToWorldEXT * vec4(v1.position - v0.position, 0.0)).xyz;
   vec3 e2_world = (gl_ObjectToWorldEXT * vec4(v2.position - v0.position, 0.0)).xyz;
   float world_area = length(cross(e1_world, e2_world));  // 2x world-space area

   // UV-space triangle area
   vec2 uv_e1 = v1.tex_coord_0 - v0.tex_coord_0;
   vec2 uv_e2 = v2.tex_coord_0 - v0.tex_coord_0;
   float uv_area = abs(uv_e1.x * uv_e2.y - uv_e1.y * uv_e2.x);  // 2x UV area

   // Pre-computed log-domain LOD constant (texture-size independent)
   payload.tri_lod_constant = 0.5 * safeLog2(uv_area / max(world_area, kMinCosTheta));
   ```

5. **Add `computeRayConeLod()` helper** to `shaders/include/sampling.glsl`:
   ```glsl
   // Compute texture-independent LOD from ray cone state and triangle constant.
   // Returns a value that must be combined with per-texture size:
   //   final_lod = 0.5 * log2(tex_w * tex_h) + computeRayConeLod(...)
   // `more_detail_on_slopes = true` uses sqrt(normalTerm) for grazing angles.
   float computeRayConeLod(float tri_lod_constant, float cone_width,
                           vec3 ray_dir, vec3 normal) {
       float filter_width = abs(cone_width);
       float normal_term = abs(dot(ray_dir, normal));
       normal_term = sqrt(normal_term);  // More detail on grazing angles
       return tri_lod_constant + safeLog2(filter_width / max(normal_term, kMinCosTheta));
   }
   ```

6. **Add `computeSpreadExpansionByPdf()` helper** to `shaders/include/sampling.glsl`:
   ```glsl
   // PDF-based cone spread expansion (from RTXPT).
   // Conservative factor of 0.3 avoids overblur; stochastic supersampling
   // handles antialiasing. For delta events (pdf -> infinity), returns ~0.
   float computeSpreadExpansionByPdf(float bsdf_pdf) {
       const float kGrowthFactor = 0.3;
       return kGrowthFactor * 2.0 * acos(
           max(-1.0, 1.0 - (1.0 / max(bsdf_pdf, kMinCosTheta)) / (2.0 * kPi)));
   }
   ```

7. **Update `raygen.rgen`:**
   - Initialize ray cone at primary ray:
     ```glsl
     // Recover vertical FOV half-tangent from inverse projection matrix
     float tan_half_fov = abs(pc.inv_proj[1][1]);
     float pixel_spread_angle = atan(2.0 * tan_half_fov / float(gl_LaunchSizeEXT.y));
     float cone_width = 0.0;
     float cone_spread = pixel_spread_angle;
     ```
   - After each hit, propagate cone distance:
     ```glsl
     cone_width = cone_spread * payload.hit_t + cone_width;
     ```
   - Compute texture-independent LOD once per hit:
     ```glsl
     float ray_cone_lod = computeRayConeLod(
         payload.tri_lod_constant, cone_width, ray_dir, shading_normal);
     ```
   - Replace `texture(bindless_textures[idx], uv)` calls with `textureLod()` including per-texture size:
     ```glsl
     // For each texture sample, add per-texture size term.
     // tex_base_lod = log2(tex_width * tex_height), precomputed or queried via textureSize().
     ivec2 tex_size = textureSize(bindless_textures[idx], 0);
     float tex_base_lod = log2(float(tex_size.x) * float(tex_size.y));
     float final_lod = 0.5 * tex_base_lod + ray_cone_lod;
     // Clamp to avoid overly coarse mips (keep at least 16×16 texel detail)
     int mip_levels = textureQueryLevels(bindless_textures[idx]);
     final_lod = min(final_lod, max(float(mip_levels) - 5.0, 0.0));
     vec4 sampled = textureLod(bindless_textures[idx], uv, final_lod);
     ```
   - After BSDF sampling, update cone spread based on scatter PDF:
     ```glsl
     if (!is_delta_event) {
         float spread_expansion = computeSpreadExpansionByPdf(bsdf_pdf);
         cone_spread = min(cone_spread + spread_expansion, 2.0 * kPi);
     }
     // For delta events (mirror reflection, perfect refraction), spread unchanged
     ```
   - Pass updated `cone_width` and `cone_spread` through payload for next bounce:
     ```glsl
     payload.cone_width = cone_width;
     payload.cone_spread_angle = cone_spread;
     ```

8. **Update `anyhit.rahit`** — hard-code LOD 0 for alpha testing:
   ```glsl
   // Alpha test always at LOD 0 to preserve silhouette accuracy
   float alpha = textureLod(bindless_textures[nonuniformEXT(base_color_tex_idx)], uv, 0.0).a;
   ```

9. **Create `tests/phase8f_test.cpp`** — see Verification section below.

### Verification

Create `tests/phase8f_test.cpp` with the following test cases:

1. **`SafeLog2ValidRange`** — Verify `safeLog2(x)` returns finite values for x in `{FLT_MIN, 0.001, 1.0, 1000.0, FLT_MAX}`. Verify `safeLog2(0.0)` does not produce NaN/Inf (clamped to FLT_MIN result).

2. **`TriLodConstantPositiveForLargeUV`** — Construct a triangle with large UV-space area and small world-space area. Verify `tri_lod_constant > 0` (higher mip = blurrier, as expected for a dense texel-to-world mapping).

3. **`TriLodConstantNegativeForSmallUV`** — Construct a triangle with small UV-space area and large world-space area. Verify `tri_lod_constant < 0` (lower mip = sharper, as expected for a sparse texel mapping).

4. **`TriLodConstantDegenerateTriangle`** — Verify that a degenerate triangle (zero world area) produces a finite, clamped LOD constant (no NaN/Inf) thanks to the `max(world_area, kMinCosTheta)` guard.

5. **`RayConeLodZeroWidthReturnsTriConstant`** — With `cone_width = 0` and ray perpendicular to surface, verify `computeRayConeLod()` returns `tri_lod_constant + safeLog2(0)` (a large negative number, clamping LOD to mip 0 in practice).

6. **`RayConeLodIncreasesWithWidth`** — Verify that increasing `cone_width` monotonically increases the returned LOD. Test with widths `{0.001, 0.01, 0.1, 1.0}`.

7. **`RayConeLodIncreasesAtGrazingAngles`** — Verify that as `dot(ray_dir, normal)` approaches 0, LOD increases (more blur at grazing angles). Test with cos_i values `{1.0, 0.5, 0.1, 0.01}`.

8. **`SpreadExpansionByPdfDecreasesWithHighPdf`** — Verify that `computeSpreadExpansionByPdf(pdf)` returns smaller angles for larger PDFs. Test with `pdf = {0.1, 1.0, 10.0, 100.0}` and verify monotonically decreasing expansion.

9. **`SpreadExpansionDeltaEvent`** — Verify that a very high PDF (delta event, e.g., `pdf = 1e6`) produces near-zero spread expansion (< 0.001 radians).

10. **`SpreadExpansionClampedToTwoPi`** — Verify that repeated spread expansions with low PDF never exceed `2π` when the raygen clamping logic is applied.

11. **`ConePropagationWidensWithDistance`** — Simulate a multi-bounce path: start with `cone_width = 0`, `cone_spread = 0.001` (narrow primary ray). Propagate through 3 hits at distances `[5.0, 10.0, 20.0]` with moderate PDF (1.0) scatter at each. Verify `cone_width` increases monotonically and `cone_spread` increases at each non-delta bounce.

12. **`FinalLodClampsToMipCount`** — Verify that the final LOD (after adding texture size term) is clamped to `max(mip_levels - 5.0, 0.0)`, preventing sampling below 16×16 texel resolution.

**Additional manual verification:**
- **Quality test:** Render a scene with a textured ground plane receding to the horizon. Compare ray-cone LOD vs fixed mip=0 — ray-cone version should show less moiré aliasing at distance.
- **Performance test:** Render DamagedHelmet.glb with and without ray cones. Measure frame time — ray cones should maintain or improve texture cache hit rate (lower L2 miss count on GPU profiler).
- **Convergence test:** Verify FLIP score at 256 spp is not degraded by ray cone LOD (energy conservation unaffected by mip selection).
- **Alpha silhouette test:** Render a foliage model (alpha-tested leaves). Verify leaf silhouettes are identical with and without ray cones (anyhit always uses LOD 0).
- No NaN/Inf in output; no Vulkan validation errors

---

## Phase 8G: Spherical Area Lights + Triangle Light Primitives

**Goal:** Extend the light system with two new light types: spherical area lights (analytic spheres with uniform emission) and triangle light primitives (for future emissive mesh decomposition). This expands light type coverage without requiring ReSTIR.

**Source:** RTXPT `PolymorphicLight.hlsli` (sphere lights, triangle lights), "Real-Time Polygonal-Light Shading with Linearly Transformed Cosines" (Heitz et al.)

### Design Decisions

- **`SphereLight` as a new scene-layer type.** A sphere light is defined by center (vec3), radius (float), and radiance (vec3). Sampling is straightforward: pick a visible point on the sphere via solid-angle uniform sampling, compute the PDF, trace a shadow ray. For small spheres viewed from a distance, the solid angle approaches a point light; the area integral ensures correct behavior at all distances.
- **`TriangleLight` as a new scene-layer type.** A triangle light is defined by three vertices (v0, v1, v2) and radiance (vec3, front-face emission). This is the fundamental primitive for decomposing emissive meshes into light sources in future phases. Sampling uses uniform random barycentric coordinates; PDF is `1 / triangle_area`.
- **Unified `PackedLight` buffer replaces `PackedAreaLight`.** All light types are packed into a single storage buffer using a type discriminator. Each packed light is 64 bytes (4 × vec4): the `.w` of the first vec4 encodes the light type as a float-encoded uint (0 = quad, 1 = sphere, 2 = triangle). Shader branching is minimal (one switch per light per shadow ray). The existing `PackedAreaLight` struct and `area_light_count` push constant are replaced by `PackedLight` and `light_count`.
- **Quad `AreaLight` retained alongside `TriangleLight`.** The quad is not deprecated — it has a simpler uniform sampling formula (no barycentric coordinates), a direct solid-angle PDF, and is the natural primitive for rectangular emitters (ceiling panels, windows, screens). Two triangles could represent a quad, but the dedicated quad path is more efficient and ergonomic for the common case. Host applications continue to use `AddAreaLight()` for rectangular emitters and `AddTriangleLight()` for arbitrary emissive geometry.
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

4. Update `GpuScene::UpdateLights()` — pack all three light types into `PackedLight[]`:
   - Iterate `AreaLights()` → pack as kQuad
   - Iterate `SphereLights()` → pack as kSphere
   - Iterate `TriangleLights()` → pack as kTriangle
   - Upload to storage buffer (binding 11)

5. Rename push constant `area_light_count` → `light_count` (total count across all types).

6. Create `shaders/include/lights.glsl` — light sampling functions:
   - `sampleQuadLight(PackedLight light, vec2 xi, vec3 shading_pos)` → `LightSample` (position, normal, pdf, radiance)
   - `sampleSphereLight(PackedLight light, vec2 xi, vec3 shading_pos)` → `LightSample` (visible-hemisphere solid angle sampling)
   - `sampleTriangleLight(PackedLight light, vec2 xi, vec3 shading_pos)` → `LightSample` (uniform barycentric sampling)
   - `sampleLight(PackedLight light, vec2 xi, vec3 shading_pos)` → dispatches by type

7. Update `raygen.rgen` — replace per-area-light loop with per-light loop using `sampleLight()`.

### Verification
- **Sphere light test:** Place a sphere light above the Cornell box floor. Verify soft circular shadow on the floor. Verify radiance falls off with distance squared.
- **Triangle light test:** Place a single triangle light on the ceiling. Verify illumination pattern. Verify two-sided emission when `two_sided = true`.
- **Backward compatibility:** Existing area light tests pass unchanged (quad lights packed as type 0).
- **Convergence test:** Cornell box at 4 spp vs 256 spp — FLIP below threshold.
- No NaN/Inf; no Vulkan validation errors.

---

## Phase 8H: Diffuse Transmission + Thin-Surface Mode

**Goal:** Add diffuse transmission (light passing through a thin surface with scattering, as in leaves and paper) and a thin-surface material flag that enables single-slab approximations without enter/exit tracking. These extend the BSDF with new lobes for translucent materials.

**Source:** RTXPT `ApplyDeltaLobes()` (diffuse transmission), glTF `KHR_materials_diffuse_transmission` extension, RTXPT nested dielectric priority model

### Design Decisions

- **Diffuse transmission as a new BSDF lobe.** When `diffuse_transmission_factor > 0`, a fraction of the incident light is transmitted diffusely through the surface (cosine-weighted hemisphere on the opposite side). This models thin translucent materials like leaves, paper, and fabric where light scatters forward through the medium. The factor controls the diffuse/transmission split: `(1 - diffuse_transmission_factor)` goes to regular diffuse reflection, `diffuse_transmission_factor` goes to diffuse transmission.
- **`thin_surface` material flag.** When true, the material uses single-slab approximations for all transmission effects: no IOR refraction (rays pass straight through), thin-slab Beer-Lambert attenuation (existing Phase 8C behavior), and the surface is treated as infinitely thin (no enter/exit state tracking). When false, the existing Phase 8C Fresnel refraction + IOR behavior applies. Most real-world thin translucent materials (leaves, curtains, lamp shades) should set `thin_surface = true`.
- **PackedMaterial extended to 8 vec4 (128 bytes).** A new 8th vec4 stores `diffuse_transmission_factor` (.r), `thin_surface` as float-encoded bool (.g), and `diffuse_transmission_color` (.rgb packed into .b and .a via two half-floats + one float, or stored in a simpler layout). To keep the packing simple: `.r = diffuse_transmission_factor`, `.g = thin_surface (0.0/1.0)`, `.b = reserved`, `.a = reserved`. The diffuse transmission color defaults to the base color (no separate field needed initially).
- **MIS update: 5-way strategy.** The diffuse transmission lobe adds a fifth sampling strategy. `SamplingProbabilities` gains a `diffuse_transmission` field. The probability is proportional to `diffuse_transmission_factor * (1 - metallic)` (metals can't transmit). Strategy selection and MIS weight computation extend naturally.

### Tasks

1. Add material fields to `MaterialDesc`:
   ```cpp
   float diffuse_transmission_factor = 0.0f;
   bool  thin_surface                = false;
   ```

2. Extend `PackedMaterial` to 8 vec4 (128 bytes):
   ```cpp
   glm::vec4 transmission_ext;  // .r = diffuse_transmission_factor,
                                 // .g = thin_surface (0.0/1.0),
                                 // .b = reserved, .a = reserved
   ```
   Update `static_assert(sizeof(PackedMaterial) == 128)`.

3. Update `GpuScene::UpdateMaterials()` to pack new fields.

4. Update `shaders/include/constants.glsl`:
   - `kMaterialStride = 8u` (was 7)

5. Update `shaders/include/mis.glsl` — add diffuse transmission strategy:
   - Add `STRATEGY_DIFFUSE_TRANSMISSION = 4` constant
   - Add `diffuse_transmission` field to `SamplingProbabilities` and `AllPDFs`
   - Update `calculateSamplingProbabilities` to compute transmission probability
   - Update `calculateAllPDFs` and `calculateMISWeight` for 5 strategies

6. Add `evaluateDiffuseTransmission()` to `shaders/include/brdf.glsl`:
   ```glsl
   vec3 evaluateDiffuseTransmission(vec3 albedo, float NdotL_back,
                                     float diffuse_transmission_factor) {
       return albedo * diffuse_transmission_factor / PI * max(-NdotL_back, 0.0);
   }
   ```

7. Update `raygen.rgen` bounce loop:
   - Read `diffuse_transmission_factor` and `thin_surface` from material
   - When `STRATEGY_DIFFUSE_TRANSMISSION` is chosen: sample cosine hemisphere on the **opposite** side of the normal (`-N`)
   - Evaluate diffuse transmission BRDF
   - For thin surfaces: skip IOR refraction in the existing transmission code path (pass through without bending)

8. Update glTF loader to parse `KHR_materials_diffuse_transmission`:
   - Read `diffuseTransmissionFactor` → `diffuse_transmission_factor`
   - Set `thin_surface = true` when the extension is present (glTF diffuse transmission implies thin surface)

### Verification
- **Leaf test:** Render a thin quad with `diffuse_transmission_factor = 0.8` and green base color, lit from behind. Verify light passes through with forward-scattered green tint.
- **Thin vs thick:** Compare `thin_surface = true` (no refraction bending) vs `thin_surface = false` (Fresnel refraction) on a glass panel. Thin surface should show straight-through transmission.
- **Energy conservation:** At high SPP, verify total outgoing energy (reflected + transmitted) does not exceed incoming for any material configuration.
- **Convergence test:** Scene with translucent materials at 4 spp vs 256 spp — FLIP below threshold.
- No NaN/Inf; no Vulkan validation errors.

---

## Phase 8I: Nested Dielectric Priority

**Goal:** Implement a material priority system for correctly handling overlapping dielectric volumes (e.g., liquid inside glass, coated objects). Without priority, the renderer cannot determine which IOR to use when exiting one volume and entering another simultaneously.

**Source:** RTXPT `NestedDielectrics.hlsli` (priority stack model), "Simple Nested Dielectrics in Ray Traced Images" (Schmidt & Budge, JGT 2002)

### Design Decisions

- **Priority-based IOR stack (simplified).** Each material with `transmission_factor > 0` has an integer `nested_priority` (0 = lowest, 255 = highest). When a ray enters a new volume, the priority determines which medium's IOR governs the interface. A higher-priority medium "wins" at shared boundaries. Default priority 0 means legacy behavior (no nesting awareness).
- **Compact stack in ray state.** The ray carries a small priority stack (4 entries, matching RTXPT) as part of the path state in the raygen shader. Each entry stores `{priority, ior}`. On entering a volume, push; on exiting, pop. The current medium's IOR is the top of the stack.
- **No PackedMaterial change for priority.** The `alpha_mode_misc.b` field was reserved in Phase 8D's PackedMaterial (7 vec4, now 8 vec4). Priority is stored in `transmission_ext.b` (the Phase 8H vec4's reserved .b slot).

### Tasks

1. Add `nested_priority` to `MaterialDesc`:
   ```cpp
   uint8_t nested_priority = 0;  // 0 = no nesting, 1-255 = priority (higher wins)
   ```

2. Pack into `PackedMaterial::transmission_ext.b` as float-encoded uint8.

3. Implement `IORStack` in `shaders/include/dielectric.glsl`:
   ```glsl
   const int kMaxIORStackDepth = 4;

   struct IORStack {
       uint priorities[kMaxIORStackDepth];
       float iors[kMaxIORStackDepth];
       int depth;
   };

   void pushIOR(inout IORStack stack, uint priority, float ior);
   void popIOR(inout IORStack stack, uint priority);
   float currentIOR(IORStack stack);  // Returns top-of-stack IOR, or 1.0 if empty
   ```

4. Update `raygen.rgen` transmission code path:
   - Initialize `IORStack` at path start (empty, depth=0)
   - On entering a transmissive volume: `pushIOR(stack, nested_priority, ior)`
   - On exiting (backface hit of same priority): `popIOR(stack, nested_priority)`
   - Use `currentIOR(stack)` as `n1` (current medium) when computing Fresnel

5. Update glTF loader to parse priority from material extensions (no standard glTF extension; use a custom property or default to 0).

### Verification
- **Glass-in-glass test:** Render a sphere (IOR 1.5, priority 1) inside a larger sphere (IOR 1.33, priority 2). Verify the inner sphere refracts correctly through the outer medium.
- **Single-volume test:** Existing glass scenes render identically when `nested_priority = 0` (no stack behavior).
- **Stack overflow test:** Verify graceful handling when depth exceeds `kMaxIORStackDepth` (should clamp, not crash).
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
- **Emissive mesh test:** Place a glTF object with emissive material in a dark room. Verify the emissive surface illuminates nearby objects via NEE (not just random hits). Compare 4 spp render with/without extraction — extracted version should show significantly less noise.
- **Performance:** Verify extraction runs in < 1ms for scenes with < 10K emissive triangles.
- **Convergence test:** Scene with emissive objects at 4 spp vs 256 spp — FLIP below threshold.
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
- **Performance test:** Render a scene with 100+ lights. Verify frame time is approximately constant regardless of light count (O(1) vs O(N)).
- **Convergence test:** Same scene at 256 spp — WRS result converges to same reference as the previous O(N) loop (FLIP < threshold). Note: WRS introduces selection noise, so low-SPP FLIP may be slightly higher than O(N); this is expected.
- **Single light test:** Verify behavior with exactly 1 light matches previous implementation (WRS trivially selects the only light).
- No NaN/Inf; no Vulkan validation errors.

---

## Phase 9A: Standalone Denoiser Library (`deni_vulkan`)

**Goal:** Build and unit-test the Deni Vulkan denoiser as a fully standalone library. This phase has **no dependency on Monti's renderer** — it can be developed in parallel with Phases 2–8.

**Source:** rtx-chessboard `render/passthrough_denoiser.h/.cpp`, `shaders/passthrough_denoise.comp`

### Tasks

1. Implement `denoise/src/vulkan/Denoiser.cpp`:
   - `Create()`: accept `DenoiserDesc` with optional `pipeline_cache` and required `allocator`; reject null allocator with an error (design decision 16 — no hidden internal allocators); create output image + image view (RGBA16F), create descriptor set layout, descriptor pool, descriptor sets, compute pipeline (using pipeline_cache if provided)
   - `Denoise()`:
     - Update descriptors with input image views
     - Transition output image to `VK_IMAGE_LAYOUT_GENERAL`
     - Bind compute pipeline + descriptors
     - Dispatch compute shader (ceil(width/16), ceil(height/16), 1)
     - Transition output image to `VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL`
     - Return output image view
   - `Resize()`: recreate output image, update descriptors
   - `LastPassTimeMs()`: GPU timestamp query (optional, return 0.0 initially)
   - Destructor: destroy all Vulkan resources

2. Implement `denoise/src/vulkan/shaders/passthrough_denoise.comp`:
   - Layout: `local_size_x = 16, local_size_y = 16`
   - Bindings: noisy_diffuse (storage image, readonly), noisy_specular (storage image, readonly), output (storage image, writeonly)
   - Operation: `output = diffuse + specular`
   - Bounds check: skip if `gl_GlobalInvocationID.xy >= image_size`

3. Compile shader: add to CMake shader compilation list

4. **Ensure standalone:** `deni_vulkan` library must compile and link without any `monti_*` dependencies. Verify the CMake target has no `monti_*` in its link list. Verify the public header has no GLM dependency — only `<vulkan/vulkan.h>` and standard library headers.

5. Write standalone unit test (`tests/deni_passthrough_test.cpp`):
   - Create a minimal Vulkan context (device, queue, command pool) — no renderer, no scene
   - Allocate two input images (RGBA16F) with known pixel data (e.g., diffuse = {0.3, 0.1, 0.2, 1.0}, specular = {0.1, 0.4, 0.05, 1.0})
   - Allocate placeholder images for the remaining DenoiserInput fields (motion_vectors, linear_depth, world_normals, diffuse_albedo, specular_albedo) — contents don't matter for passthrough
   - Create `Denoiser`, call `Denoise()`, read back output
   - Verify output = diffuse + specular per-pixel (FLIP with threshold 0.0)
   - Test `Resize()` works without crashes
   - Test output image layout is `VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL` after `Denoise()`

### Verification
- `deni_vulkan` library compiles independently (no monti dependencies, no GLM in public header)
- Standalone unit test passes: passthrough(diffuse, specular) pixel-exact match to `diffuse + specular`
- Resize works without crashes or leaks
- No Vulkan validation errors

### rtx-chessboard Reference
- [passthrough_denoiser.h/.cpp](../../rtx-chessboard/src/render/passthrough_denoiser.h): compute pipeline creation, dispatch pattern
- [passthrough_denoise.comp](../../rtx-chessboard/shaders/passthrough_denoise.comp): shader source

---

## Phase 9B: Denoiser Integration Test

**Goal:** Wire the standalone Deni denoiser into the Monti render loop and verify end-to-end correctness.

**Prerequisite:** Phase 8C (renderer produces G-buffer outputs) and Phase 9A (denoiser library built).

### Tasks

1. Wire denoiser into the `monti_view` render loop:
   - After `RenderFrame()` produces G-buffer, call `denoiser->Denoise(cmd, ...)` with G-buffer image views
   - Verify the denoised output feeds correctly into the tone mapper (Phase 10)

2. Write integration test:
   - Render Cornell box through the full pipeline: trace → denoise → verify
   - FLIP comparison: denoised output vs. CPU-computed `diffuse + specular` sum confirms passthrough is lossless

### Verification
- **Integration test:** render Cornell box — FLIP against pre-denoiser sum confirms passthrough is lossless
- No validation errors when denoiser is wired into the render loop
- Image layout transitions are correct through the full pipeline

### rtx-chessboard Reference
- [passthrough_denoiser.h/.cpp](../../rtx-chessboard/src/render/passthrough_denoiser.h): compute pipeline creation, dispatch pattern
- [passthrough_denoise.comp](../../rtx-chessboard/shaders/passthrough_denoise.comp): shader source

---

## Phase 10A: Tone Map + Present (End-to-End Pipeline)

**Goal:** Implement tone mapping and swapchain presentation as app-local code in `monti_view`, connect the full render pipeline: trace → denoise → tonemap → present. No interactive controls yet — use a fixed camera.

**Source:** rtx-chessboard `render/tone_mapper.h/.cpp`, `shaders/tonemapping.comp`, `main.cpp` render loop

### Tasks

1. Implement `app/core/tone_mapper.h` and `tone_mapper.cpp`:
   - Create LDR output image (RGBA8_UNORM), descriptor sets, compute pipeline
   - ACES filmic tone mapping (Stephen Hill's fit) with sRGB EOTF
   - Exposure control via push constant
   - Dispatch compute (16×16 workgroup)

2. Implement `app/shaders/tonemap.comp`:
   - Read HDR texel, apply exposure: `exposed = hdr * pow(2.0, exposure_ev)`
   - Apply ACES filmic
   - Apply accurate sRGB EOTF (piecewise linear + gamma 2.4)

3. Implement swapchain presentation in `monti_view`:
   - `vkCmdBlitImage()` from LDR output to swapchain (handles format conversion)
   - Transition swapchain image to `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`

4. Wire up complete render loop in `app/view/main.cpp`:
   ```
   while (running) {
       cmd = beginFrame()
       renderer.RenderFrame(cmd, gbuffer, frame_index)
       denoised = denoiser.Denoise(cmd, {gbuffer inputs})
       ToneMap(cmd, denoised, ldr_image, exposure)
       BlitToSwapchain(cmd, ldr_image, swapchain)
       endFrame()
   }
   ```

5. Test scene acquisition and integration:
   - Add CMake option `MONTI_DOWNLOAD_TEST_SCENES` (default OFF) that fetches additional glTF test scenes at configure time via `FetchContent` or a download script
   - **Core scenes** (small, committed to `tests/assets/`): Box.glb, DamagedHelmet.glb, DragonAttenuation.glb, ClearCoatTest.glb, MosquitoInAmber.glb, MaterialsVariantsShoe.glb, MorphPrimitivesTest.glb (already present), programmatic CornellBox
   - **Extended scenes** (large, downloaded on demand): Amazon Lumberyard Bistro (exterior + interior), NVIDIA Emerald Square, Sponza (Intel), Khronos ToyCar, Khronos FlightHelmet — sourced from NVIDIA RTXPT-Assets repo, Khronos glTF-Sample-Assets, and Intel OIDN sample scenes
   - Create `tests/scene_configs/` directory with per-scene JSON camera configs (position, target, up, FOV) transcribed from RTXPT `.scene.json` files where available
   - Add `tests/golden/` directory structure for storing LDR reference renders per scene (gitignored for large files; generated locally via `monti_datagen --golden`)

6. Golden test expansion:
   - Extend `tests/golden_test.cpp` (or equivalent) with parameterized test cases for each scene + camera config
   - Each test: load scene → render N SPP → tonemap → FLIP compare against stored golden reference
   - FLIP thresholds: mean < 0.05 for simple scenes (Cornell, Box), mean < 0.08 for complex scenes (Bistro, Sponza)
   - CI runs core scenes only (small assets); extended scenes run in a nightly/manual job

### Verification
- **Unit test:** feed known HDR values through ACES + sRGB on CPU, compare GPU output — per-pixel error < 1/255
- **End-to-end golden test:** full pipeline (trace → denoise → tonemap) on Cornell box, FLIP against stored LDR reference (mean < 0.05)
- **End-to-end golden test:** `DamagedHelmet.glb` full pipeline, FLIP against stored reference
- **Extended golden tests:** Bistro exterior, Sponza, ToyCar — FLIP mean < 0.08 (nightly CI only)
- Resize works through the entire pipeline (no validation errors after resize)
- Clean shutdown (no leaks reported by VMA stats or validation layers)

### rtx-chessboard Reference
- [tone_mapper.h/.cpp](../../rtx-chessboard/src/render/tone_mapper.h): pipeline setup, dispatch
- [tonemapping.comp](../../rtx-chessboard/shaders/tonemapping.comp): ACES + sRGB shader
- [main.cpp](../../rtx-chessboard/src/main.cpp): `RenderFrame()` function, blit-to-swapchain

### RTXPT Reference
- [RTXPT-Assets](https://github.com/NVIDIA-RTX/RTXPT-Assets): Bistro, Kitchen, A Beautiful Game scene data
- RTXPT `.scene.json` files: camera positions and render settings for each test scene

---

## Phase 10B: Interactive Camera + ImGui Overlay (`monti_view`)

**Goal:** Add interactive camera controls and an ImGui debug overlay to `monti_view` per [app_specification.md](app_specification.md) §6.4–§6.5.

**Source:** rtx-chessboard `input/camera_controller.h/.cpp`, `ui/ui_renderer.h/.cpp`, `main.cpp`

### Tasks

1. Implement `app/view/camera_controller.h` and `camera_controller.cpp`:
   - Fly camera (default): WASD + right-click drag for look, Q/E for up/down, mouse wheel for speed, Shift for fast
   - Orbit camera (toggle `O`): left-click drag orbit, middle-click pan, wheel zoom
   - Movement speed scales with scene bounding box diagonal
   - Update `scene.SetActiveCamera()` each frame
   - F key: focus on scene center (auto-fit camera)

2. Implement `app/view/ui_renderer.h` and `ui_renderer.cpp`:
   - Initialize ImGui with Vulkan + SDL3 backends, FreeType font rendering
   - Load single TrueType font (Inter-Regular, 16px) from `app/assets/fonts/`

3. Implement `app/view/panels.h` and `panels.cpp`:
   - **Top bar:** FPS / frame time / renderer ms / denoiser ms, scene file name, mode indicator (Fly/Orbit)
   - **Settings panel (Tab):** SPP slider, exposure EV, environment rotation, denoiser toggle, debug visualization (Off/Normals/Albedo/Depth/Motion Vectors/Noisy), camera info (FOV, position — read-only), scene info (node/mesh/material/triangle counts)
   - **Camera path panel (C):** record/save path, path preview (deferred to later if too complex)

4. Wire ImGui into the `monti_view` render loop (`app/view/main.cpp`):
   - Record ImGui draw commands after tone map, before present
   - Suppress camera input when `ImGui::GetIO().WantCaptureMouse`

### Verification
- Camera movement is smooth, motion vectors update correctly
- ImGui panels render without visual artifacts
- Changing SPP / exposure via panel takes effect immediately
- Performance: frame times within 2× of rtx-chessboard for same scene and settings
- No validation errors from ImGui rendering

### rtx-chessboard Reference
- [camera_controller.h/.cpp](../../rtx-chessboard/src/input/camera_controller.h): fly camera, input handling
- [ui_renderer.h/.cpp](../../rtx-chessboard/src/ui/ui_renderer.h): ImGui init, frame recording
- [main.cpp](../../rtx-chessboard/src/main.cpp): ImGui integration, camera controls

---

## Phase 11A: Capture Writer (`monti_capture`)

**Goal:** Implement the CPU-side OpenEXR writer as a standalone library. This phase has no GPU dependency — it takes `const float*` arrays and writes EXR files. Each frame produces **two** EXR files at different resolutions: an input EXR (noisy radiance + G-buffer) and a target EXR (high-SPP reference).

**Source:** Design spec §8

### Tasks

1. Implement `capture/src/Writer.cpp`:
   - `Create()`: validate output directory, create if needed; compute target resolution from `WriterDesc::input_width/height` and `ScaleMode` using `target_dim = floor(input_dim × scale_factor / 2) × 2`
   - `monti::capture::ScaleMode` enum mirrors `deni::vulkan::ScaleMode` (the capture writer is CPU-only with no Vulkan dependency)
   - `TargetWidth()`/`TargetHeight()` accessors return the computed target resolution
   - `WriteFrame(input, target, frame_index)` writes two EXR files per frame:
     - `{output_dir}/frame_{NNNN}_input.exr` — `InputFrame` channels at input resolution
     - `{output_dir}/frame_{NNNN}_target.exr` — `TargetFrame` channels at target resolution

2. Input EXR — per-channel bit depths (OpenEXR supports independent pixel types per channel):
   - `noisy_diffuse` → `noisy_diffuse.R`, `noisy_diffuse.G`, `noisy_diffuse.B` (HALF)
   - `noisy_specular` → `noisy_specular.R`, `noisy_specular.G`, `noisy_specular.B` (HALF)
   - `diffuse_albedo` → `diffuse_albedo.R`, `diffuse_albedo.G`, `diffuse_albedo.B` (HALF)
   - `specular_albedo` → `specular_albedo.R`, `specular_albedo.G`, `specular_albedo.B` (HALF)
   - `normal` → `normal.X`, `normal.Y`, `normal.Z`, `normal.W` (HALF)
   - `depth` → `depth.Z` (FLOAT — FP32 avoids precision loss at long view distances)
   - `motion` → `motion.X`, `motion.Y` (HALF)

3. Target EXR — all channels FP32:
   - `ref_diffuse` → `ref_diffuse.R`, `ref_diffuse.G`, `ref_diffuse.B` (FLOAT)
   - `ref_specular` → `ref_specular.R`, `ref_specular.G`, `ref_specular.B` (FLOAT)

4. Write integration test (`tests/capture_writer_test.cpp`):
   - Create known-value float arrays at two different resolutions (e.g., 64×64 input, 128×128 target via `ScaleMode::kPerformance`)
   - Write via `Writer::WriteFrame(input, target, 0)`
   - Verify `TargetWidth()` / `TargetHeight()` return the expected computed dimensions
   - Reload both EXR files via tinyexr and verify:
     - Input EXR: correct channel names, correct resolution, per-channel precision (HALF vs FLOAT)
     - Target EXR: correct channel names, correct (larger) resolution, FP32 precision
   - Verify pixel values round-trip correctly (within FP16 precision for HALF channels, exact for FLOAT)
   - Verify null pointer fields are omitted from their respective EXR

### Verification
- **Integration test:** write known data at two resolutions → reload both EXR files → verify channel names, resolutions, bit depths, and pixel values match
- Input EXR contains all enabled input layers with mixed bit depths (HALF + FLOAT)
- Target EXR contains reference layers at the larger resolution in FP32
- Null pointer fields produce no EXR channels (verified in both files)
- File sizes are reasonable (not zero, not unexpectedly large)
- `monti_capture` library compiles with no Vulkan dependency (CPU-side only)

### rtx-chessboard Reference
- No direct equivalent (rtx-chessboard doesn't have capture). Use tinyexr documentation.

---

## Phase 11B: GPU Readback + Headless Data Generator (`monti_datagen`)

**Goal:** Implement the full `monti_datagen` executable: headless Vulkan rendering at two resolutions, GPU → CPU readback, high-SPP reference rendering at the target resolution, and dual-file EXR output via the capture writer. This completes the `app/datagen/` stub from Phase 4 per [app_specification.md](app_specification.md) §7.

**Prerequisite:** Phase 10A (full render pipeline working) and Phase 11A (capture writer).

**Source:** Design spec §8, §9, §10.2. App specification §7.

### Tasks

1. Implement dual-resolution rendering via separate GBuffers:
   - Parse `--target-scale` CLI option (maps to `monti::capture::ScaleMode`: `native`=1×, `quality`=1.5×, `performance`=2×; default: `performance`)
   - Create capture `Writer` with input dimensions and `ScaleMode`; query `TargetWidth()`/`TargetHeight()` for G-buffer allocation
   - Allocate input G-buffer at `--width` × `--height` (compact formats)
   - Allocate reference G-buffer at target resolution (RGBA32F for radiance, RGBA16F for aux)
   - Create renderer with `width`/`height` set to the **target** (larger) resolution
   - Print both resolutions at startup: `Input: 960×540, Target: 1920×1080 (performance 2.0×)`

2. Implement the generation loop with two render passes per frame:
   - `renderer->SetSamplesPerPixel(spp)` → `RenderFrame(cmd, input_gbuffer, frame)` at input resolution
   - `renderer->SetSamplesPerPixel(ref_spp)` → `RenderFrame(cmd, ref_gbuffer, frame)` at target resolution
   - The renderer is format-agnostic and resolution-agnostic — it uses the GBuffer's image dimensions

3. Implement GPU → CPU readback utilities in `app/core/`:
   - Create staging buffer per G-buffer image (separate sets for input and target resolutions)
   - `vkCmdCopyImageToBuffer()` after render completes
   - Map staging buffer, copy to CPU, unmap

4. Implement `app/datagen/main.cpp` (replacing the Phase 4 stub):
   - CLI parsing: `--camera-path`, `--output`, `--width`, `--height`, `--target-scale`, `--spp`, `--ref-spp`, `--env`, `--exposure`, scene file (required)
   - `--target-scale` maps to `monti::capture::ScaleMode` (default: `performance`)
   - Built-in camera path generators: `orbit:N`, `orbit:N:elevation`, `random:N` per app specification §5
   - Headless VulkanContext (no window, no surface, no swapchain)
   - Load scene, upload geometry, create renderer and both G-buffer sets
   - No denoiser, no tone mapping — output is linear HDR

5. Implement `app/datagen/generation_session.h` and `generation_session.cpp`:
   - Synchronous generation loop per app specification §7.2:
     - For each camera position: set camera, render noisy G-buffer at input resolution, render high-SPP reference at target resolution, submit and wait, readback both, write two EXR files
   - Progress to stdout: `[N/M] frame_NNNN written (X.XXs)`
   - Summary on completion: total frames, total time, output directory, input/target resolutions
   - Exit code 0 on success, 1 on error

6. Implement `app/datagen/camera_path.h` and `camera_path.cpp`:
   - Load camera path from JSON file (nlohmann/json)
   - Built-in generators: orbit (auto-fit distance to scene bounding box), random viewpoints on sphere

### Verification
- **Integration test:** `monti_datagen` captures Cornell box frame, reload both EXR files:
  - Input EXR: verify resolution matches `--width`/`--height`, channels have expected names and mixed bit depths
  - Target EXR: verify resolution matches computed target, `ref_diffuse`/`ref_specular` channels are FP32
  - Compare `ref_diffuse + ref_specular` from target EXR to a live render via FLIP (mean < 0.01 — validates readback fidelity)
- **Resolution test:** input and target EXR files have different resolutions (target = input × scale factor)
- **Noise test:** FLIP(noisy @ 4 spp, reference @ 256 spp) > 0.1 (confirms noisy data is actually noisy, not accidentally the reference). Note: FLIP comparison requires downscaling target to input resolution or upscaling input.
- `monti_datagen` runs without a display server (no swapchain created, no window opened)
- Exit code 0 on success, 1 on error (missing scene file, invalid camera path, invalid `--target-scale`)
- Progress output is parseable (`[N/M]` format)

### rtx-chessboard Reference
- No direct equivalent (rtx-chessboard doesn't have capture). Use tinyexr documentation.

---

## Future Phases (Not in Initial Plan)

These are documented for roadmap visibility but not scheduled. See [roadmap.md](roadmap.md) for detailed breakdowns.

| Future Phase | Description | Prerequisite |
|---|---|---|
| F1 | NRD denoiser (ReLAX/ReBLUR, cross-vendor, open-source) | Phase 9A + 8E (hit distance output) |
| F2 | ReSTIR Direct Illumination | Phase 8K complete (WRS foundation) |
| F3 | Emissive mesh importance sampling via ReSTIR | Phase 8J + F2 |
| F4 | Volume enhancements (homogeneous scattering, heterogeneous media) | Phase 8I complete (nested dielectrics) |
| F5 | DLSS-RR denoiser backend (NVIDIA-only) | Phase 9A complete |
| F6 | Mobile Vulkan renderer (`monti_vulkan_mobile`) | Phase 8K complete (shared GpuScene/GeometryManager); hybrid rasterization (default) + ray query pipeline; projection-matrix jitter for TAA; format-agnostic G-buffer via `shaderStorageImageReadWithoutFormat` |
| F7 | Metal renderer (C API) | Phase 8K design patterns established |
| F8 | WebGPU renderer (C API → WASM) | Phase 8K design patterns established |
| F9 | ML denoiser training pipeline | Phase 11B complete (training data capture working) |
| F10 | Shader permutation cache | Phase 8K complete |
| F11 | ML denoiser deployment (desktop + mobile) | F9 complete (trained model weights available) |
| F12 | Super-resolution in ML denoiser | F11 complete; uses `ScaleMode` enum in `DenoiserInput` (kQuality 1.5×, kPerformance 2×) |
| F13 | Fragment shader denoiser (mobile) | F6 + F11 complete; denoise → tonemap → present as render pass subpasses; `Denoiser` auto-selects compute vs fragment based on device |
| F14 | GPU skinning + morph targets | Phase 6 complete (BLAS refit hooks); compute shader pipeline for joint transforms + morph weight blending, BLAS refit on deformed vertices |

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
  │                                │                                          ├─→ Phase 10B (monti_view: interactive + ImGui)
  │                                │                                          └─→ Phase 11B (monti_datagen: readback + headless)
  │                                └─→ Phase 10A (monti_view: tonemap + present)
  ├─→ Phase 9A (standalone denoiser)              ─→ Phase 9B
  └─→ Phase 11A (capture writer — CPU-only)        ─→ Phase 11B
```

Phases 2 and 4 can be developed in parallel. Phase 9A (standalone denoiser library) can be developed in parallel with Phases 2–8 since it has no Monti dependencies. Phase 11A (capture writer) can also be developed in parallel with Phases 2–10 since it is CPU-only with no GPU dependency. Phase 9B requires both 8D and 9A. Phase 10A (`monti_view` tonemap + present) can start after 8D + 9B. Phase 10B (`monti_view` interactive UI) depends on 10A. Phase 11B (`monti_datagen` headless data generator) depends on 10A + 11A. Phases 8E–8K can be developed in any order after 8D, except: 8J requires 8G (triangle light type), 8K requires 8G+8J (light buffer with all types).
