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
   - Shader compilation: find `glslc`, custom command for `.rgen`/`.rchit`/`.rmiss`/`.comp` → `.spv` (SPIR-V embedding as `constexpr uint32_t[]` arrays deferred to Phase 7B when real shaders exist)
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
   - `scene/include/monti/scene/Light.h` — EnvironmentLight per §5.4
   - `scene/include/monti/scene/Camera.h` — CameraParams per §5.5
   - `scene/include/monti/scene/Scene.h` — Scene class, SceneNode per §5.2
   - `renderer/include/monti/vulkan/Renderer.h` — Renderer class, RendererDesc (including `get_device_proc_addr`), GBuffer per §6.3
   - `renderer/include/monti/vulkan/GpuBufferUtils.h` — GpuBuffer, upload helpers per §6.1.1
   - `capture/include/monti/capture/Writer.h` — Writer class, WriterDesc, InputFrame, TargetFrame per §8
   - **Internal headers** (GpuScene.h, GeometryManager.h, EnvironmentMap.h, BlueNoise.h) are deferred to their respective implementation phases.

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
   - `MaterialDesc` struct (full PBR per §5.3; transmission/volume fields implemented, not deferred; emissive fields included but noted as deferred pending ReSTIR on desktop; sheen deferred — not in v1)

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

## Phase 5: GPU Scene (`monti::vulkan::GpuScene`)

**Goal:** Create the GpuScene that registers host-provided geometry buffers, packs materials, and uploads textures for ray tracing.

**Source:** rtx-chessboard `render/gpu_scene.h/.cpp`, `core/buffer.h/.cpp`, `core/image.h/.cpp`, `core/upload.h/.cpp`

### Tasks

1. Implement RAII buffer and image wrappers:
   - `Buffer` class: VMA-allocated, supports staging + device-local
   - `Image` class: VMA-allocated, supports mip generation, image views
   - `Upload` class: staging buffer management for CPU→GPU transfers

2. Implement `renderer/src/vulkan/GpuScene.h` and `GpuScene.cpp`:
   - `RegisterMeshBuffers(mesh_id, binding)`:
     - Store `MeshBufferBinding` (VkBuffer, VkDeviceAddress for vertex+index, counts, stride)
     - Host is responsible for uploading vertex/index data to GPU buffers before calling this
     - Device addresses used later for BLAS building and shader `buffer_reference` access
   - `UpdateMaterials(scene)`:
     - Pack `MaterialDesc` → `PackedMaterial` array → storage buffer
     - Include transmission/volume fields in the packed representation
   - `UploadTextures(scene, cmd)`:
     - Upload texture images to VkImage array with image views for bindless binding
   - ID → index lookup maps for meshes and materials

3. Implement GPU buffer upload helpers in `renderer/include/monti/vulkan/GpuBufferUtils.h`:
   - `UploadMeshToGpu(allocator, device, cmd, mesh_data)` → returns `GpuBuffer` pair for vertex/index
   - `CreateVertexBuffer()`, `CreateIndexBuffer()`, `DestroyGpuBuffer()`
   - Uses VMA staging → device-local transfer
   - These are optional convenience helpers in the `monti_vulkan` library (not app code)
   - Hosts with their own buffer management skip these and call `RegisterMeshBuffers()` directly

4. Wire `Renderer::GetGpuScene()` public accessor:
   - `Renderer::GetGpuScene()` returns a reference to the internal `GpuScene`
   - Verify host can register mesh buffers through this accessor

### Verification
- `RegisterMeshBuffers()` stores correct device addresses
- Material buffer contains correct packed values (readback and compare), including transmission fields
- Texture images created with correct dimensions and formats
- No VMA allocation failures or validation errors
- Host helper uploads glTF mesh data and registers successfully

### rtx-chessboard Reference
- [gpu_scene.h/.cpp](../../rtx-chessboard/src/render/gpu_scene.h): merged buffer approach, `PackedMaterial`, texture upload
- [buffer.h/.cpp](../../rtx-chessboard/src/core/buffer.h): VMA buffer wrapper
- [image.h/.cpp](../../rtx-chessboard/src/core/image.h): VMA image wrapper, mip generation
- [upload.h/.cpp](../../rtx-chessboard/src/core/upload.h): staging buffer management

---

## Phase 6: Acceleration Structures (`GeometryManager` — Internal)

**Goal:** Build BLAS per mesh and a single TLAS for all scene nodes. `GeometryManager` is internal to the renderer (§6.2 of the design doc) — it is not exposed in public headers. `RenderFrame()` calls it automatically. Tests in this phase exercise the class through its internal interface.

**Source:** rtx-chessboard `render/hw/hw_path_tracer.cpp` (BLAS/TLAS building functions)

### Tasks

1. Implement `renderer/src/vulkan/GeometryManager.h` and `GeometryManager.cpp`:
   - `BuildAllBlas(cmd, gpu_scene)`:
     - One `VkAccelerationStructureKHR` per registered mesh
     - Triangle geometry referencing host-provided vertex/index buffers via device addresses from `MeshBufferBinding`
     - Vertex format: `VK_FORMAT_R32G32B32_SFLOAT` (position)
     - Stride: from `MeshBufferBinding::vertex_stride`
     - Enable compaction flag
     - Two-pass: build uncompacted → query compact size → build compacted → destroy uncompacted
     - Cache device addresses per BLAS
   - `BuildTlas(cmd, scene, gpu_scene)`:
     - One `VkAccelerationStructureInstanceKHR` per visible scene node
     - Transform from `SceneNode::transform.ToMatrix()` → `VkTransformMatrixKHR`
     - Instance custom index: encode mesh range index + material index (12 bits each)
     - Instance mask `0xFF`
     - Single TLAS with `VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR`
   - Scratch buffer management (single large allocation, reused)

2. Allocate and manage TLAS instance buffer:
   - Staging upload per frame for instance data
   - or device-local with host-visible for per-frame updates

3. Implement `NotifyMeshDeformed()`:
   - Mark BLAS as needing refit (not full rebuild) when `topology_changed = false`
   - Mark BLAS as needing full rebuild when `topology_changed = true`

4. Implement BLAS cleanup on mesh removal:
   - When `Scene::RemoveMesh()` has been called and no nodes reference the mesh, destroy the corresponding BLAS
   - `RenderFrame()` detects removed meshes by comparing its internal state against the current `Scene`

### Verification
- BLAS build completes without validation errors
- BLAS compaction reduces memory (log before/after sizes)
- TLAS build completes with correct instance count
- Device addresses are non-zero for all BLAS entries
- Modifying a scene node transform and calling `RenderFrame()` rebuilds the TLAS correctly
- Removing a mesh via `Scene::RemoveMesh()` causes BLAS cleanup on the next `RenderFrame()`

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
   - `linear_depth`: R16F (recommended) or RGBA16F, `VK_IMAGE_USAGE_STORAGE_BIT`
   - `world_normals`: RGBA16F, `VK_IMAGE_USAGE_STORAGE_BIT`
   - `diffuse_albedo`: R11G11B10F (recommended) or RGBA16F, `VK_IMAGE_USAGE_STORAGE_BIT`
   - `specular_albedo`: R11G11B10F (recommended) or RGBA16F, `VK_IMAGE_USAGE_STORAGE_BIT`
   - Support resize/recreate on window resize
   - Require `shaderStorageImageReadWithoutFormat` + `shaderStorageImageWriteWithoutFormat` at device creation (see §6.5.1 in design doc) — enables format-agnostic storage image access, so shaders work with either compact or RGBA16F formats without permutations

2. Implement environment map loading (`renderer/src/vulkan/EnvironmentMap.h/.cpp`):
   - Load HDR equirectangular map (EXR via tinyexr) — pixel loading and CDF
     computation are both internal to `monti_vulkan`; the scene layer only
     stores a `TextureId` reference in `EnvironmentLight`
   - Create `VkImage` + `VkImageView` + `VkSampler`
   - Pre-compute marginal/conditional CDFs for importance sampling (storage buffers)
   - Follow rtx-chessboard's CDF computation in `environment_loader.cpp`

3. Implement blue noise table (`renderer/src/vulkan/BlueNoise.h/.cpp`):
   - Generate or load blue noise data
   - Create storage buffer accessible in shaders
   - Follow rtx-chessboard's `BlueNoise` class

### Verification
- Integration test: load an HDR environment map, verify CDF buffers are non-zero and sum to ~1.0 (readback marginal CDF last entry)
- G-buffer images created at correct resolution with correct formats
- Blue noise buffer allocated and populated
- No VMA allocation failures or validation errors
- Window resize recreates all images without leaks

### rtx-chessboard Reference
- [hw_path_tracer.cpp](../../rtx-chessboard/src/render/hw/hw_path_tracer.cpp): output image creation
- [environment_loader.cpp](../../rtx-chessboard/src/loaders/environment_loader.cpp): HDR loading, CDF computation
- [blue_noise.h/.cpp](../../rtx-chessboard/src/render/blue_noise.h): blue noise generation/loading

---

## Phase 7B: Descriptor Sets + Push Constants + Ray Tracing Pipeline

**Goal:** Create the Vulkan ray tracing pipeline object, descriptor set layouts, and shader binding table. No shaders yet — this is the pipeline plumbing.

**Source:** rtx-chessboard `render/hw/hw_path_tracer.cpp` (pipeline creation ~line 59, SBT setup)

### Tasks

1. Create descriptor set layout and descriptor sets:
   - Binding 0: TLAS (`VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR`)
   - Binding 1: Output storage image (noisy_diffuse)
   - Binding 2: Output storage image (noisy_specular)
   - Binding 3: Vertex buffer (storage buffer, device address)
   - Binding 4: Index buffer (storage buffer, device address)
   - Binding 5: Material buffer (storage buffer)
   - Binding 6: Mesh range buffer (storage buffer)
   - Binding 7: Bindless texture array (`VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER[]`)
   - Binding 8: Environment map + CDF buffers
   - Binding 9: Blue noise buffer
   - Binding 10+: Additional G-buffer storage images (motion, depth, normals, albedo)

2. Create push constant layout:
   - Camera: inverse view matrix, inverse projection matrix
   - Frame index, samples per pixel, max bounces
   - Image dimensions

3. Create ray tracing pipeline:
   - Implement SPIR-V embedding pipeline: CMake custom command runs `glslc` to compile `.glsl` → `.spv`, then generates a `.h` file with `constexpr uint32_t[]` array per shader (deferred from Phase 1)
   - Load SPIR-V for raygen, closest-hit, miss shaders (use placeholder/minimal shaders for compilation)
   - Create shader modules
   - Create shader groups (raygen, hit, miss)
   - Create pipeline with max recursion depth
   - Create shader binding table (SBT) buffer with correct stride and alignment

### Verification
- Pipeline creation succeeds without validation errors
- SBT buffer allocated with correct size and alignment (verify against `VkPhysicalDeviceRayTracingPipelinePropertiesKHR`)
- Descriptor sets update without validation errors when bound to the resources from Phase 7A and Phases 5–6
- Push constant range fits within device limits

### rtx-chessboard Reference
- [hw_path_tracer.cpp](../../rtx-chessboard/src/render/hw/hw_path_tracer.cpp): pipeline creation (~line 59), SBT setup, descriptor layout, push constant layout

---

## Phase 7C: Raygen + Miss + Closesthit Stub + RenderFrame

**Goal:** Write the initial shaders and wire up `Renderer::RenderFrame()` so that rays are cast and the environment map is visible on screen.

**Source:** rtx-chessboard `shaders/raygen.rgen`, `shaders/miss.rmiss`, `shaders/closesthit.rchit`

### Tasks

1. Implement basic `raygen.rgen`:
   - Compute ray origin and direction from pixel + camera inverse matrices
   - Single sample per pixel (no MIS yet, no bounce loop)
   - `traceRayEXT()` → miss returns environment map sample
   - Write result to noisy_diffuse output

2. Implement `miss.rmiss`:
   - Sample environment map using ray direction
   - Apply basic importance sampling with pre-computed CDFs
   - Write to payload

3. Implement basic `closesthit.rchit` (stub):
   - Return a flat shaded color (e.g., normal as color) to verify hits work
   - Full material shading deferred to Phase 8A

4. Implement `Renderer::RenderFrame()`:
   - Bind pipeline, descriptor sets, push constants
   - Transition output images to `VK_IMAGE_LAYOUT_GENERAL` before trace
   - `vkCmdTraceRaysKHR()` with SBT addresses

### Verification
- **Integration test:** window shows environment map rendered via ray tracing miss shader
- Loading Cornell box (programmatic) shows room silhouettes (normals as color from closest-hit stub)
- Loading `DamagedHelmet.glb` shows object silhouette with environment visible around it
- Push constants correctly pass camera matrices (verify by rotating camera — scene rotates)
- No validation errors during trace

### rtx-chessboard Reference
- [raygen.rgen](../../rtx-chessboard/shaders/raygen.rgen): camera ray generation, basic structure
- [miss.rmiss](../../rtx-chessboard/shaders/miss.rmiss): environment map sampling on miss
- [closesthit.rchit](../../rtx-chessboard/shaders/closesthit.rchit): barycentric interpolation, material fetch
- [hw_path_tracer.cpp](../../rtx-chessboard/src/render/hw/hw_path_tracer.cpp): `Trace()` method, command recording

---

## Phase 8A: GLSL Shader Library + Single-Bounce PBR

**Goal:** Port the GLSL utility library from rtx-chessboard and implement full material shading in the closest-hit shader. The raygen uses a single bounce (primary ray + one shadow/reflection) to validate correct PBR output before adding the full bounce loop.

**Source:** rtx-chessboard `shaders/include/*.glsl`, `shaders/closesthit.rchit`

### Tasks

1. Port GLSL shader includes:
   - `shaders/include/brdf.glsl`: Cook-Torrance, GGX NDF, Smith G, Schlick Fresnel
   - `shaders/include/sampling.glsl`: hemisphere sampling, GGX importance sampling, environment CDF sampling
   - `shaders/include/bluenoise.glsl`: spatial-temporal hashing for decorrelation
   - `shaders/include/mis.glsl`: PDF calculations, MIS weight computation

2. Implement full `closesthit.rchit`:
   - Barycentric interpolation of vertex attributes (position, normal, UV, tangent)
   - Fetch `PackedMaterial` from material buffer via instance custom index
   - Sample base color texture if present (bindless array)
   - Apply normal map if present
   - Return hit data in payload (position, normal, material properties, UVs)

3. Update `raygen.rgen` for single-bounce shading:
   - Primary ray → closesthit → evaluate BRDF at hit point
   - Single environment light sample with MIS weight
   - Single area light sample per quad: uniform point on quad, solid-angle PDF, shadow ray, MIS-weighted against BRDF sample. Iterate over the scene's `AreaLight` list from a storage buffer uploaded by GpuScene.
   - Write shaded result to noisy_diffuse output
   - Multi-sample per pixel (N spp, configurable via push constant)
   - Per-sample blue noise scrambling

### Verification
- **Integration test:** render `DamagedHelmet.glb` at 4 spp and 256 spp, compute FLIP score — must be below convergence threshold
- **Golden reference test:** render Cornell box at 256 spp, compare against stored reference with FLIP (mean < 0.05)
- Metallic surfaces show environment reflections
- Normal map details are visible on surfaces
- Texture sampling works (base color maps applied correctly)
- No validation errors; no NaN/Inf in output

### rtx-chessboard Reference
- [closesthit.rchit](../../rtx-chessboard/shaders/closesthit.rchit): material fetch, barycentric interpolation
- [brdf.glsl](../../rtx-chessboard/shaders/include/brdf.glsl): Cook-Torrance implementation
- [sampling.glsl](../../rtx-chessboard/shaders/include/sampling.glsl): importance sampling functions
- [bluenoise.glsl](../../rtx-chessboard/shaders/include/bluenoise.glsl): blue noise hashing
- [mis.glsl](../../rtx-chessboard/shaders/include/mis.glsl): MIS weight computation

---

## Phase 8B: Multi-Bounce MIS + Clear Coat

**Goal:** Extend the raygen shader to a full multi-bounce path tracer with 4-way MIS, Russian roulette, diffuse/specular classification, and clear coat support.

**Source:** rtx-chessboard `shaders/raygen.rgen`, `shaders/include/clearcoat.glsl`

### Tasks

1. Port clear coat GLSL include:
   - `shaders/include/clearcoat.glsl`: dual-layer clearcoat + base BRDF

2. Implement full bounce loop in `raygen.rgen`:
   - Bounce loop (max 4 bounces + 8 transparency bounces)
   - Per-bounce: trace → hit/miss → evaluate BRDF → sample next direction
   - 4-way MIS: diffuse hemisphere, GGX specular, clear coat, environment
   - Per-bounce area light sampling: sample each quad area light, shadow ray, MIS-weighted contribution
   - Russian roulette after bounce 3 (continuation probability = max throughput component)
   - Separate diffuse/specular classification based on first opaque bounce
   - Output: noisy_diffuse, noisy_specular (two separate images)

3. Wire up clear coat in closest-hit:
   - Evaluate clear coat lobe when `clear_coat > 0` in material
   - Return clear coat parameters in payload for MIS strategy selection

### Verification
- **Convergence test:** Cornell box at 4 spp vs 256 spp, FLIP score below threshold (validates multi-bounce GI convergence)
- **Golden reference test:** Cornell box at 256 spp matches stored reference (FLIP mean < 0.05)
- Metallic surfaces show recursive environment reflections (multiple bounces visible)
- Clear coat shows dual-layer effect (`ClearCoatTest.glb` renders correctly)
- Diffuse/specular split: noisy_diffuse and noisy_specular contain separated contributions
- Russian roulette terminates paths without visible bias
- No validation errors; no NaN/Inf in output

### rtx-chessboard Reference
- [raygen.rgen](../../rtx-chessboard/shaders/raygen.rgen): full path tracing loop, MIS, diffuse/specular split
- [clearcoat.glsl](../../rtx-chessboard/shaders/include/clearcoat.glsl): clearcoat BRDF

---

## Phase 8C: Transparency + Transmission + G-Buffer Auxiliary Data + Sub-pixel Jitter

**Goal:** Add transparency handling, physical transmission (Fresnel refraction, IOR, volume attenuation), write all G-buffer auxiliary channels, and implement sub-pixel jitter.

**Source:** rtx-chessboard `shaders/raygen.rgen` (transparency loop, G-buffer writes), `shaders/closesthit.rchit` (alpha handling)

### Tasks

1. Implement transparency and transmission in closest-hit + raygen:
   - Alpha masking (`AlphaMode::kMask`): discard hits below cutoff
   - Alpha blending (`AlphaMode::kBlend`): accumulate through transparency bounces
   - **Physical transmission:** Fresnel-based refraction using `transmission_factor` and `ior` from MaterialDesc
   - Volume attenuation: Beer-Lambert using `attenuation_color` and `attenuation_distance`
   - Thickness handling via `thickness_factor` for thin-surface approximation
   - Transparency bounce limit (8 additional bounces)

2. Write auxiliary G-buffer data in raygen:
   - Motion vectors: project current hit position with previous VP matrix, compute pixel delta
   - Linear depth: view-space Z distance from camera
   - World normals + roughness packed in .w channel
   - Diffuse albedo, specular albedo (for future denoiser demodulation)

3. Implement sub-pixel jitter for anti-aliasing:
   - Halton sequence (base 2, 3) per frame
   - Apply jitter to ray origin in raygen

### Verification
- **Integration test:** render `DragonAttenuation.glb` — transparent dragon shows correct Fresnel refraction, IOR bending, and volume attenuation (color shifts through thicker regions)
- **Integration test:** render `MosquitoInAmber.glb` — nested transmission with embedded geometry visible through amber
- Alpha-masked materials (foliage, fences) show correct cutout holes
- **Convergence test:** Cornell box with transparency at 4 spp vs 256 spp, FLIP below threshold
- Motion vectors are correct: test with camera dolly, verify pixel deltas match expected movement
- Linear depth increases monotonically with distance from camera
- World normals match expected surface orientation
- Sub-pixel jitter smooths edges over multiple frames (visible with accumulation)
- No NaN/Inf values in any G-buffer channel

### rtx-chessboard Reference
- [raygen.rgen](../../rtx-chessboard/shaders/raygen.rgen): transparency loop, G-buffer writes
- [closesthit.rchit](../../rtx-chessboard/shaders/closesthit.rchit): alpha handling

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

### Verification
- **Unit test:** feed known HDR values through ACES + sRGB on CPU, compare GPU output — per-pixel error < 1/255
- **End-to-end golden test:** full pipeline (trace → denoise → tonemap) on Cornell box, FLIP against stored LDR reference (mean < 0.05)
- **End-to-end golden test:** `DamagedHelmet.glb` full pipeline, FLIP against stored reference
- Resize works through the entire pipeline (no validation errors after resize)
- Clean shutdown (no leaks reported by VMA stats or validation layers)

### rtx-chessboard Reference
- [tone_mapper.h/.cpp](../../rtx-chessboard/src/render/tone_mapper.h): pipeline setup, dispatch
- [tonemapping.comp](../../rtx-chessboard/shaders/tonemapping.comp): ACES + sRGB shader
- [main.cpp](../../rtx-chessboard/src/main.cpp): `RenderFrame()` function, blit-to-swapchain

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

These are documented for roadmap visibility but not scheduled:

| Future Phase | Description | Prerequisite |
|---|---|---|
| F1 | ReLAX denoiser (desktop only) | Phase 9A complete |
| F2 | ReSTIR Direct Illumination (desktop only) | Phase 8C complete |
| F3 | Emissive mesh rendering (desktop only) | Phase 8C + F2 (needs ReSTIR for correct sampling) |
| F5 | DLSS-RR denoiser backend | Phase 9A complete |
| F6 | Mobile Vulkan renderer (`monti_vulkan_mobile`) | Phase 8C complete (shared GpuScene/GeometryManager); hybrid rasterization (default) + ray query pipeline; projection-matrix jitter for TAA; format-agnostic G-buffer via `shaderStorageImageReadWithoutFormat` |
| F7 | Metal renderer (C API) | Phase 8C design patterns established |
| F8 | WebGPU renderer (C API → WASM) | Phase 8C design patterns established |
| F9 | ML denoiser training pipeline | Phase 11B complete (training data capture working) |
| F10 | Shader permutation cache | Phase 8C complete |
| F11 | ML denoiser deployment (desktop + mobile) | F9 complete (trained model weights available) |
| F12 | Super-resolution in ML denoiser | F11 complete; uses `ScaleMode` enum in `DenoiserInput` (kQuality 1.5×, kPerformance 2×) |
| F13 | Fragment shader denoiser (mobile) | F6 + F11 complete; denoise → tonemap → present as render pass subpasses; `Denoiser` auto-selects compute vs fragment based on device |

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
  ├─→ Phase 4 (Vulkan context + app scaffolding)
  │     └─→ Phase 5 ─→ ... ─→ Phase 8C
  │                                ├─→ Phase 9B (denoiser integration) ─→ Phase 10A (monti_view: tonemap + present)
  │                                │                                          ├─→ Phase 10B (monti_view: interactive + ImGui)
  │                                │                                          └─→ Phase 11B (monti_datagen: readback + headless)
  │                                └─→ Phase 10A (monti_view: tonemap + present)
  ├─→ Phase 9A (standalone denoiser)              ─→ Phase 9B
  └─→ Phase 11A (capture writer — CPU-only)        ─→ Phase 11B
```

Phases 2 and 4 can be developed in parallel. Phase 9A (standalone denoiser library) can be developed in parallel with Phases 2–8 since it has no Monti dependencies. Phase 11A (capture writer) can also be developed in parallel with Phases 2–10 since it is CPU-only with no GPU dependency. Phase 9B requires both 8C and 9A. Phase 10A (`monti_view` tonemap + present) can start after 8C + 9B. Phase 10B (`monti_view` interactive UI) depends on 10A. Phase 11B (`monti_datagen` headless data generator) depends on 10A + 11A.
