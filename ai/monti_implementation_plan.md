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
   - `linear_depth`: R16F (recommended) or RGBA16F, `VK_IMAGE_USAGE_STORAGE_BIT`
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

**Goal:** Replace the Phase 7B skeleton shaders with initial functional shaders and wire up `Renderer::RenderFrame()` so that rays are cast and the environment map is visible on screen. The descriptor layout, push constants, pipeline structure, and SBT from Phase 7B are unchanged — only the shader source files are updated.

**Source:** rtx-chessboard `shaders/raygen.rgen`, `shaders/miss.rmiss`, `shaders/closesthit.rchit`

### Tasks

1. Update `shaders/raygen.rgen` (replace skeleton logic):
   - Compute ray origin and direction from pixel + camera inverse matrices (using `inv_view`, `inv_proj` from push constants)
   - Single sample per pixel (no MIS yet, no bounce loop)
   - `traceRayEXT()` → miss returns environment map sample
   - Write result to noisy_diffuse output

2. Update `shaders/miss.rmiss` (replace skeleton logic):
   - Sample environment map using ray direction
   - Apply basic importance sampling with pre-computed CDFs
   - Write to payload

3. Update `shaders/closesthit.rchit` (replace skeleton logic):
   - Return a flat shaded color (e.g., normal as color) to verify hits work
   - Full material shading deferred to Phase 8A

4. Implement `Renderer::RenderFrame()`:
   - Call `GpuScene::UpdateAreaLights()` and `GpuScene::UpdateMaterials()` when scene is dirty
   - Update descriptor sets with current resources (TLAS, G-buffer images, buffers)
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
   - Single area light sample per quad: uniform point on quad, solid-angle PDF, shadow ray, MIS-weighted against BRDF sample. Iterate over the scene's `AreaLight` list from the area light storage buffer (descriptor binding 11, uploaded by `GpuScene::UpdateAreaLights()`). Loop count controlled by `push_constants.area_light_count`.
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
