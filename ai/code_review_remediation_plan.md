# Code Review Remediation Plan

*Generated: March 14, 2026 — Post-Phase 11B milestone review*

This plan addresses all findings from the comprehensive code review of the Monti and Deni libraries, their tests, and the monti_view/monti_datagen applications. Work is organized into four phases executed in dependency order.

---

## Phase R1 — Bugs and Correctness Issues

### R1-1. Shadow ray missing optimization flag
**File:** `shaders/raygen.rgen` (~line 593)
**Issue:** Shadow ray uses `gl_RayFlagsTerminateOnFirstHitEXT` but omits `gl_RayFlagsSkipClosestHitShaderEXT`. The closest-hit shader runs unnecessarily on occluded shadow rays.
**Fix:** Change to `gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT`.

### R1-2. Wire jitter into NDC computation
**Files:** `shaders/raygen.rgen` (~line 87), `renderer/src/vulkan/Renderer.cpp` (~line 274)
**Issue:** `jitter_x` and `jitter_y` are set in C++ via `HaltonJitter()` but the raygen shader ignores them, using a hardcoded `+0.5` pixel center offset. The jitter values are needed for temporal accumulation and will be consumed by the future temporal denoiser.
**Fix:** In `raygen.rgen`, replace the hardcoded `+0.5` with `+0.5 + pc.jitter_x` / `+0.5 + pc.jitter_y` in the NDC computation. Verify the Halton sequence in `Renderer.cpp` produces sub-pixel offsets in the correct range ([-0.5, 0.5]).

### R1-3. Mutable `Nodes()` bypasses TLAS dirty tracking
**File:** `scene/include/monti/scene/Scene.h` (~line 51)
**Issue:** Mutable `std::vector<SceneNode>& Nodes()` lets callers modify transforms without incrementing `tlas_generation_`.
**Fix:** Return `const std::vector<SceneNode>&` only. Add a `SetNodeTransform(NodeId, Transform)` method if one doesn't exist, or verify all callers use it. If mutable iteration is needed for non-transform fields, consider a `MutableNodes()` that marks dirty.

### R1-4. `GpuScene` returns index 0 for missing IDs
**File:** `renderer/src/vulkan/GpuScene.cpp` (~lines 187, 201)
**Issue:** `GetMaterialIndex` and `GetMeshAddressIndex` return 0 on lookup failure, silently aliasing the first material/mesh.
**Fix:** Return `UINT32_MAX` as a sentinel value. Add a `constexpr uint32_t kInvalidIndex = UINT32_MAX;` in the header. Log a warning on miss.

### R1-5. `EncodeCustomIndex` silently truncates >4095
**File:** `renderer/src/vulkan/GeometryManager.cpp` (~line 110)
**Issue:** 12-bit mask truncates without warning. Scenes with >4096 meshes or materials get corrupted custom indices.
**Fix:** Add an `assert()` and a runtime warning (stderr) when either index exceeds 4095. Use the named constants from R2-2.

### R1-6. `MaterialDesc::clear_coat_roughness` default is wrong
**File:** `scene/include/monti/scene/Material.h` (~line 73)
**Issue:** Default is `0.1f` but glTF specifies `0.0f`.
**Fix:** Change default to `0.0f`.

### R1-7. GpuReadback ignores VkResult in `BeginOneShot`/`SubmitAndWait`
**File:** `capture/src/GpuReadback.cpp`
**Issue:** `vkAllocateCommandBuffers`, `vkBeginCommandBuffer`, `vkCreateFence`, `vkQueueSubmit`, `vkWaitForFences` return values are all unchecked.
**Fix:** Check VkResult for each call. Return `std::optional<VkCommandBuffer>` from `BeginOneShot` and `bool` from `SubmitAndWait`. Propagate errors to callers.

### R1-8. `ReadbackImage` hardcodes ray tracing stage in restore barrier
**File:** `capture/src/GpuReadback.cpp` (~lines 213-214)
**Issue:** Restore barrier always targets `VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR` regardless of actual consumer.
**Fix:** Add a `dst_stage` parameter (matching the existing `src_stage` parameter). Update `AccumulateFrames` callers to pass the correct stage.

### R1-9. `StagingBuffer::Destroy()` doesn't clear `allocator_`
**File:** `capture/src/GpuReadback.cpp` (~line 119)
**Issue:** After `Destroy()`, `allocator_` holds a potentially stale handle.
**Fix:** Add `allocator_ = VK_NULL_HANDLE;` to `Destroy()`.

### R1-10. `GBufferImages.h` comment is inaccurate
**File:** `app/core/GBufferImages.h` (~lines 11-13)
**Issue:** Comment says only datagen uses `TRANSFER_SRC_BIT`, but view also passes it for debug blitting.
**Fix:** Update comment to say both apps use `TRANSFER_SRC_BIT`.

### R1-11. `CameraParams::aspect_ratio` is unused dead data
**File:** `scene/include/monti/scene/Camera.h` (~line 13)
**Issue:** `ProjectionMatrix` takes its own `aspect` parameter, making this member unused.
**Fix:** Remove `aspect_ratio` from `CameraParams`. Remove all assignments to it in view `main.cpp`, datagen `CameraSetup.h`, and test code.

---

## Phase R2 — Magic Numbers → Named Constants

### R2-1. Renderer G-buffer count
**File:** `renderer/src/vulkan/Renderer.cpp`
**Fix:** Add `constexpr uint32_t kGBufferImageCount = 7;` and replace all 3 instances of the literal `7`: the `std::array<VkImageMemoryBarrier2, 7>` declaration (line 288), the `std::array<VkImage, 7>` declaration (line 289), and the `for` loop bound (line 299).

### R2-2. Custom index encoding constants
**File:** `renderer/src/vulkan/GeometryManager.cpp` AND `shaders/include/constants.glsl`
**Fix:** C++: `constexpr uint32_t kCustomIndexBits = 12; constexpr uint32_t kCustomIndexMask = 0xFFFu;`
GLSL: Add matching `const uint kCustomIndexBits = 12u; const uint kCustomIndexMask = (1u << kCustomIndexBits) - 1u;` to `constants.glsl`. Update `closesthit.rchit` and `anyhit.rahit` to use the shared constants instead of local definitions. Remove the local `kMeshAddrIndexBits`/`kMeshAddrIndexMask` definitions from both files. Add `#include "include/constants.glsl"` to `closesthit.rchit` (it does not currently include it). Verify `anyhit.rahit` already includes it.

### R2-3. Query pool capacity
**File:** `renderer/src/vulkan/GeometryManager.cpp`
**Fix:** `constexpr uint32_t kMinQueryPoolCapacity = 64;`

### R2-4. Luminance coefficients
**File:** `renderer/src/vulkan/EnvironmentMap.cpp`
**Fix:** `constexpr float kLumaR = 0.299f; constexpr float kLumaG = 0.587f; constexpr float kLumaB = 0.114f;`

### R2-5. OpenGL sampler enum values
**File:** `scene/src/gltf/GltfLoader.cpp`
**Fix:** Add named constants only for values used in explicit `case` labels: `kGlClampToEdge = 33071`, `kGlMirroredRepeat = 33648`, `kGlNearest = 9728`, `kGlNearestMipmapNearest = 9984`, `kGlNearestMipmapLinear = 9986`. Drop `kGlLinear`, `kGlLinearMipmapNearest`, and `kGlLinearMipmapLinear` since they fall through to the `default` case.

### R2-6. Normal epsilon
**File:** `scene/src/gltf/GltfLoader.cpp`
**Fix:** `constexpr float kNormalEpsilon = 1e-8f;`

### R2-7. Camera auto-fit constants
**Files:** `app/view/main.cpp` and `app/datagen/CameraSetup.h` (define inline for now; move to shared `app/core/CameraSetup.h` when R3-3 is implemented)
**Fix:** `constexpr float kCameraFitPadding = 1.1f; constexpr float kMinCameraDistance = 0.1f; constexpr float kDefaultNearPlane = 0.01f; constexpr float kDefaultFarPlane = 10000.0f; constexpr float kMinSceneDiagonal = 0.01f; constexpr float kFallbackSceneDiagonal = 10.0f;`
Replace all corresponding magic numbers in both files. The two additional constants (`kMinSceneDiagonal`, `kFallbackSceneDiagonal`) cover the `scene_diagonal < 0.01f` guard and `10.0f` fallback in `app/view/main.cpp`.

### R2-8. Datagen default constants
**File:** `app/datagen/main.cpp`
**Fix:** `constexpr uint32_t kDefaultWidth = 960; constexpr uint32_t kDefaultHeight = 540; constexpr uint32_t kDefaultRefFrames = 64;`

### R2-9. GPU readback pixel size
**File:** `capture/include/monti/capture/GpuReadback.h` AND `capture/src/GpuReadback.cpp`
**Fix:** `constexpr VkDeviceSize kRGBA16FPixelSize = 8;` — use as default parameter in the header. Also update the two hardcoded `8` literals in `GpuReadback.cpp` `AccumulateFrames` (diffuse and specular readback calls) to use `kRGBA16FPixelSize`.

### R2-10. Panels UI constants
**File:** `app/view/Panels.cpp`
**Fix:** `constexpr float kTopBarHeight = 30.0f; constexpr float kSettingsPanelX = 10.0f; constexpr float kSettingsPanelY = 40.0f; constexpr float kSettingsPanelWidth = 320.0f; constexpr int kMaxSppSlider = 64; constexpr float kMinExposure = -10.0f; constexpr float kMaxExposure = 10.0f;`

### R2-11. Shader magic numbers
**File:** `shaders/include/constants.glsl` (add new constants)
```glsl
const float kTexLodMargin = 5.0;
const uint kDebugModeDepth = 3u;
const uint kDebugModeMotionVectors = 4u;
const uint kAlphaModeMask = 1u;
const uint kAlphaModeBlend = 2u;
const float kMotionVectorVizScale = 50.0;
```
Dropped `kAlphaModeOpaque` — opaque is always the implicit default (no explicit check in any shader) and no planned work requires it.
**File:** `shaders/raygen.rgen` — replace 4 instances of `float(mip_levels) - 5.0` with `float(mip_levels) - kTexLodMargin` (base color, metallic-roughness, transmission, normal map). The 5th instance (emissive texture, line 328) is deferred to emissive material work. Replace debug mode literals (`3u`/`4u`), alpha mode literal (`2u`), and motion vector scale (`50.0`).
**File:** `shaders/anyhit.rahit` — replace `1u` alpha mode literal with `kAlphaModeMask`.

### R2-12. Blue noise shader hash primes
**File:** `shaders/include/bluenoise.glsl`
**Fix:** Rename the existing local variables to named constants: `const uint kSpatialHashPrime1 = 73856093u; const uint kSpatialHashPrime2 = 19349663u; const uint kTemporalHashPrime = 251u;`

---

## Phase R3a — App & Scene Code Consolidation

### R3-1. Extract shared `LoadShaderFile` utility
**New file:** `app/core/ShaderFile.h` (inline header-only)
**Action:** Extract `LoadShaderFile(std::string_view path) → std::vector<uint8_t>` into this shared header. Update `RaytracePipeline.cpp` and `ToneMapper.cpp` to use it. Keep the copy in `Denoiser.cpp` since deni must remain independent (zero monti dependency).

### R3-2. Extract `GBufferImages::ToGBuffer()` method
**Files:** `app/core/GBufferImages.h`, `app/core/GBufferImages.cpp`
**Action:** Add a `vulkan::GBuffer ToGBuffer() const` method. Remove the duplicate `MakeGBuffer()` free functions from `app/view/main.cpp` and `app/datagen/GenerationSession.cpp`.

### R3-3. Consolidate `SceneAABB` / `ComputeSceneAABB` / auto-fit camera
**New file:** `app/core/CameraSetup.h` (inline header)
**Action:** Move `SceneAABB`, `ComputeSceneAABB()`, and `AutoFitCamera()` from `app/view/main.cpp` into this header. Update `app/datagen/CameraSetup.h` to re-export or replace with the shared version. Update both mains to use the shared header. Also move the camera constants defined inline in R2-7 (`kCameraFitPadding`, `kMinCameraDistance`, `kDefaultNearPlane`, `kDefaultFarPlane`, `kMinSceneDiagonal`, `kFallbackSceneDiagonal`) into this shared header and remove the inline copies from `app/view/main.cpp` and `app/datagen/CameraSetup.h`.

### R3-4. Extract `Scene::FindById<T>()` template helper
**File:** `scene/src/Scene.cpp`
**Action:** Add a private `FindById` template and replace the 6 identical `find_if` patterns in `GetMesh`, `GetMaterial`, `GetNode`, `GetTexture`.

### R3-5. Extract image barrier helper functions
**New file:** `renderer/include/monti/renderer/VulkanBarriers.h` (inline header-only)
**Issue:** ~20 instances of `VkImageMemoryBarrier2` boilerplate across Upload.cpp, Renderer.cpp, GpuReadback.cpp, ToneMapper.cpp, GBufferImages.cpp, and app/view/main.cpp. Each instance manually fills 8+ struct fields.
**Action:** Add `MakeImageBarrier(VkImage, VkImageLayout old, VkImageLayout new, VkPipelineStageFlags2 src, VkAccessFlags2 src, VkPipelineStageFlags2 dst, VkAccessFlags2 dst, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) → VkImageMemoryBarrier2` and a `CmdPipelineBarrier(VkCommandBuffer, std::span<const VkImageMemoryBarrier2>)` convenience wrapper. Replace boilerplate in monti_vulkan and app code. Leave the `deni_vulkan` copy in `Denoiser.cpp` as-is (deni independence).

### R3-6. Add `Buffer::EnsureCapacity()` method
**Files:** `renderer/include/monti/renderer/Buffer.h`, `renderer/src/vulkan/Buffer.cpp`
**Issue:** 4+ instances of the "check size → Destroy → Create" pattern in GpuScene.cpp (material, mesh-address, area-light buffers) and GeometryManager.cpp (scratch buffer).
**Action:** Add `bool Buffer::EnsureCapacity(VkDeviceSize required_size, VmaAllocator allocator, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage)` that returns `true` if the buffer already has sufficient capacity, otherwise destroys and recreates. Replace the inline capacity-check patterns in GpuScene.cpp and GeometryManager.cpp.

---

## Phase R3b — Shader & Capture Deduplication

### R3-7. Extract texture LOD helper in raygen shader
**File:** `shaders/raygen.rgen`
**Verified:** `nonuniformEXT(idx)` compiles correctly inside a helper function where `idx` is a parameter. `glslc --target-env=vulkan1.2` produces correct `NonUniform` SPIR-V decorations on the sampled image and load instructions. Tested with shaderc v2026.1 / glslang 11.1.0.
**Action:** Define a helper function near the top of `raygen.rgen` (before `main()`) — not in a shared include, since it depends on the `bindless_textures` descriptor binding (set 0, binding 10):
```glsl
float computeTextureLod(uint tex_idx, float ray_cone_lod) {
    ivec2 ts = textureSize(bindless_textures[nonuniformEXT(tex_idx)], 0);
    float tex_lod = 0.5 * log2(float(ts.x) * float(ts.y)) + ray_cone_lod;
    int mip_levels = textureQueryLevels(bindless_textures[nonuniformEXT(tex_idx)]);
    return min(tex_lod, max(float(mip_levels) - kTexLodMargin, 0.0));
}
```
Replace 4 inline instances (base color ~line 212, metallic-roughness ~line 224, transmission ~line 237, normal map ~line 260) with `float tex_lod = computeTextureLod(tex_idx, ray_cone_lod);`. Each call site replaces 4 lines with 1 line.
**Returns:** The clamped LOD value only. Callers still call `textureLod(...)` separately.
**Not touched:** The 5th instance (emissive texture ~line 328) uses hardcoded `5.0` instead of `kTexLodMargin` and is deferred to emissive material work.
**No functional change:** The helper computes the identical value as the inlined code.

### R3-8. Move `kMeshAddrIndexBits`/`kMeshAddrIndexMask` to shared include
**Files:** `shaders/include/constants.glsl`, `shaders/closesthit.rchit`, `shaders/anyhit.rahit`
**Action:** Done as part of R2-2 (constants added to `constants.glsl`, local definitions removed, `#include` added to `closesthit.rchit`). Verify no remaining local definitions and that both shaders compile correctly. This item is a no-op if R2-2 is complete.

### R3-9. Unify EXR channel types and remove redundant write path
**File:** `capture/src/Writer.cpp`
**Issue:** `ChannelEntry` is functionally identical to `ExrChannel` with `is_raw_half=false` — same float data, same pixel_type. This created parallel types (`ChannelEntry` vs `ExrChannel`), parallel append functions (`AppendChannelGroup` vs `AppendFloatChannelGroup`), and parallel write functions (`WriteExr` vs `WriteExrUnified`). The `RawHalfChannelEntry` struct is defined but never used.
**Action — pure code deletion, no new abstractions:**
1. Delete `ChannelEntry` struct (~6 lines).
2. Delete `RawHalfChannelEntry` struct (~4 lines, dead code).
3. Delete `AppendChannelGroup` function (~15 lines) — `AppendFloatChannelGroup` is the superset.
4. Delete `WriteExr` function (~69 lines) — `WriteExrUnified` handles all-float channels identically.
5. Rename `WriteExrUnified` → `WriteExr`.
6. Update `WriteFrame` input section: `std::vector<ChannelEntry>` → `std::vector<ExrChannel>`, `AppendChannelGroup(...)` → `AppendFloatChannelGroup(...)`, `WriteExr(...)` → unchanged (now calls renamed function).
7. Update `WriteFrame` target section: same change.
8. Update `WriteFrameRaw` target section: same change.
**Verification:** `WriteExrUnified` with all `is_raw_half=false` channels sets `pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT` and `requested_pixel_types[i] = ch.pixel_type` — identical to what `WriteExr` did. No functional change.
**Net reduction:** ~90 lines removed, 0 lines of new abstraction.

### ~~R3-10. Deduplicate `AppendChannelGroup` variants~~ — SKIPPED
**File:** `capture/src/Writer.cpp`
**Decision:** After R3-9, two functions remain: `AppendRawHalfChannelGroup` (uint16_t input, ~15 lines) and `AppendFloatChannelGroup` (float input, ~15 lines). A template with `if constexpr` branches would be more complex than two clearly-typed 15-line functions, and would weaken the type safety distinction between half and float inputs (e.g., `AppendRawHalfChannelGroup` intentionally has no `pixel_type` parameter — it always outputs HALF). Skipped per conservative evaluation.

---

## Phase R4 — Priority 1 Test Coverage

Target: raise overall coverage from **87.45% → 90%+** (~100 additional covered lines).

### R4-0. Generate test assets
**Script:** `tests/generate_test_assets.py`
**Output:** Committed `.glb` files in `tests/assets/`
**Action:** Write a Python script using the `struct`/`json` modules (no third-party deps) that generates the following minimal glTF binary files:
1. **`NoNormals.glb`** — A unit quad (4 vertices, 2 triangles) with positions, indices, and tex coords but **no normals accessor**. Also includes a degenerate zero-area triangle (3 vertices at the same position) as a second mesh primitive to exercise the `(0, 1, 0)` fallback in `GenerateFaceWeightedNormals`.
2. **`DiffuseTransmission.glb`** — A single triangle with a material that includes the `KHR_materials_diffuse_transmission` extension in JSON. Extension values: `"diffuseTransmissionFactor": 0.75`, `"diffuseTransmissionColorFactor": [0.8, 0.2, 0.1]`. The material must also declare the extension in `extensionsUsed` at the root level. Uses `cgltf`'s `extensions` array path (not native fields), matching how the loader parses it.
3. **`NoMaterial.glb`** — A single triangle primitive with positions and indices but **no material reference** (`"material"` key absent from the primitive JSON).

Run the script once and commit the three `.glb` files to `tests/assets/`. The script itself lives in `tests/` for reproducibility.

### R4-1. GltfLoader `GenerateFaceWeightedNormals()` test
**File:** `tests/gltf_loader_test.cpp` (new TEST_CASE, tag `[gltf][normals]`)
**Asset:** `tests/assets/NoNormals.glb` (generated by R4-0)
**Action:** Load `NoNormals.glb` via `LoadGltf()`. The loader calls the anonymous-namespace `GenerateFaceWeightedNormals` when it detects a mesh primitive without a normals accessor. Verify:
- Load succeeds (`result.success`)
- Quad mesh: vertices have non-zero normals pointing in the expected direction (the unit quad lies in XZ, so normals should point along ±Y). Use `glm::dot(normal, expected) > 0.99f` tolerance.
- Degenerate mesh: all vertices get the `(0, 1, 0)` fallback normal (from the `len <= kNormalEpsilon` branch)
- Access vertex normals via `result.mesh_data[i].vertices[j].normal`
- **Uncovered lines:** GltfLoader.cpp 61-85 (~25 lines)

### R4-2. GltfLoader `KHR_materials_diffuse_transmission` test
**File:** `tests/gltf_loader_test.cpp` (new TEST_CASE, tag `[gltf][diffuse_transmission]`)
**Asset:** `tests/assets/DiffuseTransmission.glb` (generated by R4-0)
**Why not a Cornell Box variation:** The uncovered code (GltfLoader.cpp lines 325-392) is the glTF JSON extension parser invoked by `LoadGltf()`. Building a scene programmatically via `Scene::AddMaterial()` would bypass the loader entirely and not exercise the uncovered lines.
**Action:** Load `DiffuseTransmission.glb` via `LoadGltf()`. Verify:
- Load succeeds
- `scene.Materials()` has exactly 1 material
- `mat.diffuse_transmission_factor` == `0.75f` (use `WithinAbs(0.75, 1e-5)`)
- `mat.diffuse_transmission_color` == `glm::vec3(0.8f, 0.2f, 0.1f)` (check each component with `WithinAbs`)
- `mat.thin_surface` == `true`
- **Uncovered lines:** GltfLoader.cpp 325-392 (~60 lines)

### R4-3. GltfLoader `GetOrCreateDefaultMaterial` test
**File:** `tests/gltf_loader_test.cpp` (new TEST_CASE, tag `[gltf][default_material]`)
**Asset:** `tests/assets/NoMaterial.glb` (generated by R4-0)
**Action:** Load `NoMaterial.glb` via `LoadGltf()`. The loader calls `GetOrCreateDefaultMaterial` when a primitive has no material reference. Verify:
- Load succeeds
- `scene.Materials()` contains exactly 1 material
- Material `name` == `"default"`
- `base_color` == `glm::vec3(1.0f, 1.0f, 1.0f)`
- `roughness` == `0.5f`
- `metallic` == `0.0f`
- **Uncovered lines:** GltfLoader.cpp 398-412 (~15 lines)

### R4-4. `Image` move-assignment test
**File:** `tests/gpu_scene_test.cpp` (new TEST_CASE, tag `[vulkan][integration][image]`)
**Action:** Using the existing `TestContext` pattern, create two `Image` objects via `Image::Create()` (small 4×4 RGBA8 images). Move-assign one to the other. Verify:
- Destination holds valid handles (image, view non-null)
- Destination dimensions match the source's original dimensions
- Source is in moved-from state (image/view are `VK_NULL_HANDLE`, width/height are 0)
- Self-move-assignment is a no-op (assign `img` to itself, verify handles unchanged)
- Both images clean up correctly (no leak/double-free — validated by VMA and Vulkan validation layers in CI)
- **Uncovered lines:** Image.cpp 37-64 (~28 lines)

### ~~R4-5. `GpuScene::ToVkFormat` comprehensive test~~ — SKIPPED
**Decision:** `ToVkFormat` is `private static` on `GpuScene`. Testing it indirectly requires creating synthetic textures with various `PixelFormat` values and uploading them through the full GPU pipeline — significant complexity for 8 lines of a trivial switch statement. Skipped per conservative evaluation. The 8 uncovered lines are format mapping cases that are implicitly validated when real assets use those formats.

### Estimated coverage impact
- R4-1: ~25 lines
- R4-2: ~60 lines
- R4-3: ~15 lines
- R4-4: ~28 lines
- ~~R4-5: ~8 lines~~ (skipped)
- **Total: ~128 lines → coverage increase to ~90.7%**

---

## Phase R5 — Move Per-Frame Constants to UBO

Refactor `PushConstants` to move per-frame-constant data into a uniform buffer object. Currently at 248/256 bytes with only 8 bytes of headroom. Camera matrices alone consume 192 bytes of push constant space despite changing at most once per frame.

### R5-1. Create `FrameUniforms` UBO struct
**New files:** `renderer/src/vulkan/FrameUniforms.h`, `shaders/include/frame_uniforms.glsl`
**Action:** Define a `FrameUniforms` C++ struct in the internal renderer directory (alongside `RaytracePipeline.h`, `Buffer.h`, etc.) containing all fields moved from push constants:

| Field               | C++ Type     | GLSL Type | Bytes | std140 Offset |
|---------------------|-------------|-----------|-------|---------------|
| `inv_view`          | `glm::mat4` | `mat4`    | 64    | 0             |
| `inv_proj`          | `glm::mat4` | `mat4`    | 64    | 64            |
| `prev_view_proj`    | `glm::mat4` | `mat4`    | 64    | 128           |
| `env_width`         | `uint32_t`  | `uint`    | 4     | 192           |
| `env_height`        | `uint32_t`  | `uint`    | 4     | 196           |
| `env_avg_luminance` | `float`     | `float`   | 4     | 200           |
| `env_max_luminance` | `float`     | `float`   | 4     | 204           |
| `env_rotation`      | `float`     | `float`   | 4     | 208           |
| `skybox_mip_level`  | `float`     | `float`   | 4     | 212           |
| `jitter_x`          | `float`     | `float`   | 4     | 216           |
| `jitter_y`          | `float`     | `float`   | 4     | 220           |
| `area_light_count`  | `uint32_t`  | `uint`    | 4     | 224           |
| *(pad)*             | `uint32_t`  | `uint`    | 4     | 228           |

Total: **232 bytes**. The trailing pad aligns the struct to 16 bytes (std140 requires the overall struct size to round up to the alignment of its largest member, `mat4` = 16 bytes). Final size: 232 rounds up to **240 bytes** to satisfy std140 `mat4` base alignment. *(Note: re-check during implementation — `glm::mat4` columns are `vec4` with 16-byte alignment, so std140 struct size must be a multiple of 16.)*

**C++ header (`FrameUniforms.h`):**
```cpp
#pragma once
#include <glm/glm.hpp>
#include <cstdint>

namespace monti::vulkan {

struct FrameUniforms {
    glm::mat4 inv_view;              // 64 bytes, offset 0
    glm::mat4 inv_proj;              // 64 bytes, offset 64
    glm::mat4 prev_view_proj;        // 64 bytes, offset 128

    uint32_t env_width;              // 4 bytes, offset 192
    uint32_t env_height;             // 4 bytes, offset 196
    float    env_avg_luminance;      // 4 bytes, offset 200
    float    env_max_luminance;      // 4 bytes, offset 204

    float    env_rotation;           // 4 bytes, offset 208
    float    skybox_mip_level;       // 4 bytes, offset 212
    float    jitter_x;              // 4 bytes, offset 216
    float    jitter_y;              // 4 bytes, offset 220

    uint32_t area_light_count;       // 4 bytes, offset 224
    uint32_t pad0;                   // 4 bytes, offset 228 (std140 alignment)
    uint32_t pad1;                   // 4 bytes, offset 232
    uint32_t pad2;                   // 4 bytes, offset 236
};

static_assert(sizeof(FrameUniforms) == 240);
static_assert(sizeof(FrameUniforms) % 16 == 0, "std140 requires struct size multiple of 16");

}  // namespace monti::vulkan
```

**GLSL include (`shaders/include/frame_uniforms.glsl`):**
```glsl
// Frame-constant uniform buffer — matches C++ FrameUniforms struct (std140).
layout(std140, set = 0, binding = 16) uniform FrameUniformsBlock {
    mat4  inv_view;
    mat4  inv_proj;
    mat4  prev_view_proj;

    uint  env_width;
    uint  env_height;
    float env_avg_luminance;
    float env_max_luminance;

    float env_rotation;
    float skybox_mip_level;
    float jitter_x;
    float jitter_y;

    uint  area_light_count;
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
} frame;
```
**Accessed as:** `frame.inv_view`, `frame.env_width`, etc. (replacing `pc.inv_view`, `pc.env_width`).

### R5-2. Allocate and bind the per-frame UBO
**Files:** `renderer/src/vulkan/Renderer.cpp`, `renderer/src/vulkan/RaytracePipeline.h`, `renderer/src/vulkan/RaytracePipeline.cpp`

**Buffer ownership:** Renderer owns a single `Buffer` member in `Renderer::Impl` for the frame UBO. Follows the same pattern as material/mesh-address/area-light buffers in `GpuScene` (single-buffered, host-visible, written each frame via Map/memcpy/Unmap).

**Why single-buffered (no ring):** `RenderFrame()` is always called after the prior frame's fence wait completes (view: `WaitForFence(current_frame)`, datagen: `WaitIdle()`). The comment at Renderer.cpp line 122 ("cmd has completed by now") confirms this contract. All existing host-visible buffers rely on the same guarantee. No ring-buffering needed.

**No explicit pipeline barrier needed:** With `VMA_MEMORY_USAGE_AUTO` + `VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT`, VMA selects HOST_COHERENT memory on discrete GPUs. The fence-wait → memcpy → submit ordering provides the required host-write-before-device-read guarantee, identical to the material/mesh-address/area-light buffers. **Mobile note:** This assumes HOST_COHERENT memory. On a hypothetical non-coherent allocation, `vmaFlushAllocation` would be required after Unmap — but the same applies to the material/mesh-address/area-light buffers, so all four should be addressed together if mobile support is added. Add a `// HOST_COHERENT assumed — see R5-2 mobile note` comment at the Map/Unmap site.

**Descriptor set changes (set 0, binding 16):**

1. **Layout** (`CreateDescriptorSetLayout` in `RaytracePipeline.cpp`):
   - Change `std::array<VkDescriptorSetLayoutBinding, 16>` → `std::array<..., 17>`.
   - Add binding 16: `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER`, count 1, stage `VK_SHADER_STAGE_RAYGEN_BIT_KHR`.
   - Extend `binding_flags` array to 17 entries with `kUpdateAfterBind`.

2. **Pool** (`CreateDescriptorPool`):
   - Add a 5th pool size: `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER`, `descriptorCount = 1`.
   - Change `std::array<VkDescriptorPoolSize, 4>` → `std::array<..., 5>`.

3. **`DescriptorUpdateInfo`** (in `RaytracePipeline.h`):
   - Add: `VkBuffer frame_uniforms_buffer;` and `VkDeviceSize frame_uniforms_buffer_size;`

4. **`UpdateDescriptors`**:
   - Add a `VkDescriptorBufferInfo` for binding 16 and append a write via `add_write(16, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, &ubo_info, nullptr)`.
   - Update `writes.reserve(16)` → `writes.reserve(17)`.

5. **`Renderer::Impl`**:
   - Add `Buffer frame_ubo_;` member.
   - In `RenderFrame`, after populating `FrameUniforms`, call `frame_ubo_.Map()` / `memcpy` / `Unmap()`.
   - Create the buffer once (224 bytes, `VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT`, `VMA_MEMORY_USAGE_AUTO`, `VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT`). Create during first `RenderFrame` or in renderer initialization.
   - Pass `frame_ubo_.Handle()` and `frame_ubo_.Size()` in `DescriptorUpdateInfo`.

### R5-3. Slim down `PushConstants` to per-dispatch data
**Files:** `renderer/src/vulkan/RaytracePipeline.h`, `renderer/src/vulkan/RaytracePipeline.cpp`, `shaders/raygen.rgen`, `renderer/src/vulkan/Renderer.cpp`

**Remaining PushConstants struct (C++):**
```cpp
struct PushConstants {
    uint32_t frame_index;            // 4 bytes, offset 0
    uint32_t paths_per_pixel;        // 4 bytes, offset 4
    uint32_t max_bounces;            // 4 bytes, offset 8
    uint32_t debug_mode;             // 4 bytes, offset 12
};

static_assert(sizeof(PushConstants) == 16);
```
`area_light_count` moves to the UBO. `pad0` removed. Total: **16 bytes** (down from 248).

**Remaining GLSL push_constant block:**
```glsl
layout(push_constant) uniform PushConstants {
    uint  frame_index;
    uint  paths_per_pixel;
    uint  max_bounces;
    uint  debug_mode;
} pc;
```

**Push constant stage flags:** Narrow from all 4 RT stages to `VK_SHADER_STAGE_RAYGEN_BIT_KHR` only. Only `raygen.rgen` declares and reads the push_constant block — `closesthit.rchit`, `anyhit.rahit`, and `miss.rmiss` have no push_constant declaration. Update both:
- `VkPushConstantRange.stageFlags` in `CreatePipelineAndLayout()` (~line 314)
- `vkCmdPushConstants` call in `Renderer.cpp` (~line 332)

Add a comment: `// Only raygen reads push constants; other stages use descriptors.`

**Shader updates in `raygen.rgen`:**
Replace all `pc.*` references for moved fields with `frame.*` references:
- `pc.inv_view` → `frame.inv_view`
- `pc.inv_proj` → `frame.inv_proj`
- `pc.prev_view_proj` → `frame.prev_view_proj`
- `pc.jitter_x` / `pc.jitter_y` → `frame.jitter_x` / `frame.jitter_y`
- `pc.env_width` / `pc.env_height` → `frame.env_width` / `frame.env_height`
- `pc.env_avg_luminance` / `pc.env_max_luminance` → `frame.env_avg_luminance` / `frame.env_max_luminance`
- `pc.env_rotation` → `frame.env_rotation`
- `pc.skybox_mip_level` → `frame.skybox_mip_level`
- `pc.area_light_count` → `frame.area_light_count`

Add `#include "include/frame_uniforms.glsl"` to `raygen.rgen` (after existing includes).

**Renderer.cpp updates:**
Split the current `PushConstants pc{}` population into two sections:
1. Populate `FrameUniforms` → map UBO → memcpy → unmap
2. Populate slimmed `PushConstants` (4 fields only)

### ~~R5-4. Update Denoiser descriptor interface~~ — SKIPPED
**Decision:** The denoiser ([Denoiser.cpp](denoise/src/vulkan/Denoiser.cpp)) uses only 3 storage image bindings and reads none of the fields being moved (`prev_view_proj`, `jitter`, `env_*`, `area_light_count`). This becomes relevant when implementing a temporal denoiser. Skipped — no loss of functionality.

---

## Execution Notes

- **Dependency order:** R1 (bugs) and R2 (constants) can be done in parallel. R3a/R3b (dedup) depend on R2 for named constants. R4 (tests) should run last to validate all prior changes. R5 (UBO refactor) is independent and can follow R4 or be done in parallel with R3/R4.
- **Session plan (6 sessions):** R1 → R2 → R3a → R3b → R4 → R5
- **Build verification:** Run full build + test suite after each phase.
- **deni independence:** Never add monti dependencies to the deni_vulkan library. The `LoadShaderFile` copy in `Denoiser.cpp` and the `ScaleMode` duplicate in `Denoiser.h` stay as-is.
- **Shader changes:** After modifying GLSL files, recompile all SPIR-V and run the GPU integration tests (phase7c, phase8b, phase8h, phase10a).
- **Push constant changes (R1-2):** Wire jitter into NDC computation. Must update both GLSL and C++ sides atomically and recompile shaders before running tests.
- **UBO refactor (R5):** Changes the descriptor set layout and push constant struct simultaneously. Update both GLSL and C++ atomically. All shader stages that reference moved fields must be updated in the same commit.
