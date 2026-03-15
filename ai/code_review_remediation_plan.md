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
**Fix:** Add `constexpr uint32_t kGBufferImageCount = 7;` and use it in the barrier array size and loop.

### R2-2. Custom index encoding constants
**File:** `renderer/src/vulkan/GeometryManager.cpp` AND `shaders/include/constants.glsl`
**Fix:** C++: `constexpr uint32_t kCustomIndexBits = 12; constexpr uint32_t kCustomIndexMask = 0xFFFu;`
GLSL: Add matching `const uint kCustomIndexBits = 12u; const uint kCustomIndexMask = (1u << kCustomIndexBits) - 1u;` to `constants.glsl`. Update `closesthit.rchit` and `anyhit.rahit` to use the shared constants instead of local definitions.

### R2-3. Query pool capacity
**File:** `renderer/src/vulkan/GeometryManager.cpp`
**Fix:** `constexpr uint32_t kMinQueryPoolCapacity = 64;`

### R2-4. Luminance coefficients
**File:** `renderer/src/vulkan/EnvironmentMap.cpp`
**Fix:** `constexpr float kLumaR = 0.299f; constexpr float kLumaG = 0.587f; constexpr float kLumaB = 0.114f;`

### R2-5. OpenGL sampler enum values
**File:** `scene/src/gltf/GltfLoader.cpp`
**Fix:** Add named constants: `kGlClampToEdge = 33071`, `kGlMirroredRepeat = 33648`, `kGlNearest = 9728`, `kGlLinear = 9729`, `kGlNearestMipmapNearest = 9984`, `kGlLinearMipmapNearest = 9985`, `kGlNearestMipmapLinear = 9986`, `kGlLinearMipmapLinear = 9987`.

### R2-6. Normal epsilon
**File:** `scene/src/gltf/GltfLoader.cpp`
**Fix:** `constexpr float kNormalEpsilon = 1e-8f;`

### R2-7. Camera auto-fit constants
**File:** `app/core/CameraSetup.h` (new shared location, see R3-3)
**Fix:** `constexpr float kCameraFitPadding = 1.1f; constexpr float kMinCameraDistance = 0.1f; constexpr float kDefaultNearPlane = 0.01f; constexpr float kDefaultFarPlane = 10000.0f;`

### R2-8. Datagen default constants
**File:** `app/datagen/main.cpp`
**Fix:** `constexpr uint32_t kDefaultWidth = 960; constexpr uint32_t kDefaultHeight = 540; constexpr uint32_t kDefaultRefFrames = 64;`

### R2-9. GPU readback pixel size
**File:** `capture/include/monti/capture/GpuReadback.h`
**Fix:** `constexpr VkDeviceSize kRGBA16FPixelSize = 8;` — use as default parameter.

### R2-10. Panels UI constants
**File:** `app/view/Panels.cpp`
**Fix:** `constexpr float kTopBarHeight = 30.0f; constexpr float kSettingsPanelX = 10.0f; constexpr float kSettingsPanelY = 40.0f; constexpr float kSettingsPanelWidth = 320.0f; constexpr int kMaxSppSlider = 64; constexpr float kMinExposure = -10.0f; constexpr float kMaxExposure = 10.0f;`

### R2-11. Shader magic numbers
**File:** `shaders/include/constants.glsl` (add new constants)
```glsl
const float kTexLodMargin = 5.0;
const uint kDebugModeDepth = 3u;
const uint kDebugModeMotionVectors = 4u;
const uint kAlphaModeOpaque = 0u;
const uint kAlphaModeMask = 1u;
const uint kAlphaModeBlend = 2u;
const float kMotionVectorVizScale = 50.0;
```
**File:** `shaders/raygen.rgen` — replace all 5 instances of `float(mip_levels) - 5.0` with `float(mip_levels) - kTexLodMargin`, replace debug mode literals, alpha mode literals.
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
**Action:** Move `SceneAABB`, `ComputeSceneAABB()`, and `AutoFitCamera()` from `app/view/main.cpp` into this header. Update `app/datagen/CameraSetup.h` to re-export or replace with the shared version. Update both mains to use the shared header.

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
**Action:** Define a `float computeTextureLod(uint tex_idx, float ray_cone_lod)` function near the top. Replace all 5 inline instances.

### R3-8. Move `kMeshAddrIndexBits`/`kMeshAddrIndexMask` to shared include
**Files:** `shaders/include/constants.glsl`, `shaders/closesthit.rchit`, `shaders/anyhit.rahit`
**Action:** Move definitions to `constants.glsl` (done as part of R2-2). Remove local definitions from closesthit and anyhit. Add `#include "include/constants.glsl"` to both if not already present.

### R3-9. Factor `WriteExr` and `WriteExrUnified` shared code
**File:** `capture/src/Writer.cpp`
**Action:** Extract a shared `WriteExrCore(header, image, channels, filepath)` helper that handles `InitEXRHeader`, compression, `SaveEXRImageToFile`, and cleanup. Both `WriteExr` and `WriteExrUnified` become thin wrappers.

### R3-10. Deduplicate `AppendChannelGroup` variants
**File:** `capture/src/Writer.cpp`
**Action:** Template the three `AppendChannelGroup` / `AppendRawHalfChannelGroup` / `AppendFloatChannelGroup` functions into a single template or use a common inner loop with a deinterleave callback.

---

## Phase R4 — Priority 1 Test Coverage

Target: raise overall coverage from **87.45% → 90%+** (~100 additional covered lines).

### R4-1. GltfLoader `ComputeNormals()` test
**File:** `tests/gltf_loader_test.cpp` (new TEST_CASE)
**Action:** Create a minimal glTF (or load an asset without normals) that exercises `ComputeNormals`. Verify:
- Area-weighted normals point in expected direction for a known quad
- Degenerate zero-area triangle gets `(0, 1, 0)` fallback
- **Uncovered lines:** GltfLoader.cpp 61-85 (~25 lines)

### R4-2. GltfLoader `KHR_materials_diffuse_transmission` test
**File:** `tests/gltf_loader_test.cpp` (new TEST_CASE)
**Action:** Load a glTF asset with `KHR_materials_diffuse_transmission` extension. Verify:
- `diffuse_transmission_factor` correctly parsed
- `diffuse_transmission_color` correctly parsed
- `thin_surface` flag set to `true`
- **Uncovered lines:** GltfLoader.cpp 335-396 (~60 lines)

### R4-3. GltfLoader `GetOrCreateDefaultMaterial` test
**File:** `tests/gltf_loader_test.cpp` (new TEST_CASE)
**Action:** Load a glTF with a primitive that has no material assigned. Verify:
- Scene contains a material named `"default"`
- Default material has white base color, roughness 0.5, metallic 0.0
- **Uncovered lines:** GltfLoader.cpp 397-412 (~15 lines)

### R4-4. `Image` move-assignment test
**File:** `tests/gpu_scene_test.cpp` or new test file
**Action:** Create two `Image` objects on the GPU. Move-assign one to the other. Verify:
- Destination holds valid handles
- Source is zeroed out (moved-from state)
- Self-assignment is a no-op
- **Uncovered lines:** Image.cpp 37-64 (~28 lines)

### R4-5. `GpuScene::ToVkFormat` comprehensive test
**File:** `tests/gpu_scene_test.cpp` (new TEST_CASE)
**Action:** Upload materials with textures whose pixel formats cover the tail cases: `kRG16F`, `kRG16_SNORM`, `kR32F`, `kR8_UNORM`. Verify correct descriptor image info formats after upload.
- **Uncovered lines:** GpuScene.cpp 357-364 (~8 lines)

### Estimated coverage impact
- R4-1: ~25 lines
- R4-2: ~60 lines
- R4-3: ~15 lines
- R4-4: ~28 lines
- R4-5: ~8 lines
- **Total: ~136 lines → coverage increase to ~91%**

---

## Phase R5 — Move Per-Frame Constants to UBO

Refactor `PushConstants` to move per-frame-constant data into a uniform buffer object. Currently at 248/256 bytes with only 8 bytes of headroom. Camera matrices alone consume 192 bytes of push constant space despite changing at most once per frame.

### R5-1. Create `FrameUniforms` UBO struct
**New files:** `renderer/include/monti/renderer/FrameUniforms.h`, `shaders/include/frame_uniforms.glsl`
**Action:** Define a `FrameUniforms` struct containing:
- `inv_view` (mat4, 64 bytes)
- `inv_proj` (mat4, 64 bytes)
- `prev_view_proj` (mat4, 64 bytes)
- `env_width` (uint32)
- `env_height` (uint32)
- `env_avg_luminance` (float)
- `env_max_luminance` (float)
- `env_rotation` (float)
- `skybox_mip_level` (float)
- `jitter_x` (float)
- `jitter_y` (float)

Total: 224 bytes (std140-aligned). Add matching GLSL uniform block in `frame_uniforms.glsl`. Add `static_assert` on C++ struct size/alignment.

### R5-2. Allocate and bind the per-frame UBO
**Files:** `renderer/src/vulkan/Renderer.cpp`, `renderer/src/vulkan/RaytracePipeline.h`, `renderer/src/vulkan/RaytracePipeline.cpp`
**Action:**
- Add a `Buffer` member to the renderer for the frame UBO (double-buffered or persistently-mapped).
- Add a descriptor set binding (set 1, binding 0) for the UBO in the raytrace pipeline layout.
- Each frame: map the buffer, write the `FrameUniforms` struct, flush.
- Update `VkDescriptorSetLayoutBinding` and pipeline layout creation.

### R5-3. Slim down `PushConstants` to per-dispatch data
**Files:** `renderer/src/vulkan/RaytracePipeline.h`, `shaders/raygen.rgen`, `renderer/src/vulkan/Renderer.cpp`
**Action:** Remove the fields moved to the UBO from `PushConstants`. The remaining struct should contain only:
- `frame_index` (uint32)
- `paths_per_pixel` (uint32)
- `max_bounces` (uint32)
- `area_light_count` (uint32)
- `debug_mode` (uint32)

Total: ~20 bytes (down from 248). Update `static_assert`, `vkCmdPushConstants` call, and the GLSL `push_constant` block. Update all shader stages that reference the moved fields to read from the UBO instead.

### R5-4. Update Denoiser descriptor interface
**Files:** `denoise/src/vulkan/Denoiser.cpp`, `shaders/denoise.comp` (if applicable)
**Action:** If the denoiser needs access to `prev_view_proj` or jitter for temporal denoising, add a UBO binding to its own descriptor set layout. The denoiser should accept a `VkBuffer` handle via its existing interface — do not create a monti dependency.

---

## Execution Notes

- **Dependency order:** R1 (bugs) and R2 (constants) can be done in parallel. R3a/R3b (dedup) depend on R2 for named constants. R4 (tests) should run last to validate all prior changes. R5 (UBO refactor) is independent and can follow R4 or be done in parallel with R3/R4.
- **Session plan (6 sessions):** R1 → R2 → R3a → R3b → R4 → R5
- **Build verification:** Run full build + test suite after each phase.
- **deni independence:** Never add monti dependencies to the deni_vulkan library. The `LoadShaderFile` copy in `Denoiser.cpp` and the `ScaleMode` duplicate in `Denoiser.h` stay as-is.
- **Shader changes:** After modifying GLSL files, recompile all SPIR-V and run the GPU integration tests (phase7c, phase8b, phase8h, phase10a).
- **Push constant changes (R1-2):** Wire jitter into NDC computation. Must update both GLSL and C++ sides atomically and recompile shaders before running tests.
- **UBO refactor (R5):** Changes the descriptor set layout and push constant struct simultaneously. Update both GLSL and C++ atomically. All shader stages that reference moved fields must be updated in the same commit.
