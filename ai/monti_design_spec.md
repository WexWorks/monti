# Monti — Cross-Platform Path Tracing Renderer & Deni Denoiser — Architecture Design (v6)

> **Purpose:** This document defines the architecture for **Monti**, a physically-based path tracing renderer, and **Deni**, a standalone ML-ready denoiser library. Deni is the primary product — a lightweight, platform-native library (Vulkan + VMA) that any engine developer can integrate. Monti is internal tooling for generating denoiser training data and serves as a reference implementation / demo for customers. Both are designed to be clean, simple, and engine-friendly.
>
> **Architecture summary:** There is no Hardware Abstraction Layer. All GPU components use native platform types directly (Vulkan, Metal, WebGPU). A shared, platform-agnostic scene layer describes *what to render* (entities, transforms, materials, lights, cameras). Platform-specific renderer implementations describe *how to render it* (acceleration structures, ray tracing, GPU resource management). The denoiser is an independent library with no dependency on the renderer or scene layer. The host application owns all GPU resources, the frame loop, and the command buffer.
>
> **Naming:** "Monti" is a variation of "Monte Carlo". "Deni" is short for "denoiser". The Vulkan implementations use C++ interfaces. Future Metal and WebGPU implementations will expose a pure-C API for Swift and TypeScript/JavaScript (via emscripten WASM) interop.
>
> **Reference implementation:** The existing [rtx-chessboard](../../../rtx-chessboard/) Vulkan path tracer serves as the baseline for Monti's initial Vulkan implementation. Its architecture (abstract `PathTracer`/`Denoiser` interfaces, `GPUScene` upload, BLAS/TLAS management, MIS path tracing, bindless textures, push constants) is directly adapted into Monti's library structure.

---

## 1. Design Goals and Constraints

| Goal | Constraint |
|---|---|
| Deni is the product | Minimal dependencies (Vulkan + VMA), native platform types, frictionless integration |
| Monti is internal tooling | Training data generation, reference implementation, demo for customers |
| No HAL | All GPU components use native types directly; no intermediate handle system |
| Platform-agnostic scene | Entities, transforms, materials, textures, lights, cameras as CPU-side data |
| Platform-specific GPU work | Renderer and denoiser are per-platform; host provides native GPU resources |
| Host owns everything | Device, command buffers, frame lifecycle, GPU buffers — all host-owned |
| glTF 2.0 PBR materials | CPU-side material data in scene layer; renderer creates GPU representations |
| GPU buffer geometry | Host provides native GPU vertex/index buffers with device addresses; renderer references them directly, no CPU roundtrip. Optional upload helpers (`GpuBufferUtils.h`) for hosts without existing GPU buffer management |
| Incremental migration | Existing rtx-chessboard Vulkan path tracer adapted pass-by-pass |
| Identical interfaces | All platform implementations follow the same interface shape, differing only in native GPU types (see §4.6, §6.4) |

### Non-Goals (initial release)
- Custom material shaders (glTF PBR only)
- Multi-view / stereo cameras
- Streaming G-buffer capture (synchronous file output is sufficient)
- Vulkan mobile renderer (designed for, implemented later; see §6.5)
- Metal renderer (designed for, implemented later)
- WebGPU renderer (designed for, implemented later; requires WebGPU ray tracing API)
- ReSTIR-based emissive mesh importance sampling (see roadmap F3; basic emissive extraction + hybrid WRS is in Phases 8J/8K)
- Video capture output (OpenEXR sequences are sufficient initially)
- NRD denoiser (deferred; see roadmap F16 — only if cross-vendor denoising needed before ML denoiser)
- ReSTIR DI (desktop initially; designed for, implemented later; see roadmap F2)

> **Scope note:** The "initial release" covers Phases 1–8N + 9A–9D + 10A–10B + 11A–11B — all complete. This includes near-term quality improvements (firefly filter, ray cones, sphere/triangle lights, diffuse transmission, nested dielectrics, emissive extraction, hybrid WRS with direct-sample fallback, KHR_texture_transform, KHR_materials_sheen, DDS texture loading) that were added based on RTXPT comparison analysis. Phases beyond 8N (ReSTIR, volumes, skinning) remain deferred to the roadmap. See [monti_implementation_plan_completed.md](monti_implementation_plan_completed.md) for detailed phase specifications and [roadmap.md](roadmap.md) for future work. DLSS-RR is integrated at the app level in `monti_view` (F1) as a quality reference; NRD ReLAX is deferred (F16); ReBLUR is not planned.

---

## 2. System Architecture

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                        PRODUCT (Deni)                           │
  │                                                                  │
  │  deni_vulkan           Standalone denoiser, native Vulkan types  │
  │  deni_metal            (future) native Metal types               │
  │  deni_webgpu           (future) native WebGPU types              │
  │                                                                  │
  │  No dependencies on renderer, scene, or any internal tooling.   │
  │  Customer links one library, includes one header.                │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │                    INTERNAL TOOLING (Monti)                     │
  │                                                                  │
  │  monti_scene           Platform-agnostic data model              │
  │                        (transforms, materials, textures,         │
  │                         lights, cameras)                         │
  │                                                                  │
  │  monti_vulkan          Path tracer (Vulkan desktop, RT pipeline) │
  │  monti_vulkan_mobile   (future) ray query in compute             │
  │  monti_metal           (future) Metal RT                         │
  │  monti_webgpu          (future) WebGPU RT when available         │
  │                                                                  │
  │  monti_capture         OpenEXR writer (CPU-side)                 │
  │                                                                  │
  │  Tone mapper and presenter live in the host app (app-internal).  │
  │  Internal tools use the product denoiser — we are our own        │
  │  first customer.                                                 │
  └──────────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Layout

```
# ── PRODUCT (Deni) ──────────────────────────────────────────────────

denoise/
├── include/deni/vulkan/
│   └── Denoiser.h                  # Standalone Vulkan denoiser (all public types inline)
└── src/
    └── vulkan/
        ├── Denoiser.cpp            # Passthrough + ML denoiser dispatch
        ├── WeightLoader.h/cpp      # .denimodel binary format parser
        ├── MlInference.h/cpp       # GPU buffer/image management for ML inference
        └── shaders/                # GLSL compute → SPIR-V (loaded from .spv files at runtime)
            └── passthrough_denoise.comp

# ── INTERNAL TOOLING (Monti) ───────────────────────────────────────

scene/
├── include/monti/scene/
│   ├── Scene.h
│   ├── Material.h
│   ├── Light.h
│   ├── Camera.h
│   └── Types.h
└── src/
    ├── Scene.cpp
    └── gltf/
        └── GltfLoader.cpp

renderer/
├── include/monti/vulkan/
│   ├── Renderer.h
│   └── GpuBufferUtils.h           # Optional GPU buffer upload helpers
└── src/
    └── vulkan/
        ├── Renderer.cpp
        ├── Buffer.cpp              # VMA buffer wrappers
        ├── Image.cpp               # VMA image wrappers
        ├── Upload.cpp              # Staging buffer upload
        ├── GpuScene.cpp            # Material/texture/light packing, bindless descriptors
        ├── GpuBufferUtils.cpp      # GPU buffer upload helper implementations
        ├── GeometryManager.cpp     # BLAS/TLAS construction
        ├── EnvironmentMap.cpp      # CDF computation, GPU upload
        ├── BlueNoise.cpp           # Blue noise texture generation/upload
        ├── RaytracePipeline.cpp    # RT pipeline and SBT management
        ├── EmissiveLightExtractor.cpp  # Emissive mesh → TriangleLight extraction (Phase 8J)
        └── shaders/                # raygen, closesthit, miss, anyhit → SPIR-V + source

shaders/
├── raygen.rgen                     # Ray generation shader
├── closesthit.rchit                # Closest-hit shader
├── miss.rmiss                      # Miss shader
├── anyhit.rahit                    # Any-hit shader (alpha masking)
└── include/                        # Shared GLSL includes
    ├── bluenoise.glsl
    ├── brdf.glsl                   # Cook-Torrance BRDF + diffuse transmission
    ├── clearcoat.glsl
    ├── common.glsl
    ├── constants.glsl
    ├── frame_uniforms.glsl
    ├── interior_list.glsl          # Nested dielectric priority tracking
    ├── light_sampling.glsl          # WRS-based light selection (includes wrs.glsl)
    ├── lights.glsl                 # Quad/sphere/triangle light sampling + contribution estimate
    ├── mis.glsl                    # Multiple importance sampling
    ├── payload.glsl                # HitPayload struct
    ├── sampling.glsl
    ├── sheen.glsl                  # Sheen BRDF with precomputed albedo LUT
    ├── uv_transform.glsl           # Per-material UV transform
    ├── wrs.glsl                    # Weighted reservoir sampling data structure
    └── vertex.glsl

capture/
├── include/monti/capture/
│   └── Writer.h
└── src/
    └── Writer.cpp

# ── HOST APPLICATION ───────────────────────────────────────────────

app/
├── core/                           # Shared app infrastructure
│   ├── vulkan_context.cpp          # Device, swapchain, frame loop
│   ├── ToneMapper.cpp              # ACES filmic + sRGB compute shader
│   ├── GBufferImages.cpp           # G-buffer allocation
│   ├── CameraSetup.h               # Camera configuration helpers
│   ├── EnvironmentLoader.cpp       # HDR environment map loading
│   └── frame_resources.cpp         # Per-frame Vulkan resource management
├── view/                           # Interactive viewer (monti_view)
│   ├── main.cpp                    # Viewer entry point
│   ├── CameraController.cpp        # Mouse/keyboard camera control
│   ├── Panels.cpp                  # ImGui debug panels
│   └── UiRenderer.cpp              # ImGui Vulkan integration
├── datagen/                        # Training data generator (monti_datagen)
├── assets/                         # App-level assets
└── shaders/                        # App-level shaders (tonemap, UI)

# ── TRAINING PIPELINE ─────────────────────────────────────────────

training/
├── deni_train/                     # Python training package
│   └── models/                     # U-Net model definitions (unet.py, blocks.py)
├── scripts/                        # Export scripts (export_weights.py → .denimodel)
├── configs/                        # Training configuration files
├── scenes/                         # Training scene definitions
├── viewpoints/                     # Camera viewpoint definitions
├── light_rigs/                     # Light setup definitions
└── tests/                          # Training pipeline tests

# ── TESTS ──────────────────────────────────────────────────────────

tests/
├── main_test.cpp                   # Test runner entry point
├── ml_weight_loader_test.cpp       # .denimodel round-trip tests
├── scenes/
│   ├── CornellBox.h                # Programmatic Cornell box builder
│   └── CornellBox.cpp
├── assets/                         # Downloaded Khronos glTF samples (gitignored)
└── references/                     # Golden reference images per platform (committed)
    └── vulkan/
```

---

## 4. Deni — Product Denoiser API

The denoiser is a standalone library with minimal dependencies: Vulkan and VMA. It has no dependencies on Monti, the scene layer, GLM, or any internal tooling. A customer includes one header, links one library, and passes their existing VMA allocator.

> **GLM in Deni headers:** The passthrough denoiser's public header has no GLM dependency — all types are Vulkan-native or scalar. If the NRD ReLAX denoiser (§4.5) is added in the future and requires view/projection matrix inputs, those fields will use raw `float[16]` arrays rather than `glm::mat4`, keeping the public API free of GLM. This also prepares for the future pure-C API (§4.6) where GLM is unavailable.

### 4.1 Vulkan Denoiser Interface

The initial implementation is a **passthrough denoiser** that sums noisy diffuse and specular contributions — identical to the rtx-chessboard `PassthroughDenoiser`. This provides the correct pipeline plumbing and image layout contracts. The ML denoiser (F11) is the planned product upgrade; NRD ReLAX (F16) is a deferred cross-vendor fallback.

```cpp
// denoise/include/deni/vulkan/Denoiser.h
#pragma once
#include <vulkan/vulkan.h>
#include <memory>

namespace deni::vulkan {

// ── Image Layout Contract ──────────────────────────────────────────────────
//
// INPUTS: All input images must be in VK_IMAGE_LAYOUT_GENERAL before calling
//         Denoise(). The denoiser reads them via storage image descriptors.
//         After Denoise() returns, input images remain in VK_IMAGE_LAYOUT_GENERAL.
//
// OUTPUT: The denoiser's internal output image is transitioned to
//         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL by the time Denoise() returns,
//         ready for the caller's tone mapper or blit. This matches the
//         rtx-chessboard convention.
//
// The denoiser records all necessary pipeline barriers into the provided
// command buffer. The caller is responsible only for ensuring inputs are
// in the correct layout before calling Denoise().

// ── Input ──────────────────────────────────────────────────────────────────
//
// Input Specification:
//   noisy_diffuse   — Diffuse radiance (1–N spp). Linear HDR, unbounded.
//   noisy_specular  — Specular radiance (1–N spp). Linear HDR, unbounded.
//   motion_vectors  — Screen-space motion: current pixel − previous pixel, in
//                     pixel coordinates. Positive X = rightward, positive Y =
//                     downward (Vulkan screen convention). Zero = static.
//   linear_depth    — .r: View-space linear Z distance from the camera near plane.
//                     Positive, increasing with distance. Range: [near, far].
//                     .g: Primary ray hit distance (Phase 8E). Used by denoisers
//                     for adaptive spatial filtering. 0 for miss pixels.
//   world_normals   — .xyz: unit-length surface normal in world space
//                     (right-handed, Y-up per glTF convention).
//                     .w: perceptual roughness [0, 1].
//   diffuse_albedo  — Diffuse reflectance (base_color * (1 − metallic)).
//                     Linear, [0, 1] range. Used for demodulated denoising.
//   specular_albedo — Specular F0 reflectance (Fresnel at normal incidence).
//                     Linear, [0, 1] range. Used for demodulated denoising.
//
// All radiance values are pre-exposure (scene-referred linear HDR).
// The denoiser does not apply exposure or tone mapping.

enum class ScaleMode {
    kNative,       // 1.0× — denoise only, no upscaling
    kQuality,      // 1.5× — e.g. render at 720p for 1080p output (requires ML denoiser)
    kPerformance,  // 2.0× — e.g. render at 540p for 1080p output (requires ML denoiser)
};

// Per-tier field usage:
//   Passthrough: reads noisy_diffuse, noisy_specular only. Other fields are ignored.
//   ML denoiser (future F11): reads all fields; uses scale_mode for super-resolution.
//   NRD ReLAX (future F16, deferred): reads all fields.
//
// Initial release: only ScaleMode::kNative is supported. Create() will
// succeed with any ScaleMode, but Denoise() returns an error if
// scale_mode != kNative until the ML denoiser is implemented.

struct DenoiserInput {
    VkImageView noisy_diffuse;    // RGBA16F — diffuse radiance (1-N spp)
    VkImageView noisy_specular;   // RGBA16F — specular radiance (1-N spp)
    VkImageView motion_vectors;   // RG16F   — screen-space motion (pixels); RGBA16F also supported
    VkImageView linear_depth;     // RG16F   — .r = view-space linear Z, .g = hit distance; RGBA16F also supported
    VkImageView world_normals;    // RGBA16F — world normals (.xyz), roughness (.w)
    VkImageView diffuse_albedo;   // R11G11B10F — diffuse reflectance; RGBA16F also supported
    VkImageView specular_albedo;  // R11G11B10F — specular F0; RGBA16F also supported
                                  // Passthrough ignores both albedo fields.
                                  // ML denoiser uses both for demodulated denoising.
                                  // NRD ReLAX (deferred) uses both for demodulated denoising.

    uint32_t  render_width;
    uint32_t  render_height;
    ScaleMode scale_mode = ScaleMode::kNative;  // See §4.10

    bool reset_accumulation;      // True on camera cut or scene reset
};

// ── Output ─────────────────────────────────────────────────────────────────

struct DenoiserOutput {
    VkImage     denoised_image;   // RGBA16F — denoised output (GENERAL layout)
    VkImageView denoised_color;   // RGBA16F — denoised radiance
};

// ── Configuration ──────────────────────────────────────────────────────────

struct DenoiserDesc {
    VkDevice         device;
    VkPhysicalDevice physical_device;
    uint32_t         width  = 1920;
    uint32_t         height = 1080;

    // width/height set the maximum output resolution (the 1× target).
    // The denoiser pre-allocates its internal output image at this size.
    // The caller may pass smaller render_width/render_height per frame,
    // but the derived output dimensions (render_dim × scale_factor,
    // rounded to nearest even) must not exceed width/height.
    // Call Resize() to change the maximum if needed.

    // Optional. Host-managed pipeline cache for faster pipeline creation
    // (critical on mobile where shader compilation is expensive).
    // If VK_NULL_HANDLE, the denoiser creates pipelines without a cache.
    // The host is responsible for serializing/deserializing the cache
    // via vkGetPipelineCacheData / vkCreatePipelineCache with initial data.
    VkPipelineCache  pipeline_cache = VK_NULL_HANDLE;

    // Required. Host-managed VMA allocator. The denoiser uses this for all
    // internal GPU memory allocations. Sharing the host's allocator avoids
    // multiple allocators competing for the same heap.
    VmaAllocator     allocator;

    // Required. Directory containing compiled SPIR-V shader files (.spv).
    // The denoiser loads shaders from this directory at Create() time.
    std::string_view shader_dir;

    // Required. The denoiser resolves all Vulkan device functions internally
    // through this pointer. This makes Deni completely loader-agnostic — the
    // host can use volk, the Vulkan SDK loader, or any custom dispatch
    // mechanism. Deni has no build-time or link-time dependency on any Vulkan
    // loader; it only depends on Vulkan headers.
    PFN_vkGetDeviceProcAddr get_device_proc_addr;

    // Optional. Path to a .denimodel file containing trained ML denoiser
    // weights. If empty, the denoiser uses the passthrough mode
    // (diffuse + specular sum). Weight loading is deferred to the first
    // Denoise() call.
    std::string model_path;
};

// ── Denoiser ───────────────────────────────────────────────────────────────

class Denoiser {
public:
    static std::unique_ptr<Denoiser> Create(
        const DenoiserDesc& desc);
    ~Denoiser();

    // Record denoise commands into cmd.
    // Call within an active command buffer recording.
    DenoiserOutput Denoise(VkCommandBuffer cmd,
                           const DenoiserInput& input);

    // Resize internal buffers. Call on window resize.
    void Resize(uint32_t width, uint32_t height);

    float LastPassTimeMs() const;

    // Returns true if a .denimodel was provided and weights are loaded (or pending).
    bool HasMlModel() const;
};

} // namespace deni::vulkan
```

### 4.2 Customer Integration Example

```cpp
#include <deni/vulkan/Denoiser.h>

// Customer creates denoiser using their existing Vulkan device and VMA allocator.
// The host provides vkGetDeviceProcAddr from whatever Vulkan loader they use
// (volk, Vulkan SDK, custom loader, etc.). Deni resolves all device functions
// internally — no loader dependency leaks into the library.
auto denoiser = deni::vulkan::Denoiser::Create({
    .device               = my_device,
    .physical_device      = my_physical_device,
    .width                = 1920,
    .height               = 1080,
    .allocator            = my_vma_allocator,
    .shader_dir           = "shaders/deni/",
    .get_device_proc_addr = vkGetDeviceProcAddr,
    .model_path           = "models/deni_unet.denimodel",  // Optional — omit for passthrough
});

// In their existing frame loop, using their existing command buffer:
vkBeginCommandBuffer(cmd, &begin_info);

// ... their path tracer records commands ...

// Slot our denoiser in:
auto denoised = denoiser->Denoise(cmd, {
    .noisy_diffuse   = my_noisy_diffuse_view,
    .noisy_specular  = my_noisy_specular_view,
    .motion_vectors  = my_motion_view,
    .linear_depth    = my_depth_view,
    .world_normals   = my_normals_view,
    .diffuse_albedo  = my_diffuse_albedo_view,
    .specular_albedo = my_specular_albedo_view,
    .render_width    = 1920,
    .render_height   = 1080,
    .scale_mode      = deni::vulkan::ScaleMode::kNative,
    .reset_accumulation = false,
});

// Use denoised.denoised_color in their tone mapper
// ...

vkEndCommandBuffer(cmd);
```

### 4.3 Initial Implementation: Passthrough

The initial `Denoiser` is a passthrough that sums diffuse + specular in a compute shader (16×16 workgroup), matching rtx-chessboard's `PassthroughDenoiser`. This validates the full pipeline: image layout contracts, descriptor binding, command buffer recording, and host integration. When a `model_path` is provided in `DenoiserDesc`, the ML inference pipeline (Phase F11) is activated instead.

### 4.4 Loader-Agnostic Vulkan Dispatch

Deni (and Monti's renderer) have **no build-time or link-time dependency on any Vulkan loader**. They depend only on `Vulkan::Headers` (type definitions) and `VulkanMemoryAllocator` (VMA). All Vulkan device functions are resolved at runtime via the `PFN_vkGetDeviceProcAddr` provided by the host in `DenoiserDesc` / `RendererDesc`.

**Why this matters:** Deni is a product library intended for integration into any Vulkan engine. Requiring a specific loader (volk, the Vulkan SDK loader, etc.) would impose a transitive dependency on the host application and potentially conflict with the host's own Vulkan loading strategy. By accepting `PFN_vkGetDeviceProcAddr` as a required field, Deni works with any loader — or no loader at all (e.g., directly from `vkGetDeviceProcAddr` obtained via `dlsym`/`GetProcAddress`).

**Implementation:** `Create()` uses the provided `PFN_vkGetDeviceProcAddr` to resolve all needed device-level functions (descriptor set operations, pipeline creation, command buffer recording, image/view creation, etc.) and stores them in a private dispatch table. All internal Vulkan calls go through this table. VMA has its own internal dispatch and does not need additional function pointers.

**Same pattern for Monti:** `monti_vulkan` follows the same approach — `RendererDesc::get_device_proc_addr` is required and used identically. This keeps both libraries usable by any Vulkan host, with volk (or any other loader) confined to the application layer.

### 4.5 Future: NRD ReLAX Spatial-Temporal Filter (Desktop Only)

> Deferred (F16) — see [roadmap.md](roadmap.md#f5-future-platform-denoisers) for context. NRD ReLAX is a 7-pass classical denoiser that may be added as a cross-vendor fallback on desktop if the ML denoiser (F11) is not yet ready when AMD/Intel support is needed. Not planned for mobile (bandwidth constraints). ReBLUR is not planned.

### 4.6 Platform Interface Parity

> **Design intent:** All platform denoiser implementations (`deni::vulkan::Denoiser`, `deni::metal::Denoiser`, `deni::webgpu::Denoiser`) follow an **identical interface shape**. The struct and method names, parameter semantics, image layout contracts, and lifecycle are the same across platforms — only the native GPU types differ (`VkImageView` vs `MTLTexture*` vs `WGPUTextureView`).
>
> There is no abstract base class because the type signatures necessarily differ per platform. Instead, parity is enforced by convention and documentation. Each platform's `Denoiser` class has the same methods: `Create()`, `Denoise()`, `Resize()`, `LastPassTimeMs()`. Each platform's `DenoiserInput` struct has the same semantic fields with platform-native types.
>
> The Vulkan implementation uses a C++ interface. Future Metal and WebGPU implementations will expose a **pure-C API** (`deni_denoiser_create()`, `deni_denoiser_denoise()`, etc.) for interop with Swift and TypeScript/JavaScript (via emscripten WASM). The C API is a thin wrapper over the same internal implementation.

### 4.7 Shader Distribution

Compiled SPIR-V shaders are built at CMake time via `glslc` and output to a build directory (`${CMAKE_CURRENT_BINARY_DIR}/deni_shaders/` for Deni, `${CMAKE_CURRENT_BINARY_DIR}/monti_shaders/` for Monti). At runtime, shaders are **loaded from `.spv` files** via the `shader_dir` path provided in `DenoiserDesc` / `RendererDesc`. The GLSL source files are also shipped alongside the library for inspection. This approach keeps shader iteration fast (recompile SPIR-V without relinking the library) and avoids embedding binary blobs in source files.

### 4.8 Thread Safety

A single `Denoiser` instance is not thread-safe. `Create()`, `Denoise()`, and `Resize()` must not be called concurrently on the same instance. Multiple `Denoiser` instances on the same `VkDevice` are safe if they record into different command buffers.

### 4.9 Future Platform Denoisers

> See [roadmap.md](roadmap.md#f5-future-platform-denoisers) for the full table of planned denoiser backends (Metal, WebGPU, NRD ReLAX). DLSS-RR is integrated at the app level in `monti_view` (F1), not in Deni.

### 4.10 Producing DenoiserInput

Customers integrating Deni with their own path tracer must produce the `DenoiserInput` fields according to these conventions:

| Field | Definition | Coordinate system |
|---|---|---|
| `noisy_diffuse` | Diffuse radiance accumulated over N samples. Linear HDR, unbounded. | Scene-referred |
| `noisy_specular` | Specular radiance accumulated over N samples. Linear HDR, unbounded. | Scene-referred |
| `motion_vectors` | `current_pixel_pos − previous_pixel_pos` in pixel coordinates. | Vulkan screen: +X right, +Y down. Zero = static. |
| `linear_depth` | `.r`: View-space Z distance from camera near plane. `.g`: primary ray hit distance (Phase 8E, for denoiser adaptive filtering). | `.r`: Positive, increasing: `dot(hit − eye, camera_forward)`. `.g`: Euclidean hit distance (`payload.hit_t`). |
| `world_normals` | `.xyz`: unit-length surface normal in world space (glTF Y-up, right-handed). `.w`: perceptual roughness ∈ [0, 1]. | World space |
| `diffuse_albedo` | `base_color × (1 − metallic)`. Linear, [0, 1] range. | — |
| `specular_albedo` | Fresnel F0 at normal incidence. Linear, [0, 1] range. For dielectrics: `((ior−1)/(ior+1))²`. For metals: `base_color`. | — |

**Splitting diffuse and specular:** The path tracer classifies each path's contribution at the first opaque bounce. If the first bounce samples the diffuse lobe, the entire path throughput goes to `noisy_diffuse`; if it samples the specular or clear coat lobe, it goes to `noisy_specular`. This split enables demodulated denoising — the denoiser operates on demodulated (albedo-divided) radiance and remodulates after filtering, preserving texture detail.

**Image layouts:** All input images must be in `VK_IMAGE_LAYOUT_GENERAL` before `Denoise()`. The denoiser reads them via storage image descriptors. After `Denoise()`, inputs remain in `VK_IMAGE_LAYOUT_GENERAL`.

All radiance values are pre-exposure (scene-referred linear HDR). The denoiser does not apply exposure or tone mapping.

### 4.11 Super-Resolution via ScaleMode

The denoiser combines denoising and upscaling in a single inference pass, controlled by the `ScaleMode` enum in `DenoiserInput`.

| ScaleMode | Factor | Render → Output (1080p target) | Ray cost vs native |
|---|---|---|---|
| `kNative` | 1.0× | 1080p → 1080p | 1.00× |
| `kQuality` | 1.5× | 720p → 1080p | 0.44× |
| `kPerformance` | 2.0× | 540p → 1080p | 0.25× |

**Output resolution computation:**
```
output_dim = floor(render_dim × scale_factor / 2) × 2
```
Rounding to the nearest even dimension avoids chroma-subsampling alignment issues and simplifies PixelShuffle upsampling layers in the ML model.

**ML training implications:** Discrete scale factors allow training one specialized model per ratio (or a single model conditioned on the ratio). Arbitrary ratios would force the model to learn all possible upsampling kernels simultaneously, reducing quality. Integer-ratio PixelShuffle layers (used for 2×) are well-understood; 1.5× uses a 3×-upsample + 2×-downsample strategy internally.

**Why not arbitrary output dimensions?** Arbitrary `output_width`/`output_height` fields would suggest the denoiser supports any scale ratio. In practice, ML upscaling architectures are trained for specific ratios and produce poor results at untrained ratios. The `ScaleMode` enum makes the supported presets explicit and maps directly to trained model variants.

`DenoiserDesc::width/height` defines the maximum output resolution (the 1× target). The derived output dimensions at any `ScaleMode` must not exceed this maximum.

---

## 5. Scene Layer — Platform-Agnostic Data Model

The scene layer describes *what to render*. It is pure CPU-side data: transforms, material descriptions, texture pixel data, lights, cameras, and dirty tracking. It has no GPU types and no platform dependencies. Geometry (vertex/index buffers) is **not** stored in the scene layer — the host owns GPU buffers directly, and the renderer references them via device addresses.

### 5.1 Types (`scene/Types.h`)

Based on rtx-chessboard's `TypedId<Tag>` pattern with `operator<=>` and explicit `bool` conversion.

```cpp
// scene/include/monti/scene/Types.h
#pragma once
#include <compare>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace monti {

template <typename Tag>
struct TypedId {
    static constexpr uint64_t kInvalid = UINT64_MAX;
    uint64_t value = kInvalid;
    bool operator==(const TypedId&) const = default;
    auto operator<=>(const TypedId&) const = default;
    explicit operator bool() const { return value != kInvalid; }
};

struct MeshTag {};
struct MaterialTag {};
struct TextureTag {};
struct NodeTag {};

using MeshId     = TypedId<MeshTag>;
using MaterialId = TypedId<MaterialTag>;
using TextureId  = TypedId<TextureTag>;
using NodeId     = TypedId<NodeTag>;

} // namespace monti

template <typename Tag>
struct std::hash<monti::TypedId<Tag>> {
    size_t operator()(const monti::TypedId<Tag>& id) const noexcept {
        return std::hash<uint64_t>{}(id.value);
    }
};

namespace monti {

struct Transform {
    glm::vec3 translation = {0, 0, 0};
    glm::quat rotation    = {1, 0, 0, 0};
    glm::vec3 scale       = {1, 1, 1};
    glm::mat4 ToMatrix() const;
};

// Fixed vertex layout for glTF PBR path tracing.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 tangent;    // xyz = direction, w = bitangent sign
    glm::vec2 tex_coord_0;
    glm::vec2 tex_coord_1;
};

// Pixel formats for texture descriptions.
enum class PixelFormat {
    kRGBA16F,
    kRGBA32F,
    kRG16F,
    kRGBA8_UNORM,
    kRG16_SNORM,
    kR32F,
    kR8_UNORM,
    // BC compressed formats (Phase 8N — DDS texture support)
    kBC1_UNORM,
    kBC3_UNORM,
    kBC4_UNORM,
    kBC5_UNORM,
    kBC7_UNORM,
};

// Texture sampler wrap mode (matches glTF 2.0 / Vulkan conventions).
enum class SamplerWrap {
    kRepeat,
    kClampToEdge,
    kMirroredRepeat,
};

// Texture sampler filter mode.
enum class SamplerFilter {
    kLinear,
    kNearest,
};

} // namespace monti
```

### 5.2 Scene (`scene/Scene.h`)

Follows rtx-chessboard's flat-vector scene model with typed IDs. Simpler than a full ECS — sufficient for glTF scene loading and path tracing.

```cpp
// scene/include/monti/scene/Scene.h
#pragma once
#include "Types.h"
#include "Material.h"
#include "Light.h"
#include "Camera.h"
#include <memory>
#include <string_view>
#include <vector>

namespace monti {

// Each SceneNode references exactly one mesh primitive and one material.
// Multi-primitive glTF meshes are split into one SceneNode per primitive
// during loading (see §5.6). There is no concept of multi-material meshes
// at the renderer level.
struct SceneNode {
    NodeId      id;
    MeshId      mesh_id;
    MaterialId  material_id;
    Transform   transform;
    Transform   prev_transform;   // For motion vectors
    bool        visible = true;
    std::string name;
};

class Scene {
public:
    Scene() = default;

    // ── Thread Safety ───────────────────────────────────────────────────────
    // Scene is not thread-safe. The host must not modify the Scene
    // (AddMesh, AddNode, SetNodeTransform, RemoveNode, etc.) while
    // Renderer::RenderFrame() is reading it. Synchronize externally
    // (e.g., update scene on the main thread before recording the
    // command buffer, or use a mutex).

    // ── Entity lifecycle ───────────────────────────────────────────────
    MeshId     AddMesh(Mesh mesh, std::string_view name = "");
    MaterialId AddMaterial(MaterialDesc material, std::string_view name = "");
    TextureId  AddTexture(TextureDesc texture, std::string_view name = "");
    NodeId     AddNode(MeshId mesh, MaterialId material,
                       std::string_view name = "");
    void       RemoveNode(NodeId id);
    bool       RemoveMesh(MeshId id);  // Returns false if nodes still reference it

    // ── Transform ──────────────────────────────────────────────────────
    // Saves current transform to prev_transform (for motion vectors),
    // then sets the new transform. Prefer this over direct member access.
    void SetNodeTransform(NodeId id, const Transform& new_transform);

    // ── Accessors ──────────────────────────────────────────────────────
    const Mesh*         GetMesh(MeshId id) const;
    MaterialDesc*       GetMaterial(MaterialId id);
    const MaterialDesc* GetMaterial(MaterialId id) const;
    SceneNode*          GetNode(NodeId id);
    const SceneNode*    GetNode(NodeId id) const;
    const TextureDesc*  GetTexture(TextureId id) const;

    const std::vector<Mesh>&         Meshes() const;
    const std::vector<MaterialDesc>& Materials() const;
    const std::vector<SceneNode>&    Nodes() const;
    std::vector<SceneNode>&          Nodes();
    const std::vector<TextureDesc>&  Textures() const;

    // ── Lights ─────────────────────────────────────────────────────────
    void SetEnvironmentLight(const EnvironmentLight& light);
    const EnvironmentLight* GetEnvironmentLight() const;

    void AddAreaLight(const AreaLight& light);
    const std::vector<AreaLight>& AreaLights() const;

    void AddSphereLight(const SphereLight& light);
    void AddTriangleLight(const TriangleLight& light);
    const std::vector<SphereLight>& SphereLights() const;
    const std::vector<TriangleLight>& TriangleLights() const;

    // ── Camera ─────────────────────────────────────────────────────────
    void SetActiveCamera(const CameraParams& params);
    const CameraParams& GetActiveCamera() const;

    // ── TLAS Dirty Tracking ────────────────────────────────────────────
    // Monotonically increasing counter, incremented by AddNode(),
    // RemoveNode(), and SetNodeTransform(). The renderer caches the
    // last-seen value and skips TLAS rebuilds when unchanged. O(1) check
    // avoids per-node transform diffing on static frames.
    uint64_t TlasGeneration() const;

private:
    std::vector<Mesh>         meshes_;
    std::vector<MaterialDesc> materials_;
    std::vector<SceneNode>    nodes_;
    std::vector<TextureDesc>  textures_;

    std::optional<EnvironmentLight> environment_light_;
    std::vector<AreaLight> area_lights_;
    std::vector<SphereLight> sphere_lights_;
    std::vector<TriangleLight> triangle_lights_;
    CameraParams active_camera_;

    uint64_t next_mesh_id_     = 0;
    uint64_t next_material_id_ = 0;
    uint64_t next_texture_id_  = 0;
    uint64_t next_node_id_     = 0;
    uint64_t tlas_generation_  = 0;  // Incremented by AddNode/RemoveNode/SetNodeTransform
};

} // namespace monti
```

### 5.3 Material (`scene/Material.h`)

CPU-side PBR material data. For each channel, the texture is optional — if the ID is invalid, the constant factor is used alone. If both are provided, the texture sample is multiplied by the factor per glTF 2.0.

> **Emissive fields:** The `MaterialDesc` includes emissive factor, texture, and strength fields per glTF 2.0. These fields are **parsed and stored** by the glTF loader and **used by the renderer** for emissive light extraction (Phase 8J). Emissive mesh faces are decomposed into `TriangleLight` entries and sampled via explicit next-event estimation (NEE) for direct lighting. Emissive surfaces also render their emissive contribution when hit by path-traced rays.

```cpp
// scene/include/monti/scene/Material.h
#pragma once
#include "Types.h"
#include <glm/glm.hpp>
#include <optional>
#include <string>
#include <vector>

namespace monti {

// ── Mesh (metadata only — geometry lives on GPU) ─────────────────────────
//
// The scene layer stores mesh metadata. Vertex/index data is owned by the
// host as GPU buffers and registered with the renderer via
// Renderer::RegisterMeshBuffers(). This avoids CPU roundtrips and supports GPU-generated
// geometry (cloth simulation, procedural deformation).

struct Mesh {
    MeshId id;
    std::string name;
    uint32_t vertex_count  = 0;
    uint32_t index_count   = 0;
    uint32_t vertex_stride = sizeof(Vertex);  // Defaults to standard Vertex layout
    glm::vec3 bbox_min{0};
    glm::vec3 bbox_max{0};
};

// ── Transient Mesh Data (for loaders) ────────────────────────────────────
//
// Returned by glTF loader. Host uploads to GPU buffers, then discards.
// Not stored in the Scene.

struct MeshData {
    MeshId mesh_id;
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;
};

// ── Texture Registration ───────────────────────────────────────────────────

struct TextureDesc {
    TextureId   id;
    std::string name;
    uint32_t    width = 0;
    uint32_t    height = 0;
    uint32_t    mip_levels = 1;
    PixelFormat format = PixelFormat::kRGBA8_UNORM;
    std::vector<uint8_t> data;

    // Sampler parameters (from glTF sampler, or defaults per glTF 2.0 spec).
    SamplerWrap   wrap_s  = SamplerWrap::kRepeat;
    SamplerWrap   wrap_t  = SamplerWrap::kRepeat;
    SamplerFilter mag_filter = SamplerFilter::kLinear;
    SamplerFilter min_filter = SamplerFilter::kLinear;
};

// ── PBR Material ───────────────────────────────────────────────────────────

struct MaterialDesc {
    MaterialId id;
    std::string name;

    // Base PBR (implemented)
    glm::vec3 base_color       = {1, 1, 1};
    float     roughness        = 0.5f;
    float     metallic         = 0.0f;
    float     opacity          = 1.0f;
    float     ior              = 1.5f;

    std::optional<TextureId> base_color_map;
    std::optional<TextureId> normal_map;
    std::optional<TextureId> metallic_roughness_map;

    float normal_scale         = 1.0f;

    // Clear coat (implemented, matches rtx-chessboard)
    float clear_coat           = 0.0f;
    float clear_coat_roughness = 0.0f;

    // Alpha masking (implemented — any-hit shader for kMask, kBlend via blended transmission)
    enum class AlphaMode { kOpaque, kMask, kBlend };
    AlphaMode alpha_mode       = AlphaMode::kOpaque;
    float     alpha_cutoff     = 0.5f;
    bool      double_sided     = false;

    // Emissive (implemented — emissive extraction + NEE in Phase 8J)
    glm::vec3 emissive_factor    = {0, 0, 0};
    std::optional<TextureId> emissive_map;
    float     emissive_strength  = 1.0f;

    // Transmission/volume (implemented — Fresnel refraction, volume attenuation)
    float     transmission_factor  = 0.0f;
    std::optional<TextureId> transmission_map;
    glm::vec3 attenuation_color    = {1, 1, 1};
    float     attenuation_distance = 0.0f;

    // Diffuse transmission (Phase 8H — KHR_materials_diffuse_transmission)
    float     diffuse_transmission_factor = 0.0f;  // 0 = opaque, 1 = fully diffuse-transmissive
    glm::vec3 diffuse_transmission_color  = {1, 1, 1};
    bool      thin_surface                = false;  // Single-intersection in/out (leaves, fabric)

    // Nested dielectric priority (Phase 8I — overlapping transparent volumes)
    uint8_t   nested_priority = 0;  // 0 = default; higher priority wins when volumes overlap

    // Sheen (Phase 8M — KHR_materials_sheen)
    glm::vec3 sheen_color              = {0, 0, 0};
    float     sheen_roughness          = 0.0f;
    std::optional<TextureId> sheen_color_map;
    std::optional<TextureId> sheen_roughness_map;

    // KHR_texture_transform (Phase 8L — per-material UV transform)
    glm::vec2 uv_offset   = {0, 0};
    glm::vec2 uv_scale    = {1, 1};
    float     uv_rotation = 0.0f;
};

} // namespace monti
```

### 5.4 Lights (`scene/Light.h`)

Two light types are implemented in v1: `EnvironmentLight` (HDR equirectangular map) and `AreaLight` (emissive quad). Point, spot, and directional lights are intentionally omitted — they are mathematical idealizations (zero-area emitters) that don't exist physically. A small area light produces the same visual result with correct soft shadows and penumbrae; a sun disk in the environment map handles directional illumination.

Phase 8G adds `SphereLight` and `TriangleLight` for richer area light support:
- **SphereLight** — Analytic solid-angle sampling with correct MIS weight. Useful for light bulbs, orbs, and approximate emitters.
- **TriangleLight** — Triangle-shaped emitter for emissive mesh extraction (Phase 8J). Emissive mesh faces are decomposed into individual triangle lights for direct NEE sampling.

> **Why quad area lights first?** A quad emitter requires minimal path tracer changes: sample a point on the quad, compute the solid angle PDF, trace a shadow ray, and MIS-weight against the BRDF sample. This is a direct extension of the existing environment MIS logic. By contrast, emissive arbitrary-mesh lights require per-triangle CDF construction, mesh-area-weighted sampling, and ideally ReSTIR to converge with many emitters — significantly more complex. The quad area light enables the Cornell box ceiling light, window rectangles, and basic interior scenes without that complexity. Phase 8K adds hybrid NEE: scenes with ≤ 4 lights use direct per-light shadow rays (preserving current quality), while scenes above the threshold use weighted reservoir sampling (WRS) to select one light per bounce with probability proportional to estimated contribution. This enables efficient NEE with hundreds of emissive triangle lights from Phase 8J extraction.

```cpp
// scene/include/monti/scene/Light.h
#pragma once
#include "Types.h"
#include <glm/glm.hpp>

namespace monti {

struct EnvironmentLight {
    TextureId hdr_lat_long;       // HDR equirectangular map
    float     intensity  = 1.0f;
    float     rotation   = 0.0f;  // Radians around Y axis
};

// Quad area light — a planar rectangle defined by a corner and two edge vectors.
// Emits light from the front face (determined by cross(edge_a, edge_b) normal).
// This is sufficient for ceiling panels, window rectangles, and simple interior lighting.
//
// RETENTION NOTE: The quad AreaLight is intentionally retained alongside TriangleLight.
// Quads have a simpler uniform sampling formula (no barycentric coordinates) and are
// the natural primitive for rectangular emitters (ceiling panels, windows, screens).
// Two triangles can represent a quad, but the quad's solid-angle sampling PDF is more
// efficient for rectangular geometry. The unified PackedLight buffer (Phase 8G) supports
// all three types (kQuad, kSphere, kTriangle) without redundancy.
struct AreaLight {
    glm::vec3 corner   = {0, 0, 0};   // World-space corner position
    glm::vec3 edge_a   = {1, 0, 0};   // First edge from corner
    glm::vec3 edge_b   = {0, 0, 1};   // Second edge from corner
    glm::vec3 radiance = {1, 1, 1};   // Emitted radiance (linear HDR)
    bool      two_sided = false;       // Emit from both faces
};

// Spherical area light — emits uniformly from a sphere surface.
// Uses solid-angle sampling for correct MIS weights. (Phase 8G)
struct SphereLight {
    glm::vec3 center   = {0, 0, 0};   // World-space center
    float     radius   = 0.5f;        // Sphere radius
    glm::vec3 radiance = {1, 1, 1};   // Emitted radiance (linear HDR)
};

// Triangle area light — a single emissive triangle.
// Used for emissive mesh extraction (Phase 8J) where each emissive mesh face
// becomes an individually sampable triangle light for NEE.
struct TriangleLight {
    glm::vec3 v0       = {0, 0, 0};   // Vertex 0
    glm::vec3 v1       = {1, 0, 0};   // Vertex 1
    glm::vec3 v2       = {0, 0, 1};   // Vertex 2
    glm::vec3 radiance = {1, 1, 1};   // Emitted radiance (linear HDR)
    bool      two_sided = false;       // Emit from both faces
};

} // namespace monti
```

### 5.5 Camera (`scene/Camera.h`)

```cpp
// scene/include/monti/scene/Camera.h
#pragma once
#include <glm/glm.hpp>

namespace monti {

struct CameraParams {
    glm::vec3 position = {0, 0, 0};
    glm::vec3 target   = {0, 0, -1};
    glm::vec3 up       = {0, 1, 0};

    float vertical_fov_radians = 1.047197f;   // ~60 degrees
    float near_plane           = 0.1f;
    float far_plane            = 1000.0f;
    float aperture_radius      = 0.0f;        // 0 = pinhole
    float focus_distance       = 10.0f;
    float exposure_ev100       = 0.0f;

    glm::mat4 ViewMatrix() const;
    glm::mat4 ProjectionMatrix(float aspect) const;
};

} // namespace monti
```

### 5.6 glTF Loader

The loader populates the scene with mesh metadata, materials, textures, and nodes. Vertex and index data is returned as transient `MeshData` in the `LoadResult` — the host uploads this to GPU buffers and registers device addresses via `Renderer::RegisterMeshBuffers()`. This separation keeps the scene layer GPU-agnostic while supporting GPU-side-only geometry. Camera extraction is not performed; cameras are always set by the host. Skin, animation, and morph target data are silently ignored (GPU skinning is planned for F14).

```cpp
// scene/src/gltf/GltfLoader.h
#pragma once
#include <monti/scene/Scene.h>
#include <monti/scene/Material.h>  // MeshData
#include <string>
#include <vector>

namespace monti::gltf {

struct LoadResult {
    bool success = false;
    std::string error_message;
    std::vector<NodeId> nodes;
    std::vector<MeshData> mesh_data;  // CPU-side vertex/index data for host upload
};

struct LoadOptions {
    bool  generate_missing_normals  = true;
    bool  generate_missing_tangents = true;
};

// Loads glTF and populates the scene with mesh metadata, materials,
// textures, and nodes. Each glTF primitive becomes a separate Mesh +
// SceneNode. Node hierarchy is flattened — world transforms are computed
// by concatenating parent transforms and stored in SceneNode::transform.
// Vertex/index data is returned in LoadResult::mesh_data for host-driven
// GPU upload.
LoadResult LoadGltf(Scene&             scene,
                    const std::string& file_path,
                    const LoadOptions& options = {});

} // namespace monti::gltf
```

---

## 6. Monti Vulkan Renderer — Internal Tooling

The renderer reads the platform-agnostic scene data and manages all GPU resources. It takes native Vulkan types for the command buffer and outputs to host-provided G-buffer images. Materials and textures are packed from the scene's CPU-side data into GPU buffers internally. Geometry (vertex/index buffers) is host-owned — the renderer references device addresses registered by the host via `Renderer::RegisterMeshBuffers()`. The internal `GpuScene` class (adapted from rtx-chessboard's `GPUScene`) manages material packing, texture upload, and mesh buffer bindings.

### 6.1 GPU Scene (Internal, Vulkan-specific)

`GpuScene` is **internal** to the renderer — it lives in `renderer/src/vulkan/` and is not exposed in public headers. The host registers geometry buffers via `Renderer::RegisterMeshBuffers()` (§6.3), which delegates to the internal `GpuScene`. Material and texture uploads are triggered internally by the renderer on first `RenderFrame()` after `SetScene()`.

Materials are packed into a host-visible storage buffer (direct `memcpy`, no staging — material arrays are small and updated infrequently). Textures are uploaded to device-local `VkImage` objects with per-texture `VkSampler` and bound as a bindless `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` array. Geometry buffers are **not owned** by `GpuScene` — the host provides `VkBuffer` handles and device addresses via `Renderer::RegisterMeshBuffers()`.

Since geometry lives in **separate per-mesh buffers** (not merged), shaders cannot index into a single buffer by offset. Instead, `GpuScene` maintains a **buffer address table** — a storage buffer of `MeshAddressEntry` structs, one per registered mesh — providing per-mesh vertex/index device addresses for GLSL `buffer_reference` access. This avoids duplicating geometry into a merged buffer (significant memory savings for large scenes) at the cost of one extra storage buffer read per closest-hit invocation.

```cpp
// renderer/src/vulkan/GpuScene.h  (INTERNAL — not in public include/)
#pragma once
#include <monti/scene/Types.h>
#include <monti/vulkan/Renderer.h>  // MeshBufferBinding
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <cstdint>
#include <vector>
#include <unordered_map>

namespace monti::vulkan {

// GPU-packed material: eleven vec4s per material for storage buffer upload.
// PACKED STRUCTURE — 176 bytes (11 × vec4, 16-byte aligned)
//
// Evolution: 96 bytes (Phase 8C) → 112 bytes (Phase 8D, +emissive)
//            → 128 bytes (Phase 8H, +diffuse transmission/thin-surface/nested priority)
//            → 176 bytes (Phases 8L/8M, +UV transform/sheen/sheen textures)
//
// All texture indices are float-encoded uint32_t via std::bit_cast<float>().
// UINT32_MAX = no texture. Shader checks: floatBitsToUint(idx) == 0xFFFFFFFFu.
struct alignas(16) PackedMaterial {
    glm::vec4 base_color_roughness;   // .rgb = base_color, .a = roughness
    glm::vec4 metallic_clearcoat;     // .r = metallic, .g = clear_coat,
                                      // .b = clear_coat_roughness,
                                      // .a = base_color_map index
    glm::vec4 opacity_ior;            // .r = opacity, .g = ior,
                                      // .b = normal_map index,
                                      // .a = metallic_roughness_map index
    glm::vec4 transmission_volume;    // .r = transmission_factor, .g = reserved (0),
                                      // .b = attenuation_distance,
                                      // .a = transmission_map index
    glm::vec4 attenuation_color_pad;  // .rgb = attenuation_color,
                                      // .a = emissive_map index
    glm::vec4 alpha_mode_misc;        // .r = alpha_mode (float-encoded uint: 0/1/2),
                                      // .g = alpha_cutoff,
                                      // .b = normal_scale,
                                      // .a = uv_rotation
    glm::vec4 emissive;               // .rgb = emissive_factor, .a = emissive_strength
    glm::vec4 transmission_ext;       // .r = diffuse_transmission_factor,
                                      // .g = thin_surface (1.0/0.0),
                                      // .b = packHalf2x16(dt_color.rg) as float,
                                      // .a = packHalf2x16(vec2(dt_color.b, nested_priority))
    glm::vec4 uv_transform;           // .rg = uv_offset, .ba = uv_scale
    glm::vec4 sheen;                  // .rgb = sheen_color, .a = sheen_roughness
    glm::vec4 sheen_textures;         // .r = sheen_color_map index,
                                      // .g = sheen_roughness_map index,
                                      // .ba = reserved
};

static_assert(sizeof(PackedMaterial) == 176);
// kMaterialStride = 11  (number of vec4s per material in the storage buffer)

// GPU-packed light: four vec4s per light for storage buffer upload.
// PACKED STRUCTURE — 64 bytes (4 × vec4, 16-byte aligned)
//
// Unified light buffer supporting multiple light types via a type discriminator.
// Phase 8G replaces the previous PackedAreaLight with this polymorphic layout.
//
// LightType enum: kQuad = 0, kSphere = 1, kTriangle = 2
struct alignas(16) PackedLight {
    glm::vec4 position_type;     // kQuad: .xyz = corner, .w = type (0.0)
                                 // kSphere: .xyz = center, .w = type (1.0)
                                 // kTriangle: .xyz = v0, .w = type (2.0)
    glm::vec4 param_a;           // kQuad: .xyz = edge_a, .w = two_sided (1.0/0.0)
                                 // kSphere: .r = radius, .gba = unused
                                 // kTriangle: .xyz = v1, .w = two_sided (1.0/0.0)
    glm::vec4 param_b;           // kQuad: .xyz = edge_b, .w = unused
                                 // kSphere: unused
                                 // kTriangle: .xyz = v2, .w = unused
    glm::vec4 radiance;          // .xyz = emitted radiance, .w = unused (all types)
};

static_assert(sizeof(PackedLight) == 64);

class GpuScene {
public:
    GpuScene(VmaAllocator allocator, VkDevice device);
    ~GpuScene();

    GpuScene(const GpuScene&) = delete;
    GpuScene& operator=(const GpuScene&) = delete;

    // Register host-owned GPU buffers for a mesh. Called after host uploads
    // vertex/index data. Device addresses are used for BLAS/TLAS building
    // and shader-side buffer_reference access. Also adds an entry to the
    // buffer address table (mesh_address_entries_).
    void RegisterMeshBuffers(MeshId mesh, const MeshBufferBinding& binding);

    // Pack CPU-side materials from Scene into host-visible GPU storage buffer.
    // Allocates buffer on first call; reallocates if material count grows.
    bool UpdateMaterials(const class monti::Scene& scene);

    // Upload all textures from Scene to device-local VkImages with staging.
    // Creates VkImageView + VkSampler per texture. Generates mip chain when
    // TextureDesc::mip_levels > 1 via vkCmdBlitImage. Records commands into cmd.
    bool UploadTextures(const class monti::Scene& scene,
                        VkCommandBuffer cmd);

    // Upload/re-upload the mesh address table storage buffer.
    // Host-visible buffer — writes via memcpy, no command buffer needed.
    // Called by RenderFrame() when new meshes have been registered.
    void UploadMeshAddressTable();

    // Pack and upload lights from Scene to host-visible storage buffer.
    // Packs all light types (quad, sphere, triangle) into the unified PackedLight format.
    // Called by RenderFrame() when the scene changes. Returns the total light count.
    uint32_t UpdateLights(const class monti::Scene& scene);

    // Accessors for BLAS/TLAS building and descriptor binding
    const MeshBufferBinding* GetMeshBinding(MeshId id) const;
    uint32_t GetMeshAddressIndex(MeshId id) const;
    VkBuffer MeshAddressBuffer() const;
    VkDeviceSize MeshAddressBufferSize() const;
    VkBuffer MaterialBuffer() const;
    uint32_t GetMaterialIndex(MaterialId id) const;
    VkBuffer LightBuffer() const;
    VkDeviceSize LightBufferSize() const;
    uint32_t LightCount() const;
    uint32_t TextureCount() const;
    const auto& TextureImages() const { return texture_images_; }

private:
    VmaAllocator allocator_;
    VkDevice     device_;

    std::unordered_map<MeshId, MeshBufferBinding> mesh_bindings_;

    // Buffer address table: one entry per registered mesh.
    // Shader uses mesh_address_index (from instance custom index lower 12 bits)
    // to look up per-mesh vertex/index device addresses for buffer_reference.
    struct alignas(16) MeshAddressEntry {
        VkDeviceAddress vertex_address;  // 8 bytes
        VkDeviceAddress index_address;   // 8 bytes
        uint32_t        vertex_count;    // 4 bytes
        uint32_t        index_count;     // 4 bytes
        // Total: 24 bytes, padded to 32 bytes by alignas(16) + std430
    };
    std::vector<MeshAddressEntry> mesh_address_entries_;
    std::unordered_map<MeshId, uint32_t> mesh_id_to_address_index_;
    // Host-visible storage buffer for the address table

    // Material storage buffer (host-visible, VMA-allocated)
    // Bindless texture images + per-texture samplers

    // Light storage buffer (host-visible, VMA-allocated)
    // Unified PackedLight buffer for all light types (quad, sphere, triangle).
    // Packed from Scene::AreaLights(), Scene::SphereLights(), Scene::TriangleLights()
    // on each UpdateLights() call. Empty scenes bind a 1-element placeholder buffer.
    uint32_t light_count_ = 0;
};

} // namespace monti::vulkan
```

### 6.1.1 GPU Buffer Upload Helpers (Optional)

Convenience functions for hosts that do not already manage GPU geometry buffers. These upload `MeshData` (returned by the glTF loader) to device-local VMA buffers with staging copies and return a `GpuBuffer` ready for `Renderer::RegisterMeshBuffers()`. Platform-specific — each backend (Vulkan, Metal, WebGPU) provides its own equivalent.

Hosts that already maintain GPU buffers (game engines, procedural generators) skip these entirely and call `Renderer::RegisterMeshBuffers()` directly with their own buffer handles.

```cpp
// renderer/include/monti/vulkan/GpuBufferUtils.h
#pragma once
#include <monti/scene/Material.h>  // MeshData, Vertex
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <span>
#include <utility>

namespace monti::vulkan {

/// GPU buffer with device address, suitable for BLAS building
/// and shader-side buffer_reference access.
struct GpuBuffer {
    VkBuffer        buffer         = VK_NULL_HANDLE;
    VmaAllocation   allocation     = VK_NULL_HANDLE;
    VkDeviceAddress device_address = 0;
    VkDeviceSize    size           = 0;
};

/// Upload vertex/index data from a MeshData to device-local GPU buffers.
/// Allocates a staging buffer, copies data, records vkCmdCopyBuffer into
/// cmd, and returns {vertex_buffer, index_buffer}. The staging buffer is
/// freed internally after the copy commands are recorded (the copy is
/// completed when cmd is submitted and the fence signals).
///
/// The returned GpuBuffer pair is owned by the caller and must be kept
/// alive for the renderer's lifetime (until DestroyGpuBuffer() is called).
/// The device-local buffers are usable once the command buffer completes
/// (fence signal).
///
/// Buffers are created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
/// VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
/// and VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, making them immediately usable
/// for BLAS building, shader buffer_reference access, and descriptor binding.
std::pair<GpuBuffer, GpuBuffer> UploadMeshToGpu(
    VmaAllocator allocator, VkDevice device, VkCommandBuffer cmd,
    const MeshData& mesh_data);

/// Create a device-local vertex buffer from raw vertex data.
GpuBuffer CreateVertexBuffer(
    VmaAllocator allocator, VkDevice device, VkCommandBuffer cmd,
    std::span<const monti::Vertex> vertices);

/// Create a device-local index buffer from raw index data.
GpuBuffer CreateIndexBuffer(
    VmaAllocator allocator, VkDevice device, VkCommandBuffer cmd,
    std::span<const uint32_t> indices);

/// Destroy a GpuBuffer, freeing the VMA allocation.
void DestroyGpuBuffer(VmaAllocator allocator, GpuBuffer& buffer);

/// Convenience: upload all meshes from a loader result, register their
/// buffer bindings with the Renderer, and return the GpuBuffers the host
/// must keep alive for the renderer's lifetime. Equivalent to calling
/// UploadMeshToGpu + Renderer::RegisterMeshBuffers per mesh, but less
/// boilerplate. Records vkCmdCopyBuffer commands into cmd.
std::vector<GpuBuffer> UploadAndRegisterMeshes(
    Renderer& renderer, VmaAllocator allocator, VkDevice device,
    VkCommandBuffer cmd, std::span<const MeshData> mesh_data);

} // namespace monti::vulkan
```

### 6.2 Geometry Manager (Internal)

The `GeometryManager` is **internal** to the renderer — not exposed in public headers. `RenderFrame()` calls it automatically: it builds BLAS for newly registered meshes, refits/rebuilds BLAS for deformed meshes (flagged via `Renderer::NotifyMeshDeformed()`), rebuilds the TLAS when transforms change, and destroys BLAS for meshes removed from the scene.

```cpp
// renderer/src/vulkan/GeometryManager.h  (INTERNAL — not in public include/)
#pragma once
#include <monti/scene/Types.h>
#include <vulkan/vulkan.h>
#include <cstdint>

namespace monti::vulkan {

// Manages BLAS/TLAS construction from host-provided GPU buffers.
// Follows rtx-chessboard's pattern: one BLAS per unique mesh,
// single TLAS for all visible scene nodes, with compaction.
//
// Called internally by RenderFrame() — not exposed to host.
class GeometryManager {
public:
    // Build BLAS for any newly registered meshes.
    // Refit BLAS for meshes flagged as deformed (topology unchanged).
    // Rebuild BLAS for meshes flagged as topology-changed.
    bool BuildDirtyBlas(VkCommandBuffer cmd, const class GpuScene& gpu_scene);

    // Build/rebuild TLAS from scene node transforms.
    bool BuildTlas(VkCommandBuffer cmd, const class monti::Scene& scene,
                   const class GpuScene& gpu_scene);

    // Destroy BLAS for meshes no longer present in the scene.
    void CleanupRemovedMeshes(const class monti::Scene& scene);

    VkAccelerationStructureKHR Tlas() const;

    // Mark a mesh's BLAS as needing refit or rebuild.
    void NotifyMeshDeformed(MeshId mesh, bool topology_changed = false);

private:
    // BLAS entries, scratch buffers, TLAS buffer
};

} // namespace monti::vulkan
```

### 6.3 Renderer Interface

```cpp
// renderer/include/monti/vulkan/Renderer.h
#pragma once
#include <monti/scene/Scene.h>
#include <vulkan/vulkan.h>
#include <memory>

namespace monti::vulkan {

// ── G-Buffer (Vulkan native types) ─────────────────────────────────────────
// Matches rtx-chessboard's output image set for denoiser compatibility.

struct GBuffer {
    VkImageView noisy_diffuse;    // RGBA16F     — diffuse radiance (recommended)
    VkImageView noisy_specular;   // RGBA16F     — specular radiance (recommended)
    VkImageView motion_vectors;   // RG16F       — screen-space motion; RGBA16F also supported
    VkImageView linear_depth;     // RG16F       — .r = view-space linear Z, .g = hit distance; RGBA16F also supported
    VkImageView world_normals;    // RGBA16F     — world normals (.xyz), roughness (.w)
    VkImageView diffuse_albedo;   // R11G11B10F  — diffuse reflectance; RGBA16F also supported
    VkImageView specular_albedo;  // R11G11B10F  — specular F0; RGBA16F also supported
};

// ── Renderer ───────────────────────────────────────────────────────────────

struct RendererDesc {
    VkDevice         device;
    VkPhysicalDevice physical_device;
    VkQueue          queue;
    uint32_t         queue_family_index;
    VkPipelineCache  pipeline_cache = VK_NULL_HANDLE;  // Optional, host-managed
    VmaAllocator     allocator;                         // Required, host-managed
    uint32_t         width             = 1920;
    uint32_t         height            = 1080;
    uint32_t         samples_per_pixel = 4;

    // Required. Same semantics as DenoiserDesc::get_device_proc_addr.
    // The renderer resolves all Vulkan device functions through this pointer.
    PFN_vkGetDeviceProcAddr get_device_proc_addr;
};

// Host-provided GPU buffer handles and device addresses for a mesh.
// Passed to Renderer::RegisterMeshBuffers() after the host uploads
// vertex/index data to GPU buffers.
struct MeshBufferBinding {
    VkBuffer         vertex_buffer;
    VkDeviceAddress  vertex_address;
    VkBuffer         index_buffer;
    VkDeviceAddress  index_address;
    uint32_t         vertex_count;
    uint32_t         index_count;
    uint32_t         vertex_stride = sizeof(monti::Vertex);
};

class Renderer {
public:
    static std::unique_ptr<Renderer> Create(
        const RendererDesc& desc);
    ~Renderer();

    // ── Scene ─────────────────────────────────────────────────────────
    void SetScene(monti::Scene* scene);

    // ── Geometry Registration ─────────────────────────────────────────
    // Register host-owned GPU buffers for a mesh. Called after the host
    // uploads vertex/index data to GPU buffers. Delegates to the internal
    // GpuScene. Device addresses are used for BLAS building and shader
    // buffer_reference access.
    void RegisterMeshBuffers(MeshId mesh, const MeshBufferBinding& binding);

    // ── Geometry Deformation ──────────────────────────────────────────
    // Signal that the host modified a mesh's GPU vertex buffer contents.
    // Call this AFTER writing to the buffer and BEFORE RenderFrame().
    // If topology_changed is false, BLAS is refit (fast — updates bounds,
    // preserves tree structure). If true, BLAS is fully rebuilt.
    //
    // The host is responsible for pipeline barriers between their
    // compute/transfer writes and this notification. See §10.4 for
    // a complete barrier example.
    void NotifyMeshDeformed(MeshId mesh, bool topology_changed = false);

    // ── Mesh Cleanup Callback ──────────────────────────────────────────
    // Register a callback invoked during RenderFrame() when the renderer
    // destroys a BLAS for a mesh that was removed from the scene.
    // The callback receives the MeshId of the cleaned-up mesh.
    // The host can use this to schedule deferred GPU buffer cleanup
    // (e.g., destroy the buffer after in-flight frames complete).
    // Set to nullptr to disable. Only one callback is active at a time.
    using MeshCleanupCallback = std::function<void(MeshId)>;
    void SetMeshCleanupCallback(MeshCleanupCallback callback);

    // ── Render ────────────────────────────────────────────────────────
    // Thread safety: a single Renderer instance is not thread-safe.
    // Do not call RenderFrame(), SetScene(), NotifyMeshDeformed(),
    // Resize(), or RegisterMeshBuffers() concurrently on the same instance.
    // Do not modify the Scene while RenderFrame() is executing.
    // Multiple Renderer instances on the same VkDevice are safe if
    // they record into different command buffers.
    //
    // Records all GPU commands into cmd: BLAS build/refit for dirty
    // meshes, TLAS rebuild, path trace. Outputs to the GBuffer images.
    //
    // Accumulation: RenderFrame() supports arbitrarily high SPP values.
    // If the requested SPP exceeds the per-dispatch limit (tuned to
    // avoid GPU timeout / TDR), the renderer internally splits into
    // multiple trace dispatches and accumulates the results. The host
    // always calls RenderFrame() once — the accumulation is transparent.
    //
    // For high-spp reference capture, call with a separate GBuffer
    // allocated at higher precision (e.g., RGBA32F) and a higher SPP —
    // the renderer is format-agnostic and writes the same channels
    // regardless of image format or sample count.
    bool RenderFrame(VkCommandBuffer cmd, const GBuffer& output,
                     uint32_t frame_index);

    // ── Configuration ─────────────────────────────────────────────────
    void SetSamplesPerPixel(uint32_t spp);
    uint32_t GetSamplesPerPixel() const;
    void SetMaxBounces(uint32_t bounces);
    void SetDebugMode(uint32_t mode);
    void SetBackgroundMode(bool show_environment, float blur_level = 0.0f);

    // ── Resize ────────────────────────────────────────────────────────
    void Resize(uint32_t width, uint32_t height);

    float LastFrameTimeMs() const;
};

} // namespace monti::vulkan
```

### 6.4 Platform Renderer Parity

> **Design intent:** Like the denoiser (§4.6), all platform renderer implementations (`monti::vulkan::Renderer`, `monti::metal::Renderer`, `monti::webgpu::Renderer`) follow an **identical interface shape** despite using different native GPU types. There is no abstract base class because type signatures differ per platform (e.g., `VkCommandBuffer` vs `id<MTLCommandBuffer>` vs `WGPUCommandEncoder`).
>
> Parity is enforced by convention: each renderer has `Create()`, `SetScene()`, `RenderFrame()`, `Resize()`, `SetSamplesPerPixel()`. Each has a platform-specific `GBuffer` struct with the same semantic fields but native types.
>
> The Vulkan renderer uses C++. Future Metal and WebGPU renderers expose a **pure-C API** for Swift and WASM interop.

### 6.5 Mobile Vulkan Renderer (`monti_vulkan_mobile`)

> Deferred — see [roadmap.md](roadmap.md#f6-mobile-vulkan-renderer-monti_vulkan_mobile) for full design. Hybrid rasterization + ray query pipeline exploiting TBDR tile memory. Shares `GpuScene`, `GeometryManager`, and GLSL include files with the desktop renderer.

### 6.6 Future Renderers

> See [roadmap.md](roadmap.md#f7f8-future-renderers) — Metal RT (C API) and WebGPU screen-space ray march (C API → WASM).

### 6.7 Future: ReSTIR DI

> Deferred — see [roadmap.md](roadmap.md#f2-restir-di--emissive-mesh-lights). ReSTIR inserts reservoir-based resampled importance sampling when local light sources are added. Initial implementation targets desktop; mobile enablement is feasible on newer SoCs with hardware RT (Snapdragon 8 Elite+, Immortalis-G925+) at mobile render resolution.

---

## 7. Tone Mapping & Presentation — Host Application

Tone mapping (ACES filmic, Reinhard, passthrough) and swapchain presentation are trivial single-shader operations. They live in the host application, not in separate libraries. The rtx-chessboard `ToneMapper` and compositing logic can be copied directly into the app.

```
app/src/ToneMapper.cpp   — compute shader: HDR → sRGB LDR
app/src/Presenter.cpp    — full-screen blit to swapchain image
app/shaders/tonemap.comp — ACES filmic + sRGB EOTF
```

---

## 8. Capture Writer — Internal Tooling

CPU-side. Reads pixel data from host-provided buffers and writes OpenEXR training pairs. Each frame produces **two** EXR files:

1. **Input EXR** (`frame_NNNNNN_input.exr`) — Low-SPP noisy radiance + G-buffer auxiliary channels at the input (render) resolution. Channels use **mixed per-channel bit depths**: radiance and auxiliary channels use FP16 (sufficient for training inputs), while `linear_depth` uses FP32 (avoids precision loss at long view distances). OpenEXR natively supports per-channel pixel types — each channel independently specifies `HALF`, `FLOAT`, or `UINT`.

2. **Target EXR** (`frame_NNNNNN_target.exr`) — High-SPP reference diffuse and specular radiance at the **target resolution** (input resolution × scale factor). Written in FP32 for maximum precision in ground-truth data.

The two-file design cleanly separates resolutions: input channels share one resolution, target channels share another. This avoids the complexity of multi-resolution data windows within a single EXR and is directly consumable by standard ML dataloaders that expect paired files.

> **Alpha channel:** The renderer currently hardcodes alpha to 1.0 for `noisy_diffuse` and `noisy_specular`. The EXR files include the `.A` channel so the format is ready for future transparency alpha computation. The writer writes whatever the caller provides.

> **tinyexr low-level API:** The convenience `SaveEXR()` function does not support per-channel bit depths. The implementation uses the low-level `EXRHeader`/`EXRImage`/`SaveEXRImageToFile()` API with per-channel `pixel_type` settings.

```cpp
// capture/include/monti/capture/Writer.h
#pragma once
#include <string>
#include <memory>
#include <cstdint>

namespace monti::capture {

// Scale factor for target resolution relative to input resolution.
// Mirrors deni::vulkan::ScaleMode — the capture writer is CPU-only
// and does not depend on Vulkan or Deni headers.
enum class ScaleMode {
    kNative,       // 1.0× — target resolution = input resolution
    kQuality,      // 1.5× — target = input × 1.5 (rounded to even)
    kPerformance,  // 2.0× — target = input × 2
};

struct WriterDesc {
    std::string   output_dir = "./capture/";
    uint32_t      input_width;          // Input (render) resolution
    uint32_t      input_height;
    ScaleMode     scale_mode = ScaleMode::kPerformance;  // Target = input × scale
    // Target resolution is computed internally:
    //   target_dim = floor(input_dim × scale_factor / 2) × 2
};

// Input channels — all at input resolution (WriterDesc::input_width × input_height).
// All pointers are to CPU-side float arrays. Null pointers are omitted from the
// output EXR. The writer uses per-channel bit depths: FP16 for radiance and
// auxiliary channels, FP32 for linear_depth.
struct InputFrame {
    const float* noisy_diffuse      = nullptr;  // 4 floats/pixel (RGBA) → FP16
    const float* noisy_specular     = nullptr;  // 4 floats/pixel (RGBA) → FP16
    const float* diffuse_albedo     = nullptr;  // 3 floats/pixel (RGB)  → FP16
    const float* specular_albedo    = nullptr;  // 3 floats/pixel (RGB)  → FP16
    const float* world_normals      = nullptr;  // 4 floats/pixel (XYZW) → FP16
    const float* linear_depth       = nullptr;  // 1 float/pixel         → FP32
    const float* motion_vectors     = nullptr;  // 2 floats/pixel (XY)   → FP16
};

// Target channels — at target resolution (derived from input × scale_mode).
// Written in FP32 for maximum ground-truth precision.
struct TargetFrame {
    const float* ref_diffuse        = nullptr;  // 4 floats/pixel (RGBA) → FP32
    const float* ref_specular       = nullptr;  // 4 floats/pixel (RGBA) → FP32
};

class Writer {
public:
    static std::unique_ptr<Writer> Create(
        const WriterDesc& desc);
    ~Writer();

    // Target resolution derived from WriterDesc.
    uint32_t TargetWidth() const;
    uint32_t TargetHeight() const;

    // Writes two EXR files per frame:
    //   {output_dir}/frame_{NNNNNN}_input.exr   — input channels at input resolution
    //   {output_dir}/frame_{NNNNNN}_target.exr  — target channels at target resolution
    bool WriteFrame(const InputFrame& input, const TargetFrame& target,
                    uint32_t frame_index);
};

} // namespace monti::capture
```

### 8.1 Per-Channel Bit Depths (Input EXR)

OpenEXR allows each channel to have an independent pixel type (`HALF`, `FLOAT`, or `UINT`). The input EXR uses mixed bit depths to balance file size and precision:

| Channel Group | EXR Channels | EXR Type | Rationale |
|---|---|---|---|
| `noisy_diffuse` (RGBA) | `noisy_diffuse.R/G/B/A` | `HALF` | HDR radiance fits well in FP16; alpha for future transparency |
| `noisy_specular` (RGBA) | `noisy_specular.R/G/B/A` | `HALF` | Same as diffuse |
| `diffuse_albedo` (RGB) | `diffuse_albedo.R/G/B` | `HALF` | Values in [0,1]; FP16 is more than sufficient |
| `specular_albedo` (RGB) | `specular_albedo.R/G/B` | `HALF` | Same as diffuse albedo |
| `normal` (XYZW) | `normal.X/Y/Z/W` | `HALF` | Unit vectors + roughness in [0,1] |
| `depth` (Z) | `depth.Z` | `FLOAT` | FP32 avoids precision loss at long view distances |
| `motion` (XY) | `motion.X/Y` | `HALF` | Sub-pixel motion precision is sufficient |

The target EXR uses `FLOAT` (FP32) for all channels — ground-truth reference data warrants maximum precision:

| Channel Group | EXR Channels | EXR Type |
|---|---|---|
| `ref_diffuse` (RGBA) | `ref_diffuse.R/G/B/A` | `FLOAT` |
| `ref_specular` (RGBA) | `ref_specular.R/G/B/A` | `FLOAT` |

---

## 10. Host Application Examples

### 10.1 Interactive Real-Time Display (Internal Tool)

```cpp
#include <monti/scene/Scene.h>
#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>
#include <deni/vulkan/Denoiser.h>

int main() {
    // ── Vulkan setup (host-owned) ──────────────────────────────────────
    VulkanContext vk = CreateVulkanContext(window, 1920, 1080);

    // ── G-buffer images (host-owned) ───────────────────────────────────
    GBufferImages gb = CreateGBufferImages(vk, 1920, 1080);

    // ── Scene (platform-agnostic) ──────────────────────────────────────
    monti::Scene scene;
    auto result = monti::gltf::LoadGltf(scene, "bistro.glb");

    // ── Renderer (Vulkan-specific) ─────────────────────────────────────
    auto renderer = monti::vulkan::Renderer::Create({
        .device             = vk.device,
        .physical_device    = vk.physical_device,
        .queue              = vk.queue,
        .queue_family_index = vk.queue_family,
        .pipeline_cache     = vk.pipeline_cache,
        .allocator          = vk.allocator,  // Shared VMA allocator
        .width = 1920, .height = 1080,
    });
    renderer->SetScene(&scene);

    // ── Upload geometry from loader to GPU ──────────────────────────────
    // Convenience function: uploads all meshes to GPU and registers bindings.
    // Hosts with their own buffer management can skip this and call
    // RegisterMeshBuffers() directly per mesh.
    VkCommandBuffer upload_cmd = vk.BeginOneShot();
    auto mesh_buffers = monti::vulkan::UploadAndRegisterMeshes(
        *renderer, vk.allocator, vk.device, upload_cmd,
        result.mesh_data);
    vk.SubmitAndWait(upload_cmd);
    // MeshData CPU vectors can now be discarded (result goes out of scope).
    // mesh_buffers must be kept alive for the renderer's lifetime.

    // ── Denoiser (Deni product — we are our own customer) ──────────────
    auto denoiser = deni::vulkan::Denoiser::Create({
        .device          = vk.device,
        .physical_device = vk.physical_device,
        .pipeline_cache  = vk.pipeline_cache,
        .allocator       = vk.allocator,
        .shader_dir      = "shaders/deni/",
        .get_device_proc_addr = vkGetDeviceProcAddr,
        .model_path      = "models/deni_unet.denimodel",  // Optional
        .width = 1920, .height = 1080,
    });

    // ── Render loop ────────────────────────────────────────────────────
    uint32_t frame_index = 0;
    while (running) {
        // ── Update transforms (e.g., user moved an object) ─────────────
        if (object_moved) {
            scene.SetNodeTransform(selected_node_id, new_transform);
            // No other calls needed — RenderFrame() detects the transform
            // change and rebuilds the TLAS automatically.
        }

        VkCommandBuffer cmd = vk.BeginFrame();

        renderer->RenderFrame(cmd, gb.gbuffer, frame_index);

        auto denoised = denoiser->Denoise(cmd, {
            .noisy_diffuse   = gb.gbuffer.noisy_diffuse,
            .noisy_specular  = gb.gbuffer.noisy_specular,
            .motion_vectors  = gb.gbuffer.motion_vectors,
            .linear_depth    = gb.gbuffer.linear_depth,
            .world_normals   = gb.gbuffer.world_normals,
            .diffuse_albedo  = gb.gbuffer.diffuse_albedo,
            .specular_albedo = gb.gbuffer.specular_albedo,
            .render_width    = 1920,
            .render_height   = 1080,
            .scale_mode      = deni::vulkan::ScaleMode::kNative,
            .reset_accumulation = false,
        });

        // Tone map + present (app-owned, not library code)
        ToneMap(cmd, denoised.denoised_color, vk.ldr_image,
                scene.GetActiveCamera().exposure_ev100);
        BlitToSwapchain(cmd, vk.ldr_image, vk.swapchain_image_view);
        vk.EndFrame();
        ++frame_index;
    }

    // ── Cleanup ────────────────────────────────────────────────────────
    vk.WaitIdle();
    for (auto& buf : mesh_buffers)
        monti::vulkan::DestroyGpuBuffer(vk.allocator, buf);
}
```

### 10.2 Training Data Capture (Headless)

The training data generator renders each frame at **two resolutions**: a low-SPP noisy render at the input resolution, and a high-SPP reference render at a higher target resolution (input × scale factor, using the `ScaleMode` enum). The capture writer produces two EXR files per frame — one for inputs, one for targets — at their respective resolutions.

```cpp
#include <monti/scene/Scene.h>
#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>
#include <monti/capture/Writer.h>

int main() {
    constexpr uint32_t input_w = 960, input_h = 540;
    // Target resolution computed by Writer from ScaleMode::kPerformance (2×)
    // → target_w = 1920, target_h = 1080

    VulkanContext vk = CreateVulkanContextHeadless();

    // Low-SPP noisy G-buffer at input resolution (compact formats)
    GBufferImages gb = CreateGBufferImages(vk, input_w, input_h);

    // High-SPP reference G-buffer at target resolution (RGBA32F for radiance,
    // RGBA16F for aux channels — higher precision than the compact G-buffer).
    // Compute target resolution using the same formula as ScaleMode.
    constexpr uint32_t target_w = 1920, target_h = 1080;  // 960×2, 540×2
    GBufferImages ref_gb = CreateGBufferImages(vk, target_w, target_h,
                                                HighPrecisionFormats());

    monti::Scene scene;
    auto result = monti::gltf::LoadGltf(scene, "bistro.glb");

    // Renderer supports rendering at different resolutions per call —
    // it uses the GBuffer's image dimensions, not RendererDesc dimensions.
    // RendererDesc::width/height set the maximum supported resolution.
    auto renderer = monti::vulkan::Renderer::Create({
        .device = vk.device, .physical_device = vk.physical_device,
        .queue = vk.queue, .queue_family_index = vk.queue_family,
        .allocator = vk.allocator,
        .width = target_w, .height = target_h,  // Max of the two resolutions
    });
    renderer->SetScene(&scene);

    // ── Upload geometry (same pattern as interactive example) ───────────
    VkCommandBuffer upload_cmd = vk.BeginOneShot();
    auto mesh_buffers = monti::vulkan::UploadAndRegisterMeshes(
        *renderer, vk.allocator, vk.device, upload_cmd,
        result.mesh_data);
    vk.SubmitAndWait(upload_cmd);

    auto writer = monti::capture::Writer::Create({
        .output_dir    = "./training_data/",
        .input_width   = input_w,
        .input_height  = input_h,
        .scale_mode    = monti::capture::ScaleMode::kPerformance,  // 2× target
    });

    // Query derived target resolution (for verification / G-buffer allocation)
    assert(writer->TargetWidth() == target_w);
    assert(writer->TargetHeight() == target_h);

    // Load scenes...

    for (uint32_t frame = 0; frame < total_frames; ++frame) {
        VkCommandBuffer cmd = vk.BeginFrame();

        // Low-SPP noisy render at input resolution
        renderer->SetSamplesPerPixel(4);
        renderer->RenderFrame(cmd, gb.gbuffer, frame);

        // High-SPP reference render at target resolution (2× input)
        // Same RenderFrame call — different GBuffer, different resolution,
        // higher SPP. The renderer is format-agnostic and resolution-agnostic.
        renderer->SetSamplesPerPixel(256);
        renderer->RenderFrame(cmd, ref_gb.gbuffer, frame);

        vk.EndFrame();

        // Read back and write two EXR files per frame:
        //   frame_NNNNNN_input.exr  — noisy + G-buffer at input_w × input_h
        //   frame_NNNNNN_target.exr — ref diffuse + specular at target_w × target_h
        writer->WriteFrame(
            {   // InputFrame — input resolution
                .noisy_diffuse      = noisy_diffuse_pixels.data(),
                .noisy_specular     = noisy_specular_pixels.data(),
                .diffuse_albedo     = diffuse_albedo_pixels.data(),
                .specular_albedo    = specular_albedo_pixels.data(),
                .world_normals      = normals_pixels.data(),
                .linear_depth       = depth_pixels.data(),
                .motion_vectors     = motion_pixels.data(),
            },
            {   // TargetFrame — target resolution
                .ref_diffuse        = ref_diffuse_pixels.data(),
                .ref_specular       = ref_specular_pixels.data(),
            },
            frame);
    }
}
```

### 10.3 Transform Update (TLAS-Only)

The most common dynamic case: objects move, rotate, or scale. Mesh data is unchanged — only the TLAS needs rebuilding. `SetNodeTransform()` saves the current transform as `prev_transform` (for motion vectors), sets the new transform, and increments the scene's `tlas_generation_` counter. The renderer's `GeometryManager` caches the last-seen generation and compares it at the start of each `RenderFrame()` — when unchanged (fully static frame), the TLAS rebuild is skipped entirely (O(1) check). When the generation has advanced, the TLAS is rebuilt with the current transforms.

```cpp
// Chess piece moves from e2 to e4
monti::Transform new_xform = scene.GetNode(pawn_node_id)->transform;
new_xform.translation = glm::vec3(4.0f, 0.0f, 3.0f);  // e4 position
scene.SetNodeTransform(pawn_node_id, new_xform);
// SetNodeTransform increments tlas_generation_ — renderer will detect this.

// Next RenderFrame() will rebuild TLAS with the new instance transform.
// No other calls needed. Cost: one vkCmdBuildAccelerationStructuresKHR
// for the TLAS (~0.1-0.5ms).
```

### 10.4 Vertex Deformation — Topology Unchanged (BLAS Refit)

Vertex positions change but triangle count and index buffer are identical. Common for skeletal animation, cloth simulation, morph targets. `NotifyMeshDeformed()` tells the renderer to **refit** (not rebuild) the BLAS on the next `RenderFrame()` — this updates bounding volumes while preserving the tree structure.

**The host is responsible for pipeline barriers** between their compute/transfer writes and `RenderFrame()`. The renderer cannot insert these barriers because it doesn't know which pipeline stage produced the vertex data.

```cpp
// ── Host runs cloth simulation on GPU ──────────────────────────────
VkCommandBuffer cmd = vk.BeginFrame();

// 1. Host dispatches cloth simulation compute shader.
//    Writes directly into the vertex buffer that was registered
//    with Renderer::RegisterMeshBuffers().
DispatchClothSimulation(cmd, cloth_vertex_buffer, wind_params, dt);

// 2. Barrier: host's compute write → renderer's AS build read.
//    This is the HOST's responsibility — the renderer cannot know
//    what pipeline stage wrote the vertex data.
VkMemoryBarrier2 barrier{
    .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
    .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
    .dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
};
VkDependencyInfo dep{
    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    .memoryBarrierCount = 1,
    .pMemoryBarriers    = &barrier,
};
vkCmdPipelineBarrier2(cmd, &dep);

// 3. Notify renderer: vertices changed, topology same → BLAS refit.
renderer->NotifyMeshDeformed(cloth_mesh_id, /*topology_changed=*/false);

// 4. RenderFrame() processes the notification:
//    - Refits the cloth mesh's BLAS (fast — updates bounds only)
//    - Rebuilds TLAS
//    - Path traces
renderer->RenderFrame(cmd, gb.gbuffer, frame_index);

// ... denoise, tonemap, present ...
vk.EndFrame();
```

**Cost:** BLAS refit per dirty mesh (`VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR`) + TLAS rebuild. Refit is significantly faster than full rebuild because it preserves the BVH tree structure and only updates bounding boxes. Quality degrades over many frames of continuous deformation — consider a periodic full rebuild (e.g., every 60 frames) by passing `topology_changed = true`.

### 10.5 Topology Change (BLAS Full Rebuild)

The mesh's triangle count, index buffer, or vertex count change. Common for LOD switching, procedural generation, fracture/destruction. The host re-registers the buffer binding with potentially new buffers and signals a topology change.

```cpp
// ── LOD switch: replace high-detail mesh with low-detail ───────────
// The host has pre-uploaded multiple LOD levels to separate GPU buffers.

// 1. Update Mesh metadata for the new LOD.
scene.GetNode(building_node_id)->mesh_id = lod2_mesh_id;

// 2. Register buffer bindings for the new LOD mesh
//    (if not already registered during initial load).
renderer->RegisterMeshBuffers(lod2_mesh_id, {
    .vertex_buffer  = lod2_vb.buffer,
    .vertex_address = lod2_vb.device_address,
    .index_buffer   = lod2_ib.buffer,
    .index_address  = lod2_ib.device_address,
    .vertex_count   = lod2_vertex_count,
    .index_count    = lod2_index_count,
    .vertex_stride  = sizeof(monti::Vertex),
});

// 3. RenderFrame() builds a new BLAS for the LOD2 mesh (if not cached)
//    and rebuilds the TLAS with the updated node→mesh mapping.
renderer->RenderFrame(cmd, gb.gbuffer, frame_index);
```

For **in-place topology changes** (same MeshId, new buffer contents):

```cpp
// ── Procedural mesh regenerated on GPU (e.g., adaptive tessellation) ──

// 1. Host dispatches compute to regenerate mesh into the same buffer.
DispatchProceduralGeneration(cmd, terrain_vb, terrain_ib,
                             &new_vertex_count, &new_index_count);

// 2. Barrier: compute write → AS build read (host responsibility).
vkCmdPipelineBarrier2(cmd, &dep);  // Same barrier pattern as §10.4

// 3. Re-register with updated counts (buffer handle may be the same).
renderer->RegisterMeshBuffers(terrain_mesh_id, {
    .vertex_buffer  = terrain_vb.buffer,
    .vertex_address = terrain_vb.device_address,
    .index_buffer   = terrain_ib.buffer,
    .index_address  = terrain_ib.device_address,
    .vertex_count   = new_vertex_count,
    .index_count    = new_index_count,
    .vertex_stride  = sizeof(monti::Vertex),
});

// 4. Notify: topology changed → full BLAS rebuild required.
renderer->NotifyMeshDeformed(terrain_mesh_id, /*topology_changed=*/true);

// 5. RenderFrame() fully rebuilds the BLAS for this mesh + TLAS.
renderer->RenderFrame(cmd, gb.gbuffer, frame_index);
```

**Cost:** Full BLAS build + optional compaction for the affected mesh. More expensive than refit but required when triangle count or connectivity changes. `RenderFrame()` batches all dirty BLAS builds into a single `vkCmdBuildAccelerationStructuresKHR` call.

### 10.6 Object Removal

```cpp
// Register cleanup callback once at startup
renderer->SetMeshCleanupCallback([&](monti::MeshId mesh_id) {
    // Schedule deferred buffer destruction (after in-flight frames complete).
    // Production code queues these for cleanup after kFramesInFlight fences.
    pending_buffer_cleanup.push_back(mesh_id);
});

// ...

// Remove a scene node (e.g., object destroyed, picked up, etc.)
scene.RemoveNode(door_node_id);

// Next RenderFrame() excludes this node from the TLAS. If no other
// nodes reference the same mesh, the renderer destroys its BLAS and
// invokes the cleanup callback with the MeshId. The host then frees
// the GPU buffer after ensuring no in-flight frames reference it.
vk.WaitIdle();  // Simple approach; production code uses per-frame fences
for (auto id : pending_buffer_cleanup) {
    auto& buf = mesh_gpu_buffers[id];
    monti::vulkan::DestroyGpuBuffer(vk.allocator, buf.vertex);
    monti::vulkan::DestroyGpuBuffer(vk.allocator, buf.index);
}
pending_buffer_cleanup.clear();
```

---

## 11. Render Pipeline Internals (Vulkan)

### 11.1 Current Implementation (MIS Path Tracing)

Directly adapted from rtx-chessboard's `HWPathTracer`, enhanced with quality and material features from Phases 8E–8N:

```
RenderFrame(cmd, GBuffer):
│
├─ [Internal] Acceleration structure maintenance
│    ├─ Build BLAS for newly registered meshes
│    ├─ Refit BLAS for meshes flagged via NotifyMeshDeformed(id, false)
│    ├─ Rebuild BLAS for meshes flagged via NotifyMeshDeformed(id, true)
│    ├─ Destroy BLAS for meshes removed from scene
│    │   └─ Invoke MeshCleanupCallback per destroyed BLAS
│    └─ Rebuild TLAS from current scene node transforms
│
├─ [Internal] Update material/texture/light GPU buffers (GpuScene)
│    ├─ Pack materials (176 bytes/material, 11 vec4s)
│    ├─ Upload textures (RGBA8, BC1-BC7 compressed DDS support — Phase 8N)
│    └─ Pack lights (quad + sphere + triangle in unified PackedLight buffer — Phase 8G)
│
├─ [Internal] Emissive light extraction (Phase 8J)
│    └─ Scan emissive materials, extract TriangleLights for NEE sampling
│
├─ [GPU] Path Trace — MIS (may loop for high SPP)
│    For each batch of samples (split if SPP exceeds per-dispatch limit):
│      vkCmdTraceRaysKHR (gl_RayFlagsNoneEXT — non-opaque for alpha masking):
│        raygen:     per-pixel rays, N spp, blue noise sampling,
│                    projection-matrix sub-pixel jitter (Halton 2,3),
│                    thin-lens DoF (aperture_radius, focus_distance),
│                    material fetch, BRDF evaluation, direct lighting,
│                    ray cone tracking for texture LOD (Phase 8F)
│        anyhit:     alpha masking for AlphaMode::kMask (texture lookup,
│                    discard if alpha < cutoff; opaque geometry skips via
│                    VK_GEOMETRY_OPAQUE_BIT)
│        closesthit: barycentric interpolation, normal/UV via HitPayload,
│                    per-material UV transform (Phase 8L)
│        miss:       sets missed flag (environment sampled in raygen)
│      Per-path bounce loop (max bounces configurable, default 4 + 8 transparency headroom):
│        - 5-way MIS: diffuse, specular, clear coat, sheen (Phase 8M), environment
│        - Cook-Torrance BRDF + GGX microfacet
│        - Sheen BRDF with precomputed albedo LUT (Phase 8M)
│        - Fresnel refraction + thin-slab Beer-Lambert attenuation
│        - Diffuse transmission for thin surfaces (Phase 8H)
│        - Nested dielectric priority tracking (Phase 8I)
│        - Emissive surface contribution on path hits (Phase 8J)
│        - Hybrid NEE: direct per-light shadow rays (≤4 lights) or WRS single-light
│          selection (>4 lights) for quad/sphere/triangle lights (Phase 8G/8J/8K)
│        - Russian roulette after bounce 3
│        - Separate diffuse/specular classification
│        - G-buffer written from first non-fully-transparent hit
│      Accumulate results into GBuffer image views
│
├─ [GPU] Firefly filter (Phase 8E)
│    └─ Post-trace median-based outlier suppression on noisy radiance
│
└─ Done. Host calls denoiser, tone map, present.
```

### 11.2 Future Enhancement: ReSTIR DI

> Deferred — see [roadmap.md](roadmap.md#restir-pipeline-overview) for pipeline insertion details and reservoir buffer layout.

---

## 12. Build System

```cmake
cmake_minimum_required(VERSION 3.24)
project(monti LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── PRODUCT (Deni) ───────────────────────────────────────────────────────────
add_library(deni_vulkan STATIC
    denoise/src/vulkan/Denoiser.cpp
    denoise/src/vulkan/WeightLoader.cpp
    denoise/src/vulkan/MlInference.cpp)
target_include_directories(deni_vulkan PUBLIC denoise/include)
target_link_libraries(deni_vulkan
    PRIVATE Vulkan::Headers GPUOpen::VulkanMemoryAllocator)
# No GLM dependency in public headers. VMA for suballocation.
# SPIR-V shaders compiled via glslc and loaded from .spv files at runtime.

# Deni shader compilation (GLSL → SPIR-V)
set(DENI_SHADER_SOURCES
    ${DENI_SHADER_DIR}/passthrough_denoise.comp)
# Output: ${CMAKE_CURRENT_BINARY_DIR}/deni_shaders/*.spv

# ── INTERNAL TOOLING (Monti) ─────────────────────────────────────────────────
add_library(monti_scene STATIC
    scene/src/Scene.cpp
    scene/src/gltf/GltfLoader.cpp)
target_include_directories(monti_scene PUBLIC scene/include)
target_link_libraries(monti_scene PRIVATE glm cgltf)
# No Vulkan dependency — platform-agnostic.

add_library(monti_vulkan STATIC
    renderer/src/vulkan/Renderer.cpp
    renderer/src/vulkan/Buffer.cpp
    renderer/src/vulkan/Image.cpp
    renderer/src/vulkan/Upload.cpp
    renderer/src/vulkan/GpuScene.cpp
    renderer/src/vulkan/GpuBufferUtils.cpp
    renderer/src/vulkan/GeometryManager.cpp
    renderer/src/vulkan/EnvironmentMap.cpp
    renderer/src/vulkan/BlueNoise.cpp
    renderer/src/vulkan/RaytracePipeline.cpp
    renderer/src/vulkan/EmissiveLightExtractor.cpp)
target_include_directories(monti_vulkan PUBLIC renderer/include)
target_link_libraries(monti_vulkan
    PUBLIC monti_scene
    PRIVATE Vulkan::Headers glm GPUOpen::VulkanMemoryAllocator)
# SPIR-V shaders compiled via glslc and loaded from .spv files at runtime.

# Monti shader compilation (GLSL → SPIR-V)
set(MONTI_SHADER_SOURCES
    ${MONTI_SHADER_DIR}/raygen.rgen
    ${MONTI_SHADER_DIR}/miss.rmiss
    ${MONTI_SHADER_DIR}/closesthit.rchit
    ${MONTI_SHADER_DIR}/anyhit.rahit)
# Output: ${CMAKE_CURRENT_BINARY_DIR}/monti_shaders/*.spv

# monti_vulkan_mobile is deferred to roadmap phase F6.
# It will be added when the mobile renderer is implemented.

add_library(monti_capture STATIC
    capture/src/Writer.cpp)
target_include_directories(monti_capture PUBLIC capture/include)
target_link_libraries(monti_capture PRIVATE tinyexr)
# No Vulkan dependency — CPU-side only.

# ── Test Infrastructure ──────────────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(flip
    GIT_REPOSITORY https://github.com/NVlabs/flip.git
    GIT_TAG        main)
FetchContent_MakeAvailable(flip)

option(MONTI_DOWNLOAD_TEST_ASSETS "Download Khronos glTF-Sample-Assets" ON)
option(MONTI_DOWNLOAD_BENCHMARK_SCENES "Download heavy benchmark scenes" OFF)

add_executable(monti_tests tests/main_test.cpp tests/scenes/CornellBox.cpp)
target_link_libraries(monti_tests PRIVATE
    monti_vulkan monti_scene deni_vulkan flip::tool)
# Tests use real Vulkan — no mocking, no HAL. SwiftShader/lavapipe for CI.
```

### 12.1 Android NDK Cross-Compilation

The `deni_vulkan` and `monti_vulkan_mobile` libraries target Android via the NDK's CMake toolchain. The Vulkan headers are provided by the NDK (API level 30+ includes Vulkan 1.2 headers); no separate Vulkan SDK is required.

```bash
# Prerequisites: Android NDK r26+ (includes Vulkan 1.2 headers)
# Set ANDROID_NDK to your NDK installation path.

# Build deni_vulkan for Android arm64-v8a:
cmake -B build-android -S . \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-30 \
    -DANDROID_STL=c++_shared \
    -DMONTI_BUILD_APP=OFF \
    -DMONTI_BUILD_TESTS=OFF

cmake --build build-android --target deni_vulkan monti_vulkan_mobile
```

**CMake options for Android builds:**

| Option | Default | Purpose |
|---|---|---|
| `MONTI_BUILD_APPS` | `ON` | Build `monti_view` and `monti_datagen` (disable for library-only Android builds) |
| `MONTI_BUILD_TESTS` | `ON` | Build test executables (disable for Android — tests run on host) |

**Integration into an Android project (Gradle + CMake):**

In the app's `build.gradle.kts`:
```kotlin
android {
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            arguments += "-DMONTI_BUILD_APP=OFF"
            arguments += "-DMONTI_BUILD_TESTS=OFF"
        }
    }
}
```

In the app's native `CMakeLists.txt`, add the Monti repo as a subdirectory:
```cmake
add_subdirectory(${MONTI_SOURCE_DIR} ${CMAKE_BINARY_DIR}/monti)
target_link_libraries(my_native_lib PRIVATE deni_vulkan)
# Or for full renderer: target_link_libraries(my_native_lib PRIVATE monti_vulkan_mobile)
```

**Notes:**
- SPIR-V shaders are pre-compiled and loaded from files at runtime — the `shader_dir` path must point to a directory containing the `.spv` files (typically bundled as Android assets).
- `VkPipelineCache` is strongly recommended on Android (50–500ms pipeline compilation without it). The host should serialize/deserialize the cache between app launches.
- SDL3 is not needed for library-only builds. The app (when built) uses SDL3 for windowing; the libraries have no SDL3 dependency.
- **C API for Android JNI:** The Vulkan libraries currently expose a C++ API, which is usable from NDK C++ code linked via JNI. If demand arises for direct Kotlin/Java interop without a C++ bridge, a pure-C wrapper (`deni_denoiser_create()`, etc.) can be added using the same pattern planned for Metal and WebGPU (see §4.6).

---

## 13. Testing Strategy

### 13.1 Automated Render Validation

All render output tests use **NVIDIA FLIP** (BSD-3) for perceptual image comparison. FLIP produces a per-pixel error map and summary statistics (mean, median, max). This replaces manual visual inspection.

**Principles:**

- **GPU-side integration tests are the default.** Every rendering feature is tested by actually rendering images on the GPU — load a scene, configure the feature under test, render frames, read back pixels, and verify measurable properties. There is no CPU-side reimplementation of shader logic for testing purposes. If a feature runs on the GPU, it is tested on the GPU.
- **Each test targets a specific feature and detects regressions.** The scene, materials, and camera are chosen to isolate the feature, and the pass/fail criterion is chosen so that breaking the feature causes the test to fail.
- **No feature toggles for testing.** The renderer is an uber-shader — all features are always enabled. Tests isolate features through scene design (stress inputs, spatial properties, two-scene comparisons), not by compiling out or disabling features.
- **Unit tests only for complex isolated CPU logic.** Never reimplement GPU shader functions on the CPU for testing — test them through rendered output instead.
- **Vulkan validation layers are always on** in debug builds. Zero validation errors is a pass/fail gate for every GPU phase.

**Three-tier validation:**

| Tier | What it proves | Stored references? | FLIP threshold |
|------|----------------|-------------------|----------------|
| **Feature-specific regression** | A specific rendering feature produces the expected measurable signal — scene designed so a regression changes the signal | No — both images rendered at test time | Per-test (pixel property checks, variance ratios, FLIP deltas between two scene configurations) |
| **Self-consistency (convergence)** | Renderer converges — low SPP and high SPP produce the same image up to noise | No — both images rendered at test time | Mean FLIP < configurable threshold |
| **Golden reference (regression)** | Output hasn't changed from a known-good baseline | Yes — per-platform images committed to `tests/references/` | Mean FLIP < 0.05 |

Feature-specific regression tests and convergence tests are the primary automated gates. Golden reference tests catch regressions but require updating stored images when intentional changes occur. All test types produce FLIP error maps and/or diagnostic PNGs as artifacts for debugging failures.

### 13.2 Real GPU Testing — No Mocking

Tests run against the real Vulkan (or Metal / WebGPU) API. No abstract base class, no hardware abstraction layer, no mocks. Each platform's test binary links its native backend and exercises the actual GPU code path.

For CI environments without a physical GPU, use **SwiftShader** or **lavapipe** (Mesa's software Vulkan ICD). Both support `VK_KHR_ray_tracing_pipeline` to varying degrees. Reference images for software renderers are stored separately from hardware references due to floating-point divergence.

Cross-platform strategy: each platform runs its own native test suite. Vulkan tests run on Windows/Linux, Metal tests run on macOS, WebGPU tests use Dawn's headless mode.

### 13.3 Test Scenes

| Scene | Source | Purpose |
|-------|--------|---------|
| **Cornell box** (programmatic) | `tests/scenes/CornellBox.h` | Basic path tracing correctness — GI, color bleeding, convergence |
| `Box.glb` | Khronos glTF-Sample-Assets | Simplest geometry load validation |
| `DamagedHelmet.glb` | Khronos glTF-Sample-Assets | PBR materials: metallic, roughness, normal maps, 5 texture maps |
| `DragonAttenuation.glb` | Khronos glTF-Sample-Assets | Transparency, volume attenuation, refraction |
| `MosquitoInAmber.glb` | Khronos glTF-Sample-Assets | Nested transparency, subsurface-like effects |
| `ClearCoatTest.glb` | Khronos glTF-Sample-Assets | Clear coat BRDF validation |
| `MaterialsVariantsShoe.glb` | Khronos glTF-Sample-Assets | Multi-material PBR stress test |

Heavy benchmark scenes (opt-in via `MONTI_DOWNLOAD_BENCHMARK_SCENES`): Amazon Lumberyard Bistro (~2.8M triangles), Intel Sponza (~262K triangles), San Miguel (~7.8M triangles). These validate performance and large-scene stability.

---

## 14. Open Questions

1. **Shader permutations** — Bitmask-keyed pipeline cache for material variants. Consider when profiling shows register pressure or divergence from uber-shader approach. An uber-shader with dynamic branching (like rtx-chessboard) is acceptable initially.

2. **Video capture** — H.265 via FFmpeg to reduce training data storage. Evaluate before large-scale generation.

3. **Reservoir buffer layout** — Define packed format (16 bytes/pixel) before ReSTIR implementation.

4. **Temporal upscaling / super-resolution** — The ML denoiser can combine denoising + upscaling in a single pass. The `DenoiserInput` already supports distinct render/output dimensions (see §4.11). The capture writer produces dual-resolution training pairs: input EXR at render resolution, target EXR at render × scale factor (controlled by `ScaleMode` — initially 2× via `kPerformance`). On mobile, rendering at 540p and upscaling to 1080p is the expected usage pattern.

5. ~~**Reference render accumulation**~~ — **Resolved.** `RenderFrame()` supports arbitrarily high SPP values. If the requested SPP exceeds the per-dispatch limit (tuned to avoid GPU timeout / TDR), the renderer internally splits into multiple trace dispatches and accumulates the results into the G-buffer. The host always calls `RenderFrame()` once per frame regardless of SPP. See §6.3.

6. **G-buffer lifetime** — Denoiser needs frame N-1 history (when temporal denoising is implemented). Host must not destroy/resize G-buffer between frames without `reset_accumulation = true`.

7. **GPU deformation synchronization** — The host is responsible for pipeline barriers between their compute/transfer writes to vertex buffers and the next `RenderFrame()` call. The renderer cannot insert these barriers because it doesn't know which pipeline stage wrote the vertex data. See §10.4 for a complete code example with `VkMemoryBarrier2` from `VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT` to `VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR`.

8. **Mobile uber-shader register pressure** — The single compute shader for mobile path tracing combines all material evaluation into one entry point. Profile on Adreno/Mali to determine if shader specialization constants or permutations are needed to maintain occupancy.

9. ~~**ML denoiser training**~~ — **Resolved.** The training pipeline is implemented in `training/`. A 3-level U-Net (~120K params, 13 input channels, 3 output channels) is trained on captured EXR pairs. Weights are exported to `.denimodel` binary format via `export_weights.py` and loaded by `WeightLoader` at runtime. GPU inference via GLSL compute shaders is being implemented (Phase F11).

10. **C API design for Metal/WebGPU** — The pure-C API wrapper for Swift and WASM interop needs a consistent naming convention (e.g., `deni_vulkan_create()`, `monti_metal_render_frame()`). Define the C API pattern before implementing Metal or WebGPU backends.

---

## 15. Design Decisions Log

Key decisions made during the design process and their rationale:

1. **No HAL.** A Hardware Abstraction Layer was considered and rejected. All GPU components use native platform types directly. The denoiser product benefits most — customers use their existing Vulkan/Metal/WebGPU types with zero foreign abstractions.

2. **Deni (denoiser) is the product; Monti (renderer) is internal tooling.** The denoiser has the widest addressable market. The renderer generates training data and serves as a reference implementation. Internal tools use the product denoiser (we are our own first customer), never the reverse.

3. **Platform-specific components throughout.** Renderer, denoiser are per-platform with native types. Tone mapping and presentation live in the host app (not separate libraries). Only the scene layer and capture writer are platform-agnostic.

4. **Identical interface shape, no abstract base class.** All platform implementations of both the renderer and denoiser follow the same method names, parameter semantics, and lifecycle — but with platform-native types. Since `VkCommandBuffer` ≠ `id<MTLCommandBuffer>` ≠ `WGPUCommandEncoder`, a shared base class is impossible without a HAL. Parity is enforced by convention, documentation, and testing.

5. **C++ for Vulkan, pure-C API for Metal and WebGPU.** The Vulkan implementation targets desktop C++ applications. Metal needs a C API callable from Swift. WebGPU needs a C API callable from TypeScript/JavaScript via emscripten WASM. The C APIs are thin wrappers over the same platform-native implementations.

6. **Host owns GPU resources, including geometry buffers.** The host creates the device, command buffers, frame synchronization, G-buffer images, and vertex/index GPU buffers. Components record into the host's command buffer. The renderer references host-provided buffers via device addresses — it never uploads vertex/index data itself.

7. **Materials/textures as CPU-side scene data; geometry as GPU-side buffer references.** Materials and textures are stored in the scene layer as CPU descriptions and packed/uploaded by the renderer's GpuScene. Geometry (vertex/index buffers) is owned entirely by the host — the renderer receives device addresses and builds acceleration structures from them. This avoids CPU↔GPU roundtrips for geometry and supports GPU-generated meshes.

8. **GLM as an explicit dependency for Monti; not for Deni.** Monti uses GLM for vector/matrix types in its public headers (scene layer, renderer desc). Deni's public header has no GLM dependency — all types are Vulkan-native or scalar. When ReLAX adds matrix inputs, they will use raw `float[16]` arrays to keep the denoiser header GLM-free and compatible with the future pure-C API.

9. **Tagged template IDs with `operator<=>`.** Directly adopted from rtx-chessboard's `TypedId<Tag>` pattern — type-safe, sortable, hashable.

10. **Passthrough denoiser first, ReLAX later.** The initial denoiser validates the full pipeline (image contracts, command recording, host integration) without the complexity of temporal-spatial filtering. ReLAX is a drop-in upgrade.

11. **Emissive materials rendered via extraction + hybrid NEE.** The glTF loader parses emissive attributes per-spec. The `EmissiveLightExtractor` (Phase 8J) scans emissive materials, decomposes their mesh faces into `TriangleLight` entries, and adds them to the scene for explicit next-event estimation (NEE) sampling. Emissive surfaces also contribute radiance when hit by path-traced rays. For scenes with few lights (≤ 4), all lights are sampled directly with per-light shadow rays (preserving quality). For scenes with many lights (e.g., dozens of extracted emissive triangles), Phase 8K's weighted reservoir sampling (WRS) selects one light per bounce proportional to estimated contribution, with one-sample MIS against the BSDF strategy. This provides correct emissive rendering without requiring ReSTIR — ReSTIR DI (Phase F2) adds temporal/spatial reservoir resampling on top for further variance reduction.

12. **Separate diffuse/specular output.** Following rtx-chessboard, the path tracer classifies contributions by first opaque bounce into separate images. This enables demodulated denoising (DLSS-RR style) and is required for future ML denoiser training data.

13. **Transmission implemented, not deferred.** Fresnel refraction, IOR, and volume attenuation are included in the initial path tracer. Transmission is essential for glass, water, and gem materials in test scenes (DragonAttenuation, MosquitoInAmber). Deferring would leave a conspicuous gap in material support. Thickness factor/texture (a rasterizer approximation) is intentionally excluded — the path tracer uses actual ray-traced distances (`payload.hit_t`) through manifold volumes, matching NVIDIA RTXPT's approach.

14. **Tone mapper and presenter in app, not libraries.** These are trivial single-shader operations (ACES filmic compute, swapchain blit). Packaging them as separate libraries with their own `Create()`/`Destroy()` lifecycle adds complexity without value. They live in the host app and can be copied from rtx-chessboard.

15. **Mobile via hybrid rasterization + ray query.** Mobile GPUs support `VK_KHR_ray_query` but not `VK_KHR_ray_tracing_pipeline`. The mobile renderer rasterizes primary visibility in a TBDR-friendly render pass, then uses `rayQueryEXT` in compute shaders for indirect bounces. This exploits tile memory for the G-buffer pass and cuts ray tracing workload ~40–60%. The same scene layer, GpuScene, and GLSL include library are shared with the desktop renderer.

16. **VMA required in desc.** Both `deni_vulkan` and `monti_vulkan` require a host-provided `VmaAllocator` for all internal GPU memory allocations. This avoids hidden internal allocators competing with the host for memory and simplifies the implementation (no conditional allocator creation). VMA is a header-only library and widely adopted — requiring it is not a meaningful burden for customers already using Vulkan.

17. **SPIR-V loaded from files, GLSL shipped.** Compiled SPIR-V shaders are built by `glslc` at CMake time and output to build directories. At runtime, libraries load `.spv` files from a `shader_dir` path provided in the descriptor. GLSL source files are shipped alongside for debugging, extension, and audit. This approach avoids embedding binary blobs in source and simplifies shader iteration (recompile SPIR-V without relinking). The `shader_dir` is a required field in both `DenoiserDesc` and `RendererDesc`.

18. **BLAS/TLAS building inside RenderFrame, not exposed to host.** The `GeometryManager` is internal to the renderer. `RenderFrame()` automatically builds BLAS for new meshes, refits BLAS for deformed meshes, rebuilds BLAS for topology changes, cleans up BLAS for removed meshes, and rebuilds the TLAS. The host signals geometry changes via `RegisterMeshBuffers()` (for new/changed buffers) and `NotifyMeshDeformed()` (for vertex updates). This keeps the host API simple and avoids exposing acceleration structure internals. The alternative — host calls `BuildAllBlas()` + `BuildTlas()` explicitly — was considered for engine integrations that need control over barrier placement and build timing, but rejected because (a) the common case (static geometry + transform animation) requires zero host calls beyond `SetNodeTransform()`, and (b) the deformation case only requires a barrier + `NotifyMeshDeformed()` before `RenderFrame()`.

19. **Mesh lifecycle managed via Scene; renderer syncs automatically.** When the host calls `Scene::RemoveNode()` or `Scene::RemoveMesh()`, the renderer detects the removal at the next `RenderFrame()` and cleans up internal BLAS entries. There is no separate `UnregisterMeshBuffers()` call — the renderer's view of the scene is always derived from the Scene's current state. The host is notified of BLAS cleanup via `SetMeshCleanupCallback()` — a callback invoked during `RenderFrame()` for each cleaned-up mesh. The host is responsible for ensuring GPU buffers are not freed while in-flight frames still reference them (wait for fence or defer cleanup by N frames before calling `DestroyGpuBuffer()`).

20. **Optional GPU buffer upload helpers.** `monti::vulkan::UploadMeshToGpu()` and `UploadAndRegisterMeshes()` are convenience helpers for hosts that don't already manage GPU geometry buffers. They live in `GpuBufferUtils.h`, use VMA for allocation, and produce `GpuBuffer` structs ready for `Renderer::RegisterMeshBuffers()`. `UploadAndRegisterMeshes()` combines upload + registration in one call to reduce boilerplate (see §10.1). Hosts with their own buffer management (game engines) skip these entirely and call `RegisterMeshBuffers()` directly. Each platform backend provides its own equivalent (Metal: `MTLBuffer` helpers, WebGPU: `GPUBuffer` helpers).

21. **Format-agnostic G-buffer access.** Shaders use `shaderStorageImageReadWithoutFormat` / `shaderStorageImageWriteWithoutFormat` to read and write G-buffer images in whatever format the host allocated. The recommended compact formats (RG16F motion, R16F depth, R11G11B10F albedo) yield 32% bandwidth savings (38 vs 56 bytes/pixel) and are the default in the app, but RGBA16F is fully supported for any channel. No shader permutations or format negotiation required — the host simply allocates images in its preferred format.

22. **ReLAX and ReSTIR target desktop initially.** ReLAX’s 7 full-screen compute passes consume ~800+ MB bandwidth at 1080p, exceeding the mobile per-frame budget on older GPUs. ReSTIR adds 3 more full-screen passes. On mobile, the ML-trained denoiser (single-pass inference) is the planned denoiser; environment-only MIS with 1–2 SPP is the baseline lighting strategy. However, newer mobile SoCs with dedicated RT hardware (Snapdragon 8 Elite+, Dimensity 9400+, Immortalis-G925+) can handle ReSTIR at mobile render resolution (540p), where bandwidth costs are ~4× lower. Mobile ReSTIR enablement is a follow-up to the desktop implementation once the mobile renderer (F6) is in place.

23. **Hybrid rasterization + ray query as default mobile option.** Primary visibility is rasterized in a standard render pass (exploiting TBDR tile memory) by default on mobile. Only indirect bounces, shadows, and reflections use `rayQueryEXT` in compute. This cuts ray tracing workload ~40–60% and is the only way to get TBDR benefits in a path tracing pipeline. Camera jitter is applied as a sub-pixel projection matrix offset (standard TAA technique), providing equivalent temporal AA accumulation to per-ray jitter at 1 SPP. A pure ray-query compute path remains available for the mobile renderer when TBDR is not a factor or maximum single-frame quality is preferred.

24. **Fragment shader denoiser on mobile.** The ML denoiser on mobile runs as a fullscreen fragment shader instead of compute, enabling the denoise → tone map → present chain to execute as render pass subpasses with intermediate results staying in tile memory. The `Denoiser` selects compute (desktop) or fragment (mobile) automatically based on device properties. This is an internal implementation choice, not an API change — `Denoise(cmd, input)` returns `DenoiserOutput` in both cases. The additional setup (~100 lines: render pass, framebuffer, graphics pipeline vs. compute pipeline) is contained within the denoiser implementation.

25. **Super-resolution via discrete ScaleMode presets.** The `ScaleMode` enum in `DenoiserInput` controls combined denoise + upscale at three fixed ratios (1×, 1.5×, 2×). Each ratio maps to a trained ML model variant. Discrete presets avoid the quality loss of arbitrary-ratio upscaling and simplify training. Output dimensions are derived as `floor(render_dim × scale_factor / 2) × 2`. This is particularly valuable on mobile (540p → 1080p at `kPerformance` = 4× fewer rays).

26. **Host-provided Vulkan dispatch (required).** Both `DenoiserDesc` and `RendererDesc` require `PFN_vkGetDeviceProcAddr get_device_proc_addr`. All Vulkan device-level functions are resolved through it at `Create()` time and stored in an internal dispatch table. This makes Deni and Monti completely loader-agnostic — the host can use volk, the Vulkan SDK loader, or any custom dispatch mechanism. The libraries link only against `Vulkan::Headers` (type definitions), not `Vulkan::Vulkan` (the loader).

27. **No separate ReferenceTarget — GBuffer for both interactive and capture.** The renderer has a single `RenderFrame(cmd, GBuffer, frame_index)` entry point. For capture, the host allocates a second GBuffer at the **target resolution** (input × scale factor) with higher-precision formats (e.g., RGBA32F for radiance channels) and renders at high SPP. The renderer is format-agnostic (design decision 21) and resolution-agnostic (it uses the GBuffer's image dimensions), so the same shader code writes to compact interactive images at one resolution or full-precision capture images at another. The capture writer produces **two EXR files per frame**: an input EXR (noisy radiance + G-buffer at input resolution, mixed FP16/FP32 per-channel) and a target EXR (high-SPP reference `ref_diffuse` + `ref_specular` at target resolution, FP32). This eliminates a separate struct and a second `RenderFrame` overload while giving training more data (split diffuse/specular reference instead of merged) at the resolution needed for super-resolution training.

28. **RenderFrame accumulates internally for high SPP.** `RenderFrame()` supports arbitrarily high SPP values set via `SetSamplesPerPixel()`. If the requested SPP exceeds a per-dispatch limit (tuned to avoid GPU timeout / TDR, e.g., 16–32 SPP per dispatch on desktop), the renderer internally issues multiple `vkCmdTraceRaysKHR` dispatches within the same command buffer, accumulating results into the G-buffer between dispatches. The host always calls `RenderFrame()` once per frame. This keeps the host API simple — `SetSamplesPerPixel(256)` followed by `RenderFrame()` just works for high-SPP reference capture without the host managing accumulation loops.

29. **Mesh cleanup via callback, not polling.** `SetMeshCleanupCallback()` replaces the earlier `TakeCleanedUpMeshIds()` polling pattern. The callback is invoked during `RenderFrame()` for each mesh whose BLAS was destroyed (because no scene nodes reference it anymore). This is push-based — the host cannot forget to poll. The callback receives the `MeshId` so the host can schedule deferred GPU buffer destruction (after in-flight frames complete). The callback runs synchronously within `RenderFrame()`, so the host should not perform blocking GPU operations inside it.

30. **Separate per-mesh buffers with buffer address table, not merged buffers.** rtx-chessboard merges all vertex/index data into a single buffer pair with a `MeshGPURange` offset table. Monti keeps per-mesh separate buffers owned by the host. Shaders access vertex/index data via GLSL `buffer_reference` using device addresses looked up from a `MeshAddressEntry` storage buffer (the buffer address table). This avoids duplicating all geometry into a merged copy (~190 MB savings for a 2.8M triangle scene), preserves the host-owned-buffer contract (the renderer never copies geometry), and supports GPU-generated meshes with zero overhead. The trade-off is one extra storage buffer read per closest-hit invocation — negligible compared to texture sampling and BRDF evaluation. Instance custom index encoding packs `mesh_address_index` (lower 12 bits) + `material_index` (upper 12 bits), supporting up to 4096 unique meshes and 4096 unique materials per scene.

32. **Any-hit shader for alpha masking, not raygen discard.** `AlphaMode::kMask` is handled by a dedicated `anyhit.rahit` shader that samples the base color texture and calls `ignoreIntersectionEXT` when alpha < cutoff. Opaque geometry sets `VK_GEOMETRY_OPAQUE_BIT` in the BLAS build, bypassing the any-hit shader entirely. This is the standard Vulkan RT approach — it avoids wasting closest-hit processing on masked fragments and lets the hardware skip any-hit invocations for fully opaque geometry. Ray flags use `gl_RayFlagsNoneEXT` (not `gl_RayFlagsOpaqueEXT`) to enable any-hit invocations.

33. **Projection-matrix sub-pixel jitter, not per-ray origin jitter.** Camera jitter is applied as a sub-pixel offset in the projection matrix (standard TAA technique) rather than perturbing ray origins in the raygen shader. This is consistent with the mobile hybrid rasterization path (design decision 23), produces identical jitter for both ray-traced and rasterized primary visibility, and simplifies the jitter implementation to a single matrix modification. Uses Halton(2,3) sequence with a 16-frame period, reset on camera movement.

34. **Ray-traced Beer-Lambert attenuation; full volumetric deferred.** Transmission uses Beer-Lambert absorption with ray-traced distances: `exp(-absorption * hit_t)` where `absorption = -log(attenuation_color) / attenuation_distance` and `hit_t` is the actual distance traveled through the volume. This requires manifold (closed) meshes — thin-shell geometry will show no absorption. Thickness factor/texture (the rasterizer `thickness / NdotV` approximation from KHR_materials_volume) is not used; the glTF loader warns when a thickness texture is encountered. Currently absorption is applied only on exit from the volume; per-bounce absorption while traversing the interior is planned (see F4). Full volume rendering (participating media, heterogeneous volumes) is deferred to a future phase.

35. **double_sided stored, rendering deferred.** The `MaterialDesc::double_sided` field is parsed from glTF and stored in the scene layer. It is packed into the GPU material buffer (`alpha_mode_misc` vec4) and available to shaders, but face culling logic (back-face test flipping in closest-hit/any-hit) is not yet implemented. The default Vulkan RT behavior (back-faces are valid intersections but normals may point away from the ray) is acceptable for current scenes. Proper double_sided support will be added when test scenes require it.

36. **TLAS dirty tracking via scene generation counter.** `Scene::tlas_generation_` is a monotonically increasing `uint64_t` counter incremented by `AddNode()`, `RemoveNode()`, and `SetNodeTransform()`. The renderer's `GeometryManager` caches the last-seen value and skips the TLAS rebuild entirely when it matches — an O(1) check that avoids per-node transform diffing on static frames. The alternative (per-node dirty flags) would require O(N) scanning, and diffing transforms directly would require storing shadow copies. The generation counter is trivial to implement and covers all mutation paths through the `Scene` API. If the host mutates `SceneNode` fields directly (bypassing `SetNodeTransform()`), the generation counter will not advance and the TLAS will be stale — this is documented as incorrect usage.

37. **Firefly filter as post-trace median suppression.** Phase 8E adds a firefly filter that runs after the path trace dispatch. It uses a median-based outlier detection on the noisy radiance buffers to suppress extremely bright firefly pixels caused by MIS weight spikes or low-probability light paths. This is cheaper and more robust than clamping individual sample contributions during tracing (which biases the estimator).

38. **Ray cones for texture LOD.** Phase 8F implements ray cone tracking for automatic texture level-of-detail selection. The ray cone spread angle is propagated through bounces and used to compute a footprint at each hit point, which maps to a mip level for texture sampling. This avoids over-sharpening on distant surfaces and provides correct filtering without screen-space derivatives (which are unavailable in ray tracing).

39. **Unified light buffer for quad/sphere/triangle lights.** Phase 8G introduces a `PackedLight` struct (64 bytes, 4 × vec4) with a type discriminator field supporting three light types: `kQuad = 0`, `kSphere = 1`, `kTriangle = 2`. All light types are packed into a single GPU storage buffer. Phase 8K adds hybrid NEE: when `light_count <= kMaxDirectSampleLights` (4), all lights are sampled directly with per-light shadow rays (preserving quality for simple scenes). Above the threshold, weighted reservoir sampling (WRS) selects one light per bounce with probability proportional to its estimated unshadowed contribution, with one-sample MIS between the WRS-NEE and BSDF strategies. This avoids separate light buffers per type and scales from 1 to hundreds of lights.

40. **Diffuse transmission for thin surfaces.** Phase 8H implements `KHR_materials_diffuse_transmission` for materials like leaves, fabric, and paper. When `diffuse_transmission_factor > 0`, light scatters diffusely through the surface. The `thin_surface` flag enables single-intersection in/out behavior (no refraction, just Lambertian transmission through the surface plane).

41. **Nested dielectric priority for overlapping volumes.** Phase 8I implements a priority-based interior list for correctly handling overlapping transparent volumes (e.g., liquid inside glass). Each material has a `nested_priority` value; when a ray enters an overlapping volume, the higher-priority material's IOR is used for refraction. The interior list is tracked per-ray in `interior_list.glsl`.

42. **Emissive mesh extraction via EmissiveLightExtractor.** Phase 8J adds `EmissiveLightExtractor`, which scans scene materials for emissive surfaces (luminance >= `kMinEmissiveLuminance`), reads triangle vertices from the mesh data, transforms them to world space, and adds them as `TriangleLight` entries to the scene. This enables explicit NEE sampling of emissive meshes without requiring ReSTIR. The extraction runs at scene load time, not per-frame. Extracted triangle lights are sampled efficiently via the hybrid WRS algorithm (Phase 8K) — scenes with many emissive triangles automatically use WRS rather than tracing a shadow ray per light.

43. **Per-material UV transform.** Phase 8L implements `KHR_texture_transform` via per-material `uv_offset`, `uv_scale`, and `uv_rotation` fields. The UV transform is applied in the closest-hit shader before texture sampling, using the transform values packed into the `uv_transform` vec4 of `PackedMaterial`.

44. **Sheen BRDF with precomputed albedo LUT.** Phase 8M implements `KHR_materials_sheen` for fabric-like retroreflective highlights. The sheen lobe uses a precomputed albedo lookup texture (256×256 R16F) indexed by `(NdotV, sheen_roughness)` to compute the directional albedo scaling. Sheen is added as a fifth MIS lobe alongside diffuse, specular, clear coat, and environment.

45. **DDS/BC compressed texture support.** Phase 8N adds support for loading DDS texture files with BC1, BC3, BC4, BC5, and BC7 block compression formats. Compressed textures are uploaded directly to the GPU without CPU-side decompression, reducing memory usage and upload bandwidth. The `PixelFormat` enum and `TextureDesc` are extended with BC format variants.

46. **ML denoiser weights via .denimodel binary format.** Phase F11-1 implements a binary weight format (`.denimodel`) for shipping trained ML denoiser weights. The format has a simple header (magic "DENI", version 1) followed by per-layer records (name string, shape dimensions, float32 data). `WeightLoader` parses and validates the format; `MlInference` manages GPU buffer allocation for weights and feature maps. Weight upload to the GPU is deferred to the first `Denoise()` call via a staging buffer.
