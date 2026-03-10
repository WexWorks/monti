# Monti — Cross-Platform Path Tracing Renderer & Deni Denoiser — Architecture Design (v5)

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
| Identical interfaces | All platform implementations follow the same interface shape, differing only in native GPU types (see §4.5, §6.4) |

### Non-Goals (initial release)
- Custom material shaders (glTF PBR only)
- Multi-view / stereo cameras
- Streaming G-buffer capture (synchronous file output is sufficient)
- Vulkan mobile renderer (designed for, implemented later; see §6.5)
- Metal renderer (designed for, implemented later)
- WebGPU renderer (designed for, implemented later; requires WebGPU ray tracing API)
- Emissive mesh light auto-registration (designed for, implemented later; see §5.3 for material fields)
- Video capture output (OpenEXR sequences are sufficient initially)
- ReLAX denoiser (desktop only; designed for, implemented later; see §4.4)
- ReSTIR DI (desktop only; designed for, implemented later; see §6.7, §11.2)

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
│   ├── Denoiser.h                  # Standalone Vulkan denoiser
│   └── Types.h                     # Input/output structs (Vulkan types)
└── src/
    ├── vulkan/
    │   ├── Denoiser.cpp             # Passthrough (initial), ReLAX (future)
    │   └── shaders/                # GLSL compute → SPIR-V
    ├── dlss/                       # (future) DLSS-RR wrapper
    └── metalfx/                    # (future) MetalFX wrapper

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
│   └── Renderer.h
└── src/
    ├── vulkan/
    │   ├── Renderer.cpp
    │   ├── GpuScene.cpp            # Material packing, texture upload
    │   ├── GeometryManager.cpp     # BLAS/TLAS construction
    │   ├── EnvironmentMap.cpp      # HDR loading (EXR via tinyexr), CDF computation, GPU upload
    │   └── shaders/                # raygen, closesthit, miss → SPIR-V + source
    ├── metal/                      # (future)
    └── webgpu/                     # (future)

capture/
├── include/monti/capture/
│   └── Writer.h
└── src/
    └── Writer.cpp

# ── HOST APPLICATION ───────────────────────────────────────────────

app/
├── main.cpp                        # Demo / training data host
├── VulkanContext.cpp               # Device, swapchain, frame loop
├── ToneMapper.cpp                  # ACES filmic + sRGB compute shader
└── Presenter.cpp                   # Swapchain blit

# ── TESTS ──────────────────────────────────────────────────────────

tests/
├── main_test.cpp                   # Test runner entry point
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

> **GLM in Deni headers:** The passthrough denoiser's public header has no GLM dependency — all types are Vulkan-native or scalar. When the ReLAX denoiser (§4.4) adds view/projection matrix inputs, those fields will use raw `float[16]` arrays rather than `glm::mat4`, keeping the public API free of GLM. This also prepares for the future pure-C API (§4.5) where GLM is unavailable.

### 4.1 Vulkan Denoiser Interface

The initial implementation is a **passthrough denoiser** that sums noisy diffuse and specular contributions — identical to the rtx-chessboard `PassthroughDenoiser`. This provides the correct pipeline plumbing and image layout contracts. The ReLAX spatial-temporal filter is a future addition (§4.4).

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
//   linear_depth    — View-space linear Z distance from the camera near plane.
//                     Positive, increasing with distance. Range: [near, far].
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
//   ReLAX (future): reads all fields.
//   ML denoiser (future): reads all fields; uses scale_mode for super-resolution.
//
// Initial release: only ScaleMode::kNative is supported. Create() will
// succeed with any ScaleMode, but Denoise() returns an error if
// scale_mode != kNative until the ML denoiser is implemented.

struct DenoiserInput {
    VkImageView noisy_diffuse;    // RGBA16F — diffuse radiance (1-N spp)
    VkImageView noisy_specular;   // RGBA16F — specular radiance (1-N spp)
    VkImageView motion_vectors;   // RG16F   — screen-space motion (pixels); RGBA16F also supported
    VkImageView linear_depth;     // R16F    — view-space linear Z; RGBA16F also supported
    VkImageView world_normals;    // RGBA16F — world normals (.xyz), roughness (.w)
    VkImageView diffuse_albedo;   // R11G11B10F — diffuse reflectance; RGBA16F also supported
    VkImageView specular_albedo;  // R11G11B10F — specular F0; RGBA16F also supported
                                  // Passthrough ignores both albedo fields.
                                  // ReLAX uses both for demodulated denoising.
                                  // ML denoiser uses both for demodulated denoising.

    uint32_t  render_width;
    uint32_t  render_height;
    ScaleMode scale_mode = ScaleMode::kNative;  // See §4.9

    bool reset_accumulation;      // True on camera cut or scene reset
};

// ── Output ─────────────────────────────────────────────────────────────────

struct DenoiserOutput {
    VkImageView denoised_color;   // RGBA16F — denoised radiance
                                  // Layout: VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
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

    // Optional. If non-null, the denoiser loads all Vulkan device functions
    // through this pointer instead of the statically-linked vkGetDeviceProcAddr.
    // This supports hosts using Volk, custom loaders, or any other dispatch
    // mechanism. When null, the denoiser uses the linked Vulkan loader directly.
    PFN_vkGetDeviceProcAddr get_device_proc_addr = nullptr;
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
};

} // namespace deni::vulkan
```

### 4.2 Customer Integration Example

```cpp
#include <deni/vulkan/Denoiser.h>

// Customer creates denoiser using their existing Vulkan device and VMA allocator.
// Hosts using Volk pass vkGetDeviceProcAddr; others omit it (uses linked loader).
auto denoiser = deni::vulkan::Denoiser::Create({
    .device               = my_device,
    .physical_device      = my_physical_device,
    .width                = 1920,
    .height               = 1080,
    .allocator            = my_vma_allocator,
    .get_device_proc_addr = vkGetDeviceProcAddr,  // or nullptr
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

The initial `Denoiser` is a passthrough that sums diffuse + specular in a compute shader (16×16 workgroup), matching rtx-chessboard's `PassthroughDenoiser`. This validates the full pipeline: image layout contracts, descriptor binding, command buffer recording, and host integration.

### 4.4 Future: ReLAX Spatial-Temporal Filter (Desktop Only)

> Deferred — see [roadmap.md](roadmap.md#f1-relax-spatial-temporal-filter-desktop-only) for full design. ReLAX is a 7-pass classical denoiser that will be added as a drop-in upgrade on desktop. Not planned for mobile (bandwidth constraints).

### 4.5 Platform Interface Parity

> **Design intent:** All platform denoiser implementations (`deni::vulkan::Denoiser`, `deni::metal::Denoiser`, `deni::webgpu::Denoiser`) follow an **identical interface shape**. The struct and method names, parameter semantics, image layout contracts, and lifecycle are the same across platforms — only the native GPU types differ (`VkImageView` vs `MTLTexture*` vs `WGPUTextureView`).
>
> There is no abstract base class because the type signatures necessarily differ per platform. Instead, parity is enforced by convention and documentation. Each platform's `Denoiser` class has the same methods: `Create()`, `Denoise()`, `Resize()`, `LastPassTimeMs()`. Each platform's `DenoiserInput` struct has the same semantic fields with platform-native types.
>
> The Vulkan implementation uses a C++ interface. Future Metal and WebGPU implementations will expose a **pure-C API** (`deni_denoiser_create()`, `deni_denoiser_denoise()`, etc.) for interop with Swift and TypeScript/JavaScript (via emscripten WASM). The C API is a thin wrapper over the same internal implementation.

### 4.6 Shader Distribution

Compiled SPIR-V shaders are **embedded in the library source** as `constexpr uint32_t[]` arrays, generated at build time by a CMake custom command. The GLSL source files are also shipped alongside the library for inspection. This ensures "link one library" works without external shader files.

### 4.7 Thread Safety

A single `Denoiser` instance is not thread-safe. `Create()`, `Denoise()`, and `Resize()` must not be called concurrently on the same instance. Multiple `Denoiser` instances on the same `VkDevice` are safe if they record into different command buffers.

### 4.8 Future Platform Denoisers

> See [roadmap.md](roadmap.md#f5-future-platform-denoisers) for the full table of planned denoiser backends (Metal, WebGPU, DLSS-RR, ReLAX).

### 4.9 Producing DenoiserInput

Customers integrating Deni with their own path tracer must produce the `DenoiserInput` fields according to these conventions:

| Field | Definition | Coordinate system |
|---|---|---|
| `noisy_diffuse` | Diffuse radiance accumulated over N samples. Linear HDR, unbounded. | Scene-referred |
| `noisy_specular` | Specular radiance accumulated over N samples. Linear HDR, unbounded. | Scene-referred |
| `motion_vectors` | `current_pixel_pos − previous_pixel_pos` in pixel coordinates. | Vulkan screen: +X right, +Y down. Zero = static. |
| `linear_depth` | View-space Z distance from the camera near plane. | Positive, increasing with distance: `dot(hit − eye, camera_forward)`. |
| `world_normals` | `.xyz`: unit-length surface normal in world space (glTF Y-up, right-handed). `.w`: perceptual roughness ∈ [0, 1]. | World space |
| `diffuse_albedo` | `base_color × (1 − metallic)`. Linear, [0, 1] range. | — |
| `specular_albedo` | Fresnel F0 at normal incidence. Linear, [0, 1] range. For dielectrics: `((ior−1)/(ior+1))²`. For metals: `base_color`. | — |

**Splitting diffuse and specular:** The path tracer classifies each path's contribution at the first opaque bounce. If the first bounce samples the diffuse lobe, the entire path throughput goes to `noisy_diffuse`; if it samples the specular or clear coat lobe, it goes to `noisy_specular`. This split enables demodulated denoising — the denoiser operates on demodulated (albedo-divided) radiance and remodulates after filtering, preserving texture detail.

**Image layouts:** All input images must be in `VK_IMAGE_LAYOUT_GENERAL` before `Denoise()`. The denoiser reads them via storage image descriptors. After `Denoise()`, inputs remain in `VK_IMAGE_LAYOUT_GENERAL`.

All radiance values are pre-exposure (scene-referred linear HDR). The denoiser does not apply exposure or tone mapping.

### 4.10 Super-Resolution via ScaleMode

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

    // ── Camera ─────────────────────────────────────────────────────────
    void SetActiveCamera(const CameraParams& params);
    const CameraParams& GetActiveCamera() const;

private:
    std::vector<Mesh>         meshes_;
    std::vector<MaterialDesc> materials_;
    std::vector<SceneNode>    nodes_;
    std::vector<TextureDesc>  textures_;

    std::optional<EnvironmentLight> environment_light_;
    std::vector<AreaLight> area_lights_;
    CameraParams active_camera_;

    uint64_t next_mesh_id_     = 0;
    uint64_t next_material_id_ = 0;
    uint64_t next_texture_id_  = 0;
    uint64_t next_node_id_     = 0;
};

} // namespace monti
```

### 5.3 Material (`scene/Material.h`)

CPU-side PBR material data. For each channel, the texture is optional — if the ID is invalid, the constant factor is used alone. If both are provided, the texture sample is multiplied by the factor per glTF 2.0.

> **Emissive fields:** The `MaterialDesc` includes emissive factor, texture, and strength fields per glTF 2.0. These fields are **parsed and stored** by the glTF loader but **not used by the renderer** in the initial implementation. Emissive light source support will be added alongside ReSTIR in a future phase. Until then, emissive surfaces render as opaque with their base color.

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
// host as GPU buffers and registered with the renderer's GpuScene via
// device addresses. This avoids CPU roundtrips and supports GPU-generated
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
    float clear_coat_roughness = 0.1f;

    // Alpha (implemented)
    enum class AlphaMode { kOpaque, kMask, kBlend };
    AlphaMode alpha_mode       = AlphaMode::kOpaque;
    float     alpha_cutoff     = 0.5f;
    bool      double_sided     = false;

    // Emissive (parsed and stored; rendering DEFERRED — requires ReSTIR for proper sampling)
    glm::vec3 emissive_factor    = {0, 0, 0};
    std::optional<TextureId> emissive_map;
    float     emissive_strength  = 1.0f;

    // Transmission/volume (implemented — Fresnel refraction, volume attenuation)
    float     transmission_factor  = 0.0f;
    std::optional<TextureId> transmission_map;
    glm::vec3 attenuation_color    = {1, 1, 1};
    float     attenuation_distance = 0.0f;
    float     thickness_factor     = 0.0f;

    // Sheen — DEFERRED. Will be added when a concrete rendering plan exists.
    // Removed from v1 to avoid dead API surface. See KHR_materials_sheen.
};

} // namespace monti
```

### 5.4 Lights (`scene/Light.h`)

Two light types are implemented: `EnvironmentLight` (HDR equirectangular map) and `AreaLight` (emissive quad). Together these cover all practical lighting scenarios for a physically-based path tracer. Point, spot, and directional lights are intentionally omitted — they are mathematical idealizations (zero-area emitters) that don't exist physically. A small area light produces the same visual result with correct soft shadows and penumbrae; a sun disk in the environment map handles directional illumination. Emissive mesh lights (arbitrary geometry) will be added alongside ReSTIR DI when needed.

> **Why quad area lights?** A quad emitter requires minimal path tracer changes: sample a point on the quad, compute the solid angle PDF, trace a shadow ray, and MIS-weight against the BRDF sample. This is a direct extension of the existing environment MIS logic. By contrast, emissive arbitrary-mesh lights require per-triangle CDF construction, mesh-area-weighted sampling, and ideally ReSTIR to converge with many emitters — significantly more complex. The quad area light enables the Cornell box ceiling light, window rectangles, and basic interior scenes without that complexity.

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
struct AreaLight {
    glm::vec3 corner   = {0, 0, 0};   // World-space corner position
    glm::vec3 edge_a   = {1, 0, 0};   // First edge from corner
    glm::vec3 edge_b   = {0, 0, 1};   // Second edge from corner
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
    float aspect_ratio         = 16.0f / 9.0f;
    float near_plane           = 0.1f;
    float far_plane            = 1000.0f;
    float aperture_radius      = 0.0f;        // 0 = pinhole
    float focus_distance       = 10.0f;
    float exposure_ev100       = 0.0f;
};

} // namespace monti
```

### 5.6 glTF Loader

The loader populates the scene with mesh metadata, materials, textures, and nodes. Vertex and index data is returned as transient `MeshData` in the `LoadResult` — the host uploads this to GPU buffers and registers device addresses with the renderer's `GpuScene`. This separation keeps the scene layer GPU-agnostic while supporting GPU-side-only geometry. Camera extraction is not performed; cameras are always set by the host. Skin, animation, and morph target data are silently ignored.

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

The renderer reads the platform-agnostic scene data and manages all GPU resources. It takes native Vulkan types for the command buffer and outputs to host-provided G-buffer images. Materials and textures are packed from the scene's CPU-side data into GPU buffers. Geometry (vertex/index buffers) is host-owned — the renderer references device addresses registered by the host. This follows the rtx-chessboard `GPUScene` pattern, renamed to `monti::vulkan::GpuScene`.

### 6.1 GPU Scene (Vulkan-specific)

Materials packed into a storage buffer, textures in a bindless array. Geometry buffers are **not owned** by GpuScene — the host provides `VkBuffer` handles and device addresses via `RegisterMeshBuffers()`. This avoids CPU↔GPU roundtrips and supports GPU-generated geometry (cloth, procedural deformation).

```cpp
// renderer/src/vulkan/GpuScene.h
#pragma once
#include <monti/scene/Types.h>
#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>
#include <unordered_map>

namespace monti::vulkan {

struct alignas(16) PackedMaterial {
    glm::vec4 base_color_roughness;   // .rgb = base_color, .a = roughness
    glm::vec4 metallic_clearcoat;     // .r = metallic, .g = clear_coat,
                                      // .b = clear_coat_roughness,
                                      // .a = base_color_map index (UINT32_MAX = none)
    glm::vec4 opacity_ior;            // .r = opacity, .g = ior
    glm::vec4 transmission_volume;    // .r = transmission_factor, .g = thickness,
                                      // .b = attenuation_distance, .a = (reserved)
    glm::vec4 attenuation_color_pad;  // .rgb = attenuation_color, .a = (reserved)
};

struct MeshBufferBinding {
    VkBuffer         vertex_buffer;
    VkDeviceAddress  vertex_address;
    VkBuffer         index_buffer;
    VkDeviceAddress  index_address;
    uint32_t         vertex_count;
    uint32_t         index_count;
    uint32_t         vertex_stride = sizeof(monti::Vertex);  // Default matches standard Vertex layout
};

class GpuScene {
public:
    // Register host-owned GPU buffers for a mesh. Called after host uploads
    // vertex/index data. Device addresses are used for BLAS/TLAS building
    // and shader-side buffer_reference access.
    void RegisterMeshBuffers(MeshId mesh, const MeshBufferBinding& binding);

    // Pack CPU-side materials from Scene into GPU storage buffer,
    // upload textures into bindless image array.
    bool UpdateMaterials(const class monti::Scene& scene);
    bool UploadTextures(const class monti::Scene& scene,
                        VkCommandBuffer cmd);

    // Accessors for BLAS/TLAS building and descriptor binding
    const MeshBufferBinding* GetMeshBinding(MeshId id) const;
    VkBuffer MaterialBuffer() const;
    uint32_t GetMaterialIndex(MaterialId id) const;
    uint32_t TextureCount() const;

private:
    std::unordered_map<MeshId, MeshBufferBinding> mesh_bindings_;
    // Material storage buffer (VMA-allocated)
    // Bindless texture images
};

} // namespace monti::vulkan
```

### 6.1.1 GPU Buffer Upload Helpers (Optional)

Convenience functions for hosts that do not already manage GPU geometry buffers. These upload `MeshData` (returned by the glTF loader) to device-local VMA buffers with staging copies and return a `GpuBuffer` ready for `RegisterMeshBuffers()`. Platform-specific — each backend (Vulkan, Metal, WebGPU) provides its own equivalent.

Hosts that already maintain GPU buffers (game engines, procedural generators) skip these entirely and call `RegisterMeshBuffers()` directly with their own buffer handles.

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
/// Buffers are created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT and
/// VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
/// making them immediately usable for BLAS building and shader access.
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

/// Convenience: upload all meshes from a loader result and register them
/// with the GpuScene in a single call. Returns the GpuBuffers the host
/// must keep alive for the renderer's lifetime. Equivalent to calling
/// UploadMeshToGpu + RegisterMeshBuffers per mesh, but less boilerplate.
std::vector<GpuBuffer> UploadAndRegisterMeshes(
    GpuScene& gpu_scene, VmaAllocator allocator, VkDevice device,
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
    VkImageView linear_depth;     // R16F        — view-space linear Z; RGBA16F also supported
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

    // Optional. Same semantics as DenoiserDesc::get_device_proc_addr.
    PFN_vkGetDeviceProcAddr get_device_proc_addr = nullptr;
};

class Renderer {
public:
    static std::unique_ptr<Renderer> Create(
        const RendererDesc& desc);
    ~Renderer();

    // ── Scene ─────────────────────────────────────────────────────────
    void SetScene(monti::Scene* scene);

    // Host accesses GpuScene to register geometry buffers.
    GpuScene& GetGpuScene();

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
    // Resize(), or GpuScene mutators concurrently on the same instance.
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

    // ── Resize ────────────────────────────────────────────────────────
    void Resize(uint32_t width, uint32_t height);

    float LastFrameTimeMs() const;
};

} // namespace monti::vulkan
```

### 6.4 Platform Renderer Parity

> **Design intent:** Like the denoiser (§4.5), all platform renderer implementations (`monti::vulkan::Renderer`, `monti::metal::Renderer`, `monti::webgpu::Renderer`) follow an **identical interface shape** despite using different native GPU types. There is no abstract base class because type signatures differ per platform (e.g., `VkCommandBuffer` vs `id<MTLCommandBuffer>` vs `WGPUCommandEncoder`).
>
> Parity is enforced by convention: each renderer has `Create()`, `SetScene()`, `RenderFrame()`, `Resize()`, `SetSamplesPerPixel()`. Each has a platform-specific `GBuffer` struct with the same semantic fields but native types.
>
> The Vulkan renderer uses C++. Future Metal and WebGPU renderers expose a **pure-C API** for Swift and WASM interop.

### 6.5 Mobile Vulkan Renderer (`monti_vulkan_mobile`)

> Deferred — see [roadmap.md](roadmap.md#f6-mobile-vulkan-renderer-monti_vulkan_mobile) for full design. Hybrid rasterization + ray query pipeline exploiting TBDR tile memory. Shares `GpuScene`, `GeometryManager`, and GLSL include files with the desktop renderer.

### 6.6 Future Renderers

> See [roadmap.md](roadmap.md#f7f8-future-renderers) — Metal RT (C API) and WebGPU screen-space ray march (C API → WASM).

### 6.7 Future: ReSTIR DI (Desktop Only)

> Deferred — see [roadmap.md](roadmap.md#f2f4-restir-di--local-light-sources-desktop-only). ReSTIR inserts reservoir-based resampled importance sampling when local light sources are added. Not planned for mobile (bandwidth constraints).

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

CPU-side. Reads pixel data from host-provided buffers and writes OpenEXR.

```cpp
// capture/include/monti/capture/Writer.h
#pragma once
#include <string>
#include <memory>
#include <cstdint>

namespace monti::capture {

enum class CaptureFormat {
    kFloat16,  // FP16 EXR channels (smaller files, sufficient for training)
    kFloat32,  // FP32 EXR channels (maximum precision for reference data)
};

struct WriterDesc {
    std::string   output_dir = "./capture/";
    uint32_t      width;
    uint32_t      height;
    CaptureFormat format = CaptureFormat::kFloat32;  // EXR output precision
};

// Fixed-field capture frame. All pointers are to CPU-side float arrays.
// The host reads back GPU images into float arrays (converting from the
// GPU texture format as needed). Null pointers are omitted from the
// output EXR. The Writer converts to the precision specified by
// WriterDesc::format (kFloat16 or kFloat32) when writing.
struct CaptureFrame {
    const float* noisy_diffuse      = nullptr;  // 4 floats/pixel (RGBA)
    const float* noisy_specular     = nullptr;  // 4 floats/pixel (RGBA)
    const float* ref_diffuse        = nullptr;  // 4 floats/pixel (RGBA) — high-spp reference
    const float* ref_specular       = nullptr;  // 4 floats/pixel (RGBA) — high-spp reference
    const float* diffuse_albedo     = nullptr;  // 3 floats/pixel (RGB)
    const float* specular_albedo    = nullptr;  // 3 floats/pixel (RGB)
    const float* world_normals      = nullptr;  // 4 floats/pixel (XYZW; .w = roughness)
    const float* linear_depth       = nullptr;  // 1 float/pixel
    const float* motion_vectors     = nullptr;  // 2 floats/pixel (XY)
};

class Writer {
public:
    static std::unique_ptr<Writer> Create(
        const WriterDesc& desc);
    ~Writer();

    bool WriteFrame(const CaptureFrame& frame, uint32_t frame_index);
};

} // namespace monti::capture
```

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
        renderer->GetGpuScene(), vk.allocator, vk.device, upload_cmd,
        result.mesh_data);
    vk.SubmitAndWait(upload_cmd);
    // MeshData CPU vectors can now be discarded (result goes out of scope).

    // ── Denoiser (Deni product — we are our own customer) ──────────────
    auto denoiser = deni::vulkan::Denoiser::Create({
        .device          = vk.device,
        .physical_device = vk.physical_device,
        .pipeline_cache  = vk.pipeline_cache,
        .allocator       = vk.allocator,
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

```cpp
#include <monti/scene/Scene.h>
#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>
#include <monti/capture/Writer.h>

int main() {
    VulkanContext vk = CreateVulkanContextHeadless();
    GBufferImages gb = CreateGBufferImages(vk, 1920, 1080);
    // High-precision G-buffer for reference render (RGBA32F for radiance,
    // RGBA16F for aux channels — higher precision than the compact interactive G-buffer).
    GBufferImages ref_gb = CreateGBufferImages(vk, 1920, 1080, HighPrecisionFormats());

    monti::Scene scene;
    auto result = monti::gltf::LoadGltf(scene, "bistro.glb");

    auto renderer = monti::vulkan::Renderer::Create({
        .device = vk.device, .physical_device = vk.physical_device,
        .queue = vk.queue, .queue_family_index = vk.queue_family,
        .allocator = vk.allocator,
        .width = 1920, .height = 1080,
    });
    renderer->SetScene(&scene);

    // ── Upload geometry (same pattern as interactive example) ───────────
    VkCommandBuffer upload_cmd = vk.BeginOneShot();
    auto mesh_buffers = monti::vulkan::UploadAndRegisterMeshes(
        renderer->GetGpuScene(), vk.allocator, vk.device, upload_cmd,
        result.mesh_data);
    vk.SubmitAndWait(upload_cmd);

    auto writer = monti::capture::Writer::Create({
        .output_dir = "./training_data/",
        .width = 1920, .height = 1080,
    });

    // Load scenes...

    for (uint32_t frame = 0; frame < total_frames; ++frame) {
        VkCommandBuffer cmd = vk.BeginFrame();

        // Low-spp noisy render (interactive G-buffer, compact formats)
        renderer->SetSamplesPerPixel(4);
        renderer->RenderFrame(cmd, gb.gbuffer, frame);

        // High-spp reference render (high-precision G-buffer)
        // Same RenderFrame call — just a different GBuffer and higher SPP.
        renderer->SetSamplesPerPixel(256);
        renderer->RenderFrame(cmd, ref_gb.gbuffer, frame);

        vk.EndFrame();

        // Read back and write EXR
        // Reference = sum of high-spp diffuse + specular (computed by writer
        // or offline). Training data also includes the split channels.
        writer->WriteFrame({
            .noisy_diffuse      = noisy_diffuse_pixels.data(),
            .noisy_specular     = noisy_specular_pixels.data(),
            .ref_diffuse        = ref_diffuse_pixels.data(),
            .ref_specular       = ref_specular_pixels.data(),
            .diffuse_albedo     = diffuse_albedo_pixels.data(),
            .specular_albedo    = specular_albedo_pixels.data(),
            .world_normals      = normals_pixels.data(),
            .linear_depth       = depth_pixels.data(),
            .motion_vectors     = motion_pixels.data(),
        }, frame);
    }
}
```

### 10.3 Transform Update (TLAS-Only)

The most common dynamic case: objects move, rotate, or scale. Mesh data is unchanged — only the TLAS needs rebuilding. `SetNodeTransform()` saves the current transform as `prev_transform` (for motion vectors) and sets the new one. `RenderFrame()` detects the change and rebuilds the TLAS automatically.

```cpp
// Chess piece moves from e2 to e4
monti::Transform new_xform = scene.GetNode(pawn_node_id)->transform;
new_xform.translation = glm::vec3(4.0f, 0.0f, 3.0f);  // e4 position
scene.SetNodeTransform(pawn_node_id, new_xform);

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
//    with GpuScene::RegisterMeshBuffers().
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
renderer->GetGpuScene().RegisterMeshBuffers(lod2_mesh_id, {
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
renderer->GetGpuScene().RegisterMeshBuffers(terrain_mesh_id, {
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

### 11.1 Initial Implementation (MIS Path Tracing)

Directly adapted from rtx-chessboard's `HWPathTracer`:

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
├─ [Internal] Update material/texture GPU buffers (GpuScene)
│
├─ [GPU] Path Trace — MIS (may loop for high SPP)
│    For each batch of samples (split if SPP exceeds per-dispatch limit):
│      vkCmdTraceRaysKHR:
│        raygen:     per-pixel rays, N spp, blue noise sampling
│        closesthit: barycentric interpolation, material fetch, transmission
│        miss:       environment map sampling
│      Per-path bounce loop (max 4 bounces + 8 transparency):
│        - 4-way MIS: diffuse, specular, clear coat, environment
│        - Cook-Torrance BRDF + GGX microfacet
│        - Fresnel refraction + volume attenuation (transmission)
│        - Russian roulette after bounce 3
│        - Separate diffuse/specular classification
│      Accumulate results into GBuffer image views
│
└─ Done. Host calls denoiser, tone map, present.
```

### 11.2 Future Enhancement: ReSTIR DI (Desktop Only)

> Deferred — see [roadmap.md](roadmap.md#restir-pipeline-insertion-point) for pipeline insertion details and reservoir buffer layout.

---

## 12. Build System

```cmake
cmake_minimum_required(VERSION 3.24)
project(monti LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── PRODUCT (Deni) ───────────────────────────────────────────────────────────
add_library(deni_vulkan
    denoise/src/vulkan/Denoiser.cpp)
target_include_directories(deni_vulkan PUBLIC denoise/include)
target_link_libraries(deni_vulkan PRIVATE Vulkan::Vulkan)
# No GLM dependency in public headers. Internal VMA for suballocation.
# Compiled SPIR-V embedded as constexpr uint32_t[] arrays in source.

# ── INTERNAL TOOLING (Monti) ─────────────────────────────────────────────────
add_library(monti_scene
    scene/src/Scene.cpp
    scene/src/gltf/GltfLoader.cpp)
target_include_directories(monti_scene PUBLIC scene/include)
target_link_libraries(monti_scene PRIVATE glm cgltf)
# No Vulkan dependency — platform-agnostic.

add_library(monti_vulkan
    renderer/src/vulkan/Renderer.cpp
    renderer/src/vulkan/GpuScene.cpp
    renderer/src/vulkan/GeometryManager.cpp)
target_include_directories(monti_vulkan PUBLIC renderer/include)
target_link_libraries(monti_vulkan PUBLIC monti_scene PRIVATE Vulkan::Vulkan glm VulkanMemoryAllocator)
# SPIR-V embedded; GLSL source shipped in shaders/ for reference.

# monti_vulkan_mobile is deferred to roadmap phase F6.
# It will be added when the mobile renderer is implemented.

add_library(monti_capture
    capture/src/Writer.cpp)
target_include_directories(monti_capture PUBLIC capture/include)
target_link_libraries(monti_capture PRIVATE tinyexr)
# No Vulkan dependency — CPU-side only.

# ── Shader Compilation ───────────────────────────────────────────────────────
# Find glslc from Vulkan SDK, compile .rgen/.rchit/.rmiss/.comp → .spv
# following the rtx-chessboard CMakeLists.txt pattern.
# Compiled SPIR-V is then embedded into C++ source as constexpr arrays
# (e.g., via xxd or a CMake custom command generating .h files).

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
- SPIR-V shaders are embedded at compile time — no runtime file I/O for shaders.
- `VkPipelineCache` is strongly recommended on Android (50–500ms pipeline compilation without it). The host should serialize/deserialize the cache between app launches.
- SDL3 is not needed for library-only builds. The app (when built) uses SDL3 for windowing; the libraries have no SDL3 dependency.
- **C API for Android JNI:** The Vulkan libraries currently expose a C++ API, which is usable from NDK C++ code linked via JNI. If demand arises for direct Kotlin/Java interop without a C++ bridge, a pure-C wrapper (`deni_denoiser_create()`, etc.) can be added using the same pattern planned for Metal and WebGPU (see §4.5).

---

## 13. Testing Strategy

### 13.1 Automated Render Validation

All render output tests use **NVIDIA FLIP** (BSD-3) for perceptual image comparison. FLIP produces a per-pixel error map and summary statistics (mean, median, max). This replaces manual visual inspection.

**Two-tier validation:**

| Tier | What it proves | Stored references? | FLIP threshold |
|------|----------------|-------------------|----------------|
| **Self-consistency (convergence)** | Renderer converges — low SPP and high SPP produce the same image up to noise | No — both images rendered at test time | Mean FLIP < configurable threshold |
| **Golden reference (regression)** | Output hasn't changed from a known-good baseline | Yes — per-platform images committed to `tests/references/` | Mean FLIP < 0.05 |

Convergence tests are the primary automated gate. Golden reference tests catch regressions but require updating stored images when intentional changes occur.

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

4. **Temporal upscaling / super-resolution** — The ML denoiser can combine denoising + upscaling in a single pass. The `DenoiserInput` already supports distinct render/output dimensions (see §4.9). On mobile, rendering at 540p and upscaling to 1080p is the expected usage pattern.

5. ~~**Reference render accumulation**~~ — **Resolved.** `RenderFrame()` supports arbitrarily high SPP values. If the requested SPP exceeds the per-dispatch limit (tuned to avoid GPU timeout / TDR), the renderer internally splits into multiple trace dispatches and accumulates the results into the G-buffer. The host always calls `RenderFrame()` once per frame regardless of SPP. See §6.3.

6. **G-buffer lifetime** — Denoiser needs frame N-1 history (when temporal denoising is implemented). Host must not destroy/resize G-buffer between frames without `reset_accumulation = true`.

7. **GPU deformation synchronization** — The host is responsible for pipeline barriers between their compute/transfer writes to vertex buffers and the next `RenderFrame()` call. The renderer cannot insert these barriers because it doesn't know which pipeline stage wrote the vertex data. See §10.4 for a complete code example with `VkMemoryBarrier2` from `VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT` to `VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR`.

8. **Mobile uber-shader register pressure** — The single compute shader for mobile path tracing combines all material evaluation into one entry point. Profile on Adreno/Mali to determine if shader specialization constants or permutations are needed to maintain occupancy.

9. **ML denoiser training** — The ReLAX denoiser (future) is a classical signal processing approach. The product roadmap includes an ML-based denoiser trained on data generated by the capture system. The training pipeline is not covered by this document.

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

11. **Emissive material fields stored but not rendered.** The glTF loader parses emissive attributes per-spec. Rendering emissive surfaces as light sources requires either importance sampling (ReSTIR) or explicit light registration, both deferred.

12. **Separate diffuse/specular output.** Following rtx-chessboard, the path tracer classifies contributions by first opaque bounce into separate images. This enables demodulated denoising (DLSS-RR style) and is required for future ML denoiser training data.

13. **Transmission implemented, not deferred.** Fresnel refraction, IOR, volume attenuation, and thickness are included in the initial path tracer. Transmission is essential for glass, water, and gem materials in test scenes (DragonAttenuation, MosquitoInAmber). Deferring would leave a conspicuous gap in material support.

14. **Tone mapper and presenter in app, not libraries.** These are trivial single-shader operations (ACES filmic compute, swapchain blit). Packaging them as separate libraries with their own `Create()`/`Destroy()` lifecycle adds complexity without value. They live in the host app and can be copied from rtx-chessboard.

15. **Mobile via hybrid rasterization + ray query.** Mobile GPUs support `VK_KHR_ray_query` but not `VK_KHR_ray_tracing_pipeline`. The mobile renderer rasterizes primary visibility in a TBDR-friendly render pass, then uses `rayQueryEXT` in compute shaders for indirect bounces. This exploits tile memory for the G-buffer pass and cuts ray tracing workload ~40–60%. The same scene layer, GpuScene, and GLSL include library are shared with the desktop renderer.

16. **VMA required in desc.** Both `deni_vulkan` and `monti_vulkan` require a host-provided `VmaAllocator` for all internal GPU memory allocations. This avoids hidden internal allocators competing with the host for memory and simplifies the implementation (no conditional allocator creation). VMA is a header-only library and widely adopted — requiring it is not a meaningful burden for customers already using Vulkan.

17. **SPIR-V embedded, GLSL shipped.** Compiled SPIR-V is embedded in library source as `constexpr uint32_t[]` arrays. GLSL source files are shipped alongside for debugging, extension, and audit. Libraries do not depend on runtime shader compilation.

18. **BLAS/TLAS building inside RenderFrame, not exposed to host.** The `GeometryManager` is internal to the renderer. `RenderFrame()` automatically builds BLAS for new meshes, refits BLAS for deformed meshes, rebuilds BLAS for topology changes, cleans up BLAS for removed meshes, and rebuilds the TLAS. The host signals geometry changes via `RegisterMeshBuffers()` (for new/changed buffers) and `NotifyMeshDeformed()` (for vertex updates). This keeps the host API simple and avoids exposing acceleration structure internals. The alternative — host calls `BuildAllBlas()` + `BuildTlas()` explicitly — was considered for engine integrations that need control over barrier placement and build timing, but rejected because (a) the common case (static geometry + transform animation) requires zero host calls beyond `SetNodeTransform()`, and (b) the deformation case only requires a barrier + `NotifyMeshDeformed()` before `RenderFrame()`.

19. **Mesh lifecycle managed via Scene; renderer syncs automatically.** When the host calls `Scene::RemoveNode()` or `Scene::RemoveMesh()`, the renderer detects the removal at the next `RenderFrame()` and cleans up internal BLAS entries. There is no separate `UnregisterMeshBuffers()` call — the renderer's view of the scene is always derived from the Scene's current state. The host is notified of BLAS cleanup via `SetMeshCleanupCallback()` — a callback invoked during `RenderFrame()` for each cleaned-up mesh. The host is responsible for ensuring GPU buffers are not freed while in-flight frames still reference them (wait for fence or defer cleanup by N frames before calling `DestroyGpuBuffer()`).

20. **Optional GPU buffer upload helpers.** `monti::vulkan::UploadMeshToGpu()` and `UploadAndRegisterMeshes()` are convenience helpers for hosts that don't already manage GPU geometry buffers. They live in `GpuBufferUtils.h`, use VMA for allocation, and produce `GpuBuffer` structs ready for `RegisterMeshBuffers()`. `UploadAndRegisterMeshes()` combines upload + registration in one call to reduce boilerplate (see §10.1). Hosts with their own buffer management (game engines) skip these entirely. Each platform backend provides its own equivalent (Metal: `MTLBuffer` helpers, WebGPU: `GPUBuffer` helpers).

21. **Format-agnostic G-buffer access.** Shaders use `shaderStorageImageReadWithoutFormat` / `shaderStorageImageWriteWithoutFormat` to read and write G-buffer images in whatever format the host allocated. The recommended compact formats (RG16F motion, R16F depth, R11G11B10F albedo) yield 32% bandwidth savings (38 vs 56 bytes/pixel) and are the default in the app, but RGBA16F is fully supported for any channel. No shader permutations or format negotiation required — the host simply allocates images in its preferred format.

22. **ReLAX and ReSTIR are desktop-only.** ReLAX’s 7 full-screen compute passes consume ~800+ MB bandwidth at 1080p, exceeding the mobile per-frame budget. ReSTIR adds 3 more full-screen passes. On mobile, the ML-trained denoiser (single-pass inference) is the planned denoiser; environment-only MIS with 1–2 SPP is the planned lighting strategy.

23. **Hybrid rasterization + ray query as default mobile option.** Primary visibility is rasterized in a standard render pass (exploiting TBDR tile memory) by default on mobile. Only indirect bounces, shadows, and reflections use `rayQueryEXT` in compute. This cuts ray tracing workload ~40–60% and is the only way to get TBDR benefits in a path tracing pipeline. Camera jitter is applied as a sub-pixel projection matrix offset (standard TAA technique), providing equivalent temporal AA accumulation to per-ray jitter at 1 SPP. A pure ray-query compute path remains available for the mobile renderer when TBDR is not a factor or maximum single-frame quality is preferred.

24. **Fragment shader denoiser on mobile.** The ML denoiser on mobile runs as a fullscreen fragment shader instead of compute, enabling the denoise → tone map → present chain to execute as render pass subpasses with intermediate results staying in tile memory. The `Denoiser` selects compute (desktop) or fragment (mobile) automatically based on device properties. This is an internal implementation choice, not an API change — `Denoise(cmd, input)` returns `DenoiserOutput` in both cases. The additional setup (~100 lines: render pass, framebuffer, graphics pipeline vs. compute pipeline) is contained within the denoiser implementation.

25. **Super-resolution via discrete ScaleMode presets.** The `ScaleMode` enum in `DenoiserInput` controls combined denoise + upscale at three fixed ratios (1×, 1.5×, 2×). Each ratio maps to a trained ML model variant. Discrete presets avoid the quality loss of arbitrary-ratio upscaling and simplify training. Output dimensions are derived as `floor(render_dim × scale_factor / 2) × 2`. This is particularly valuable on mobile (540p → 1080p at `kPerformance` = 4× fewer rays).

26. **Host-provided Vulkan dispatch.** Both `DenoiserDesc` and `RendererDesc` accept an optional `PFN_vkGetDeviceProcAddr get_device_proc_addr`. When provided, all Vulkan device-level functions are resolved through it at `Create()` time and stored in an internal dispatch table. When null (default), the library calls the statically-linked `vkGetDeviceProcAddr` from the Vulkan loader. This supports hosts using Volk (which #defines away the global function pointers), custom loaders, or layer-wrapped dispatch without requiring the library to link against or depend on any specific loader mechanism.

27. **No separate ReferenceTarget — GBuffer for both interactive and capture.** The renderer has a single `RenderFrame(cmd, GBuffer, frame_index)` entry point. For capture, the host allocates a second GBuffer with higher-precision formats (e.g., RGBA32F for radiance channels) and renders at high SPP. The renderer is format-agnostic (design decision 21), so the same shader code writes to compact interactive images or full-precision capture images. The reference ground truth for training is the sum of the high-spp diffuse and specular channels (computed by the capture writer or offline tooling). This eliminates a separate struct and a second `RenderFrame` overload while giving training more data (split diffuse/specular reference instead of merged).

28. **RenderFrame accumulates internally for high SPP.** `RenderFrame()` supports arbitrarily high SPP values set via `SetSamplesPerPixel()`. If the requested SPP exceeds a per-dispatch limit (tuned to avoid GPU timeout / TDR, e.g., 16–32 SPP per dispatch on desktop), the renderer internally issues multiple `vkCmdTraceRaysKHR` dispatches within the same command buffer, accumulating results into the G-buffer between dispatches. The host always calls `RenderFrame()` once per frame. This keeps the host API simple — `SetSamplesPerPixel(256)` followed by `RenderFrame()` just works for high-SPP reference capture without the host managing accumulation loops.

29. **Mesh cleanup via callback, not polling.** `SetMeshCleanupCallback()` replaces the earlier `TakeCleanedUpMeshIds()` polling pattern. The callback is invoked during `RenderFrame()` for each mesh whose BLAS was destroyed (because no scene nodes reference it anymore). This is push-based — the host cannot forget to poll. The callback receives the `MeshId` so the host can schedule deferred GPU buffer destruction (after in-flight frames complete). The callback runs synchronously within `RenderFrame()`, so the host should not perform blocking GPU operations inside it.
