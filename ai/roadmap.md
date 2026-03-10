# Monti/Deni — Roadmap

> These features are **not in the initial release** (Vulkan desktop path tracer + passthrough/ML denoiser). They are documented here for planning visibility. See [renderer_design_v5.md](renderer_design_v5.md) for the current-release architecture.

---

## Future Phase Summary

| Phase | Feature | Prerequisite |
|---|---|---|
| F1 | ReLAX denoiser (desktop only) | Passthrough denoiser complete |
| F2 | ReSTIR Direct Illumination (desktop only) | Multi-bounce MIS complete |
| F3 | Emissive mesh rendering (desktop only) | F2 (needs ReSTIR for correct sampling) |
| F5 | DLSS-RR denoiser backend | Passthrough denoiser complete |
| F6 | Mobile Vulkan renderer (`monti_vulkan_mobile`) | Shared GpuScene/GeometryManager ready |
| F7 | Metal renderer (C API) | Desktop design patterns established |
| F8 | WebGPU renderer (C API → WASM) | Desktop design patterns established |
| F9 | ML denoiser training pipeline | Capture writer complete |
| F10 | Shader permutation cache | Multi-bounce MIS complete |
| F11 | ML denoiser deployment (desktop + mobile) | F9 complete (trained weights available) |
| F12 | Super-resolution in ML denoiser | F11 complete; uses `ScaleMode` enum |
| F13 | Fragment shader denoiser (mobile) | F6 + F11 complete |

---

## F1: ReLAX Spatial-Temporal Filter (Desktop Only)

> **Desktop only.** ReLAX (Relaxed A-Trous Spatial-Temporal) will be added as a drop-in upgrade to the passthrough denoiser on desktop Vulkan. The class interface remains the same; additional input fields (`current_view_proj`, `previous_view_proj`) will be added to `DenoiserInput`. The existing `diffuse_albedo` and `specular_albedo` fields are used for demodulated denoising. The algorithm executes 7 compute passes: reproject → prefilter → 4× à-trous (strides 1/2/4/8) → temporal accumulate. Expected cost: ~3–6 ms at 1080p on desktop.
>
> ReLAX is not planned for mobile. Its 7 full-screen compute passes consume ~800+ MB of memory bandwidth at 1080p — exceeding the entire per-frame bandwidth budget on mobile GPUs (~833 MB/frame at 50 GB/s, 60 fps). The ML-trained denoiser (F11) is the planned mobile denoiser: a single-pass inference model that reads inputs once and writes output once, fitting within mobile bandwidth constraints.

---

## F2: ReSTIR DI + Emissive Mesh Lights (Desktop Only)

> **Desktop only.** ReSTIR Direct Illumination will be added to support emissive mesh lights (arbitrary geometry emitters). Quad area lights are already supported in v1 via direct solid-angle sampling + MIS, which is sufficient for a small number of rectangular emitters. When scenes contain many emissive meshes with complex geometry, ReSTIR provides efficient resampled importance sampling. The initial MIS path tracing approach (environment + area light + GGX importance sampling, power heuristic) is sufficient for environment + quad area light scenes.
>
> ReSTIR is not planned for mobile. The temporal and spatial reuse passes add 3 additional full-screen read/write cycles, exceeding the mobile bandwidth budget. On mobile, the environment + area light MIS approach with 1–2 SPP + ML denoising is the target pipeline.

### ReSTIR Pipeline Insertion Point

ReSTIR inserts reservoir-based resampled importance sampling before the path trace bounce loop. It requires the material model (target PDF) and light source knowledge from the scene layer. The pass sequence:

1. Candidate generation (shadow rays, reservoir fill)
2. Temporal reuse (merge with frame N-1 reservoirs)
3. Spatial reuse (neighbor gathering)

Reservoir buffer layout (16 bytes/pixel packed format) will be defined before implementation.

---

## F5: Future Platform Denoisers

| Library | Platform | Implementation |
|---|---|---|
| `deni_vulkan` (ML) | Desktop + Mobile | ML-trained single-pass denoiser (the product denoiser) |
| `deni_vulkan` (ReLAX) | Desktop only | ReLAX spatial-temporal filter (classical, pre-ML baseline) |
| `deni_metal` | iOS / macOS | ML denoiser in Metal compute (C API for Swift) |
| `deni_webgpu` | Web | ML denoiser in WebGPU compute (C API for WASM/JS) |
| `deni_dlss` | NVIDIA | DLSS 3.5 Ray Reconstruction wrapper |

---

## F6: Mobile Vulkan Renderer (`monti_vulkan_mobile`)

Mobile GPUs (Qualcomm Adreno on Snapdragon 8 Gen 3+, ARM Mali Immortalis-G720+) support `VK_KHR_ray_query` and `VK_KHR_acceleration_structure` but **do not** support `VK_KHR_ray_tracing_pipeline`. There are no dedicated raygen/closesthit/miss shader stages on mobile — ray tracing is done via `rayQueryEXT` inside compute (or fragment) shaders.

**Approach:** The mobile renderer uses a **hybrid rasterization + ray query** pipeline to exploit TBDR tile memory for primary visibility while using ray tracing only for indirect effects:

1. **Rasterization G-buffer pass** (render pass, TBDR-friendly) — Rasterize the scene to produce primary visibility: depth, world normals, material ID, albedo, motion vectors. This goes through the tile pipeline at full TBDR efficiency. Multiple outputs stay in tile memory via subpass attachments or MRT (multi-render-target); only the final resolved images are written to DRAM.

2. **Ray trace compute pass** (compute, `rayQueryEXT`) — Reads the G-buffer, traces indirect bounces (diffuse GI, specular reflections, shadow rays) using `rayQueryEXT`. Writes noisy diffuse and noisy specular to storage images.

3. **Denoise pass** — ML denoiser (or passthrough) processes the noisy outputs.

This hybrid approach cuts ray tracing workload by ~40–60% (primary rays are cheapest to rasterize but most expensive to trace through the BVH) and gets full TBDR benefits for the G-buffer pass.

```cpp
// renderer/include/monti/vulkan/MobileRenderer.h
#pragma once
#include <monti/scene/Scene.h>
#include <vulkan/vulkan.h>
#include <memory>

namespace monti::vulkan::mobile {

struct RendererDesc {
    VkDevice         device;
    VkPhysicalDevice physical_device;
    VkQueue          queue;
    uint32_t         queue_family_index;
    VkPipelineCache  pipeline_cache = VK_NULL_HANDLE;  // Critical on mobile
    VmaAllocator     allocator      = VK_NULL_HANDLE;
    uint32_t         width             = 1920;
    uint32_t         height            = 1080;
    uint32_t         samples_per_pixel = 1;  // Lower default for mobile
};

class Renderer {
public:
    static std::unique_ptr<Renderer> Create(const RendererDesc& desc);
    ~Renderer();

    void SetScene(monti::Scene* scene);
    GpuScene& GetGpuScene();
    void NotifyMeshDeformed(MeshId mesh, bool topology_changed = false);
    bool RenderFrame(VkCommandBuffer cmd, const GBuffer& output,
                     uint32_t frame_index);
    void SetSamplesPerPixel(uint32_t spp);
    uint32_t GetSamplesPerPixel() const;
    void Resize(uint32_t width, uint32_t height);
    float LastFrameTimeMs() const;
};

} // namespace monti::vulkan::mobile
```

**Shared code:** `GpuScene`, `GeometryManager`, `EnvironmentMap`, `BlueNoise`, and all GLSL include files (`brdf.glsl`, `sampling.glsl`, `mis.glsl`, etc.) are identical between desktop and mobile renderers. The GLSL material evaluation and BRDF code is shared between the desktop closesthit shader and the mobile ray trace compute shader.

**Key differences from desktop:**

| Aspect | Desktop (`monti_vulkan`) | Mobile (`monti_vulkan_mobile`) |
|--------|--------------------------|--------------------------------|
| Extension | `VK_KHR_ray_tracing_pipeline` | `VK_KHR_ray_query` + `VK_KHR_acceleration_structure` |
| Primary visibility | Ray traced (raygen shader) | **Rasterized** (render pass, TBDR tile memory) |
| Indirect bounces | `traceRayEXT` (SBT dispatch) | `rayQueryEXT` (compute) |
| Shader structure | raygen + closesthit + miss + anyhit | Rasterization vertex/fragment + uber-compute for indirect |
| Material branching | SBT per-geometry hit shader | Dynamic `switch` on material type |
| G-buffer formats | Compact formats (see GBuffer struct) | Same compact formats |
| Default SPP | 4 | 1 |
| Max bounces | 4 + 8 transparency | 2 + 4 transparency (tunable) |
| Denoiser | Passthrough → ReLAX → ML | Passthrough → ML (no ReLAX) |

### G-Buffer Formats

G-buffer images are **format-agnostic at the API level** — the renderer and denoiser accept `VkImageView` and use `shaderStorageImageReadWithoutFormat` / `shaderStorageImageWriteWithoutFormat` to read and write images in whatever format the host allocated. This means the host is free to use either compact or RGBA16F formats for any channel. No shader permutations, no API changes, no format negotiation.

The **recommended compact formats** provide 32% bandwidth savings and are the default in our app:

| Image | Recommended Format | Bytes/pixel | RGBA16F Alternative | Notes |
|---|---|---|---|---|
| `noisy_diffuse` | `VK_FORMAT_R16G16B16A16_SFLOAT` | 8 | (same) | HDR radiance, needs full FP16 range |
| `noisy_specular` | `VK_FORMAT_R16G16B16A16_SFLOAT` | 8 | (same) | HDR radiance, needs full FP16 range |
| `motion_vectors` | `VK_FORMAT_R16G16_SFLOAT` | 4 | RGBA16F (8 B) | Only .xy used |
| `linear_depth` | `VK_FORMAT_R16_SFLOAT` | 2 | RGBA16F (8 B) | Single depth value |
| `world_normals` | `VK_FORMAT_R16G16B16A16_SFLOAT` | 8 | (same) | .xyz normal + .w roughness |
| `diffuse_albedo` | `VK_FORMAT_B10G11R11_UFLOAT_PACK32` | 4 | RGBA16F (8 B) | Reflectance; no alpha, LDR-range |
| `specular_albedo` | `VK_FORMAT_B10G11R11_UFLOAT_PACK32` | 4 | RGBA16F (8 B) | F0 reflectance; no alpha, LDR-range |
| **Total (compact)** | | **38** | **56 (RGBA16F)** | **32% bandwidth savings with compact** |

Both `shaderStorageImageReadWithoutFormat` and `shaderStorageImageWriteWithoutFormat` are universally supported on Vulkan 1.2+ desktop GPUs and on all mobile GPUs that support ray query (Adreno 740+, Mali-G720+). The renderer and denoiser require these features at device creation time.

### TBDR and Mobile Pipeline

Mobile GPUs use Tile-Based Deferred Rendering (TBDR), where render pass attachments stay in fast on-chip tile memory and only the final resolved output is written to DRAM. The compute-only ray tracing passes cannot use tile memory (storage images always go to DRAM). The hybrid approach maximizes TBDR benefits:

- **Rasterization G-buffer pass:** Depth, normals, material ID, albedo, and motion vectors are produced as render pass color/depth attachments. On TBDR hardware, these stay in tile memory during the pass and are resolved to DRAM only once at the end. Camera jitter is applied as a sub-pixel offset to the projection matrix (identical to TAA jitter in rasterization engines), providing temporal anti-aliasing accumulation across frames. At 1 SPP on mobile, per-frame projection jitter is equivalent to per-pixel ray jitter — both see exactly one sample per pixel per frame.
- **Ray trace compute pass:** Reads the resolved G-buffer from DRAM; traces indirect bounces (diffuse GI, specular reflections, shadow rays) using `rayQueryEXT`. Writes noisy radiance to storage images. This pass has no tile memory benefit.
- **Fragment shader denoiser:** The ML denoiser implementation uses a fullscreen fragment shader instead of compute on mobile. This reads G-buffer inputs as texture samples (through the texture cache) and writes denoised output as a render pass color attachment — keeping the output in tile memory for a subsequent tone-map subpass. The chain (denoise → tone map → present) executes as subpasses within a single render pass, with intermediate results never touching DRAM. This is invisible to the caller — `Denoise(cmd, input)` returns `DenoiserOutput` regardless of whether the implementation uses compute or fragment shaders internally. On desktop, the compute path is used (no TBDR benefit from fragment). On mobile, the fragment path is preferred. The `Denoiser` selects the implementation automatically based on device properties.

### Mobile Shader Best Practices

- **FP16 throughout.** Mobile denoiser and renderer shaders should use `float16_t` (via `GL_EXT_shader_explicit_arithmetic_types_float16`) or `mediump` for ALU operations. Mobile GPUs typically have 2× FP16 throughput vs FP32. Only subpixel motion reprojection needs FP32 precision.
- **Workgroup size.** Use 8×8 workgroups for mobile compute shaders (vs 16×16 on desktop). Mobile GPUs have smaller wavefronts (Adreno: 64, Mali: 16) and benefit from smaller workgroup tiles that fit shader register files.

### Mobile Complications

1. **Uber-shader complexity.** The ray trace compute shader combines all material evaluation, miss handling, and transparency logic. This is larger than the desktop pipeline's per-stage shaders (more register pressure, possible occupancy loss). Shader specialization constants can reduce this per-material if profiling shows it's needed.
2. **No SBT dispatch.** Per-geometry hit shader selection via SBT is unavailable. Material branching is dynamic — the compute shader loads the material index from the hit and branches to the correct BRDF evaluation path.
3. **Performance budget.** Mobile GPUs lack dedicated RT cores (or have minimal acceleration). Expect 1–2 SPP at interactive rates vs 4+ on desktop. The denoiser becomes even more critical on mobile.
4. **VkPipelineCache importance.** Pipeline compilation on mobile drivers is significantly slower than desktop (50–500ms per pipeline). On mobile, `pipeline_cache` should be treated as effectively required. The denoiser and renderer should log a warning if `pipeline_cache` is `VK_NULL_HANDLE` on a mobile device.
5. **Acceleration structure build cost on mobile.** BLAS building on mobile GPUs (Adreno, Mali) is significantly slower than on desktop NVIDIA/AMD hardware with dedicated RT cores. On desktop, building 50 BLAS for topology-changed meshes in a single frame is practical; on mobile, the same workload can consume the entire frame budget (~16ms). Mitigation strategies:
   - **Amortize across frames.** When many meshes change topology simultaneously (e.g., LOD transitions, batch spawning), spread BLAS rebuilds across multiple frames. `RenderFrame()` can prioritize: build N highest-priority BLAS per frame, reuse stale BLAS for the rest. Priority can be based on screen-space size, distance to camera, or frame count since last rebuild.
   - **Prefer refit over rebuild.** For continuous deformation (animation, cloth), BLAS refit is 2–5× faster than full rebuild on mobile. Accept slightly looser BVH quality in exchange for meeting frame budget.
   - **Async compute overlap.** On mobile GPUs that support async compute queues (some Adreno 700+), BLAS/TLAS builds can run on a compute queue overlapped with the previous frame's fragment work. This hides latency but requires careful synchronization.
   - **Budget parameter (future).** If profiling shows this is a real bottleneck, add an optional `max_blas_builds_per_frame` parameter to the mobile `RendererDesc`. `RenderFrame()` would then limit how many dirty BLAS it processes per call, deferring the rest to subsequent frames. This is not needed for the initial implementation — measure first.
6. **Render resolution.** Mobile should default to a lower internal render resolution (720p or 540p) with the ML denoiser providing super-resolution upscaling to the display resolution via `ScaleMode::kPerformance` (2×) or `ScaleMode::kQuality` (1.5×). Rendering at 540p and upscaling to 1080p cuts ray tracing cost by 4× and G-buffer memory by 4×.

---

## F7/F8: Future Renderers

| Renderer | Platform | Strategy |
|---|---|---|
| `monti_metal` | iOS / macOS | Metal RT + `MTLAccelerationStructure` (C API) |
| `monti_webgpu` | Web | Rasterize depth → Hi-Z → screen-space ray march (C API → WASM) |
