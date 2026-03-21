# Monti/Deni — Roadmap

> These features are **not in the initial release** (Vulkan desktop path tracer + passthrough/ML denoiser). They are documented here for planning visibility. See [renderer_design_v5.md](renderer_design_v5.md) for the current-release architecture.

---

## Recommended Implementation Order

Based on RTXPT comparison analysis, this ordering maximizes visual quality payoff per unit of incremental effort, respecting phase dependencies.

### Wave 1 — Immediate Quality Wins (High Impact)

| Order | Phase | Status | Integration Depth | Rationale |
|---|---|---|---|---|
| 1 | **8E** — Firefly filter + hit distance | **Done** | Low | ~50 LOC. Post-processing clamp + G-buffer channel widen. No MIS, BRDF, or energy changes. |
| 2 | **8F** — Ray cone texture LOD | **Done** | Low | ~100 LOC. Mechanical `textureLod()` conversion. No MIS, BRDF, or energy changes. |
| 3 | **8H** — Diffuse transmission + thin-surface | **Done** | **High** | Material surface area is small, but extends 4→5-way MIS (all MIS functions, probability floors, CDF selection). Three-way Fresnel/specular/diffuse energy split. NaN edge cases at strategy boundaries. |
| 4 | **8I** — Nested dielectric priority | **Done** | Medium | ~80 LOC core (IOR stack). No MIS strategy changes — only affects Fresnel input IOR. Main risk: enter/exit tracking edge cases (missed exits, double-entry, stack overflow). |

### Wave 2 — Light System Upgrade (Medium-High Effort, High Impact)

| Order | Phase | Status | Integration Depth | Rationale |
|---|---|---|---|---|
| 5 | **8G** — Sphere + triangle lights | Remaining | Medium | Unified PackedLight buffer with 3 solid-angle PDF functions. Light PDFs must be compatible with BRDF-side MIS. Sphere light has degenerate edge cases (shading point on/inside sphere). |
| 6 | **8J** — Emissive mesh extraction | Remaining | Medium | CPU data pipeline — `EmissiveLightExtractor` class. Uses 8G's `sampleTriangleLight()`. No MIS or energy changes. |
| 7 | **8K** — WRS for NEE | Remaining | **High** | Replaces O(N) loop with O(1) reservoir. Requires reformulating NEE light PDF in MIS weight (uniform 1/N → weight-proportional). Reservoir overflow and weight_sum=0 edge cases. Foundational for ReSTIR (F2). |

### Wave 3 — Denoiser & ML Pipeline (High Effort, Transformative Impact)

| Order | Phase | Rationale |
|---|---|---|
| 8 | **F1** — DLSS-RR in `monti_view` | App-level NVIDIA denoiser for interactive development and quality reference. Leverages existing rtx-chessboard DLSS-RR + Volk integration. Transforms noisy 1–4 SPP renders into temporally stable output on NVIDIA hardware. |
| 9 | **F9** — ML denoiser training pipeline | Training data generation via `monti_datagen`. Uses DLSS-RR output as quality ceiling comparison. |
| 10 | **F11** — ML denoiser deployment in Deni | Trained model deployed via ncnn Vulkan inference. Cross-vendor, the product denoiser. |

### Wave 4 — Advanced Lighting (High Effort, Many-Light Scenes)

| Order | Phase | Rationale |
|---|---|---|
| 11 | **F2** — ReSTIR DI | Spatiotemporal reservoir resampling. Major quality improvement for many-light scenes (cities, interiors with many emissive surfaces). Requires 8K as foundation. |
| 12 | **F3** — Emissive mesh ReSTIR | Full temporal/spatial resampling of emissive triangle lights. Builds directly on F2. |

### Standalone — Depth of Field (No Dependencies)

| Order | Phase | Status | Integration Depth | Rationale |
|---|---|---|---|---|
| — | **DoF-1** — Core thin-lens DoF | Remaining | Low | ~50 LOC. Thin-lens ray perturbation in raygen, f-stop/focus UI. Pinhole-equivalent G-buffer for denoiser compatibility. No MIS, BRDF, or energy changes. |
| — | **DoF-2** — Polygonal bokeh | Remaining | Low | ~15 LOC. Regular polygon aperture sampling for shaped bokeh highlights. Deferred until DoF-1 validated. |

See [dof_plan.md](dof_plan.md) for full implementation details, denoiser interaction analysis, and ML training data considerations.

### Standalone — Material Extensions (Scene Compatibility)

| Order | Phase | Status | Integration Depth | Rationale |
|---|---|---|---|---|
| — | **8L** — KHR_texture_transform | Remaining | Low | ~80 LOC. UV offset/scale/rotation applied before texture sampling. No MIS, BRDF, or energy changes. Per-material transform packed into 1 new vec4 + reuse of reserved slot. Needed for ToyCar, SheenChair, Intel Sponza. |
| — | **8M** — KHR_materials_sheen | Remaining | Medium | ~200 LOC. Charlie sheen BSDF lobe, layered atop base BRDF following clearcoat pattern (deterministic evaluation, not a separate MIS strategy). Energy-preserving attenuation of base layer. Needed for ToyCar, SheenChair. |

These phases are independent of each other and of Waves 1–4. They should be implemented before Phase F9-6 (Extended Scenes + Data Augmentation) to enable training with ToyCar and other sheen/tiled-texture models. Both depend only on Phase 8D (PBR textures complete). 8M additionally follows the clearcoat layering pattern from Phase 8B.

### Standalone — DDS Texture Loading (Large Scene Compatibility)

| Order | Phase | Status | Integration Depth | Rationale |
|---|---|---|---|---|
| — | **8N** — DDS texture loading | Remaining | Medium | ~200 LOC. GPU-native BC1/BC3/BC4/BC5/BC7 compressed texture upload via dds-ktx. No shader changes — hardware decompression during `textureLod()`. Pre-generated mipmaps from DDS files. Needed for GPUOpen Cauldron-Media scenes (BistroInterior, AbandonedWarehouse, Brutalism) which use DDS textures exclusively. |

Phase 8N depends only on Phase 8D (PBR textures complete) and is independent of all other phases. It should be implemented before F9-6b to enable training data generation from large architectural scenes with full-frame geometry coverage.

### Wave 5 — Deferred Features (As-Needed)

Remaining phases are lower priority and should be tackled as use cases demand:

| Phase | When to implement |
|---|---|
| **F4** — Volume enhancements | When scenes with fog, smoke, or subsurface scattering are needed |
| **F14** — GPU skinning | When animated character scenes are needed |
| **F15** — ReSTIR GI | When indirect illumination quality at low SPP becomes the bottleneck |
| **F16** — NRD ReLAX in Deni | When cross-vendor denoising is needed (AMD/Intel) before ML denoiser quality is sufficient |

### Key Dependencies

```
Wave 1:  8E ──→ 8F ──→ 8H ──→ 8I     (independent, can be reordered within wave)
Wave 2:  8G ──→ 8J ──→ 8K             (strictly sequential)
Wave 3:  10B ──→ F1 (DLSS-RR + denoiser UI)  (app-level, NVIDIA quality reference)
          11B ──→ F9 (ML training)    (training data generation)
          F9  ──→ F11 (ML in Deni)    (product denoiser deployment)
Wave 4:  8K ──→ F2 ──→ F3             (ReSTIR builds on WRS)
MatExt:  8D ──→ 8L (KHR_texture_transform)   (independent of all waves)
         8D ──→ 8M (KHR_materials_sheen)      (independent of all waves)
         8D ──→ 8N (DDS texture loading)       (independent of all waves)
         8L + 8M ──→ F9-6b (training scenes that use these extensions)
         8N ──→ F9-6b (Cauldron-Media scenes require DDS support)
F9-6:    F9-6a ──→ F9-6b ──→ F9-6c ──→ F9-6d (strictly sequential)
         F9-6a: C++ multi-viewpoint rendering (no renderer feature dependencies)
         F9-6b: scene downloads + viewpoint generation (needs 8L/8M for ToyCar/SheenChair, 8N for Cauldron-Media)
         F9-6c: PyTorch augmentation transforms (no renderer dependencies)
         F9-6d: full dataset generation + validation (needs F9-6a/b/c complete)
DoF:     DoF-1 ──→ DoF-2              (independent of all waves)
         DoF-1 ──→ F9-4/F9-6d         (training data should include DoF scenes)
```

Waves 1 and 2 can be interleaved since they are independent. Phases 8E, 8F, 8H, and 8I are complete. Next: Wave 2 (8G, light system). Material extensions (8L, 8M) can be implemented at any time after 8D and should be completed before F9-6b to enable training with ToyCar, SheenChair, and Intel Sponza. Wave 3 proceeds in parallel with Waves 1–2: DLSS-RR (F1) provides interactive denoised viewing during development and serves as the quality ceiling for ML denoiser training. NRD ReLAX (F16) is deferred until cross-vendor denoising is needed; ReBLUR is not planned. Phases that add MIS strategies or modify MIS weight formulas (8H, 8K) have proven to be high-complexity regardless of feature surface area — the MIS probability distribution is a cross-cutting invariant.

**Recommended next-session order for ML denoiser training data:** 8L → 8M → 8N → F9-6a → F9-6b → F9-6c → F9-6d. Phase 8L (~80 LOC, one session) is a prerequisite for rendering Intel Sponza, ToyCar, and SheenChair with correct UV tiling. Phase 8M (~200 LOC, one session) adds the sheen BSDF needed for ToyCar and SheenChair. Phase 8N (~200 LOC, one session) adds DDS texture loading for GPUOpen Cauldron-Media scenes (BistroInterior, AbandonedWarehouse, Brutalism). F9-6a (C++-only, one session) adds --viewpoints JSON batch mode to monti_datagen. F9-6b (Python-only, one session) expands scene downloads and generates viewpoints. F9-6c (PyTorch-only, one session) implements augmentation transforms. F9-6d (orchestration, one session) wires everything together for full dataset generation. F9-6a has no renderer feature dependencies and could be implemented before or in parallel with 8L/8M/8N.

---

## RTXPT Reference Test Scenes

The NVIDIA RTXPT project (and its companion [RTXPT-Assets](https://github.com/NVIDIA-RTX/RTXPT-Assets) repository) provides several glTF test scenes that are useful for validating Monti rendering quality. Since RTXPT and Monti both consume glTF 2.0 with the same PBR extensions, these scenes can serve as direct comparison targets.

### Recommended Test Scenes for Monti

| Scene | Tests | Monti Phases Exercised |
|---|---|---|
| **Amazon Lumberyard Bistro** | Many-light interiors, emissive signage, complex geometry, reflections | 8G, 8J, 8K, 8N, F1, F2 |
| **Kitchen** | Interior lighting, glossy reflections, transparent objects (glass, liquid) | 8E, 8I, F1 |
| **DragonAttenuation** | Transmission, volume attenuation, IOR, material variants | 8C, 8H, 8I |
| **A Beautiful Game** | Chess set with transmission + volumetric effects, clearcoat | 8C, 8H, 8I (direct comparison with rtx-chessboard) |
| **Transparent Machines** | Complex transparency, nested dielectrics | 8I, 8H |
| **Mosquito In Amber** | Nested transmission + IOR + volume | 8I, F4 |
| **Toy Car** | Clearcoat, sheen, transmission, texture transform | 8C, 8H, 8L, 8M |
| **Sponza** | Architectural lighting benchmark, environment + area lights | 8E, 8F, 8G, F1 |
| **Khronos glTF 2.0 sample models** | Material feature conformance (50+ models) | All material phases |

### Integration Approach

1. **Conformance testing:** Use Khronos glTF sample models as unit test references. Render at reference SPP (4096) and compare against known-good images.
2. **Visual comparison:** Render Bistro and Kitchen scenes in both RTXPT and Monti at matched settings (same camera, same SPP, same bounce count). Use FLIP error metric for quantitative comparison.
3. **Performance benchmarking:** Use Sponza and Bistro as standard benchmarks. Track frame time across phases to measure performance impact of new features.
4. **Scene format:** All RTXPT scenes use `.scene.json` configuration files that reference `.glb` models. Monti's glTF loader already handles `.glb`; the `.scene.json` wrapper (camera positions, light overrides) can be parsed with a small utility or manually transcribed.

---

## Future Phase Summary

> **Deni shader loading:** Phase 9A loads compiled SPIR-V from disk (`build/deni_shaders/*.spv`). A future cleanup pass should embed SPIR-V as C++ byte arrays at build time to eliminate the runtime file dependency and make Deni fully self-contained.

| Phase | Feature | Prerequisite |
|---|---|---|
| F1 | DLSS-RR in `monti_view` (NVIDIA-only, app-level quality reference) + denoiser selection UI | Phase 10B (interactive viewer with ImGui) |
| F2 | ReSTIR Direct Illumination | Phase 8K (WRS foundation) |
| F3 | Emissive mesh ReSTIR importance sampling | F2 (needs ReSTIR for correct sampling) |
| F4 | Volume enhancements (homogeneous + heterogeneous) | Phase 8I (nested dielectrics) |
| F6 | Mobile Vulkan renderer (`monti_vulkan_mobile`) | Shared GpuScene/GeometryManager ready |
| F7 | Metal renderer (C API) | Desktop design patterns established |
| F8 | WebGPU renderer (C API → WASM) | Desktop design patterns established |
| F9 | ML denoiser training pipeline | Capture writer complete |
| F10 | Shader permutation cache | Multi-bounce MIS complete |
| F11 | ML denoiser deployment in Deni (desktop + mobile) | F9 complete (trained weights available) |
| F12 | Super-resolution in ML denoiser | F11 complete; uses `ScaleMode` enum |
| F13 | Fragment shader denoiser (mobile) | F6 + F11 complete |
| F14 | GPU skinning + morph targets | Phase 6 (GeometryManager) |
| F15 | ReSTIR GI (indirect illumination reuse) | F2 complete |
| F16 | NRD ReLAX denoiser in Deni (cross-vendor) | F11 complete (deferred until cross-vendor denoising needed) |

---

## F1: DLSS-RR in `monti_view` (NVIDIA Quality Reference) + Denoiser Selection UI

> **NVIDIA only, app-level integration.** DLSS-RR (Deep Learning Super Sampling — Ray Reconstruction) is NVIDIA's ML-based denoiser providing the highest quality denoising available for ray-traced content. It is integrated directly in `monti_view` as app-level code — not in the Deni library — because it is vendor-locked to NVIDIA GPUs and serves as a development-time quality reference, not a shipping product denoiser.
>
> **Purpose:** DLSS-RR serves as the quality ceiling comparison for ML denoiser development. During training (F9), DLSS-RR output provides the perceptual reference for evaluating ML denoiser quality. During development, it provides interactive denoised viewing in `monti_view` for scene authoring and camera placement.
>
> **Reference implementation:** The [rtx-chessboard](../../../rtx-chessboard/) project has a working DLSS-RR integration using Volk for Vulkan function loading. This implementation can be directly adapted for `monti_view`.
>
> **Required inputs:** DLSS-RR consumes the same G-buffer channels already output by Monti's path tracer: noisy diffuse/specular radiance, motion vectors, linear depth, world normals + roughness, diffuse/specular albedo. No additional renderer modifications needed.
>
> **Integration approach:**
> 1. Add NVIDIA DLSS SDK as a dependency (downloaded, not committed — proprietary license).
> 2. Implement `app/view/dlss_rr_denoiser.h/.cpp` as app-local code (not in Deni).
> 3. Wire into `monti_view` render loop: trace → DLSS-RR denoise → tonemap → present.
> 4. Detect NVIDIA GPU at startup; fall back to Deni passthrough on non-NVIDIA hardware.
> 5. Add denoiser selection UI to `monti_view` settings panel (Panels.cpp): "Passthrough" / "DLSS-RR" toggle. Greyed out on non-NVIDIA hardware. This is the first denoiser UI — Phase 10B intentionally deferred the toggle until a second denoiser option existed.
>
> **Not in Deni:** DLSS-RR is intentionally not part of the Deni library. Deni is cross-vendor by design. DLSS-RR is an NVIDIA-only development tool used in the apps (`monti_view`, `monti_datagen`) for quality comparison during ML denoiser development.
>
> Expected cost: ~2–4 ms at 1080p on RTX 4090.

---

## F2: ReSTIR DI + Emissive Mesh Lights

> ReSTIR Direct Illumination will be added to support efficient many-light scenes, building on the WRS (Weighted Reservoir Sampling) foundation established in Phase 8K and emissive mesh light extraction from Phase 8J. While Phase 8K provides single-frame O(1) light selection, ReSTIR adds **temporal and spatial resampling** to dramatically improve sample quality by reusing light samples across frames and neighboring pixels.
>
> **Mobile considerations:** ReSTIR's 3 additional full-screen compute passes (temporal, spatial, final shading) add significant bandwidth. On older mobile GPUs without hardware RT acceleration, this exceeds the frame budget. However, newer mobile SoCs with dedicated RT hardware (Snapdragon 8 Elite, Dimensity 9400, Immortalis-G925+) can handle ReSTIR at mobile render resolution (540p): the reservoir buffer is ~16 MB double-buffered, and the 3 passes add ~50 MB/frame bandwidth — within budget when combined with ML upscale. The initial implementation targets desktop; mobile enablement is a follow-up once the mobile renderer (F6) is in place.

### ReSTIR Pipeline Overview

ReSTIR DI extends the Phase 8K WRS reservoir with spatiotemporal resampling. The pipeline inserts between camera ray generation and the path trace bounce loop:

1. **Candidate generation** — For each pixel, generate M light candidates using the Phase 8K WRS sampler. Evaluate unshadowed target PDF `p̂(x) = Le × BSDF × G / pdf_source` for each candidate. Store the best candidate in a per-pixel reservoir.
2. **Temporal resampling** — Merge the current frame's reservoir with the previous frame's reservoir at the reprojected pixel location (using motion vectors). Apply M-capping to bound bias (`max_temporal_M = 20`). This gives each pixel an effective sample count of ~20× without additional shadow rays.
3. **Spatial resampling** — For each pixel, gather K neighbor reservoirs (K = 3–5) within a radius and merge. Neighbor selection uses a randomized spiral pattern. Accept/reject based on normal similarity and depth ratio thresholds to avoid light leaking across geometric discontinuities.
4. **Final shading** — Trace a single shadow ray for the reservoir's selected light sample and apply the final contribution with the combined MIS weight.

### Reservoir Buffer Layout

Each pixel stores a compact reservoir (16 bytes packed):

```
struct Reservoir {
    uint   selected_light;  // Light index (Phase 8G unified light buffer)
    float  weight_sum;      // Running weight sum (for streaming WRS)
    float  target_pdf;      // p̂(selected sample) — target PDF evaluation
    uint   M;               // Number of candidates seen (for bias correction)
};
```

Double-buffered: `reservoir_current` + `reservoir_previous` (ping-pong per frame). Total memory: 2 × 16 bytes × width × height = ~64 MB at 4K.

### NEE-AT (Adaptive Temporal Feedback)

RTXPT uses NEE-AT to learn which lights matter over time. A per-light importance weight is maintained and updated each frame based on the actual contribution of selected lights. High-contribution lights are sampled more frequently in subsequent frames. This is implemented as a feedback buffer written during final shading and read during candidate generation to bias the WRS weight computation.

### Screen-Space Cache (SSC)

An optional tile-based local light cache (16×16 pixel tiles) stores the top-K most important lights per screen tile. During candidate generation, some fraction of candidates are drawn from the local tile cache (nearby lights likely to contribute) and the remainder from the global light importance map. This spatial locality heuristic improves convergence for scenes with clustered area lights.

### Integration with Phase 8K

Phase 8K provides the core WRS algorithm and `light_sampling.glsl`. ReSTIR extends this with:
- `restir_temporal.comp` — Temporal resampling compute shader
- `restir_spatial.comp` — Spatial resampling compute shader
- `restir_final.comp` — Final shading with MIS weight application
- Reservoir buffer allocation and ping-pong management in `GpuScene`

---

## F3: Emissive Mesh Importance Sampling via ReSTIR

> **Prerequisite:** F2 (ReSTIR DI pipeline).
>
> **Relationship to Phase 8J:** Phase 8J provides *basic* emissive mesh light extraction — a compute shader scans materials, decomposes emissive mesh faces into individual `TriangleLight` primitives, and adds them to the unified light buffer for single-frame WRS sampling (Phase 8K). This works well for small numbers of emissive meshes. However, scenes with many emissive meshes (hundreds of emissive triangles) degrade because even WRS selects only one light per pixel per bounce — the probability of selecting the *correct* emissive triangle is low without spatiotemporal resampling.
>
> **F3 scope:** Integrate the Phase 8J-extracted emissive triangle lights into the F2 ReSTIR DI pipeline. This means:
> - Emissive triangle lights participate in ReSTIR candidate generation alongside quad, sphere, and environment lights.
> - Temporal resampling reuses successful emissive light samples across frames (critical for convergence — an emissive TriangleLight that contributed last frame is likely still relevant).
> - Spatial resampling shares emissive light discoveries between neighboring pixels.
> - NEE-AT (F2) learns per-emissive-light importance over time.
>
> **Without F3:** Phase 8J + 8K provide basic emissive NEE (one WRS sample per bounce, no temporal reuse). This is functional but noisy for complex emissive-heavy scenes.
> **With F3:** Full ReSTIR treatment of emissive lights — converges dramatically faster for scenes like neon-lit streets, emissive signage, and complex interior lighting.

---

## F4: Volume Enhancements (Desktop Only)

> **Prerequisite:** Phase 8I (nested dielectric priority stack). Volume enhancements build on the IOR priority stack to properly track which medium the ray is currently inside.

### Homogeneous Scattering (Phase 1)

Add Beer-Lambert distance tracking for homogeneous participating media. When a ray enters a dielectric volume, sample a free-flight distance from an exponential distribution `t = -ln(ξ) / σ_t` where `σ_t` is the extinction coefficient. If the sampled distance is less than the next surface hit, a scattering event occurs in the medium:

- **Absorption:** Attenuate throughput by `exp(-σ_a × t)` along the ray segment.
- **Scattering:** At the scatter point, sample a new direction from a Henyey-Greenstein phase function with asymmetry parameter `g` (stored in `MaterialDesc`).
- **Transmittance:** If the ray exits the volume without scattering, apply `exp(-σ_t × distance)` transmittance.

New material properties: `vec3 scattering_coefficient`, `float scattering_anisotropy` (Henyey-Greenstein `g ∈ [-1, 1]`). These map to the KHR_materials_volume extension's `attenuationColor` + `attenuationDistance` (already partially supported in Phase 8C) with an added scattering term.

### Heterogeneous Media (Phase 2)

Add ray marching through spatially varying density fields (3D textures or procedural volumes). This requires:

- **Delta tracking** (Woodcock tracking): Step through the volume with a majorant extinction coefficient, accept/reject scattering events based on the local-to-majorant ratio. Unbiased.
- **3D density texture binding:** Extend `MaterialDesc` with an optional density volume texture index.
- **Enter/exit medium tracking:** Use the Phase 8I IOR stack to know which volume the ray is currently traversing. The top of the stack determines the active medium's scattering properties.

Heterogeneous media is considerably more expensive (many ray march steps per volume traversal) and should only be applied to materials flagged with volumetric density textures.

---

## F5: Future Platform Denoisers

> **Denoising strategy:** The ML-trained denoiser (F11) is the product denoiser — deployed in Deni across all platforms. DLSS-RR (F1) is an NVIDIA-only, app-level quality reference used during development and ML training. NRD ReLAX (F16) is a deferred cross-vendor fallback for AMD/Intel GPUs if needed before the ML denoiser is ready. ReBLUR is not planned.

| Library | Platform | Implementation |
|---|---|---|
| `deni_vulkan` (ML) | Desktop + Mobile | ML-trained single-pass denoiser (the product denoiser, F11) |
| `deni_vulkan` (NRD ReLAX) | Desktop only | NRD ReLAX spatial-temporal filter (F16, deferred, cross-vendor fallback) |
| `deni_metal` | iOS / macOS | ML denoiser in Metal compute (C API for Swift) |
| `deni_webgpu` | Web | ML denoiser in WebGPU compute (C API for WASM/JS) |
| DLSS-RR (app-level) | NVIDIA only | DLSS 3.5 Ray Reconstruction in `monti_view` / `monti_datagen` (F1, quality reference) |

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
    void RegisterMeshBuffers(MeshId mesh, const MeshBufferBinding& binding);
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
| Denoiser | Passthrough → ML (DLSS-RR in monti_view for dev) | Passthrough → ML (no ReLAX) |

### G-Buffer Formats

G-buffer images are **format-agnostic at the API level** — the renderer and denoiser accept `VkImageView` and use `shaderStorageImageReadWithoutFormat` / `shaderStorageImageWriteWithoutFormat` to read and write images in whatever format the host allocated. This means the host is free to use either compact or RGBA16F formats for any channel. No shader permutations, no API changes, no format negotiation.

The **recommended compact formats** provide 32% bandwidth savings and are the default in our app:

| Image | Recommended Format | Bytes/pixel | RGBA16F Alternative | Notes |
|---|---|---|---|---|
| `noisy_diffuse` | `VK_FORMAT_R16G16B16A16_SFLOAT` | 8 | (same) | HDR radiance, needs full FP16 range |
| `noisy_specular` | `VK_FORMAT_R16G16B16A16_SFLOAT` | 8 | (same) | HDR radiance, needs full FP16 range |
| `motion_vectors` | `VK_FORMAT_R16G16_SFLOAT` | 4 | RGBA16F (8 B) | Only .xy used |
| `linear_depth` | `VK_FORMAT_R16G16_SFLOAT` | 4 | RGBA16F (8 B) | .r = depth, .g = hit distance (Phase 8E) |
| `world_normals` | `VK_FORMAT_R16G16B16A16_SFLOAT` | 8 | (same) | .xyz normal + .w roughness |
| `diffuse_albedo` | `VK_FORMAT_B10G11R11_UFLOAT_PACK32` | 4 | RGBA16F (8 B) | Reflectance; no alpha, LDR-range |
| `specular_albedo` | `VK_FORMAT_B10G11R11_UFLOAT_PACK32` | 4 | RGBA16F (8 B) | F0 reflectance; no alpha, LDR-range |
| **Total (compact)** | | **40** | **56 (RGBA16F)** | **29% bandwidth savings with compact** |

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

---

## F14: GPU Skinning + Morph Targets

> **Prerequisite:** Phase 6 (GeometryManager with `NotifyMeshDeformed()` and BLAS refit support).

GPU-driven skeletal animation and morph target blending, removing the requirement for the host application to perform vertex deformation on the CPU before uploading.

### Joint Matrix Compute Shader

A compute shader transforms joint matrices from local bone space to world space using the skeleton hierarchy. Input: array of local joint transforms + parent indices. Output: array of world-space joint matrices in an SSBO. This runs once per skeleton per frame.

### Vertex Skinning Compute Shader

Reads the bind-pose vertex buffer and joint matrices SSBO. For each vertex, accumulates `Σ(weight_i × jointMatrix_i)` (up to 4 joints per vertex, matching glTF `JOINTS_0` / `WEIGHTS_0` attributes). Writes deformed position + normal + tangent to a staging vertex buffer. Workgroup size: 256 threads, one vertex per thread.

### Morph Target Blending

Before or combined with skinning, apply morph target deltas: `position += Σ(morph_weight_i × delta_position_i)`. Morph targets are stored as delta buffers (position + normal + tangent offsets). The blend weights are provided per frame by the host via `Scene::SetMorphWeights(MeshId, span<float>)`.

### BLAS Integration

After vertex deformation, call `GeometryManager::NotifyMeshDeformed(mesh_id, false)` to trigger a BLAS **refit** (not rebuild) for the deformed mesh. Refit is 2–5× faster than rebuild and acceptable for continuous animation where topology doesn't change. The existing Phase 6 BLAS refit path handles this.

### Host Interface

```cpp
// scene/include/monti/scene/Scene.h additions
void SetSkeleton(MeshId mesh, std::span<const glm::mat4> inverse_bind_matrices,
                 std::span<const int32_t> parent_indices);
void SetJointTransforms(MeshId mesh, std::span<const glm::mat4> local_transforms);
void SetMorphWeights(MeshId mesh, std::span<const float> weights);
```

The renderer detects dirty skeletons/morph weights per frame and dispatches the skinning compute shader before TLAS build.

---

## F15: ReSTIR GI (Indirect Illumination Reuse)

> **Prerequisite:** F2 (ReSTIR DI complete).

ReSTIR GI extends the reservoir framework to cache and reuse **secondary bounce** illumination. While ReSTIR DI resamples direct light sources, ReSTIR GI resamples indirect lighting paths:

- At each primary hit, trace a secondary ray and record the hit point + incoming radiance as a "GI sample."
- Store GI samples in per-pixel reservoirs (similar to F2's light reservoirs but storing path segments instead of light indices).
- Temporal resampling: merge with previous frame's GI reservoir at the reprojected location.
- Spatial resampling: gather neighbor GI reservoirs with Jacobian-corrected resampling weights.

ReSTIR GI dramatically improves indirect illumination quality at low SPP by amortizing expensive secondary bounces across frames and pixels. It is the primary technique for achieving real-time global illumination quality comparable to offline rendering.

Memory cost: ~32 bytes/pixel for GI reservoirs (double-buffered: ~128 MB at 4K). The temporal/spatial passes add ~2–3 ms at 1080p.
