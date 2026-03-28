# Deni & Monti — Platform Overview

> **Deni** is the product: a per-game trained ML denoiser that ships as a drop-in native library.
> **Monti** is the tooling: a reference path tracer and training data pipeline that powers Deni.

---

## The Problem

Real-time path tracing at 1 sample per pixel produces noisy images. Denoising recovers a clean frame, but today's options force a trade-off:

- **Vendor-locked denoisers** (DLSS-RR, XeSS) deliver great quality — but only on one vendor's hardware, with a fixed model trained on generic data not tunable to a specific game.
- **Generic open denoisers** (OIDN, NRD) work everywhere — but ship a one-size-fits-all model with no per-game optimization.

Neither option lets a studio train on their own assets, lighting conditions, and art style.

## What We Provide

A **complete system** — not just a denoiser, but the path tracer, denoiser, and training pipeline together:

```
Game assets (glTF) ──→ Monti (path tracer) ──→ Training data (multi-channel EXR)
                                                        │
                                                        ▼
                                         ML training (PyTorch) ──→ Trained model weights
                                                                          │
                                                                          ▼
                              Game engine ──→ Deni (denoiser library) ──→ Clean frame
```

1. **Monti** renders the customer's scenes at low spp (noisy input) and high spp (ground truth), exporting 9-channel EXR frames: noisy diffuse, noisy specular, reference diffuse, reference specular, normals, depth, motion vectors, diffuse albedo, and specular albedo.
2. A **training pipeline** fine-tunes the ML denoiser model on this game-specific data.
3. **Deni** ships as a standalone library (~single header + compiled lib) with the trained weights. The customer integrates it with a few API calls and ships it with their game.

---

## Key Benefits

### 1. Per-Game Trained Denoiser

Unlike fixed denoisers, Deni is trained on each customer's actual assets and lighting. A studio rendering stylized environments, hard-surface robots, or organic landscapes gets a model that understands *their* noise distribution — not an average across unrelated scenes. The training data pipeline is included, not an external dependency.

### 2. Native GPU API — No Abstraction Layer

Deni and Monti use platform-native types directly. On Vulkan, the API accepts `VkDevice`, `VkCommandBuffer`, `VkImageView`. On Metal (future), `MTLDevice`, `MTLCommandBuffer`, `MTLTexture`. On WebGPU (future), `WGPUDevice`, `WGPUCommandEncoder`, `WGPUTextureView` — compiled to WASM for browser deployment. No intermediate handle system, no hidden allocations, no translation layer.

This means:
- **Zero integration friction** — customers pass resources they already own.
- **Full host control** — the game engine owns the device, synchronization, and memory. Deni records GPU work into the engine's command buffer.
- **No hidden costs** — Deni allocates GPU memory through the host's allocator (VMA on Vulkan, `MTLDevice`/`MTLHeap` on Metal, `WGPUDevice` on WebGPU), so all GPU memory is visible in one place. No secret allocation pools.

```cpp
// Integration is this simple:
auto denoiser = deni::vulkan::Denoiser::Create({
    .device          = my_device,
    .physical_device = my_physical_device,
    .allocator       = my_vma_allocator,
    .width           = 1920,
    .height          = 1080,
});

// Per frame — record into the engine's command buffer:
auto output = denoiser->Denoise(cmd, input);
// output.denoised_color is a VkImageView ready for tone mapping
```

### 3. Bandwidth-Conscious Architecture

The system is designed for efficiency across all platforms:

- **Super-resolution built in** — The denoiser combines denoising and upscaling in a single inference pass, controlled by a `ScaleMode` enum (Native 1×, Quality 1.5×, Performance 2×). On mobile this is essential (540p → 1080p at Performance = 4× fewer rays); on desktop it frees GPU headroom for higher bounce counts or denser scenes. Each scale mode maps to a trained ML model variant for maximum quality.
- **Uniform FP16 G-buffer** — All G-buffer channels use FP16 formats (RGBA16F or RG16F). This eliminates format conversions between the renderer, denoiser, and data generation pipeline — the renderer's output images are passed directly to the ML denoiser without copies, format changes, or full-screen passes. Shaders access G-buffer images through format-less storage image reads/writes (`shaderStorageImageReadWithoutFormat`), so the host can use RG16F for motion vectors and depth (4 bytes/pixel) and RGBA16F for all other channels (8 bytes/pixel) — same shaders, same API, no permutations.
- **Hybrid rendering on mobile** — Primary visibility is rasterized (exploiting TBDR tile memory on mobile GPUs), while indirect bounces use hardware ray queries. This cuts ray tracing workload 40–60%.
- **Fragment shader denoiser on mobile** — The ML denoiser runs as a fullscreen fragment shader, chaining denoise → tone map → present as render pass subpasses. Intermediate results stay in on-chip tile memory and never touch main memory. The API is identical on desktop and mobile.

### 4. Denoiser Progression Path

Deni provides two denoiser tiers, each usable independently:

| Tier | What It Is | Use Case |
|---|---|---|
| **Passthrough** | Sum diffuse + specular; no denoising | Pipeline validation, baseline comparison |
| **ML Denoiser** | Per-game trained neural network | Production quality; trained on customer's assets |

The ML denoiser is the primary target. Passthrough validates the pipeline from day one. During development, DLSS-RR (NVIDIA-only) is used at the app level in `monti_view` as a quality reference and for interactive viewing — it is not part of the Deni library. NRD ReLAX may be added later as a cross-vendor classical fallback if needed before ML weights are available. All tiers share the same API — switching between them is a configuration change, not a code change.

### 5. Cross-Platform with Per-Platform Quality

Each platform backend is written in native code, not generated from an abstraction:

| Platform | Renderer | Denoiser | API |
|---|---|---|---|
| **Vulkan (desktop)** | Ray tracing pipeline | Compute shader | C++ |
| **Vulkan (mobile)** | Hybrid rasterize + ray query | Fragment shader | C++ |
| **Metal** (planned) | Metal RT | Metal compute/fragment | C |
| **WebGPU** (planned) | Screen-space ray march | Compute shader | C (→ WASM) |

All backends follow the same interface shape (`Create`, `Denoise`, `Resize`, `Destroy`) and parameter semantics. Porting from one platform to another requires changing types, not logic.

### 6. Training Data Pipeline Included

Monti's capture system generates production-grade training data:

- **Multi-layer EXR output** per frame (9 data layers): noisy diffuse, noisy specular, reference diffuse (high-spp), reference specular (high-spp), diffuse albedo, specular albedo, normals, depth, motion vectors
- **Headless batch mode** — `monti_datagen` runs automated camera paths, CLI-driven, suitable for CI farms
- **Interactive mode** — `monti_view` provides fly/orbit camera with real-time preview for scene inspection
- **Perceptual validation** — NVIDIA FLIP convergence tests verify renderer correctness before any training data is produced

A customer loads their glTF scenes, defines camera paths, and generates 10K–100K training frames. The pipeline is self-contained.

---

## Integration Effort

Deni integration requires:

1. **Create** — `Denoiser::Create(desc)` at startup.
2. **Per frame** — Fill `DenoiserInput` (7 `VkImageView` fields + resolution + reset flag), call `Denoise(cmd, input)`.
3. **Resize** — `Resize(new_width, new_height)` on window resize.
4. **Destroy** — Destructor cleans up.

The host engine owns all GPU resources (device, images, command buffers). Deni creates only its internal pipeline objects and descriptors. No global state, no singletons, no thread-unsafe initialization.

---

## Why Not Just Use DLSS / OIDN / NRD?

| | DLSS-RR | OIDN | NRD | **Deni** |
|---|---|---|---|---|
| Hardware | NVIDIA only | CPU (slow) / NVIDIA GPU | Any GPU | **Any GPU** |
| Per-game training | No | No | No | **Yes** |
| Mobile support | No | No | Limited | **Designed for it** |
| Source access | Closed | Open | Open | **Open interface** |
| Super-resolution | Combined (fixed) | No | No | **Combined (trainable)** |
| Integration model | SDK + runtime | Library | Library | **Library (native types)** |
| Training pipeline | Not provided | Not provided | Not provided | **Included** |

The core differentiator: Deni is not a better generic denoiser — it's a system for building *your* denoiser, trained on *your* data, shipped with *your* game.

---

## Target Hardware

**Desktop:** Any Vulkan 1.2+ GPU with `VK_KHR_ray_tracing_pipeline` — NVIDIA RTX 2000+, AMD RX 6000+, Intel Arc.

**Mobile:** Any Vulkan 1.2+ GPU with `VK_KHR_ray_query` + `VK_KHR_acceleration_structure` — Qualcomm Adreno 740+ (Snapdragon 8 Gen 3), ARM Mali-G720+ (Dimensity 9300), Samsung Xclipse (Exynos 2400).

**Future:** Metal RT (Apple A17 Pro+, M1+), WebGPU (all modern browsers).

---

## Summary

Deni + Monti provide a **vertically integrated path tracing and denoising system** where:

- The **denoiser is the product** — a lightweight native library customers drop into their engine.
- The **path tracer is the training tool** — generates game-specific training data from the customer's own assets.
- The **ML model is per-customer** — trained on real game data, not generic scenes, delivering quality tailored to each title.
- The **architecture is mobile-first** — TBDR-optimized, bandwidth-conscious, with super-resolution built into the interface.
- The **API is native** — no abstraction tax, no hidden allocations, full engine control over GPU resources. On every platform, library memory is allocated through the host's allocator — VMA on Vulkan, the host's `MTLDevice` or `MTLHeap` on Metal, the host's `WGPUDevice` on WebGPU.
