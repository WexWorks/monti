# ML Path Tracer Denoiser — Implementation Plan

## Overview

This document summarizes the design decisions for a machine learning denoiser targeting a
Vulkan-based path tracer, similar in concept to DLSS-RR. The system targets **non-NVIDIA
desktop Vulkan GPUs** (AMD, Intel) and **mobile Android Vulkan GPUs** (Qualcomm Adreno,
ARM Mali). It covers the full pipeline: training data generation, network architecture,
training strategy, inference deployment, and a phased rollout plan that uses ReLAX as a
working denoiser while the ML pipeline is developed.

---

## Part 1: Path Tracer Prerequisites

### 1.1 G-Buffer Layout

The G-buffer must be designed correctly from the start. Changing it later propagates through
every shader. Required channels per pixel:

```
// Geometry
vec3  worldPosition
vec3  worldNormal           // shading normal
vec3  geometryNormal        // needed for ReLAX specular/diffuse separation
float depth
float linearDepth
vec2  motionVector          // includes jitter delta correction (see section 3.3)

// Material
vec3  albedo                // demodulated from lighting
float roughness
float metallic
float ior
vec3  emissive

// Radiance — MUST be separated into two buffers
vec3  diffuseRadiance       // diffuse indirect lighting only
vec3  specularRadiance      // specular indirect lighting only
float diffuseHitT           // distance to first diffuse bounce hit
float specularHitT          // distance to first specular bounce hit

// Reprojection confidence
float reprojectionValidity  // 0=invalid, 1=fully valid (depth+normal+screen tests)
```

Key design decisions:
- **Albedo must be demodulated** — store albedo separately and divide it out of radiance
  buffers. Both ReLAX and the ML denoiser should denoise lighting variation, not albedo
  variation.
- **Diffuse and specular radiance must be separate buffers** — ReLAX applies different
  temporal accumulation strategies to each. This information cannot be recovered if combined.
- **HitT values are critical** — both ReLAX and the ML denoiser use hit distance to modulate
  filter radius. A distant hit point indicates low-frequency indirect lighting (blur
  aggressively); a nearby hit point is high frequency (preserve detail).

### 1.2 Path State Tracking

Add these fields to your path tracing loop's per-path state struct. They are pervasive changes
that are painful to add later:

```glsl
struct PathState {
    vec3  throughput;
    vec3  radiance;
    int   bounceCount;
    bool  hasSpecularBounce;      // true if any specular bounce has occurred
    bool  lastBounceSpecular;     // for ReLAX lobe routing
    float lastBounceRoughness;    // for specular/diffuse blend decisions
    float diffuseHitT;            // world-space distance to first diffuse hit
    float specularHitT;           // world-space distance to first specular hit
    bool  diffuseHitTSet;
    bool  specularHitTSet;
};
```

### 1.3 BSDF Sample Struct

Your BSDF sampling must track which lobe was sampled:

```glsl
struct BSDFSample {
    vec3  weight;
    vec3  direction;
    float pdf;
    bool  isSpecular;
    bool  isDiffuse;
    bool  isTransmission;
};
```

### 1.4 Render Pass Graph

Structure your renderer as an explicit pass graph from the start, even if most passes are
stubs. This is required for ReSTIR's multi-pass structure:

```
GBufferPass
  → PathTracingPass         (outputs separated diffuse/specular radiance + hitT)
  → [ReSTIR passes]         (stubs initially)
  → DenoisingPass           (ReLAX initially, ML denoiser later)
  → TemporalAccumulationPass
  → TonemappingPass
  → PresentPass
```

---

## Part 2: Advanced Lighting — ReSTIR and Caustics

### 2.1 Why Standard Path Tracing Fails for Difficult Lighting

Two failure modes that require special handling:

**The bright-light-through-a-crack problem:** A large intense light source seen through a
narrow aperture (door crack, window). The visible solid angle from inside the room is tiny.
Standard NEE shadow rays almost always hit the door rather than the crack. Acceptance rate
≈ crack_area / door_area.

**Caustics:** Light paths of the form Light → S+ → D (one or more specular bounces terminating
on a diffuse surface). A camera ray tracing this in reverse almost never arrives at the light.
Produces extreme fireflies at low SPP.

### 2.2 Portal Sampling (Near-Term Fix for the Crack Problem)

Explicitly tag apertures (doors, windows) as portals. Sample directions through the portal
using spherical rectangle sampling (Ureña et al. 2013 — "An Area-Preserving Parametrization
for Spherical Rectangles"). Combine with direct light sampling using MIS power heuristic:

```glsl
// Sample direction through portal
SphericalRect sr = buildSphericalRect(shadingPoint, portalCorners);
vec3 dir = sampleSphericalRect(sr, u1, u2, pdf_portal);

// Check if direction hits the light
if (intersectsLight(dir) && !occluded(shadingPoint, lightHit)) {
    float w = powerHeuristic(pdf_portal, pdf_light);
    contribution += Le * bsdf * cosTheta * w / pdf_portal;
}
```

### 2.3 Bidirectional Path Tracing (BDPT) for Caustics

BDPT traces paths from both camera and light, connecting subpath vertices. Light subpaths
naturally trace through refractive geometry and deposit at the correct caustic receiver
locations. Camera subpaths find these connection points.

Key implementation reference: Veach's thesis and PBRT (Chapter on Light Transport).

Caustic identification: any surface where `hasSpecularBounce == true` in the path state is
a potential caustic receiver. No explicit scene labeling required — emerges from path flags.

### 2.4 ReSTIR — Phased Addition

ReSTIR is a sampling strategy layered on top of your existing path tracer. It does not require
restructuring core BVH, material, or BSDF code.

**ReSTIR DI (Direct Illumination):** Per-pixel reservoir resampling for direct lighting.
Dramatically improves sampling efficiency for complex lighting including partially occluded
lights.

```glsl
struct Reservoir {
    LightSample y;      // currently selected sample
    float W;            // unbiased contribution weight
    float Wsum;         // running sum of candidate weights
    int   M;            // number of candidates processed
};
```

**ReSTIR GI (Global Illumination):** Extends resampling to indirect bounce hit points.
Improves one-bounce indirect but multi-bounce caustics remain challenging.

**Memory budget:** Plan for one to two full-screen reservoir buffers. On mobile this is a
non-trivial allocation — budget for it early even if ReSTIR is not yet implemented.

**Implementation order:** ReSTIR DI first, ReSTIR GI later. Both can be added without
restructuring existing code if the pass graph is in place.

---

## Part 3: ReLAX Denoiser — Phase 1 Denoiser

### 3.1 Why ReLAX Before the ML Denoiser

ReLAX serves multiple roles in the development plan:

- **Immediate working denoiser** while the ML pipeline is developed (months of work)
- **Runs on all target hardware** — non-NVIDIA desktop and mobile Android, unlike DLSS-RR
- **Quality reference** for ML denoiser development — use it to identify what the ML denoiser
  needs to improve upon
- **Fallback path** for devices where ML inference is too expensive
- **Foundation is shared** — the G-buffer layout ReLAX requires is identical to what the ML
  denoiser requires. No duplicated work.
- **DLSS-RR as perceptual reference** — use DLSS-RR (on NVIDIA hardware) as a quality ceiling
  comparison, ReLAX as a technical debugging reference

### 3.2 ReLAX vs SVGF

ReLAX extends SVGF with:

- **Specular/diffuse separation:** Separate filter parameters per lobe. Diffuse gets long
  temporal history (view-independent, stable). Specular gets shorter history (view-dependent,
  changes rapidly).
- **Virtual motion vectors for specular:** Specular reflections do not follow surface motion
  vectors. ReLAX computes a virtual hit point using specularHitT and traces where it moves
  under camera and scene motion. This is why specularHitT must be output from the path tracer.
- **History confidence / anti-lag:** Per-pixel history length counter. Decrements rapidly on
  reprojection failure. Modulates both temporal blend factor and spatial filter radius.
- **Adaptive α:** Exponential moving average with per-pixel adaptive blend factor driven by
  variance, motion, and history confidence. More principled than SVGF's fixed α.

### 3.3 Fixing the Subpixel Jitter Problem (SVGF Baseline Fix)

The subpixel jitter / geometric edge sparkling problem from the existing WebGPU SVGF
implementation has three fixes, in order of priority:

**Fix 1 — Jitter-corrected motion vectors (2 lines of code):**
```glsl
vec2 jitterDelta = currentJitter - previousJitter;
vec2 motionVector = geometricMotion + jitterDelta;
```

**Fix 2 — Neighbourhood clamping (20 lines):**
```glsl
// Compute 3x3 neighbourhood bounds in current frame
vec3 neighbourMin = vec3(1e10);
vec3 neighbourMax = vec3(-1e10);
for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
        vec3 s = sampleCurrentRadiance(uv + vec2(i,j) * texelSize);
        neighbourMin = min(neighbourMin, s);
        neighbourMax = max(neighbourMax, s);
    }
}
// Clip reprojected history into neighbourhood before blending
vec3 history = sampleReprojected(uv, motionVector);
history = clamp(history, neighbourMin, neighbourMax);
vec3 result = mix(history, current, alpha);
```

**Fix 3 — Variance-adaptive α:**
High local variance indicates jitter-troubled pixels. Reduce α (increase weight on current
frame) at high-variance pixels.

### 3.4 ReLAX Implementation Order

1. Fix SVGF with the three changes above (~1 week)
2. Add specular/diffuse radiance separation to G-buffer and path tracer
3. Implement separate temporal accumulation per lobe
4. Add virtual motion vectors using specularHitT
5. Add history confidence / anti-lag
6. Tune

**Reference implementation:** NVIDIA NRD (Real-time Denoisers) — pure HLSL compute shaders,
open source, HLSL→GLSL translation is mechanical.
- https://github.com/NVIDIAGameWorks/RayTracingDenoiser

---

## Part 4: ML Denoiser — Network Architecture

### 4.1 Architecture: Temporal U-Net with Warped Frame Concatenation

**Core:** U-Net encoder-decoder with skip connections. Multi-scale noise handling, high
frequency detail preservation via skip connections.

**Temporal strategy:** Warp N=4–8 previous frames into current frame space using motion
vectors. Concatenate as additional input channels. Simpler and more parallelizable than
recurrent approaches (ConvLSTM). Equivalent to DLSS 2's approach.

**Attention:** Single spatial attention layer at the bottleneck only. Full attention at every
scale is too expensive. Channel attention (Squeeze-and-Excitation) is a cheaper alternative
if spatial attention proves too costly at inference.

### 4.2 Input Channels

Per-frame input (×N frames after warping):
```
noisy color RGB          3
noise variance           1
world-space normals      3
albedo (demodulated)     3
depth (linear)           1
motion vectors           2
roughness                1
metallic                 1
IOR                      1
reprojection validity    1  ← mask indicating where warped history is trustworthy
─────────────────────────
per frame total         17
```

For N=4 temporal frames: 17 × 4 = 68 input channels plus current frame G-buffer = ~85 total.
In practice use random crops and consider reducing N initially.

### 4.3 Recommended Channel Counts by Target

| Target | Parameters | Encoder channels | Approach |
|---|---|---|---|
| Mobile (< 5ms) | 500K–1M | 16/32/64 | ncnn Vulkan, FP16 |
| Desktop mid-range | 2M–5M | 32/64/128 | ncnn Vulkan |
| Desktop high-end | 5M–20M | 64/128/256 | ncnn Vulkan, optimize later |

Start with the mid-range configuration and scale down if mobile performance requires it.

### 4.4 Layer Types Required

| Layer | Complexity | Notes |
|---|---|---|
| 3x3 Convolution | High | Core cost; weight storage, channel loops, ping-pong buffers |
| 1x1 Convolution | Medium | Matrix multiply per pixel; simpler than 3x3 |
| Depthwise separable conv | Medium | Preferred for mobile; two-pass implementation |
| Group normalization | Low-Medium | Preferred over instance norm; computable in single invocation |
| Batch norm (inference) | Trivial | Fold into conv weights at export time — disappears |
| ReLU / LeakyReLU | Trivial | One line |
| GELU | Low | Fast tanh approximation |
| Bilinear upsample | Low | Use hardware bilinear filtering |
| Skip concatenation | Low | Memory lifetime management |
| Residual add | Trivial | Two texture samples and add |
| Max/avg pooling | Low | Render target sizing |
| Spatial attention | Very High | O(N²) — use windowed or channel attention instead |
| Channel attention (SE) | Low-Medium | Global avg pool reduction + small MLP |

**Use group normalization, not instance normalization.** Instance norm requires a spatial
reduction pass (two full-screen passes per layer). Group norm computes statistics over channels
within a single invocation — much more shader-friendly and equally effective.

**Use depthwise separable convolutions** in the encoder for mobile. Standard 3x3 conv in
decoder where quality matters most.

**Note on separability:** Learned 3x3 convolution kernels are not separable in general.
Post-training SVD decomposition can identify approximately rank-1 kernels and apply separable
optimization selectively, but this is a late-stage optimization.

---

## Part 5: Training Data Generation

### 5.1 Training Pairs

Each training pair consists of:
- **Input:** Low-SPP noisy G-buffer sequence (N frames) with motion
- **Target:** High-SPP clean single frame (no motion blur) at higher resolution

Targets are **per-frame clean renders, not motion-blurred.** Motion vectors handle temporal
alignment. The network learns frame-accurate reconstruction; motion blur is not part of the
denoiser's job.

### 5.2 Reprojection Validity Mask

Render a per-pixel validity mask indicating whether warped history is trustworthy:
- Is the reprojected pixel within screen bounds?
- Does reprojected depth match current depth within threshold?
- Do reprojected normals agree with current normals?
- (Optional) Do surface IDs match?

Feed as input channel. Weight the temporal consistency loss term by this mask — downweight
or zero the temporal loss at invalid reprojection regions (disocclusions, fast motion, new
geometry).

### 5.3 Loss Function

```
L_total = λ1 · L_reconstruction + λ2 · L_perceptual + λ3 · L_temporal
```

- **L_reconstruction:** L1 loss (not L2 — L2 produces blurry results)
- **L_perceptual:** VGG feature loss — preserves perceptual sharpness
- **L_temporal:** Penalizes frame-to-frame differences after warping; weighted by
  reprojection validity mask
- **Compute all losses in tonemapped space** (ACES or Reinhard), not linear HDR. Bright
  fireflies dominate gradients in linear space.
- **Normalize inputs by per-frame exposure estimate** before computing loss.

### 5.4 Synthetic Stress Scenes

Generate programmatically to cover failure modes underrepresented in naturalistic data.
Sweep parameters continuously rather than hand-authoring discrete variants.

**Geometric / Aliasing:**
- Combs and gratings at Nyquist limit (sweep frequency and orientation)
- Chain-link fence (thin geometry + transparency + depth complexity)
- Venetian blinds (thin occluders + directional lighting)
- Dense foliage alpha cards

**Specular / Lighting:**
- Mirror spheres (reflections move differently from geometry — breaks motion vector assumptions)
- Thin specular highlights on curved surfaces
- Caustics (wine glass, water surface)
- Emissive geometry behind glass (refraction breaks motion/light relationship)

**Motion:**
- Helicopter/propeller blades (rotational — motion vectors exceed pixel width at blade tips)
- Oscillating pendulums at varying frequencies
- Camera zoom (radial motion vectors — underrepresented in typical data)
- Objects emerging from occlusion (canonical disocclusion test)
- Rotating textured sphere (zero motion vectors despite texture moving)
- Screen-edge motion (objects entering/leaving frame)

**Temporal Stability:**
- Flickering lights (strobing intensity)
- Exploding/spawning particles
- Iridescent / thin-film materials (specular color changes with view angle)

**Training batch sampler** should draw proportionally from the hard end of each parameter
range, not uniformly. A network trained with uniform sampling will see mostly easy cases.

**Mix stress phenomena** in single scenes — real content combines multiple difficult cases
simultaneously.

### 5.5 Training Infrastructure

- **Framework:** PyTorch with FP16 mixed precision (AMP)
- **Patch-based training:** Random 256×256 or 512×512 crops. Increases effective batch size,
  acts as data augmentation, makes full-resolution dataset appear much larger.
- **Batch size:** 4–8 at full resolution or 8–16 with patches on a single 4090
- **Data storage:** Google Cloud Storage or AWS S3 — not GitHub (100MB file limit,
  not designed for binary training data)
- **Model/code storage:** GitHub for network architecture, training scripts, exported
  weights, inference shaders

### 5.6 Training Hardware and Cost Estimate

| Phase | Hardware | Estimated Cost |
|---|---|---|
| Architecture validation | Local 4090 | Free |
| Ablation studies (parallel) | Vast.ai / RunPod 4090s | $200–500 |
| Serious training runs | Lambda Labs 8×A100 | $500–1500 |
| Final training + retraining | Lambda Labs 8×A100 | $500–1000 |
| **Total** | | **$1500–3000** |

**Biggest cost risk:** Discovering a G-buffer data quality problem after expensive training
runs. Validate every G-buffer channel visually before any cloud training. Run a 1000-iteration
sanity check locally first.

**Multi-GPU:** PyTorch DistributedDataParallel (DDP) requires minimal code changes and works
identically on cloud multi-GPU nodes as on local multi-GPU. Use `torchrun` launcher.

---

## Part 6: Inference Deployment — ncnn on Vulkan

### 6.1 Why ncnn

ncnn (Tencent) is the recommended inference framework for this target:

- Designed specifically for Android mobile Vulkan deployment on Adreno and Mali GPUs
- Vulkan compute is the primary GPU backend (not an afterthought)
- Targets Vulkan 1.0 core — minimal extension requirements, same feature set as your
  path tracer
- Vendor-agnostic compute shaders — works correctly on AMD and Intel desktop Vulkan
- Automatic FP16 detection and use (VK_KHR_16bit_storage) — significant mobile performance
  win at no cost
- Production-proven in major shipping mobile applications

### 6.2 Model Export Pipeline

```
PyTorch (training)
    ↓  torch.onnx.export()
ONNX model
    ↓  ncnn onnx2ncnn converter
ncnn .param + .bin files
    ↓  ncnn optimization tool (FP16 quantization, layer fusion)
Optimized ncnn model (.param + .bin)
    ↓  loaded at runtime via ncnn API
ncnn Vulkan inference
```

Stick to standard PyTorch nn.Module components. Exotic ops or custom layers risk conversion
failures in onnx2ncnn.

### 6.3 Vulkan Integration

ncnn supports importing existing VkImage and VkBuffer resources — no copy needed from your
renderer's memory into ncnn-managed buffers. Validate this early with a simple passthrough
test.

```
Your Vulkan render pass
    → G-buffer pass
    → Path tracing pass (outputs diffuse/specular radiance + hitT)
    → ncnn denoiser inference
        reads: your VkImages (radiance, G-buffer channels, previous frames)
        writes: denoised output VkImage
    → Tonemapping / present pass
```

### 6.4 Compute vs Fragment Shader Tradeoff on Mobile TBDR

The standard advice (compute > fragment) inverts on mobile TBDR in specific cases:

| Operation | Recommended shader type | Reason |
|---|---|---|
| 3x3 convolution | Compute | Neighborhood access spans tiles; shared memory tiling wins |
| 1x1 convolution | Fragment (subpass) | Pointwise — stays in tile memory |
| Activations, group norm, residual add | Fragment (subpass) | Pointwise — stays in tile memory |
| Bilinear upsample + skip concat | Fragment | Natural fullscreen quad operation |
| Attention | Compute | Non-local access pattern defeats tile optimization |
| Final composite | Fragment | Natural render pass integration |

**ncnn handles this automatically** in its Vulkan backend. This optimization is relevant only
if you later replace ncnn with custom shaders.

### 6.5 Early Validation Steps (Before Committing to Full Integration)

1. Run ncnn squeezenet example on target Android device
2. Verify VkImage import works in your renderer
3. Export a small test U-Net from PyTorch → ONNX → ncnn and verify conversion
4. Run inference and check output before investing in full architecture

### 6.6 When to Move Beyond ncnn (Later Optimization Phase)

- Specific layer is a profiling bottleneck and a custom shader would do better
- Memory bandwidth on mobile requires tile memory optimizations for specific passes
- Model uses operations onnx2ncnn cannot convert (fix: register custom layer in ncnn first)

---

## Part 7: Phased Implementation Plan

### Phase 1 — Foundation (Now)

**Goal:** Correct G-buffer, stable path tracer output, working baseline denoiser.

- [ ] Implement separated diffuse/specular radiance G-buffer buffers
- [ ] Output diffuseHitT and specularHitT from path tracer
- [ ] Add lobe tracking (isSpecular, isDiffuse) to BSDFSample struct
- [ ] Add path state flags (hasSpecularBounce, lastBounceSpecular, etc.)
- [ ] Add geometry normal alongside shading normal in G-buffer
- [ ] Structure renderer as explicit pass graph
- [ ] Fix SVGF subpixel jitter: jitter-corrected motion vectors + neighbourhood clamping +
      variance-adaptive α
- [ ] Add reprojection validity mask output

**Deliverable:** Stable path tracer with correct G-buffer. Fixed SVGF with no edge sparkling.

### Phase 2 — ReLAX (Working Denoiser)

**Goal:** Production-quality spatiotemporal denoiser running on all target hardware.

- [ ] Extend SVGF to separate specular/diffuse accumulation
- [ ] Implement virtual motion vectors for specular using specularHitT
- [ ] Implement history confidence / anti-lag
- [ ] Implement adaptive α driven by variance and history confidence
- [ ] Validate on AMD/Intel desktop and Android target devices
- [ ] Compare quality against DLSS-RR (on NVIDIA) as perceptual ceiling reference

**Deliverable:** ReLAX running on desktop Vulkan and mobile Android. Shippable quality.

### Phase 3 — Training Data Generation

**Goal:** High-quality training dataset covering difficult cases.

- [ ] Implement high-SPP clean frame renderer for ground truth targets
- [ ] Implement motion vector export for temporal sequences
- [ ] Build programmatic synthetic stress scene generator (see Section 5.4)
- [ ] Set up GCS or S3 storage for dataset
- [ ] Validate every G-buffer channel visually before large-scale generation
- [ ] Generate initial dataset: naturalistic scenes + synthetic stress scenes
- [ ] Implement data loader with proportional sampling toward hard cases

**Deliverable:** Training dataset with verified quality.

### Phase 4 — ML Denoiser Training

**Goal:** Trained U-Net denoiser meeting or exceeding ReLAX quality on benchmark scenes.

- [ ] Implement temporal U-Net in PyTorch (start with mid-range channel counts)
- [ ] Implement loss function: L1 + perceptual + temporal consistency weighted by validity mask
- [ ] Run architecture validation locally on 4090 (free)
- [ ] Run ablation studies on Vast.ai (parallel, cheap)
- [ ] Run final training on Lambda Labs 8×A100
- [ ] Compare output against ReLAX on held-out stress scenes
- [ ] Compare output against DLSS-RR perceptually

**Deliverable:** Trained model weights + ONNX export.

### Phase 5 — Inference Integration

**Goal:** ML denoiser running via ncnn on desktop and mobile Vulkan.

- [ ] Validate onnx2ncnn conversion of trained model
- [ ] Integrate ncnn into Vulkan render pass graph
- [ ] Verify VkImage import from renderer into ncnn
- [ ] Profile on target AMD/Intel desktop
- [ ] Profile on target Android device (Adreno and Mali)
- [ ] Tune batch size / precision for mobile performance targets

**Deliverable:** ML denoiser running on all target hardware.

### Phase 6 — ReSTIR and Advanced Lighting (Later)

**Goal:** Improved sampling for difficult lighting scenarios.

- [ ] Implement portal sampling (spherical rectangle — Ureña 2013)
- [ ] Implement ReSTIR DI reservoir structure and spatial/temporal reuse
- [ ] Implement ReSTIR GI for one-bounce indirect
- [ ] Evaluate BDPT for caustics in training data ground truth renders
- [ ] Generate caustic-specific synthetic stress scenes for ML denoiser retraining

### Phase 7 — Optimization (When Profiling Demands It)

**Goal:** Meet latency targets on mobile and low-end desktop.

- [ ] Profile ncnn dispatch to identify bottleneck layers
- [ ] Evaluate hybrid compute/fragment approach for mobile TBDR
- [ ] Consider post-training SVD decomposition for approximately separable kernels
- [ ] Evaluate hybrid ReLAX + ML denoiser pipeline for high-end desktop tier
- [ ] Consider tiered quality levels: ML denoiser (high-end), ReLAX (mid), SVGF (low/mobile)

---

## Part 8: Key References

| Topic | Reference |
|---|---|
| ReLAX / REBLUR denoiser | NVIDIA NRD: https://github.com/NVIDIAGameWorks/RayTracingDenoiser |
| SVGF | Schied et al. 2017 — "Spatiotemporal Variance-Guided Filtering" |
| ReSTIR DI | Bitterli et al. 2020 — "Spatiotemporal reservoir resampling for real-time ray tracing" |
| Portal sampling | Ureña et al. 2013 — "An Area-Preserving Parametrization for Spherical Rectangles" |
| Flash Attention (tiling strategy) | Dao et al. — https://github.com/Dao-AILab/flash-attention |
| ncnn inference framework | https://github.com/Tencent/ncnn |
| ONNX Runtime Web (WGSL ops reference) | https://github.com/microsoft/onnxruntime (js/web/lib/wasm/jsep/webgpu/ops/) |
| llama.cpp Vulkan shaders (matmul reference) | https://github.com/ggerganov/llama.cpp |
| OIDN (U-Net inference pipeline reference) | https://github.com/OpenImageDenoise/oidn |
| BDPT / MIS | Veach thesis; PBRT book (Light Transport chapter) |

---

## Part 9: Critical Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Temporal strategy | Warped frame concatenation (N=4–8) | Simpler than RNN, parallelizable, no hidden state management |
| Normalization layer | Group normalization | Single-invocation friendly; instance norm requires expensive reduction passes |
| Upsampling | Bilinear (hardware) | Free; transposed conv adds unnecessary complexity |
| Attention | Channel (SE block) or windowed spatial at bottleneck only | Full spatial attention is O(N²) — intractable in real-time shaders |
| Inference framework | ncnn | Mobile-first Vulkan design; AMD/Intel desktop compatible; FP16 automatic |
| Training loss | L1 + perceptual + temporal (tonemapped space) | L2 blurs; linear HDR dominated by fireflies |
| Phase 1 denoiser | ReLAX (SVGF extension) | Works on all target hardware; shared G-buffer with ML denoiser; quality reference |
| Desktop quality reference | DLSS-RR (perceptual) + ReLAX (technical) | DLSS-RR is black box NVIDIA-only; ReLAX is debuggable and cross-platform |
| Data storage | GCS or S3 | GitHub limits (100MB/file, ~1GB repo) make it unsuitable for training data |
| Training hardware | Local 4090 → Vast.ai → Lambda Labs | Cheapest path to validated model; total budget $1500–3000 |
