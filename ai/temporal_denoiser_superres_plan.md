# Temporal Denoiser — Phases T6–T8: Super-Resolution and Mobile

> **Continuation of [temporal_denoiser_plan.md](completed/temporal_denoiser_plan.md)** — Phases T1–T5 are complete and archived in `ai/completed/`. This file covers the remaining phases.
>
> **End goal:** Temporal residual denoiser with super-resolution, running at <3ms on RTX 4090 (1080p output) and <4ms on Adreno 750 (720p→1080p). Quality approaching commercial denoisers via temporal accumulation and learned upsampling.
>
> **Key insight:** Rendering at half resolution reduces ray tracing cost by 4×. A learned upsampler recovers high-frequency detail using temporal accumulation to approach native-resolution quality. On mobile, fragment shader inference keeps intermediate feature maps in on-chip tile SRAM, avoiding the bandwidth penalty of compute shaders on TBDR GPUs.
>
> **Starting point:** Phase T5 ✅ — Temporal residual network inference deployed. v3 model (~30K params, base_channels=32) running in `monti_view`, replacing the v1 single-frame denoiser. Depthwise separable GLSL shaders (`depthwise_conv.comp`, `pointwise_conv.comp`) in production. Estimated 2-5ms on RTX 4090 at 1080p. Temporal accumulation provides ~1.3× quality improvement over single-frame baseline.

## Architecture Overview

```
T6: Super-resolution — training (perf: 4× fewer pixels to denoise)
T7: Super-resolution — inference (perf: full pipeline, render at half res)
T8: Mobile fragment shader backend (platform: mobile deployment via ncnn or custom)
```

### Cumulative Performance Estimates (1080p output, RTX 4090)

| After Phase | GFLOPS | Est. Time (current shaders) | Est. Time (coop matrix) | Quality vs Baseline |
|---|---|---|---|---|
| T4+T5 ✅ (temporal residual, base_channels=32) | ~22 | 3-5ms | 0.2-0.5ms | ~1.3× (temporal accumulation) |
| T6+T7 (super-res, render@540p) | ~14 | 1-3ms | 0.1-0.3ms | ~1.2× |

### Cumulative Performance Estimates (720p→1080p, Adreno 750 mobile)

| After Phase | GFLOPS | Est. Time (fragment) | Quality |
|---|---|---|---|
| T5 ✅ (temporal, base_channels=32) | ~10 (720p input) | 4-7ms | 1.3× |
| T7 (super-res) | ~7 | 2-4ms | 1.2× |
| T8 (mobile backend) | ~7 | 2-4ms (TBDR optimized) | 1.2× |

---

## Phase T6: Super-Resolution — Training

**Goal:** Train a super-resolution variant of the temporal residual network that denoises at half resolution and upscales 2× to the output resolution. This renders 4× fewer pixels and denoises 4× fewer pixels.

**Motivation:** The biggest cost in a real-time path tracer is ray tracing, not denoising. Rendering at 540p instead of 1080p reduces ray tracing cost by 4×. The super-resolution denoiser then upscales to 1080p, leveraging temporal accumulation to recover high-frequency detail that the low-resolution input lacks. This is the same principle as DLSS Performance/Quality modes.

**Retraining required.** The network architecture changes to include an upsampling tail. Requires paired low-res/high-res training data.

### Performance & Quality Estimates

- **Performance (at inference):** Denoiser runs on 960×540 input, adds a learned 2× upsampler to reach 1080p. Total GFLOPS: ~12 (denoise at half res) + ~4 (upsample) = ~16. Ray tracing cost drops 4×.
- **Quality:** At static camera, approaches native resolution quality after temporal convergence (10+ frames). During motion, ~1-2 dB below native temporal denoiser. Large net quality improvement over non-temporal approaches because the 4× ray tracing savings can be reinvested as more SPP at lower resolution.

### Tasks

#### 1. Generate paired low-res/high-res training data

Update `generate_training_data.py` to also render at half resolution with `--render-scale 0.5`:

```bash
python scripts/generate_training_data.py \
    --monti-datagen build/app/datagen/Release/monti_datagen.exe \
    --config training/configs/scenes.json \
    --viewpoints-dir training/viewpoints/ \
    --output training_data_superres/ \
    --render-scale 0.5 \
    --target-resolution 1920x1080
```

**`monti_datagen` changes:** Add `--render-scale` flag that renders at `scale × target_resolution` for the noisy input while accumulating the reference (target) at full `target_resolution`. Motion vectors are in the low-res coordinate space (screen-space pixels at render resolution).

Each frame produces:
- Noisy input: 19ch at 960×540 (low-res, same G-buffer as native)
- Reference target: 7ch at 1920×1080 (high-res: 6ch ref irradiance + 1ch hit mask)
- Motion vectors: 2ch at 960×540 (included in the 19ch input)

#### 2. Super-resolution architecture — `training/deni_train/models/superres_unet.py`

Extend the temporal residual network with a learned 2× upsampler tail:

```python
class DeniSuperResNet(nn.Module):
    """Temporal residual denoiser + 2× learned upsampler.
    
    Inputs (same as temporal residual: 26ch at low resolution):
        Same 26ch input as DeniTemporalResidualNet:
        - reprojected_diffuse: 3ch, reprojected_specular: 3ch
        - disocclusion_mask: 1ch
        - G-buffer: 19ch (noisy d/s irradiance, normals, roughness, depth, motion, albedo d/s)
    
    Outputs:
        - Denoised demodulated irradiance at 2× input resolution: 6ch (3ch diffuse + 3ch specular)
        - Remodulation happens in the output shader (same as T5)
    """
    
    def __init__(self, base_channels=8):
        super().__init__()
        c = base_channels
        
        # Temporal residual core (same as v3, operates at low res)
        # Returns cat(denoised_d, denoised_s) = 6ch
        self.temporal_core = DeniTemporalResidualNet(base_channels=c)
        
        # Learned 2× upsampler (operates at low res, outputs high res)
        # Input: 6ch demodulated irradiance from temporal core
        # PixelShuffle upsampling: 4c channels → c channels at 2× resolution
        self.upsample_pre = DepthwiseSeparableConvBlock(6, c * 4)
        self.pixel_shuffle = nn.PixelShuffle(2)  # (B, 4c, H, W) → (B, c, 2H, 2W)
        self.upsample_refine = DepthwiseSeparableConvBlock(c, c)
        self.upsample_out = nn.Conv2d(c, 6, kernel_size=1)  # 6ch demod irradiance at high res
    
    def forward(self, reprojected_d, reprojected_s, disocclusion,
                noisy_d, noisy_s, normals, roughness, depth, motion,
                albedo_d, albedo_s):
        # Low-res denoised output (6ch = cat(denoised_d, denoised_s), H×W)
        denoised_lr = self.temporal_core(reprojected_d, reprojected_s, disocclusion,
                                          noisy_d, noisy_s, normals, roughness,
                                          depth, motion, albedo_d, albedo_s)
        
        # Upsample 2× to high res (6ch demod irradiance, 2H×2W)
        up = self.upsample_pre(denoised_lr)   # → (4c)ch
        up = self.pixel_shuffle(up)            # → c ch at 2× res
        up = self.upsample_refine(up)          # → c ch
        up = self.upsample_out(up)             # → 6ch (high-res demod irradiance)
        
        return up  # Remodulation (denoised_d * albedo_d + denoised_s * albedo_s) in output shader
```

**Parameter count:** ~25-30K (temporal core ~15-20K + upsampler ~10K).

**PixelShuffle why:** PixelShuffle (sub-pixel convolution) is more efficient than transposed convolution for learned upsampling. It avoids checkerboard artifacts that plague transposed convolution and naturally distributes spatial information across a 2×2 output neighborhood.

**Upsampling in demodulated space:** The upsampler operates on 6ch demodulated irradiance (not remodulated RGB) to preserve the per-lobe separation. Albedo remodulation happens in the output shader at high resolution, where the high-res albedo G-buffer is available. This gives the sharpest texture detail.

#### 3. Super-resolution training config — `training/configs/v4_superres.yaml`

```yaml
model:
  type: superres
  base_channels: 8
  use_depthwise_separable: true

data:
  data_dir: "../training_data_superres"
  render_scale: 0.5
  crop_size: 128     # Low-res crop (outputs 256×256 high-res)
  batch_size: 4

loss:
  lambda_l1: 1.0
  lambda_perceptual: 0.1
  lambda_temporal: 0.5

training:
  epochs: 250
  learning_rate: 1.0e-4
```

#### 4. Training strategy

**Two-stage training:**
1. **Stage 1 (100 epochs):** Freeze temporal core weights (loaded from v3 checkpoint), train only the upsampler. This lets the upsampler learn to produce clean high-res output from the temporal core's low-res output.
2. **Stage 2 (150 epochs):** Unfreeze all weights, fine-tune end-to-end with lower learning rate (1e-5). This allows the temporal core to adapt its output to what the upsampler needs.

This two-stage approach converges faster and more stably than training everything from scratch.

### Numerical Validation Tests (PyTorch-side)

**Test: `test_superres_model.py` — Shape validation**

1. Create `DeniSuperResNet(base_channels=8)`
2. Forward pass: all inputs at 128×128, output should be 256×256
3. **Pass criteria:** Output shape = (B, 6, 256, 256); parameter count in 25K-35K range

**Test: `test_superres_pixel_shuffle.py` — PixelShuffle correctness**

1. Create known 4-channel input at 2×2
2. Apply PixelShuffle(2)
3. **Pass criteria:** Output is 1-channel at 4×4, correctly interleaved

**Test: `test_superres_training.py` — Training converges on synthetic data**

1. Overfit on 2 synthetic low-res→high-res pairs
2. Train for 100 steps
3. **Pass criteria:** Loss decreases, output PSNR improves

### Verification
- Super-resolution model trains successfully in both stages
- Evaluation: high-res output PSNR within 1-2 dB of native-res temporal denoiser
- Temporal stability maintained at high resolution
- Weight export produces valid `.denimodel`

---

## Phase T7: Super-Resolution — Inference

**Goal:** Implement the super-resolution upsampler on the GPU and wire the full pipeline: render at half resolution → temporal denoise at half res → learned upsample to full res.

**No retraining required.** Uses v4 weights from T6.

### Performance & Quality Estimates

- **Performance:** Full pipeline at 1080p output: denoise at 540p (~8 GFLOPS) + upsample to 1080p (~4 GFLOPS) = ~12 GFLOPS total. Estimated 1-3ms on RTX 4090. Plus ray tracing at 540p is 4× cheaper.
- **Quality:** High-res output quality approaches native temporal denoiser after temporal convergence. During fast motion, 1-2 dB below native. Far superior to single-frame denoiser.

### Tasks

#### 1. PixelShuffle shader — `pixel_shuffle.comp`

```glsl
#version 460
layout(local_size_x = 16, local_size_y = 16) in;

layout(constant_id = 0) const uint IN_CHANNELS = 32;   // Must be divisible by 4
layout(constant_id = 1) const uint SCALE = 2;

layout(push_constant) uniform PushConstants {
    uint in_width;
    uint in_height;
};

layout(set = 0, binding = 0, rgba16f) uniform readonly image2DArray feature_in;
layout(set = 0, binding = 1, rgba16f) uniform writeonly image2DArray feature_out;

void main() {
    // Output pixel coordinates (high resolution)
    uint ox = gl_GlobalInvocationID.x;
    uint oy = gl_GlobalInvocationID.y;
    uint out_width = in_width * SCALE;
    uint out_height = in_height * SCALE;
    if (ox >= out_width || oy >= out_height) return;
    
    // Map to input coordinates
    uint ix = ox / SCALE;
    uint iy = oy / SCALE;
    uint sub_x = ox % SCALE;
    uint sub_y = oy % SCALE;
    
    // PixelShuffle: input channel index = out_ch * SCALE² + sub_y * SCALE + sub_x
    uint out_channels = IN_CHANNELS / (SCALE * SCALE);
    
    for (uint oc = 0; oc < out_channels; ++oc) {
        uint ic = oc * SCALE * SCALE + sub_y * SCALE + sub_x;
        // Read from input image array
        uint in_layer = ic / 4;
        uint in_comp = ic % 4;
        vec4 texel = imageLoad(feature_in, ivec3(ix, iy, in_layer));
        float val = texel[in_comp];
        
        // Write to output image array
        uint out_layer = oc / 4;
        uint out_comp = oc % 4;
        // ... accumulate and write (group output writes by layer)
    }
}
```

#### 2. Scale mode support — `deni/vulkan/Denoiser.h`

Add `ScaleMode` to the public API:

```cpp
enum class ScaleMode {
    kNative,       // Render and denoise at output resolution (1:1)
    kQuality,      // Render at 2/3 resolution, upsample (1.5× pixel reduction)
    kPerformance,  // Render at 1/2 resolution, upsample (4× pixel reduction)
};
```

`DenoiserInput` gets a new field:
```cpp
uint32_t output_width;   // Target output resolution (may differ from render_width)
uint32_t output_height;
```

When `render_width != output_width`, the inference pipeline automatically uses the super-resolution path.

#### 3. Reprojection at mixed resolutions

The reprojection shader (T3) needs to operate at the **high-res** output resolution, because the previous frame's output is high-res. The reprojected result is then downsampled to low-res for the temporal residual network input.

Add a simple 2× downsample shader (average pooling) for reprojection:

```
Previous high-res output → reproject at high-res → downsample 2× → feed to temporal core
```

Alternatively, reproject at low-res directly from a downsampled copy of the previous output. This is cheaper and simpler:

```
Previous high-res output → downsample 2× → reproject at low-res → feed to temporal core
```

The second approach is preferable — fewer pixels to warp and no high-res intermediate.

#### 4. Update `Infer()` dispatch sequence for super-resolution

```cpp
if (is_superres) {
    // 1. Downsample previous high-res output to low-res
    DispatchDownsampleAvg(cmd, frame_history_.denoised_output, lowres_prev);
    
    // 2. Reproject at low resolution
    DispatchReproject(cmd, lowres_prev, input.motion_vectors, ...);
    
    // 3. Temporal residual network at low resolution
    //    (same as T5, but at render_width × render_height)
    DispatchTemporalCore(cmd, ...);  // Output: low-res denoised (3ch)
    
    // 4. Learned upsample: low-res → high-res
    DispatchDepthwiseConv(cmd, lowres_denoised, ...);  // upsample_pre
    DispatchPointwiseConv(cmd, ...);
    DispatchGroupNorm(cmd, ...);
    DispatchPixelShuffle(cmd, ...);                     // 2× spatial
    DispatchDepthwiseConv(cmd, ...);                    // upsample_refine
    DispatchPointwiseConv(cmd, ...);
    DispatchGroupNorm(cmd, ...);
    DispatchSuperResOutputConv(cmd, ...);               // → high-res RGB
    
    // 5. Save high-res output
    CopyImage(cmd, output_image, frame_history_.denoised_output);
}
```

### Numerical Validation Tests

**Test: `[deni][numerical][golden]` — v4 super-res model matches PyTorch**

1. Golden reference: 32×32 input, 64×64 output (2× upscale)
2. Run GPU inference
3. **Pass criteria:** RMSE < 0.01, max_abs_error < 0.05

**Test: `[deni][superres][pixel_shuffle]` — PixelShuffle shader correctness**

1. Create known 4×4 input with 16 channels
2. Apply PixelShuffle(2)
3. **Pass criteria:** Output is 4-channel at 8×8, correctly interleaved (matches PyTorch `nn.PixelShuffle`)

**Test: `[deni][superres][scale_modes]` — Both scale modes produce valid output**

1. Run with `ScaleMode::kNative` (should use temporal core only)
2. Run with `ScaleMode::kPerformance` (should use temporal core + upsampler)
3. **Pass criteria:** Both produce non-NaN output, superres output is at 2× resolution

### Verification
- All `[deni][numerical][golden]` tests pass with v4 model
- PixelShuffle shader matches PyTorch reference
- Full pipeline: 540p render → denoise → 1080p output works end-to-end
- GPU timestamp: total denoiser <3ms at 1080p output on RTX 4090
- No Vulkan validation errors

---

## Phase T8: Mobile Fragment Shader Backend

**Goal:** Implement an ncnn-compatible or custom fragment shader inference backend for mobile tile-based GPUs. This enables the temporal denoiser + super-resolution pipeline on mobile devices (Adreno, Mali, Apple GPU).

**Motivation:** Mobile GPUs use tile-based deferred rendering (TBDR) where fragment shader execution benefits from on-chip tile SRAM. Storage buffer compute shaders bypass this optimization. By implementing ML inference as a series of fullscreen quad draws with texture inputs and render target outputs, the intermediate feature maps stay on-chip. This is how ncnn's Vulkan fragment shader backend works, and it's the standard approach for mobile ML inference.

**No retraining required.** Same v4 model weights — only the GPU execution strategy changes.

### Performance & Quality Estimates

- **Performance:** Mobile Adreno 750 at 720p input → 1080p output: estimated 2-4ms (tile SRAM hides memory bandwidth). Mobile Mali-G720: 3-6ms.
- **Quality:** Identical to desktop (same weights, same math). FP16 throughout for mobile — no precision difference since desktop already uses FP16 features.

### Tasks

#### 1. Fragment shader infrastructure — `denoise/src/vulkan/MlFragmentInference.h/cpp`

Create a new inference backend alongside `MlInference`:

```cpp
class MlFragmentInference {
public:
    // Same interface as MlInference
    void Initialize(VkDevice device, VmaAllocator allocator, ...);
    void LoadWeights(std::string_view model_path);
    void Resize(uint32_t width, uint32_t height);
    void Infer(VkCommandBuffer cmd, const DenoiserInput& input, VkImageView output);
    
private:
    // Render passes for each layer (VkRenderPass + VkFramebuffer)
    // One render pass per conv layer output
    // Input textures (VkSampler + VkImageView) for feature maps
    // Fullscreen quad vertex buffer (or use gl_VertexIndex trick)
};
```

#### 2. Fragment shader convolution — `conv_frag.frag`

```glsl
#version 460

layout(location = 0) out vec4 out_color;  // One RGBA16F output (4 channels)

layout(push_constant) uniform PushConstants {
    uint out_channel_group;  // Which group of 4 output channels this draw produces
};

// Input feature maps as texture arrays (sampler provides hardware bilinear)
layout(set = 0, binding = 0) uniform sampler2DArray feature_in;
// Weights stay in storage buffers (small, accessed linearly)
layout(set = 0, binding = 1) readonly buffer WeightBuffer { float weights[]; };

void main() {
    ivec2 pos = ivec2(gl_FragCoord.xy);
    uint base_oc = out_channel_group * 4;
    
    vec4 sums = vec4(0.0);
    
    // Depthwise separable: 3×3 depthwise + 1×1 pointwise
    // ... same math as compute shader but reads via texelFetch
    
    out_color = sums;
}
```

Each draw call renders a fullscreen quad and produces 4 output channels (one RGBA16F render target). A 16-channel layer requires 4 draw calls. MRT (multiple render targets) can output up to 8 channels per draw if supported.

#### 3. Render pass setup

For each conv layer:
1. Create `VkRenderPass` with RGBA16F color attachment (output feature layer)
2. Create `VkFramebuffer` with the output feature `VkImageView` (one layer of the image array)
3. Bind input feature textures and weight buffers
4. Draw fullscreen quad (3 vertices, `gl_VertexIndex` trick — no vertex buffer needed)
5. End render pass

Transition input textures to `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` before each draw, and output to `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL`.

#### 4. GroupNorm in fragment shader

GroupNorm requires global statistics (mean/variance across all pixels in a group). This can't be done in a single fragment pass. Two options:

**Option A (simpler):** Use compute shader for GroupNorm reduction, fragment for everything else. This is a hybrid approach — compute for reduction, fragment for the spatial operations.

**Option B (pure fragment):** Two fullscreen passes:
1. **Pass 1:** Render to a 1×1 texture using GL_EXT_fragment_shader_interlock or atomic operations to accumulate sum/sum-of-squares. This is slow and non-portable.
2. **Pass 2:** Read the 1×1 statistics texture, normalize in fragment shader.

**Recommendation:** Option A (hybrid). GroupNorm reduction is a tiny fraction of compute — using a compute shader for it doesn't negate the TBDR benefits of fragment-based convolution.

#### 5. Downsample and upsample in fragment shader

- **Downsample:** Render to half-resolution render target. Fragment shader reads 2×2 texels via `texelFetch` and computes max. Trivial.
- **Bilinear upsample:** Render to double-resolution render target. Fragment shader reads input via `texture()` (hardware bilinear interpolation). Essentially free — the texture unit does the work.
- **PixelShuffle:** Fragment shader reads from the correct input channel based on `gl_FragCoord` sub-pixel position. Same math as compute shader.

#### 6. Backend selection — `Denoiser.cpp`

Add backend selection in `DenoiserDesc`:

```cpp
enum class InferenceBackend {
    kCompute,    // Storage images + compute shaders (desktop)
    kFragment,   // Fragment shaders + render passes (mobile TBDR)
    kAuto,       // Auto-detect: fragment on mobile, compute on desktop
};
```

Auto-detection uses `VkPhysicalDeviceProperties` to identify mobile GPUs (Adreno, Mali, Apple by vendor ID), or the app can override.

#### 7. ncnn export (optional)

If using ncnn instead of custom fragment shaders:
- Export model to ncnn format via ONNX → ncnn converter
- ncnn handles all the fragment shader generation, render pass setup, and GroupNorm decomposition
- The deni integration wraps ncnn's `VkCompute` class
- **Trade-off:** ncnn manages its own command buffers (breaks deni's API contract of recording into caller's command buffer). Requires a workaround — either (a) submit ncnn's work on a separate queue and synchronize with semaphores, or (b) modify ncnn to record into an external command buffer (requires ncnn fork).

**Recommendation:** Start with custom fragment shaders to maintain API compatibility. Consider ncnn if the custom path proves too complex or if ONNX ecosystem tooling is needed.

### Numerical Validation Tests

**Test: `[deni][mobile][fragment_vs_compute]` — Fragment backend matches compute**

1. Run the same model/input through both compute and fragment backends
2. Compare outputs pixel-by-pixel
3. **Pass criteria:** RMSE < 0.005 (may not be bit-exact due to different texture filtering paths)

**Test: `[deni][mobile][fragment_golden]` — Fragment backend matches PyTorch**

1. Run golden reference test with fragment backend
2. **Pass criteria:** RMSE < 0.01, max_abs_error < 0.05 (same as compute)

**Test: `[deni][mobile][tbdr_perf]` — Performance measurement**

1. Run inference on a mobile device (or emulator) with GPU timestamps
2. **Pass criteria:** Report timing (no specific threshold — performance characterization)

### Verification
- Fragment backend produces output matching compute backend within tolerance
- Golden reference tests pass with fragment backend
- Full pipeline (reproject → temporal denoise → upsample) works in fragment mode
- Performance measurements on available hardware recorded
- No Vulkan validation errors

---

## Summary: Complete Pipeline Comparison

### Architecture Evolution

```
v1 (F18):    G-buffer 19ch ──► 3-level U-Net (120K) ──► demod irradiance 6ch
                              16→32→64 standard conv
                              ~122 GFLOPS, single-frame, albedo demodulation

             (v2 eliminated — depthwise blocks go directly into v3)

v3 (T5):     [Reprojected d+s 6ch + Disocclusion 1ch + G-buffer 19ch = 26ch]
                       ──► 2-level Temporal Residual Net (30K) ──► demod irradiance 6ch
                       12→24 depthwise separable
                       ~22 GFLOPS, temporal accumulation, per-lobe blending + remodulation

v4 (T7):     [Low-res reprojected + G-buffer] @ 540p (26ch)
                       ──► Temporal Residual (30K) @ 540p
                       ──► Learned 2× Upsample (10K) → 1080p demod irradiance 6ch
                       ~14 GFLOPS, temporal + super-resolution
```

### Performance Summary

| Phase | Model | RTX 4090 (1080p) | Adreno 750 (720p→1080p) | Quality |
|---|---|---|---|---|
| Baseline (F18) | v1, 120K params | 15-40ms | — | 1.0× |
| ~~T1 (textures)~~ | ~~v1~~ | ~~8-20ms~~ | — | **SKIPPED** (regression) |
| T2+T3 (PyTorch blocks + reprojection) | v1 + reprojection infra | 15-40ms | — | 1.0× |
| T4+T5 (temporal) | v3, 30K params | 2-5ms | — | 1.3× |
| T6+T7 (super-res) | v4, 40K params | 1-3ms | — | 1.2× |
| T8 (mobile) | v4 | 1-3ms | 2-4ms | 1.2× |

### Training Pipeline Summary

| Phase | Retrain? | Dataset Change | Training Time (RTX 4090) |
|---|---|---|---|
| ~~T1~~ | **SKIPPED** | — | — |
| T2 | No | — (PyTorch blocks only) | — |
| T3 | No | — | — |
| T4 | Yes | New temporal sequences | ~1-2 hours |
| T5 | No | — | — |
| T6 | Yes | New low-res/high-res pairs | ~2-3 hours |
| T7 | No | — | — |
| T8 | No | — | — |

### Test Coverage Summary

| Phase | New Tests | Tags |
|---|---|---|
| ~~T1~~ | **SKIPPED** | — |
| T2 | PyTorch block shape, determinism, gradient flow | `test_depthwise_block.py` |
| T3 | reproject identity/shift/disocclusion/dual_lobe, history lifecycle, depth copy | `[deni][temporal][reproject_*]`, `[deni][temporal][frame_history_*]` |
| T4 | model shape, channel assertion, blend weight bounds, gradient flow, first-frame, reprojection, training convergence, PSNR progression, temporal stability | `test_temporal_*.py` |
| T5 | golden, golden_multiframe, depthwise shader, first frame, accumulation, reset, remodulation, history demodulated, sigmoid stability | `[deni][numerical][*]`, `[deni][temporal][*]` |
| T6 | PyTorch model shape, pixel shuffle, training convergence | `test_superres_*.py` |
| T7 | golden, pixel_shuffle shader, scale modes | `[deni][superres][*]` |
| T8 | fragment_vs_compute match, fragment golden, perf | `[deni][mobile][*]` |

### Key Dependencies

```
F18 ✅ (single-frame inference + albedo demodulation)
  │
  ├─ ~~T1 (texture feature maps)~~         ─  SKIPPED (regression)
  ├─ T2 (depthwise separable PyTorch blocks) ─┤ ◄── parallelizable (Wave 1)
  ├─ T3 (reprojection infrastructure)     ─┘
  │
  ├─ Session 3 (sequential rendering in monti_datagen) ◄── required for temporal training data
  │
  └─ T4 (temporal training) ──► train v3
      └─ T5 (temporal inference + depthwise GLSL)
          └─ T6 (super-res training) ──► train v4
              └─ T7 (super-res inference)
                  └─ T8 (mobile fragment backend)
```

~~T1~~ is **SKIPPED** (performance regression — see T1 section). T2 and T3 are **parallelizable** (both are infrastructure with no model changes and no interdependency). T4 requires T2, T3, and Session 3 (camera path sequential rendering). T5 is the first phase that changes monti_view's denoiser output. Each training phase (T4, T6) requires the previous inference phase for data generation or baseline comparison.