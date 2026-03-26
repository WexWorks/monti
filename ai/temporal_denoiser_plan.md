# Temporal Denoiser — Performance & Quality Optimization Plan

> **Purpose:** Incrementally transform the existing single-frame ML denoiser into a high-performance temporal denoiser with super-resolution, targeting both desktop (Vulkan compute) and mobile (fragment shader, TBDR) deployment. Each phase is sized for a single Copilot Claude Opus 4.6 session and includes numerical validation tests, training pipeline changes, and performance/quality estimates.
>
> **Starting point:** Phase F11-3 ✅ — single-frame DeniUNet (120K params, 13→3 channels, 3-level U-Net with channels 16→32→64). Inference via 7 GLSL compute shaders dispatched into caller's command buffer. FP16 feature buffers, FP32 weights, ~20 dispatches per frame. Validated against PyTorch golden reference (RMSE < 0.01).
>
> **Prerequisite: F18 (albedo demodulation).** Phases T2 and T4 retrain the model. To avoid redundant retraining, F18 must be completed first so that T2's retraining already uses demodulated inputs and T4's temporal architecture builds on the established demodulated representation. T1 and T3 are infrastructure changes (no model change) and can proceed before or after F18. See [roadmap.md](roadmap.md) for dependency ordering.
>
> **End goal:** Temporal residual denoiser with super-resolution, running at <5ms on RTX 4090 (1080p) and <15ms on Adreno 750 (720p→1080p). Quality approaching commercial denoisers via temporal accumulation.
>
> **Key insight:** Rather than denoising from scratch every frame, reproject the previous frame's clean output and use a smaller network to correct only the residual errors (disocclusion, ghosting, noise in new samples). This simultaneously improves quality (temporal accumulation) and performance (smaller network, fewer FLOPS).
>
> **Relationship to existing plans:** Replaces the outline phases F11-4, F11-5, F12, and F13 in [ml_denoiser_plan.md](ml_denoiser_plan.md) and [roadmap.md](roadmap.md) with a unified, sequenced implementation plan.
>
> **Session sizing:** Each phase is scoped to fit within a single Copilot Claude Opus 4.6 context session, following the convention in [ml_denoiser_plan.md](ml_denoiser_plan.md).

---

## Architecture Overview

The plan proceeds through 8 phases, each building on the previous:

```
T1: Texture-backed feature maps (perf: 1.5-2× faster inference, no quality change)
T2: Depthwise separable convolutions in training + inference (perf: 2-3× fewer FLOPS, small quality trade-off). Prerequisite: F18 (retrains on demodulated inputs).
T3: Motion reprojection infrastructure (no ML change, foundation for temporal)
T4: Temporal residual network — training (quality: major improvement, temporal stability). Prerequisite: F18 (demodulated inputs).
T5: Temporal residual network — inference (quality + perf: smaller network, temporal accumulation, albedo remodulation in output shader)
T6: Super-resolution — training (perf: 4× fewer pixels to denoise)
T7: Super-resolution — inference (perf: full pipeline, render at half res)
T8: Mobile fragment shader backend (platform: mobile deployment via ncnn or custom)
```

### Cumulative Performance Estimates (1080p, RTX 4090)

| After Phase | GFLOPS | Est. Time (current shaders) | Est. Time (coop matrix) | Quality vs Baseline |
|---|---|---|---|---|
| Baseline (F11-3) | ~122 | 15-40ms | — | 1.0× |
| T1 (texture features) | ~122 | 8-20ms | — | 1.0× (identical) |
| T2 (depthwise sep) | ~35 | 3-8ms | — | ~0.95× (small PSNR drop) |
| T3 (reprojection) | ~35 + warp | 4-9ms | — | ~0.95× |
| T4+T5 (temporal residual) | ~18 | 2-5ms | 0.2-0.5ms | ~1.3× (temporal accumulation) |
| T6+T7 (super-res, render@540p) | ~12 | 1-3ms | 0.1-0.3ms | ~1.2× |

### Cumulative Performance Estimates (720p→1080p, Adreno 750 mobile)

| After Phase | GFLOPS | Est. Time (fragment) | Quality |
|---|---|---|---|
| T5 (temporal) | ~8 (720p input) | 3-6ms | 1.3× |
| T7 (super-res) | ~6 | 2-4ms | 1.2× |
| T8 (mobile backend) | ~6 | 2-4ms (TBDR optimized) | 1.2× |

---

## Phase T1: Texture-Backed Feature Maps

**Goal:** Replace flat storage buffer feature maps with 2D image arrays (RGBA16F). This improves spatial cache locality for 3×3 convolutions. No model or training changes — inference produces bit-identical results.

**Motivation:** The current `conv.comp` stores features in channel-major `[C][H][W]` flat storage buffers. For a 3×3 kernel, neighboring rows are `width` elements apart in memory — poor 2D cache locality. GPU texture units use tiled/swizzled memory layouts (Morton order) optimized for 2D spatial access, and the texture cache is separate from the L1/L2 cache used by storage buffer loads. Switching to `image2DArray` with RGBA16F packing (4 channels per layer) gives us hardware-optimized 2D locality and frees the L1 cache for weight data.

**No retraining required.** The mathematical operation is identical — only the memory layout of intermediate activations changes.

### Design Decisions

- **GroupNorm in-place strategy:** Use a single readwrite `image2DArray` binding (no `readonly`/`writeonly` qualifier). Each thread owns a unique `(x, y)` pixel — no data race. Read vec4 (4 channels), normalize, write vec4. Zero extra memory, zero copies.
- **Skip connection copies eliminated:** Write encoder conv2 output directly to skip images (`skip0_`, `skip1_`), then downsample reads from the skip image. Removes 2× `vkCmdCopyBuffer` per frame. `buf0_b_` is freed for other use (or removed if unneeded).
- **Image utility:** Self-contained `FeatureImage` struct inside `denoise/` — no dependency on `renderer/Image` (which lacks array layer support).
- **Old buffer path:** Remove entirely. Validate against PyTorch golden reference only (no buffer-vs-texture comparison test).
- **Usage flags:** `VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT`. No `SAMPLED_BIT` needed yet (can add later for T3 bilinear reprojection).

### Prerequisites (verified)

- `MlInference` stores `VmaAllocator`, `VkDevice`, `MlDeviceDispatch` (includes `vkCreateImageView` / `vkDestroyImageView`)
- Descriptor pool already sized for `STORAGE_IMAGE` (384 slots) in addition to `STORAGE_BUFFER` (256 slots)
- All 7 compute shaders compiled via `glslc --target-env=vulkan1.2`; specialization constants are runtime (`VkSpecializationInfo`)
- Golden reference generator (`tests/generate_golden_reference.py`) and numerical test infrastructure (`tests/ml_inference_numerical_test.cpp`) with Catch2 exist
- VMA image allocation pattern already used in test code (`ml_inference_numerical_test.cpp`)

### Performance & Quality Estimates

- **Performance:** 1.5-2× faster on memory-bound layers (level 0 at full resolution, where bandwidth dominates). ~1.2× on compute-bound layers (bottleneck). **Overall ~1.5× speedup.**
- **Quality:** Bit-identical at FP16 precision (same data, same math, different memory layout). Boundary handling keeps explicit bounds checks for zero-padding correctness.

### Tasks

#### A. FeatureImage Infrastructure — `MlInference.h` / `MlInference.cpp`

**A1.** Define `FeatureImage` struct in `MlInference.h` (replacing `FeatureBuffer`):

```cpp
struct FeatureImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    uint32_t layers = 0;  // number of RGBA16F layers = ceil(channels / 4)
    uint32_t width = 0, height = 0;
};
```

**A2.** Implement `AllocateFeatureImage()` replacing `AllocateFeatureBuffer()`:
- `VkImageCreateInfo`: `VK_IMAGE_TYPE_2D`, `VK_FORMAT_R16G16B16A16_SFLOAT`, `arrayLayers = ceil(channels / 4)`, `VK_IMAGE_TILING_OPTIMAL`
- Usage: `VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT`
- Create `VkImageView` with `viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY`, full `layerCount`
- Channel packing: channels 0-3 → layer 0, 4-7 → layer 1, etc. Last layer pads unused components with zero.

**A3.** Implement `DestroyFeatureImage()` replacing `DestroyFeatureBuffer()`:
- `vkDestroyImageView` then `vmaDestroyImage`. Reset struct to default.

**A4.** Update `Resize()` to allocate `FeatureImage` for all 10 feature maps:
- `buf0_a_`, `buf0_b_`, `skip0_`, `buf1_a_`, `buf1_b_`, `skip1_`, `buf2_a_`, `buf2_b_`, `concat1_`, `concat0_`
- Transition all to `VK_IMAGE_LAYOUT_GENERAL` at allocation time (one-shot command buffer or deferred barrier). Keep in GENERAL permanently.
- `reduction_buffer_` stays as storage buffer (1D reduction data, not spatial).

**A5.** Verify `MlDeviceDispatch` entries:
- VMA handles `vkCreateImage`/`vkDestroyImage` internally — no dispatch entries needed.
- `vkCmdCopyBuffer` for skip connections is removed (skip copy elimination).
- May need `vkCmdClearColorImage` for zero-init if `channels % 4 != 0` on last layer.

#### B. Shader Rewrites — `denoise/src/vulkan/shaders/`

**B1. `conv.comp`** — Full rewrite of data access pattern:
- Bindings 0, 1 change from `buffer { float16_t data[]; }` to `image2DArray` (RGBA16F)
- Binding 2 (weights) remains `buffer { float weights[]; }`
- Restructure outer loop from per-output-channel to per-output-layer (groups of 4):
  ```glsl
  for (uint og = 0; og < OUT_CHANNELS / 4; ++og) {
      vec4 accum = vec4(0.0);  // 4 output channels
      for (uint ic = 0; ic < IN_CHANNELS; ++ic) {
          uint in_layer = ic / 4;
          uint in_comp = ic % 4;
          for (int ky = -1; ky <= 1; ++ky) {
              for (int kx = -1; kx <= 1; ++kx) {
                  int sx = int(x) + kx, sy = int(y) + ky;
                  float val = 0.0;
                  if (sx >= 0 && sx < int(width) && sy >= 0 && sy < int(height))
                      val = imageLoad(feature_in, ivec3(sx, sy, in_layer))[in_comp];
                  // Accumulate into accum[0..3] using weights for oc = og*4+{0,1,2,3}
              }
          }
      }
      // Add bias for og*4+{0,1,2,3}
      imageStore(feature_out, ivec3(x, y, og), accum);
  }
  ```
- Keep explicit bounds checks for zero-padding (imageLoad with clamped coords would return border texels, not zero).
- Push constants unchanged: `{width, height}`

**B2. `encoder_input_conv.comp`** — Output binding change only:
- Input: 5 G-buffer storage images (bindings 0-4, unchanged)
- Output: binding 5 changes from storage buffer to `writeonly image2DArray`
- Weights: binding 6 storage buffer (unchanged)
- Restructure output loop to write in groups of 4 channels per `imageStore`

**B3. `output_conv.comp`** — Input binding change only:
- Input: binding 0 changes from storage buffer to `readonly image2DArray`
- Output: binding 1 single 2D storage image (unchanged)
- Weights: binding 2 storage buffer (unchanged)

**B4. `group_norm_reduce.comp`** — Image indexing rewrite:
- Binding 0 changes from storage buffer to `readonly image2DArray`
- Convert linear index to image coordinates:
  ```glsl
  uint spatial = i % hw;
  uint ch_local = i / hw;
  uint x = spatial % width;
  uint y = spatial / width;
  uint ch = base_channel + ch_local;
  uint layer = ch / 4;
  uint comp = ch % 4;
  float val = imageLoad(data, ivec3(x, y, layer))[comp];
  ```
- Binding 1 (reduction buffer) stays as storage buffer (1D data)
- Subgroup reduction algorithm unchanged

**B5. `group_norm_apply.comp`** — Readwrite image binding:
- Single `image2DArray` binding (no `readonly`/`writeonly` qualifier) for readwrite access
- Each thread owns pixel `(x, y)` exclusively — no data race
- Process per-layer (4 channels at a time):
  ```glsl
  for (uint layer = 0; layer < NUM_LAYERS; ++layer) {
      vec4 val = imageLoad(data, ivec3(x, y, layer));
      // For each component 0-3: normalize, apply gamma/beta, optional LeakyReLU
      // Handle last layer: only process valid components (channels % 4)
      imageStore(data, ivec3(x, y, layer), val);
  }
  ```
- Norm params buffer and reduction buffer stay as storage buffers

**B6. `downsample.comp`** — Image2DArray max pool:
- Input/output both become `image2DArray`
- Process per-layer: read 2×2 block (4 texels), take component-wise `max()`, write one output texel
- Clamp source coordinates to input dimensions for odd-sized inputs

**B7. `upsample_concat.comp`** — Image2DArray bilinear + concat:
- All 3 bindings (input, skip, output) become `image2DArray`
- Bilinear upsample: read 4 neighbors from input image, interpolate per-component per-layer
- Concatenation by layer offset: write upsampled layers at `[0..N-1]`, skip layers at `[N..N+M-1]`
- Output image has `ceil((IN_CHANNELS + SKIP_CHANNELS) / 4)` layers
- Spec constants: `IN_CHANNELS`, `SKIP_CHANNELS` (unchanged)

#### C. Descriptor & Pipeline Updates — `MlInference.cpp`

**C1.** Update descriptor set layouts — all feature map bindings change from `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` → `VK_DESCRIPTOR_TYPE_STORAGE_IMAGE`:

| Pipeline | Binding changes |
|---|---|
| Conv | 0 (input), 1 (output) → `STORAGE_IMAGE` |
| EncoderInputConv | 5 (output) → `STORAGE_IMAGE` |
| OutputConv | 0 (input) → `STORAGE_IMAGE` |
| GroupNormReduce | 0 (data) → `STORAGE_IMAGE` |
| GroupNormApply | 0 (data) → `STORAGE_IMAGE` |
| Downsample | 0 (input), 1 (output) → `STORAGE_IMAGE` |
| UpsampleConcat | 0 (input), 1 (skip), 2 (output) → `STORAGE_IMAGE` |

Weight, norm param, and reduction buffer bindings stay as `STORAGE_BUFFER`. Pool already has 384 `STORAGE_IMAGE` slots — no resize needed.

**C2.** Update descriptor set writes:
- `VkDescriptorBufferInfo` → `VkDescriptorImageInfo` with `imageView` from `FeatureImage`, `imageLayout = VK_IMAGE_LAYOUT_GENERAL`
- Update `DispatchConv`, `DispatchGroupNorm`, `DispatchDownsample`, `DispatchUpsampleConcat` signatures from `VkBuffer` → `const FeatureImage&`

**C3.** Update pipeline barriers:
- `InsertBufferBarrier()` → `InsertImageBarrier()` using `VkImageMemoryBarrier2`
- Same access masks: `VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT` / `VK_ACCESS_2_SHADER_STORAGE_READ_BIT`
- Image layout stays `VK_IMAGE_LAYOUT_GENERAL` (no transitions during inference)
- Barrier targets all subresources (`VK_IMAGE_ASPECT_COLOR_BIT`, all layers)

#### D. Dispatch Flow Updates — `MlInference.cpp` `Infer()`

**D1.** Eliminate skip copies by redirecting conv2 output to skip images:

```
BEFORE:                                  AFTER:
down0.conv2: buf0_a → buf0_b            down0.conv2: buf0_a → skip0
norm: buf0_b                             norm: skip0
copy: buf0_b → skip0                     (removed)
downsample: buf0_b → buf1_a             downsample: skip0 → buf1_a

down1.conv2: buf1_b → buf1_a            down1.conv2: buf1_b → skip1
norm: buf1_a                             norm: skip1
copy: buf1_a → skip1                     (removed)
downsample: buf1_a → buf2_a             downsample: skip1 → buf2_a
```

This removes 2× buffer copy commands per frame and eliminates a WAR hazard per level.

**D2.** Update all dispatch helper call sites from `VkBuffer` → `const FeatureImage&`. Push constants unchanged: `{width, height}`.

**D3.** Feature images stay in `VK_IMAGE_LAYOUT_GENERAL` permanently (transitioned at allocation in A4). No layout transitions needed in `Infer()`.

#### E. Test Updates — `tests/`

**E1.** Update `ml_inference_numerical_test.cpp`:
- Output readback unchanged (output_conv already writes to a 2D image, not a feature buffer)
- Golden reference data unchanged (PyTorch model unchanged)
- Add `[texture]` tag to existing golden reference test

**E2.** Update `ml_inference_test.cpp` integration tests:
- `"MlInference: feature map allocation at 256x256"` → verify `FeatureImage` creation (dimensions, layer counts, format)
- Other integration tests (weight upload, resize) adapted for `FeatureImage`

**E3.** Add GPU timestamp logging before/after inference for performance comparison.

#### F. Golden Reference — `tests/`

**F1.** Verify `tests/data/golden_ref.bin` remains valid (no regeneration needed — PyTorch model unchanged, final output is already a 2D image readback).

### Numerical Validation Tests

**Test: `[deni][numerical][golden][texture]` — Texture feature maps produce same output as PyTorch**

1. Load golden reference (same weights, same input as existing test)
2. Run GPU inference with texture-backed feature maps
3. Compare against PyTorch reference output
4. **Pass criteria:** RMSE < 0.01, max_abs_error < 0.05 (same as existing tolerance)

### Files to Modify

| File | Change |
|---|---|
| `denoise/src/vulkan/MlInference.h` | `FeatureBuffer` → `FeatureImage` struct, member variable types |
| `denoise/src/vulkan/MlInference.cpp` | Allocation, descriptors, barriers, dispatch flow (~1400 lines, extensive) |
| `denoise/src/vulkan/shaders/conv.comp` | Full rewrite: buffer → image2DArray, per-layer output loop |
| `denoise/src/vulkan/shaders/encoder_input_conv.comp` | Output binding: buffer → image2DArray |
| `denoise/src/vulkan/shaders/output_conv.comp` | Input binding: buffer → image2DArray |
| `denoise/src/vulkan/shaders/group_norm_reduce.comp` | Buffer indexing → imageLoad with coordinate conversion |
| `denoise/src/vulkan/shaders/group_norm_apply.comp` | Readwrite image2DArray, per-layer processing loop |
| `denoise/src/vulkan/shaders/downsample.comp` | Buffer → image2DArray max pool |
| `denoise/src/vulkan/shaders/upsample_concat.comp` | Buffer → image2DArray bilinear + layer-based concat |
| `tests/ml_inference_test.cpp` | Update allocation tests for FeatureImage |
| `tests/ml_inference_numerical_test.cpp` | Add `[texture]` tag, verify readback |

### Verification

1. All shaders compile without errors via `glslc --target-env=vulkan1.2`
2. `[deni][numerical][golden]` test passes with RMSE < 0.01
3. All `[deni][integration]` tests pass (feature allocation, weight upload, resize)
4. Zero Vulkan validation layer errors (`VK_LAYER_KHRONOS_validation`)
5. Visual check: denoised output in monti viewer identical to pre-change
6. GPU timestamps show measurable speedup (expect 1.3-2× on full-resolution layers)

---

## Phase T2: Depthwise Separable Convolutions

**Goal:** Replace standard 3×3 convolutions with depthwise separable convolutions in interior U-Net layers to reduce FLOP count by ~3-4×. Requires retraining the PyTorch model and regenerating the golden reference.

**Motivation:** A standard `Conv2d(C_in, C_out, 3×3)` performs $C_{in} \times C_{out} \times 9$ MADs per pixel. A depthwise separable convolution splits this into:
1. **Depthwise:** `Conv2d(C, C, 3×3, groups=C)` — 1 filter per channel, $C \times 9$ MADs
2. **Pointwise:** `Conv2d(C, C_out, 1×1)` — channel mixing, $C \times C_{out}$ MADs
3. **Total:** $C \times 9 + C \times C_{out}$ MADs vs $C_{in} \times C_{out} \times 9$ MADs

For a 32→32 layer: standard = 9,216 vs separable = 1,312 MADs — **7× reduction**. For the full U-Net, the savings are ~3-4× because the first and last layers (interfacing with raw data) keep standard convolutions.

**Quality trade-off:** Depthwise separable convolutions have less expressive power because the spatial and channel dimensions are decoupled. For denoising, this is less damaging than for recognition tasks because spatial filtering in a denoiser is mostly guided by the G-buffer (normals, depth provide explicit geometry), and channel mixing provides the learned combination. Expected PSNR drop: 0.5-1 dB, which is acceptable given the large performance gain.

**Retraining required.** The network architecture changes — must retrain from scratch and regenerate golden reference. This retraining uses F18's demodulated input representation (irradiance instead of raw radiance), so F18 must be completed first. The v2 model trains on demodulated inputs from the start, avoiding a redundant intermediate retraining step.

### Performance & Quality Estimates

- **Performance:** ~3.5× fewer FLOPS (122 GFLOPS → ~35 GFLOPS). With texture feature maps from T1, estimated inference time drops from 8-20ms to 3-8ms on RTX 4090.
- **Quality:** ~0.5-1 dB PSNR drop vs standard convolutions on validation set. Acceptable for real-time use, especially once temporal accumulation (T4) recovers the quality.

### Tasks

#### 1. PyTorch architecture changes — `training/deni_train/models/blocks.py`

Add `DepthwiseSeparableConvBlock`:

```python
class DepthwiseSeparableConvBlock(nn.Module):
    """Depthwise 3×3 + Pointwise 1×1 + GroupNorm + LeakyReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self._init_weights()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)
```

#### 2. Hybrid architecture — `training/deni_train/models/unet.py`

Create `DeniUNetV2` that uses standard `ConvBlock` for:
- First encoder conv (raw G-buffer → features): `ConvBlock(13, c)`
- Last decoder conv (features → pre-output): `ConvBlock(c, c)`
- Output projection: `Conv2d(c, 3, 1×1)` (unchanged, already pointwise)

And `DepthwiseSeparableConvBlock` for all other interior layers:
- Second conv in each encoder level
- Both convs in bottleneck
- Both convs in each decoder level (except last)

This preserves full expressiveness at the data interfaces while reducing compute in the interior.

Update `DeniUNet.__init__` to accept a `use_depthwise_separable=False` parameter so both architectures coexist. When `True`, interior `ConvBlock`s are replaced with `DepthwiseSeparableConvBlock`. The `forward()` method is unchanged — both block types have the same interface.

#### 3. Training config — `training/configs/v2_depthwise.yaml`

```yaml
model:
  in_channels: 13
  out_channels: 3
  base_channels: 16
  use_depthwise_separable: true

training:
  epochs: 150          # More epochs to compensate for reduced capacity
  learning_rate: 1.0e-4
  # ... rest same as default.yaml
```

#### 4. Retrain from scratch

```bash
python -m deni_train.train --config configs/v2_depthwise.yaml
```

Expected training time: ~20-45 minutes on RTX 4090 (small dataset, 150 epochs). Log PSNR/SSIM on validation split at each epoch. Compare final metrics against v1 (standard convolutions) to quantify the quality trade-off.

#### 5. New GLSL shader — `depthwise_conv.comp`

Implements depthwise 3×3 convolution (groups = channels):

```glsl
layout(set = 0, binding = 0, rgba16f) uniform readonly image2DArray feature_in;
layout(set = 0, binding = 1, rgba16f) uniform writeonly image2DArray feature_out;
layout(set = 0, binding = 2) readonly buffer WeightBuffer { float weights[]; };
// Weight layout: [CHANNELS][3][3] + bias[CHANNELS]

void main() {
    // Each thread processes one pixel, all channels
    for (uint ch = 0; ch < CHANNELS; ++ch) {
        float sum = 0.0;
        for (uint ky = 0; ky < 3; ++ky) {
            for (uint kx = 0; kx < 3; ++kx) {
                // imageLoad from feature_in at (sx, sy, ch/4)[ch%4]
                sum += input_val * weights[ch * 9 + ky * 3 + kx];
            }
        }
        sum += weights[CHANNELS * 9 + ch];  // bias
        // Write to temporary or directly to output
    }
}
```

The pointwise 1×1 convolution reuses the existing `conv.comp` shader with kernel_size=1 (or a dedicated `pointwise_conv.comp` that skips the 3×3 loop and only does the channel-mixing dot product). The pointwise shader is simpler and faster than the general conv:

```glsl
// pointwise_conv.comp — 1×1 convolution (channel mixing only)
for (uint oc = 0; oc < OUT_CHANNELS; ++oc) {
    float sum = 0.0;
    for (uint ic = 0; ic < IN_CHANNELS; ++ic) {
        sum += input[ic] * weights[oc * IN_CHANNELS + ic];
    }
    sum += bias[oc];
    output[oc] = sum;
}
```

#### 6. Update dispatch sequence — `MlInference.cpp`

For depthwise separable layers, dispatch sequence changes from:
```
conv.comp (3×3, C_in→C_out)  →  group_norm  →  activation
```
to:
```
depthwise_conv.comp (3×3, C→C)  →  pointwise_conv.comp (1×1, C→C_out)  →  group_norm  →  activation
```

Two dispatches instead of one per ConvBlock, but each dispatch is much cheaper. Total dispatch count increases from ~20 to ~30, but total FLOPS decrease ~3.5×.

Detect architecture from weight file: if `down0.conv2.depthwise.weight` exists in the `.denimodel` file, the model uses depthwise separable convolutions. This auto-detection allows the inference engine to support both v1 and v2 models without a configuration flag.

#### 7. Weight export update — `training/scripts/export_weights.py`

No changes needed — `export_weights.py` already exports the full `state_dict()` with whatever layer names PyTorch generates. The depthwise layers will produce keys like `down0.conv2.depthwise.weight`, `down0.conv2.pointwise.weight`, etc.

#### 8. Golden reference regeneration

Update `tests/generate_golden_reference.py` to use `DeniUNetV2` with `use_depthwise_separable=True`. Regenerate `tests/data/golden_ref.bin`.

### Numerical Validation Tests

**Test: `[deni][numerical][golden]` — Updated golden reference matches GPU output**

Same test as before, but with v2 model weights. Pass criteria: RMSE < 0.01, max_abs_error < 0.05.

**Test: `[deni][numerical][depthwise]` — Depthwise separable conv matches PyTorch**

1. Create a standalone test with a single depthwise separable ConvBlock (known weights)
2. Run through GPU shader pipeline (depthwise_conv + pointwise_conv + group_norm)
3. Compare against PyTorch `DepthwiseSeparableConvBlock` output
4. Pass criteria: RMSE < 0.005 (tighter than full model, single layer)

**Test: `[deni][quality][v1_vs_v2]` — Quality regression check**

1. Run both v1 and v2 models on a shared test image
2. Compute PSNR of each against ground truth
3. **Pass criteria:** v2 PSNR >= v1 PSNR - 1.5 dB (allows up to 1.5 dB regression)

### Verification
- v2 model trains successfully, loss converges
- Quality evaluation: v2 PSNR within 1.5 dB of v1 on validation set
- All `[deni][numerical][golden]` tests pass with v2 weights
- Depthwise separable shader produces correct output
- GPU timestamp shows ~2-3× speedup vs T1 baseline

---

## Phase T3: Motion Reprojection Infrastructure

**Goal:** Implement the GPU infrastructure for reprojecting the previous frame's denoised output to the current frame using motion vectors. This phase adds no ML changes — it creates the foundation that the temporal residual network (T4/T5) will build on.

**Motivation:** Temporal denoising requires comparing the current noisy frame against a reprojected version of the previous clean output. Motion reprojection warps each pixel from the previous frame to its new position using screen-space motion vectors. The result is an initial estimate of the current frame that is clean (from the previous denoiser output) but may have errors in disoccluded regions (newly visible areas) and fast-moving objects.

**No retraining required.** This phase only adds GPU infrastructure (a reprojection shader and frame history management). The ML model is not modified.

### Performance & Quality Estimates

- **Performance:** Adds ~0.1-0.2ms for the reprojection pass (single fullscreen dispatch at output resolution). Negligible.
- **Quality:** No change — reprojection output is not used by the denoiser yet (that happens in T5).

### Tasks

#### 1. Frame history management — `MlInference.h/cpp`

Add frame history storage:

```cpp
struct FrameHistory {
    FeatureImage denoised_output;     // Previous frame's denoised RGB (RGBA16F, 2D)
    FeatureImage reprojected;         // Warped previous output (RGBA16F, 2D)
    FeatureImage disocclusion_mask;   // Binary mask (R16F, 2D): 1.0 = valid, 0.0 = disoccluded
    bool valid = false;               // False on first frame or after reset
};

FrameHistory frame_history_;
```

Allocate these images in `Resize()`. When resolution changes, set `frame_history_.valid = false` (forces full denoise on next frame).

#### 2. Reprojection shader — `reproject.comp`

```glsl
#version 460
layout(local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
};

layout(set = 0, binding = 0, rg16f) uniform readonly image2D motion_vectors;  // Current frame MV
layout(set = 0, binding = 1, rgba16f) uniform readonly image2D prev_output;   // Previous denoised
layout(set = 0, binding = 2, r16f) uniform readonly image2D prev_depth;       // Previous linear depth
layout(set = 0, binding = 3, r16f) uniform readonly image2D curr_depth;       // Current linear depth
layout(set = 0, binding = 4, rgba16f) uniform writeonly image2D reprojected;  // Output: warped image
layout(set = 0, binding = 5, r16f) uniform writeonly image2D disocclusion;    // Output: validity mask

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (pos.x >= int(width) || pos.y >= int(height)) return;

    // Motion vector: screen-space displacement (pixels)
    vec2 mv = imageLoad(motion_vectors, pos).rg;

    // Source position in previous frame
    vec2 src = vec2(pos) + mv + 0.5;  // +0.5 for pixel center
    ivec2 src_pos = ivec2(floor(src));

    // Bounds check
    if (src_pos.x < 0 || src_pos.x >= int(width) ||
        src_pos.y < 0 || src_pos.y >= int(height)) {
        imageStore(reprojected, pos, vec4(0.0));
        imageStore(disocclusion, pos, vec4(0.0));  // Disoccluded
        return;
    }

    // Bilinear fetch from previous output
    // (Could use sampler2D for hardware bilinear, but imageLoad
    //  gives us explicit control for the depth test below)
    vec4 prev_color = imageLoad(prev_output, src_pos);

    // Depth-based disocclusion detection
    float prev_d = imageLoad(prev_depth, src_pos).r;
    float curr_d = imageLoad(curr_depth, pos).r;
    float depth_ratio = (prev_d > 0.0) ? abs(curr_d - prev_d) / max(prev_d, 1e-6) : 1.0;
    float valid = (depth_ratio < 0.1) ? 1.0 : 0.0;  // 10% depth tolerance

    imageStore(reprojected, pos, prev_color);
    imageStore(disocclusion, pos, vec4(valid));
}
```

Depth threshold (10%) is a tuning parameter — can be exposed as a push constant later.

#### 3. Depth history management

To perform depth-based disocclusion detection, we need the previous frame's linear depth. Add to `FrameHistory`:

```cpp
FeatureImage prev_depth;   // Copy of previous frame's linear depth
```

At the end of each `Infer()` call, copy the current frame's depth image to `prev_depth` (via `vkCmdCopyImage` or a simple fullscreen copy shader). This is cheap — depth is a single R16F image.

Also copy the current denoised output to `frame_history_.denoised_output` after the output conv writes it. This becomes the "previous output" for the next frame's reprojection.

#### 4. Wire into `Infer()` dispatch sequence

At the start of `Infer()`, before the U-Net dispatches:

```cpp
if (frame_history_.valid) {
    DispatchReproject(cmd, input.motion_vectors, input.linear_depth);
    InsertImageBarrier(cmd, frame_history_.reprojected);
    InsertImageBarrier(cmd, frame_history_.disocclusion_mask);
}
```

At the end of `Infer()`, after the output conv:

```cpp
// Save current output and depth for next frame
CopyImageToImage(cmd, output_image, frame_history_.denoised_output);
CopyImageToImage(cmd, input.linear_depth, frame_history_.prev_depth);
frame_history_.valid = true;
```

The reprojected image and disocclusion mask are computed but not consumed yet — they will be wired into the temporal residual network in T5.

#### 5. Reset handling

Add `DenoiserInput::reset_accumulation` handling: when `true`, set `frame_history_.valid = false`. This is already provided by `monti_view` on camera movement, scene changes, etc.

### Numerical Validation Tests

**Test: `[deni][temporal][reproject_identity]` — Static camera produces identity reprojection**

1. Set motion vectors to zero everywhere
2. Set previous output to a known pattern (e.g., horizontal gradient)
3. Run reprojection
4. **Pass criteria:** Reprojected output exactly matches previous output; disocclusion mask = 1.0 everywhere

**Test: `[deni][temporal][reproject_shift]` — Known motion produces correct warp**

1. Set motion vectors to uniform (5, 3) displacement
2. Set previous output to a checkered pattern
3. Run reprojection
4. **Pass criteria:** Reprojected output is shifted by (-5, -3) relative to previous; border pixels have disocclusion = 0.0

**Test: `[deni][temporal][reproject_disocclusion]` — Depth discontinuity detected**

1. Set current depth to a value much larger than previous depth at certain pixels
2. Run reprojection
3. **Pass criteria:** Pixels with >10% depth ratio have disocclusion = 0.0

### Verification
- All reprojection tests pass
- `Infer()` runs without Vulkan validation errors
- Frame history is correctly allocated and populated
- GPU timing: reprojection pass adds <0.3ms
- No existing `[deni][numerical][golden]` tests are broken (the U-Net path is unchanged)

---

## Phase T4: Temporal Residual Network — Training

**Goal:** Design and train the temporal residual network in PyTorch. The network takes the reprojected previous output, disocclusion mask, current noisy input, and auxiliary G-buffer channels as input, and outputs a correction delta plus a per-pixel blend weight. This phase handles only the PyTorch side — GPU inference is T5.

**Motivation:** Instead of denoising from scratch, the temporal residual network learns to:
1. **Preserve** clean reprojected pixels where the previous frame is valid (blend_weight ≈ 0)
2. **Correct** ghosting/lag artifacts where motion compensation is imperfect
3. **Fill in** disoccluded regions using only the current frame's noisy input (blend_weight ≈ 1)
4. **Refine** noise by blending current samples with temporal history

This is fundamentally easier than full denoising — the network only needs to handle the residual error, not reconstruct the entire image. A smaller, faster network suffices.

**Retraining required.** This is a new architecture — requires sequential training data and training from scratch.

### Performance & Quality Estimates

- **Performance (training only):** No GPU inference change in this phase. Training produces the v3 model weights.
- **Quality:** Expected 2-4 dB PSNR improvement over v2 single-frame on sequences with moderate motion. Much better temporal stability (less flicker between frames).

### Tasks

#### 1. Sequential training data generation — `training/scripts/generate_viewpoints.py` + `generate_training_data.py`

Temporal training requires ordered frame sequences. Rather than a separate script, the existing pipeline is extended in two steps:

**Step 1 — Camera path generation in `generate_viewpoints.py`**

Replace the old random-variation approach with structured path generation driven by a new `training/configs/scenes.json` config. Each manual seed viewpoint is amplified into one path per path type — structured amplification instead of random strategy selection.

`scenes.json` makes the implicit scene-name → seed-file mapping explicit:

```json
{
  "version": 1,
  "defaults": {
    "sequence_length": 8,
    "max_displacement": 2.0
  },
  "scenes": [
    {
      "name": "Sponza",
      "scene_type": "interior",
      "scene": "scenes/khronos/Sponza/Sponza.gltf",
      "seeds": "viewpoints/manual/Sponza.json"
    },
    {
      "name": "DamagedHelmet",
      "scene_type": "object",
      "scene": "scenes/khronos/DamagedHelmet.glb",
      "seeds": "viewpoints/manual/DamagedHelmet.json"
    }
  ]
}
```

All paths are relative to `training/`. Per-scene fields override `defaults`. Scene type (`interior` vs `object`) controls which path algorithms are permitted. Unlisted scenes are an error — `scenes.json` is required.

**Path algorithms** (each seed generates one path per permitted type):

| Type | Motion | Interior | Object |
|---|---|---|---|
| `strafe` | position + target translated perpendicular to view direction | ✓ | ✓ |
| `dolly` | position + target translated along view direction | ✓ | ✓ |
| `zoom` | FOV sweep, position + target fixed | ✓ | ✓ |
| `orbit` | spherical sweep around target | — | ✓ |
| `handheld` | per-frame correlated micro-jitter (Halton sequence, position only) | ✓ | ✓ |

The random element within each path type (displacement direction, sweep magnitude) is seeded by `scene_name + seed_index + path_type` for reproducibility. Total paths per scene ≈ `len(seeds) × len(path_types)`.

**Safety:** displacement is bounded by `max_displacement` (distance from seed position). A degenerate-path guard rejects any frame where `dist(position, target) < 0.3 × seed_dist`. Invalid rendered viewpoints are caught downstream by `remove_invalid_viewpoints.py`.

**Viewpoint format** — the existing viewpoint JSON gains two fields and drops `id`:

```json
{
  "path_id": "a1b2c3d4",
  "frame": 0,
  "position": [...],
  "target": [...],
  "fov": 60.0,
  "exposure": 0.0,
  "environment": "..."
}
```

`path_id` groups frames into ordered sequences. `frame` defines render order within the path (0-indexed). The old per-viewpoint `id` field is removed — `{path_id}_{frame:04d}` serves as the unique render identifier. All frames of the same path share the same `environment`/`lights`/`exposure` assignment.

**Step 2 — Sequential rendering in `generate_training_data.py`**

Extend the existing script to detect and handle paths. Group viewpoints by `path_id`, sort by `frame`, and render in order so that `monti_datagen` sees consecutive camera positions and motion vectors reflect the camera motion between frames.

```bash
python scripts/generate_training_data.py \
    --monti-datagen build/app/datagen/Release/monti_datagen.exe \
    --config training/configs/scenes.json \
    --viewpoints-dir training/viewpoints/ \
    --output training_data/
```

**`monti_datagen` changes required:** When frames belonging to the same `path_id` are rendered sequentially, the renderer must maintain temporal state between frames (previous camera transform, motion vector computation). Add a `--sequence-start` / `--sequence-continue` flag pair, or accept the full path's frame list in one call.

**Data format:** For each path, frames are stored flat within the per-scene directory:

```
training_data/
  Sponza/
    a1b2c3d4_0000_input.exr
    a1b2c3d4_0000_target.exr
    a1b2c3d4_0001_input.exr
    a1b2c3d4_0001_target.exr
    ...
    e5f6g7h8_0000_input.exr    ← different path, same scene
    ...
```

The temporal dataset loader identifies paths by grouping on the 8-hex prefix and pairs consecutive frames naturally.

#### 2. Temporal dataset loader — `training/deni_train/data/temporal_dataset.py`

```python
class TemporalDataset(Dataset):
    """Loads pairs of consecutive frames for temporal residual training.
    
    Discovers paths by grouping EXR files on their 8-hex path_id prefix, then
    pairs consecutive frames (frame N-1 target → frame N input/target).
    
    Each sample returns:
        prev_target: Previous frame's ground truth (serves as "previous denoised output")
        curr_input:  Current frame's 13-channel noisy G-buffer
        curr_target: Current frame's ground truth (supervision signal)
        motion_vectors: Current frame's motion vectors (from curr_input channels 11-12)
    
    During training, the reprojection is done in PyTorch (matching the GPU shader)
    so the network sees the same reprojection artifacts it will encounter at inference.
    """
    
    def __init__(self, data_dir, transform=None):
        # Glob *_0000_input.exr to find path starts; scan consecutive frames per path_id
        ...
    
    def __getitem__(self, idx):
        # Load frame N-1 target (prev clean) and frame N input (curr noisy) + target
        prev_target = load_frame(self.pairs[idx].prev_target_path)  # 3ch RGB
        curr_input = load_frame(self.pairs[idx].curr_input_path)    # 13ch G-buffer
        curr_target = load_frame(self.pairs[idx].curr_target_path)  # 3ch RGB
        
        # Apply reprojection in PyTorch (simulates GPU reproject.comp)
        motion_vectors = curr_input[11:13]  # channels 11-12
        reprojected, disocclusion = self.reproject(prev_target, motion_vectors, 
                                                    curr_depth, prev_depth)
        
        return {
            'reprojected': reprojected,       # 3ch warped previous clean
            'disocclusion': disocclusion,     # 1ch binary mask
            'noisy_input': curr_input[:6],    # 6ch noisy diffuse + specular
            'normals': curr_input[6:10],       # 4ch normals + roughness
            'depth': curr_input[10:11],        # 1ch depth
            'target': curr_target              # 3ch ground truth
        }
```

#### 3. Temporal residual architecture — `training/deni_train/models/temporal_unet.py`

> **Note:** This architecture assumes F18 (albedo demodulation) is complete. The noisy radiance inputs are demodulated irradiance (radiance / albedo), not raw radiance. The network denoises in demodulated space, and albedo remodulation happens in the output shader (T5). This means the network operates on a smoother, lower-frequency signal — easier to denoise and more amenable to temporal accumulation.

```python
class DeniTemporalResidualNet(nn.Module):
    """Temporal residual denoiser (operates in demodulated irradiance space).
    
    Inputs (14 channels total):
        - reprojected: 3ch (warped previous clean output, in demodulated space)
        - disocclusion_mask: 1ch (0=invalid, 1=valid)
        - noisy_irradiance: 6ch (current demodulated diffuse + specular irradiance)
        - world_normals: 3ch (XYZ)
        - roughness: 1ch
    
    Outputs (4 channels):
        - correction_delta: 3ch (added to reprojected to get final output)
        - blend_weight: 1ch (sigmoid, 0=use reprojected, 1=use noisy+correction)
    
    Final output (demodulated) = reprojected + blend_weight * correction_delta
    Remodulation (in output shader): final_rgb = output_d * albedo_d + output_s * albedo_s
    """
    
    def __init__(self, base_channels=8):
        super().__init__()
        c = base_channels   # 8 (much smaller than v1/v2's 16)
        
        # 2-level U-Net (not 3 — smaller network for residual task)
        self.down0 = DownBlock(14, c, use_depthwise_separable=True)
        self.bottleneck1 = DepthwiseSeparableConvBlock(c, c * 2)
        self.bottleneck2 = DepthwiseSeparableConvBlock(c * 2, c * 2)
        self.up0 = UpBlock(c * 2, c, c, use_depthwise_separable=True)
        self.out_conv = nn.Conv2d(c, 4, kernel_size=1)  # 3ch delta + 1ch weight
    
    def forward(self, reprojected, disocclusion, noisy, normals, roughness):
        x = torch.cat([reprojected, disocclusion, noisy, normals, roughness], dim=1)
        
        pooled, skip0 = self.down0(x)
        b = self.bottleneck1(pooled)
        b = self.bottleneck2(b)
        up = self.up0(b, skip0)
        
        out = self.out_conv(up)
        delta = out[:, :3]                    # Correction
        weight = torch.sigmoid(out[:, 3:4])   # Blend weight [0, 1]
        
        # For disoccluded pixels, force blend_weight toward 1.0
        # (network should learn this, but this provides a strong prior)
        weight = torch.max(weight, 1.0 - disocclusion)
        
        return reprojected + weight * delta
```

**Parameter count:** ~15-20K parameters (2-level U-Net, base_channels=8, depthwise separable). About 6-8× smaller than v2.

**GFLOPS estimate:** At 1080p: ~18 GFLOPS (vs ~35 for v2, ~122 for v1).

#### 4. Temporal training loop modifications — `training/deni_train/train_temporal.py`

Key differences from single-frame training:
- **First frame handling:** For the first frame in each sequence (no previous output available), fall back to single-frame denoiser output (run v2 model) as the "previous clean" input. This trains the temporal network to handle the cold-start case.
- **Temporal stability loss:** Add a term penalizing flicker between consecutive outputs:
  ```python
  # Warp current output to next frame, compare against next frame's output
  temporal_loss = L1(warp(output_t, motion_t_to_t1), output_t1.detach())
  total_loss = lambda_l1 * l1_loss + lambda_perceptual * perceptual_loss 
             + lambda_temporal * temporal_loss
  ```
  `lambda_temporal = 0.5` — strong enough to enforce stability without suppressing legitimate changes.
- **Sequence batching:** DataLoader returns batches of frame pairs from the same sequence.

#### 5. Training config — `training/configs/v3_temporal.yaml`

```yaml
model:
  type: temporal_residual
  base_channels: 8
  use_depthwise_separable: true

data:
  data_dir: "../training_data"
  sequence_length: 8
  crop_size: 256
  batch_size: 4     # Smaller batch (2 frames per sample = 2× memory)

loss:
  lambda_l1: 1.0
  lambda_perceptual: 0.1
  lambda_temporal: 0.5

training:
  epochs: 200
  learning_rate: 1.0e-4
```

#### 6. Export and evaluate

Export v3 weights: `python scripts/export_weights.py --checkpoint configs/checkpoints/v3_temporal_best.pt --output models/deni_v3_temporal.denimodel`

Evaluate on held-out temporal sequences, measuring:
- Per-frame PSNR (should exceed v2 by 2-4 dB on average)
- Temporal stability (warp error between consecutive outputs < threshold)
- Disocclusion quality (PSNR specifically in disoccluded regions)

### Numerical Validation Tests (PyTorch-side only, no GPU tests in this phase)

**Test: `test_temporal_model.py` — Architecture shape validation**

1. Create `DeniTemporalResidualNet(base_channels=8)`
2. Forward pass with random inputs: reprojected(B,3,256,256), disocclusion(B,1,256,256), noisy(B,6,256,256), normals(B,3,256,256), roughness(B,1,256,256)
3. **Pass criteria:** Output shape = (B, 3, 256, 256); parameter count in 15K-25K range

**Test: `test_temporal_reproject.py` — PyTorch reprojection matches reference**

1. Create a known image + motion vectors
2. Run PyTorch reprojection
3. **Pass criteria:** Output matches expected shifted image, disocclusion mask correct at boundaries

**Test: `test_temporal_training.py` — Training loop converges**

1. Overfit on 2 synthetic temporal sequences (4 frames each)
2. Train for 100 steps
3. **Pass criteria:** Loss decreases to < 0.05

### Verification
- v3 model trains on temporal data, loss converges
- Evaluation shows PSNR improvement over v2 on temporal sequences
- Temporal stability metric improves (less flicker)
- All PyTorch-side tests pass
- Weight export produces valid `.denimodel` file

---

## Phase T5: Temporal Residual Network — Inference

**Goal:** Implement the temporal residual network's GPU inference, replacing the single-frame U-Net with the new temporal pipeline. Wire the reprojection output (T3) into the residual network input.

**Motivation:** This is where the quality and performance gains materialize on the GPU. The temporal residual network is ~6-8× smaller than the single-frame v2 model, so inference is dramatically faster. And quality improves because each frame leverages the accumulated history from all previous frames.

**No retraining required.** Uses v3 weights from T4.

### Performance & Quality Estimates

- **Performance:** ~18 GFLOPS at 1080p. With texture feature maps: estimated 2-5ms on RTX 4090. With cooperative matrix (future): <0.5ms.
- **Quality:** Matches or exceeds single-frame v1 quality (120K params) despite having only 15-20K params, because temporal accumulation provides information that a larger single-frame network can't access.

### Tasks

#### 1. New shaders for temporal input

**`temporal_encoder_input_conv.comp`** — Replaces `encoder_input_conv.comp` for the temporal model. Reads 14 input channels from:
- Reprojected previous output: `image2D` (3ch, from T3's reprojection shader)
- Disocclusion mask: `image2D` (1ch, from T3's reprojection shader)
- Noisy diffuse: `image2D` (3ch, from G-buffer)
- Noisy specular: `image2D` (3ch, from G-buffer)
- World normals: `image2D` (3ch + roughness = 4ch from G-buffer)

Outputs to first feature level `image2DArray` (base_channels layers).

#### 2. Temporal output shader — `temporal_output_conv.comp`

Modified output conv that produces 4 channels (3ch delta + 1ch blend weight), applies the temporal blending, and remodulates with albedo (F18):

```glsl
// Final demodulated output = reprojected + sigmoid(weight) * delta
vec3 delta = vec3(sum_r, sum_g, sum_b);   // From 1×1 conv
float weight = 1.0 / (1.0 + exp(-sum_w)); // Sigmoid

// Force weight=1 for disoccluded pixels
float disocc = imageLoad(disocclusion_mask, pos).r;
weight = max(weight, 1.0 - disocc);

vec3 reprojected = imageLoad(reprojected_prev, pos).rgb;
vec3 denoised_irradiance = reprojected + weight * delta;

// Albedo remodulation (F18): convert demodulated irradiance back to radiance
// The network outputs denoised irradiance (diffuse+specular combined in demodulated space).
// Remodulate with albedo to restore texture detail.
// For separate diffuse/specular outputs (6ch), remodulate each lobe independently:
//   final = denoised_d * albedo_d + denoised_s * albedo_s
// For combined 3ch output, use combined albedo approximation.
vec3 albedo_d = imageLoad(diffuse_albedo, pos).rgb;
vec3 albedo_s = imageLoad(specular_albedo, pos).rgb;
// Split denoised irradiance proportionally (or use 6ch output if available)
vec3 final = denoised_irradiance * max(albedo_d + albedo_s, vec3(0.001));
imageStore(output_image, pos, vec4(final, 1.0));
```

This fuses the blending and remodulation into the output shader — no extra dispatch needed. The exact remodulation strategy (combined vs. per-lobe) is determined by F18's implementation.

#### 3. Update `Infer()` dispatch sequence

The new sequence for temporal inference:

```cpp
void MlInference::Infer(VkCommandBuffer cmd, const DenoiserInput& input,
                         VkImageView output_view) {
    // 1. Reproject previous output (if valid)
    if (frame_history_.valid) {
        DispatchReproject(cmd, input);
    }
    
    // 2. Temporal residual U-Net (2-level, smaller)
    //    Encoder level 0
    DispatchTemporalEncoderInput(cmd, input);  // 14ch → base_channels
    DispatchGroupNorm(cmd, ...);
    DispatchDepthwiseConv(cmd, ...);
    DispatchPointwiseConv(cmd, ...);
    DispatchGroupNorm(cmd, ...);
    // Save skip0, downsample
    
    //    Bottleneck (at H/2 × W/2 — only 2 levels)
    DispatchDepthwiseConv(cmd, ...);
    DispatchPointwiseConv(cmd, ...);
    DispatchGroupNorm(cmd, ...);
    DispatchDepthwiseConv(cmd, ...);
    DispatchPointwiseConv(cmd, ...);
    DispatchGroupNorm(cmd, ...);
    
    //    Decoder level 0
    DispatchUpsampleConcat(cmd, ...);
    DispatchDepthwiseConv(cmd, ...);
    DispatchPointwiseConv(cmd, ...);
    DispatchGroupNorm(cmd, ...);
    
    //    Temporal output (delta + blend + fuse)
    DispatchTemporalOutputConv(cmd, ...);
    
    // 3. Save current output for next frame
    CopyImage(cmd, output_image, frame_history_.denoised_output);
    CopyImage(cmd, input.linear_depth, frame_history_.prev_depth);
    frame_history_.valid = true;
}
```

Total dispatches: ~14 (vs ~20 for single-frame v1). Each dispatch does less work due to smaller channels and depthwise separable convolutions.

#### 4. Fallback for first frame

When `frame_history_.valid == false` (first frame, resolution change, accumulation reset):
- Set reprojected to black (zero)
- Set disocclusion mask to 0.0 everywhere (all pixels "disoccluded")
- The network's forced `weight = max(weight, 1.0 - disocclusion)` ensures weight=1 for all pixels
- The output becomes `0 + 1.0 * delta = delta`, i.e., the network produces the output directly from the noisy input, behaving as a single-frame denoiser

This is exactly what the network was trained to handle (first-frame fallback in T4).

#### 5. Auto-detect model version

The weight loader should detect which model type is loaded and configure the dispatch path accordingly:

```cpp
enum class ModelVersion { kV1_SingleFrame, kV2_Depthwise, kV3_Temporal };

// Detected from weight layer names:
// - Contains "down1" → 3-level U-Net (v1 or v2)
// - Contains "depthwise" → v2 or v3
// - Does NOT contain "down1" + contains "depthwise" → v3 (2-level temporal)
```

This allows loading any model version without manual configuration.

### Numerical Validation Tests

**Test: `[deni][numerical][golden]` — v3 temporal model matches PyTorch**

1. Generate golden reference with v3 temporal model:
   - Known input: reprojected + disocclusion + noisy G-buffer
   - Run PyTorch inference, store expected output
2. Run GPU inference with same inputs
3. **Pass criteria:** RMSE < 0.01, max_abs_error < 0.05

**Test: `[deni][temporal][first_frame]` — First frame produces reasonable output**

1. Run inference with `frame_history_.valid = false` 
2. Verify output is non-zero and non-NaN
3. **Pass criteria:** Output PSNR > 15 dB vs ground truth (very loose — just ensures the fallback produces something reasonable)

**Test: `[deni][temporal][accumulation]` — Quality improves over frames**

1. Run 5 frames with static camera (same input, different noise seed if possible)
2. Measure PSNR of each frame
3. **Pass criteria:** Frame 5 PSNR > Frame 1 PSNR (temporal accumulation is helping)

**Test: `[deni][temporal][reset]` — Accumulation reset works**

1. Run 3 frames normally
2. Set `reset_accumulation = true`
3. Run frame 4
4. **Pass criteria:** Frame 4 output is valid (no NaN, reasonable PSNR)

### Verification
- All `[deni][numerical][golden]` tests pass with v3 model
- Temporal accumulation visibly improves quality over successive frames
- Reset handling works correctly
- GPU timestamp confirms <5ms inference at 1080p on RTX 4090
- No Vulkan validation errors

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
- Noisy input: 13ch at 960×540 (low-res)
- Reference target: 3ch at 1920×1080 (high-res)
- Motion vectors: 2ch at 960×540

#### 2. Super-resolution architecture — `training/deni_train/models/superres_unet.py`

Extend the temporal residual network with a learned 2× upsampler tail:

```python
class DeniSuperResNet(nn.Module):
    """Temporal residual denoiser + 2× learned upsampler.
    
    Inputs (same as temporal residual: 14ch at low resolution):
        - reprojected: 3ch (warped previous *high-res* output, downsampled to low-res)
        - disocclusion_mask: 1ch
        - noisy_radiance: 6ch
        - normals + roughness: 4ch
    
    Outputs:
        - Denoised RGB at 2× input resolution (high-res)
    """
    
    def __init__(self, base_channels=8):
        super().__init__()
        c = base_channels
        
        # Temporal residual core (same as v3, operates at low res)
        self.temporal_core = DeniTemporalResidualNet(base_channels=c)
        
        # Learned 2× upsampler (operates at low res, outputs high res)
        # PixelShuffle upsampling: 4c channels → c channels at 2× resolution
        self.upsample_pre = DepthwiseSeparableConvBlock(3, c * 4)
        self.pixel_shuffle = nn.PixelShuffle(2)  # (B, 4c, H, W) → (B, c, 2H, 2W)
        self.upsample_refine = DepthwiseSeparableConvBlock(c, c)
        self.upsample_out = nn.Conv2d(c, 3, kernel_size=1)  # Linear, no activation
    
    def forward(self, reprojected, disocclusion, noisy, normals, roughness):
        # Low-res denoised output (3ch, H×W)
        denoised_lr = self.temporal_core(reprojected, disocclusion, noisy, 
                                          normals, roughness)
        
        # Upsample 2× to high res (3ch, 2H×2W)
        up = self.upsample_pre(denoised_lr)   # → (4c)ch
        up = self.pixel_shuffle(up)            # → c ch at 2× res
        up = self.upsample_refine(up)          # → c ch
        up = self.upsample_out(up)             # → 3ch (high-res RGB)
        
        return up
```

**Parameter count:** ~25-30K (temporal core ~15-20K + upsampler ~10K).

**PixelShuffle why:** PixelShuffle (sub-pixel convolution) is more efficient than transposed convolution for learned upsampling. It avoids checkerboard artifacts that plague transposed convolution and naturally distributes spatial information across a 2×2 output neighborhood.

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
3. **Pass criteria:** Output shape = (B, 3, 256, 256); parameter count in 25K-35K range

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
v1 (F11-3):  G-buffer 13ch ──► 3-level U-Net (120K) ──► RGB 3ch
                              16→32→64 standard conv
                              ~122 GFLOPS, single-frame

v2 (T2):     G-buffer 13ch ──► 3-level U-Net (35K) ──► RGB 3ch
                              16→32→64 depthwise separable
                              ~35 GFLOPS, single-frame

v3 (T5):     [Reprojected 3ch + Disocclusion 1ch + Noisy 10ch]
                       ──► 2-level Temporal Residual Net (20K) ──► RGB 3ch
                       8→16 depthwise separable
                       ~18 GFLOPS, temporal accumulation

v4 (T7):     [Low-res reprojected + noisy] @ 540p
                       ──► Temporal Residual (20K) @ 540p
                       ──► Learned 2× Upsample (10K) → 1080p RGB
                       ~12 GFLOPS, temporal + super-resolution
```

### Performance Summary

| Phase | Model | RTX 4090 (1080p) | Adreno 750 (720p→1080p) | Quality |
|---|---|---|---|---|
| Baseline (F11-3) | v1, 120K params | 15-40ms | — | 1.0× |
| T1 (textures) | v1 | 8-20ms | — | 1.0× |
| T1+T2 (depthwise) | v2, 35K params | 3-8ms | — | 0.95× |
| T3+T4+T5 (temporal) | v3, 20K params | 2-5ms | — | 1.3× |
| T6+T7 (super-res) | v4, 30K params | 1-3ms | — | 1.2× |
| T8 (mobile) | v4 | 1-3ms | 2-4ms | 1.2× |

### Training Pipeline Summary

| Phase | Retrain? | Dataset Change | Training Time (RTX 4090) |
|---|---|---|---|
| T1 | No | — | — |
| T2 | Yes | Same single-frame data | ~30 min |
| T3 | No | — | — |
| T4 | Yes | New temporal sequences | ~1-2 hours |
| T5 | No | — | — |
| T6 | Yes | New low-res/high-res pairs | ~2-3 hours |
| T7 | No | — | — |
| T8 | No | — | — |

### Test Coverage Summary

| Phase | New Tests | Tags |
|---|---|---|
| T1 | buffer_vs_texture match, golden | `[deni][numerical][texture]` |
| T2 | depthwise shader, golden, quality regression | `[deni][numerical][depthwise]`, `[deni][quality]` |
| T3 | reproject identity/shift/disocclusion | `[deni][temporal][reproject_*]` |
| T4 | PyTorch model shape, reprojection, training convergence | `test_temporal_*.py` |
| T5 | golden, first frame, accumulation, reset | `[deni][temporal][*]` |
| T6 | PyTorch model shape, pixel shuffle, training convergence | `test_superres_*.py` |
| T7 | golden, pixel_shuffle shader, scale modes | `[deni][superres][*]` |
| T8 | fragment_vs_compute match, fragment golden, perf | `[deni][mobile][*]` |

### Key Dependencies

```
F11-3 ✅ (single-frame inference working)
  │
  ├─ T1 (texture feature maps)
  │   └─ T2 (depthwise separable convolutions) ──► retrain v2
  │       └─ T3 (reprojection infrastructure)
  │           └─ T4 (temporal training) ──► retrain v3
  │               └─ T5 (temporal inference)
  │                   └─ T6 (super-res training) ──► retrain v4
  │                       └─ T7 (super-res inference)
  │                           └─ T8 (mobile fragment backend)
```

All phases are strictly sequential. Each inference phase (T1, T2, T3, T5, T7, T8) requires the previous phase's code. Each training phase (T4, T6) requires the previous inference phase for data generation or baseline comparison. T2 requires retraining before its inference code changes take effect.
