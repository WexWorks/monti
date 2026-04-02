# Temporal Denoiser — Performance & Quality Optimization Plan

> **Purpose:** Incrementally transform the existing single-frame ML denoiser into a high-performance temporal denoiser with super-resolution, targeting both desktop (Vulkan compute) and mobile (fragment shader, TBDR) deployment. Each phase is sized for a single Copilot Claude Opus 4.6 session and includes numerical validation tests, training pipeline changes, and performance/quality estimates.
>
> **Starting point:** Phase F18 ✅ — single-frame DeniUNet with albedo demodulation (19→6 channels, 3-level U-Net with channels 16→32→64). Inference via 7 GLSL compute shaders dispatched into caller's command buffer. FP16 feature buffers, FP32 weights, ~20 dispatches per frame. Validated against PyTorch golden reference (RMSE < 0.01).
>
> **Prerequisite: F18 ✅ (albedo demodulation) — COMPLETE.** The current model already denoises in demodulated irradiance space with 19-ch input (demodulated diffuse/specular irradiance + normals + roughness + depth + motion vectors + diffuse/specular albedo) and 6-ch output (separate diffuse/specular irradiance). All temporal phases build on this demodulated representation.
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
~~T1: Texture-backed feature maps~~ — **SKIPPED** (implemented and reverted; regression on RTX 4090, see below)
T2: Depthwise separable convolution blocks — PyTorch only (code infrastructure for T4/T5)
T3: Motion reprojection infrastructure (no ML change, foundation for temporal)
T4: Temporal residual network — training (quality: major improvement, temporal stability). Prerequisite: F18 ✅ (demodulated inputs). Includes depthwise separable blocks from T2.
T5: Temporal residual network — inference (quality + perf: smaller network, temporal accumulation, depthwise GLSL shaders, albedo remodulation in output shader)
T6: Super-resolution — training (perf: 4× fewer pixels to denoise)
T7: Super-resolution — inference (perf: full pipeline, render at half res)
T8: Mobile fragment shader backend (platform: mobile deployment via ncnn or custom)
```

> **Note on v2 model elimination:** The original plan included a standalone v2 model (depthwise separable single-frame denoiser) deployed between v1 (F18) and v3 (temporal). This intermediate model has been eliminated. The depthwise separable PyTorch blocks are added in T2 and first used in T4 training. The depthwise GLSL shaders are added in T5. monti_view jumps directly from v1 (F18) to v3 (temporal). This saves one training run, one golden reference, and one GPU integration cycle.

### Cumulative Performance Estimates (1080p, RTX 4090)

| After Phase | GFLOPS | Est. Time (current shaders) | Est. Time (coop matrix) | Quality vs Baseline |
|---|---|---|---|---|
| Baseline (F18 ✅) | ~122 | 15-40ms | — | 1.0× |
| ~~T1 (texture features)~~ | ~~8-20ms~~ | — | — | **SKIPPED** — regression, see T1 section |
| T2 (PyTorch blocks only) | — | — | — | (no inference change) |
| T3 (reprojection) | ~122 + warp | 15-40ms | — | 1.0× (reprojection not yet wired to model) |
| T4+T5 (temporal residual) | ~22 | 2-5ms | 0.2-0.5ms | ~1.3× (temporal accumulation) |
| T6+T7 (super-res, render@540p) | ~14 | 1-3ms | 0.1-0.3ms | ~1.2× |

### Cumulative Performance Estimates (720p→1080p, Adreno 750 mobile)

| After Phase | GFLOPS | Est. Time (fragment) | Quality |
|---|---|---|---|
| T5 (temporal) | ~8 (720p input) | 3-6ms | 1.3× |
| T7 (super-res) | ~6 | 2-4ms | 1.2× |
| T8 (mobile backend) | ~6 | 2-4ms (TBDR optimized) | 1.2× |

---

## ~~Phase T1: Texture-Backed Feature Maps~~ — SKIPPED

> **Status: SKIPPED.** T1 was fully implemented (all 9 denoise files rewritten: 7 GLSL compute shaders converted from storage buffer to `image2DArray`, `MlInference.h`/`.cpp` updated with `FeatureImage` structs, descriptor layout changes, and image barriers) and passed all 22 `[deni]` tests (3,455 assertions, golden reference RMSE < 0.01). However, performance testing on RTX 4090 (Sponza, 1280×720) showed a **~25% regression** (114–118ms vs 87–92ms baseline). Optimization attempts (input-first loop reorder to reduce redundant `imageLoad` calls) worsened the regression to ~3.5× slower (314–324ms) due to register pressure from accumulator arrays crushing GPU occupancy. All T1 changes were reverted.
>
> **Root causes of regression:**
> 1. `imageLoad` returns FP32 `vec4`, expanding from FP16 — double the register pressure and bandwidth vs the existing `float16_t` buffer access
> 2. RTX 4090's 96MB L2 cache already serves flat storage buffer loads efficiently; the texture cache advantage is minimal for this workload pattern
> 3. Each `imageLoad` fetches a full `vec4` (4 channels, 8 bytes of FP16) even when only 1 channel component is needed at that position, reducing effective bandwidth utilization
>
> **Impact on later phases:** T1 has no downstream dependencies — T2 (PyTorch blocks), T3 (reprojection), and T4+ (temporal) do not require texture-backed feature maps. The flat storage buffer representation is retained.

**Original goal:** Replace flat storage buffer feature maps with 2D image arrays (RGBA16F). This was expected to improve spatial cache locality for 3×3 convolutions. No model or training changes — inference produces bit-identical results.

**Original motivation:** The current `conv.comp` stores features in channel-major `[C][H][W]` flat storage buffers. For a 3×3 kernel, neighboring rows are `width` elements apart in memory — poor 2D cache locality. GPU texture units use tiled/swizzled memory layouts (Morton order) optimized for 2D spatial access, and the texture cache is separate from the L1/L2 cache used by storage buffer loads. Switching to `image2DArray` with RGBA16F packing (4 channels per layer) gives us hardware-optimized 2D locality and frees the L1 cache for weight data.

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

**Test: `[deni][texture][layer_counts]` — Feature images have correct layer counts**

1. After `Resize()`, inspect each `FeatureImage`
2. **Pass criteria:** `layers == ceil(channels / 4)` for every feature map (e.g., 16ch → 4 layers, 32ch → 8 layers, 64ch → 16 layers)

**Test: `[deni][texture][intermediate_match]` — Per-layer intermediates match PyTorch**

1. Generate PyTorch intermediate activations for each U-Net stage (after each conv+norm)
2. Run GPU inference, read back the feature image after each corresponding dispatch
3. **Pass criteria:** RMSE < 0.01 at each stage. This catches per-shader bugs that might cancel out in the final output.

**Test: `[deni][texture][boundary_padding]` — Zero-padding at image borders**

1. Create a small test input (8×8) with non-zero border pixels
2. Run a single 3×3 conv dispatch
3. **Pass criteria:** Border output pixels use zero-padding (not clamped or wrapped), verified against PyTorch with `padding=1` (which uses zero-padding by default)

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

## Phase T2: Depthwise Separable Convolution Blocks (PyTorch Only)

**Goal:** Add depthwise separable convolution building blocks to the PyTorch training codebase and unit test them. No GLSL shaders, no model training, no GPU inference changes — those happen in T4 (training) and T5 (inference) respectively.

**Motivation:** A standard `Conv2d(C_in, C_out, 3×3)` performs $C_{in} \times C_{out} \times 9$ MADs per pixel. A depthwise separable convolution splits this into:
1. **Depthwise:** `Conv2d(C, C, 3×3, groups=C)` — 1 filter per channel, $C \times 9$ MADs
2. **Pointwise:** `Conv2d(C, C_out, 1×1)` — channel mixing, $C \times C_{out}$ MADs
3. **Total:** $C \times 9 + C \times C_{out}$ MADs vs $C_{in} \times C_{out} \times 9$ MADs

For a 32→32 layer: standard = 9,216 vs separable = 1,312 MADs — **7× reduction**. For the full U-Net, the savings are ~3-4× because the first and last layers (interfacing with raw data) keep standard convolutions.

**Quality trade-off:** Depthwise separable convolutions have less expressive power because the spatial and channel dimensions are decoupled. For denoising, this is less damaging than for recognition tasks because spatial filtering in a denoiser is mostly guided by the G-buffer (normals, depth provide explicit geometry), and channel mixing provides the learned combination. Expected PSNR drop: 0.5-1 dB, which is acceptable given the large performance gain.

**No standalone v2 model is trained or deployed.** The first model to use depthwise separable convolutions is the v3 temporal model (T4). This avoids an intermediate retraining step and an intermediate GPU inference integration. GLSL depthwise/pointwise shaders are implemented in T5 alongside the temporal inference pipeline, where they're validated both in isolation (dedicated shader unit tests) and integrated (temporal golden reference).

**No retraining, no GLSL changes, no GPU inference changes.**

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

#### 2. Verify block compatibility with existing `DownBlock` / `UpBlock`

The `DepthwiseSeparableConvBlock` has the same interface as `ConvBlock` (`__init__(in_ch, out_ch)`, `forward(x) → tensor`). Verify it can be dropped into `DownBlock` and `UpBlock` as a replacement for interior `ConvBlock` layers by running the existing single-frame model with one conv swapped to depthwise separable and confirming shapes match.

### Numerical Validation Tests

**Test: `test_depthwise_block.py` — Block output shape and parameter count**

1. Create `DepthwiseSeparableConvBlock(16, 32)`
2. Forward pass with random input (B=2, C=16, H=64, W=64)
3. **Pass criteria:**
   - Output shape = (2, 32, 64, 64)
   - Parameter count: depthwise(16×3×3) + pointwise(16×32) + norm(2×32) + biases = 720 (vs ConvBlock ~4,768 for same config)
   - Output is finite (no NaN/Inf)

**Test: `test_depthwise_determinism.py` — Same weights produce same output**

1. Create two `DepthwiseSeparableConvBlock(16, 16)` with identical manual weights
2. Forward pass with same input
3. **Pass criteria:** Outputs are bit-identical

**Test: `test_depthwise_gradient_flow.py` — Gradients flow through all parameters**

1. Create `DepthwiseSeparableConvBlock(16, 32)`
2. Forward pass, compute loss = output.sum(), backward
3. **Pass criteria:** All parameters have non-zero `.grad` (depthwise.weight, pointwise.weight, norm.weight, norm.bias)

### Files to Modify

| File | Change |
|---|---|
| `training/deni_train/models/blocks.py` | Add `DepthwiseSeparableConvBlock` class |
| `training/tests/test_depthwise_block.py` | New — unit tests for the block |

### Verification
- All unit tests pass
- Block produces correct output shapes for various channel configs
- Gradient flows through all parameters
- No changes to existing model, shaders, or inference code

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
    FeatureImage denoised_diffuse;    // Previous frame's denoised diffuse irradiance (RGBA16F, 2D)
    FeatureImage denoised_specular;   // Previous frame's denoised specular irradiance (RGBA16F, 2D)
    FeatureImage reprojected_diffuse; // Warped previous diffuse (RGBA16F, 2D)
    FeatureImage reprojected_specular;// Warped previous specular (RGBA16F, 2D)
    FeatureImage disocclusion_mask;   // Binary mask (R16F, 2D): 1.0 = valid, 0.0 = disoccluded
    bool valid = false;               // False on first frame or after reset
};

FrameHistory frame_history_;
```

Allocate these images in `Resize()`. When resolution changes, set `frame_history_.valid = false` (forces full denoise on next frame).

#### 2. Reprojection shader — `reproject.comp`

Warps both the previous diffuse and specular outputs using motion vectors, producing two reprojected images and a shared disocclusion mask.

```glsl
#version 460
layout(local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
};

layout(set = 0, binding = 0, rg16f) uniform readonly image2D motion_vectors;    // Current frame MV
layout(set = 0, binding = 1, rgba16f) uniform readonly image2D prev_diffuse;    // Previous denoised diffuse
layout(set = 0, binding = 2, rgba16f) uniform readonly image2D prev_specular;   // Previous denoised specular
layout(set = 0, binding = 3, r16f) uniform readonly image2D prev_depth;         // Previous linear depth
layout(set = 0, binding = 4, r16f) uniform readonly image2D curr_depth;         // Current linear depth
layout(set = 0, binding = 5, rgba16f) uniform writeonly image2D reprojected_d;  // Output: warped diffuse
layout(set = 0, binding = 6, rgba16f) uniform writeonly image2D reprojected_s;  // Output: warped specular
layout(set = 0, binding = 7, r16f) uniform writeonly image2D disocclusion;      // Output: validity mask

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
        imageStore(reprojected_d, pos, vec4(0.0));
        imageStore(reprojected_s, pos, vec4(0.0));
        imageStore(disocclusion, pos, vec4(0.0));  // Disoccluded
        return;
    }

    // Fetch from previous denoised outputs (both lobes)
    vec4 prev_d = imageLoad(prev_diffuse, src_pos);
    vec4 prev_s = imageLoad(prev_specular, src_pos);

    // Depth-based disocclusion detection
    float prev_z = imageLoad(prev_depth, src_pos).r;
    float curr_z = imageLoad(curr_depth, pos).r;
    float depth_ratio = (prev_z > 0.0) ? abs(curr_z - prev_z) / max(prev_z, 1e-6) : 1.0;
    float valid = (depth_ratio < 0.1) ? 1.0 : 0.0;  // 10% depth tolerance

    imageStore(reprojected_d, pos, prev_d);
    imageStore(reprojected_s, pos, prev_s);
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

Also copy the current denoised output (both lobes) to `frame_history_.denoised_diffuse` and `frame_history_.denoised_specular` after the output conv writes them. These become the "previous output" for the next frame's reprojection.

#### 4. Wire into `Infer()` dispatch sequence

At the start of `Infer()`, before the U-Net dispatches:

```cpp
if (frame_history_.valid) {
    DispatchReproject(cmd, input.motion_vectors, input.linear_depth);
    InsertImageBarrier(cmd, frame_history_.reprojected_diffuse);
    InsertImageBarrier(cmd, frame_history_.reprojected_specular);
    InsertImageBarrier(cmd, frame_history_.disocclusion_mask);
}
```

At the end of `Infer()`, after the output conv:

```cpp
// Save current output (both lobes) and depth for next frame
CopyImageToImage(cmd, output_diffuse, frame_history_.denoised_diffuse);
CopyImageToImage(cmd, output_specular, frame_history_.denoised_specular);
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

**Test: `[deni][temporal][reproject_dual_lobe]` — Both lobes warped identically**

1. Set previous diffuse to all 1.0, previous specular to all 0.5, motion = (2, 0)
2. Run reprojection
3. **Pass criteria:** Both reprojected_d and reprojected_s are shifted by (-2, 0); the ratio reproj_s / reproj_d = 0.5 everywhere (lobes warped identically, not blended)

**Test: `[deni][temporal][frame_history_lifecycle]` — History valid flag transitions**

1. Create `MlInference`, verify `frame_history_.valid == false`
2. Run one frame of `Infer()`, verify `frame_history_.valid == true`
3. Call `Resize()` with same dimensions, verify `frame_history_.valid == false` (conservative reset)
4. Run `Infer()` with `reset_accumulation = true`, verify `frame_history_.valid == false` after dispatch

**Test: `[deni][temporal][depth_copy_fidelity]` — Previous depth buffer matches input**

1. Run one `Infer()` with a known depth image (e.g., linear ramp 0→1)
2. Read back `frame_history_.prev_depth`
3. **Pass criteria:** Exact bit-match with the input depth (copy, not interpolation)

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
- **Quality:** Expected 2-4 dB PSNR improvement over v1 single-frame (F18) on sequences with moderate motion. Much better temporal stability (less flicker between frames).

### Tasks

#### 1. Sequential training data generation — `monti_view` path tracking + `generate_training_data.py`

Temporal training requires ordered frame sequences. Camera paths are captured directly in `monti_view` using a tracking mode, then rendered sequentially by `generate_training_data.py`. `generate_viewpoints.py` is deleted — its random-variation approach is superseded by direct capture.

**Step 1 — Camera path capture in `monti_view`**

`monti_view` gains a path tracking mode (press `P` to toggle). While tracking is enabled, any camera motion automatically starts recording a new path (delta-based: a new frame is buffered whenever the camera moves by more than a minimum threshold). When motion stops for ~500ms, the path is automatically flushed to the viewpoints JSON file and a new path begins on the next movement. Pressing `P` again disables tracking and discards any in-progress path.

- The currently loaded environment map path and current `environmentRotation` value from the UI panel are embedded in every captured frame, enabling diverse lighting to be captured naturally by rotating the environment map during a recording session.
- Backspace deletes the last flushed path from the JSON file, allowing the user to undo accidental fly-throughs.
- The UI panel shows a recording indicator and frame count while capturing.
- Paths are buffered in memory and flushed to disk atomically (single JSON rewrite on stop) — no per-frame file I/O.

**Viewpoint format** — the existing viewpoint JSON field `id` is replaced by `path_id` + `frame`:

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

`path_id` groups frames into ordered sequences. `frame` defines render order within the path (0-indexed). The old per-viewpoint `id` field is removed — `{path_id}_{frame:04d}` serves as the unique render identifier. All frames of the same path share the same `environment`/`lights`/`exposure` assignment (captured at recording time in `monti_view`).

**Step 2 — Sequential rendering in `generate_training_data.py`**

Update the existing script to group viewpoints by `path_id`, sort by `frame`, and render in order so that `monti_datagen` sees consecutive camera positions and motion vectors reflect the camera motion between frames. All frames in every captured path are rendered — there is no truncation or sequence-length cap in datagen.

```bash
python scripts/generate_training_data.py \
    --monti-datagen build/app/datagen/Release/monti_datagen.exe \
    --viewpoints-dir training/viewpoints/ \
    --output training_data/
```

**`monti_datagen` changes required:** When frames belonging to the same `path_id` are rendered sequentially, the renderer must maintain temporal state between frames (previous camera transform, motion vector computation). Add a `--sequence-start` / `--sequence-continue` flag pair, or accept the full path's frame list in one call.

**Path lengths:** Captured paths are variable-length (a typical recording session produces paths of 30–300+ frames). `monti_datagen` renders all frames without truncation. The training preprocessor (below) is responsible for splitting into fixed-length windows.

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

The temporal preprocessor identifies paths by grouping on the 8-hex prefix and builds sliding windows of consecutive frames.

#### 2. Temporal preprocessor — `training/scripts/preprocess_temporal.py`

Rather than performing windowing and crop extraction at training time (which would require loading full-resolution EXR images on every training step, causing disk I/O saturation), a dedicated offline preprocessor converts rendered EXR sequences into pre-cropped safetensors files suitable for fast training.

**Design rationale:** Temporal training has an additional correctness requirement that single-frame training does not: all frames in an 8-frame window must share exactly the same crop coordinates. A `RandomCrop` applied independently per frame (as in `ExrDataset`) would destroy the spatial correspondence needed for reprojection. The preprocessor selects crop coordinates once per window and applies them identically across all 8 frames.

```python
# preprocess_temporal.py
#
# For each scene's rendered EXR directory:
#   1. Glob {path_id}_{frame:04d}_input.exr, group by path_id, sort by frame
#   2. Build sliding 8-frame windows (configurable stride, default = 4, giving 50% overlap)
#   3. For each window:
#      a. Load and demodulate all 8 input EXR + target EXR files (same logic as
#         convert_to_safetensors.py: divide radiance by albedo with epsilon=0.001)
#      b. Select N random 384×384 crop positions (N configurable, default = 4)
#      c. Apply the same crop coordinates to ALL 8 frames in the window
#      d. Write one .safetensors per crop:
#           input:  float16, (8, 19, 384, 384)  ← 8-frame sequence, 19 G-buffer channels
#           target: float16, (8, 7,  384, 384)  ← 8-frame sequence, 6ch ref irradiance + 1ch hit mask
#
# Note: The safetensors stores 19ch G-buffer per frame. The 26ch temporal model input is
# assembled at training time by concatenating G-buffer data with reprojected previous output
# (6ch) and disocclusion mask (1ch) computed from the model's own predictions. The 7ch
# target is NOT the network output format (3ch delta_d + 3ch delta_s + 1ch blend_weight)
# — it is 6ch reference demodulated irradiance + 1ch hit mask (same format as static model).
# 
# Parallelized with ProcessPoolExecutor (same pattern as convert_to_safetensors.py).
# Output files named: {path_id}_{window_start:04d}_crop{N}.safetensors
```

**Dataset size:** With ~2500 rendered frames per scene (stride=4, 4 crops/window), this produces ~2300 training samples per scene — comparable to the current single-frame dataset. At training time, each sample loads one small pre-cropped file, eliminating the disk I/O bottleneck entirely.

#### 3. Temporal training dataset — `training/deni_train/data/temporal_safetensors_dataset.py`

```python
class TemporalSafetensorsDataset(Dataset):
    """Loads pre-cropped 8-frame sequence safetensors for temporal residual training.
    
    Each .safetensors file contains one pre-extracted crop across 8 consecutive frames.
    Windowing and crop selection are handled offline by preprocess_temporal.py —
    this class simply loads one file per sample.
    
    Each sample returns:
        input:  float16, (8, 19, H_crop, W_crop)  ← sequence × channels × spatial
        target: float16, (8,  7, H_crop, W_crop)
    """

    def __init__(self, data_dir: str):
        self.files = sorted(glob.glob(os.path.join(data_dir, "**", "*.safetensors"),
                                      recursive=True))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        tensors = load_file(self.files[idx])
        inp = tensors["input"].clamp(-65504.0, 65504.0)
        tgt = tensors["target"].clamp(-65504.0, 65504.0)
        inp = torch.nan_to_num(inp, nan=0.0, posinf=0.0, neginf=0.0)
        tgt = torch.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0)
        return {"input": inp, "target": tgt}
```

`train.py` instantiates `TemporalSafetensorsDataset` when the config specifies `model.type: temporal_residual`. `convert_to_safetensors.py` is retained unchanged for the single-frame v1 (F18) pipeline.

#### 4. Temporal residual architecture — `training/deni_train/models/temporal_unet.py`

> **Note:** This architecture assumes F18 ✅ (albedo demodulation) is complete. The noisy radiance inputs are demodulated irradiance (radiance / albedo), not raw radiance. The network denoises in demodulated space, and albedo remodulation happens in the output shader (T5). The temporal model retains ALL 19 G-buffer channels from the static model and ADDS temporal-specific channels (reprojected previous output + disocclusion mask), because the network benefits from full material/geometry context for edge-aware temporal filtering and blend weight estimation.

```python
class DeniTemporalResidualNet(nn.Module):
    """Temporal residual denoiser (operates in demodulated irradiance space).
    
    Inputs (26 channels total):
        Temporal channels (7ch):
        - reprojected_diffuse: 3ch (warped previous denoised diffuse irradiance)
        - reprojected_specular: 3ch (warped previous denoised specular irradiance)
        - disocclusion_mask: 1ch (0=invalid, 1=valid)
        
        Current frame G-buffer (19ch, same as static model):
        - noisy_diffuse_irradiance: 3ch (current demodulated diffuse)
        - noisy_specular_irradiance: 3ch (current demodulated specular)
        - world_normals: 3ch (XYZ)
        - roughness: 1ch
        - linear_depth: 1ch
        - motion_vectors: 2ch (XY)
        - diffuse_albedo: 3ch
        - specular_albedo: 3ch
    
    Outputs (7 channels):
        - diffuse_correction_delta: 3ch (added to reprojected diffuse)
        - specular_correction_delta: 3ch (added to reprojected specular)
        - blend_weight: 1ch (sigmoid, 0=use reprojected, 1=use noisy+correction)
    
    Final output (demodulated):
        denoised_d = reprojected_d + blend_weight * delta_d
        denoised_s = reprojected_s + blend_weight * delta_s
    Remodulation (in output shader):
        final_rgb = denoised_d * albedo_d + denoised_s * albedo_s
    """
    
    def __init__(self, base_channels=12):
        super().__init__()
        c = base_channels   # 12 (slightly larger than originally planned 8 to handle 26 input channels)
        
        # 2-level U-Net (not 3 — smaller network for residual task)
        self.down0 = DownBlock(26, c, use_depthwise_separable=True)
        self.bottleneck1 = DepthwiseSeparableConvBlock(c, c * 2)
        self.bottleneck2 = DepthwiseSeparableConvBlock(c * 2, c * 2)
        self.up0 = UpBlock(c * 2, c, c, use_depthwise_separable=True)
        self.out_conv = nn.Conv2d(c, 7, kernel_size=1)  # 3ch diffuse delta + 3ch specular delta + 1ch weight
    
    def forward(self, reprojected_d, reprojected_s, disocclusion,
                noisy_d, noisy_s, normals, roughness, depth, motion,
                albedo_d, albedo_s):
        x = torch.cat([reprojected_d, reprojected_s, disocclusion,
                        noisy_d, noisy_s, normals, roughness, depth, motion,
                        albedo_d, albedo_s], dim=1)  # (B, 26, H, W)
        
        pooled, skip0 = self.down0(x)
        b = self.bottleneck1(pooled)
        b = self.bottleneck2(b)
        up = self.up0(b, skip0)
        
        out = self.out_conv(up)         # (B, 7, H, W)
        delta_d = out[:, :3]            # Diffuse correction
        delta_s = out[:, 3:6]           # Specular correction
        weight = torch.sigmoid(out[:, 6:7])  # Blend weight [0, 1]
        
        # For disoccluded pixels, force blend_weight toward 1.0
        # (network should learn this, but this provides a strong prior)
        weight = torch.max(weight, 1.0 - disocclusion)
        
        denoised_d = reprojected_d + weight * delta_d
        denoised_s = reprojected_s + weight * delta_s
        return torch.cat([denoised_d, denoised_s], dim=1)  # (B, 6, H, W)
```

**Why 26 input channels (not 14 as originally drafted)?** The original T4 design dropped linear depth, motion vectors, and albedo from the temporal model's inputs under the assumption that the residual network only needed a minimal set. This was incorrect:

1. **Albedo (6ch):** Edge-aware filtering needs albedo to distinguish texture edges from noise boundaries, especially in disoccluded regions where the network falls back to single-frame denoising. Dropping albedo would regress quality at material boundaries — the exact problem F18 solved.
2. **Linear depth (1ch):** Provides continuous geometric context beyond the binary disocclusion mask. Essential for depth-discontinuity-aware filtering.
3. **Motion vectors (2ch):** Motion magnitude helps the network adaptively weight the temporal blend factor — faster motion should trust reprojection less. The reprojection shader uses motion vectors for warping, but the network benefits from seeing motion magnitude directly.
4. **Separate diffuse/specular reprojection (6ch instead of 3ch):** Required to maintain per-lobe remodulation from F18. Without separate lobe reprojection, the temporal model would have to combine lobes before blending, losing the quality advantage of per-lobe `denoised_d * albedo_d + denoised_s * albedo_s` remodulation.

**Parameter count:** ~25-35K parameters (2-level U-Net, base_channels=12, depthwise separable). About 4× smaller than the static v1 (F18) model.

**GFLOPS estimate:** At 1080p: ~22 GFLOPS (vs ~122 for v1). Slightly higher than 14-channel version (~18 GFLOPS) due to the larger first conv layer, but still a major reduction from the static model.

#### 5. Temporal training loop modifications — `training/deni_train/train_temporal.py`

Key differences from single-frame training:
- **First frame handling:** For the first frame in each sequence (no previous output available), use zeros for the reprojected input channels (6ch reprojected d/s = 0, disocclusion = 1). This trains the temporal network to handle the cold-start case by falling back to single-frame behavior.
- **Temporal stability loss:** Add a term penalizing flicker between consecutive outputs:
  ```python
  # Warp current output to next frame, compare against next frame's output
  temporal_loss = L1(warp(output_t, motion_t_to_t1), output_t1.detach())
  total_loss = lambda_l1 * l1_loss + lambda_perceptual * perceptual_loss 
             + lambda_temporal * temporal_loss
  ```
  `lambda_temporal = 0.5` — strong enough to enforce stability without suppressing legitimate changes.
- **Sequence batching:** DataLoader returns batches of frame pairs from the same sequence.

#### 6. Training config — `training/configs/default.yaml` (update for temporal)

The existing `default.yaml` is updated for temporal training (no separate config file):

```yaml
model:
  type: temporal_residual
  in_channels: 26
  out_channels: 7     # 3ch diffuse delta + 3ch specular delta + 1ch blend weight
  base_channels: 12
  use_depthwise_separable: true

data:
  data_dir: "../training_data_temporal_st"  # pre-processed by preprocess_temporal.py --window 8
  data_format: "temporal_safetensors"
  precropped: true
  sequence_length: 8                        # encoded in the safetensors shape
  batch_size: 4     # Smaller batch (8-frame sequence per sample = 8× memory vs single-frame)

loss:
  lambda_l1: 1.0
  lambda_perceptual: 0.1
  lambda_temporal: 0.5

training:
  epochs: 200
  learning_rate: 1.0e-4
```

#### 7. Export and evaluate

Export v3 weights: `python scripts/export_weights.py --checkpoint configs/checkpoints/v3_temporal_best.pt --output models/deni_v3_temporal.denimodel`

Evaluate on held-out temporal sequences, measuring:
- Per-frame PSNR (should exceed v1 single-frame by 2-4 dB on average)
- Temporal stability (warp error between consecutive outputs < threshold)
- Disocclusion quality (PSNR specifically in disoccluded regions)

### Numerical Validation Tests (PyTorch-side only, no GPU tests in this phase)

**Test: `test_temporal_model.py` — Architecture shape validation**

1. Create `DeniTemporalResidualNet(base_channels=12)`
2. Forward pass with random inputs: reprojected_d(B,3,256,256), reprojected_s(B,3,256,256), disocclusion(B,1,256,256), noisy_d(B,3,256,256), noisy_s(B,3,256,256), normals(B,3,256,256), roughness(B,1,256,256), depth(B,1,256,256), motion(B,2,256,256), albedo_d(B,3,256,256), albedo_s(B,3,256,256)
3. **Pass criteria:** Output shape = (B, 6, 256, 256); parameter count in 25K-35K range

**Test: `test_temporal_model.py` — Internal channel count assertion**

1. Verify the concatenated input tensor is exactly (B, 26, H, W) by instrumenting the `forward()` method
2. Verify `out_conv` weight shape is `(7, base_channels, 1, 1)`
3. **Pass criteria:** Shapes match spec; assertion failures if channel counts drift

**Test: `test_temporal_model.py` — Blend weight bounds and disocclusion forcing**

1. Forward pass with `disocclusion = 0.0` everywhere (all pixels disoccluded)
2. **Pass criteria:** All blend weights = 1.0 (forced by `max(weight, 1.0 - disocclusion)`)
3. Forward pass with `disocclusion = 1.0` everywhere (all valid)
4. **Pass criteria:** All blend weights ∈ [0, 1] (sigmoid range)

**Test: `test_temporal_model.py` — Gradient flow through all parameters**

1. Forward pass, compute `loss = output.sum()`, `loss.backward()`
2. **Pass criteria:** Every parameter in the model has non-zero `.grad` (catches disconnected subgraphs in the 2-level U-Net, depthwise blocks, and skip connections)

**Test: `test_temporal_model.py` — First-frame equivalence**

1. Forward with reprojected_d = zeros, reprojected_s = zeros, disocclusion = 0.0 (all disoccluded)
2. Verify output is deterministic (same random seed → same output)
3. **Pass criteria:** Output is non-zero (network produces something from noisy input alone)

**Test: `test_temporal_reproject.py` — PyTorch reprojection matches reference**

1. Create a known image + motion vectors
2. Run PyTorch reprojection
3. **Pass criteria:** Output matches expected shifted image, disocclusion mask correct at boundaries

**Test: `test_temporal_training.py` — Training loop converges**

1. Overfit on 2 synthetic temporal sequences (4 frames each)
2. Train for 100 steps
3. **Pass criteria:** Loss decreases to < 0.05

**Test: `test_temporal_training.py` — Temporal PSNR progression**

1. Train on a synthetic 8-frame sequence with moderate noise
2. Run trained model on the sequence, measure PSNR at each frame
3. **Pass criteria:** PSNR(frame 8) > PSNR(frame 1) by at least 1 dB (temporal accumulation is helping)

**Test: `test_temporal_training.py` — Temporal stability loss effect**

1. Train once with `lambda_temporal = 0.0`, once with `lambda_temporal = 0.5`
2. Measure frame-to-frame output difference (L1 between consecutive outputs warped by motion)
3. **Pass criteria:** `lambda_temporal = 0.5` model has lower frame-to-frame difference (less flicker)

### Verification
- v3 model trains on temporal data, loss converges
- Evaluation shows PSNR improvement over v1 single-frame model on temporal sequences
- Temporal stability metric improves (less flicker)
- All PyTorch-side tests pass
- Weight export produces valid `.denimodel` file

---

## Phase T5: Temporal Residual Network — Inference

**Goal:** Implement the temporal residual network's GPU inference, replacing the single-frame U-Net with the new temporal pipeline. Wire the reprojection output (T3) into the residual network input.

**Motivation:** This is where the quality and performance gains materialize on the GPU. The temporal residual network is ~6× smaller than the single-frame v1 model (F18), so inference is dramatically faster. And quality improves because each frame leverages the accumulated history from all previous frames. This phase also introduces the depthwise separable GLSL shaders (`depthwise_conv.comp`, `pointwise_conv.comp`) — the first model to use them on the GPU is v3.

**No retraining required.** Uses v3 weights from T4.

### Performance & Quality Estimates

- **Performance:** ~18 GFLOPS at 1080p. With texture feature maps: estimated 2-5ms on RTX 4090. With cooperative matrix (future): <0.5ms.
- **Quality:** Matches or exceeds single-frame v1 quality (120K params) despite having only 15-20K params, because temporal accumulation provides information that a larger single-frame network can't access.

### Tasks

#### 1. New shaders for temporal input

**`temporal_encoder_input_conv.comp`** — Replaces `encoder_input_conv.comp` for the temporal model. Reads 26 input channels from:
- Reprojected previous diffuse: `image2D` (3ch, from T3's reprojection shader)
- Reprojected previous specular: `image2D` (3ch, from T3's reprojection shader)
- Disocclusion mask: `image2D` (1ch, from T3's reprojection shader)
- Noisy diffuse: `image2D` (3ch, demodulated on-the-fly from G-buffer, gated by hit mask)
- Noisy specular: `image2D` (3ch, demodulated on-the-fly, gated by hit mask)
- World normals: `image2D` (3ch + roughness in .w = 4ch from G-buffer)
- Linear depth: `image2D` (1ch from G-buffer)
- Motion vectors: `image2D` (2ch from G-buffer)
- Diffuse albedo: `image2D` (3ch from G-buffer)
- Specular albedo: `image2D` (3ch from G-buffer)

Demodulation of noisy irradiance channels uses the same logic as the static model's `encoder_input_conv.comp` (F18 ✅). The temporal-specific bindings (reprojected_d, reprojected_s, disocclusion) are additional inputs from the T3 reprojection pass.

Outputs to first feature level `image2DArray` (base_channels layers).

#### 2. Depthwise separable GLSL shaders — `depthwise_conv.comp`, `pointwise_conv.comp`

New shaders implementing depthwise separable convolutions on the GPU. These are first introduced in T5 (no standalone v2 model needed them earlier).

**`depthwise_conv.comp`** — Depthwise 3×3 convolution (groups = channels):

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
    }
}
```

**`pointwise_conv.comp`** — 1×1 convolution (channel mixing only):

```glsl
// Per-pixel: read all input channels, compute weighted sum per output channel
for (uint oc = 0; oc < OUT_CHANNELS; ++oc) {
    float sum = 0.0;
    for (uint ic = 0; ic < IN_CHANNELS; ++ic) {
        sum += input[ic] * weights[oc * IN_CHANNELS + ic];
    }
    sum += bias[oc];
    output[oc] = sum;
}
```

For depthwise separable layers, the dispatch sequence is:
```
depthwise_conv.comp (3×3, C→C)  →  pointwise_conv.comp (1×1, C→C_out)  →  group_norm  →  activation
```

Two dispatches instead of one per ConvBlock, but each dispatch is much cheaper.

#### 3. Temporal output shader — `temporal_output_conv.comp`

Modified output conv that produces 7 channels (3ch diffuse delta + 3ch specular delta + 1ch blend weight), applies the temporal blending per-lobe, and remodulates with albedo (F18 ✅):

```glsl
// 1×1 conv: IN_CHANNELS → 7 output channels
// ... (same conv loop as output_conv.comp but with 7 outputs)
vec3 delta_d = vec3(sums[0], sums[1], sums[2]);    // Diffuse correction delta
vec3 delta_s = vec3(sums[3], sums[4], sums[5]);    // Specular correction delta
float weight = 1.0 / (1.0 + exp(-sums[6]));        // Sigmoid → blend weight

// Force weight=1 for disoccluded pixels
float disocc = imageLoad(disocclusion_mask, pos).r;
weight = max(weight, 1.0 - disocc);

// Apply per-lobe temporal blending
vec3 reproj_d = imageLoad(reprojected_diffuse, pos).rgb;
vec3 reproj_s = imageLoad(reprojected_specular, pos).rgb;
vec3 denoised_d = reproj_d + weight * delta_d;
vec3 denoised_s = reproj_s + weight * delta_s;

// Albedo remodulation (F18): per-lobe irradiance × albedo → radiance
vec3 albedo_d = imageLoad(diffuse_albedo, pos).rgb;
vec3 albedo_s = imageLoad(specular_albedo, pos).rgb;
float hit = imageLoad(noisy_diffuse, pos).a;

vec3 final_d = (hit > 0.5) ? denoised_d * max(albedo_d, vec3(0.001)) : denoised_d;
vec3 final_s = (hit > 0.5) ? denoised_s * max(albedo_s, vec3(0.001)) : denoised_s;
vec3 final_rgb = final_d + final_s;

imageStore(output_image, pos, vec4(final_rgb, 1.0));

// Also write denoised irradiance (pre-remodulation) to frame history for next frame's reprojection
imageStore(history_diffuse, pos, vec4(denoised_d, 1.0));
imageStore(history_specular, pos, vec4(denoised_s, 1.0));
```

This fuses the per-lobe blending, remodulation, and history write into the output shader — no extra dispatches needed. Writing the denoised irradiance (before remodulation) to frame history ensures that reprojection in the next frame operates in demodulated space.

#### 4. Update `Infer()` dispatch sequence

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
    DispatchTemporalEncoderInput(cmd, input);  // 26ch → base_channels
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
    
    //    Temporal output (per-lobe delta + blend + remodulate + history write)
    DispatchTemporalOutputConv(cmd, ...);
    
    // 3. Save current depth for next frame (denoised irradiance is written by output shader)
    CopyImage(cmd, input.linear_depth, frame_history_.prev_depth);
    frame_history_.valid = true;
}
```

Total dispatches: ~14 (vs ~20 for single-frame v1). Each dispatch does less work due to smaller channels and depthwise separable convolutions.

#### 5. Fallback for first frame

When `frame_history_.valid == false` (first frame, resolution change, accumulation reset):
- Set reprojected diffuse and specular to black (zero)
- Set disocclusion mask to 0.0 everywhere (all pixels "disoccluded")
- The network's forced `weight = max(weight, 1.0 - disocclusion)` ensures weight=1 for all pixels
- The output becomes `0 + 1.0 * delta = delta`, i.e., the network produces the output directly from the noisy input, behaving as a single-frame denoiser

This is exactly what the network was trained to handle (first-frame fallback in T4).

#### 6. Auto-detect model version

The weight loader should detect which model type is loaded and configure the dispatch path accordingly:

```cpp
enum class ModelVersion { kV1_SingleFrame, kV3_Temporal };

// Detected from weight layer names:
// - Contains "down1" → 3-level U-Net (v1 single-frame, standard convolutions)
// - Does NOT contain "down1" + contains "depthwise" → v3 (2-level temporal, depthwise separable)
```

This allows loading any model version without manual configuration.

### Numerical Validation Tests

**Test: `[deni][numerical][golden]` — v3 temporal model matches PyTorch**

1. Generate golden reference with v3 temporal model:
   - Known input: reprojected + disocclusion + noisy G-buffer
   - Run PyTorch inference, store expected output
2. Run GPU inference with same inputs
3. **Pass criteria:** RMSE < 0.01, max_abs_error < 0.05

**Test: `[deni][numerical][golden_multiframe]` — Multi-frame golden reference**

1. Generate PyTorch golden output for a 3-frame sequence (frame 1: cold start, frame 2-3: with history)
2. Run GPU inference for all 3 frames sequentially, storing intermediate history
3. Compare GPU output at EACH frame against PyTorch reference
4. **Pass criteria:** RMSE < 0.01 at every frame. This catches history write/read bugs that a single-frame golden test would miss.

**Test: `[deni][numerical][depthwise]` — Depthwise separable conv matches PyTorch**

1. Create a standalone test with a single depthwise separable ConvBlock (known weights)
2. Run through GPU shader pipeline (depthwise_conv.comp + pointwise_conv.comp + group_norm)
3. Compare against PyTorch `DepthwiseSeparableConvBlock` output
4. **Pass criteria:** RMSE < 0.005 (tighter than full model, single layer)

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

**Test: `[deni][temporal][remodulation]` — Per-lobe remodulation produces correct RGB**

1. Set up known denoised_d = (0.5, 0.5, 0.5), denoised_s = (0.1, 0.1, 0.1), albedo_d = (0.8, 0.2, 0.2), albedo_s = (1.0, 1.0, 1.0), hit = 1.0
2. Run temporal output shader
3. **Pass criteria:** final_rgb = (0.5 × 0.8 + 0.1 × 1.0, 0.5 × 0.2 + 0.1 × 1.0, 0.5 × 0.2 + 0.1 × 1.0) = (0.5, 0.2, 0.2). Exact FP16-precision match.

**Test: `[deni][temporal][history_demodulated]` — Frame history stores demodulated irradiance**

1. Run one frame of inference with known albedo and noisy input
2. Read back `frame_history_.denoised_diffuse` and `frame_history_.denoised_specular`
3. **Pass criteria:** History values are demodulated irradiance (NOT remodulated RGB). Verify by checking that history_d ≠ output_rgb and that history_d * albedo_d + history_s * albedo_s ≈ output_rgb.

**Test: `[deni][temporal][sigmoid_stability]` — No NaN/Inf from extreme blend weight inputs**

1. Set up a synthetic input that produces very large pre-sigmoid values (sums[6] = ±88.0, near FP16 overflow)
2. Run temporal output shader
3. **Pass criteria:** blend_weight ∈ {~0.0, ~1.0} (saturated sigmoid), no NaN/Inf in output

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
