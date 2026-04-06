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
> **Relationship to existing plans:** Replaces the outline phases F11-4, F11-5, F12, and F13 in [ml_denoiser_plan.md](completed/ml_denoiser_plan.md) and [roadmap.md](roadmap.md) with a unified, sequenced implementation plan.
>
> **Session sizing:** Each phase is scoped to fit within a single Copilot Claude Opus 4.6 context session, following the convention in [ml_denoiser_plan.md](completed/ml_denoiser_plan.md).

---

## Architecture Overview

The plan proceeds through 8 phases, each building on the previous:

```
~~T1: Texture-backed feature maps~~ — **SKIPPED** (implemented and reverted; regression on RTX 4090, see below)
T2: Depthwise separable convolution blocks — PyTorch only ✅
T3: Motion reprojection infrastructure ✅
T4: Temporal residual network — training ✅
T5: Temporal residual network — inference ✅ (quality + perf: smaller network, temporal accumulation, depthwise GLSL shaders, albedo remodulation in output shader)
T6–T8: Super-resolution and mobile — see [temporal_denoiser_superres_plan.md](../temporal_denoiser_superres_plan.md)
```

> **Note on v2 model elimination:** The original plan included a standalone v2 model (depthwise separable single-frame denoiser) deployed between v1 (F18) and v3 (temporal). This intermediate model has been eliminated. The depthwise separable PyTorch blocks are added in T2 and first used in T4 training. The depthwise GLSL shaders are added in T5. monti_view jumps directly from v1 (F18) to v3 (temporal). This saves one training run, one golden reference, and one GPU integration cycle.

### Cumulative Performance Estimates (1080p, RTX 4090)

| After Phase | GFLOPS | Est. Time (current shaders) | Est. Time (coop matrix) | Quality vs Baseline |
|---|---|---|---|---|
| Baseline (F18 ✅) | ~122 | 15-40ms | — | 1.0× |
| ~~T1 (texture features)~~ | ~~8-20ms~~ | — | — | **SKIPPED** — regression, see T1 section |
| T2 (PyTorch blocks only) | — | — | — | (no inference change) |
| T3 (reprojection) | ~122 + warp | 15-40ms | — | 1.0× (reprojection not yet wired to model) |
| T4+T5 ✅ (temporal residual, base_channels=32) | ~22 | 3-5ms | 0.2-0.5ms | ~1.3× (temporal accumulation) |

### Cumulative Performance Estimates (720p→1080p, Adreno 750 mobile)

| After Phase | GFLOPS | Est. Time (fragment) | Quality |
|---|---|---|---|
| T5 ✅ (temporal, base_channels=32) | ~10 (720p input) | 4-7ms | 1.3× |

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

## Phase T2: Depthwise Separable Convolution Blocks (PyTorch Only) ✅

> **Status: COMPLETE.** `DepthwiseSeparableConvBlock` added to `training/deni_train/models/blocks.py` with depthwise 3×3 (groups=in_ch, no bias) + pointwise 1×1 + GroupNorm + LeakyReLU. 9 unit tests in `training/tests/test_depthwise_block.py` covering output shape, parameter count (752 vs 4,704 for ConvBlock), determinism, gradient flow, and drop-in compatibility with DownBlock/UpBlock. All tests pass.

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
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(
            self.depthwise.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu"
        )
        nn.init.kaiming_normal_(
            self.pointwise.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu"
        )
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)
```

> **Design decisions:**
> - `bias=False` on depthwise conv — the subsequent pointwise+GroupNorm recenters activations, making depthwise bias redundant. This is standard practice (MobileNetV2/V3, EfficientNet).
> - Same `kaiming_normal_(fan_out)` init for both convs — for depthwise conv with `groups=in_ch`, `fan_in == fan_out == 9`, so the mode choice is irrelevant.
> - No intermediate norm/act between depthwise and pointwise — simpler, fewer GLSL dispatches in T5, and standard for lightweight denoising architectures.

#### 2. Verify block compatibility with existing `DownBlock` / `UpBlock`

The `DepthwiseSeparableConvBlock` has the same interface as `ConvBlock` (`__init__(in_ch, out_ch)`, `forward(x) → tensor`). Verify via a formal test (in `test_depthwise_block.py`) that it can be dropped into `DownBlock` and `UpBlock` as a replacement for interior `ConvBlock` layers by constructing those blocks with `DepthwiseSeparableConvBlock` substituted and confirming output shapes match.

### Numerical Validation Tests

All tests live in a single file: `training/tests/test_depthwise_block.py`.

**Test: output shape and parameter count**

1. Create `DepthwiseSeparableConvBlock(16, 32)`
2. Forward pass with random input (B=2, C=16, H=64, W=64)
3. **Pass criteria:**
   - Output shape = (2, 32, 64, 64)
   - Parameter count = 752: depthwise weight (16×1×3×3=144) + pointwise weight (32×16×1×1=512) + pointwise bias (32) + norm weight (32) + norm bias (32). No depthwise bias.
   - For comparison, `ConvBlock(16, 32)` has 4,704 params — ~6.3× more.
   - Output is finite (no NaN/Inf)

**Test: determinism — same weights produce same output**

1. Create two `DepthwiseSeparableConvBlock(16, 16)` with identical manual weights
2. Forward pass with same input
3. **Pass criteria:** Outputs are bit-identical

**Test: gradient flow through all parameters**

1. Create `DepthwiseSeparableConvBlock(16, 32)`
2. Forward pass, compute loss = output.sum(), backward
3. **Pass criteria:** All parameters have non-zero `.grad` (depthwise.weight, pointwise.weight, pointwise.bias, norm.weight, norm.bias)

**Test: drop-in compatibility with DownBlock and UpBlock**

1. Construct a `DownBlock` and `UpBlock` with `DepthwiseSeparableConvBlock` replacing interior `ConvBlock` layers
2. Forward pass with appropriately shaped inputs
3. **Pass criteria:** Output shapes match those from the standard `ConvBlock`-based blocks

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

## Phase T3: Motion Reprojection Infrastructure ✅

> **Status: COMPLETE.** GPU reprojection infrastructure implemented: `HistoryImage`/`FrameHistory` structs in `MlInference.h`, `reproject.comp` shader (8 bindings, motion vector warping, 10% depth disocclusion), `CopyImageToHistory` for per-frame output/depth archival, `DispatchReproject` wired into `Infer()` flow, descriptor pool ring-buffer (kPoolCount=3) matching frames-in-flight. 4 standalone reprojection tests in `temporal_reproject_test.cpp` (identity, shift, disocclusion, dual_lobe). All 26 `[deni]` tests pass (3,615 assertions), zero Vulkan validation errors in Debug monti_view. The `frame_history_lifecycle` and `depth_copy_fidelity` tests were omitted (require private member access; lifecycle is exercised end-to-end by the 4 reprojection tests).

**Goal:** Implement the GPU infrastructure for reprojecting the previous frame's denoised output to the current frame using motion vectors. This phase adds no ML changes — it creates the foundation that the temporal residual network (T4/T5) will build on.

**Motivation:** Temporal denoising requires comparing the current noisy frame against a reprojected version of the previous clean output. Motion reprojection warps each pixel from the previous frame to its new position using screen-space motion vectors. The result is an initial estimate of the current frame that is clean (from the previous denoiser output) but may have errors in disoccluded regions (newly visible areas) and fast-moving objects.

**No retraining required.** This phase only adds GPU infrastructure (a reprojection shader and frame history management). The ML model is not modified.

### Performance & Quality Estimates

- **Performance:** Adds ~0.1-0.2ms for the reprojection pass (single fullscreen dispatch at output resolution). Negligible.
- **Quality:** No change — reprojection output is not used by the denoiser yet (that happens in T5).

### Tasks

#### 1. Frame history management — `MlInference.h/cpp`

Add a lightweight `HistoryImage` struct for 2D image history (distinct from `FeatureBuffer`, which is a flat storage buffer for intermediate feature maps):

```cpp
// 2D VkImage for frame history storage (temporal reprojection).
// Unlike FeatureBuffer (flat [C][H][W] storage buffer), this is a proper
// VkImage suitable for vkCmdCopyImage and imageLoad/imageStore in shaders.
struct HistoryImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0;
};

struct FrameHistory {
    HistoryImage denoised_diffuse;    // Previous frame's denoised diffuse irradiance (RGBA16F, 2D)
    HistoryImage denoised_specular;   // Previous frame's denoised specular irradiance (RGBA16F, 2D)
    HistoryImage reprojected_diffuse; // Warped previous diffuse (RGBA16F, 2D)
    HistoryImage reprojected_specular;// Warped previous specular (RGBA16F, 2D)
    HistoryImage disocclusion_mask;   // Binary mask (R16F, 2D): 1.0 = valid, 0.0 = disoccluded
    HistoryImage prev_depth;          // Previous frame's linear depth (RG16F, 2D — matches DenoiserInput format)
    bool valid = false;               // False on first frame or after reset
};

FrameHistory frame_history_;
```

Allocate these images in `Resize()`. When resolution changes, set `frame_history_.valid = false` (forces full denoise on next frame).

> **Why HistoryImage and not FeatureBuffer?** Frame history needs `vkCmdCopyImage` from the renderer's output image (a `VkImage`), and the reprojection shader reads history via `imageLoad`. `FeatureBuffer` is a flat storage buffer with no `VkImage`. The `FeatureImage` type from (reverted) T1 also does not exist in the codebase. `HistoryImage` is a minimal new struct with just the fields needed.

> **Why RG16F for prev_depth?** The renderer's `linear_depth` is RG16F (see `DenoiserInput`). `vkCmdCopyImage` requires matching formats between source and destination. Using RG16F for history depth avoids a format conversion; the shader reads only `.r`.

#### 2. Reprojection shader — `reproject.comp`

Warps both the previous diffuse and specular outputs using motion vectors, producing two reprojected images and a shared disocclusion mask.

```glsl
#version 460
layout(local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
};

layout(set = 0, binding = 0, rg16f)   uniform readonly  image2D motion_vectors;    // Current frame MV (normalized [0,1] screen-space)
layout(set = 0, binding = 1, rgba16f) uniform readonly  image2D prev_diffuse;      // Previous denoised diffuse
layout(set = 0, binding = 2, rgba16f) uniform readonly  image2D prev_specular;     // Previous denoised specular
layout(set = 0, binding = 3, rg16f)   uniform readonly  image2D prev_depth;        // Previous linear depth (RG16F, .r = linear Z)
layout(set = 0, binding = 4, rg16f)   uniform readonly  image2D curr_depth;        // Current linear depth (RG16F, .r = linear Z)
layout(set = 0, binding = 5, rgba16f) uniform writeonly image2D reprojected_d;     // Output: warped diffuse
layout(set = 0, binding = 6, rgba16f) uniform writeonly image2D reprojected_s;     // Output: warped specular
layout(set = 0, binding = 7, r16f)    uniform writeonly image2D disocclusion;      // Output: validity mask

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (pos.x >= int(width) || pos.y >= int(height)) return;

    // Motion vector convention (from raygen.rgen):
    //   mv = screen_current - screen_prev  (normalized [0,1] screen-space)
    // To find previous-frame position: prev = current - mv
    vec2 mv_norm = imageLoad(motion_vectors, pos).rg;
    vec2 mv_pixels = mv_norm * vec2(width, height);

    // Source position in previous frame (pixel coordinates)
    vec2 src = vec2(pos) + 0.5 - mv_pixels;  // +0.5 for pixel center, then subtract MV
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

> **MV sign convention:** The renderer computes `motion = screen_current - screen_prev` in normalized [0,1] screen-space (see `raygen.rgen`). To find the previous-frame source pixel, we subtract: `src = current_pixel - mv * resolution`. The plan's original `src = pos + mv` was incorrect (inverted sign and missing normalized→pixel conversion).

> **Depth format:** Bindings 3-4 use `rg16f` (not `r16f`) to match the renderer's `linear_depth` format. Only `.r` is read.

Depth threshold (10%) is a tuning parameter — can be exposed as a push constant later.

#### 3. Depth history management

The `prev_depth` field is already included in the `FrameHistory` struct (task 1). At the end of each `Infer()` call, copy the current frame's depth image to `prev_depth` via `vkCmdCopyImage`. The depth image is RG16F (matching `DenoiserInput::linear_depth` format), so the copy is format-compatible.

Also copy the current denoised output (both lobes) to `frame_history_.denoised_diffuse` and `frame_history_.denoised_specular` after the output conv writes them. These become the "previous output" for the next frame's reprojection.

> **Dispatch table addition:** `MlDeviceDispatch` currently lacks `vkCmdCopyImage`. Add it:
> ```cpp
> PFN_vkCmdCopyImage vkCmdCopyImage = nullptr;
> ```
> Load it in `MlDeviceDispatch::Load()` alongside the existing entries.

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
// Save current output (both lobes) and depth for next frame via vkCmdCopyImage.
// All images remain in VK_IMAGE_LAYOUT_GENERAL (no layout transitions needed).
// Source images: output_diffuse/specular are written by output_conv; input.linear_depth
// comes from the renderer. Destination: HistoryImage members in frame_history_.
CopyImage(cmd, output_diffuse, frame_history_.denoised_diffuse);
CopyImage(cmd, output_specular, frame_history_.denoised_specular);
CopyImage(cmd, input.linear_depth, frame_history_.prev_depth);
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

1. Set motion vectors to uniform normalized value equivalent to (5, 3) pixel displacement (i.e., `(5/width, 3/height)` in normalized screen-space)
2. Set previous output to a checkered pattern
3. Run reprojection
4. **Pass criteria:** Reprojected output is shifted by (-5, -3) pixels relative to previous (since `prev = curr - mv`); border pixels have disocclusion = 0.0

**Test: `[deni][temporal][reproject_disocclusion]` — Depth discontinuity detected**

1. Set current depth to a value much larger than previous depth at certain pixels
2. Run reprojection
3. **Pass criteria:** Pixels with >10% depth ratio have disocclusion = 0.0

**Test: `[deni][temporal][reproject_dual_lobe]` — Both lobes warped identically**

1. Set previous diffuse to all 1.0, previous specular to all 0.5, motion = normalized equivalent of (2, 0) pixel displacement
2. Run reprojection
3. **Pass criteria:** Both reprojected_d and reprojected_s are shifted by (-2, 0) pixels; the ratio reproj_s / reproj_d = 0.5 everywhere (lobes warped identically, not blended)

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

## Phase T4: Temporal Residual Network — Training ✅

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

#### 1. Sequential training data generation — ✅ COMPLETE

> **All training data prerequisites are ✅ COMPLETE:**
> - Sessions 1–2 (camera path recording in `monti_view`, viewpoint JSON with `path_id`+`frame`) ✅
> - Session 3 (sequential rendering in `monti_datagen` with temporal state continuity, `ResetTemporalState()` on path boundaries) ✅
> - Session 4A (static pre-cropped safetensors pipeline) ✅
> - Exposure wedge removed from `generate_training_data.py` — output filenames are `{scene}_{path_id}_{frame:04d}_input.exr` (no `_ev+N` suffix)
>
> See `camera_path_recording_spec.md` for details. Session 4B (temporal windowed crops +
> temporal dataset loader) is ✅ COMPLETE — all data pipeline prerequisites for T4 are done.
> Session 4B is scoped as data-pipeline-only (preprocess + dataset loader); the training
> loop and model code are part of T4.

**Data format:** For each path, frames are stored flat within the per-scene directory:

```
training_data/
  Sponza/
    Sponza_a1b2c3d4_0000_input.exr
    Sponza_a1b2c3d4_0000_target.exr
    Sponza_a1b2c3d4_0001_input.exr
    Sponza_a1b2c3d4_0001_target.exr
    ...
    Sponza_e5f6g7h8_0000_input.exr    ← different path, same scene
    ...
```

After `convert_to_safetensors.py`: `Sponza_a1b2c3d4_0000.safetensors`, etc.

The temporal preprocessor identifies paths by grouping on the `{scene}_{path_id}` prefix and builds sliding windows of consecutive frames.

#### 2. Temporal preprocessor — `training/scripts/preprocess_temporal.py` (Session 4B)

> **Status:** ✅ COMPLETE. 4A static crop extraction and 4B temporal windowing are both
> implemented in `preprocess_temporal.py`. Use `--window W --stride S` for temporal mode.

Extends the existing `preprocess_temporal.py` (which currently only does static crop extraction) with `--window W` and `--stride S` flags for temporal windowing.

**Design rationale:** Temporal training has an additional correctness requirement that single-frame training does not: all frames in an 8-frame window must share exactly the same crop coordinates. A `RandomCrop` applied independently per frame (as in `ExrDataset`) would destroy the spatial correspondence needed for reprojection. The preprocessor selects crop coordinates once per window and applies them identically across all 8 frames.

**Input:** Full-resolution safetensors (from `convert_to_safetensors.py`), with filenames following the pattern `{scene}_{path_id}_{frame:04d}.safetensors`.

**Output:** Pre-cropped temporal safetensors with shape `(W, 19, crop_size, crop_size)` for input and `(W, 7, crop_size, crop_size)` for target. Output files named: `{path_id}_{window_start:04d}_crop{N}.safetensors`.

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

## Phase T5: Temporal Residual Network — Inference ✅

> **Status: COMPLETE ✅**

**Goal:** Implement the temporal residual network's GPU inference, replacing the single-frame U-Net with the new temporal pipeline. Wire the reprojection output (T3) into the residual network input.

**Motivation:** This is where the quality and performance gains materialize on the GPU. The temporal residual network is ~6× smaller than the single-frame v1 model (F18), so inference is dramatically faster. And quality improves because each frame leverages the accumulated history from all previous frames. This phase also introduces the depthwise separable GLSL shaders (`depthwise_conv.comp`, `pointwise_conv.comp`) — the first model to use them on the GPU is v3.

**No retraining required.** Uses v3 weights from T4.

### Performance & Quality Estimates

- **Performance:** ~22 GFLOPS at 1080p with base_channels=32 (~15.6K params). With flat FP16 storage buffers: estimated 3-5ms on RTX 4090. With cooperative matrix (future): <0.5ms.
- **Quality:** Matches or exceeds single-frame v1 quality (121K params) despite having only ~15.6K params, because temporal accumulation provides information that a larger single-frame network can't access. The base_channels=32 choice provides headroom for edge preservation and specular handling in disoccluded regions while staying well under the 8ms inference budget.

### Tasks

#### 1. New shader for temporal input gathering

**`temporal_input_gather.comp`** — Reads 26 input channels from G-buffer images and temporal history images, writes to a flat FP16 storage buffer. This replaces the v1 `encoder_input_conv.comp` for the temporal model, but does **not** fuse the first convolution — it only gathers image data into the flat buffer format that the generic depthwise/pointwise shaders consume.

> **Design rationale:** The v3 model's `down0.conv1` is a `DepthwiseSeparableConvBlock` (depthwise 3×3 + pointwise 1×1), NOT a standard 3×3 conv like v1's `encoder_input_conv.comp`. Fusing a depthwise separable conv with 10 image reads into a single shader would be complex and hard to validate. Instead, we separate concerns: (1) gather 26 channels from images → flat buffer, (2) run generic `depthwise_conv.comp`, (3) run generic `pointwise_conv.comp`. This adds one extra dispatch but keeps each shader simple and independently testable.

Reads 26 input channels from:
- Reprojected previous diffuse: `image2D` (3ch, from T3's reprojection shader, binding 0)
- Reprojected previous specular: `image2D` (3ch, from T3's reprojection shader, binding 1)
- Disocclusion mask: `image2D` (1ch, from T3's reprojection shader, binding 2)
- Noisy diffuse: `image2D` (3ch, demodulated on-the-fly from G-buffer, binding 3)
- Noisy specular: `image2D` (3ch, demodulated on-the-fly, binding 4)
- World normals: `image2D` (3ch + roughness in .w = 4ch from G-buffer, binding 5)
- Linear depth: `image2D` (1ch from G-buffer, binding 6)
- Motion vectors: `image2D` (2ch from G-buffer, binding 7)
- Diffuse albedo: `image2D` (3ch from G-buffer, binding 8)
- Specular albedo: `image2D` (3ch from G-buffer, binding 9)
- Output buffer: flat FP16 `[26][H][W]` storage buffer (binding 10)

Channels 0-6 are temporal (reprojected_d, reprojected_s, disocclusion). Channels 7-25 are G-buffer, using the same demodulation logic as v1's `encoder_input_conv.comp` (F18 ✅): `irradiance = radiance / max(albedo, 0.001)`. On the first frame (`frame_history_.valid == false`), the reprojection shader is not dispatched; instead the temporal history images contain zeros (from the UNDEFINED→GENERAL transition clear), and the gather shader reads those zeros for channels 0-5, and 0.0 for channel 6 (disocclusion), producing the correct cold-start input.

Outputs to a flat FP16 `[26][H][W]` storage buffer — the same format that `depthwise_conv.comp` reads.

#### 2. Depthwise separable GLSL shaders — `depthwise_conv.comp`, `pointwise_conv.comp`

New shaders implementing depthwise separable convolutions on the GPU. These use the same flat FP16 `[C][H][W]` storage buffer representation as the existing v1 shaders (NOT `image2DArray` — T1 texture-backed feature maps were SKIPPED).

**`depthwise_conv.comp`** — Depthwise 3×3 convolution (groups = channels):

```glsl
layout(local_size_x = 16, local_size_y = 16) in;

layout(constant_id = 0) const uint CHANNELS = 32;

layout(push_constant) uniform PushConstants { uint width; uint height; };

layout(set = 0, binding = 0) readonly buffer InputBuffer { float16_t data_in[]; };
layout(set = 0, binding = 1) writeonly buffer OutputBuffer { float16_t data_out[]; };
layout(set = 0, binding = 2) readonly buffer WeightBuffer { float weights[]; };
// Weight layout: [CHANNELS][1][3][3] — NO bias (PyTorch depthwise has bias=False)
// Total weight count: CHANNELS * 9

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= width || y >= height) return;

    uint hw = width * height;

    for (uint ch = 0; ch < CHANNELS; ++ch) {
        float sum = 0.0;
        uint w_base = ch * 9;

        for (uint ky = 0; ky < 3; ++ky) {
            for (uint kx = 0; kx < 3; ++kx) {
                int sx = int(x) + int(kx) - 1;
                int sy = int(y) + int(ky) - 1;

                if (sx >= 0 && sx < int(width) && sy >= 0 && sy < int(height)) {
                    sum += float(data_in[ch * hw + sy * width + sx])
                         * weights[w_base + ky * 3 + kx];
                }
            }
        }

        // No bias — depthwise conv has bias=False in PyTorch DepthwiseSeparableConvBlock
        data_out[ch * hw + y * width + x] = float16_t(clamp(sum, -65504.0, 65504.0));
    }
}
```

**`pointwise_conv.comp`** — 1×1 convolution (channel mixing only):

```glsl
layout(local_size_x = 16, local_size_y = 16) in;

layout(constant_id = 0) const uint IN_CHANNELS = 32;
layout(constant_id = 1) const uint OUT_CHANNELS = 32;

layout(push_constant) uniform PushConstants { uint width; uint height; };

layout(set = 0, binding = 0) readonly buffer InputBuffer { float16_t data_in[]; };
layout(set = 0, binding = 1) writeonly buffer OutputBuffer { float16_t data_out[]; };
layout(set = 0, binding = 2) readonly buffer WeightBuffer { float weights[]; };
// Weight layout: [OUT_CHANNELS][IN_CHANNELS][1][1] followed by bias[OUT_CHANNELS]
// Total weight count: OUT_CHANNELS * IN_CHANNELS + OUT_CHANNELS

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= width || y >= height) return;

    uint hw = width * height;
    uint pixel_idx = y * width + x;

    for (uint oc = 0; oc < OUT_CHANNELS; ++oc) {
        float sum = 0.0;
        uint w_base = oc * IN_CHANNELS;

        for (uint ic = 0; ic < IN_CHANNELS; ++ic) {
            sum += float(data_in[ic * hw + pixel_idx]) * weights[w_base + ic];
        }

        // Bias (pointwise conv has bias=True in PyTorch DepthwiseSeparableConvBlock)
        sum += weights[OUT_CHANNELS * IN_CHANNELS + oc];

        data_out[oc * hw + pixel_idx] = float16_t(clamp(sum, -65504.0, 65504.0));
    }
}
```

> **Weight naming convention for v3 model:** Each `DepthwiseSeparableConvBlock` named `{layer}` produces:
> - `{layer}.depthwise.weight` — shape `[C_in, 1, 3, 3]`, **no bias**
> - `{layer}.pointwise.weight` — shape `[C_out, C_in, 1, 1]`
> - `{layer}.pointwise.bias` — shape `[C_out]`
> - `{layer}.norm.weight` — shape `[C_out]` (GroupNorm gamma)
> - `{layer}.norm.bias` — shape `[C_out]` (GroupNorm beta)
>
> This differs from v1's `ConvBlock` naming: `{layer}.conv.weight` shape `[C_out, C_in, 3, 3]`, `{layer}.conv.bias`, `{layer}.norm.weight`, `{layer}.norm.bias`.

For depthwise separable layers, the dispatch sequence is:
```
depthwise_conv.comp (3×3, C→C, no bias)  →  pointwise_conv.comp (1×1, C→C_out, with bias)  →  group_norm_reduce + group_norm_apply (with activation)
```

Three dispatches instead of one per ConvBlock (vs two for v1: conv + group_norm), but each dispatch is much cheaper due to depthwise separable factorization.

#### 3. Temporal output shader — `temporal_output_conv.comp`

Fused output shader that reads from the last flat FP16 feature buffer and produces: 1×1 conv → 7 channels, sigmoid blend weight, disocclusion forcing, per-lobe temporal blending, albedo remodulation, AND writes demodulated irradiance to history images. This replaces the v1 `output_conv.comp` for the temporal model.

**Bindings (11 total):**
```glsl
layout(local_size_x = 16, local_size_y = 16) in;

layout(constant_id = 0) const uint IN_CHANNELS = 32;

layout(push_constant) uniform PushConstants { uint width; uint height; };

// Network output
layout(set = 0, binding = 0) readonly buffer InputBuffer { float16_t data_in[]; };
layout(set = 0, binding = 1) readonly buffer WeightBuffer { float weights[]; };
// Weight layout: out_conv.weight [7][IN_CHANNELS][1][1] + out_conv.bias [7]

// Final output
layout(set = 0, binding = 2, rgba16f) uniform writeonly image2D output_image;

// Temporal blending inputs (from reproject.comp / frame history)
layout(set = 0, binding = 3, rgba16f) uniform readonly image2D reprojected_diffuse;
layout(set = 0, binding = 4, rgba16f) uniform readonly image2D reprojected_specular;
layout(set = 0, binding = 5, r16f)    uniform readonly image2D disocclusion_mask;

// Remodulation inputs (from G-buffer)
layout(set = 0, binding = 6, rgba16f) uniform readonly image2D noisy_diffuse;   // .a = hit mask
layout(set = 0, binding = 7, rgba16f) uniform readonly image2D diffuse_albedo;
layout(set = 0, binding = 8, rgba16f) uniform readonly image2D specular_albedo;

// Demodulated history output (for next frame's reprojection)
layout(set = 0, binding = 9,  rgba16f) uniform writeonly image2D history_diffuse;
layout(set = 0, binding = 10, rgba16f) uniform writeonly image2D history_specular;
```

**Shader body:**
```glsl
void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= width || y >= height) return;

    ivec2 pos = ivec2(x, y);
    uint hw = width * height;
    uint pixel_idx = y * width + x;
    const uint OUT_CH = 7;

    // 1×1 conv: IN_CHANNELS → 7 output channels
    float sums[7];
    for (uint oc = 0; oc < OUT_CH; ++oc) {
        float sum = 0.0;
        uint w_base = oc * IN_CHANNELS;
        for (uint ic = 0; ic < IN_CHANNELS; ++ic) {
            sum += float(data_in[ic * hw + pixel_idx]) * weights[w_base + ic];
        }
        sums[oc] = sum + weights[OUT_CH * IN_CHANNELS + oc];  // bias
    }

    vec3 delta_d = vec3(sums[0], sums[1], sums[2]);    // Diffuse correction delta
    vec3 delta_s = vec3(sums[3], sums[4], sums[5]);    // Specular correction delta
    float weight = 1.0 / (1.0 + exp(-clamp(sums[6], -20.0, 20.0)));  // Sigmoid, clamped for FP16 stability

    // Force weight=1 for disoccluded pixels
    float disocc = imageLoad(disocclusion_mask, pos).r;
    weight = max(weight, 1.0 - disocc);

    // Apply per-lobe temporal blending
    vec3 reproj_d = imageLoad(reprojected_diffuse, pos).rgb;
    vec3 reproj_s = imageLoad(reprojected_specular, pos).rgb;
    vec3 denoised_d = reproj_d + weight * delta_d;
    vec3 denoised_s = reproj_s + weight * delta_s;

    // Write demodulated irradiance to frame history (BEFORE remodulation)
    // Next frame's reproject.comp will warp these for temporal accumulation
    imageStore(history_diffuse, pos, vec4(denoised_d, 1.0));
    imageStore(history_specular, pos, vec4(denoised_s, 1.0));

    // Albedo remodulation (F18): per-lobe irradiance × albedo → radiance
    vec3 albedo_d = imageLoad(diffuse_albedo, pos).rgb;
    vec3 albedo_s = imageLoad(specular_albedo, pos).rgb;
    float hit = imageLoad(noisy_diffuse, pos).a;

    const float DEMOD_EPS = 0.001;
    vec3 final_d = (hit > 0.5) ? denoised_d * max(albedo_d, vec3(DEMOD_EPS)) : denoised_d;
    vec3 final_s = (hit > 0.5) ? denoised_s * max(albedo_s, vec3(DEMOD_EPS)) : denoised_s;
    vec3 final_rgb = final_d + final_s;

    imageStore(output_image, pos, vec4(final_rgb, 1.0));
}
```

This fuses the per-lobe blending, remodulation, and history write into the output shader — no extra dispatches needed. Writing the denoised irradiance (before remodulation) to `history_diffuse` / `history_specular` ensures that reprojection in the next frame operates in demodulated space. This replaces the v1 approach of `CopyImageToHistory(cmd, output_image, ...)` which copied remodulated RGB.

#### 4. Update `Infer()` dispatch sequence

The new sequence for temporal inference (v3 path, selected when `model_version_ == kV3_Temporal`):

```cpp
void MlInference::InferV3Temporal(VkCommandBuffer cmd, const DenoiserInput& input,
                                   VkImageView output_view, VkImage output_image) {
    uint32_t c0 = level0_channels_;  // base_channels (e.g. 32)
    uint32_t c1 = level1_channels_;  // base_channels * 2 (e.g. 64)
    uint32_t w0 = width_, h0 = height_;
    uint32_t w1 = DivCeil(w0, 2), h1 = DivCeil(h0, 2);

    // 1. Reproject previous output (if valid)
    if (frame_history_.valid) {
        DispatchReproject(cmd, input);
        // Barriers on reprojected_d, reprojected_s, disocclusion_mask
    }

    // 2. Gather 26-ch temporal input from images → flat buffer
    //    Reads: 7 G-buffer images + 3 temporal history images → buf_input_26ch
    DispatchTemporalInputGather(cmd, input);  // → buf_input_ [26][H][W]
    InsertBufferBarrier(cmd);

    // 3. Encoder level 0: down0.conv1 (DepthwiseSeparableConvBlock)
    //    down0.conv1.depthwise: [26,1,3,3], no bias
    DispatchDepthwiseConv(cmd, buf_input_.buffer, buf0_a_.buffer,
                          "down0.conv1.depthwise", /*channels=*/26, w0, h0);
    //    down0.conv1.pointwise: [c0,26,1,1] + bias[c0]
    DispatchPointwiseConv(cmd, buf0_a_.buffer, buf0_b_.buffer,
                          "down0.conv1.pointwise", /*in_ch=*/26, /*out_ch=*/c0, w0, h0);
    //    down0.conv1.norm: GroupNorm + LeakyReLU
    DispatchGroupNorm(cmd, buf0_b_.buffer, "down0.conv1.norm", c0, w0, h0);

    // 4. down0.conv2 (DepthwiseSeparableConvBlock)
    DispatchDepthwiseConv(cmd, buf0_b_.buffer, buf0_a_.buffer,
                          "down0.conv2.depthwise", c0, w0, h0);
    DispatchPointwiseConv(cmd, buf0_a_.buffer, buf0_b_.buffer,
                          "down0.conv2.pointwise", c0, c0, w0, h0);
    DispatchGroupNorm(cmd, buf0_b_.buffer, "down0.conv2.norm", c0, w0, h0);

    // Save skip0+ downsample to level 1
    { VkBufferCopy copy{}; copy.size = buf0_b_.size_bytes;
      dispatch_.vkCmdCopyBuffer(cmd, buf0_b_.buffer, skip0_.buffer, 1, &copy); }
    DispatchDownsample(cmd, buf0_b_.buffer, buf1_a_.buffer, c0, w0, h0);

    // 5. Bottleneck at H/2 × W/2 (only 2 levels — no level 2)
    //    bottleneck1: c0 → c1
    DispatchDepthwiseConv(cmd, buf1_a_.buffer, buf1_b_.buffer,
                          "bottleneck1.depthwise", c0, w1, h1);
    DispatchPointwiseConv(cmd, buf1_b_.buffer, buf1_a_.buffer,
                          "bottleneck1.pointwise", c0, c1, w1, h1);
    DispatchGroupNorm(cmd, buf1_a_.buffer, "bottleneck1.norm", c1, w1, h1);

    //    bottleneck2: c1 → c1
    DispatchDepthwiseConv(cmd, buf1_a_.buffer, buf1_b_.buffer,
                          "bottleneck2.depthwise", c1, w1, h1);
    DispatchPointwiseConv(cmd, buf1_b_.buffer, buf1_a_.buffer,
                          "bottleneck2.pointwise", c1, c1, w1, h1);
    DispatchGroupNorm(cmd, buf1_a_.buffer, "bottleneck2.norm", c1, w1, h1);

    // 6. Decoder level 0: upsample + concat skip0
    DispatchUpsampleConcat(cmd, buf1_a_.buffer, skip0_.buffer,
                           concat0_.buffer, c1, c0, w0, h0);

    //    up0.conv1: (c1+c0) → c0
    DispatchDepthwiseConv(cmd, concat0_.buffer, buf0_a_.buffer,
                          "up0.conv1.depthwise", c1 + c0, w0, h0);
    DispatchPointwiseConv(cmd, buf0_a_.buffer, buf0_b_.buffer,
                          "up0.conv1.pointwise", c1 + c0, c0, w0, h0);
    DispatchGroupNorm(cmd, buf0_b_.buffer, "up0.conv1.norm", c0, w0, h0);

    //    up0.conv2: c0 → c0
    DispatchDepthwiseConv(cmd, buf0_b_.buffer, buf0_a_.buffer,
                          "up0.conv2.depthwise", c0, w0, h0);
    DispatchPointwiseConv(cmd, buf0_a_.buffer, buf0_b_.buffer,
                          "up0.conv2.pointwise", c0, c0, w0, h0);
    DispatchGroupNorm(cmd, buf0_b_.buffer, "up0.conv2.norm", c0, w0, h0);

    // 7. Temporal output (fused: 1×1 conv → 7ch, sigmoid, blend, remodulate, history write)
    DispatchTemporalOutputConv(cmd, buf0_b_.buffer, output_view, output_image, input);

    // 8. Save current depth for next frame (denoised irradiance written by output shader)
    if (input.linear_depth_image != VK_NULL_HANDLE) {
        CopyImageToHistory(cmd, input.linear_depth_image, frame_history_.prev_depth, w0, h0);
    }
    frame_history_.valid = (output_image != VK_NULL_HANDLE &&
                            input.linear_depth_image != VK_NULL_HANDLE);
}
```

Total dispatches: ~18 (gather + 5×depthwise + 5×pointwise + 5×groupnorm + downsample + upsample_concat + output). More dispatches than v1's ~20, but each is dramatically cheaper due to depthwise separable factorization and smaller channel counts at a 2-level architecture.

> **Buffer allocation for v3:** The v3 path requires a `buf_input_` buffer at level 0 resolution with 26 channels (for the gathered temporal input). Level 2 buffers (`buf2_a_`, `buf2_b_`) and `skip1_`/`concat1_` are NOT needed (only 2-level U-Net). `Resize()` should allocate based on `model_version_`.

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

// Detected from weight layer names in InferArchitectureFromWeights():
//
// V1 (3-level, standard convolutions):
//   - Has "down0.conv1.conv.weight" shape [c0, 19, 3, 3]
//   - Has "down1" layers (3-level encoder)
//   - base_channels = shape[0] of "down0.conv1.conv.weight"
//   - level0=c0, level1=c0*2, level2=c0*4
//
// V3 (2-level, depthwise separable, temporal):
//   - Has "down0.conv1.depthwise.weight" shape [26, 1, 3, 3]
//   - Has "down0.conv1.pointwise.weight" shape [c0, 26, 1, 1]
//   - Does NOT have "down1" layers (2-level encoder)
//   - base_channels = shape[0] of "down0.conv1.pointwise.weight"
//   - level0=c0, level1=c0*2 (no level2)
//   - kInputChannels = 26 (not 19)
//   - kOutputChannels = 7 (not 6) — 3ch delta_d + 3ch delta_s + 1ch blend weight
```

The existing `InferArchitectureFromWeights()` currently only checks for `down0.conv1.conv.weight` (v1). It must be extended to also check for `down0.conv1.depthwise.weight` (v3) and set `model_version_` accordingly. The `Infer()` method then branches on `model_version_` to select the v1 or v3 dispatch sequence.

This allows loading any model version without manual configuration.

#### 7. Update `export_weights.py` for temporal model support

The existing `training/scripts/export_weights.py` only handles `DeniUNet` (v1). It must be updated to auto-detect the model type from the checkpoint and instantiate the correct model class.

**Changes:**
- Read `checkpoint['model_config']['type']` to determine model class:
  - `'temporal_residual'` → `DeniTemporalResidualNet(base_channels=cfg['base_channels'])`
  - Default / missing → `DeniUNet(in_channels=cfg['in_channels'], ...)`
- Import `DeniTemporalResidualNet` from `deni_train.models.temporal_unet`
- The `.denimodel` binary format itself does NOT change — it serializes all `state_dict` entries by name and shape. The v3 model simply has different layer names (e.g. `down0.conv1.depthwise.weight` instead of `down0.conv1.conv.weight`). The C++ `WeightLoader` is format-agnostic.
- ONNX export: update the `export_onnx()` call to use 26 input channels for temporal models

**CLI (unchanged):**
```bash
python scripts/export_weights.py \
    --checkpoint configs/checkpoints/model_best.pt \
    --output models/deni_v3_temporal.denimodel \
    --install
```

#### 8. Update `generate_golden_reference.py` for temporal model

The existing `tests/generate_golden_reference.py` generates a golden reference `.bin` for single-frame v1 (19-ch input, v1 model, 3-ch remodulated output). It must be extended to support v3 temporal golden references.

**Changes for single-frame v3 golden (cold start):**
- Detect model type from `.denimodel` weights (same heuristic as C++: presence of `down0.conv1.depthwise.weight`)
- Create `DeniTemporalResidualNet` instead of `DeniUNet`
- Generate 26-ch input: 7 temporal channels (zeros for reprojected_d/s, 0.0 for disocclusion = all disoccluded) + 19ch G-buffer
- Model outputs 6 channels (denoised demodulated irradiance), not 3ch remodulated RGB
- The golden `.bin` stores the 6ch demodulated output as the expected reference; the C++ test compares the denoised irradiance BEFORE remodulation (remodulation is tested separately)

**Updated golden reference binary format (version 2):**
```
[4B magic] "GREF"
[4B version] 2
[4B width, height, base_channels]
[4B model_type]  — NEW: 0 = v1_single_frame, 1 = v3_temporal
[4B denimodel_size]
[denimodel_size bytes] embedded .denimodel file

[4B num_input_images]  — 7 for v1, 10 for v3 (adds reprojected_d/s, disocclusion)
For each input image:
  [4B channels]
  [4B format] VkFormat enum
  [channel×width×height×2 bytes] FP16 data

[4B num_output_channels]  — 3 for v1 (remodulated RGB), 6 for v3 (demod irradiance)
[num_output_channels×width×height×4 bytes] FP32 expected output (channel-major)
```

**Multi-frame golden reference (new `generate_temporal_golden_reference.py`):**
- Generates a 3-frame sequence: frame 0 (cold start), frame 1-2 (with history)
- Each frame's expected output stored sequentially
- The PyTorch `_build_temporal_input()` and `reproject()` utilities handle the autoregressive state
- Stores per-frame expected outputs so the C++ test can validate each frame independently

**C++ golden test updates (`ml_inference_numerical_test.cpp`):**
- Load version 2 golden `.bin`, detect model type from `model_type` field
- For v3: create 26-ch input (bind 10 images instead of 7), compare 6ch demod irradiance output
- Multi-frame test: run `Infer()` 3 times, compare each frame's output against stored reference

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
