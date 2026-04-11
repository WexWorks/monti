# Temporal Model Improvement Plan (Post-v1)

**Context:** First temporal model trained at 32 base_channels, 6,452 samples, 15,633 parameters.
Input images are 4 SPP. Primary artifacts: blotchiness, hue shifts, and permanent motion ghosting.

---

## 1. Ghosting

### Root Cause

The model has no training signal that tells it *when* history is wrong. The blend weight `w`
only learns to minimize the reconstruction + temporal consistency losses. If `w` learns to
heavily trust history (low value → `reprojected + small_delta`), it minimizes both losses
simultaneously — but any bad reprojection gets permanently baked in with no recovery mechanism.
The disocclusion-mask prior (`weight = max(weight, 1 - disocclusion)`) forces `w→1` at
depth discontinuities but is blind to lateral ghosting from moving geometry.

---

### Fix A — Supervised Blend Weight Loss

**Question: any manually-set parameters?**

Yes, one: the error threshold that separates "reprojection was good" from "reprojection was
bad." Everything else is computed dynamically from training data.

The threshold is a hyperparameter applied to the per-pixel reprojection MAE in
ACES-tonemapped demodulated space, which has a bounded range of [0, 1]. A good default is
**0.05** (i.e., ~5% of the tonemapped dynamic range). Add to `temporal.yaml`:

```yaml
loss:
  lambda_l1: 1.0
  lambda_perceptual: 0.1
  lambda_temporal: 0.5
  lambda_blend_weight: 0.5        # new: weight on supervised blend supervision
  blend_weight_threshold: 0.05    # new: reprojection MAE above this → force w=1
```

**Implementation in `train_temporal.py` — inside `_process_sequence()`:**

After `reprojected_d, valid_mask = reproject(...)` and before computing `frame_loss`:

```python
# Supervised blend weight: teach the network to fire w=1 wherever reprojection
# was inaccurate, regardless of whether depth-based disocclusion fired.
if t > 0 and lambda_blend_weight > 0:
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        # GT reprojection error in tonemapped space (gradient does not flow through target)
        repro_err_d = torch.abs(
            aces_tonemap(reprojected_d) - aces_tonemap(target[:, :3])
        ).mean(dim=1, keepdim=True).detach()  # (B, 1, H, W)
        repro_err_s = torch.abs(
            aces_tonemap(reprojected_s) - aces_tonemap(target[:, 3:6])
        ).mean(dim=1, keepdim=True).detach()
        
        # Where reprojection is bad → GT says use current frame (w should be 1)
        gt_weight = torch.max(
            (repro_err_d > blend_weight_threshold).float(),
            (repro_err_s > blend_weight_threshold).float(),
        )  # (B, 1, H, W)
        
        # predicted_weight must be extracted from the model's internal output.
        # See architecture change below.
        blend_loss = F.binary_cross_entropy(predicted_weight, gt_weight)
        frame_losses.append(lambda_blend_weight * blend_loss)
```

**Architecture change required:** The current `DeniTemporalResidualNet.forward()` computes the
blend weight internally and does not return it. To compute the supervised loss, the training
code needs access to the raw sigmoid weight. Two options:

- **Option A (recommended):** Return `(denoised, blend_weight)` from `forward()` and update
  all callers. Training gets the weight; inference ignores it (already consumed in blending).
- **Option B:** Add a `training_forward()` method that returns the extra tensor.

Option A is cleaner. Change the forward signature in `temporal_unet.py`:

```python
def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (denoised (B,6,H,W), blend_weight (B,1,H,W))."""
    ...
    return torch.cat([denoised_d, denoised_s], dim=1), weight
```

Update `evaluate_temporal.py` callers to unpack `denoised, _ = model(temporal_input)`.

**No new GPU-side changes needed** — the blend weight is consumed inside the model and never
appears in the `.denimodel` weights as a separate output. The GPU shader already reads the
6-channel denoised output. The supervised loss is purely a training-time signal.

---

### Fix B — Velocity-Based Blend Weight Floor

**Question: what are `18:20` in the example code?**

These are **architecture constants**, not parameters, derived from the 26-channel temporal
input layout:

```
Temporal input channel layout (26ch):
  [0-2]   reprojected diffuse irradiance
  [3-5]   reprojected specular irradiance
  [6]     disocclusion mask
  [7-25]  G-buffer (19ch):
    G-buf [0-2]  = input[7-9]   — demod diffuse irradiance
    G-buf [3-5]  = input[10-12] — demod specular irradiance
    G-buf [6-8]  = input[13-15] — world normals XYZ
    G-buf [9]    = input[16]    — roughness
    G-buf [10]   = input[17]    — linear depth
    G-buf [11-12]= input[18-19] — motion vectors XY  ← these are indices 18:20
    G-buf [13-15]= input[20-22] — diffuse albedo
    G-buf [16-18]= input[23-25] — specular albedo
```

`slice(18, 20)` (Python) = `x[:, 18:20]` (PyTorch indexing). Already defined as `_CH_MOTION`
in `train_temporal.py`. The implementation should use that constant:

```python
# In temporal_unet.py, import is not needed — pass as pre-extracted channels,
# OR simply use it at the model level. Recommended: do it in the forward pass
# using the known slice, with a doc comment referencing the layout table.
mv = x[:, 18:20]  # motion vectors XY — input channels 18-19 per layout table
mv_magnitude = torch.norm(mv, dim=1, keepdim=True)
velocity_prior = torch.clamp(mv_magnitude / kMaxMV, 0.0, 1.0)
weight = torch.max(weight, torch.max(1.0 - disocclusion, velocity_prior))
```

`kMaxMV` is the normalization scale for motion vectors (in pixels, normalized to [0,1] screen
space, then scaled by resolution). A screen-space motion of **0.05** (5% of frame width) is a
reasonable saturation point where ghosting always becomes visually obvious. Add to the model
constructor:

```yaml
model:
  type: temporal_residual
  base_channels: 32
  max_mv_for_weight: 0.05   # new: motion magnitude at which velocity prior saturates to 1.0
```

These are not independent parameters to be tuned — they encode domain knowledge about the
rendering setup and should be set once and documented, not swept in hyperparameter search.

---

### Fix C — Longer Training Sequences

**Question: how to coordinate across scripts?**

The window size flows through three places:

| Script | Role | How it uses window size |
|---|---|---|
| `preprocess_temporal.py` | Data generation | CLI `--window W` bakes W into the `.safetensors` shape `(W, C, H, W)` |
| `temporal_safetensors_dataset.py` | Data loading | Reads W automatically from tensor shape — **no config needed** |
| `train_temporal.py` | Training | Reads W from data at runtime: `B, W, C_in, H, W_spatial = inp_seq.shape` — **no config needed** |
| `MlInference` (GPU) | Inference | Processes one frame at a time — **W is irrelevant** |

The only script that requires an explicit window size is `preprocess_temporal.py`. The
training and dataset scripts adapt automatically. Therefore:

- **Do not** add `window_size` to `temporal.yaml` — the training code already reads it from data.
- **Do** document the expected window size in a comment in `temporal.yaml` as a reminder of
  what the training data was generated with:

```yaml
data:
  data_dir: "D:/training_data_temporal_st"
  data_format: "temporal_safetensors"
  # Window size is baked into the data by preprocess_temporal.py --window N
  # Current data was generated with --window 8. Regenerate data to change this.
  batch_size: 4
  num_workers: 4
```

**To extend to 16 frames:** Regenerate training data with `--window 16 --stride 8`. The
training loop runs with the new W automatically. GPU inference is unchanged. VGG perceptual
loss inside `DenoiserLoss` pads spatial dims for VGG but does not require any changes — it
already operates per-frame.

**GPU inference (MlInference):** Entirely frame-by-frame. History is one frame deep (the
previous denoised output and its depth). There is no concept of a training window at inference
time. The benefit of longer training sequences is that the model learns to *recover* from
accumulated errors — ghosts that need 10+ frames to flush only appear in training data if the
window is >= 10.

---

## 2. Blotchiness

### Why 3 Levels and Not 4?

**Spatial divisibility at 1080p:**

| U-Net Levels | Required divisor | 1920 divisible? | 1080 divisible? |
|---|---|---|---|
| 2 (current) | 4 | ✓ (480×4) | ✓ (270×4) |
| 3 | 8 | ✓ (240×8) | ✓ (135×8) |
| 4 | 16 | ✓ (120×16) | **✗** (67.5×16) |

1080 is divisible by 8 but **not** by 16. A 4-level U-Net at 1080p requires padding the height
to 1088 (+8 pixels) before inference and cropping after. This creates cascading changes:

- `evaluate_temporal.py` currently pads to a multiple of 4; must change to 16.
- The GPU `MlInference` allocates `FeatureBuffer` levels with exact `width/2^n` dimensions.
  Padding must happen before `DispatchEncoderInputConv()` or the level buffers must be
  over-allocated. This requires changes to `MlInference.cpp`, the downsampling/upsampling
  shaders, and the output crop.
- History images (`denoised_diffuse`, `prev_depth`, etc.) are allocated at `render_width ×
  render_height`. Adding padding means either the history dimensions diverge from the render
  dimensions, or the entire frame history system is resized.
- The golden reference test uses `32×32` which is already divisible by 16, so tests still
  pass, meaning the bug only manifests at rendering resolution — making it easy to miss.

A 3-level U-Net avoids all of this. 1080 / 8 = 135 (exact), 1920 / 8 = 240 (exact). The only
change needed is updating `evaluate_temporal.py`'s pad-to-multiple from 4 to 8, and adding one
additional level to `MlInference`. 

**Receptive field improvement (3-level vs 2-level):**

- Current 2-level: two 3×3 convs at full res + two 3×3 convs at H/2. Effective receptive field
  at output ≈ 10px.
- 3-level: adds a full 2-conv block at H/4. Adding one more downsample step roughly doubles the
  effective receptive field to ~20px, which is sufficient to reason about blotch patterns
  spanning 20-40px.
- 4-level adds a block at H/8 for ~40px — likely overkill for residual denoising, and not
  worth the integration cost described above.

**Parameter cost:**

With base_channels=32:
- Current (2-level): ~15.6k parameters
- 3-level adds: `DownBlock(c, c*2)` + `UpBlock(c*2, c*2, c*2)` + intermediate skips
  ≈ +22k parameters → ~38k total (well within a 64k budget)
- 4-level would add another block at ~48k → ~63k total

---

## 3. Hue Shifts

### Demodulation/Remodulation Mismatch (Bug)

**Status: confirmed mismatch between training and inference.**

**Python demodulation (identical in both `exr_dataset.py` and `preprocess_temporal.py`):**
```python
_DEMOD_EPS = 0.001
input_arrays[0:3] = input_arrays[0:3] / np.maximum(albedo_d, _DEMOD_EPS)
input_arrays[3:6] = input_arrays[3:6] / np.maximum(albedo_s, _DEMOD_EPS)
```

**GLSL encoder (`encoder_input_conv.comp`):**
```glsl
const float DEMOD_EPS = 0.001;
// ...
return raw / max(a, DEMOD_EPS);  // ✓ matches Python
```

**Training loss remodulation (`denoiser_loss.py`):**
```python
pred_radiance = predicted[:, :3] * albedo_d + predicted[:, 3:6] * albedo_s
#                                ↑ no epsilon clamp — BUG
```

**GLSL output shader (`output_conv.comp`):**
```glsl
const float DEMOD_EPS = 0.001;
diffuse_rad  *= max(albedo_d, vec3(DEMOD_EPS));   // has epsilon clamp
specular_rad *= max(albedo_s, vec3(DEMOD_EPS));   // has epsilon clamp
```

**Effect:** For near-black materials (albedo < 0.001), the training loss remodulates by ~0,
making those pixels invisible to the perceptual loss. But at inference the GPU uses
`max(albedo, 0.001)`, so the model's output in those regions is multiplied differently than it
was trained to expect. The result is a systematic hue bias on very dark or dielectric materials
with near-zero specular F0.

**Fix (`denoiser_loss.py`):**

```python
_DEMOD_EPS = 1e-3  # must match GPU DEMOD_EPS in encoder_input_conv.comp / output_conv.comp

# In DenoiserLoss.forward():
pred_radiance = (predicted[:, :3] * torch.clamp(albedo_d, min=_DEMOD_EPS) +
                 predicted[:, 3:6] * torch.clamp(albedo_s, min=_DEMOD_EPS))
tgt_radiance  = (target[:, :3]    * torch.clamp(albedo_d, min=_DEMOD_EPS) +
                 target[:, 3:6]   * torch.clamp(albedo_s, min=_DEMOD_EPS))
```

This is a one-line change and should be implemented regardless of other improvements. It costs
nothing and eliminates a systematic inconsistency. **The GLSL shaders do not need to change.**

---

### Loss Weight Recommendations

**Current values:**
```yaml
loss:
  lambda_l1: 1.0
  lambda_perceptual: 0.1
  lambda_temporal: 0.5
```

**Analysis:**

`lambda_l1 = 1.0`: The L1 loss is computed on ACES-tonemapped demodulated irradiance, which is
bounded [0, 1]. This is a well-calibrated anchor — do not change it.

`lambda_perceptual = 0.1`: `DenoiserLoss` sums three VGG L1 losses (relu1_2: 64ch at full
spatial, relu2_2: 128ch at 1/2 spatial, relu3_3: 256ch at 1/4 spatial). The sum of these
three terms is numerically larger than a raw pixel-space L1, so 0.1 is already a deprioritized
weight. Increasing to **0.2–0.3** will improve hue fidelity. Start with 0.2.

`lambda_temporal = 0.5`: This is the flicker penalty between adjacent frames (warped L1 in
valid regions). At 0.5 relative to lambda_l1=1.0, temporal smoothness is weighted at 50% of
per-frame reconstruction quality. This is likely *too high* — it may cause the blend weight to
learn conservatism (low `w`, trust history) as the easiest way to minimize flicker. If we add
supervised blend weight loss, temporal stability becomes implicit in the supervision signal.
Recommend reducing to **0.2–0.3** once Fix A is added.

**Recommended `temporal.yaml` changes:**

```yaml
loss:
  lambda_l1: 1.0
  lambda_perceptual: 0.2       # was 0.1 — improves hue fidelity
  lambda_temporal: 0.3         # was 0.5 — reduce after adding supervised blend weight loss
  lambda_blend_weight: 0.5     # new — supervised ghosting rejection
  blend_weight_threshold: 0.05 # new — reprojection MAE threshold in tonemapped space
```

---

## 4. Training Data Scale

Current: 6,452 samples across the current scene set.
Target: ~20,000 samples.

**Recommended additional scenes (in priority order):**

1. **Sun Temple** — large indoor architectural space with strong directional light and complex
   occlusion boundaries. High motion-vector variance when walking through corridors.
2. **San Miguel** — heavy foliage with sub-pixel geometry details. Stress-tests the model's
   ability to handle high-frequency normals with aggressive disocclusion.
3. Additional Bistro viewpoint paths beyond glass patches (the current set is heavily biased
   toward the exterior).

**Preprocessing parameters for new data:**

```bash
python scripts/preprocess_temporal.py \
    --input-dir ../training_data_new/ \
    --output-dir ../training_data_temporal_st/ \
    --window 16 --stride 8 \
    --crops 4 --crop-size 384 \
    --workers 8
```

Using `--window 16` (instead of 8) requires no code changes — the training loop adapts. It
roughly doubles the memory cost per batch, so reduce `batch_size` to 2 if GPU OOM.

---

## 5. Model Capacity (base_channels)

**Current:** `base_channels = 32` → 15,633 parameters.

Increasing to 48 or 64 is reasonable once the 3-level U-Net is in place:

| base_channels | U-Net Levels | Approx. params | Notes |
|---|---|---|---|
| 32 | 2 (current) | ~15.6k | Baseline |
| 32 | 3 | ~38k | Recommended next step |
| 48 | 3 | ~85k | After 3-level validated |
| 64 | 3 | ~150k | Only if blotchiness persists at 48 |

Do not increase `base_channels` and `num_levels` simultaneously — change one at a time so that
regressions are attributable.

---

## 6. Recommended Implementation Order

| Priority | Change | Files | Risk |
|---|---|---|---|
| 1 | Fix DEMOD_EPS mismatch in loss | `denoiser_loss.py` | Very low — one-line fix |
| 2 | Supervised blend weight loss | `temporal_unet.py`, `train_temporal.py`, `temporal.yaml` | Medium — requires return value change |
| 3 | Increase `lambda_perceptual` to 0.2 | `temporal.yaml` | Very low |
| 4 | 3-level U-Net | `temporal_unet.py`, `MlInference.cpp`, `evaluate_temporal.py`, shaders | Medium |
| 5 | Velocity prior (Fix B) | `temporal_unet.py`, `temporal.yaml` | Low (training only) |
| 6 | Extend training window to 16 frames | `preprocess_temporal.py` re-run, `temporal.yaml` comment | Low (data regen required) |
| 7 | Add Sun Temple + San Miguel scenes | `generate_training_data.py`, scene files | Low |
| 8 | base_channels = 48 | `temporal.yaml` | Low (after 3-level validated) |

Items 1–3 can be done in a single training run immediately with no data regen. Items 4–6
require data regen or inference code changes and should follow once 1–3 are validated.
