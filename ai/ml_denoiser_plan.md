# ML Denoiser — Training & Deployment Plan

> **Purpose:** Full pipeline plan for training an ML denoiser in PyTorch and deploying it in the Deni denoiser library via custom GLSL compute shaders on Vulkan. Covers training data generation, network architecture, training scripts, weight export, inference shaders, and integration. Organized as sequential phases sized for single Copilot sessions.
>
> **Prerequisites:** Phase 11A ✅ (Capture Writer) and Phase 11B (GPU Readback + Headless Datagen) must be complete before starting. The monti renderer must be functional through at least Phase 10A (full render pipeline).
>
> **Relationship to existing plans:** This plan implements roadmap phases F9 (ML denoiser training pipeline) and F11 (ML denoiser deployment in Deni).
>
> **Session sizing:** Each phase (or sub-phase) is scoped to fit within a single Copilot Claude Opus 4.6 context session, following the convention in [monti_implementation_plan.md](monti_implementation_plan.md).

---

## Key Architecture Decisions

### 1. Network: Small U-Net (No Temporal, No Super-Resolution)

The initial network is a 3-level U-Net encoder-decoder with skip connections:
- **Encoder channels:** 16 → 32 → 64 (bottleneck)
- **Parameters:** ~120K (small enough for fast local training and real-time GPU inference)
- **Normalization:** Group normalization (8 groups) — single-invocation friendly for GPU inference
- **Activation:** LeakyReLU (slope 0.01)
- **No attention mechanisms** — unnecessary complexity for MVP
- **No temporal frames** — single-frame denoising only (N=1); temporal added in future phases
- **No super-resolution** — `ScaleMode::kNative` (1×) only; upscaling added in F12

This is deliberately small. The goal is to validate the full pipeline (training → export → inference → integration) with the simplest possible network. Quality improvements come from scaling up the architecture and expanding the training set in future phases.

### 2. Input Channels (13 total)

The network input matches what monti's G-buffer produces and the capture writer already saves:

| Channel | Components | EXR Channel Names | Notes |
|---|---|---|---|
| Noisy diffuse radiance | RGB (3) | `diffuse.R/G/B` | Alpha (`diffuse.A`) = geometry mask (1 = hit, 0 = miss); discarded for training |
| Noisy specular radiance | RGB (3) | `specular.R/G/B` | Alpha (`specular.A`) discarded |
| World normals | XYZ (3) | `normal.X/Y/Z` | World-space surface normal; miss sentinel = (0, 0, 1) |
| Roughness | 1 | `normal.W` | Material roughness packed in normal alpha; 0 at miss pixels |
| Linear depth | 1 | `depth.Z` | FP32 in EXR (all other input channels FP16) |
| Motion vectors | XY (2) | `motion.X/Y` | Screen-space motion |
| **Total** | **13** | |

**Excluded input channels:** `albedo_d.R/G/B` and `albedo_s.R/G/B` (diffuse and specular albedo) are written to the input EXR but not used by the network. These will be added in a future phase for demodulated denoising, where the network learns to denoise in albedo-divided space and albedo is remodulated after inference.

Output: **RGB (3 channels)** — denoised combined radiance (diffuse + specular). The network learns to combine and denoise both lobes. Albedo remodulation is deferred to a future phase (requires the network to learn demodulated denoising, which adds training complexity).

### 3. Inference: Custom GLSL Compute Shaders (Not ncnn)

The deni `Denoiser::Denoise(VkCommandBuffer cmd, ...)` API records GPU commands into a caller-provided command buffer. This is a fundamental design contract — the caller owns the command buffer lifecycle, and the denoiser slots into the existing recording.

**ncnn cannot record into an external command buffer.** Its Vulkan backend manages its own command buffers and queue submissions internally. Using ncnn would require either breaking deni's API contract or inserting fence-waits mid-frame, adding latency and architectural complexity.

**Custom GLSL compute shaders** record directly into the caller's command buffer via `vkCmdDispatch`, preserving the existing API perfectly. The small U-Net requires 7 compute shaders:

| Shader | Purpose |
|---|---|
| `encoder_input_conv.comp` | First encoder Conv3×3 reading directly from G-buffer image views (13→16ch) |
| `conv.comp` | Conv3×3 + bias between feature buffers (all other encoder/decoder/bottleneck convolutions) |
| `group_norm_reduce.comp` | Partial sum/sum-of-squares reduction for GroupNorm (pass 1) |
| `group_norm_apply.comp` | Finalize GroupNorm mean/var, normalize + gamma/beta + optional LeakyReLU (pass 2) |
| `downsample.comp` | 2× spatial downsampling (max pool) |
| `upsample_concat.comp` | 2× bilinear upsample + skip connection concatenation |
| `output_conv.comp` | Conv1×1 final projection → RGB output image |

Weight data is loaded from a binary file into GPU storage buffers at initialization. The inference dispatcher chains compute dispatches with pipeline barriers matching the U-Net architecture. Changing the model requires re-exporting weights from PyTorch; architecture changes require updating the dispatch sequence.

**ncnn remains the recommended approach for mobile** (future F13), where its TBDR fragment shader optimizations and automatic FP16 handling provide significant benefits, and a separate submission model is acceptable.

### 4. Training Strategy

- **Framework:** PyTorch with FP16 mixed precision (AMP)
- **Loss:** L1 + VGG perceptual, computed in ACES-tonemapped space (not linear HDR — prevents firefly gradient domination)
- **Ground truth:** High-SPP (256+) multi-frame accumulated renders from monti_datagen (no DLSS-RR dependency)
- **Patch training:** Random 256×256 crops for memory efficiency and data augmentation
- **Initial hardware:** Local NVIDIA RTX 4090 (24 GB VRAM)
- **Future hardware:** Cloud multi-GPU for hyperparameter sweeps and larger datasets

### 5. Quality Target

**MVP:** Noticeably better than passthrough (raw noisy input), acceptable for interactive preview. Not expected to match DLSS-RR quality initially. The pipeline is designed for iterative improvement — quality scales with training data breadth, network capacity, and architectural enhancements (temporal, super-resolution) in future phases.

### 6. Weight File Format

Custom binary format (`.denimodel`) — simple, no protobuf/ONNX dependency:

```
Header:
  [4 bytes]  magic: "DENI"
  [4 bytes]  version: 1
  [4 bytes]  num_layers
  [4 bytes]  total_weight_bytes

Per layer (repeated num_layers times):
  [4 bytes]  name_length
  [N bytes]  name (UTF-8, not null-terminated)
  [4 bytes]  num_dims
  [4 bytes × num_dims]  shape
  [4 bytes × product(shape)]  float32 weight data (little-endian)
```

The Python export script writes this format. The C++ weight loader reads it and uploads to GPU storage buffers. ONNX export is also provided for portability and debugging, but is not used by deni.

---

## Repository Layout

```
# ── TRAINING (Python, in monti repo) ─────────────────────────────────

training/
├── pyproject.toml                  # Package metadata + dependencies
├── requirements.txt                # Pinned dependencies for reproducibility
├── README.md                       # Quick-start guide
├── configs/
│   ├── default.yaml                # Default training hyperparameters
│   └── small_test.yaml             # Fast config for pipeline validation
├── scripts/
│   ├── generate_training_data.py   # Batch monti_datagen invocation
│   ├── generate_viewpoints.py      # Compute camera viewpoints per scene
│   ├── validate_dataset.py         # EXR channel stats + thumbnail gallery
│   ├── export_weights.py           # PyTorch → .denimodel binary export
│   └── download_scenes.py          # Download extended training scenes
├── deni_train/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── exr_dataset.py          # EXR pair dataset (input + target)
│   │   └── transforms.py           # Random crop, flip, exposure jitter
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py                 # U-Net architecture
│   │   └── blocks.py               # ConvBlock, DownBlock, UpBlock
│   ├── losses/
│   │   ├── __init__.py
│   │   └── denoiser_loss.py        # L1 + VGG perceptual loss
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── tonemapping.py          # ACES tonemap for loss computation
│   │   └── metrics.py              # PSNR, SSIM evaluation
│   ├── train.py                    # Main training entry point
│   └── evaluate.py                 # Evaluation / comparison script
└── tests/
    ├── test_dataset.py             # Dataset loader tests
    ├── test_model.py               # Forward pass shape tests
    └── test_export.py              # Weight export round-trip test

# ── INFERENCE (C++ / GLSL, in deni library) ──────────────────────────

denoise/
├── include/deni/vulkan/
│   └── Denoiser.h                  # Unchanged public API
└── src/vulkan/
    ├── Denoiser.cpp                # Updated: mode selection (passthrough vs ML)
    ├── MlInference.h               # Internal: weight loading + dispatch sequencing
    ├── MlInference.cpp             # Internal: U-Net inference pipeline
    ├── WeightLoader.h              # Internal: .denimodel file parser
    ├── WeightLoader.cpp
    └── shaders/
        ├── passthrough.comp        # Existing passthrough shader
        ├── conv_block.comp         # Conv3×3 + GroupNorm + LeakyReLU
        ├── downsample.comp         # 2× spatial downsampling
        ├── upsample_concat.comp    # 2× bilinear upsample + skip concat
        └── output_conv.comp        # Conv1×1 → RGB output

# ── TRAINING DATA (generated, gitignored) ─────────────────────────────

training_data/                      # Output from monti_datagen (gitignored)
├── cornell_box/
│   ├── frame_000000_input.exr
│   ├── frame_000000_target.exr
│   └── ...
├── damaged_helmet/
│   └── ...
└── ...

# ── TRAINED MODELS (checked in, small) ────────────────────────────────

models/                             # Exported weights (checked into repo)
├── deni_v1.denimodel               # Binary weights for inference
└── deni_v1.onnx                    # ONNX export for reference
```

---

## Phase Overview

| Phase(✅) | Deliverable | Verifiable Outcome |
|---|---|---|
| F9-1 ✅ | Python project scaffold + EXR dataset loader | Loader reads synthetic EXR pairs, produces correct tensor shapes |
| F9-2 ✅ | U-Net architecture + loss functions | Forward pass with random input produces correct output shape; loss computes without error |
| F9-3 ✅ | Training loop + export scripts | Overfit on 2 synthetic samples: loss decreases to near zero; weights export to `.denimodel` |
| F9-4 ✅ | Initial training data generation | Cornell Box exported to .glb; scenes downloaded; 15 EXR pairs generated (3 scenes × 5 exposures); validation script confirms channel integrity |
| F9-5 ✅ | First training run + quality baseline | Model trained on real data; PSNR improvement over noisy input measurable on held-out validation split; auto-generated baseline report |
| F9-5b ⏭️ | Hyperparameter sensitivity sweep (optional) | Not executed — deferred in favor of expanded dataset (F9-6). Sweep planned but no configs, checkpoints, or results produced. |
| F9-6a ✅ | Multi-viewpoint rendering in `monti_datagen` | `--position`/`--target`/`--fov` CLI args + `--viewpoints` JSON batch mode; `Scene&` in `GenerationSession`; Writer subdirectory param; `nlohmann_json` linked; scene/Vulkan resources reused across viewpoints |
| F9-6b ✅ | Extended scene downloads + viewpoint generation | 10 new Khronos GLB models + 2 multi-file glTF downloaded; viewpoint JSONs generated per scene (24 viewpoints each) |
| F9-6c ✅ | Data augmentation transforms | `RandomRotation180`, `ExposureJitter`; `RandomHorizontalFlip` removed from pipeline (world-space normals incompatible with screen flips); unit tests |
| F9-6d ✅ | Full dataset generation + validation | `generate_training_data.py` updated with `--viewpoints` + `--env` support; ~1,680 frames rendered and validated; training dataloader confirmed working |
| F9-6e ✅ | HDRI collection + lighting rigs | 5 CC0 HDRIs downloaded; `--lights` CLI in monti_datagen; `generate_light_rigs.py` for overhead + key-fill-rim rigs; 560 supplementary frames with diverse lighting |
| F9-7 ✅ | Production training run | Retrained model with full dataset; quality assessment documented |
| F11-1 ✅ | Weight loading + inference buffers in Deni | Weights loaded from `.denimodel`, GPU buffers allocated, sizes verified |
| F11-2 ✅ | GLSL inference compute shaders | Inference dispatches produce output image; correctness validated against PyTorch reference |
| F11-3 ✅ | End-to-end integration + validation | ML denoiser in monti_view; A/B comparison with passthrough; integration test passes |

### Future Phases (Outlined)

| Phase | Feature | Prerequisite |
|---|---|---|
| F11-4 | Temporal extension — training (N=2–4 frame input, frame warping) + `--camera-path` JSON support in `monti_datagen` for multi-frame temporal sequences | F11-3 |
| F11-5 | Temporal extension — inference (frame history management in deni) | F11-4 |
| F12 | Super-resolution training + inference (`ScaleMode::kQuality`, `kPerformance`); add `--target-scale` CLI to `monti_datagen` | F11-3 |
| F13 | Mobile fragment shader inference (ncnn or custom, TBDR-optimized) | F11-3 + F6 |
| F18 | Albedo demodulation — add `albedo_d`/`albedo_s` as network inputs, train in albedo-divided space, remodulate after inference | F11-3 |
| F19 | Transparency output — use `diffuse.A`/`specular.A` alpha as transparency mask (currently geometry hit mask) | Renderer alpha support |
| F20 | Cloud training scripts (multi-GPU DDP, hyperparameter sweeps) | F9-7 |
| F21 | Broader scene acquisition + stress scene generation | F9-6d |

### Key Dependencies

```
11B (datagen) ──→ F9-4 (generate data) ──→ F9-5 (first training)
                                            ↓
F9-1 (scaffold) → F9-2 (model) → F9-3 (training loop) ─────→ F9-5
                                                                ↓
                                                      F9-5b (optional sweep)
                                                                ↓
                                            F9-6a/b/c/d/e (more scenes + augmentation + lighting) → F9-7 (production training)
                                                                    ↓
                            F11-1 (weights in deni) → F11-2 (shaders) → F11-3 (integration)
```

F9-1 through F9-3 (Python infrastructure) can proceed in parallel with Phase 11B completion, using synthetic test data. F9-4 onward requires a functional monti_datagen.

---

## Phase F9-1: Python Project Scaffold + EXR Dataset Loader ✅

**Goal:** Rename EXR channel prefixes in the capture writer for conciseness, establish the Python training project structure, and implement a dataset loader that reads monti_datagen's EXR file pairs. Verify with synthetic test data.

### Tasks

0. **Rename EXR channel prefixes in `Writer.cpp` and tests** — change the string literals used as EXR channel name prefixes to shorter, ML-friendly names. This is a contained string-literal rename (no API or struct changes):

   | Old Prefix | New Prefix | File Scope |
   |---|---|---|
   | `noisy_diffuse` | `diffuse` | `Writer.cpp` (×2: `WriteFrame`, `WriteFrameRaw`) |
   | `noisy_specular` | `specular` | `Writer.cpp` (×2) |
   | `diffuse_albedo` | `albedo_d` | `Writer.cpp` (×2) |
   | `specular_albedo` | `albedo_s` | `Writer.cpp` (×2) |
   | `ref_diffuse` | `diffuse` | `Writer.cpp` (×1, target EXR only) |
   | `ref_specular` | `specular` | `Writer.cpp` (×1, target EXR only) |
   | `normal` | `normal` | *(unchanged)* |
   | `depth` | `depth` | *(unchanged)* |
   | `motion` | `motion` | *(unchanged)* |

   Also update all channel name string literals in `tests/capture_writer_test.cpp` and `tests/phase11b_test.cpp` to match. C++ struct field names (`InputFrame::noisy_diffuse`, `TargetFrame::ref_diffuse`, etc.) remain unchanged — only EXR string prefixes change.

1. Create `training/pyproject.toml` with package metadata and dependencies:
   ```toml
   [project]
   name = "deni-train"
   version = "0.1.0"
   requires-python = ">=3.10"
   dependencies = [
       "torch>=2.0",
       "torchvision>=0.15",
       "OpenEXR>=3.2",    # or tinyexr Python bindings
       "Imath>=3.1",
       "numpy>=1.24",
       "pyyaml>=6.0",
       "tensorboard>=2.12",
       "Pillow>=10.0",
   ]
   ```

2. Create `training/requirements.txt` with pinned versions for reproducibility.

3. Create `training/deni_train/data/exr_dataset.py`:
   - `ExrDataset(torch.utils.data.Dataset)` class
   - Constructor takes `data_dir` path, finds all `frame_*_input.exr` / `frame_*_target.exr` pairs
   - `__getitem__` loads an EXR pair and returns `(input_tensor, target_tensor)`
   - Input tensor channels (13) in CHW layout, loaded as `float16` (FP16) tensors:
     - `diffuse.R/G/B` (3) — noisy diffuse radiance (`.A` alpha discarded)
     - `specular.R/G/B` (3) — noisy specular radiance (`.A` alpha discarded)
     - `normal.X/Y/Z` (3) — world-space surface normals
     - `normal.W` (1) — material roughness
     - `depth.Z` (1) — linear depth (FP32 in EXR, converted to FP16 on load)
     - `motion.X/Y` (2) — screen-space motion vectors
   - Target tensor channels (3) in CHW layout, loaded as `float16`:
     - Combined radiance: `target_rgb = diffuse.R/G/B + specular.R/G/B` (sum of reference diffuse and specular from target EXR; alpha channels discarded)
   - FP16 tensors reduce memory footprint ~2× with negligible quality impact for this network size. The loss function (F9-2) upcasts to FP32 internally via AMP.
   - Handles missing pairs gracefully (skip with warning)
   - Ignores albedo channels (`albedo_d.*`, `albedo_s.*`) — reserved for future demodulated denoising phase

4. Create `training/deni_train/data/transforms.py`:
   - `RandomCrop(size)` — random spatial crop applied identically to input and target
   - `RandomHorizontalFlip(p=0.5)` — flip both input and target (negate motion vector X)
   - `Compose(transforms)` — chain transforms
   - All transforms operate on `(input, target)` tensor pairs to maintain spatial alignment

5. Create `training/scripts/generate_synthetic_data.py`:
   - Generates a small synthetic dataset (10 pairs) for pipeline validation
   - Input: random noise + smooth gradients (simulating noisy G-buffer channels)
   - Target: Gaussian-blurred version of the noisy radiance (crude ground truth proxy)
   - Writes EXR files matching monti_datagen's channel naming convention
   - This synthetic data is used only for testing the training pipeline — never for actual model training

6. Create `training/tests/test_dataset.py`:
   - Test that `ExrDataset` finds and loads synthetic EXR pairs
   - Verify input tensor shape: `(13, H, W)`
   - Verify target tensor shape: `(3, H, W)`
   - Verify `RandomCrop` produces expected spatial dimensions
   - Verify `RandomHorizontalFlip` negates motion vector X channel

### Verification
- `capture_writer_test` and `phase11b_test` pass with renamed EXR channel prefixes
- `pip install -e .` succeeds in a fresh virtual environment
- `generate_synthetic_data.py` writes 10 EXR pairs to a temp directory
- `test_dataset.py` passes: loader produces correct shapes (FP16), transforms work correctly
- No import errors, no dependency conflicts

---

## Phase F9-2: U-Net Architecture + Loss Functions ✅

**Goal:** Implement the denoiser U-Net in PyTorch and the composite loss function. Verify with random data.

### Network Architecture Detail

```
Input (13 channels, H×W)
  │
  ├─ Encoder Level 0: ConvBlock(13→16) → ConvBlock(16→16)  ─── skip_0 (16ch, H×W)
  │   └─ MaxPool 2×2
  ├─ Encoder Level 1: ConvBlock(16→32) → ConvBlock(32→32)  ─── skip_1 (32ch, H/2×W/2)
  │   └─ MaxPool 2×2
  ├─ Bottleneck: ConvBlock(32→64) → ConvBlock(64→64)       (64ch, H/4×W/4)
  │
  ├─ Upsample 2×2 → Concat(skip_1) → ConvBlock(96→32) → ConvBlock(32→32)
  ├─ Upsample 2×2 → Concat(skip_0) → ConvBlock(48→16) → ConvBlock(16→16)
  │
  └─ Conv1×1(16→3) → Output (3 channels, H×W)
```

Each `ConvBlock`: Conv2d(3×3, padding=1) → GroupNorm(8 groups) → LeakyReLU(0.01)

### Tasks

1. Create `training/deni_train/models/blocks.py`:
   - `ConvBlock(in_ch, out_ch)` — Conv2d(3×3, pad=1) + GroupNorm(min(8, out_ch)) + LeakyReLU(0.01)
   - `DownBlock(in_ch, out_ch)` — two `ConvBlock`s + MaxPool2d(2)
   - `UpBlock(in_ch, skip_ch, out_ch)` — Upsample(scale=2, bilinear) + cat(skip) + two ConvBlocks. `in_ch` is the channel count from the layer below (pre-concatenation); the first ConvBlock receives `in_ch + skip_ch` channels after concatenation with the skip connection.
   - All blocks use `nn.Module` properly for parameter registration
   - Weight initialization: Kaiming normal (`fan_out`, `leaky_relu`) for all Conv2d layers; zeros for biases. Applied via a `_init_weights()` method called in the module constructor. This is the standard initialization for LeakyReLU networks and provides better convergence than PyTorch's default Kaiming uniform.

2. Create `training/deni_train/models/unet.py`:
   - `DeniUNet(in_channels=13, out_channels=3, base_channels=16)` — `nn.Module`
   - Encoder: 2 `DownBlock`s (16→32→64 at bottleneck)
   - Bottleneck: 2 `ConvBlock`s at 64 channels
   - Decoder: 2 `UpBlock`s (64→32→16) with skip connections
   - Output: `nn.Conv2d(16, 3, kernel_size=1)` — no activation (linear HDR output)
   - `forward(x)` returns denoised RGB
   - Report parameter count via `extra_repr()` override (PyTorch-idiomatic way to extend `nn.Module.__repr__`)

3. Create `training/deni_train/utils/tonemapping.py`:
   - `aces_tonemap(x)` — ACES filmic tone mapping (matches monti's Stephen Hill fitted curve from `tonemap.comp`)
   - Implements the same RRT+ODT fit: `v = m1 @ hdr`, rational polynomial `a/b`, `result = clamp(m2 @ (a/b), 0, 1)`
   - Operates on batched tensors `(B, 3, H, W)` — applied to raw linear HDR (no exposure multiplication; exposure is a display-time concern and training data has fixed lighting baked in)
   - Used by the loss function to compute losses in perceptually uniform space
   - No changes needed to the renderer or monti_datagen — training data remains in linear HDR, and the tonemap is applied only inside the loss function for gradient weighting

4. Create `training/deni_train/losses/denoiser_loss.py`:
   - `DenoiserLoss(lambda_l1=1.0, lambda_perceptual=0.1)` — `nn.Module`
   - L1 loss: `|aces(predicted) - aces(target)|` in tonemapped space
   - Perceptual loss: VGG-16 feature matching (relu1_2, relu2_2, relu3_3) on tonemapped RGB
   - VGG input normalization: tonemapped [0,1] RGB is normalized with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` before passing to VGG feature extractor. This is the standard approach used by LPIPS and perceptual loss implementations — VGG was trained on ImageNet-normalized inputs, so feeding unnormalized values would produce misaligned feature activations.
   - VGG features extracted once, frozen, no gradient (`torch.no_grad()` + `requires_grad_(False)`)
   - Total: `lambda_l1 * L1 + lambda_perceptual * L_perceptual`
   - Tonemapping before loss prevents HDR firefly pixels from dominating gradients

5. Create `training/tests/test_model.py`:
   - Verify `DeniUNet` forward pass: input `(B=2, 13, 256, 256)` → output `(B=2, 3, 256, 256)`
   - Verify parameter count is in expected range (100K–200K)
   - Verify `DenoiserLoss` computes without NaN on random input
   - Verify loss decreases over 10 gradient steps on a fixed random batch (sanity check)

### Verification
- All tests pass
- Forward pass produces correct output shape
- Loss function computes and backpropagates without error
- Parameter count printed and within expected range

> **Note:** The actual parameter count (~120K) is smaller than the initial planning estimate of 500K–1M. This is correct for a 3-level U-Net with base_channels=16. The small size is intentional for MVP — quality improvements come from scaling up base_channels and adding encoder levels in future phases.

---

## Phase F9-3: Training Loop + Export Scripts ✅

**Goal:** Implement the training script with configuration, logging, checkpointing, and weight export. Verify by overfitting on synthetic data.

### Tasks

1. Create `training/configs/default.yaml`:
   ```yaml
   data:
     data_dir: "../training_data"
     crop_size: 256
     batch_size: 8
     num_workers: 4

   model:
     in_channels: 13
     out_channels: 3
     base_channels: 16

   loss:
     lambda_l1: 1.0
     lambda_perceptual: 0.1

   training:
     epochs: 100
     learning_rate: 1.0e-4
     weight_decay: 1.0e-5
     grad_clip_norm: 1.0     # max gradient L2 norm (standard for ML denoiser pipelines)
     lr_scheduler: cosine    # cosine annealing
     warmup_epochs: 5
     mixed_precision: true   # FP16 AMP
     checkpoint_interval: 10 # epochs
     log_interval: 50        # steps
     sample_interval: 10     # epochs between TensorBoard image samples
     seed: 42                # RNG seed for reproducibility (torch, numpy, python)

   export:
     output_dir: "../models"
   ```

2. Create `training/configs/small_test.yaml`:
   - **Standalone file** (not merged on top of `default.yaml`) — all fields present with overrides applied: `batch_size: 2`, `epochs: 50`, `crop_size: 128`, `checkpoint_interval: 25`, `sample_interval: 10`. All other fields copied from `default.yaml`. The training script loads exactly one YAML file — no config inheritance or merge mechanism.

3. Create `training/deni_train/train.py`:
   - CLI: `python -m deni_train.train --config configs/default.yaml [--resume checkpoint.pt]`
   - Loads config (YAML), creates dataset + dataloader + model + loss + optimizer (AdamW)
   - **Deterministic seeding:** Sets `torch.manual_seed()`, `numpy.random.seed()`, `random.seed()`, and `torch.backends.cudnn.deterministic = True` from `training.seed` config value. Simplifies debugging by producing reproducible training runs.
   - Training loop with:
     - FP16 mixed precision via `torch.amp.GradScaler`
     - Gradient clipping via `torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)` — standard practice for HDR data with mixed precision to prevent gradient explosions
     - Cosine annealing LR scheduler with linear warmup
     - TensorBoard logging (loss curves, learning rate, sample images every `training.sample_interval` epochs)
     - **Full state checkpointing:** saves model `state_dict`, optimizer state, LR scheduler state, `GradScaler` state, current epoch, and best validation loss. `--resume` restores all of these so training continues exactly where it left off.
     - Best-model tracking (lowest validation loss)
   - Validation split: last 10% of dataset (or separate `val_dir` if provided)
   - Sample visualization: every `training.sample_interval` epochs (default 10), log input/predicted/target image triplets to TensorBoard (tonemapped for display)

4. Create `training/deni_train/evaluate.py`:
   - CLI: `python -m deni_train.evaluate --checkpoint model_best.pt --data_dir ../training_data --output_dir results/`
   - Loads model + dataset, runs inference on full images (no cropping)
   - **Input padding:** If image dimensions are not divisible by 4 (required by 2-level U-Net with 2× MaxPool), pad with reflect mode to the next multiple of 4, run inference, then crop output back to original size
   - Computes per-image metrics: PSNR, SSIM (on tonemapped output)
   - Computes aggregate metrics: mean PSNR, mean SSIM
   - Saves side-by-side comparison PNGs to `--output_dir`: noisy input | denoised | ground truth
   - Prints results table to stdout

5. Create `training/deni_train/utils/metrics.py`:
   - `compute_psnr(predicted, target)` — PSNR in tonemapped space
   - `compute_ssim(predicted, target)` — SSIM via `torchvision.transforms.functional` or `skimage`

6. Create `training/scripts/export_weights.py`:
   - CLI: `python scripts/export_weights.py --checkpoint model_best.pt --output models/deni_v1.denimodel`
   - Loads PyTorch checkpoint, iterates `model.state_dict()`
   - Writes `.denimodel` binary format (header + per-layer name/shape/weights)
   - **Layer names:** Uses PyTorch's native `state_dict()` key names verbatim (e.g., `encoder.down0.conv1.weight`). This is the standard/idiomatic approach — the C++ weight loader in F11-1 will use these same names to map weights to compute shader buffers.
   - Also exports ONNX via `torch.onnx.export(opset_version=18)` for reference. Opset 18 provides native GroupNorm support (avoids decomposition into Reshape+InstanceNorm workarounds from older opsets). Dynamic axes for batch and spatial dimensions (`{0: 'batch', 2: 'height', 3: 'width'}`).
   - Prints layer summary: name, shape, parameter count

7. Create `training/tests/test_export.py`:
   - Create a small model, export to `.denimodel`, reload and verify weights match
   - Verify ONNX export runs without error
   - Verify `.denimodel` file is parseable (read header, verify magic/version/layer count)

### Verification
- Overfit test: use `small_test.yaml` with `data_dir` pointing to the existing `generate_synthetic_data.py` output (10 pairs). Train for 50 epochs — loss drops to < 0.01, confirming the model can memorize small data
- Checkpoint saves and resumes correctly (loss continues from where it left off — verified by `--resume`)
- TensorBoard shows loss curves and sample images at configured `sample_interval`
- `export_weights.py` produces a `.denimodel` file with correct structure
- `evaluate.py` runs with `--output_dir` and prints metrics table + saves comparison PNGs
- All tests pass

---

## Phase F9-4: Initial Training Data Generation ✅

**Goal:** Generate the first real training dataset from monti's existing test scenes using `monti_datagen`. Create validation tooling. Export the programmatic Cornell Box to `.glb` for use as a training scene.

**Prerequisite:** Phase 11B (monti_datagen functional)

> **Note:** The initial `monti_datagen` uses auto-fitted camera only (no `--position`/`--target`/`--fov` or `--viewpoints` support yet — added in F9-6a). Each invocation produces exactly one EXR pair (noisy input + accumulated reference target). Camera path support (`--camera-path` JSON for multi-frame temporal sequences) will be added in a future phase alongside temporal denoising (F11-4). For now, multiple viewpoints are achieved by running `monti_datagen` multiple times with different `--exposure` values per scene, producing a small but varied dataset.

### Tasks

1. Create `training/scripts/export_cornell_box.py`:
   - Python script using `pygltflib` to generate a Cornell Box `.glb` matching the geometry in `tests/scenes/CornellBox.cpp`
   - 7 meshes: floor, ceiling, back wall, left wall, right wall, short box, tall box
   - 4 materials: white diffuse, red diffuse, green diffuse, light emissive
   - Area light on ceiling, camera at canonical viewpoint
   - Output: `training/scenes/cornell_box.glb`
   - Add `pygltflib` to `requirements.txt` (build-time dependency for scene export only)
   - **Rationale:** monti uses cgltf (read-only). Rather than adding a C++ glTF writer to the engine, a standalone Python exporter is simpler and stays within the training toolchain.

2. Create `training/scripts/download_scenes.py`:
   - Downloads glTF sample models from the Khronos glTF-Sample-Assets GitHub repository:
     - `DamagedHelmet.glb` — PBR textures, normal maps, emissive
     - `DragonAttenuation.glb` — Transmission, volume attenuation
   - Output: `training/scenes/`
   - Validates downloads (file size sanity check, basic glTF parse via `pygltflib`)
   - Skips already-downloaded files (idempotent)
   - CLI: `python scripts/download_scenes.py [--output training/scenes/]`

3. Create `training/scripts/generate_training_data.py`:
   - Python script that invokes `monti_datagen` via `subprocess` for each scene at multiple exposure levels
   - CLI args: `--monti-datagen`, `--output`, `--scenes`, `--width`, `--height`, `--spp`, `--ref-frames`
   - Default: 960×540, 4 SPP noisy, 64 ref frames (64 × 4 = 256 effective reference SPP), `ScaleMode::kNative` (input = target resolution). **Note:** 540 is divisible by 4, which is required by the U-Net's 2-level 2× MaxPool. All training data resolutions should be divisible by 4 to avoid padding overhead during training and evaluation.
   - For each scene, runs `monti_datagen` at 5 exposure levels: −1.0, −0.5, 0.0, +0.5, +1.0 EV
   - Each invocation produces 1 EXR pair → 5 pairs per scene × 3 scenes = **15 total pairs**
   - Creates per-scene output directories, prints progress and elapsed time
   - Example invocation:
     ```bash
     python scripts/generate_training_data.py --monti-datagen ../build/app/datagen/Release/monti_datagen.exe
     ```
   - **Note:** `--position`, `--target`, `--fov`, and `--viewpoints` flags are added in F9-6a. `--camera-path` (multi-frame temporal sequences) and `--target-scale` flags do not yet exist and will be added in future phases.

4. Create `training/scripts/validate_dataset.py`:
   - CLI: `python scripts/validate_dataset.py --data_dir ../training_data`
   - Recursively finds all EXR pairs under the data directory (supports per-scene/per-exposure subdirectories)
   - For each EXR pair:
     - Verify all expected channels present (21 input channels, 8 target channels)
     - Check for NaN/Inf values (fatal — indicates renderer bug)
     - Compute per-channel statistics: min, max, mean, std
     - Flag suspiciously low variance (possible black render)
   - Generate thumbnail gallery HTML page: ACES-tonemapped (matching training loss space) input vs target side-by-side at original resolution
   - Print summary: total pairs, any issues found, channel statistics table

5. Create `training/.gitignore`:
   - Ignore `training_data/`, `data_cache/`, `scenes/` (downloaded/generated .glb files), and `results/`
   - These directories contain large binary files that should not be committed

### Verification
- `export_cornell_box.py` produces a valid `scenes/cornell_box.glb` that `monti_datagen` can load
- `download_scenes.py` downloads DamagedHelmet and DragonAttenuation without errors
- `generate_training_data.py` runs and produces 15 EXR pairs across 3 scenes × 5 exposures
- `validate_dataset.py` reports no NaN/Inf, reasonable channel statistics
- Thumbnail gallery shows recognizable noisy inputs and clean targets across exposure range
- `ExrDataset` (from F9-1) loads the generated data without errors

### Initial Training Scenes

| Scene | Source | Viewpoints × Exposures | Tests |
|---|---|---|---|
| Cornell box | `export_cornell_box.py` → `.glb` | 1 × 5 | Diffuse GI, area light, color bleeding |
| DamagedHelmet.glb | Khronos glTF-Sample-Assets | 1 × 5 | PBR textures, normal maps, emissive |
| DragonAttenuation.glb | Khronos glTF-Sample-Assets | 1 × 5 | Transmission, volume attenuation |
| **Total** | | **15 pairs** | |

This is a small dataset — sufficient for validating the full pipeline and getting a first quality signal, but far too small for production quality. The exposure variation provides brightness diversity to help the network generalize slightly beyond a single lighting condition. Multi-viewpoint rendering (via `--position`/`--target`/`--fov` CLI args and `--viewpoints` JSON batch mode) and extended scene downloads are expanded in F9-6a/b/c/d. Multi-frame temporal camera paths (`--camera-path`) are added alongside temporal denoising in F11-4.

---

## Phase F9-5: First Training Run + Quality Baseline ✅

**Goal:** Train the U-Net on real rendered data from F9-4. Establish a quality baseline and validate the full training→evaluation pipeline with real content.

**Prerequisite:** Phase F9-4 verified (15 EXR pairs generated from 3 scenes × 5 exposures, validation script passes)

### Tasks

1. Run training:
   - Use `default.yaml` config with `data_dir` pointing to F9-4 output (`../training_data`)
   - Train for 100 epochs on local RTX 4090
   - Monitor loss curves in TensorBoard (`tensorboard --logdir configs/runs/`)
   - Expected training time: ~10–30 minutes for 15 pairs with 256×256 crops (small dataset, fast iteration)
   - Checkpoints saved to `training/configs/checkpoints/` (managed by `train.py` — relative to config file location)
   - Best model saved as `training/configs/checkpoints/model_best.pt` (lowest validation loss)
   - Note: with 15 total pairs, the 10% validation split yields 2 held-out pairs and 13 training pairs

2. Evaluate quality on held-out validation split:
   - Run `evaluate.py` on the **held-out validation data only** (last 10% of dataset, same split used during training) to measure generalization rather than memorization
   - Enhance `evaluate.py` to accept a `--val-split` flag that evaluates only the last 10% of pairs (matching the training split logic in `train.py`), and a `--report` flag that generates a Markdown report
   - Example invocation:
     ```
     python -m deni_train.evaluate \
         --checkpoint configs/checkpoints/model_best.pt \
         --data_dir ../training_data \
         --output_dir results/v1_baseline/ \
         --val-split \
         --report results/v1_baseline/v1_baseline.md
     ```
   - Compute PSNR and SSIM metrics (in ACES-tonemapped space, matching training loss)
   - Generate side-by-side comparison PNGs (noisy | denoised | ground truth)
   - Expected: PSNR improvement of 3–6 dB over raw noisy input (modest but measurable)
   - Also run without `--val-split` to get full-dataset metrics for reference (note in report that full-dataset numbers include training data)

3. Export trained weights:
   - Export to `training/models/` (already gitignored in both `training/.gitignore` and root `.gitignore`):
     ```
     python scripts/export_weights.py \
         --checkpoint configs/checkpoints/model_best.pt \
         --output models/deni_v1.denimodel
     ```
   - Produces `training/models/deni_v1.denimodel` + `training/models/deni_v1.onnx`
   - Verify `.denimodel` file is reasonable size (~0.5–1 MB for ~120K params)
   - **Note:** These are intermediate training artifacts. The C++ denoiser weight loader (F11-1) will load `.denimodel` files from a configurable path at runtime. The exported weights are copied or referenced by path when integrating into deni — no need to move them to `denoise/` until F11-1.

4. Auto-generate quality assessment document:
   - Enhance `evaluate.py` with `--report <path>` flag that auto-generates a Markdown file containing:
     - Model configuration summary (channels, parameters, training epochs)
     - Per-image PSNR and SSIM table (Markdown table format)
     - Aggregate mean PSNR and SSIM
     - Comparison image references (relative paths to generated PNGs)
     - Noisy-input baseline PSNR (computed by comparing raw noisy input to ground truth)
     - Delta PSNR (denoised improvement over noisy baseline)
     - Timestamp and checkpoint path for reproducibility
   - Output: `training/results/v1_baseline/v1_baseline.md` (gitignored under `training/results/`)
   - TensorBoard loss curve screenshot added manually as a follow-up (referenced in the report as a placeholder path)
   - This document serves as the baseline for measuring future improvements

### Verification
- Training completes without error; loss curve shows convergence
- PSNR improvement over noisy input is measurable (> 2 dB minimum) on held-out validation split
- Visual comparison shows denoised output is visibly smoother than noisy input
- Exported weights file is valid and correctly sized
- No NaN/Inf in model output
- Auto-generated `v1_baseline.md` contains valid metrics and image references
- `--val-split` evaluation matches expected held-out pair count (2 of 15)

---

## Phase F9-5b: Hyperparameter Sensitivity Sweep (Optional) ⏭️ Skipped

**Goal:** Explore a small grid of hyperparameter variations to identify improvements over the F9-5 default configuration. Strictly optional — only proceed if F9-5 baseline is satisfactory and time permits.

**Prerequisite:** Phase F9-5 complete (baseline model trained and evaluated)

### Tasks

1. Create sweep configs:
   - Each config is a standalone YAML (no inheritance), based on `default.yaml` with one axis varied:
     - `configs/sweep_perceptual_0.05.yaml` — `lambda_perceptual: 0.05`
     - `configs/sweep_perceptual_0.2.yaml` — `lambda_perceptual: 0.2`
     - `configs/sweep_channels_32.yaml` — `base_channels: 32` (~480K params, ~4× larger)
     - `configs/sweep_lr_3e-4.yaml` — `learning_rate: 3.0e-4`
     - `configs/sweep_lr_5e-5.yaml` — `learning_rate: 5.0e-5`
   - All configs use different TensorBoard `run_name` suffixes for comparison (e.g., `runs/sweep_perceptual_0.05/`)

2. Run each sweep variant:
   - Train each config for 100 epochs (same as baseline)
   - Use separate checkpoint directories per config (automatic — `train.py` places checkpoints relative to config file, but sweep configs can override or use a `run_name` field)
   - Record wall-clock training time for the `base_channels=32` variant (expected ~2–4× slower)

3. Evaluate all variants:
   - Run `evaluate.py --val-split` for each variant's best checkpoint
   - Auto-generate a per-variant report in `results/sweep_<name>/`

4. Create comparison summary:
   - `training/results/sweep_summary.md` (auto-generated or manual):
     - Table: config variant | val PSNR | val SSIM | training time | parameter count
     - Identify best performer
     - If a variant clearly outperforms default, update `default.yaml` with winning hyperparameters

### Verification
- All sweep variants train without error
- Comparison table shows meaningful variation between configs
- If `default.yaml` is updated, re-run F9-5 evaluation to confirm improvement

---

## Phase F9-6: Extended Scenes + Data Augmentation ✅

**Goal:** Broaden the training set with additional scenes and viewpoints, and implement a data augmentation pipeline to increase effective dataset size. This phase is split into five sub-phases (F9-6a through F9-6e), each scoped to fit in a single Copilot session.

### Prerequisites

- Phase F9-5 complete (first training run baseline)
- `monti_datagen` builds and runs (Phase 11B)
- Phase 8L (KHR_texture_transform) and Phase 8M (KHR_materials_sheen) complete — required for correct rendering of ToyCar and SheenChair. Models that use these extensions can be included without 8L/8M but will render with incorrect textures (untiled) or missing sheen. If 8L/8M are not yet complete, defer ToyCar and SheenChair from the scene list.

---

### Phase F9-6a: Multi-Viewpoint Rendering in `monti_datagen` ✅

**Goal:** Enable `monti_datagen` to render multiple camera viewpoints from a single process invocation, avoiding the cost of re-parsing the scene and rebuilding Vulkan resources (~2-5 seconds per launch) for each viewpoint. Also add `--position`/`--target`/`--fov` CLI arguments for single-viewpoint use.

**Context:** The `GenerationSession` class already iterates `kNumFrames` in a loop where each iteration independently renders a noisy frame + reference accumulation + writes output. The loop body is fully independent — extending it to iterate over viewpoints primarily requires changing the camera between iterations and making `kNumFrames` configurable from a viewpoints file. `GenerationSession` currently has no reference to the `Scene` — one must be added to the constructor so it can call `scene.SetActiveCamera()` between viewpoints.

**Session scope:** C++ only — `main.cpp`, `GenerationSession.h/.cpp`, `Writer.h/.cpp`, and `CMakeLists.txt` modifications. No Python changes.

#### Tasks

1. **Add `--position`, `--target`, and `--fov` CLI arguments** (`app/datagen/main.cpp`):
   - Add `--position` option: `std::vector<float>` with `->expected(3)` for 3 floats `X Y Z` (camera world-space position)
   - Add `--target` option: `std::vector<float>` with `->expected(3)` for 3 floats `X Y Z` (camera look-at target point)
   - Add `--fov` option: single float, vertical FOV in degrees (default: `kDefaultFovDegrees` = 60.0)
   - Both `--position` and `--target` are optional. When omitted, use the existing `ComputeDefaultCamera()` auto-fit behavior.
   - When `--position`/`--target` are provided, construct a `CameraParams` directly from the specified values — do NOT call `ComputeDefaultCamera()`. Use `--fov` (or its default), `kDefaultNearPlane`/`kDefaultFarPlane` from `CameraSetup.h`, and up vector `(0, 1, 0)`.
   - Validation: if only one of `--position`/`--target` is provided, print error and exit
   - Convert `std::vector<float>` to `glm::vec3` after parsing
   - Print the active camera position/target/fov in the configuration output for reproducibility

2. **Add `--viewpoints` CLI argument** (`app/datagen/main.cpp`):
   - Add `--viewpoints` option: path to a JSON file containing an array of viewpoint entries, with `->check(CLI::ExistingFile)`
   - JSON format:
     ```json
     [
       {"position": [1.0, 2.0, 3.0], "target": [0.0, 0.0, 0.0]},
       {"position": [2.0, 1.0, -1.0], "target": [0.0, 0.5, 0.0]}
     ]
     ```
   - Optional `exposure` override per entry: `{"position": [...], "target": [...], "exposure": 0.5}`
   - Optional `fov` override per entry (degrees): `{"position": [...], "target": [...], "fov": 45.0}` — defaults to the global `--fov` value (or 60.0) when omitted
   - If `--viewpoints` is provided, `--position`/`--target` must NOT also be provided. Use CLI11 `->excludes()` on the `--viewpoints` option:
     ```cpp
     auto* pos_opt = app.add_option("--position", position_vec, ...)->expected(3);
     auto* tgt_opt = app.add_option("--target", target_vec, ...)->expected(3);
     auto* vp_opt  = app.add_option("--viewpoints", viewpoints_path, ...)
                         ->check(CLI::ExistingFile)
                         ->excludes(pos_opt)->excludes(tgt_opt);
     ```
   - Parse JSON using `nlohmann/json` (already fetched via `FetchContent` as `nlohmann_json`, `#include <nlohmann/json.hpp>`)
   - Store as `std::vector<ViewpointEntry>` (defined in `GenerationSession.h`, see Task 3):
     ```cpp
     struct ViewpointEntry {
         glm::vec3 position;
         glm::vec3 target;
         float fov_degrees = kDefaultFovDegrees;  // From CameraSetup.h
         std::optional<float> exposure;            // Override per-viewpoint, or use global --exposure
     };
     ```

3. **Extend `GenerationConfig` and `GenerationSession` with viewpoints and scene access**:
   - Define `ViewpointEntry` struct in `GenerationSession.h` (above `GenerationConfig`):
     ```cpp
     struct ViewpointEntry {
         glm::vec3 position;
         glm::vec3 target;
         float fov_degrees = kDefaultFovDegrees;
         std::optional<float> exposure;
     };
     ```
   - Add `std::vector<ViewpointEntry> viewpoints;` to `GenerationConfig`
   - Add a `Scene&` parameter to `GenerationSession`'s constructor and store it as `Scene& scene_` member. Update the call site in `main.cpp` accordingly.
   - If `--viewpoints` JSON provided → populate `viewpoints` from file
   - If `--position`/`--target` provided → single-entry vector (with `--fov` and `--exposure`)
   - If neither → single-entry from `ComputeDefaultCamera()` (current behavior)

4. **Add `nlohmann_json` link to `monti_datagen` CMake target** (`CMakeLists.txt`):
   - Add `nlohmann_json::nlohmann_json` to the `target_link_libraries` for `monti_datagen`:
     ```cmake
     target_link_libraries(monti_datagen PRIVATE
         ${CORE_LIBS}
         monti_capture
         CLI11::CLI11
         nlohmann_json::nlohmann_json
     )
     ```

5. **Add subdirectory parameter to `Writer::WriteFrame` and `Writer::WriteFrameRaw`** (`capture/include/monti/capture/Writer.h`, `capture/src/Writer.cpp`):
   - Add an optional `std::string_view subdirectory = ""` parameter to both `WriteFrame()` and `WriteFrameRaw()`
   - When non-empty, insert the subdirectory into the output path: `{output_dir}/{subdirectory}/frame_{NNNNNN}_input.exr`
   - Create the subdirectory (via `std::filesystem::create_directories`) if it doesn't already exist
   - When empty, behavior is unchanged: `{output_dir}/frame_{NNNNNN}_input.exr`

6. **Update `GenerationSession::Run()` to iterate viewpoints**:
   - Replace `constexpr uint32_t kNumFrames = 1;` with `config_.viewpoints.size()`
   - At the top of each loop iteration, build a `CameraParams` from the `ViewpointEntry` (position, target, `fov_degrees` → radians, `kDefaultNearPlane`/`kDefaultFarPlane`, up = `(0,1,0)`, exposure from per-viewpoint override or global config). Call `scene_.SetActiveCamera(camera)` to apply it.
   - Pass subdirectory `std::format("vp_{}", i)` to `WriteFrame`/`WriteFrameRaw` for per-viewpoint output organization: `output_dir/vp_0/frame_000000_input.exr`, `output_dir/vp_1/frame_000000_input.exr`, etc. Always use `vp_N/` subdirectories regardless of viewpoint count.
   - Print progress: `[viewpoint 3/10] pos=(1.0, 2.0, 3.0) target=(0.0, 0.0, 0.0) fov=60.0`
   - The renderer's `SetScene()` is already called once — camera changes between viewpoints only require updating the scene's active camera. The BLAS/TLAS, textures, and pipeline are reused.

7. **Handle frame_index for reference accumulation**:
   - With multiple viewpoints, reset the frame_index to 0 at the start of each viewpoint iteration. Each viewpoint gets a fresh jitter sequence starting from `frame_index = 0` for the noisy pass and `base_frame_index = 1` for reference accumulation.
   - This means every viewpoint uses the same jitter pattern, which is fine since different camera positions produce different pixel content.

#### Verification
- `monti_datagen scene.glb` — auto-fit viewpoint, output in `vp_0/`
- `monti_datagen scene.glb --position 1 2 3 --target 0 0 0` — renders from specified viewpoint in `vp_0/`
- `monti_datagen scene.glb --position 1 2 3 --target 0 0 0 --fov 45` — uses 45° vertical FOV
- `monti_datagen scene.glb --position 1 2 3` (no target) — prints error, exits non-zero
- `monti_datagen scene.glb --viewpoints vps.json` — renders all viewpoints, creates `vp_0/`, `vp_1/`, ... subdirectories
- `monti_datagen scene.glb --viewpoints vps.json --position 1 2 3 --target 0 0 0` — prints error (mutually exclusive), exits non-zero
- Viewpoint renders from JSON match equivalent `--position`/`--target` single-invocation renders (FLIP < 0.01 at 64 spp)
- Performance: 10-viewpoint JSON invocation completes in <50% of the time of 10 separate single-viewpoint invocations

---

### Phase F9-6b: Extended Scene Downloads + Viewpoint Generation ✅

**Goal:** Expand the training scene library and generate camera viewpoints for each scene.

**Session scope:** Python only — `download_scenes.py`, `generate_viewpoints.py`, `test_viewpoints.py`. No C++ or PyTorch changes.

**Resolved decisions:**
- SheenChair.glb added to download list (10 new GLB models + 2 multi-file glTF)
- Phases 8L and 8M are complete — ToyCar and SheenChair included without deferral
- FlightHelmet multi-file glTF download implemented (parse `.gltf` JSON, download referenced buffers/textures)
- Khronos Sponza multi-file glTF download also implemented alongside FlightHelmet
- Viewpoint parameters auto-computed from GLB/glTF bounding boxes (accessor min/max) — no hardcoded per-scene values
- Unit tests added in `tests/test_viewpoints.py` for orbit/hemisphere geometry, bounding box parsing, NaN/validity checks
- `generate_training_data.py` only gets new scene entries in `_SCENES` — viewpoint wiring deferred to F9-6d

#### Tasks

1. **Expand `download_scenes.py`** to download additional Khronos glTF-Sample-Assets GLB models:

   | Model | URL (Khronos glTF-Sample-Assets `main` branch) | Min Size | Features Exercised |
   |---|---|---|---|
   | WaterBottle.glb | `.../Models/WaterBottle/glTF-Binary/WaterBottle.glb` | 1 MB | PBR metal/roughness textures |
   | AntiqueCamera.glb | `.../Models/AntiqueCamera/glTF-Binary/AntiqueCamera.glb` | 1 MB | Detailed PBR, small geometry |
   | Lantern.glb | `.../Models/Lantern/glTF-Binary/Lantern.glb` | 1 MB | PBR wood/metal |
   | ToyCar.glb | `.../Models/ToyCar/glTF-Binary/ToyCar.glb` | 1 MB | Clearcoat, transmission, **sheen (8M), texture transform (8L)** |
   | ABeautifulGame.glb | `.../Models/ABeautifulGame/glTF-Binary/ABeautifulGame.glb` | 3 MB | Chess set, transmission, volume |
   | MosquitoInAmber.glb | `.../Models/MosquitoInAmber/glTF-Binary/MosquitoInAmber.glb` | 3 MB | Nested transmission, IOR, volume |
   | GlassHurricaneCandleHolder.glb | `.../Models/GlassHurricaneCandleHolder/glTF-Binary/GlassHurricaneCandleHolder.glb` | 1 MB | Glass transmission, volume |
   | BoomBox.glb | `.../Models/BoomBox/glTF-Binary/BoomBox.glb` | 5 MB | PBR, emissive front panel |
   | SheenChair.glb | `.../Models/SheenChair/glTF-Binary/SheenChair.glb` | 1 MB | **Sheen (8M), texture transform (8L)**, fabric |

   All URLs use the base `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/` prefix.

   Multi-file glTF scenes (no GLB variant available) — download `.gltf` + all referenced `.bin`/texture files:

   | Model | Base Path (Khronos glTF-Sample-Assets `main` branch) | Min Size | Features Exercised |
   |---|---|---|---|
   | FlightHelmet | `.../Models/FlightHelmet/glTF/` | 5 MB total | Multi-mesh PBR, leather/glass |
   | Sponza (Crytek) | `.../Models/Sponza/glTF/` | 30 MB total | Large interior, many materials, core PBR |

   - For each multi-file glTF: download the `.gltf` file, parse its JSON to discover referenced `buffers[].uri` and `images[].uri`, then download each into the same subdirectory under `scenes/`. The cgltf loader already supports multi-file `.gltf`.
   - All models use **CC0 or CC-BY 4.0** licenses — no approval required
   - Download to `training/scenes/` (gitignored), validate GLB magic bytes, skip existing files

   **Deferred scenes — not included in automated downloads:**

   - **Intel Sponza (PBR remaster):** 3.71 GB ZIP including glTF format. Creative Commons Attribution license. Manual download required from Intel:
     `https://cdrdv2.intel.com/v1/dl/getContent/830833`
     After download, extract the `glTF/` subdirectory to `training/scenes/IntelSponza/`. The `download_scenes.py` script should detect and validate this directory if present, and skip Sponza data generation if absent. Sponza uses core PBR features plus KHR_texture_transform (requires Phase 8L) and is fully compatible with the current renderer once 8L is complete.
   - **Amazon Lumberyard Bistro:** Distributed by NVIDIA ORCA in OBJ/FBX format only — no glTF version available. Would require format conversion. Deferred to Phase 10A-2 (Extended Scene Acquisition) in the implementation plan, which includes CMake-driven scene downloads.

2. **Create `training/scripts/generate_viewpoints.py`** — compute camera viewpoints for each scene:
   - `compute_bounding_box(scene_path)` — parses GLB/glTF accessor min/max for POSITION attributes, returns scene-wide AABB (center, extents)
   - `compute_orbit_viewpoints(center, radius, num_views, elevation_deg)` — generates evenly-spaced orbit positions around a center point at a given elevation angle
   - `compute_hemisphere_viewpoints(center, radius, num_views)` — generates quasi-uniform viewpoints on a hemisphere above the center (Fibonacci spiral distribution)
   - Per-scene viewpoint parameters auto-computed from bounding box: center = AABB center, radius = 2.5× half-diagonal, orbit_views = 8, elevations = [0, 20, -10] (default). Cornell box uses hardcoded parameters (programmatic scene, not parsed from GLB accessors).

   - Output: JSON file per scene (`viewpoints/<scene_name>.json`) matching the `--viewpoints` format from F9-6a:
     ```json
     [
       {"position": [1.0, 2.0, 3.0], "target": [0.0, 1.0, 0.0]},
       {"position": [2.0, 1.0, -1.0], "target": [0.0, 1.0, 0.0]}
     ]
     ```
   - CLI: `python scripts/generate_viewpoints.py [--output viewpoints/]`

3. **Update `generate_training_data.py`** — add new scene entries to `_SCENES` only:
   - Add all newly downloaded scenes to the `_SCENES` list
   - Viewpoint-based invocation (`--viewpoints` JSON) deferred to F9-6d
   - Existing exposure-sweep-only logic preserved for now

4. **Create `tests/test_viewpoints.py`** — unit tests for viewpoint generation:
   - Test `compute_orbit_viewpoints` produces correct count and geometry (positions lie on expected circle)
   - Test `compute_hemisphere_viewpoints` produces quasi-uniform distribution on hemisphere
   - Test `compute_bounding_box` returns correct AABB for a known GLB (cornell_box.glb)
   - Test no NaN in generated positions, target ≠ position for all viewpoints
   - Test JSON output format matches F9-6a schema
   - Test orbit at 0° elevation produces positions in the horizontal plane

#### Verification
- All downloaded scenes have valid GLB magic bytes (or valid `.gltf` JSON)
- Generated viewpoint JSONs contain valid camera positions (no NaN, target ≠ position)
- Unit tests pass for orbit, hemisphere, bounding box, and JSON format
- `generate_training_data.py` scene list updated with new entries

---

### Phase F9-6c ✅: Data Augmentation Transforms

**Goal:** Implement rotation and photometric augmentation transforms with correct per-channel adjustment rules. Screen-space flips are deferred because world-space normals don't transform correctly under them.

**Session scope:** Python/PyTorch only — `transforms.py` in `training/deni_train/data/`. Plus unit tests.

#### Design Note: No Flips (World-Space Normals)

Normals are stored in **world space** (transformed via `gl_ObjectToWorldEXT` in the shader, never converted to view/camera space). Flipping the screen image horizontally or vertically simulates rendering a mirrored scene — but the world-space normals in the G-buffer still point in the original direction, which is incorrect for the mirrored geometry. The same issue applies to 90°/270° rotations (they also imply a different camera orientation that would change the mapping between world axes and screen axes).

**Only 180° rotation is geometrically safe** — it is equivalent to rotating the camera 180° around its forward axis, which doesn't change which world axis maps to which screen axis (just negates both screen X and screen Y). Both motion vector components negate, and world-space normals remain unchanged.

Flips and arbitrary rotations can be revisited once normals are converted to view space in the capture writer (or as a dataset loader transform), which would make them correctly transformable under any screen-space spatial operation.

#### Tasks

1. **Remove `RandomHorizontalFlip` from the default transform pipeline** — the existing class remains in `transforms.py` but is no longer included in the `Compose` chain. The world-space normal issue makes it incorrect for training. Add a comment explaining why it is excluded.

2. **Implement `RandomRotation180(p=0.5)`** — random 180° rotation (the only safe rotation under world-space normals):
   - Apply `torch.rot90(tensor, k=2, dims=[-2, -1])` to all channels of both input and target (equivalent to flipping both axes)
   - Negate motion vector X and Y: `ch[11] = -ch[11]`, `ch[12] = -ch[12]`
   - World normals are **unchanged** — 180° screen rotation doesn't change world-space directions
   - The 3-channel target tensor (RGB) is spatially rotated but needs no channel adjustment

3. **Implement `ExposureJitter(range=(-1.0, 1.0))`**:
   - Sample `jitter ~ Uniform(range[0], range[1])`
   - Compute `scale = 2^jitter`
   - Multiply input radiance channels (diffuse RGB [0-2], specular RGB [3-5]) by `scale`
   - Multiply target radiance channels (RGB [0-2]) by the same `scale`
   - Leave guide channels [6-12] unchanged
   - **FP16 overflow protection:** After scaling, clamp using a luminance-preserving approach to avoid color shifts. Compute per-pixel luminance `L = 0.2126*R + 0.7152*G + 0.0722*B`. If `L > kFP16Max` (65504), scale the pixel's RGB uniformly by `kFP16Max / L`. This prevents overflow to Inf while preserving chromaticity. Apply the same clamp to both input radiance and target radiance after scaling.
   - The math should be performed in float32 (promote before scaling, convert back after clamping) to avoid intermediate overflow.

4. **Update default transform pipeline** in `train.py`:
   ```python
   transform = Compose([
       RandomCrop(cfg.data.crop_size),
       RandomRotation180(),
       ExposureJitter(range=(-1.0, 1.0)),
   ])
   ```

5. **Unit tests** for all transforms:
   - Construct synthetic 13-channel input tensors with known motion vector and world normal values
   - `RandomRotation180`: verify motion vectors are negated, normals are unchanged, spatial pixels are rotated
   - `ExposureJitter`: verify input radiance and target radiance are scaled by the same factor, guide channels are unchanged
   - `ExposureJitter` overflow: construct a tensor with values near FP16 max, apply positive jitter, verify no Inf values and chromaticity is preserved (R/G/B ratios unchanged)
   - `ExposureJitter` at jitter=0: verify input and target are unchanged

#### Per-Channel Transform Rules Reference

The 13-channel input tensor has channel indices:
```
[0–2]   diffuse RGB         (radiance — scaled by ExposureJitter)
[3–5]   specular RGB        (radiance — scaled by ExposureJitter)
[6–8]   world normals XYZ   (world-space — invariant to all current transforms)
[9]     roughness           (scalar — invariant)
[10]    linear depth Z      (scalar — invariant)
[11–12] motion vectors XY   (screen-space — negated by RandomRotation180)
```

#### Future: Enabling Full Spatial Augmentation

To enable flips and 90°/270° rotations, convert normals from world space to **view/camera space** in the capture writer (multiply by the camera's view matrix). View-space normals transform correctly under screen-space spatial operations (flip X ↔ negate view-normal X, etc.). This conversion should be done as a separate phase that also updates the inference shaders and any existing trained models.

#### Verification
- Unit tests pass for `RandomRotation180` and `ExposureJitter`
- Transforms produce correctly aligned input/target pairs (visual inspection on a few samples)
- Training runs without NaN/Inf with all augmentations enabled
- No Inf values produced by ExposureJitter on HDR inputs near FP16 max

---

### Phase F9-6d: Full Dataset Generation + Validation ✅

**Goal:** Update `generate_training_data.py` to pass per-scene viewpoint JSONs to `monti_datagen --viewpoints`, generate the complete extended training dataset (~1,680 frames), validate quality, and verify the training pipeline handles the larger dataset.

**Session scope:** Code changes to `generate_training_data.py` (viewpoint integration, HDRI support, disk space estimation), then orchestration — run scripts, validate output, adjust configs.

**Prerequisites:**
- Phase F9-6a ✅ (`monti_datagen --viewpoints` CLI)
- Phase F9-6b ✅ (`download_scenes.py`, `generate_viewpoints.py`)
- Phase F9-6c ✅ (`RandomRotation180`, `ExposureJitter` transforms)
- HDRI environment map downloaded (see Task 1)

#### Tasks

1. **Download an HDRI environment map** for standalone object scenes:
   - Download a free CC0 HDRI from [Poly Haven](https://polyhaven.com/hdris) — recommended: `studio_small_09_2k.exr` (indoor studio, ~8 MB, good for object rendering)
   - Save to `training/environments/studio_small_09_2k.exr` (gitignored)
   - **Rationale:** Monti does not support `KHR_lights_punctual` (by design — see monti_design_spec.md). The renderer supports only environment maps and emissive quad area lights. Standalone object models (WaterBottle, AntiqueCamera, Lantern, BoomBox, FlightHelmet, etc.) have no embedded lights and rely entirely on environment lighting. The default mid-gray fallback produces flat, low-variance images unsuitable for ML training. Cornell box (emissive ceiling quad) and Sponza (emissive surfaces) have self-contained lighting and don't require an HDRI.

2. **Update `generate_training_data.py`** — integrate viewpoint JSONs and HDRI support:
   - Add `--viewpoints-dir` CLI argument (default: `viewpoints/`): directory containing per-scene `<scene_name>.json` viewpoint files generated by `generate_viewpoints.py`
   - Add `--env` CLI argument (default: empty): optional path to an HDR environment map (`.exr`), forwarded to `monti_datagen --env`
   - Add `--max-viewpoints N` CLI argument (default: None = use all): truncate each scene's viewpoint list to the first N entries. This enables incremental pipeline validation without maintaining separate viewpoint files:
     - `--max-viewpoints 1` → 14 scenes × 1 vp × 5 exposures = 70 frames (~10.5 GB, ~35–70 min)
     - `--max-viewpoints 2` → 14 scenes × 2 vp × 5 exposures = 140 frames (~21 GB, ~70–140 min)
     - Omit → full 24 viewpoints per scene (1,680 frames, ~252 GB, ~14–28 hours)
   - For each scene, look up `<viewpoints_dir>/<scene_name>.json`. If the file exists, load the JSON array, truncate to `--max-viewpoints` if set, write a temp JSON with the truncated list, and pass `--viewpoints <temp_path>` to `monti_datagen`. If not found, fall back to single auto-fit viewpoint (current behavior, no `--viewpoints` flag).
   - When `--env` is provided, append `--env <path>` to every `monti_datagen` invocation
   - Update the total-pairs calculation and progress display to account for viewpoints per scene: `total_frames = sum(min(viewpoints_per_scene[s], max_vp) * len(exposures) for s in scenes)`
   - Add a `--dry-run` flag that prints the full generation plan (scenes, viewpoints, exposures, estimated disk space, estimated time) without running `monti_datagen`

3. **Add disk space estimation to `generate_training_data.py`**:
   - Estimate ~150 MB per EXR pair (input + target at 960×540, 29 FP16/FP32 channels total)
   - Before generation, compute `estimated_gb = total_frames * 0.15` and print it
   - Check available disk space on the output volume (via `shutil.disk_usage`). If estimated space exceeds 90% of available free space, print a warning and prompt for confirmation (skip prompt with `--yes` flag)

4. **Run `python scripts/download_scenes.py`** — download all scene models to `training/scenes/`

5. **Run `python scripts/generate_viewpoints.py`** — create viewpoint JSONs for all scenes in `training/viewpoints/`

6. **Pipeline validation run** — render 2 viewpoints per scene to verify pipeline and multi-viewpoint iteration:
   - Run with `--max-viewpoints 2` and `--dry-run` first to review the plan:
     ```
     python scripts/generate_training_data.py --env environments/studio_small_09_2k.exr \
         --viewpoints-dir viewpoints/ --scenes scenes/ --output training_data_test/ \
         --max-viewpoints 2 --dry-run
     ```
   - Then run without `--dry-run` to generate 140 frames (14 scenes × 2 viewpoints × 5 exposures, ~21 GB, ~70–140 min)
   - Using 2 viewpoints validates both single-frame rendering and multi-viewpoint iteration (first and second viewpoints are at different azimuths on the orbit, so they confirm distinct camera positions). By induction, if viewpoints 0 and 1 produce correct output, viewpoints 2–23 (same orbit/elevation logic) will too.
   - **Verify:** Input EXRs show noisy but recognizable images; target EXRs show clean converged renders; no black/NaN frames; standalone objects are lit by the HDRI; the two viewpoints per scene show visibly different camera angles
   - Run `python scripts/validate_dataset.py --data-dir training_data_test/` on the test output
   - Visually inspect the HTML gallery for each scene
   - If any scene produces obviously broken output, debug before committing to full generation

7. **Full dataset generation** — render all (scene × viewpoint × exposure) combinations:
   - Run: `python scripts/generate_training_data.py --env environments/studio_small_09_2k.exr --viewpoints-dir viewpoints/ --scenes scenes/ --output training_data/`
   - Expected: 14 scenes × 24 viewpoints × 5 exposures = **1,680 EXR pairs** (3,360 files)
   - Estimated disk space: **~252 GB** (1,680 × 150 MB)
   - Estimated render time: **~14–28 hours** on RTX 4090 (30–60 seconds per frame at 960×540, 4 SPP noisy + 256 SPP reference)

8. **Run `python scripts/validate_dataset.py`** — verify all EXR files:
   - No NaN/Inf values
   - Reasonable per-channel value ranges
   - All expected files present (1,680 input + 1,680 target)

9. **Spot-check rendered thumbnails** — visually inspect the HTML gallery generated by `validate_dataset.py` for each scene to verify:
   - Standalone objects are properly lit (not flat/gray)
   - Cornell box ceiling light illuminates the scene
   - Sponza interior has visible illumination from emissive surfaces
   - No obviously broken materials (untiled textures, missing normals)

10. **Run a short training test** (5 epochs) with the full dataset:
    - Verify the dataloader handles ~1,680 frames without out-of-memory errors
    - Verify training loss decreases over 5 epochs
    - Verify validation loss is computed without errors

11. **Update `training/configs/default.yaml`**:
    - Increase `epochs` from 100 to 200 (larger dataset benefits from more training)
    - Consider increasing `batch_size` from 8 to 12 or 16 if VRAM permits (monitor during test training)
    - Verify `data_dir` still points to the correct relative path (`../training_data`)

#### Verification
- All 1,680 training frames generated successfully (14 scenes × 24 viewpoints × 5 exposures)
- Validation script reports no NaN/Inf issues
- Visual inspection confirms all scenes are properly lit and rendered
- Pipeline validation (Task 6) passed before full generation
- Training dataloader iterates the full dataset without errors
- 5-epoch test training shows decreasing loss

### Extended Training Scenes

| Scene | Source | Viewpoints | Exposures | Total Frames | Features Exercised |
|---|---|---|---|---|---|
| Cornell box | `export_cornell_box.py` | 24 | 5 | 120 | Diffuse GI, area light |
| DamagedHelmet | Khronos (existing) | 24 | 5 | 120 | PBR textures, normal maps |
| DragonAttenuation | Khronos (existing) | 24 | 5 | 120 | Transmission, volume |
| WaterBottle | Khronos (new) | 24 | 5 | 120 | PBR metal/roughness |
| AntiqueCamera | Khronos (new) | 24 | 5 | 120 | Detailed PBR geometry |
| Lantern | Khronos (new) | 24 | 5 | 120 | PBR wood/metal |
| ToyCar | Khronos (new) | 24 | 5 | 120 | Clearcoat, transmission, sheen (8M), texture transform (8L) |
| ABeautifulGame | Khronos (new) | 24 | 5 | 120 | Transmission, volume, complex scene |
| MosquitoInAmber | Khronos (new) | 24 | 5 | 120 | Nested transmission, IOR |
| GlassHurricaneCandleHolder | Khronos (new) | 24 | 5 | 120 | Glass transmission |
| BoomBox | Khronos (new) | 24 | 5 | 120 | PBR, emissive |
| SheenChair | Khronos (new) | 24 | 5 | 120 | Sheen (8M), texture transform (8L), fabric |
| FlightHelmet | Khronos (new, glTF) | 24 | 5 | 120 | Multi-mesh PBR |
| Sponza | Khronos (Crytek, glTF) | 24 | 5 | 120 | Large interior, many materials, emissive surfaces |
| **Total** | | | | **1,680** | |

If Intel Sponza is manually downloaded, add 120 additional frames (24 viewpoints × 5 exposures).

With online augmentation (random crop + 180° rotation + exposure jitter), each epoch sees effectively unique samples. At 1,680 source frames with 256×256 crops from 960×540 images, each frame yields ~6 non-overlapping crop positions × 2 geometric states (identity + 180° rotation) × continuous exposure jitter — the effective training diversity is substantial without pre-generating offline augmentation files.

### Disk Space and Time Estimates

| Metric | Estimate |
|---|---|
| EXR pair size (input + target, 960×540) | ~150 MB |
| Total frames | 1,680 |
| Total disk space | **~252 GB** |
| Render time per frame (4 SPP + 256 SPP ref) | ~30–60 seconds |
| Total render time (RTX 4090) | **~14–28 hours** |

The generation only needs to be done once. The `--dry-run` flag prints the full plan and disk estimate before committing. The script checks available disk space and warns if insufficient.

### Environment Lighting Strategy

Monti does not support `KHR_lights_punctual` (by design — zero-area emitters are non-physical). The renderer supports environment maps and emissive quad area lights only.

| Scene Category | Lighting Source | HDRI Needed? |
|---|---|---|
| Cornell box | Emissive ceiling quad (self-contained) | No |
| Sponza | Emissive surfaces + HDRI skylight | Optional (improves quality) |
| Standalone objects (11 scenes) | Environment map only | **Yes** |

All standalone object scenes require `--env <hdri.exr>` for meaningful illumination. The default mid-gray fallback produces flat, low-variance images that would harm training quality. The `generate_training_data.py` script unconditionally passes `--env` to all scenes when the flag is provided — scenes with embedded lighting (Cornell box) benefit from the additional ambient contribution.

Phase F9-6e expands this to 5 diverse HDRIs with round-robin cycling across viewpoints via `--envs-dir`. If F9-6e is complete before F9-6d Task 7 (full generation), use `--envs-dir environments/` instead of `--env environments/studio_small_09_2k.exr` for maximum lighting diversity in the base dataset.

### Future Augmentation Enhancements

The initial augmentation pipeline (180° rotation + exposure jitter) is sufficient for MVP quality. The following techniques, used by production denoisers like DLSS, should be evaluated once the baseline model is trained:

- **Noise-level jitter** — Vary `--spp` across training frames (1, 2, 4, 8, 16) so the network learns to handle different input noise levels. The current plan uses a fixed SPP for all training data, which may cause the model to underperform at SPP values it wasn't trained on.
- **Auxiliary channel dropout** — During training, randomly zero out entire guide channels (normals, albedo, depth, motion vectors) with probability ~5–10% per channel. This prevents the network from over-relying on any single guide and improves robustness when guides are noisy or unreliable.
- **Random SPP mixing** — Render the same viewpoint at multiple SPP levels and randomly select one as input during training. More expensive than noise-level jitter (requires multiple renders per viewpoint) but produces better generalization across input quality levels.

---

### Phase F9-6e: HDRI Collection + Programmatic Lighting Rigs

**Goal:** Download diverse CC0 HDRIs from Poly Haven and add programmatic area light support to `monti_datagen`, then generate supplementary training frames with varied lighting conditions. Different lighting produces fundamentally different noise distributions — HDRI diversity and analytical area lights are more impactful augmentation than geometric transforms alone.

**Prerequisites:**
- Phase F9-6a ✅ (`monti_datagen` CLI patterns, `nlohmann_json` linked)
- Phase F9-6b ✅ (scene downloads, `generate_viewpoints.py` bounding box computation)

**Session scope:** C++ changes to `monti_datagen` (`--lights` flag), Python scripts (`download_hdris.py`, `generate_light_rigs.py`), updates to `generate_training_data.py` (`--envs-dir`, `--lights-dir`), supplementary dataset generation and validation.

**Ordering note:** This phase can run before or after F9-6d's pipeline validation (Tasks 1–6). If completed before F9-6d Task 7 (full generation), the base dataset benefits from HDRI cycling via `--envs-dir` instead of single `--env`. If completed after, the supplementary light-rig frames augment the existing single-HDRI dataset.

#### Tasks

1. **Create `scripts/download_hdris.py`** — download 5 diverse CC0 HDRIs from Poly Haven:
   - Download URL pattern: `https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/2k/<name>_2k.exr`
   - Save to `training/environments/` (gitignored)
   - HDRIs (2K resolution, ~5–10 MB each):

   | HDRI | Character | Training Purpose |
   |---|---|---|
   | `studio_small_09` | Indoor studio, neutral | Even diffuse lighting baseline |
   | `kloppenheim_06` | Overcast outdoor, cool | Soft shadows, low contrast |
   | `sunset_fairway` | Warm directional sunset | Strong directional shadows, warm tones |
   | `moonlit_golf` | Dim nighttime | Low-light noise patterns, high dynamic range |
   | `royal_esplanade` | Bright cloudy sky | High-key exterior, specular highlights |

   - Validate each file has valid OpenEXR magic bytes (`0x76 0x2f 0x31 0x01`)
   - Skip files that already exist (idempotent re-runs)
   - **Selection rationale:** Spans indoor/outdoor, warm/cool, bright/dark, and directional/diffuse lighting — the axes that most affect denoising difficulty

2. **Add `--lights <json_file>` CLI flag to `monti_datagen`** (C++ — `app/datagen/main.cpp`):
   - New optional CLI option accepting a JSON file path
   - JSON format: array of area light objects matching the `AreaLight` struct:
     ```json
     [
       {
         "corner": [x, y, z],
         "edge_a": [ax, ay, az],
         "edge_b": [bx, by, bz],
         "radiance": [r, g, b],
         "two_sided": false
       }
     ]
     ```
   - After scene loading and environment light setup, parse the JSON and call `scene.AddAreaLight()` for each entry
   - Reuse existing `nlohmann_json` dependency (already linked for `--viewpoints`)
   - Validation: reject empty arrays, require 3-component vectors, non-negative radiance
   - The JSON contains absolute world-space coordinates — Python scripts handle bounding-box-relative positioning when generating the rig files

3. **Create `scripts/generate_light_rigs.py`** — generate per-scene light rig JSON files:
   - CLI: `python scripts/generate_light_rigs.py --scenes-dir scenes/ --output light_rigs/ [--seed 42]`
   - For each scene, load the model with `trimesh` (reusing the approach from `generate_viewpoints.py`) and compute bounding box center, extent, and radius
   - Generate two rig types per scene:

     **Overhead rig** (`light_rigs/<scene_name>/overhead.json`):
     - Single quad area light centered above the bounding box
     - Height: `bbox_max_y + 1.5 × bbox_extent_y`
     - Width and depth: `uniform(0.5, 2.0) × bbox_horizontal_extent`
     - Normal faces downward (`cross(edge_a, edge_b)` points toward −Y)
     - Radiance: warm white `[5.0, 4.8, 4.5]` × `uniform(0.5, 2.0)` random scale
     - `two_sided: false`

     **Key-fill-rim rig** (`light_rigs/<scene_name>/key_fill_rim.json`):
     - Three area lights at `2.0 × bbox_radius` from bounding box center:
       - **Key** (45° azimuth, 30° elevation): radiance `[8.0, 7.5, 7.0]` ± 15%
       - **Fill** (−45° azimuth, 15° elevation): ~0.3× key, cooler `[2.0, 2.2, 2.5]`
       - **Rim** (180° azimuth, 45° elevation): ~0.5× key, warm `[4.0, 3.8, 3.5]`
     - Each light quad sized at `0.5 × bbox_extent`
     - Position perturbation: ±10° azimuth, ±5° elevation, ±10% radius from nominal
     - All lights oriented to face toward bounding box center
     - `two_sided: false`

   - Fixed random seed (default: 42) for reproducibility across re-runs
   - Output: `light_rigs/<scene_name>/overhead.json` and `light_rigs/<scene_name>/key_fill_rim.json`

4. **Update `generate_training_data.py`** — add HDRI cycling and light rig support:
   - **`--envs-dir <dir>`**: directory containing multiple `.exr` HDRIs (mutually exclusive with `--env`). Discovers all `*.exr` files sorted alphabetically. For each `(scene, viewpoint_index)`, selects HDRI by round-robin: `hdri = hdris[viewpoint_index % len(hdris)]`. With 24 viewpoints and 5 HDRIs, each scene gets ~5 viewpoints per HDRI across all lighting conditions.
   - **`--lights-dir <dir>`**: directory containing per-scene subdirectories with light rig JSONs (`<dir>/<scene_name>/*.json`). When present, generates **additional** frames beyond the base pass — one full viewpoints × exposures sweep per rig, passing `--lights <rig.json>` to `monti_datagen` alongside the HDRI.
   - **`--lights-max-viewpoints N`** (default: 4): independent viewpoint limit for light-rig passes only (the base pass uses `--max-viewpoints`). Keeps supplementary frame count manageable without affecting base dataset coverage.
   - Light-rig frames stored in separate subdirectories: `<output>/<scene>/<rig_name>/ev_<ev>/` — no conflict with base frames in `<output>/<scene>/ev_<ev>/`
   - Update total frame count calculation, progress display, and `--dry-run` output to include light-rig passes
   - `--envs-dir` and `--env` are mutually exclusive (print error and exit if both provided)

5. **Unit tests:**
   - `test_download_hdris.py`: URL construction, EXR magic byte validation, idempotent skip-if-exists
   - `test_generate_light_rigs.py`: output JSON matches `--lights` schema, bounding-box-relative positioning, light normals face scene center, positive radiance
   - `test_generate_training_data.py` (additions): `--envs-dir` round-robin cycling, `--lights-dir` discovery and frame count, `--envs-dir`/`--env` mutual exclusivity, `--lights-max-viewpoints` default

6. **Pipeline validation — render sample light-rig frames:**
   - Generate light rigs: `python scripts/generate_light_rigs.py --scenes-dir scenes/ --output light_rigs/`
   - Dry-run review:
     ```
     python scripts/generate_training_data.py --envs-dir environments/ \
         --viewpoints-dir viewpoints/ --scenes scenes/ --output training_data_lights_test/ \
         --lights-dir light_rigs/ --max-viewpoints 1 --lights-max-viewpoints 1 --dry-run
     ```
   - Expected: 14 scenes × (1 base + 2 rigs) × 1 viewpoint × 5 exposures = **210 frames** (~31.5 GB, ~1.75–3.5 hours)
   - Render and visually inspect:
     - Overhead frames: top-down area light with soft shadows visible
     - Key-fill-rim frames: three-point lighting with distinct key/fill/rim contributions
     - Base pass frames: varied HDRI environments visible across scenes (not uniform)
   - Run `validate_dataset.py` on the test output

7. **Generate supplementary dataset** (production):
   - After F9-6d's base dataset exists in `training_data/`, generate light-rig supplementary frames into the same directory:
     ```
     python scripts/generate_training_data.py --envs-dir environments/ \
         --viewpoints-dir viewpoints/ --scenes scenes/ --output training_data/ \
         --lights-dir light_rigs/ --lights-max-viewpoints 4
     ```
   - The base pass re-renders 1,680 frames with HDRI cycling (if previously generated with single `--env`, this replaces those frames with per-viewpoint HDRI rotation — same count, more diverse lighting)
   - Light-rig supplementary: 14 scenes × 2 rigs × 4 viewpoints × 5 exposures = **560 frames** (~84 GB, ~4.7–9.3 hours)
   - Total dataset: **2,240 frames** (1,680 base + 560 light rigs) ≈ **336 GB**
   - Run `validate_dataset.py` on the full combined dataset
   - Spot-check HTML gallery: overhead rig frames show visible top-down illumination; key-fill-rim frames show three-point lighting pattern

#### Dataset Summary (with F9-6e)

| Pass | Scenes | Viewpoints | HDRIs | Exposures | Light Rig | Frames |
|---|---|---|---|---|---|---|
| Base (HDRI cycling) | 14 | 24 | 5 (round-robin) | 5 | — | 1,680 |
| Overhead rig | 14 | 4 | 5 (round-robin) | 5 | overhead | 280 |
| Key-fill-rim rig | 14 | 4 | 5 (round-robin) | 5 | key_fill_rim | 280 |
| **Total** | | | | | | **2,240** |

Disk estimate: ~336 GB (2,240 × 150 MB). Render time: ~18.7–37.3 hours on RTX 4090 (total, not incremental over F9-6d).

The supplementary frame count is tunable via `--lights-max-viewpoints`. Using 8 instead of 4 doubles rig frames to 1,120 (2,800 total, ~420 GB). Start with 4 and increase if the trained model shows lighting-dependent quality variance.

#### Verification
- All 5 HDRIs download successfully and have valid EXR magic bytes
- `monti_datagen --lights` parses JSON and renders scenes with visible area light contributions
- Light rig JSONs produce physically plausible illumination (shadows and highlights in expected directions)
- Pipeline validation (210 frames) renders without errors; visual inspection confirms overhead and key-fill-rim lighting patterns
- `validate_dataset.py` passes on the combined 2,240-frame dataset (no NaN/Inf, all expected files present)
- Unit tests pass for `download_hdris`, `generate_light_rigs`, and updated `generate_training_data`

---

## Phase F9-7: Production Training Run

**Goal:** Train the U-Net on a full production dataset generated from the updated pipeline (viewpoint-centric data amplification, flat `<scene>_<id>_{input,target}.exr` naming). Establish a production quality baseline with per-scene metrics. Export production weights for F11 deployment.

**Prerequisites:**
- Phase F9-6e complete ✅: HDRIs, light rigs, environment cycling, viewpoint generation with embedded exposure/env/lights amplification
- Prune dark viewpoints plan complete ✅: unique IDs per viewpoint, flat file naming, `remove_invalid_viewpoints.py` operational
- Pipeline validated with 8 viewpoints per scene in `training_data_test/`

**Session scope:** Code changes to `train.py` (stratified validation split, early stopping), `evaluate.py` (per-scene metrics), and `default.yaml` (production config). Then: smoke-test training on the existing 8-viewpoint test dataset, generate full production viewpoints and training data, and run production training.

### Dataset Architecture

All data amplification (exposure levels, environment maps, light rigs) is embedded in the viewpoint JSON entries. Each viewpoint has a unique 8-hex ID and maps 1:1 to an EXR pair. The generation script renders viewpoints directly with no additional loops.

**File naming:** `<SceneName>_<8-hex-id>_{input,target}.exr` (flat, no subdirectories)

**Scene name extraction:** From filename — strip `_<8-hex-id>_{input,target}.exr` suffix. The scene name is everything before the last `_<8-hex>_` segment.

**Viewpoint generation settings for production (recommended):**
```
python scripts\generate_viewpoints.py `
    --scenes scenes `
    --output viewpoints `
    --seeds viewpoints\manual `
    --variations-per-seed 4 `
    --envs-dir environments `
    --lights-dir light_rigs `
    --exposures 0 -1 1 -2 2
```

With 67 seed viewpoints across 14 scenes, `--variations-per-seed 4`, and 5 exposure levels, this produces ~1,340 total viewpoints. After ~10% attrition from `remove_invalid_viewpoints.py`, expect ~1,200 usable training pairs.

### Tasks

#### Infrastructure changes

1. **Add per-scene stratified validation split** to `train.py` and `evaluate.py`:
   - The current validation split takes the last 10% of all pairs globally. With pairs sorted by filename (grouped by scene name), the last 10% would be only WaterBottle + ToyCar frames, leaving most scenes unrepresented in validation.
   - Change to: hold out the **last 10% of pairs for each scene**. Group pairs by scene name (extracted from filename), sort within each group, and take the last 10% (minimum 1) of each scene's pairs as validation.
   - Implementation: `ExrDataset` discovers pairs via glob `*_input.exr`. The scene name is extractable from the filename by stripping `_<8-hex-id>_input.exr`. Group pairs by scene name, take the last 10% of each scene's sorted pairs as validation, and the rest as training.
   - Add a shared `deni_train/data/splits.py` module with `scene_name_from_pair(pair)` and `stratified_split(pairs)` → `(train_indices, val_indices)` that both `train.py` and `evaluate.py` use.
   - `evaluate.py --val-split` should use the same stratified logic so validation metrics match what training held out.

2. **Add early stopping with `patience`** to `train.py`:
   - Add `patience` field to `default.yaml` under `training:` (default: `30` epochs)
   - Track epochs since last validation loss improvement. If `patience` epochs pass without improvement, stop training early and print a message.
   - The best model checkpoint (`model_best.pt`) is already saved on each improvement, so early stopping just terminates the loop — no additional checkpointing logic needed.
   - Early stopping is based on validation loss (already computed each epoch). The metric is automated — no manual monitoring required.

3. **Add per-scene metric grouping** to `evaluate.py` report:
   - Extract scene name from each evaluation pair's filename using the shared `scene_name_from_pair()` helper
   - Group results by scene name and compute per-scene mean PSNR, SSIM, delta PSNR
   - Add a "Per-Scene Summary" section to the Markdown report:
     ```
     ## Per-Scene Summary
     | Scene | Frames | Mean Noisy PSNR | Mean Denoised PSNR | Mean Delta PSNR | Mean SSIM |
     ```
   - Print per-scene summary to console during evaluation
   - This enables the verification criterion "no quality regression on any scene type"

4. **Update `default.yaml` for production training:**
   - `epochs: 200` (early stopping will terminate sooner if the model converges)
   - `patience: 30` (stop if no val loss improvement for 30 epochs)
   - `batch_size: 8` (monitor VRAM usage during smoke test and increase to 12 or 16 if headroom permits)
   - `base_channels: 16` (no sweep data to justify 32; start small and evaluate)
   - `warmup_epochs: 5`, `learning_rate: 1.0e-4` (keep unchanged)
   - `data_dir: "../training_data"` (production dataset)

#### Smoke-test training (8-viewpoint test dataset)

5. **Run a smoke-test training** on the existing 8-viewpoint-per-scene test dataset in `training_data_test/`:
   - Create `configs/smoke_test.yaml` with: `data_dir: "../training_data_test"`, `epochs: 10`, `patience: 5`, `checkpoint_interval: 5`
   - Run: `python -m deni_train.train --config configs/smoke_test.yaml`
   - **Verify:**
     - Stratified validation split reports frames from multiple scenes (not just the last scene alphabetically)
     - Training loss decreases over 10 epochs
     - Validation loss is computed without errors
     - Checkpoint saves work (both periodic and best-model)
     - Early stopping counter is printed each epoch
   - Run evaluation: `python -m deni_train.evaluate --checkpoint configs/checkpoints/smoke_test/model_best.pt --data_dir ../training_data_test --output_dir results/smoke_test/ --val-split --report results/smoke_test/smoke_test.md`
   - **Verify:**
     - Per-scene summary table appears in the report with all scenes represented
     - Per-image metrics table is present
     - Comparison PNGs are generated

#### Production dataset generation

6. **Generate production viewpoints:**
   ```
   python scripts\generate_viewpoints.py `
       --scenes scenes `
       --output viewpoints `
       --seeds viewpoints\manual `
       --variations-per-seed 4 `
       --envs-dir environments `
       --lights-dir light_rigs `
       --exposures 0 -1 1 -2 2
   ```

7. **Render production training data:**
   ```
   python scripts\generate_training_data.py `
       --monti-datagen ..\build\Release\monti_datagen.exe `
       --scenes scenes `
       --viewpoints-dir viewpoints `
       --output training_data `
       --width 960 --height 540 `
       --spp 4 --ref-frames 64 `
       --jobs 3 -y
   ```

8. **Prune invalid viewpoints:**
   ```
   python scripts\remove_invalid_viewpoints.py `
       --training-data training_data `
       --viewpoints-dir viewpoints
   ```

9. **Validate dataset:**
   ```
   python scripts\validate_dataset.py `
       --data_dir training_data `
       --gallery training_data\gallery.html
   ```

#### Production training

10. **Run full production training:**
    - Run: `python -m deni_train.train --config configs/default.yaml`
    - Local RTX 4090
    - Monitor via TensorBoard: `tensorboard --logdir configs/runs/`
    - Watch for: training loss convergence, validation loss tracking training loss (no divergence = no overfitting), early stopping trigger
    - If batch_size=8 causes OOM, reduce to 4 and restart

11. **Evaluate production model:**
    - Run: `python -m deni_train.evaluate --checkpoint configs/checkpoints/model_best.pt --data_dir ../training_data --output_dir results/v2_production/ --val-split --report results/v2_production/v2_production.md`
    - Review the per-scene summary: check that all 14 scenes have positive delta PSNR (denoised better than noisy)
    - Manually inspect comparison PNGs for representative scenes:
      - A diffuse-dominated scene (cornell_box)
      - A specular/transmission scene (GlassHurricaneCandleHolder or DragonAttenuation)
      - A complex multi-material scene (Sponza or ABeautifulGame)
    - Note any scenes where the model struggles (expected: highly specular or transmissive content)

12. **Export production weights:**
    - `python scripts/export_weights.py --checkpoint configs/checkpoints/model_best.pt --output models/deni_v1.denimodel`
    - Verify file size is reasonable (~0.5–1 MB for ~120K params)
    - Verify export script prints layer summary without errors

13. **Document results:**
    - The auto-generated `results/v2_production/v2_production.md` serves as the production quality report
    - Note in the report: training epochs completed (did early stopping trigger?), final train/val loss, per-scene analysis
    - Note areas where the model struggles and potential improvements for future phases

### Verification
- ✅ Stratified validation split holds out the last ~10% of each scene's pairs (all scenes represented in validation)
- ✅ Early stopping terminates training if validation loss plateaus for `patience` epochs
- ✅ Per-scene evaluation report shows metrics for all 14 scenes
- ✅ Smoke test on 8-viewpoint test data completes without errors (training + evaluation + report generation)
- ✅ Production model achieves positive delta PSNR (improvement over noisy input) on held-out validation data (+1.37 dB mean)
- ⚠️ 9 of 14 scenes show positive delta PSNR; 5 scenes regress (BoomBox -5.53, GlassHurricaneCandleHolder -8.45, SheenChair -3.29, ToyCar -0.79, WaterBottle -0.54) — regressions are on specular/transmissive content as anticipated
- ✅ Visual quality is noticeably better than passthrough (noisy) input on diffuse-dominated scenes
- ✅ Exported weights are valid and correctly sized (475,779 params, 1.9 MB .denimodel, 1.9 MB .onnx)

#### Production Training Results
- **Model:** DeniUNet, base_channels=32, 475,779 parameters
- **Dataset:** 1,270 safetensors samples (1,144 train / 126 val), 14 scenes
- **Training:** 148 epochs (early stopped at patience=30), best val_loss=0.086232 at epoch 118
- **Evaluation:** Mean +1.37 dB delta PSNR, mean SSIM 0.7303 on validation split
- **Top performers:** cornell_box (+11.58 dB), DragonAttenuation (+7.85 dB), MosquitoInAmber (+4.69 dB)
- **Regressions:** GlassHurricaneCandleHolder (-8.45 dB), BoomBox (-5.53 dB), SheenChair (-3.29 dB) — specular/transmissive content
- **Bug fixed:** evaluate.py and export_weights.py now infer model hyperparameters from checkpoint (base_channels was hardcoded to 16 default, mismatching the 32 used in training)

---

## Phase F11-1: Weight Loading + Inference Buffers in Deni

**Goal:** Add the `.denimodel` weight loader and GPU buffer allocation for ML inference to the Deni denoiser library. No inference yet — just the data loading and memory management.

### Tasks

1. Create `denoise/src/vulkan/WeightLoader.h` and `WeightLoader.cpp`:
   - `WeightLoader` class: parses `.denimodel` binary format
   - `Load(std::string_view path)` → returns `WeightData` struct containing:
     - Vector of `LayerWeights` (name, shape, float data)
     - Total parameter count
   - Validates magic number, version, and checksums
   - Error handling via `std::optional` (returns `std::nullopt` on invalid file)

2. Create `denoise/src/vulkan/MlInference.h` and `MlInference.cpp` (buffer management only):
   - `MlInference` class (internal to deni):
     - Constructor: `MlInference(VkDevice, VmaAllocator, PFN_vkGetDeviceProcAddr, uint32_t width, uint32_t height)`
     - `LoadWeights(const WeightData&, VkCommandBuffer cmd)` — uploads weight data to GPU storage buffers via staging
     - `Resize(uint32_t width, uint32_t height)` — (re)allocates intermediate feature map images
   - Weight storage: one `VkBuffer` per conv layer (kernel weights + bias concatenated)
   - Intermediate feature map allocation:
     - VkImages at each resolution level (full, half, quarter)
     - Channel counts matching the U-Net architecture (16, 32, 64)
     - Format: `VK_FORMAT_R16G16B16A16_SFLOAT` — packs 4 channels per image; multiple images per level for higher channel counts (16ch = 4 images, 32ch = 8 images, 64ch = 16 images)
     - Skip connection buffers (level 0 and level 1 encoder outputs)
   - Peak intermediate memory estimate at 1080p:
     - Level 0 (1920×1080): 16ch × 2 bytes × 1920 × 1080 = 66 MB
     - Level 1 (960×540): 32ch × 2 bytes × 960 × 540 = 33 MB
     - Level 2 (480×270): 64ch × 2 bytes × 480 × 270 = 17 MB
     - Skip buffers + processing temps: ~150 MB total
   - All allocations through VMA; freed in destructor (RAII)

3. Add `MlInference` as a private member of `Denoiser`:
   - Created conditionally — only when a `.denimodel` file path is provided
   - Add `model_path` field to `DenoiserDesc` (optional `std::string`)
   - When `model_path` is set and file is valid: create `MlInference`, load weights
   - When `model_path` is empty or invalid: fall back to passthrough

4. Write integration test (`tests/ml_weight_loader_test.cpp`):
   - Generate a small test `.denimodel` file (use Python export script or create a C++ test helper)
   - Verify `WeightLoader::Load()` parses it correctly
   - Verify layer names, shapes, and weight values round-trip
   - Verify `MlInference` buffer allocation succeeds at a test resolution (256×256)
   - Verify allocation sizes are non-zero and reasonable

### Verification
- `WeightLoader` parses valid `.denimodel` files and rejects invalid ones
- GPU buffers for weights are allocated and uploaded
- Intermediate feature map images are allocated at correct dimensions
- `Denoiser::Create()` with `model_path` set creates `MlInference`; without it, falls back to passthrough
- Integration test passes with zero Vulkan validation errors
- RAII cleanup: no leaks on destruction

### Intermediate Buffer Strategy

The U-Net dispatcher needs ping-pong buffers at each resolution. To minimize allocation count, use a small pool of VkImages per resolution level:

```
Level 0 (H×W):     buf_a[4], buf_b[4]    — 16 channels = 4 RGBA16F images
Level 1 (H/2×W/2): buf_c[8], buf_d[8]    — 32 channels = 8 RGBA16F images
Level 2 (H/4×W/4): buf_e[16], buf_f[16]  — 64 channels = 16 RGBA16F images
Skip 0:             skip_0[4]              — 16 channels saved from encoder
Skip 1:             skip_1[8]              — 32 channels saved from encoder
```

An alternative approach uses `VK_FORMAT_R16_SFLOAT` images with one channel per image (simpler indexing, more images). The RGBA16F approach packs 4 channels per image, reducing descriptor count by 4× at the cost of slightly more complex indexing in shaders. Either approach is valid; RGBA16F is preferred for bandwidth efficiency.

---

## Phase F11-2: GLSL Inference Compute Shaders ✅

**Goal:** Implement the GLSL compute shaders for U-Net inference and the dispatch sequence in `MlInference`. Validate output correctness against PyTorch reference.

### Design Decisions

1. **Feature map storage: flat `VkBuffer` (not storage images).** Intermediate feature maps are stored as flat FP16 storage buffers with layout `[C][H][W]` (channel-major). This simplifies indexing in shaders (linear addressing by channel/pixel) compared to managing arrays of RGBA16F images with 4-channel packing. Buffers use `VK_FORMAT_UNDEFINED`; the shader reads/writes via `float16_t` SSBO access.
   > **Future enhancement:** Switch to `VK_FORMAT_R16G16B16A16_SFLOAT` storage images for 2D texture cache benefits once the pipeline is validated. Profile to determine if the cache hit rate justifies the indexing complexity.

2. **GroupNorm: separate global reduction pass.** GroupNorm requires mean/variance over the full spatial extent (H×W) per group, which cannot be computed within a single 16×16 workgroup tile. The convolution and normalization are split into two dispatches: `conv.comp` (conv + bias) followed by `group_norm.comp` (global reduction → normalize + scale/shift + activation).
   > **Future enhancement:** If the extra dispatch becomes a bottleneck, consider: (a) fusing norm into conv via atomic global memory accumulation with a completion counter, (b) replacing GroupNorm with a per-pixel normalization that doesn't require spatial reduction, or (c) using subgroup operations for partial reductions to reduce shared memory pressure.

3. **Specialization constants for channel counts.** `in_channels` and `out_channels` are specialization constants, allowing the GLSL compiler to unroll channel loops and optimize register allocation. `width` and `height` remain push constants (change on resize). `num_groups` is a specialization constant (fixed per pipeline). `activation` (0=none, 1=LeakyReLU) is a specialization constant.

4. **Pre-allocated descriptor sets (one per dispatch step).** A fixed pool of descriptor sets is allocated at `Resize()` / pipeline creation time — one per dispatch step in the U-Net (~20 dispatches). Each is written once with the appropriate input/output/weight buffer bindings. This is the simplest approach.
   > **Future enhancement:** Consider `VK_KHR_push_descriptor` to eliminate descriptor set allocation, or `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC` with dynamic offsets to reduce set count.

5. **First encoder conv reads G-buffer directly.** The first encoder conv (13→16) reads from the `DenoiserInput` image views directly, with the G-buffer channel mapping hardcoded in the shader (diffuse RGB from binding 0, specular RGB from binding 1, normals XYZW from binding 2, depth Z from binding 3, motion XY from binding 4). This avoids an extra input-assembly dispatch.
   > **Future enhancement:** If the G-buffer layout changes or temporal inputs are added (F11-4), introduce an `input_assemble.comp` shader to decouple the first conv from the specific G-buffer format.

6. **Shader loading: file-based `.spv`.** Shaders are compiled to SPIR-V by CMake (`glslc`) and loaded from `.spv` files at runtime, consistent with the existing passthrough shader approach.

7. **Weight-to-layer mapping: name→index map.** `MlInference::LoadWeights()` builds an `std::unordered_map<std::string, uint32_t>` mapping PyTorch `state_dict` key names to weight buffer indices. The dispatch sequence looks up weight buffers by name (e.g., `"down0.conv1.conv.weight"`).

### Tasks

1. **Update `MlInference` feature map storage from images to buffers:**
   - Replace `FeatureImage` / `FeatureLevel` (VkImage-based) with `FeatureBuffer` structs holding a single `VkBuffer` + `VmaAllocation` per level
   - Buffer size per level: `channels × height × width × sizeof(float16_t)` bytes
   - Allocate ping/pong pairs at each level, plus skip connection buffers:
     - Level 0 (H×W): 2 × 16ch buffers (ping/pong) + 1 × 16ch (skip_0)
     - Level 1 (H/2×W/2): 2 × 32ch buffers + 1 × 32ch (skip_1)
     - Level 2 (H/4×W/4): 2 × 64ch buffers (no skip needed)
     - Upsample-concat scratch: 1 × 96ch at H/2×W/2, 1 × 48ch at H×W
   - All buffers: `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`, `VMA_MEMORY_USAGE_GPU_ONLY`
   - Add `weight_index_` member: `std::unordered_map<std::string, uint32_t>`

2. **Create `denoise/src/vulkan/shaders/conv.comp`** (convolution + bias only, no normalization):
   - Workgroup: 16×16 threads, one output pixel per thread
   - Specialization constants: `IN_CHANNELS`, `OUT_CHANNELS`
   - Push constants: `width`, `height`
   - Descriptor set bindings:
     - `binding 0`: input buffer (SSBO, `float` array, channel-major `[C][H][W]`)
     - `binding 1`: output buffer (SSBO, `float` array)
     - `binding 2`: weight buffer (SSBO, conv kernel `[out_ch][in_ch][3][3]` + bias `[out_ch]`)
   - Each thread computes one spatial position `(x, y)` across all output channels:
     - 3×3 convolution: `sum += weight[oc][ic][ky][kx] * input[ic][y+ky-1][x+kx-1]`
     - Add bias: `sum += bias[oc]`
     - Write result to output buffer at `output[oc * H * W + y * W + x]`
   - Boundary handling: zero-padding (matches PyTorch `nn.Conv2d` `padding=1`)
   - **Note:** Output is raw conv+bias. GroupNorm and activation are applied by the subsequent `group_norm.comp` dispatch.

3. **Create `denoise/src/vulkan/shaders/group_norm.comp`** (normalize + scale + shift + activation):
   - **Two-dispatch approach:**
     - **Pass 1 (`group_norm_reduce.comp`):** Compute per-group partial sums and sum-of-squares. Workgroup: 256×1 threads. Each workgroup processes a tile of spatial positions for one group. Accumulates into shared memory, then writes partial results to a small reduction buffer (one entry per workgroup per group).
     - **Pass 2 (`group_norm_apply.comp`):** Finalize mean/variance from partial sums, then normalize + apply gamma/beta + LeakyReLU (if `ACTIVATION == 1`). Workgroup: 16×16, one output pixel per thread.
   - Specialization constants: `CHANNELS`, `NUM_GROUPS`, `ACTIVATION` (0=none, 1=LeakyReLU)
   - Push constants: `width`, `height`
   - Descriptor set bindings (pass 2):
     - `binding 0`: input/output buffer (SSBO, in-place normalization)
     - `binding 1`: norm params buffer (SSBO, GroupNorm `gamma[channels]` + `beta[channels]`)
     - `binding 2`: reduction buffer (SSBO, per-group mean + variance from pass 1)
   - LeakyReLU: `max(x, 0.01 * x)`
   > **Future enhancement:** If the two-dispatch overhead is measurable, consider fusing the reduction into a single dispatch using atomicAdd to global memory with a workgroup completion counter, or switching to a normalization scheme that doesn't require global reduction.

4. **Create `denoise/src/vulkan/shaders/encoder_input_conv.comp`** (first encoder conv, reads G-buffer):
   - Specialization constants: `OUT_CHANNELS` (16)
   - Push constants: `width`, `height`
   - Descriptor set bindings:
     - `binding 0`: noisy diffuse (storage image, RGBA16F, readonly) — channels 0–2 (RGB)
     - `binding 1`: noisy specular (storage image, RGBA16F, readonly) — channels 3–5 (RGB)
     - `binding 2`: world normals (storage image, RGBA16F, readonly) — channels 6–8 (XYZ), channel 9 (.w = roughness)
     - `binding 3`: linear depth (storage image, R16F or R32F, readonly) — channel 10
     - `binding 4`: motion vectors (storage image, RG16F, readonly) — channels 11–12
     - `binding 5`: output buffer (SSBO, `float` array, 16 channels)
     - `binding 6`: weight buffer (SSBO, conv kernel `[16][13][3][3]` + bias `[16]`)
   - Each thread: reads 13 input channels from the 5 G-buffer image views, applies 3×3 conv + bias, writes 16 output channels to buffer
   - Boundary handling: zero-padding (matches PyTorch `nn.Conv2d` `padding=1`)
   - **Note:** G-buffer layout is hardcoded here. If the G-buffer format changes, update this shader (or replace with input_assemble.comp + generic conv.comp).

5. **Create `denoise/src/vulkan/shaders/downsample.comp`:**
   - Workgroup: 16×16
   - 2×2 max pooling: read 4 input values per channel, write max to output at half resolution
   - Specialization constants: `CHANNELS`
   - Push constants: `in_width`, `in_height`
   - Descriptor set bindings:
     - `binding 0`: input buffer (SSBO)
     - `binding 1`: output buffer (SSBO)

6. **Create `denoise/src/vulkan/shaders/upsample_concat.comp`:**
   - Workgroup: 16×16
   - Bilinear 2× upsampling of the input feature map buffer
   - Concatenates with skip connection buffer: output channels = `IN_CHANNELS + SKIP_CHANNELS`
   - Specialization constants: `IN_CHANNELS`, `SKIP_CHANNELS`
   - Push constants: `out_width`, `out_height` (output = skip resolution)
   - Descriptor set bindings:
     - `binding 0`: input buffer (SSBO, at half resolution)
     - `binding 1`: skip buffer (SSBO, at output resolution)
     - `binding 2`: output buffer (SSBO, concatenated)

7. **Create `denoise/src/vulkan/shaders/output_conv.comp`:**
   - Workgroup: 16×16
   - 1×1 convolution: 3 output channels from `IN_CHANNELS` (16) input channels
   - No normalization, no activation (linear output)
   - Specialization constant: `IN_CHANNELS`
   - Push constants: `width`, `height`
   - Descriptor set bindings:
     - `binding 0`: input buffer (SSBO)
     - `binding 1`: output image (storage image, RGBA16F — the denoiser output image)
     - `binding 2`: weight buffer (SSBO, `[3][16][1][1]` + bias `[3]`)
   - Writes RGB to output image, alpha = 1.0

8. **Implement dispatch sequence in `MlInference::Infer(VkCommandBuffer cmd, const DenoiserInput& input, VkImageView output_view)`:**

   Full dispatch sequence with buffer barriers between each step:

   ```
   EncoderInputConv(G-buffer → buf0_a, 13→16)        // reads G-buffer images directly
   GroupNormReduce(buf0_a, 16ch)
   GroupNormApply(buf0_a, 16ch, LeakyReLU)
   Conv(buf0_a → buf0_b, 16→16)
   GroupNormReduce(buf0_b, 16ch)
   GroupNormApply(buf0_b, 16ch, LeakyReLU)
   Copy(buf0_b → skip0)
   Downsample(buf0_b → buf1_a, 16ch)

   Conv(buf1_a → buf1_b, 16→32)
   GroupNormReduce(buf1_b, 32ch)
   GroupNormApply(buf1_b, 32ch, LeakyReLU)
   Conv(buf1_b → buf1_a, 32→32)
   GroupNormReduce(buf1_a, 32ch)
   GroupNormApply(buf1_a, 32ch, LeakyReLU)
   Copy(buf1_a → skip1)
   Downsample(buf1_a → buf2_a, 32ch)

   Conv(buf2_a → buf2_b, 32→64)              // bottleneck
   GroupNormReduce(buf2_b, 64ch)
   GroupNormApply(buf2_b, 64ch, LeakyReLU)
   Conv(buf2_b → buf2_a, 64→64)
   GroupNormReduce(buf2_a, 64ch)
   GroupNormApply(buf2_a, 64ch, LeakyReLU)

   UpsampleConcat(buf2_a + skip1 → concat1, 64+32=96ch)
   Conv(concat1 → buf1_b, 96→32)
   GroupNormReduce(buf1_b, 32ch)
   GroupNormApply(buf1_b, 32ch, LeakyReLU)
   Conv(buf1_b → buf1_a, 32→32)
   GroupNormReduce(buf1_a, 32ch)
   GroupNormApply(buf1_a, 32ch, LeakyReLU)

   UpsampleConcat(buf1_a + skip0 → concat0, 32+16=48ch)
   Conv(concat0 → buf0_b, 48→16)
   GroupNormReduce(buf0_b, 16ch)
   GroupNormApply(buf0_b, 16ch, LeakyReLU)
   Conv(buf0_b → buf0_a, 16→16)
   GroupNormReduce(buf0_a, 16ch)
   GroupNormApply(buf0_a, 16ch, LeakyReLU)

   OutputConv(buf0_a → output_image, 16→3)
   ```

   - Buffer memory barriers (`VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT` → `VK_ACCESS_2_SHADER_STORAGE_READ_BIT`) between each dispatch
   - All dispatches recorded into the caller's `VkCommandBuffer`
   - `Copy` steps use `vkCmdCopyBuffer` to save encoder outputs for skip connections

9. **Create pipeline and descriptor set infrastructure:**
   - Compute pipelines: one per unique specialization constant combination. Created in `MlInference` constructor after weights are loaded (channel counts are known from the architecture constants). Shader `.spv` files loaded from `shader_dir` at runtime, matching the existing passthrough shader convention.
   - **Shaders (7 `.spv` files):** `encoder_input_conv.comp.spv`, `conv.comp.spv`, `group_norm_reduce.comp.spv`, `group_norm_apply.comp.spv`, `downsample.comp.spv`, `upsample_concat.comp.spv`, `output_conv.comp.spv`
   - Descriptor sets: pre-allocate a fixed pool of ~35 descriptor sets (one per dispatch step) at pipeline creation time. Each set is written once with the appropriate input/output/weight buffer bindings and updated on `Resize()`.
     > **Future enhancement:** Use `VK_KHR_push_descriptor` to bind descriptors inline and eliminate the fixed pool, or use dynamic buffer offsets to reduce descriptor set count.
   - Pipeline layouts: one shared layout per shader type (each shader defines its own binding layout). Push constant range: `{width, height}` (8 bytes).
   - A small reduction buffer for GroupNorm intermediate results: `num_groups × num_workgroups × 2 × sizeof(float)` (mean + variance partials). Allocated once at `Resize()`.

10. **Update `CMakeLists.txt`:** Add the 7 new `.comp` shaders to `DENI_SHADER_SOURCES` so they are compiled to SPIR-V alongside `passthrough_denoise.comp`.

11. **Create `training/scripts/generate_reference_output.py`:**
    - CLI: `python scripts/generate_reference_output.py --checkpoint model.pt --input input.exr --output reference.exr`
    - Loads the DeniUNet model from a checkpoint
    - Reads the input EXR, assembles the 13-channel tensor (matching G-buffer channel order)
    - Runs PyTorch inference (FP32)
    - Saves the 3-channel RGB output as an EXR file
    - **Note:** Does not compute comparison metrics — that is left to separate validation scripts or the C++ test.

### Verification
- All 7 shaders compile to SPIR-V without errors via `glslc`
- `MlInference::Infer()` records commands and the command buffer submits without Vulkan validation errors
- Output image contains non-zero, non-NaN data
- Compare GLSL output against PyTorch reference (`generate_reference_output.py`): max pixel difference < 0.01 (FP16 precision tolerance)
- Performance: inference time < 50ms at 1080p on RTX 4090 (very conservative target for MVP)

### Golden Reference Test (Model-Shader Sync Contract)

The primary numerical validation is the **golden reference test** (`tests/ml_inference_numerical_test.cpp`), which compares GPU shader output against PyTorch model inference. The test data (`tests/data/golden_ref.bin`) is generated by `tests/generate_golden_reference.py`, which imports `DeniUNet` directly from `training/deni_train/models/unet.py` — the actual training model code.

**Sync lifecycle when the model architecture changes** (e.g. adding temporal anti-aliasing):
1. Modify the PyTorch model in `training/deni_train/models/`
2. Regenerate the golden reference: `cd training && python ../tests/generate_golden_reference.py --output ../tests/data/golden_ref.bin`
3. Update the GLSL compute shaders in `denoise/src/vulkan/shaders/` to match the new architecture
4. Run tests — the golden reference test will fail if the shaders don't match the new model output

This ensures training model and GPU shaders stay in sync without maintaining a separate CPU reference implementation.

### GroupNorm Implementation Note

GroupNorm requires computing mean and variance over spatial+channel groups. For a feature map of size `(C, H, W)` with G groups, each group has `C/G` channels. The reduction is over `(C/G) × H × W` elements.

**Approach: separate global reduction dispatch** (`group_norm_reduce.comp` + `group_norm_apply.comp`):
1. **Reduce dispatch:** Each workgroup processes a tile of spatial positions for one group, accumulating partial sums and sum-of-squares into shared memory. Final partial results are written to a small global reduction buffer.
2. **Apply dispatch:** Reads the reduction buffer to compute final mean/variance per group, then normalizes each element, applies learned gamma/beta, and optionally applies LeakyReLU activation. Operates in-place on the feature buffer.

The reduction buffer is small: `G × ceil(H×W / workgroup_size) × 2 floats` — a few KB at most. The two-dispatch approach adds one extra dispatch per ConvBlock but is clean and correct.

> **Future enhancement alternatives:** (a) Atomic global memory accumulation with a completion counter (single dispatch, complex synchronization). (b) Replace GroupNorm with InstanceNorm or a per-pixel scheme that doesn't need spatial reduction (requires retraining). (c) Use subgroup shuffle operations for warp-level partial reductions to reduce shared memory traffic.

---

## Phase F11-3: End-to-End Integration + Validation ✅

**Goal:** Wire the ML denoiser into deni's public `Denoise()` API and monti_view. Run end-to-end quality validation.

**Prerequisite:** F11-2 (GLSL inference shaders + dispatch logic in `MlInference::Infer()`) — **COMPLETE**. All 7 compute shaders compiled, `MlInference::Infer()` dispatches the full U-Net, golden reference test validates GPU output against PyTorch (RMSE 0.000369, max error 0.003389). 19 deni tests pass with 3337 assertions.

### Design Decisions

- **Passthrough fallback:** The passthrough compute shader remains as a permanent fallback path. `Denoise()` branches: if `MlInference::IsReady()`, dispatch ML inference; otherwise dispatch passthrough. No performance cost when ML is active — the passthrough pipeline/descriptors are created at init but not dispatched.
- **Default denoiser mode:** ML inference is the default when a model file is found at the configured path. Users can toggle to passthrough via keyboard shortcut or UI. If no model file exists, passthrough is used silently.
- **Model file location:** The trained model is copied from `training/models/deni_v1.denimodel` into `denoise/models/deni_v1.denimodel` at build/install time. This keeps the deni library self-contained and avoids a runtime dependency on the training directory. A `DENI_MODEL_DIR` compile-time define (analogous to `DENI_SHADER_SPV_DIR`) provides the path.
- **Timing:** `LastPassTimeMs()` uses CPU wall-clock timing (`std::chrono`) around the `Denoise()` call. GPU timestamp queries are deferred to a future optimization pass.
- **Integration test pattern:** The test creates a headless Vulkan context following the existing patterns in `tests/` (e.g., `gpu_scene_test.cpp`, `deni_passthrough_test.cpp`) and `app/datagen/main.cpp`. Uses `VulkanContext::CreateDevice(std::nullopt)` for headless, `test_helpers.h` for proc address filling and image readback.
- **Quality comparison:** `compare_denoisers.py` uses pre-rendered EXRs from `monti_datagen` (noisy + reference pairs already produced by the training data pipeline). No significant modifications to `monti_datagen` required.

### Tasks

#### Task 1: Denoiser.cpp — timing + staging buffer cleanup

**Already implemented in F11-2:** `Denoise()` already dispatches ML inference when `ml_inference_->inference.IsReady()` is true ($\S$ Denoiser.cpp lines ~175-185), with passthrough fallback. Deferred weight upload on first `Denoise()` call is also in place.

**Remaining work:**
- Implement `LastPassTimeMs()` — currently returns hardcoded `0.0f`. Add `std::chrono::steady_clock` timing around the dispatch section. Store in a `float last_pass_time_ms_` member.
- Call `ml_inference_->inference.FreeStagingBuffer()` after the first successful weight upload (the staging buffer currently leaks until `Denoiser` destruction).

**Note:** CPU wall-clock measures command recording time, not GPU execution. Sufficient for development profiling; GPU timestamp queries deferred to optimization pass.

#### Task 2: Runtime denoiser mode switching

Add a `DenoiserMode` concept to support runtime toggling:
- Add `enum class DenoiserMode { kPassthrough, kMl };` to `Denoiser.h`.
- Add `void SetMode(DenoiserMode mode)` and `DenoiserMode Mode() const` to the `Denoiser` class.
- Add `DenoiserMode mode_ = DenoiserMode::kPassthrough;` private member.
- Update `Denoise()` dispatch logic (currently at lines ~175-185):
  - When `mode_ == kPassthrough`, always dispatch passthrough regardless of ML readiness.
  - When `mode_ == kMl`, dispatch ML inference if ready, else fall back to passthrough.
- In `Create()`: set `mode_ = DenoiserMode::kMl` when `model_path` was provided and weights loaded successfully; otherwise `kPassthrough`.
- Integrate mode with timing: only time the active dispatch path.

#### Task 3: Model file deployment + build integration

- Create `denoise/models/` directory for shipping model files.
- Add CMake logic to copy `training/models/deni_v1.denimodel` → `${CMAKE_CURRENT_BINARY_DIR}/deni_models/deni_v1.denimodel` at build time (similar to the shader SPIR-V output directory pattern).
- Add `DENI_MODEL_DIR` compile definition pointing to the output directory.
- In `Denoiser::Create()`: if `desc.model_path` is empty but `DENI_MODEL_DIR "/deni_v1.denimodel"` exists, auto-load it. This enables zero-config ML denoising when the model is available.
- `DenoiserDesc::model_path` remains available for explicit override (e.g., testing a different model).

#### Task 4: monti_view integration

Update `app/view/main.cpp` (denoiser creation is at lines ~577-592):
- Currently `denoiser_desc.model_path` is never set — it defaults to empty, so ML inference is never activated in monti_view.
- Set `denoiser_desc.model_path` to `DENI_MODEL_DIR "/deni_v1.denimodel"` (after Task 3 adds the define). If auto-discovery is implemented in Task 3, leave empty and let `Create()` handle it.
- Wire `PanelState` denoiser fields from `Denoiser` state each frame.
- Add `D` keybind to toggle `SetMode()` between `kMl` and `kPassthrough`.

Update `app/view/Panels.h` — add to `PanelState`:
- `DenoiserMode denoiser_mode` — current mode (for display and toggling)
- `bool has_ml_model` — whether ML model is loaded
- `float denoiser_time_ms` — last pass timing

Update `app/view/Panels.cpp`:
- Add "Denoiser" collapsing header in `DrawSettingsPanel()`:
  - Radio buttons: "Passthrough" / "ML" (disabled if `!has_ml_model`)
  - Text: denoiser pass time in ms
  - Text: model status ("ML model loaded" / "No model — passthrough only")
- Add keyboard shortcut `D` to toggle denoiser mode (when ML model is available).

Update top bar: show current denoiser mode label (e.g., "ML" or "Passthrough") after the debug mode indicator.

#### Task 5: Integration test

Create `tests/ml_denoiser_integration_test.cpp`:
- Follow the headless Vulkan pattern from `deni_passthrough_test.cpp` (which already tests `Denoiser::Create()`, `Denoise()`, `Resize()` at 64×64).
- Use the existing test model file (`test_output/ml_inference/test_model.denimodel`, created by `ml_inference_test.cpp`) or export a small one via `export_weights.py`.
- **Test 1 — ML denoiser produces output:**
  - Create `Denoiser` with `model_path` pointing to test `.denimodel` file
  - Call `Denoise()` with synthetic G-buffer data → read back output pixels
  - Assert: output is non-zero, no NaN/Inf values
  - Assert: output differs from passthrough-only output (ML actually ran)
- **Test 2 — Mode switching:**
  - Create denoiser with ML model, confirm `Mode() == kMl`
  - `SetMode(kPassthrough)` → output matches passthrough behavior
  - `SetMode(kMl)` → output differs from passthrough (ML ran)
- **Test 3 — Graceful fallback:**
  - Create denoiser with empty `model_path`
  - Confirm `Mode() == kPassthrough` and `HasMlModel() == false`
  - `Denoise()` still produces output (passthrough)
- Zero Vulkan validation errors (validation layers enabled in test context).

**Removed from original plan:** Test 2 (ML improves PSNR over passthrough) requires the production model and a rendered scene with reference. This is covered by manual quality validation and `compare_denoisers.py` (Task 6) instead of a unit test. The test model uses random weights and cannot be expected to improve PSNR.

#### Task 6: Quality comparison script

Create `training/scripts/compare_denoisers.py`:
- **Input:** A directory of pre-rendered EXR pairs from `monti_datagen` (noisy 4-SPP + reference 256-frame accumulated). These are the same EXR files already produced by the training data pipeline.
- **Processing:**
  1. Load noisy input EXR (diffuse + specular channels → combined noisy RGB)
  2. Load reference EXR (ground truth RGB)
  3. Load ML denoised EXR (if available — produced by running inference in Python using the exported ONNX model)
  4. Compute metrics: PSNR, SSIM, FLIP for noisy-vs-ref and denoised-vs-ref
- **Output:**
  - Per-scene metrics table (stdout + JSON)
  - HTML comparison gallery with side-by-side images (noisy | denoised | reference) with FLIP error maps
  - Summary statistics across all scenes
- **Usage:** `python scripts/compare_denoisers.py --data-dir training_data/ --model models/deni_v1.onnx --output results/comparison/`
- **Fallback mode:** If `--model` is not provided, compare only noisy vs reference (useful for baseline measurement).
- This script reuses the existing `metrics.py` (PSNR, SSIM) from F9-3. FLIP computation uses the `flip` Python package (add to `requirements.txt`).

#### Task 7: Performance measurement

- After integration is working, measure `LastPassTimeMs()` on RTX 4090 at 1080p.
- Record in a brief results table (not a separate document — add to the phase completion notes):
  - Passthrough time (expected: < 0.1 ms)
  - ML inference time (target: < 20 ms for interactive use)
- If ML inference exceeds 20 ms, note it as a follow-up optimization item but do not block the phase.

### Verification
- `monti_view` launches with ML denoiser active by default (when model file present)
- Denoised output is visibly smoother than passthrough (noisy) output
- `D` key toggles between ML and passthrough; visual difference is immediately apparent
- Integration test passes: all 4 test cases green, zero Vulkan validation errors
- ImGui "Denoiser" section shows mode, timing, and model status
- Top bar shows current denoiser mode
- `compare_denoisers.py` generates HTML gallery with metrics
- Quality comparison confirms ML PSNR > passthrough PSNR on validation scenes

### Implementation Notes

**Already implemented (from F11-2):**
- `Denoise()` dispatches `MlInference::Infer()` when model is loaded and ready (Denoiser.cpp ~line 175)
- Passthrough fallback when ML is not available
- `HasMlModel()` returns true when ML state exists
- `model_path` in `DenoiserDesc` loads weights at creation time

**Estimated new/modified files:**

| Category | New Files | Modified Files |
|---|---|---|
| Deni library | — | `Denoiser.h` (DenoiserMode enum, SetMode/Mode), `Denoiser.cpp` (mode dispatch, timing, staging cleanup) |
| Build | — | `CMakeLists.txt` (model copy rule, DENI_MODEL_DIR define) |
| App | — | `app/view/main.cpp` (model path, D key), `app/view/Panels.h` (denoiser state), `app/view/Panels.cpp` (denoiser UI) |
| Tests | `tests/ml_denoiser_integration_test.cpp` | `CMakeLists.txt` (test target) |
| Training | `training/scripts/compare_denoisers.py` | `training/requirements.txt` (flip package) |

---

## Future Phase Details (Outlined)

### F11-4: Temporal Extension — Training

**Goal:** Extend the training pipeline to use N=2–4 consecutive frames.

- Add `--camera-path` JSON support to `monti_datagen` for coherent frame sequences (smooth motion)
- Update `ExrDataset` to load frame sequences and apply motion vector warping
- Add `prev_frame_warped` input channels to the network (3 additional channels per frame: warped denoised RGB)
- Update U-Net `in_channels` from 13 to 13 + 3×(N-1) for warped history
- Add temporal consistency loss: L1 between warped previous prediction and current prediction, weighted by reprojection validity mask
- Retrain with sequential frame pairs
- Export updated weights
- **DoF training data:** Include camera configurations with varying f-stop (1.4–16.0) and focus distance. Include focus pulls (rack focus) in temporal sequences so the network learns aperture sample accumulation over time. See [dof_plan.md](dof_plan.md) for details.

### F11-5: Temporal Extension — Inference

**Goal:** Add frame history management to deni's ML inference path.

- `MlInference` maintains a ring buffer of N-1 previous denoised frames
- Per-frame: warp previous frames using motion vectors → concatenate with current G-buffer → infer
- Handle `reset_accumulation = true` by clearing the history buffer
- Update `conv_block.comp` and related shaders for increased input channel count
- Validate temporal stability on animated sequences

### F12: Super-Resolution Training + Inference

**Goal:** Train and deploy models for `ScaleMode::kQuality` (1.5×) and `ScaleMode::kPerformance` (2×).

- Add PixelShuffle upsampling layer to the decoder output (2× case)
- Train separate models per scale factor (or a single model conditioned on scale)
- Input at render resolution; output at target resolution (§4.11 of design spec)
- Update `MlInference::Resize()` to allocate output at target resolution
- 1.5× uses 3×-upsample + 2×-downsample internally (or learned 1.5× PixelShuffle)

### F13: Mobile Fragment Shader Inference (ncnn or Custom)

**Goal:** Deploy the ML denoiser on mobile Vulkan with TBDR-optimized fragment shaders.

- Evaluate ncnn's Vulkan backend for mobile deployment:
  - ncnn manages its own command buffers — acceptable on mobile where the frame loop structure differs
  - Automatic FP16, TBDR fragment shader path
- Alternative: write custom fragment shaders for tile-memory-friendly inference
- Depthwise separable convolutions for mobile (reduce ALU cost)
- Target: < 5ms at 540p on Snapdragon 8 Gen 3+

### Cloud Training Infrastructure

**Goal:** Enable parallel hyperparameter sweeps and large-scale training on cloud GPUs.

- `training/scripts/cloud_train.sh` — launch script for Lambda Labs / RunPod
- Docker container with all dependencies
- PyTorch DistributedDataParallel (DDP) for multi-GPU training via `torchrun`
- Weights & Biases or TensorBoard integration for remote experiment tracking
- Automated model export and upload to artifact storage

### Training Data Expansion

**Goal:** Broader scene coverage for improved generalization.

- **Stress scenes:** Programmatic generators for difficult cases (thin geometry, specular highlights, caustics, motion blur regions)
- **Community scenes:** Lumberyard Bistro variants, kitchen scenes, open-world outdoor scenes
- **Synthetic augmentation:** Procedural noise injection, lighting variation, material parameter sweeps
- **Hard example mining:** After initial training, identify high-loss regions and generate targeted training data

---

## Summary of Files Created per Phase

| Phase | New Files | Modified Files |
|---|---|---|
| F9-1 | `pyproject.toml`, `requirements.txt`, `exr_dataset.py`, `transforms.py`, `generate_synthetic_data.py`, `test_dataset.py` | `.gitignore` |
| F9-2 | `blocks.py`, `unet.py`, `tonemapping.py`, `denoiser_loss.py`, `test_model.py` | — |
| F9-3 | `default.yaml`, `small_test.yaml`, `train.py`, `evaluate.py`, `metrics.py`, `export_weights.py`, `test_export.py` | — |
| F9-4 | `export_cornell_box.py`, `download_scenes.py`, `generate_training_data.py`, `validate_dataset.py`, `training/.gitignore` | `requirements.txt` |
| F9-5 | `results/v1_baseline/v1_baseline.md` (auto-generated), `models/deni_v1.denimodel`, `models/deni_v1.onnx` | `evaluate.py` (add `--val-split`, `--report` flags) |
| F9-5b | `configs/sweep_*.yaml`, `results/sweep_summary.md` | `default.yaml` (if winner found) |
| F9-6a | `main.cpp` (`--position`/`--target`/`--fov`/`--viewpoints`), `GenerationSession.cpp` (multi-viewpoint loop + `Scene&`) | `GenerationSession.h` (`ViewpointEntry`, `GenerationConfig`, constructor), `Writer.h`/`Writer.cpp` (subdirectory param), `CMakeLists.txt` (`nlohmann_json` link) |
| F9-6b | `download_scenes.py` (expanded), `generate_viewpoints.py` (new), `test_viewpoints.py` (new) | `generate_training_data.py` (scene list only) |
| F9-6c | `RandomRotation180`, `ExposureJitter`; remove `RandomHorizontalFlip` from pipeline | `transforms.py`, `test_transforms.py` |
| F9-6d | Dataset generation orchestration | `generate_training_data.py`, `validate_dataset.py` |
| F9-7 | Updated `deni_v1.denimodel` | `default.yaml` |
| F11-1 | `WeightLoader.h`, `WeightLoader.cpp`, `MlInference.h`, `MlInference.cpp`, `ml_weight_loader_test.cpp` | `Denoiser.h`, `Denoiser.cpp` |
| F11-2 | `encoder_input_conv.comp`, `conv.comp`, `group_norm_reduce.comp`, `group_norm_apply.comp`, `downsample.comp`, `upsample_concat.comp`, `output_conv.comp`, `generate_reference_output.py` | `MlInference.h`, `MlInference.cpp`, `CMakeLists.txt` |
| F11-3 | `ml_denoiser_integration_test.cpp`, `compare_denoisers.py`, `denoise/models/deni_v1.denimodel` (copied) | `Denoiser.h`, `Denoiser.cpp`, `CMakeLists.txt`, `app/view/main.cpp`, `app/view/Panels.h`, `app/view/Panels.cpp`, `training/requirements.txt` |
