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

**Custom GLSL compute shaders** record directly into the caller's command buffer via `vkCmdDispatch`, preserving the existing API perfectly. The small U-Net requires only 4–5 parameterized compute shaders:

| Shader | Purpose |
|---|---|
| `conv_block.comp` | Conv3×3 + GroupNorm + LeakyReLU (handles all encoder/decoder blocks) |
| `downsample.comp` | 2× spatial downsampling (max pool or strided read) |
| `upsample_concat.comp` | 2× bilinear upsample + skip connection concatenation |
| `output_conv.comp` | Conv1×1 final projection → RGB output |

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
| F9-5b ✅ | Hyperparameter sensitivity sweep (optional) | 5 config variants trained; comparison summary identifies best hyperparameters |
| F9-6a ✅ | Multi-viewpoint rendering in `monti_datagen` | `--position`/`--target`/`--fov` CLI args + `--viewpoints` JSON batch mode; `Scene&` in `GenerationSession`; Writer subdirectory param; `nlohmann_json` linked; scene/Vulkan resources reused across viewpoints |
| F9-6b ✅ | Extended scene downloads + viewpoint generation | 10 new Khronos GLB models + 2 multi-file glTF downloaded; viewpoint JSONs generated per scene (24 viewpoints each) |
| F9-6c ✅ | Data augmentation transforms | `RandomRotation180`, `ExposureJitter`; `RandomHorizontalFlip` removed from pipeline (world-space normals incompatible with screen flips); unit tests |
| F9-6d | Full dataset generation + validation | `generate_training_data.py` updated with `--viewpoints` + `--env` support; ~1,680 frames rendered and validated; training dataloader confirmed working |
| F9-7 | Production training run | Retrained model with full dataset; quality assessment documented |
| F11-1 | Weight loading + inference buffers in Deni | Weights loaded from `.denimodel`, GPU buffers allocated, sizes verified |
| F11-2 | GLSL inference compute shaders | Inference dispatches produce output image; correctness validated against PyTorch reference |
| F11-3 | End-to-end integration + validation | ML denoiser in monti_view; A/B comparison with passthrough; integration test passes |

### Future Phases (Outlined)

| Phase | Feature | Prerequisite |
|---|---|---|
| F11-4 | Temporal extension — training (N=2–4 frame input, frame warping) + `--camera-path` JSON support in `monti_datagen` for multi-frame temporal sequences | F11-3 |
| F11-5 | Temporal extension — inference (frame history management in deni) | F11-4 |
| F12 | Super-resolution training + inference (`ScaleMode::kQuality`, `kPerformance`); add `--target-scale` CLI to `monti_datagen` | F11-3 |
| F13 | Mobile fragment shader inference (ncnn or custom, TBDR-optimized) | F11-3 + F6 |
| — | Albedo demodulation — add `albedo_d`/`albedo_s` as network inputs, train in albedo-divided space, remodulate after inference | F11-3 |
| — | Transparency output — use `diffuse.A`/`specular.A` alpha as transparency mask (currently geometry hit mask) | Renderer alpha support |
| — | Cloud training scripts (multi-GPU DDP, hyperparameter sweeps) | F9-7 |
| — | Broader scene acquisition + stress scene generation | F9-6d |

### Key Dependencies

```
11B (datagen) ──→ F9-4 (generate data) ──→ F9-5 (first training)
                                            ↓
F9-1 (scaffold) → F9-2 (model) → F9-3 (training loop) ─────→ F9-5
                                                                ↓
                                                      F9-5b (optional sweep)
                                                                ↓
                                            F9-6a/b/c/d (more scenes + augmentation) → F9-7 (production training)
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

## Phase F9-5b: Hyperparameter Sensitivity Sweep (Optional) ✅

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

**Goal:** Broaden the training set with additional scenes and viewpoints, and implement a data augmentation pipeline to increase effective dataset size. This phase is split into four sub-phases (F9-6a through F9-6d), each scoped to fit in a single Copilot session.

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

### Phase F9-6d: Full Dataset Generation + Validation

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

### Future Augmentation Enhancements

The initial augmentation pipeline (180° rotation + exposure jitter) is sufficient for MVP quality. The following techniques, used by production denoisers like DLSS, should be evaluated once the baseline model is trained:

- **Noise-level jitter** — Vary `--spp` across training frames (1, 2, 4, 8, 16) so the network learns to handle different input noise levels. The current plan uses a fixed SPP for all training data, which may cause the model to underperform at SPP values it wasn't trained on.
- **Auxiliary channel dropout** — During training, randomly zero out entire guide channels (normals, albedo, depth, motion vectors) with probability ~5–10% per channel. This prevents the network from over-relying on any single guide and improves robustness when guides are noisy or unreliable.
- **Random SPP mixing** — Render the same viewpoint at multiple SPP levels and randomly select one as input during training. More expensive than noise-level jitter (requires multiple renders per viewpoint) but produces better generalization across input quality levels.

---

## Phase F9-7: Production Training Run

**Goal:** Retrain with the full expanded dataset. Assess quality improvement over the F9-5 (or F9-5b, if completed) baseline. Export production weights.

### Tasks

1. Update `default.yaml` for production training:
   - Increase `epochs` to 200–300 (more data needs more training)
   - Adjust `batch_size` based on VRAM usage with larger dataset
   - Consider increasing `base_channels` to 32 if the small network underfits

2. Train on full dataset:
   - Local RTX 4090, expected 4–12 hours depending on dataset size and config
   - Monitor for overfitting: watch validation loss diverge from training loss
   - Use early stopping if validation loss plateaus

3. Evaluate quality:
   - Run `evaluate.py` on held-out validation set (frames reserved from each scene)
   - Compare metrics against F9-5 baseline
   - Generate comparison gallery: F9-5 model vs F9-7 model vs ground truth

4. Export production weights:
   - `export_weights.py` → `models/deni_v1.denimodel` (overwrite)
   - Verify file integrity

5. Document results:
   - Update `training/results/` with production training metrics
   - Record which scenes/augmentations contributed most to quality
   - Note areas where the model still struggles (expected: specular highlights, disocclusion regions)

### Verification
- Production model outperforms F9-5 baseline on held-out validation data
- PSNR improvement is measurable across all scene types
- Visual quality is noticeably better than passthrough (noisy) input
- No quality regression on any scene type relative to baseline
- Exported weights are valid

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

## Phase F11-2: GLSL Inference Compute Shaders

**Goal:** Implement the GLSL compute shaders for U-Net inference and the dispatch sequence in `MlInference`. Validate output correctness against PyTorch reference.

### Tasks

1. Create `denoise/src/vulkan/shaders/conv_block.comp`:
   - Workgroup: 16×16 threads, one output pixel per thread
   - Push constants: `in_channels`, `out_channels`, `width`, `height`, `num_groups` (for GroupNorm), `activation` (0=none, 1=LeakyReLU)
   - Descriptor set bindings:
     - `binding 0`: input images (storage image array)
     - `binding 1`: output images (storage image array)
     - `binding 2`: weight buffer (conv kernel `[out_ch][in_ch][3][3]` + bias `[out_ch]`)
     - `binding 3`: norm params buffer (GroupNorm `gamma[out_ch]` + `beta[out_ch]`)
   - Each thread computes one spatial position across all output channels:
     - 3×3 convolution: accumulate `sum += weight[oc][ic][ky][kx] * input[ic][y+ky-1][x+kx-1]`
     - Add bias
     - GroupNorm: compute per-group mean/variance via subgroup operations or shared memory reduction, then normalize + scale + shift
     - LeakyReLU: `max(x, 0.01 * x)`
   - Boundary handling: clamp-to-edge (zero-pad is noisier)

2. Create `denoise/src/vulkan/shaders/downsample.comp`:
   - Workgroup: 16×16
   - 2×2 max pooling: read 4 input pixels, write max to output (half resolution)
   - Push constants: `channels`, `in_width`, `in_height`

3. Create `denoise/src/vulkan/shaders/upsample_concat.comp`:
   - Workgroup: 16×16
   - Bilinear 2× upsampling of the input feature map
   - Concatenates with skip connection (reads from both input and skip image arrays)
   - Writes concatenated result to output image array
   - Push constants: `in_channels`, `skip_channels`, `width`, `height`

4. Create `denoise/src/vulkan/shaders/output_conv.comp`:
   - Workgroup: 16×16
   - 1×1 convolution: `out_channels` (3) from `in_channels` (16)
   - No normalization, no activation (linear output)
   - Writes directly to the denoiser's output image

5. Implement dispatch sequence in `MlInference::Infer(VkCommandBuffer cmd, ...)`:
   - **Input assembly:** Copy the 13 G-buffer input channels into the level-0 input images (simple copy/reformat compute shader, or direct binding if format-compatible)
   - **Encoder level 0:** ConvBlock(13→16) → ConvBlock(16→16) → save to skip_0 → Downsample
   - **Encoder level 1:** ConvBlock(16→32) → ConvBlock(32→32) → save to skip_1 → Downsample
   - **Bottleneck:** ConvBlock(32→64) → ConvBlock(64→64)
   - **Decoder level 1:** UpsampleConcat(64, skip_1=32 → 96) → ConvBlock(96→32) → ConvBlock(32→32)
   - **Decoder level 0:** UpsampleConcat(32, skip_0=16 → 48) → ConvBlock(48→16) → ConvBlock(16→16)
   - **Output:** OutputConv(16→3) → write to denoiser output image
   - Pipeline barriers between each dispatch (image memory barriers for WAR hazards)
   - All dispatches recorded into the caller's `VkCommandBuffer`

6. Create pipeline and descriptor set infrastructure:
   - Compute pipelines created once in `MlInference` constructor (4 shader variants)
   - Descriptor sets allocated per-dispatch (or re-bound with different buffer offsets)
   - Pipeline layout with push constants for per-dispatch parameters
   - Shader SPIR-V embedded as `constexpr uint32_t[]` arrays (matching deni convention from §4.7 of design spec)

7. Create `training/scripts/generate_reference_output.py`:
   - Given a `.denimodel` and an input EXR, runs PyTorch inference and saves the output
   - Used to verify GLSL inference matches PyTorch output
   - Compares: max absolute difference should be < 0.01 (FP16 precision tolerance)

### Verification
- All shaders compile to SPIR-V without errors
- `MlInference::Infer()` records commands and the command buffer submits without validation errors
- Output image contains non-zero, non-NaN data
- Compare GLSL output against PyTorch reference: max pixel difference < 0.01 (FP16 tolerance)
- Performance: inference time < 50ms at 1080p on RTX 4090 (very conservative target for MVP)

### GroupNorm Implementation Note

GroupNorm requires computing mean and variance over spatial+channel groups. For a feature map of size `(C, H, W)` with G groups, each group has `C/G` channels. The reduction is over `(C/G) × H × W` elements.

**Approach:** Two-pass within the conv_block shader:
1. First pass: compute partial sums and sum-of-squares per group using workgroup shared memory. Each thread accumulates over its spatial position for all channels in the group. Use `subgroupAdd` (if available) or shared memory atomic adds for cross-thread reduction.
2. Second pass: normalize using computed mean/variance, apply learned gamma/beta.

For the small feature map sizes in this network (especially at lower resolutions), the GroupNorm reduction is cheap. If profiling shows it's a bottleneck, it can be split into a separate reduction shader.

---

## Phase F11-3: End-to-End Integration + Validation

**Goal:** Wire the ML denoiser into deni's public `Denoise()` API and monti_view. Run end-to-end quality validation.

### Tasks

1. Update `denoise/src/vulkan/Denoiser.cpp`:
   - `Denoise()` checks if `MlInference` is initialized:
     - If yes: call `MlInference::Infer(cmd, input_images, output_image)`
     - If no: run existing passthrough compute shader
   - The 13 input channels are assembled from `DenoiserInput` fields:
     - `noisy_diffuse` → channels 0–2 (RGB)
     - `noisy_specular` → channels 3–5 (RGB)
     - `world_normals` → channels 6–8 (XYZ) and channel 9 (.w = roughness)
     - `linear_depth` → channel 10 (.r)
     - `motion_vectors` → channels 11–12 (XY)
   - Output image transitions to `VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL` as per existing contract

2. Update `DenoiserDesc` to accept optional model path:
   - Add `std::string model_path;` field (empty = passthrough mode)
   - Document in header: "Path to a `.denimodel` weight file. If empty or file not found, the denoiser operates in passthrough mode."

3. Update `app/view/` to support denoiser mode selection:
   - Add `--denoiser` CLI argument: `passthrough` (default) or `ml`
   - When `ml`: set `DenoiserDesc::model_path` to `models/deni_v1.denimodel`
   - Add ImGui toggle in the settings panel (if denoiser UI exists from F1, extend it; otherwise add it)
   - Display denoiser pass time from `Denoiser::LastPassTimeMs()`

4. Create integration test (`tests/ml_denoiser_integration_test.cpp`):
   - Render a Cornell box at 4 SPP (noisy)
   - Run ML denoiser with test weights
   - Read back output pixels
   - Verify: output is non-zero, non-NaN, different from passthrough output
   - Verify: output PSNR vs 64-SPP reference is better than noisy input PSNR vs same reference
   - Verify: zero Vulkan validation errors

5. Quality comparison script (`training/scripts/compare_denoisers.py`):
   - Renders a set of validation scenes in monti_view (or loads pre-rendered EXRs)
   - Compares: noisy (passthrough) vs ML denoised vs ground truth
   - Computes PSNR, SSIM, FLIP for each
   - Generates HTML comparison gallery
   - This script becomes the standard quality assessment tool for future model iterations

6. Performance measurement:
   - Measure `LastPassTimeMs()` on RTX 4090 at 1080p
   - Record baseline: passthrough time vs ML inference time
   - Target: < 20ms for interactive use in monti_view (at 1080p on RTX 4090)
   - Note: performance optimization is a future concern — correctness and quality first

### Verification
- `monti_view --denoiser ml` launches and displays denoised output
- Denoised output is visibly smoother than passthrough (noisy) output
- Integration test passes: ML denoiser produces measurably better PSNR than passthrough
- Zero Vulkan validation errors
- ImGui panel shows denoiser mode and timing
- Quality comparison gallery generated successfully

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
| F11-2 | `conv_block.comp`, `downsample.comp`, `upsample_concat.comp`, `output_conv.comp`, `generate_reference_output.py` | `MlInference.cpp` |
| F11-3 | `ml_denoiser_integration_test.cpp`, `compare_denoisers.py` | `Denoiser.cpp`, `app/view/main.cpp`, `app/view/Panels.cpp` |
