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
- **Parameters:** ~500K–1M (small enough for fast local training)
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
│   ├── generate_training_data.ps1  # Batch monti_datagen invocation
│   ├── generate_camera_paths.py    # Programmatic camera path generation
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

| Phase | Deliverable | Verifiable Outcome |
|---|---|---|
| F9-1 | Python project scaffold + EXR dataset loader | Loader reads synthetic EXR pairs, produces correct tensor shapes |
| F9-2 | U-Net architecture + loss functions | Forward pass with random input produces correct output shape; loss computes without error |
| F9-3 | Training loop + export scripts | Overfit on 2 synthetic samples: loss decreases to near zero; weights export to `.denimodel` |
| F9-4 | Initial training data generation | EXR pairs generated from test scenes; validation script confirms channel integrity |
| F9-5 | First training run + quality baseline | Model trained on real data; PSNR improvement over noisy input measurable |
| F9-6 | Extended scenes + data augmentation | Expanded dataset from Sponza, Bistro, etc.; augmentation pipeline functional |
| F9-7 | Production training run | Retrained model with full dataset; quality assessment documented |
| F11-1 | Weight loading + inference buffers in Deni | Weights loaded from `.denimodel`, GPU buffers allocated, sizes verified |
| F11-2 | GLSL inference compute shaders | Inference dispatches produce output image; correctness validated against PyTorch reference |
| F11-3 | End-to-end integration + validation | ML denoiser in monti_view; A/B comparison with passthrough; integration test passes |

### Future Phases (Outlined)

| Phase | Feature | Prerequisite |
|---|---|---|
| F11-4 | Temporal extension — training (N=2–4 frame input, frame warping) + camera path support in `monti_datagen` (`--camera-path` JSON, orbit/random generators) | F11-3 |
| F11-5 | Temporal extension — inference (frame history management in deni) | F11-4 |
| F12 | Super-resolution training + inference (`ScaleMode::kQuality`, `kPerformance`); add `--target-scale` CLI to `monti_datagen` | F11-3 |
| F13 | Mobile fragment shader inference (ncnn or custom, TBDR-optimized) | F11-3 + F6 |
| — | Albedo demodulation — add `albedo_d`/`albedo_s` as network inputs, train in albedo-divided space, remodulate after inference | F11-3 |
| — | Transparency output — use `diffuse.A`/`specular.A` alpha as transparency mask (currently geometry hit mask) | Renderer alpha support |
| — | Cloud training scripts (multi-GPU DDP, hyperparameter sweeps) | F9-7 |
| — | Broader scene acquisition + stress scene generation | F9-6 |

### Key Dependencies

```
11B (datagen) ──→ F9-4 (generate data) ──→ F9-5 (first training)
                                            ↓
F9-1 (scaffold) → F9-2 (model) → F9-3 (training loop) ─────→ F9-5
                                                                ↓
                                            F9-6 (more scenes) → F9-7 (production training)
                                                                    ↓
                            F11-1 (weights in deni) → F11-2 (shaders) → F11-3 (integration)
```

F9-1 through F9-3 (Python infrastructure) can proceed in parallel with Phase 11B completion, using synthetic test data. F9-4 onward requires a functional monti_datagen.

---

## Phase F9-1: Python Project Scaffold + EXR Dataset Loader

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

## Phase F9-2: U-Net Architecture + Loss Functions

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
   - `UpBlock(in_ch, skip_ch, out_ch)` — Upsample(scale=2, bilinear) + cat(skip) + two ConvBlocks
   - All blocks use `nn.Module` properly for parameter registration

2. Create `training/deni_train/models/unet.py`:
   - `DeniUNet(in_channels=13, out_channels=3, base_channels=16)` — `nn.Module`
   - Encoder: 2 `DownBlock`s (16→32→64 at bottleneck)
   - Bottleneck: 2 `ConvBlock`s at 64 channels
   - Decoder: 2 `UpBlock`s (64→32→16) with skip connections
   - Output: `nn.Conv2d(16, 3, kernel_size=1)` — no activation (linear HDR output)
   - `forward(x)` returns denoised RGB
   - Print parameter count in `__repr__`

3. Create `training/deni_train/utils/tonemapping.py`:
   - `aces_tonemap(x)` — ACES filmic tone mapping (matches monti's ACES implementation)
   - Operates on batched tensors `(B, 3, H, W)` in-place friendly
   - Used by the loss function to compute losses in perceptually uniform space

4. Create `training/deni_train/losses/denoiser_loss.py`:
   - `DenoiserLoss(lambda_l1=1.0, lambda_perceptual=0.1)` — `nn.Module`
   - L1 loss: `|aces(predicted) - aces(target)|` in tonemapped space
   - Perceptual loss: VGG-16 feature matching (relu1_2, relu2_2, relu3_3) on tonemapped RGB
   - VGG features extracted once, frozen, no gradient
   - Total: `lambda_l1 * L1 + lambda_perceptual * L_perceptual`
   - Tonemapping before loss prevents HDR firefly pixels from dominating gradients

5. Create `training/tests/test_model.py`:
   - Verify `DeniUNet` forward pass: input `(B=2, 13, 256, 256)` → output `(B=2, 3, 256, 256)`
   - Verify parameter count is in expected range (500K–1.5M)
   - Verify `DenoiserLoss` computes without NaN on random input
   - Verify loss decreases over 10 gradient steps on a fixed random batch (sanity check)

### Verification
- All tests pass
- Forward pass produces correct output shape
- Loss function computes and backpropagates without error
- Parameter count printed and within expected range

---

## Phase F9-3: Training Loop + Export Scripts

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
     lr_scheduler: cosine    # cosine annealing
     warmup_epochs: 5
     mixed_precision: true   # FP16 AMP
     checkpoint_interval: 10 # epochs
     log_interval: 50        # steps

   export:
     output_dir: "../models"
   ```

2. Create `training/configs/small_test.yaml`:
   - Override for fast pipeline validation: `batch_size: 2`, `epochs: 5`, `crop_size: 128`

3. Create `training/deni_train/train.py`:
   - CLI: `python -m deni_train.train --config configs/default.yaml [--resume checkpoint.pt]`
   - Loads config (YAML), creates dataset + dataloader + model + loss + optimizer (AdamW)
   - Training loop with:
     - FP16 mixed precision via `torch.amp.GradScaler`
     - Cosine annealing LR scheduler with linear warmup
     - TensorBoard logging (loss curves, learning rate, sample images every N epochs)
     - Periodic checkpoint saving (`model.pt` + optimizer state + epoch + best loss)
     - Best-model tracking (lowest validation loss)
   - Validation split: last 10% of dataset (or separate `val_dir` if provided)
   - Sample visualization: every 10 epochs, log input/predicted/target image triplets to TensorBoard (tonemapped for display)

4. Create `training/deni_train/evaluate.py`:
   - CLI: `python -m deni_train.evaluate --checkpoint model_best.pt --data_dir ../training_data`
   - Loads model + dataset, runs inference on full images (no cropping)
   - Computes per-image metrics: PSNR, SSIM (on tonemapped output)
   - Computes aggregate metrics: mean PSNR, mean SSIM
   - Saves side-by-side comparison PNGs: noisy input | denoised | ground truth
   - Prints results table to stdout

5. Create `training/deni_train/utils/metrics.py`:
   - `compute_psnr(predicted, target)` — PSNR in tonemapped space
   - `compute_ssim(predicted, target)` — SSIM via `torchvision.transforms.functional` or `skimage`

6. Create `training/scripts/export_weights.py`:
   - CLI: `python scripts/export_weights.py --checkpoint model_best.pt --output models/deni_v1.denimodel`
   - Loads PyTorch checkpoint, iterates `model.state_dict()`
   - Writes `.denimodel` binary format (header + per-layer name/shape/weights)
   - Also exports ONNX via `torch.onnx.export()` for reference
   - Prints layer summary: name, shape, parameter count

7. Create `training/tests/test_export.py`:
   - Create a small model, export to `.denimodel`, reload and verify weights match
   - Verify ONNX export runs without error
   - Verify `.denimodel` file is parseable (read header, verify magic/version/layer count)

### Verification
- Overfit test: train on 2 synthetic samples for 50 epochs with `small_test.yaml` — loss drops to < 0.01
- Checkpoint saves and resumes correctly (loss continues from where it left off)
- TensorBoard shows loss curves and sample images
- `export_weights.py` produces a `.denimodel` file with correct structure
- `evaluate.py` runs and prints metrics table
- All tests pass

---

## Phase F9-4: Initial Training Data Generation

**Goal:** Generate the first real training dataset from monti's existing test scenes using `monti_datagen`. Create validation tooling.

**Prerequisite:** Phase 11B (monti_datagen functional)

> **Note:** The initial `monti_datagen` uses auto-fitted camera only (no `--camera-path` support). Each invocation produces a single viewpoint. Camera path support (JSON file with multiple frames, orbit generators) will be added in a future phase alongside temporal denoising (F11-4), which requires multi-frame sequences from coherent camera motion. For now, multiple viewpoints are achieved by running `monti_datagen` multiple times with different `--exposure` or scene transforms, or by implementing a simple shell loop that perturbs camera parameters.

### Tasks

1. Create `training/scripts/generate_training_data.ps1`:
   - PowerShell script that invokes `monti_datagen` for each test scene
   - Configurable: `$MontiDatagen`, `$OutputDir`, `$Width`, `$Height`, `$Spp`, `$RefFrames`
   - Default: 960×540, 4 SPP noisy, 64 ref frames (64 × 4 = 256 effective reference SPP), `ScaleMode::kNative` (input = target resolution)
   - Iterates over scene files, creates per-scene output directories
   - Prints progress and total frame count
   - Example invocation for one scene:
     ```powershell
     & $MontiDatagen --output "$OutputDir/cornell_box/" `
         --width 960 --height 540 `
         --spp 4 --ref-frames 64 `
         "$ScenesDir/cornell_box.glb"
     ```
   - **Note:** `--camera-path` and `--target-scale` flags do not yet exist. Each run produces a single auto-fitted viewpoint. Multi-viewpoint generation and super-resolution scale modes will be added in future phases.

2. Create `training/scripts/validate_dataset.py`:
   - CLI: `python scripts/validate_dataset.py --data_dir ../training_data`
   - For each EXR pair:
     - Verify all expected channels present
     - Check for NaN/Inf values (fatal — indicates renderer bug)
     - Compute per-channel statistics: min, max, mean, std
     - Flag suspiciously low variance (possible black render)
   - Generate thumbnail gallery HTML page: tonemapped input vs target side-by-side
   - Print summary: total pairs, any issues found, channel statistics table

3. Add `training_data/` and `training/data_cache/` to `.gitignore`

### Verification
- `generate_training_data.ps1` runs and produces EXR file pairs in the output directory
- `validate_dataset.py` reports no NaN/Inf, reasonable channel statistics
- Thumbnail gallery shows recognizable noisy inputs and clean targets
- `ExrDataset` (from F9-1) loads the generated data without errors

### Initial Training Scenes

| Scene | Viewpoints | Tests |
|---|---|---|
| Cornell box (programmatic) | 1 | Diffuse GI, area light, color bleeding |
| DamagedHelmet.glb | 1 | PBR textures, normal maps, emissive |
| DragonAttenuation.glb | 1 | Transmission, volume attenuation |
| **Total** | **3** | |

This is a minimal dataset — sufficient for validating the full pipeline and getting a first quality signal, but far too small for production quality. Multi-viewpoint camera paths are added alongside temporal denoising in F11-4. Extended scenes are added in F9-6.

---

## Phase F9-5: First Training Run + Quality Baseline

**Goal:** Train the U-Net on real rendered data from F9-4. Establish a quality baseline and validate the full training→evaluation pipeline with real content.

### Tasks

1. Run training:
   - Use `default.yaml` config with `data_dir` pointing to F9-4 output
   - Train for 100 epochs on local RTX 4090
   - Monitor loss curves in TensorBoard
   - Expected training time: ~10–30 minutes for 3 viewpoints with 256×256 crops (small dataset, fast iteration)

2. Evaluate quality:
   - Run `evaluate.py` on the training data (note: this is training set evaluation, not generalization — acceptable for pipeline validation)
   - Compute PSNR and SSIM metrics
   - Generate side-by-side comparison images
   - Expected: PSNR improvement of 3–6 dB over raw noisy input (modest but measurable)

3. Export trained weights:
   - `export_weights.py` → `models/deni_v1.denimodel` + `models/deni_v1.onnx`
   - Verify `.denimodel` file is reasonable size (~2–6 MB for 500K–1M params)

4. Quality assessment document:
   - Record metrics in `training/results/v1_baseline.md`:
     - Per-scene PSNR and SSIM
     - Training loss curve (screenshot from TensorBoard)
     - Sample comparison images (3–5 representative frames)
     - Known limitations and observations
   - This document serves as the baseline for measuring future improvements

5. Hyperparameter sensitivity check (if time permits):
   - Try 2–3 variations: different `lambda_perceptual`, `base_channels=32`, `learning_rate`
   - Record which variation produces best PSNR
   - Update `default.yaml` if a clear winner emerges

### Verification
- Training completes without error; loss curve shows convergence
- PSNR improvement over noisy input is measurable (> 2 dB minimum)
- Visual comparison shows denoised output is visibly smoother than noisy input
- Exported weights file is valid and correctly sized
- No NaN/Inf in model output

---

## Phase F9-6: Extended Scenes + Data Augmentation

**Goal:** Broaden the training set with additional scenes and implement a data augmentation pipeline to increase effective dataset size.

### Tasks

1. Create `training/scripts/download_scenes.py`:
   - Download publicly available glTF scenes:
     - **Intel Sponza** — architectural lighting, environment + area lights (~262K triangles)
     - **Amazon Lumberyard Bistro** — complex interior/exterior, many lights (~2.8M triangles)
     - Other freely available glTF scenes as identified (NVIDIA ORCA collection, Khronos samples)
   - Downloads to a configurable directory (default: `training/scenes/`)
   - Validates downloaded files (checksums or basic parse check)
   - Scenes are gitignored — the download script is the reproducibility mechanism

2. Create camera paths for extended scenes:
   - `sponza_walkthrough.json` — 128-frame walkthrough of Sponza interior
   - `sponza_orbit.json` — 64-frame orbit from outside
   - `bistro_interior.json` — 128-frame path through bistro interior
   - `bistro_exterior.json` — 64-frame path around bistro exterior
   - Use `generate_camera_paths.py` with manually tuned center/radius/elevation parameters

3. Update `generate_training_data.ps1` to include extended scenes:
   - Add sections for each new scene
   - Total target: ~1000–2000 frames across all scenes

4. Implement additional augmentation in `training/deni_train/data/transforms.py`:
   - `RandomRotation90()` — 0°/90°/180°/270° rotation (both input and target, adjust motion vectors)
   - `ExposureJitter(range=(-1.0, 1.0))` — multiply radiance channels by random exposure factor (both input and target, identically)
   - `RandomVerticalFlip(p=0.5)` — vertical flip (negate motion vector Y)
   - Update `Compose` to chain all augmentations

5. Create `training/scripts/augment_dataset.py` (optional offline augmentation):
   - Pre-generates augmented crops from full-resolution EXR pairs
   - Useful for reducing dataloader overhead during training
   - Generates N random crops per source frame (default N=8)
   - Writes augmented crops as separate EXR files or numpy `.npz` for faster loading

6. Generate extended dataset:
   - Run `generate_training_data.ps1` with all scenes
   - Run `validate_dataset.py` to confirm quality
   - Expected: 1000–2000 total training frames

### Verification
- Extended scenes download and load successfully in monti_datagen
- Camera paths produce reasonable viewpoints (validate with monti_view if needed)
- Augmentation transforms produce correctly aligned input/target pairs
- Augmented motion vectors are correctly adjusted for flips/rotations
- Validation script reports no issues with the expanded dataset
- Dataset loader handles the larger dataset without memory issues

### Extended Training Scenes

| Scene | Frames | Purpose |
|---|---|---|
| Cornell box | 64 | Diffuse GI baseline |
| DamagedHelmet | 128 | PBR textures |
| DragonAttenuation | 64 | Transmission/refraction |
| Intel Sponza | 192 | Architectural lighting, environment maps |
| Bistro (interior) | 128 | Complex interior, many emissive surfaces |
| Bistro (exterior) | 64 | Outdoor lighting, large geometry |
| Additional Khronos samples | ~200 | Material variety (ClearCoatTest, MaterialsVariantsShoe, etc.) |
| **Total** | **~850** | |

With 8× augmentation crops per frame: ~6800 effective training samples.

### Future Augmentation Enhancements

The initial augmentation pipeline (geometric transforms + exposure jitter) is sufficient for MVP quality. The following techniques, used by production denoisers like DLSS, should be evaluated once the baseline model is trained:

- **Noise-level jitter** — Vary `--spp` across training frames (1, 2, 4, 8, 16) so the network learns to handle different input noise levels. The current plan uses a fixed SPP for all training data, which may cause the model to underperform at SPP values it wasn't trained on.
- **Auxiliary channel dropout** — During training, randomly zero out entire guide channels (normals, albedo, depth, motion vectors) with probability ~5–10% per channel. This prevents the network from over-relying on any single guide and improves robustness when guides are noisy or unreliable.
- **Random SPP mixing** — Render the same viewpoint at multiple SPP levels and randomly select one as input during training. More expensive than noise-level jitter (requires multiple renders per viewpoint) but produces better generalization across input quality levels.

---

## Phase F9-7: Production Training Run

**Goal:** Retrain with the full expanded dataset. Assess quality improvement over the F9-5 baseline. Export production weights.

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

- Update `monti_datagen` camera paths to generate coherent frame sequences (smooth motion)
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
| F9-4 | `cornell_box_orbit.json`, `damaged_helmet_orbit.json`, `dragon_attenuation_orbit.json`, `generate_camera_paths.py`, `generate_training_data.ps1`, `validate_dataset.py` | `.gitignore` |
| F9-5 | `results/v1_baseline.md`, `deni_v1.denimodel`, `deni_v1.onnx` | — |
| F9-6 | `download_scenes.py`, extended camera paths, `augment_dataset.py` | `transforms.py`, `generate_training_data.ps1` |
| F9-7 | Updated `deni_v1.denimodel` | `default.yaml` |
| F11-1 | `WeightLoader.h`, `WeightLoader.cpp`, `MlInference.h`, `MlInference.cpp`, `ml_weight_loader_test.cpp` | `Denoiser.h`, `Denoiser.cpp` |
| F11-2 | `conv_block.comp`, `downsample.comp`, `upsample_concat.comp`, `output_conv.comp`, `generate_reference_output.py` | `MlInference.cpp` |
| F11-3 | `ml_denoiser_integration_test.cpp`, `compare_denoisers.py` | `Denoiser.cpp`, `app/view/main.cpp`, `app/view/Panels.cpp` |
