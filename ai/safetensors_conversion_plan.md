# Plan: Safetensors Training Data Format

> **Purpose:** Eliminate the per-file EXR parsing bottleneck in the training
> `DataLoader` by converting training data to the safetensors format. Current
> training runs at ~435 s/epoch with GPU utilization at 2-20%, caused by
> Python/OpenEXR overhead of reading 19 channels across 2 files per sample (×1,270
> samples per epoch). Expected improvement: 5-8× epoch time reduction.
>
> **Prerequisites:** Working EXR training pipeline (train, evaluate, validate_dataset).
> Production training already running successfully on EXR data.
>
> **Relationship to existing plans:** Extends
> [ml_denoiser_plan.md](ml_denoiser_plan.md) (F9 training infrastructure).
> Independent of [datagen_performance_plan.md](datagen_performance_plan.md) (datagen
> write path unchanged — datagen still produces EXR, conversion is a post-processing
> step).
>
> **Session sizing:** 5 phases, each scoped for implementation + testing in a single
> Copilot session. Total: ~1 session for all phases.

---

## Background

### The I/O Bottleneck

The current `ExrDataset` (in `training/deni_train/data/exr_dataset.py`) reads two
EXR files per sample via the Python `OpenEXR` bindings:

| Per-sample I/O | Details |
|---|---|
| **2 file opens** | `OpenEXR.InputFile()` for input + target |
| **19 channel reads** | 13 input channels + 6 target channels (diffuse+specular RGB) |
| **Per-channel overhead** | Header parse, channel seek, float32 decode, `np.frombuffer` |
| **Float16 conversion** | `torch.from_numpy(...).to(torch.float16)` per tensor |
| **Target summation** | `diffuse + specular` combined on CPU per sample |

This produces ~435 s/epoch on an i9-12900K + RTX 4090 with NVMe storage, despite
the GPU being idle 80-98% of the time. The bottleneck is not disk throughput but
Python/C boundary crossings and per-channel seeks in the OpenEXR library.

### Why Safetensors

| Format | Pros | Cons |
|---|---|---|
| **EXR** (current) | Industry standard, human-inspectable | Slow per-channel parsing, no memory mapping |
| **HDF5** | Fast reads, compression options | Fork-safety issues with `num_workers > 0`, requires `h5py` |
| **.pt** (pickle) | Native PyTorch | Pickle deserialization overhead, arbitrary code execution risk |
| **safetensors** ✅ | Memory-mapped, zero-copy, no pickle, JSON header + raw bytes, HuggingFace standard | Extra conversion step, not human-inspectable |

Safetensors stores tensors as a JSON metadata header followed by raw byte buffers.
`safetensors.torch.load_file()` memory-maps the file and returns tensors with
zero-copy reads. This eliminates per-channel parsing, float conversion, and Python
object allocation overhead.

---

## Phase Overview

| Phase | Scope | Files Changed | Test |
|---|---|---|---|
| **S1** | Conversion script | `scripts/convert_to_safetensors.py` (new) | Convert smoke test data, verify tensor shapes/dtypes |
| **S2** | Safetensors dataset class | `data/safetensors_dataset.py` (new), `data/__init__.py` | Load converted data, compare tensors to EXR loader |
| **S3** | Training integration | `train.py`, `configs/default.yaml` | Train 2 epochs on converted data, verify loss matches |
| **S4** | Evaluation integration | `evaluate.py` | Evaluate checkpoint on converted data |
| **S5** | Pipeline integration | `validate_dataset.py`, documentation | End-to-end pipeline test |

---

## Phase S1: Conversion Script

> **Scope:** Create `training/scripts/convert_to_safetensors.py` that reads EXR
> pairs and writes one `.safetensors` file per sample containing the pre-processed
> input and target tensors. Add `safetensors` to project dependencies.

### Rationale

Pre-processing at conversion time moves all the expensive work (EXR parsing, channel
extraction, float16 conversion, diffuse+specular summation) to a one-time offline
step. The training loop then loads pre-computed tensors directly.

### File Format

Each `.safetensors` file contains two tensors:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `"input"` | `(13, H, W)` | `float16` | 13-channel denoiser input (same as `ExrDataset`) |
| `"target"` | `(3, H, W)` | `float16` | Combined diffuse+specular radiance |

The filename preserves the original EXR pair identity:
- Directory-based: `SceneName/variation_001.safetensors` (from `SceneName/variation_001/input.exr`)
- Flat naming: `SceneName/variation_001.safetensors` (from `SceneName/variation_001_input.exr`)

### Changes

**`training/scripts/convert_to_safetensors.py`** (new)

```python
"""Convert EXR training data pairs to safetensors format.

Usage:
    python scripts/convert_to_safetensors.py --data_dir training_data/ --output_dir training_data_st/
    python scripts/convert_to_safetensors.py --data_dir training_data/ --output_dir training_data_st/ --verify
"""
```

- **CLI arguments:**
  - `--data_dir` (required): Path to EXR training data directory.
  - `--output_dir` (required): Path to output safetensors directory.
  - `--verify`: After conversion, reload each `.safetensors` file and compare to
    the EXR source (max absolute error < 1e-3 for float16 round-trip).
- **Logic:**
  1. Discover EXR pairs using the same glob logic as `ExrDataset.__init__`.
  2. For each pair, read input/target EXR channels using `_read_exr_channels`.
  3. Assemble tensors identical to `ExrDataset.__getitem__` (13-ch input, 3-ch
     target, both float16).
  4. Write `{"input": input_tensor, "target": target_tensor}` via
     `safetensors.torch.save_file()`.
  5. Preserve directory structure (create subdirectories under `output_dir`).
  6. Print progress bar and summary (total files, output size, time elapsed).
- **Verification mode (`--verify`):**
  - Reload each `.safetensors` file.
  - Reload corresponding EXR pair via `_read_exr_channels`.
  - Assert `torch.allclose(st_input, exr_input, atol=1e-3)` for both tensors.
  - Report pass/fail count.

**Dependency:** Add `safetensors>=0.4` to project dependencies.

### Acceptance Criteria

1. `python scripts/convert_to_safetensors.py --data_dir training_data/ --output_dir training_data_st/ --verify` completes with 0 failures.
2. Output directory structure mirrors input directory structure.
3. Each `.safetensors` file contains `"input"` (13, 540, 960) float16 and `"target"` (3, 540, 960) float16.
4. Total output size is within 5% of theoretical minimum (13 + 3 = 16 channels × 540 × 960 × 2 bytes = ~15.8 MB per file, ~20 GB total for 1,270 pairs).

---

## Phase S2: Safetensors Dataset Class

> **Scope:** Create `SafetensorsDataset` in
> `training/deni_train/data/safetensors_dataset.py` that loads `.safetensors` files
> with the same interface as `ExrDataset`.

### Changes

**`training/deni_train/data/safetensors_dataset.py`** (new)

```python
"""Safetensors dataset loader for pre-converted monti training data."""
```

- **Class `SafetensorsDataset(Dataset)`:**
  - `__init__(self, data_dir: str, transform=None)`:
    - Discover all `*.safetensors` files in `data_dir` (recursive glob).
    - Sort for deterministic ordering.
    - Store `self.files` list and `self.transform`.
  - `__len__()`: Return `len(self.files)`.
  - `__getitem__(self, idx)`:
    - Load file via `safetensors.torch.load_file(self.files[idx])`.
    - Extract `"input"` and `"target"` tensors.
    - Apply `self.transform` if present.
    - Return `(input_tensor, target_tensor)`.

- **Interface contract:** Returns the same `(input_tensor, target_tensor)` pair as
  `ExrDataset.__getitem__`, with identical shapes, dtypes, and value ranges.

**`training/deni_train/data/__init__.py`**

- Export both `ExrDataset` and `SafetensorsDataset` for clean imports.

### Acceptance Criteria

1. `SafetensorsDataset` returns tensors identical (within float16 precision) to
   `ExrDataset` for the same underlying data.
2. `len(SafetensorsDataset(...))` equals `len(ExrDataset(...))` for the same dataset.
3. Loading a single sample from `SafetensorsDataset` is measurably faster than
   `ExrDataset` (simple timing comparison in a test script or inline verification).

---

## Phase S3: Training Integration

> **Scope:** Update `train.py` to auto-detect safetensors data and use
> `SafetensorsDataset` when available. Add `data_format` config option as override.

### Changes

**`training/deni_train/train.py`**

- **Auto-detection logic** (in dataset construction, around the `ExrDataset` call):
  1. If `config.get("data_format") == "exr"`: force `ExrDataset`.
  2. If `config.get("data_format") == "safetensors"`: force `SafetensorsDataset`.
  3. Otherwise (default `"auto"`): check if any `*.safetensors` files exist in
     `data_dir`. If yes, use `SafetensorsDataset`. If no, fall back to `ExrDataset`.
  4. Log which dataset class is being used.

- **Import:** Add conditional import of `SafetensorsDataset`.

**`training/configs/default.yaml`**

- Add `data_format: "auto"` with a comment explaining the options (`auto`, `exr`,
  `safetensors`).

### Acceptance Criteria

1. `python -m deni_train.train --config configs/default.yaml` with safetensors data
   in `data_dir` automatically uses `SafetensorsDataset`.
2. Same command with only EXR data automatically uses `ExrDataset`.
3. Explicit `data_format: "exr"` forces EXR loading even when safetensors exist.
4. Train 2 epochs on converted safetensors data — loss values are comparable to
   EXR training (within normal stochastic variation).
5. Epoch time is significantly reduced (target: < 100 s/epoch vs ~435 s/epoch).

---

## Phase S4: Evaluation Integration

> **Scope:** Update `evaluate.py` to support safetensors data with the same
> auto-detection logic.

### Changes

**`training/deni_train/evaluate.py`**

- **Auto-detection:** Same logic as Phase S3 — check for `*.safetensors` files in
  `data_dir`, use `SafetensorsDataset` if found, else `ExrDataset`.
- **CLI option:** Add `--data-format` argument (`auto`, `exr`, `safetensors`) with
  default `"auto"`.
- **Import:** Add conditional import of `SafetensorsDataset`.

**Note:** `evaluate.py` extracts scene names from file paths via
`scene_name_from_pair()`. The `SafetensorsDataset` must expose a `files` attribute
(list of file paths) so that `scene_name_from_pair()` (or a similar helper) can
extract scene names from `.safetensors` paths.

### Scene Name Extraction

`scene_name_from_pair()` currently extracts scene names from EXR path tuples. For
safetensors, the scene name extraction needs to work on single file paths.
Options:
- Add `scene_name_from_file(path)` helper in `splits.py` that extracts the scene
  name from a safetensors path (parent directory name).
- Update `evaluate.py` to use the new helper when working with `SafetensorsDataset`.

### Acceptance Criteria

1. `python -m deni_train.evaluate --checkpoint model_best.pt --data_dir training_data_st/`
   produces correct per-image and per-scene metrics.
2. Per-scene grouping works correctly with safetensors file paths.
3. PSNR/SSIM values match (within float16 precision) those from EXR evaluation.

---

## Phase S5: Pipeline Integration and Documentation

> **Scope:** Update `validate_dataset.py` to support safetensors data. Update or
> create documentation for the new pipeline step.

### Changes

**`training/scripts/validate_dataset.py`**

- `validate_dataset.py` performs deep validation (NaN/Inf detection, per-channel
  statistics, variance checks, HTML gallery). For safetensors:
  - Add auto-detection: if `*.safetensors` files found, validate those instead of
    EXR.
  - Load tensors via `safetensors.torch.load_file()` and run the same statistical
    checks on the `"input"` and `"target"` tensors.
  - Map tensor channel indices back to logical channel names for reporting (e.g.,
    channel 0-2 = diffuse RGB, 3-5 = specular RGB, etc.).
  - Gallery generation: extract noisy radiance (channels 0-2 + 3-5) and target from
    tensors for ACES-tonemapped thumbnails.

**`training/README.md`**

Insert a new step between the existing step 6 (validate_dataset) and step 8 (training)
that documents the safetensors conversion:

```
7b. Convert EXR training data to safetensors for faster training:
\```
python scripts\convert_to_safetensors.py `
    --data_dir training_data `
    --output_dir training_data_st `
    --verify
\```
Converts each EXR input/target pair into a single `.safetensors` file with
pre-processed float16 tensors. The `--verify` flag re-reads each converted file
and compares it to the EXR source. This step is optional — training auto-detects
safetensors data if present, otherwise falls back to EXR.

7c. Validate the converted safetensors dataset:
\```
python scripts\validate_dataset.py `
    --data_dir training_data_st `
    --gallery training_data_st\gallery.html
\```
```

Also add a note to the existing training step (step 8/10) explaining the
`data_format` config option:

```
Training auto-detects safetensors data in `data_dir` when available. To force
a specific format, set `data_format: "exr"` or `data_format: "safetensors"` in
the config YAML.
```

**Pipeline documentation update:**

The full training pipeline becomes:

```
generate_viewpoints.py       → viewpoints/*.json
generate_light_rigs.py       → light_rigs/*.json
generate_training_data.py    → training_data/SceneName/*_{input,target}.exr
remove_invalid_viewpoints.py → moves invalid pairs to invalid_training_data/
validate_dataset.py          → training_data/gallery.html (validates EXR)
convert_to_safetensors.py    → training_data_st/SceneName/*.safetensors  [NEW]
validate_dataset.py          → training_data_st/gallery.html (validates ST) [NEW]
train.py                     → configs/checkpoints/model_best.pt
evaluate.py                  → results/ (metrics + comparison PNGs)
export_weights.py            → models/deni_v1.denimodel + ONNX
```

### Acceptance Criteria

1. `python scripts/validate_dataset.py --data_dir training_data_st/` completes
   successfully with 0 errors on converted data.
2. HTML gallery generates correctly from safetensors data.
3. `training/README.md` updated with conversion step and `data_format` note.
4. Full pipeline runs end-to-end: convert → validate → train → evaluate → export.

---

## Implementation Notes

### Safetensors Library Usage

```python
from safetensors.torch import save_file, load_file

# Write
save_file({"input": input_tensor, "target": target_tensor}, "sample.safetensors")

# Read (memory-mapped, zero-copy)
tensors = load_file("sample.safetensors")
input_tensor = tensors["input"]   # (13, H, W) float16
target_tensor = tensors["target"] # (3, H, W) float16
```

### File Size Estimates

| Metric | Value |
|---|---|
| Channels per sample | 16 (13 input + 3 target) |
| Resolution | 960 × 540 |
| Bytes per pixel | 2 (float16) |
| Per-file size | 16 × 960 × 540 × 2 = 16,588,800 bytes ≈ 15.8 MB |
| Total (1,270 files) | ~20.1 GB |
| Overhead vs EXR | ~0% (EXR files are similar size uncompressed) |

### Performance Expectations

| Metric | EXR (current) | Safetensors (expected) |
|---|---|---|
| Per-sample load time | ~340 ms | ~5-20 ms |
| Epoch time (1,270 samples) | ~435 s | ~55-90 s |
| GPU utilization | 2-20% | 60-90% |
| Bottleneck | I/O (EXR parsing) | Compute (model forward/backward) |

### Backward Compatibility

- EXR training data is **preserved** — conversion is additive, not destructive.
- `ExrDataset` remains fully functional and is the fallback when safetensors data is
  not available.
- All tools (`train.py`, `evaluate.py`, `validate_dataset.py`) default to
  `"auto"` detection, so existing workflows continue to work unchanged.
- The `data_format` config/CLI option provides explicit override for any scenario.

### Reuse of Existing Code

- `convert_to_safetensors.py` reuses `_read_exr_channels` and the channel assembly
  logic from `exr_dataset.py` to ensure identical tensor construction.
- `SafetensorsDataset` shares the same `(input_tensor, target_tensor)` interface and
  transform contract as `ExrDataset`.
- `splits.py` stratified split logic works on both dataset types (operates on
  indices, not file formats).
