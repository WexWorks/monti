# Plan: Skip Intermediate Safetensor Stage in Training Pipeline

## Overview

The current pipeline has an intermediate stage (`convert_to_safetensors.py`) that converts
compressed EXR pairs from `training_data/` into uncompressed full-resolution `.safetensors`
files in `training_data_st/`, which are then consumed by `preprocess_temporal.py` to extract
crops. This is wasteful: the intermediate safetensors consume significantly more disk space
than the compressed EXRs, and the crops consume even more again.

**Pipeline before:**
```
training_data/ (EXR)  →  convert_to_safetensors  →  training_data_st/ (safetensors)
                                                   →  preprocess_temporal  →  training_data_cropped_st/
```

**Pipeline after:**
```
training_data/ (EXR)  →  preprocess_temporal  →  training_data_cropped_st/
```

`preprocess_temporal.py` will absorb the EXR loading and preprocessing logic from
`convert_to_safetensors.py`. The output format of `preprocess_temporal.py` (the cropped
`.safetensors` files consumed by the training DataLoader) is **unchanged**.

---

## What `convert_to_safetensors.py` Does — Full Transformation

This is the complete set of operations the script applies, which must be replicated in
`preprocess_temporal.py`:

### Input EXR → `"input"` tensor

1. **Read** 19 named channels from `*_input.exr` as `float32` numpy arrays using
   `_read_exr_channels(path, _INPUT_CHANNEL_NAMES)` from `deni_train/data/exr_dataset.py`.

2. **Stack** channels into shape `(19, H, W)` in the order defined by
   `_INPUT_CHANNEL_NAMES` (see below).

3. **Extract albedo** arrays from the stacked input:
   - `albedo_d = input_arrays[13:16]`   — diffuse albedo RGB
   - `albedo_s = input_arrays[16:19]`   — specular albedo RGB

4. **Demodulate diffuse irradiance** (channels 0–2):
   ```python
   input_arrays[0:3] = input_arrays[0:3] / np.maximum(albedo_d, _DEMOD_EPS)
   ```

5. **Demodulate specular irradiance** (channels 3–5):
   ```python
   input_arrays[3:6] = input_arrays[3:6] / np.maximum(albedo_s, _DEMOD_EPS)
   ```

6. **Clip** all 19 channels to float16 range:
   ```python
   np.clip(input_arrays, -65504.0, 65504.0, out=input_arrays)
   ```

7. **Cast** to `torch.float16`:
   ```python
   input_tensor = torch.from_numpy(input_arrays).to(torch.float16)
   ```
   Result shape: `(19, H, W)` float16.

### Target EXR → `"target"` tensor

1. **Read** 6 named channels from `*_target.exr` as float32:
   - `_TARGET_DIFFUSE = ("diffuse.R", "diffuse.G", "diffuse.B")`
   - `_TARGET_SPECULAR = ("specular.R", "specular.G", "specular.B")`

2. **Stack** into `target_d (3, H, W)` and `target_s (3, H, W)` separately.

3. **Demodulate target using the albedo extracted from the input EXR**
   (albedo is a material property, identical regardless of SPP):
   ```python
   target_d = target_d / np.maximum(albedo_d, _DEMOD_EPS)
   target_s = target_s / np.maximum(albedo_s, _DEMOD_EPS)
   ```

4. **Clip** each to `[-65504.0, 65504.0]` separately:
   ```python
   np.clip(target_d, -65504.0, 65504.0, out=target_d)
   np.clip(target_s, -65504.0, 65504.0, out=target_s)
   ```

5. **Concatenate** and cast:
   ```python
   target_arrays = np.concatenate([target_d, target_s], axis=0)  # (6, H, W)
   target_tensor = torch.from_numpy(target_arrays).to(torch.float16)
   ```
   Result shape: `(6, H, W)` float16.

### Channel Name Constants (from `exr_dataset.py`)

```python
_INPUT_CHANNEL_NAMES = [
    "diffuse.R", "diffuse.G", "diffuse.B",           # ch 0-2  noisy diffuse
    "specular.R", "specular.G", "specular.B",         # ch 3-5  noisy specular
    "normal.X", "normal.Y", "normal.Z",               # ch 6-8  world normals
    "normal.W",                                        # ch 9    roughness
    "depth.Z",                                         # ch 10   linear depth
    "motion.X", "motion.Y",                            # ch 11-12 motion vectors
    "albedo_d.R", "albedo_d.G", "albedo_d.B",         # ch 13-15 diffuse albedo
    "albedo_s.R", "albedo_s.G", "albedo_s.B",         # ch 16-18 specular albedo
]
_TARGET_DIFFUSE  = ("diffuse.R", "diffuse.G", "diffuse.B")
_TARGET_SPECULAR = ("specular.R", "specular.G", "specular.B")
_DEMOD_EPS = 0.001
```

These are all exported from `deni_train/data/exr_dataset.py` alongside
`_read_exr_channels(path, channel_names) -> dict[str, np.ndarray]`.

---

## Files to Change

| File | Action |
|---|---|
| `training/scripts/preprocess_temporal.py` | Modify: absorb EXR loading + preprocessing |
| `training/scripts/run_training_pipeline.py` | Modify: remove steps 3 and 6, update step 4, renumber |
| `training/scripts/convert_to_safetensors.py` | **Delete** |
| `training/scripts/clean_training_run.py` | Modify: remove `training_data_st/` target |

---

## Phase 1: Modify `preprocess_temporal.py` to Load EXR Directly

This is the primary and most complex change. Output format (cropped `.safetensors`) is
**unchanged** — the training DataLoader is not affected.

### 1.1  Add imports at the top of the file

After the existing imports, add:

```python
import numpy as np

# Reuse EXR reading infrastructure from exr_dataset.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deni_train.data.exr_dataset import (
    _INPUT_CHANNEL_NAMES,
    _TARGET_DIFFUSE,
    _TARGET_SPECULAR,
    _DEMOD_EPS,
    _read_exr_channels,
)
```

`sys` and `os` are already imported. `torch` is already imported. Remove the source-side
`from safetensors.torch import load_file, save_file` import and replace with only
`from safetensors.torch import save_file` (we still write crops as safetensors).

### 1.2  Add `_load_exr_pair()` helper function

Insert this function after the module-level constants (`_MIN_COVERAGE`, `_OVERSAMPLE_FACTOR`,
`_FNAME_RE`) and before `_windows()`:

```python
def _load_exr_pair(
    input_exr: str, target_exr: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load an EXR pair and return (input, target) float16 tensors.

    Applies the same preprocessing as convert_to_safetensors.py:
      - Reads 19 input channels as float32
      - Demodulates diffuse irradiance (ch 0-2) by albedo_d (ch 13-15)
      - Demodulates specular irradiance (ch 3-5) by albedo_s (ch 16-18)
      - Clips to float16 range [-65504.0, 65504.0] and casts to float16
      - Target (6 ch) demodulated using same albedo from input EXR

    Returns:
        input:  float16 (19, H, W)
        target: float16 (6,  H, W)
    """
    input_data = _read_exr_channels(input_exr, _INPUT_CHANNEL_NAMES)
    input_arrays = np.stack(
        [input_data[name] for name in _INPUT_CHANNEL_NAMES], axis=0
    )  # (19, H, W) float32

    albedo_d = input_arrays[13:16]
    albedo_s = input_arrays[16:19]

    input_arrays[0:3] = input_arrays[0:3] / np.maximum(albedo_d, _DEMOD_EPS)
    input_arrays[3:6] = input_arrays[3:6] / np.maximum(albedo_s, _DEMOD_EPS)
    np.clip(input_arrays, -65504.0, 65504.0, out=input_arrays)
    input_tensor = torch.from_numpy(input_arrays).to(torch.float16)

    target_channel_names = list(_TARGET_DIFFUSE) + list(_TARGET_SPECULAR)
    target_data = _read_exr_channels(target_exr, target_channel_names)

    target_d = np.stack([target_data[n] for n in _TARGET_DIFFUSE], axis=0)
    target_s = np.stack([target_data[n] for n in _TARGET_SPECULAR], axis=0)

    target_d = target_d / np.maximum(albedo_d, _DEMOD_EPS)
    target_s = target_s / np.maximum(albedo_s, _DEMOD_EPS)
    np.clip(target_d, -65504.0, 65504.0, out=target_d)
    np.clip(target_s, -65504.0, 65504.0, out=target_s)

    target_arrays = np.concatenate([target_d, target_s], axis=0)  # (6, H, W)
    target_tensor = torch.from_numpy(target_arrays).to(torch.float16)

    return input_tensor, target_tensor
```

### 1.3  Replace `_FNAME_RE`

Old:
```python
_FNAME_RE = re.compile(r"^(.+)_([0-9a-f]{8})_(\d{4})\.safetensors$")
```

New:
```python
_FNAME_RE = re.compile(r"^(.+)_([0-9a-f]{8})_(\d{4})_input\.exr$")
```

Same 3 capture groups: `(scene, path_id, frame)`. Only the suffix changes.

### 1.4  Update `_process_one()` signature and loading

Current signature:
```python
def _process_one(
    input_path: str,   # path to .safetensors
    rel_path: str,
    output_dir: str,
    n_crops: int,
    crop_size: int,
) -> tuple[int, int, int]:
    tensors = load_file(input_path)
    inp = tensors["input"]
    tgt = tensors["target"]
    ...
    stem = rel_path[: -len(".safetensors")] if rel_path.endswith(".safetensors") else rel_path
```

New — replace the loading block and stem derivation:
```python
def _process_one(
    input_exr: str,    # path to *_input.exr
    rel_path: str,     # relative path of *_input.exr (used for stem + RNG seed)
    output_dir: str,
    n_crops: int,
    crop_size: int,
) -> tuple[int, int, int]:
    target_exr = input_exr[: -len("_input.exr")] + "_target.exr"
    inp, tgt = _load_exr_pair(input_exr, target_exr)
    ...
    stem = rel_path[: -len("_input.exr")] if rel_path.endswith("_input.exr") else rel_path
```

Everything else in `_process_one()` (coverage check, RNG, slicing, saving) is identical.

**Note:** The RNG seed is `rel_path` (the relative path of the file). This must remain the
relative path of the `*_input.exr` file (not the target). This produces different crop
positions than the old safetensors-based run, but that is acceptable since this is a clean
break with no backward compatibility requirement.

### 1.5  Update `_process_temporal_window()` frame loading

Current loading loop:
```python
for frame_num in frame_numbers:
    tensors = load_file(frame_paths[frame_num])
    all_inp.append(tensors["input"])
    all_tgt.append(tensors["target"])
```

`frame_paths[frame_num]` is now a `*_input.exr` path. Replace the loading block:
```python
for frame_num in frame_numbers:
    inp_exr = frame_paths[frame_num]
    tgt_exr = inp_exr[: -len("_input.exr")] + "_target.exr"
    inp, tgt = _load_exr_pair(inp_exr, tgt_exr)
    all_inp.append(inp)
    all_tgt.append(tgt)
```

Everything else in `_process_temporal_window()` is identical.

### 1.6  Update `preprocess()` file discovery

Current:
```python
all_files = sorted(
    glob.glob(os.path.join(input_dir, "**", "*.safetensors"), recursive=True)
)
if not all_files:
    print(f"No .safetensors files found in {input_dir}")
    return
```

New:
```python
all_files = sorted(
    glob.glob(os.path.join(input_dir, "**", "*_input.exr"), recursive=True)
)
if not all_files:
    print(f"No *_input.exr files found in {input_dir}")
    return
```

The `futures` loop that calls `_process_one(fpath, rel, ...)` stays the same — `fpath` is now
a `*_input.exr` path, which matches the updated `_process_one` signature.

The progress print `f"Found {len(all_files)} safetensors files"` should also be updated to
`f"Found {len(all_files)} EXR pairs"`.

### 1.7  Update `preprocess_temporal()` file discovery

Current:
```python
all_files = sorted(
    glob.glob(os.path.join(input_dir, "**", "*.safetensors"), recursive=True)
)
if not all_files:
    print(f"No .safetensors files found in {input_dir}")
    return
```

New:
```python
all_files = sorted(
    glob.glob(os.path.join(input_dir, "**", "*_input.exr"), recursive=True)
)
if not all_files:
    print(f"No *_input.exr files found in {input_dir}")
    return
```

The grouping logic that calls `_FNAME_RE.match(fname)` stays the same — `_FNAME_RE` now
matches EXR filenames. The `frame_map[frame_num] = fpath` assignment stores the `*_input.exr`
path, which is what `_process_temporal_window` now expects.

Update the progress print from `f"Found {len(all_files)} safetensors files"` to
`f"Found {len(all_files)} EXR input files"`.

### 1.8  Update `verify()` source loading

`verify()` maps `src_stem → source path` and reloads source data to reconstruct crop
positions for bit-exact comparison.

**Change 1:** Source discovery — replace safetensors glob with EXR glob:

Current:
```python
src_files = sorted(
    glob.glob(os.path.join(input_dir, "**", "*.safetensors"), recursive=True)
)
src_by_stem: dict[str, str] = {}
for fpath in src_files:
    rel = os.path.relpath(fpath, input_dir)
    stem = rel[: -len(".safetensors")] if rel.endswith(".safetensors") else rel
    src_by_stem[stem] = fpath
```

New:
```python
src_files = sorted(
    glob.glob(os.path.join(input_dir, "**", "*_input.exr"), recursive=True)
)
src_by_stem: dict[str, str] = {}
for fpath in src_files:
    rel = os.path.relpath(fpath, input_dir).replace("\\", "/")
    stem = rel[: -len("_input.exr")] if rel.endswith("_input.exr") else rel
    src_by_stem[stem] = fpath
```

**Change 2:** Source data loading when verifying a crop — replace `np_load_file` with
`_load_exr_pair` + numpy conversion:

Current (inside the crop loop):
```python
src_tensors = np_load_file(src_by_stem[src_stem])
src_inp = src_tensors["input"]
src_tgt = src_tensors["target"]
_, h, w = src_inp.shape
```

New:
```python
inp_exr = src_by_stem[src_stem]
tgt_exr = inp_exr[: -len("_input.exr")] + "_target.exr"
inp_tensor, tgt_tensor = _load_exr_pair(inp_exr, tgt_exr)
src_inp = inp_tensor.numpy()  # (19, H, W) float16
src_tgt = tgt_tensor.numpy()  # (6,  H, W) float16
_, h, w = src_inp.shape
```

Remove the now-unused import:
```python
from safetensors.numpy import load_file as np_load_file
```

**Note:** `_load_exr_pair` is accessible inside `verify()` because it is module-level, and
`verify()` is called only via `main()` in the same process (not a worker subprocess).

### 1.9  Update module docstring

Replace references to "safetensors format (from convert_to_safetensors.py)" with EXR:

```
Input EXR format (from generate_training_data.py / monti_datagen):
    *_input.exr:  19 named channels (diffuse, specular, normals, depth, motion, albedo)
    *_target.exr:  6 named channels (diffuse.RGB, specular.RGB — high-SPP reference)
```

Update the Usage examples to use `--input-dir ../training_data/` instead of
`--input-dir ../training_data_st/`.

### Phase 1 Testing and Verification

#### Automated: Shape and dtype check

After implementing, run a quick Python smoke test (from `training/`):

```python
# smoke_test_preprocess.py
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from scripts.preprocess_temporal import _load_exr_pair
import torch, glob

# Pick the first *_input.exr in training_data/
exr_files = glob.glob("training_data/**/*_input.exr", recursive=True)
assert exr_files, "No EXR files found in training_data/"

inp_exr = exr_files[0]
tgt_exr = inp_exr[:-len("_input.exr")] + "_target.exr"
assert os.path.exists(tgt_exr), f"Missing target: {tgt_exr}"

inp, tgt = _load_exr_pair(inp_exr, tgt_exr)
assert inp.dtype == torch.float16, f"Expected float16, got {inp.dtype}"
assert tgt.dtype == torch.float16, f"Expected float16, got {tgt.dtype}"
assert inp.shape[0] == 19, f"Expected 19 input channels, got {inp.shape[0]}"
assert tgt.shape[0] == 6,  f"Expected 6 target channels, got {tgt.shape[0]}"
assert inp.shape[1:] == tgt.shape[1:], "H,W mismatch between input and target"
print(f"PASS: inp {tuple(inp.shape)}, tgt {tuple(tgt.shape)}")
```

#### Numerical: Cross-validate against `convert_to_safetensors.py` output

If any existing `training_data_st/` safetensors files are present from a prior run, verify
that `_load_exr_pair` produces bit-identical results to what was stored:

```python
# cross_validate.py
import torch
from safetensors.torch import load_file
from scripts.preprocess_temporal import _load_exr_pair
import glob, os

st_files = glob.glob("training_data_st/**/*.safetensors", recursive=True)[:10]
for st_path in st_files:
    stem = os.path.relpath(st_path, "training_data_st")[:-len(".safetensors")]
    inp_exr = os.path.join("training_data", stem + "_input.exr")
    tgt_exr = os.path.join("training_data", stem + "_target.exr")
    if not os.path.exists(inp_exr): continue

    old = load_file(st_path)
    new_inp, new_tgt = _load_exr_pair(inp_exr, tgt_exr)

    assert torch.equal(new_inp, old["input"]), f"input mismatch: {stem}"
    assert torch.equal(new_tgt, old["target"]), f"target mismatch: {stem}"
    print(f"PASS: {stem}")
```

If this passes, `_load_exr_pair` is a perfect drop-in for the old safetensors loading.

#### Functional: End-to-end crop extraction (static mode)

```
python scripts/preprocess_temporal.py \
    --input-dir training_data \
    --output-dir /tmp/test_crops_static \
    --window 1 \
    --crops 2 \
    --workers 1
```

Check:
- Output files named `{scene}_{path_id}_{frame}_crop0.safetensors`, `..._crop1.safetensors`
- Load one: `inp` shape `(19, 384, 384)` float16, `tgt` shape `(6, 384, 384)` float16
- Channels 6–8 (normals) have `norm > 0.01` on at least 10% of pixels (coverage check)

#### Functional: End-to-end crop extraction (temporal mode)

```
python scripts/preprocess_temporal.py \
    --input-dir training_data \
    --output-dir /tmp/test_crops_temporal \
    --window 4 \
    --stride 2 \
    --crops 2 \
    --workers 1
```

Check:
- Output files named `{path_id}_{window_start:04d}_crop0.safetensors`
- Load one: `inp` shape `(4, 19, 384, 384)` float16, `tgt` shape `(4, 6, 384, 384)` float16
- All 4 frames share the same spatial crop position (verify by checking normals are
  consistent across the temporal dimension for static geometry pixels)

#### Functional: `--verify` flag (static mode)

```
python scripts/preprocess_temporal.py \
    --input-dir training_data \
    --output-dir /tmp/test_crops_verify \
    --window 1 \
    --crops 2 \
    --workers 1 \
    --verify
```

Should print `Verification: N crops checked, 0 errors`.

#### Manual inspection

Load a crop in a Python REPL and spot-check expected value ranges:

```python
from safetensors.torch import load_file
t = load_file("/tmp/test_crops_temporal/SomeScene/abcd1234_0000_crop0.safetensors")
inp = t["input"]   # (4, 19, 384, 384)
tgt = t["target"]  # (4, 6,  384, 384)

# Demodulated diffuse irradiance — should be >= 0, typically < 10 for lit surfaces
print(inp[:, 0:3].min(), inp[:, 0:3].max())

# World normals — unit vectors, so norm should be ~1.0 for geometry pixels
import torch
norms = inp[:, 6:9].float().norm(dim=1)  # (4, 384, 384)
print("normal norms min/max:", norms.min().item(), norms.max().item())
# normals are 0 for background, ~1 for geometry

# Motion vectors — should be small (sub-pixel to a few pixels in 960x540 space)
print("motion min/max:", inp[:, 11:13].min().item(), inp[:, 11:13].max().item())

# Depth — should be positive
print("depth min/max:", inp[:, 10].min().item(), inp[:, 10].max().item())
```

---

## Phase 2: Update `run_training_pipeline.py` and Delete `convert_to_safetensors.py`

### 2.1  Update `run_training_pipeline.py`

#### Remove `--convert-jobs` argument

Delete:
```python
    # Convert
    parser.add_argument("--convert-jobs", type=int, default=8, help="Parallel safetensors conversion workers")
```

#### Remove `--skip-convert` and `--skip-evaluate` arguments

Delete:
```python
    parser.add_argument("--skip-convert", action="store_true", help="Skip EXR→safetensors conversion")
```
```python
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip evaluation")
```

#### Replace steps 3, 4, 5, 6, 7, 8 with new steps 3, 4, 5, 6

**Remove step 3 (convert) entirely** — delete the block:
```python
    # ── 3. Convert EXR → safetensors ─────────────────────────────────────
    if not args.skip_convert:
        _run(
            [
                sys.executable, r"scripts\convert_to_safetensors.py",
                "--data_dir", "training_data",
                "--output_dir", "training_data_st",
                "--jobs", str(args.convert_jobs),
            ],
            "Step 3/8: Convert EXR pairs to safetensors",
            dry_run=dry,
        )
```

**Update step 4 (crop)** — change `--input-dir` from `training_data_st` to `training_data`
and update the step number in the description string:

```python
    # ── 3. Extract pre-cropped safetensors ───────────────────────────────
    if not args.skip_crop:
        _run(
            [
                sys.executable, r"scripts\preprocess_temporal.py",
                "--input-dir", "training_data",
                "--output-dir", "training_data_cropped_st",
                "--crops", str(args.crops),
                "--crop-size", str(args.crop_size),
                "--workers", str(args.crop_workers),
            ],
            "Step 3/6: Extract pre-cropped safetensors from EXR",
            dry_run=dry,
        )
```

Note: `--verify` is removed from the pipeline invocation. Verification is slow and should be
run manually when needed. It can still be invoked manually via `--verify` on the CLI.

**Remove step 6 (evaluate) entirely** — delete the block:
```python
    # ── 6. Evaluate ──────────────────────────────────────────────────────
    if not args.skip_evaluate:
        _run(
            [
                sys.executable, "-m", "deni_train.evaluate",
                "--checkpoint", r"configs\checkpoints\model_best.pt",
                "--data_dir", "training_data_st",
                "--output_dir", r"results\production",
                "--val-split",
                "--report", r"results\production\report.md",
            ],
            "Step 6/8: Evaluate trained model",
            dry_run=dry,
        )
```

**Renumber remaining steps:**
- Old step 5 (train) → step 4/6
- Old step 7 (export) → step 5/6
- Old step 8 (golden ref) → step 6/6

**Update module docstring:**

Replace:
```
"""End-to-end training pipeline: clean → render → convert → crop → train → evaluate → export.
...
    python scripts/run_training_pipeline.py --skip-clean --skip-render  # resume from convert step
```

With:
```
"""End-to-end training pipeline: clean → render → crop → train → export.
...
    python scripts/run_training_pipeline.py --skip-clean --skip-render  # resume from crop step
```

### 2.2  Delete `convert_to_safetensors.py`

```
del training/scripts/convert_to_safetensors.py
```

### 2.3  Update `clean_training_run.py`

Remove the `training_data_st` entry from the `targets` list. Find and delete this line:

```python
        (training_dir / "training_data_st",         "Safetensors training data",           False),
```

### Phase 2 Testing and Verification

#### Dry-run check

From `training/`:
```
python scripts/run_training_pipeline.py --skip-clean --skip-render --dry-run
```

Expected output: exactly 4 step headers printed (steps 3–6), no mention of
`convert_to_safetensors.py` or `training_data_st`, step 3 shows
`--input-dir training_data`.

#### Smoke-test clean script

```
python scripts/clean_training_run.py --dry-run
```

Verify that `training_data_st` does not appear in the output. Verify that other targets
(`training_data`, `training_data_cropped_st`, `configs/checkpoints`, etc.) are still listed.

#### Full pipeline dry-run

```
python scripts/run_training_pipeline.py --dry-run
```

Verify all 6 step commands are printed correctly and none reference
`convert_to_safetensors` or `training_data_st`.

#### Manual: attempt to import deleted script

```python
import importlib.util
spec = importlib.util.spec_from_file_location(
    "c", "training/scripts/convert_to_safetensors.py"
)
print("File should not exist")
```

Should raise `FileNotFoundError`.

---

## Summary of All Changes

### `preprocess_temporal.py` changes

| Location | What changes |
|---|---|
| Imports | Add `import numpy as np`; add `sys.path.insert`; import 5 symbols from `exr_dataset`; change `from safetensors.torch import load_file, save_file` to `from safetensors.torch import save_file` |
| `_FNAME_RE` | Match `_input.exr` suffix instead of `.safetensors` |
| New function `_load_exr_pair()` | Added after constants, before `_windows()` |
| `_process_one()` | `input_path` param renamed to `input_exr`; derive `target_exr` from it; replace `load_file()` with `_load_exr_pair()`; update stem derivation |
| `_process_temporal_window()` | Replace `load_file()` loop with `_load_exr_pair()` calls |
| `preprocess()` | Glob `*_input.exr` instead of `*.safetensors`; update not-found message |
| `preprocess_temporal()` | Glob `*_input.exr` instead of `*.safetensors`; update not-found message |
| `verify()` | Glob `*_input.exr` for source discovery; update stem computation; replace `np_load_file` + `safetensors.numpy` import with `_load_exr_pair().numpy()` |
| Module docstring | Update input format description and usage examples |

### `run_training_pipeline.py` changes

| Location | What changes |
|---|---|
| Module docstring | Remove "convert" from pipeline description; update resume example |
| `--convert-jobs` arg | Removed |
| `--skip-convert` arg | Removed |
| `--skip-evaluate` arg | Removed |
| Step 3 (convert) block | Removed |
| Step 4 (crop) block | `--input-dir training_data`; renumbered to 3/6; remove `--verify` |
| Step 5 (train) block | Renumbered to 4/6 |
| Step 6 (evaluate) block | Removed |
| Step 7 (export) block | Renumbered to 5/6 |
| Step 8 (golden ref) block | Renumbered to 6/6 |

### `clean_training_run.py` changes

| Location | What changes |
|---|---|
| `targets` list | Remove `training_data_st` entry |

### Deleted files

- `training/scripts/convert_to_safetensors.py`
