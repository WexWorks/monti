== Python Configuration

On Windows, in the training dir, start the project venv:
```
& c:\Users\wex\src\WexWorks\temporal\monti\.venv\Scripts\Activate.ps1
```

== Dataset Generation

All commands below assume your working directory is `training/`.
`generate_training_data.py` and the training commands use CWD-relative defaults
(e.g. `../build/`, `../scenes/`, `viewpoints/`) and must be run from `training/`.

0. *(Optional)* Clean artifacts from a previous training run before starting fresh:
```
python scripts\clean_training_run.py
```
Permanently deletes all generated artifacts and frees disk space immediately —
deletions bypass the Recycle Bin.

The following are removed:
- `training_data/` — rendered EXR pairs, skipped JSONs, gallery HTML
- `training_data_test/` — test datasets
- `training_data_temporal_st/` — pre-cropped temporal safetensors
- `configs/checkpoints/` — model checkpoint `.pt` files
- `configs/runs/` — TensorBoard event logs
- `models/` — exported `.denimodel` files
- `results/` — evaluation output directories

Optional arguments:
- `--dry-run` — preview what would be deleted without removing anything
- `--yes` / `-y` — skip the confirmation prompt
- `--light-rigs` — also remove the auto-generated `light_rigs/` directory

1. Record camera path viewpoints using `monti_view`:
```
..\ build\Release\monti_view.exe ..\scenes\khronos\DamagedHelmet.glb `
    --env .\environments\kloppenheim_06_2k.exr `
    --viewpoint-dir viewpoints
```
Press `P` to toggle path tracking mode. Fly the camera to record viewpoints;
paths auto-save after 500ms of idle. Press `Backspace` to undo the last path.
monti_view writes `<SceneName>.json` (e.g. `DamagedHelmet.json`) into the
viewpoint-dir automatically, based on the scene filename stem.

2. Render noisy and target images using `generate_training_data.py`:
```
python scripts\generate_training_data.py `
    --monti-datagen ..\build\Release\monti_datagen.exe `
    --scenes ..\scenes\khronos ..\scenes\training ..\scenes\extended\Cauldron-Media `
    --viewpoints-dir viewpoints `
    --output D:\training_data `
    --width 1920 --height 1080 `
    --spp 4 --ref-frames 384 --ref-spp 32 `
    --jobs 1
```
Viewpoints sharing the same environment/lights combo are batched into a single
monti_datagen invocation. Use `--jobs N` to run up to N invocations in parallel
(default: 3). For a quick test run, add `--max-viewpoints 3` to limit viewpoints
per scene and use `--output training_data_test` with lower `--ref-frames` (e.g. 64).

monti_datagen automatically normalizes each viewpoint to mid-gray (0.18) and
rejects near-black or NaN-corrupted frames (no EXR written for those viewpoints).

If any viewpoints were skipped (near-black or excessive NaN), monti_datagen
writes per-invocation `skipped-<scene>-<N>.json` files to the output directory
(via `--skipped-path`).

2b. *(Optional)* Prune skipped viewpoints from the source JSON files to avoid
re-rendering them on subsequent runs:
```
python scripts\prune_viewpoints.py `
    --skipped D:\training_data\skipped-*.json `
    --viewpoints-dir viewpoints
```
Use `--dry-run` to preview which viewpoints would be removed without modifying
any files.

3. Extract pre-cropped temporal safetensors for training:
```
python scripts\prepare_temporal.py `
    --input-dir D:\training_data `
    --output-dir D:\training_data_temporal_st `
    --window 16 --stride 8 --crops 4 --crop-size 384 `
    --workers 8
```
Groups frames by camera path, builds sliding windows of consecutive frames, and
ensures all frames in each window share the same crop position. Each output file
contains a 16-frame sequence: input `(16, 19, 384, 384)` and target
`(16, 6, 384, 384)`. The `--stride 8` produces overlapping windows for more
training samples (stride < window). Crops with less than 30% geometry coverage
are rejected via 3× oversampling.

== Training

4. Train the temporal residual denoiser:
```
python -m deni_train.train_temporal --config configs/temporal.yaml
```
Trains on pre-cropped temporal safetensors in `training_data_temporal_st/` with
autoregressive frame processing and temporal stability loss. The temporal model
is a 2-level U-Net with depthwise separable convolutions (~3.4K parameters).
Monitor progress:
```
tensorboard --logdir configs/runs/
```

4b. Resume an interrupted training run from the last periodic checkpoint:
```
python -m deni_train.train_temporal --config configs/temporal.yaml `
    --resume configs/checkpoints/checkpoint_epoch199.pt
```
Restores the full training state: model weights, optimizer, LR scheduler, and
gradient scaler. Training continues from the next epoch with the same LR schedule
and patience counter as when the checkpoint was saved. Use the most recent
`checkpoint_epochNNN.pt` (saved every `checkpoint_interval` epochs) rather than
`model_best.pt`, since `model_best.pt` may be from an earlier epoch.

4c. Warm restart: continue training from the best weights with a fresh LR schedule:
```
python -m deni_train.train_temporal --config configs/temporal.yaml `
    --resume configs/checkpoints/model_best.pt `
    --weights-only
```
Loads only the model weights from the checkpoint. Optimizer state, LR scheduler,
and patience counter are all reset to their initial values per the config. This is
useful after a full cosine-annealing run (where LR decayed to zero) — the model
starts from its best weights and receives a fresh round of learning with the full
LR schedule.

5. Evaluate the temporal model:
```
python -m deni_train.evaluate_temporal `
    --checkpoint configs/checkpoints/model_best.pt `
    --data_dir D:/training_data_temporal_st `
    --output_dir results/temporal/ `
    --val-split `
    --report results/temporal/report.md
```

For a quick visual check during training, use `--max-per-scene N` to evaluate
only N sequences per scene:
```
python -m deni_train.evaluate_temporal `
    --checkpoint configs/checkpoints/model_best.pt `
    --data_dir D:/training_data_temporal_st `
    --output_dir results/temporal_quick/ `
    --val-split `
    --max-per-scene 3
```

Each output comparison PNG shows **Noisy | Denoised | Ground Truth** (ACES tonemapped)
for a single frame. Frames within a sequence are named `<seq>_f00_comparison.png` through
`<seq>_f15_comparison.png`, so you can visually inspect temporal consistency across frames.
The model runs autoregressively — frame 0 uses zero history, frames 1–15 reproject the
previous denoised output.

6. Export production weights and install into the denoiser library:
```
python scripts/export_weights.py `
    --checkpoint configs/checkpoints/model_best.pt `
    --output models/deni_v3.denimodel `
    --install
```
The `--install` flag copies the exported model to `denoise/models/deni_v3.denimodel`,
which is where CMake picks it up for both `deni_vulkan` and `monti_tests`. Without
`--install`, the model is only written to `training/models/` and must be copied
manually. The next CMake build will automatically copy the installed model into
the build directory.

7. Regenerate the golden reference for GPU shader validation:
```
python ../tests/generate_golden_reference.py --output ../tests/data/golden_ref_v3.bin
```
This must be done whenever the model architecture changes (not for weight-only
changes). The golden reference embeds random weights and deterministic input,
then records the PyTorch output. The C++ GPU tests compare shader output against
this reference. If they disagree, the GLSL shaders need updating. See the sync
requirement notes in `tests/generate_golden_reference.py` and
`tests/ml_inference_numerical_test.cpp`.

== Full Pipeline Script

`run_training_pipeline.py` automates the entire sequence (steps 0–7) in a single
invocation. It assumes viewpoint JSONs already exist in `viewpoints/` from
monti_view path recording (step 1).

Full pipeline from scratch:
```
python scripts\run_training_pipeline.py
```

Preview what would run without executing anything:
```
python scripts\run_training_pipeline.py --dry-run
```

Resume from crop extraction (skip clean and render):
```
python scripts\run_training_pipeline.py --skip-clean --skip-render
```

Retrain only (crops already extracted):
```
python scripts\run_training_pipeline.py --skip-clean --skip-render --skip-crop
```

Use `--help` for the full set of options (resolution, spp, ref-frames, crop
count, worker counts, config path, etc.).
