== Python Configuration

On Windows, in the training dir, start the project venv:
```
& c:\Users\wex\src\WexWorks\monti\training\.venv\Scripts\Activate.ps1
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
- `training_data_test/`, `training_data_st/` — test and safetensors datasets
- `training_data_cropped_st/` — pre-cropped safetensors (extracted crops)
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
    --spp 4 --ref-frames 256 --ref-spp 16 `
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

3. Convert EXR training data to safetensors:
```
python scripts\convert_to_safetensors.py `
    --data_dir D:\training_data `
    --output_dir D:\training_data_st `
    --jobs 14
```
Converts each EXR input/target pair into a single `.safetensors` file with
pre-processed float16 tensors. Use `--verify` to check each converted file
against the in-memory tensors. Use `--delete-exr` to verify and then delete
the source EXR pair — avoiding the need for 2x disk space. Use `--jobs N` to
run up to N workers in parallel (default: min(cpu_count, 8)).

4. Extract pre-cropped 384×384 safetensors for training:
```
python scripts\preprocess_temporal.py `
    --input-dir D:\training_data_st `
    --output-dir D:\training_data_cropped_st `
    --crops 8 --crop-size 384 `
    --workers 14
```
Extracts 8 random 384×384 crops per source image into individual safetensors
files. Crops with less than 10% geometry coverage (hit mask) are rejected and
replaced via 3× oversampling. Crop positions are deterministic (seeded by
filename) so re-running produces identical output. The `--verify` flag adds a
post-extraction pass that reloads every crop and checks bit-exact equality
against the source region.

Training reads from `training_data_cropped_st/` by default (see
`configs/default.yaml`). Pre-cropping eliminates the per-epoch disk I/O
bottleneck of loading and cropping full-resolution images during training.

== Training

5. Production training on the full dataset:
```
python -m deni_train.train --config configs/default.yaml
```
Trains on pre-cropped safetensors in `training_data_cropped_st/` with early
stopping (patience=30). The default config expects pre-cropped 384×384 files
from step 4. To train on full-resolution data instead, change `data_dir` in
the config to `"../training_data_st"` (safetensors) or `"../training_data"`
(EXR). Monitor progress:
```
tensorboard --logdir configs/runs/
```

5b. Resume an interrupted training run from the last periodic checkpoint:
```
python -m deni_train.train --config configs/default.yaml `
    --resume configs/checkpoints/checkpoint_epoch199.pt
```
Restores the full training state: model weights, optimizer, LR scheduler, and
gradient scaler. Training continues from the next epoch with the same LR schedule
and patience counter as when the checkpoint was saved. Use the most recent
`checkpoint_epochNNN.pt` (saved every `checkpoint_interval` epochs) rather than
`model_best.pt`, since `model_best.pt` may be from an earlier epoch.

5c. Warm restart: continue training from the best weights with a fresh LR schedule:
```
python -m deni_train.train --config configs/default.yaml `
    --resume configs/checkpoints/model_best.pt `
    --weights-only
```
Loads only the model weights from the checkpoint. Optimizer state, LR scheduler,
and patience counter are all reset to their initial values per the config. This is
useful after a full cosine-annealing run (where LR decayed to zero) — the model
starts from its best weights and receives a fresh round of learning with the full
LR schedule. Typically used with a lower `learning_rate` in the config (e.g.
`1.0e-5` instead of `1.0e-4`) to fine-tune without overshooting.

6. Evaluate the production model:
```
python -m deni_train.evaluate `
    --checkpoint configs/checkpoints/model_best.pt `
    --data_dir D:\training_data_st `
    --output_dir results/v2_production/ `
    --val-split `
    --report results/v2_production/v2_production.md
```

For a quick visual check during training, use `--max-per-scene N` to evaluate
only N evenly-spaced samples from each scene:
```
python -m deni_train.evaluate `
    --checkpoint configs/checkpoints/model_best.pt `
    --data_dir D:\training_data_st `
    --output_dir results/quick/ `
    --val-split `
    --max-per-scene 3
```

7. Export production weights and install into the denoiser library:
```
python scripts/export_weights.py `
    --checkpoint configs/checkpoints/model_best.pt `
    --output models/deni_v1.denimodel `
    --install
```
The `--install` flag copies the exported model to `denoise/models/deni_v1.denimodel`,
which is where CMake picks it up for both `deni_vulkan` and `monti_tests`. Without
`--install`, the model is only written to `training/models/` and must be copied
manually. The next CMake build will automatically copy the installed model into
the build directory.

8. Regenerate the golden reference for GPU shader validation:
```
python ../tests/generate_golden_reference.py --output ../tests/data/golden_ref.bin
```
For the V3 temporal model, also generate the temporal golden reference:
```
python ../tests/generate_golden_reference.py --v3-only
```
This must be done whenever the model architecture changes (not for weight-only
changes). The golden reference embeds random weights and deterministic input,
then records the PyTorch output. The C++ GPU tests compare shader output against
this reference. If they disagree, the GLSL shaders need updating. See the sync
requirement notes in `tests/generate_golden_reference.py` and
`tests/ml_inference_numerical_test.cpp`.

== Full Pipeline Script

`run_training_pipeline.py` automates the entire sequence (steps 0–8) in a single
invocation. It assumes viewpoint JSONs already exist in `viewpoints/` from
monti_view path recording (step 1).

== Temporal Training

The temporal denoiser uses the same data generation pipeline (steps 1–3) as the
static model. The only difference is step 4: `preprocess_temporal.py` is run with
`--window 8` to produce 8-frame sequence crops instead of single-frame crops. This
groups frames by camera path, builds sliding windows of consecutive frames, and
ensures all frames in each window share the same crop position.

4t. Extract pre-cropped temporal safetensors for training:
```
python scripts\preprocess_temporal.py `
    --input-dir D:\training_data_st `
    --output-dir D:\training_data_temporal_st `
    --window 8 --stride 4 --crops 4 --crop-size 384 `
    --workers 14
```
Each output file contains an 8-frame sequence: input `(8, 19, 384, 384)` and
target `(8, 6, 384, 384)`. The `--stride 4` produces overlapping windows for
more training samples (stride < window).

5t. Train the temporal residual denoiser:
```
python -m deni_train.train_temporal --config configs/temporal.yaml
```
This uses `train_temporal.py` (not `train.py`) with autoregressive frame
processing and temporal stability loss. The temporal model is a smaller 2-level
U-Net with depthwise separable blocks (~3.4K parameters vs ~120K for the static
model). Monitor progress:
```
tensorboard --logdir configs/runs/
```

5t-b. Resume or warm-restart temporal training:
```
python -m deni_train.train_temporal --config configs/temporal.yaml `
    --resume configs/checkpoints/checkpoint_epoch199.pt

python -m deni_train.train_temporal --config configs/temporal.yaml `
    --resume configs/checkpoints/model_best.pt `
    --weights-only
```

6t. Evaluate the temporal model:
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
`<seq>_f07_comparison.png`, so you can visually inspect temporal consistency across frames.
The model runs autoregressively — frame 0 uses zero history, frames 1–7 reproject the
previous denoised output.

The static pipeline (steps 4–8 with `default.yaml`) remains fully functional for
training the single-frame v1 model.

Full pipeline from scratch:
```
python scripts\run_training_pipeline.py
```

Preview what would run without executing anything:
```
python scripts\run_training_pipeline.py --dry-run
```

Resume from safetensors conversion (skip clean and render):
```
python scripts\run_training_pipeline.py --skip-clean --skip-render
```

Resume from crop extraction (data already converted):
```
python scripts\run_training_pipeline.py --skip-clean --skip-render --skip-convert
```

Retrain only (crops already extracted):
```
python scripts\run_training_pipeline.py --skip-clean --skip-render --skip-convert --skip-crop
```

Use `--help` for the full set of options (resolution, spp, ref-frames, crop
count, worker counts, config path, etc.).