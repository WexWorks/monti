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
deletions bypass the Recycle Bin. Auto-generated viewpoints (`viewpoints/*.json`)
are removed while hand-crafted seeds in `viewpoints\manual\` are preserved.

The following are removed:
- `training_data/` — rendered EXR pairs, skipped JSONs, gallery HTML
- `training_data_test/`, `training_data_st/` — test and safetensors datasets
- `training_data_cropped_st/` — pre-cropped safetensors (extracted crops)
- `configs/checkpoints/` — model checkpoint `.pt` files
- `configs/runs/` — TensorBoard event logs
- `models/` — exported `.denimodel` files
- `results/` — evaluation output directories
- `viewpoints/*.json` — auto-generated viewpoints (manual seeds preserved)

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
    --output training_data `
    --width 960 --height 540 `
    --spp 4 --ref-frames 256 `
    --jobs 8
```
Viewpoints sharing the same environment/lights combo are batched into a single
monti_datagen invocation. Use `--jobs N` to run up to N invocations in parallel
(default: 3). For a quick test run, add `--max-viewpoints 3` to limit viewpoints
per scene and use `--output training_data_test` with lower `--ref-frames` (e.g. 64).

monti_datagen automatically normalizes each viewpoint to mid-gray (0.18) and
rejects near-black or NaN-corrupted frames (no EXR written for those viewpoints).
After rendering, an exposure wedge amplifies each EXR pair into multiple EV-shifted
copies. Use `--exposure-steps N` to control the number of wedge offsets (default: 5).

**Odd N** (e.g. 3, 5, 7) uses a full symmetric wedge: the offsets are exactly
`-N//2, …, 0, …, +N//2`. For example, `--exposure-steps 5` produces `-2, -1, 0, +1, +2`.

**Even N** (e.g. 1, 2, 4) always includes EV = 0 and randomly samples N − 1 additional
offsets from a balanced pool of N + 1 candidates. For example, `--exposure-steps 4`
draws 4 offsets from the pool `[-2, -1, 0, +1, +2]`, always including 0; the
remaining 3 are chosen per-viewpoint using a deterministic seed so results are
reproducible. `--exposure-steps 2` samples 1 additional offset from `[-1, 0, +1]`.
Any positive integer is accepted.

If any viewpoints were skipped (near-black or excessive NaN), monti_datagen
writes per-invocation `skipped-<scene>-<N>.json` files to the output directory
(via `--skipped-path`).

2b. *(Optional)* Prune skipped viewpoints from the source JSON files to avoid
re-rendering them on subsequent runs:
```
python scripts\prune_viewpoints.py `
    --skipped training_data\skipped-*.json `
    --viewpoints-dir viewpoints
```
Use `--dry-run` to preview which viewpoints would be removed without modifying
any files.

3. Convert EXR training data to safetensors:
```
python scripts\convert_to_safetensors.py `
    --data_dir training_data `
    --output_dir training_data_st `
    --jobs 8
```
Converts each EXR input/target pair into a single `.safetensors` file with
pre-processed float16 tensors. Use `--verify` to check each converted file
against the in-memory tensors. Use `--delete-exr` to verify and then delete
the source EXR pair — avoiding the need for 2x disk space. Use `--jobs N` to
run up to N workers in parallel (default: min(cpu_count, 8)).

4. Extract pre-cropped 384×384 safetensors for training:
```
python scripts\preprocess_temporal.py `
    --input-dir training_data_st `
    --output-dir training_data_cropped_st `
    --crops 8 --crop-size 384 `
    --workers 4 --verify
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
    --data_dir training_data_st `
    --output_dir results/v2_production/ `
    --val-split `
    --report results/v2_production/v2_production.md
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