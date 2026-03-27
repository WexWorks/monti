== Dataset Generation

All commands below assume your working directory is `training/`.
Scripts `generate_viewpoints.py` and `generate_light_rigs.py` resolve scene
paths relative to their own location, so they work from any directory.
`generate_training_data.py` and the training commands use CWD-relative defaults
(e.g. `../build/`, `../scenes/`, `viewpoints/`) and must be run from `training/`.

1. Manually generate seed viewpoints using `monti_view`:
```
..\build\Release\monti_view.exe ..\scenes\khronos\DamagedHelmet.glb `
    --env .\environments\kloppenheim_06_2k.exr `
    --viewpoint-dir viewpoints\manual
```
Note: monti_view writes `<SceneName>.json` (e.g. `DamagedHelmet.json`) into the
viewpoint-dir automatically, based on the scene filename stem.

2. Generate area light rigs for non-emissive scenes:
```
python scripts\generate_light_rigs.py `
    --scenes-dir ..\scenes\khronos ..\scenes\training ..\scenes\extended\Cauldron-Media `
    --output light_rigs
```

3. Automatically generate viewpoints using `generate_viewpoints.py`:
```
python scripts\generate_viewpoints.py `
    --scenes ..\scenes\khronos ..\scenes\training ..\scenes\extended\Cauldron-Media `
    --output viewpoints `
    --seeds viewpoints\manual `
    --variations-per-seed 4 `
    --envs-dir environments `
    --lights-dir light_rigs
```
Environment maps and light rigs are embedded directly in each viewpoint JSON entry.

4. Render noisy and target images using `generate_training_data.py`:
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
copies. Use `--exposure-steps N` to control the number of wedge offsets (default: 5
produces offsets -2, -1, 0, +1, +2 EV; choices: 3, 5, 7).

If any viewpoints were skipped (near-black or excessive NaN), monti_datagen
writes per-invocation `skipped-<scene>-<N>.json` files to the output directory
(via `--skipped-path`).

4b. *(Optional)* Prune skipped viewpoints from the source JSON files to avoid
re-rendering them on subsequent runs:
```
python scripts\prune_viewpoints.py `
    --skipped training_data\skipped-*.json `
    --viewpoints-dir viewpoints
```
Use `--dry-run` to preview which viewpoints would be removed without modifying
any files.

5. Verify the results in a web page using `validate_dataset.py`:
```
python scripts\validate_dataset.py `
    --data_dir training_data `
    --gallery training_data\gallery.html
```

6. Open the HTML file in a browser:
```
Start-Process training_data\gallery.html
```

6b. *(Optional)* Convert EXR training data to safetensors for faster training:
```
python scripts\convert_to_safetensors.py `
    --data_dir training_data `
    --output_dir training_data_st `
    --verify `
    --jobs 8
```
Converts each EXR input/target pair into a single `.safetensors` file with
pre-processed float16 tensors. The `--verify` flag re-reads each converted file
and compares it to the EXR source. Use `--jobs N` to run up to N workers in
parallel (default: min(cpu_count, 8)). Training auto-detects safetensors data if
present, otherwise falls back to EXR.

6c. Validate the converted safetensors dataset:
```
python scripts\validate_dataset.py `
    --data_dir training_data_st `
    --gallery training_data_st\gallery.html
```

== Training

7. Smoke-test training on a small test dataset:
```
python -m deni_train.train --config configs/smoke_test.yaml
```
Uses the test dataset in `training_data_test/` (generated with `--max-viewpoints 3`
as described above). Trains for 10 epochs with early stopping patience of 5.
Checkpoints saved to `configs/checkpoints/`.

8. Evaluate the smoke-test model:
```
python -m deni_train.evaluate `
    --checkpoint configs/checkpoints/model_best.pt `
    --data_dir ../training_data_test `
    --output_dir results/smoke_test/ `
    --val-split `
    --report results/smoke_test/smoke_test.md
```
Generates per-image and per-scene metrics, comparison PNGs, and a Markdown report.
`--val-split` evaluates only the held-out validation split (last ~10% per scene).

9. Production training on the full dataset:
```
python -m deni_train.train --config configs/default.yaml
```
Trains on `training_data/` with early stopping (patience=30). If safetensors data
exists in `data_dir`, it is used automatically for faster loading. To force a
specific format, set `data_format: "exr"` or `data_format: "safetensors"` in the
config YAML. Monitor progress:
```
tensorboard --logdir configs/runs/
```

10. Evaluate the production model:
```
python -m deni_train.evaluate `
    --checkpoint configs/checkpoints/model_best.pt `
    --data_dir training_data_st `
    --output_dir results/v2_production/ `
    --val-split `
    --report results/v2_production/v2_production.md
```

11. Export production weights and install into the denoiser library:
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

12. Regenerate the golden reference for GPU shader validation:
```
python ../tests/generate_golden_reference.py --output ../tests/data/golden_ref.bin
```
This must be done whenever the model architecture changes (not for weight-only
changes). The golden reference embeds random weights and deterministic input,
then records the PyTorch output. The C++ GPU tests compare shader output against
this reference. If they disagree, the GLSL shaders need updating. See the sync
requirement notes in `tests/generate_golden_reference.py` and
`tests/ml_inference_numerical_test.cpp`.