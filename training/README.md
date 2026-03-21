== Dataset Generation

1. Manually generate seed viewpoints using `monti_view`:
```
..\build\Release\monti_view.exe .\scenes\DamagedHelmet.glb `
    --env .\environments\kloppenheim_06_2k.exr `
    --viewpoint-dir viewpoints\manual
```
Note: monti_view writes `<SceneName>.json` (e.g. `DamagedHelmet.json`) into the
viewpoint-dir automatically, based on the scene filename stem.

2. Generate area light rigs for non-emissive scenes:
```
python scripts\generate_light_rigs.py `
    --scenes-dir scenes `
    --output light_rigs
```

3. Automatically generate viewpoints (with exposure amplification) using `generate_viewpoints.py`:
```
python scripts\generate_viewpoints.py `
    --scenes scenes `
    --output viewpoints `
    --seeds viewpoints\manual `
    --variations-per-seed 4 `
    --envs-dir environments `
    --lights-dir light_rigs
```
Environment maps, light rigs, and exposure levels (default: 0, -1, +1, -2, +2 EV)
are embedded directly in each viewpoint JSON entry. Use `--no-exposures` to disable
exposure amplification, or `--exposures 0 -1 1` to customise the EV levels.

4. Render noisy and target images using `generate_training_data.py`:
```
python scripts\generate_training_data.py `
    --monti-datagen ..\build\Release\monti_datagen.exe `
    --scenes scenes `
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

5. Remove near-black and corrupted viewpoints using `remove_invalid_viewpoints.py`:
```
python scripts\remove_invalid_viewpoints.py `
    --training-data training_data `
    --viewpoints-dir viewpoints
```
Invalid EXR pairs are moved to `invalid_training_data/` and the corresponding
viewpoint entries are removed from the scene JSON files. Use `--dry-run` to preview
without modifying anything.

6. Verify the results in a web page using `validate_dataset.py`:
```
python scripts\validate_dataset.py `
    --data_dir training_data `
    --gallery training_data\gallery.html
```

7. Open the HTML file in a browser:
```
Start-Process training_data\gallery.html
```

7b. *(Optional)* Convert EXR training data to safetensors for faster training:
```
python scripts\convert_to_safetensors.py `
    --data_dir training_data `
    --output_dir training_data_st `
    --verify
```
Converts each EXR input/target pair into a single `.safetensors` file with
pre-processed float16 tensors. The `--verify` flag re-reads each converted file
and compares it to the EXR source. Training auto-detects safetensors data if
present, otherwise falls back to EXR.

7c. Validate the converted safetensors dataset:
```
python scripts\validate_dataset.py `
    --data_dir training_data_st `
    --gallery training_data_st\gallery.html
```

== Training

8. Smoke-test training on a small test dataset:
```
python -m deni_train.train --config configs/smoke_test.yaml
```
Uses the test dataset in `training_data_test/` (generated with `--max-viewpoints 3`
as described above). Trains for 10 epochs with early stopping patience of 5.
Checkpoints saved to `configs/checkpoints/`.

9. Evaluate the smoke-test model:
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

10. Production training on the full dataset:
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

11. Evaluate the production model:
```
python -m deni_train.evaluate `
    --checkpoint configs/checkpoints/model_best.pt `
    --data_dir ../training_data `
    --output_dir results/v2_production/ `
    --val-split `
    --report results/v2_production/v2_production.md
```

12. Export production weights:
```
python scripts/export_weights.py `
    --checkpoint configs/checkpoints/model_best.pt `
    --output models/deni_v1.denimodel
```