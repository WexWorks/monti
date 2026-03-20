== Training

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
    --output training_data_test `
    --max-viewpoints 3 `
    --width 960 --height 540 `
    --spp 4 --ref-frames 64 `
    --jobs 3
```
Viewpoints sharing the same environment/lights combo are batched into a single
monti_datagen invocation. Use `--jobs N` to run up to N invocations in parallel
(default: 3).

5. Verify the results in a web page using `validate_dataset.py`:
```
python scripts\validate_dataset.py `
    --data_dir training_data_test `
    --gallery training_data_test\gallery.html
```

6. Open the HTML file in a browser:
```
Start-Process training_data_test\gallery.html
```