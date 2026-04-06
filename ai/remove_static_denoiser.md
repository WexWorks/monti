# Remove Legacy Static Frame (V1) ML Denoiser

Remove all V1 (`kV1_SingleFrame`, `DeniUNet`, 19ch→6ch single-frame) code from the Python training
framework, C++ inference engine, GLSL shaders, model files, and tests.

**Backwards compatibility is a non-goal.** The final pipeline is exclusively temporal (V3):

```
generate_training_data.py
    → prepare_temporal.py  (--window 8 --stride 4)
    → deni_train.train_temporal  (configs/temporal.yaml)
    → export_weights.py
    → GPU inference (MlInference / deni_v3.denimodel)
```

---

## Background: V1 vs. V3

| Aspect | V1 (static, to remove) | V3 (temporal, to keep) |
|--------|------------------------|------------------------|
| Model class | `DeniUNet` | `DeniTemporalResidualNet` |
| Enum value | `kV1_SingleFrame` | `kV3_Temporal` |
| Input channels | 19 (G-buffer only) | 26 (19 G-buffer + 7 temporal history) |
| Output channels | 6 (diffuse.RGB + specular.RGB) | 7 (3 delta_d + 3 delta_s + 1 blend) |
| U-Net depth | 3 levels, standard 3×3 conv | 2 levels, depthwise-separable conv |
| Encoder shader | `encoder_input_conv.comp` | `temporal_input_gather.comp` |
| Output shader | `output_conv.comp` | `temporal_output_conv.comp` |
| Conv shaders | `conv.comp` | `depthwise_conv.comp` + `pointwise_conv.comp` |
| Model file | `deni_v1.denimodel` | `deni_v3.denimodel` |
| Golden binary | `tests/data/golden_ref.bin` (version=1) | `tests/data/golden_ref_v3.bin` (version=2) |

Architecture is auto-detected in `MlInference::InferArchitectureFromWeights()`:
- V1 detected by presence of layer name `"down0.conv1.conv.weight"`
- V3 detected by presence of layer name `"down0.conv1.depthwise.weight"`

---

## Phase 1 — Python Training Framework

All steps in this phase are independent and can be done in parallel.

### 1. Delete `training/deni_train/train.py`
Static frame U-Net training. Replaced by `train_temporal.py`.

### 2. Delete `training/deni_train/evaluate.py`
Static frame evaluation. Replaced by `evaluate_temporal.py`.

### 3. Delete V1 model class file(s) in `training/deni_train/models/`
Files present:
- `unet.py` — contains `DeniUNet` (V1 static U-Net). **Delete.**
- `temporal_unet.py` — contains `DeniTemporalResidualNet` (V3). **Keep.**
- `blocks.py` — shared building blocks for both models. **Inspect**: if it contains
  blocks used exclusively by `DeniUNet` and not by `DeniTemporalResidualNet`, delete those
  blocks; otherwise keep the file unchanged.
- `__init__.py` — remove any import of `DeniUNet` or `unet`; keep `DeniTemporalResidualNet` import.

Confirm by grepping: `grep -r "DeniUNet\|from.*unet import" training/deni_train/`

### 4. Delete V1 data loader(s) in `training/deni_train/data/`
Files present:
- `safetensors_dataset.py` — single-frame static safetensors loader. **Delete.**
- `exr_dataset.py` — raw EXR loader (used by static `train.py` with `data_format: "exr"`).
  Inspect: if it's only referenced by the deleted `train.py`, **delete it**.
  If referenced by temporal code, keep.
- `temporal_safetensors_dataset.py` — **Keep** (used by `train_temporal.py`).
- `splits.py` — train/val split utilities. Inspect: if only used by static `train.py`, delete;
  otherwise keep.
- `transforms.py` — image transforms. Inspect: if referenced by temporal code, keep.
- `__init__.py` — remove imports for deleted modules.

### 5. Delete training config files
- `training/configs/default.yaml` — V1 production config (`in_channels: 19, out_channels: 6`,
  `data_dir: "D:/training_data_cropped_st"`, `data_format: "safetensors"`). **Delete.**
- `training/configs/small_test.yaml` — V1 quick-test config (13ch in, non-temporal). **Delete.**
- `training/configs/smoke_test.yaml` — V1 minimal CI config (13→3ch). **Delete.**
- `training/configs/temporal.yaml` — V3 production config. **Keep.**

> Note: `small_test.yaml` and `smoke_test.yaml` are deleted without replacement. If a quick
> smoke-test temporal config is needed for CI, create `temporal_smoke.yaml` as a follow-up.

---

## Phase 2 — Python Pipeline Scripts

### 6. Rename and modify `training/scripts/preprocess_temporal.py` → `prepare_temporal.py`

Rename the file, then remove the `--window 1` (static) code path:

- Remove the `preprocess()` function (single-frame crop extraction, ~lines 430–590).
- In `main()`, remove the `if args.window > 1` branch; make temporal processing unconditional.
- Change `--window` default from `1` to `8`.
- Remove all output directory references to `training_data_cropped_st` — replace with
  `training_data_temporal_st` everywhere.
- Update the module docstring and `--help` text to describe temporal-only behavior.

After renaming: update all callers (`run_training_pipeline.py`, `README.md`) to use `prepare_temporal.py`.

### 7. Rewrite `training/scripts/run_training_pipeline.py` for temporal-only pipeline

Replace the entire static V1 pipeline with the temporal V3 pipeline. The rewritten script must:

**Arguments to change:**
- `--config` default: `"configs/default.yaml"` → `"configs/temporal.yaml"`
- Remove `--skip-export`-guarded golden ref step that generates `golden_ref.bin` (V1);
  replace with a step that generates `golden_ref_v3.bin` if a V3 generator script exists.
- Keep the existing `--evaluate` flag (`action="store_true"`, disabled by default) that
  was added to the current script. The evaluate step calls `deni_train.evaluate_temporal`.

**Step changes:**

| Old step | New step |
|----------|----------|
| Step 3: `preprocess_temporal.py` without `--window` | Step 3: `prepare_temporal.py --window 8 --stride 4 --output-dir training_data_temporal_st` |
| Step 4: `deni_train.train --config configs/default.yaml` | Step 4: `deni_train.train_temporal --config configs/temporal.yaml` |
| (no evaluate step) | Step 5 (opt-in): `deni_train.evaluate_temporal --config configs/temporal.yaml` |
| Step 5: export to `models/deni_v1.denimodel` | Step 6: export to `models/deni_v3.denimodel` |
| Step 6: regenerate `golden_ref.bin` (V1) | Step 7: regenerate `golden_ref_v3.bin` if applicable |

**Remove entirely:**
- References to `training_data_cropped_st`
- References to `deni_v1.denimodel`
- References to `golden_ref.bin` (version=1)

### 8. Delete `training/scripts/generate_training_data.py.bak`
Orphaned backup file. Delete unconditionally.

### 9. Delete `training/scripts/generate_reference_output.py`
Invokes `DeniUNet` (V1) to generate PyTorch reference output for GLSL shader validation.
V3 GLSL validation uses `tests/generate_golden_reference.py` (which generates `golden_ref_v3.bin`).
Confirm by grepping: `grep -n "DeniUNet\|unet" training/scripts/generate_reference_output.py`
If V1-only, **delete**. If it has a V3 path, update to use `DeniTemporalResidualNet` only.

### 10. Review `training/scripts/compare_denoisers.py`
Grep: `grep -n "DeniUNet\|unet\|v1\|deni_v1" training/scripts/compare_denoisers.py`
If it hard-codes V1 model loading, update to use `DeniTemporalResidualNet` and `deni_v3.denimodel`.
If model-agnostic (takes model file as CLI arg), leave unchanged.

### 11. Rewrite `training/README.md`

The README currently has two variants of steps 3–7: "Static Pipeline" and "== Temporal Training".
Remove the static variant entirely. The README should document a single linear pipeline:

1. (Optional) Clean previous run
2. Record viewpoints in monti_view → `viewpoints/<scene>.json`
3. Render EXR pairs — `generate_training_data.py`
4. Prepare temporal windows — `prepare_temporal.py --window 8 --stride 4`
5. Train — `python -m deni_train.train_temporal --config configs/temporal.yaml`
6. (Optional) Evaluate — `python -m deni_train.evaluate_temporal --config configs/temporal.yaml`
7. Export weights — `export_weights.py --output models/deni_v3.denimodel --install`
8. Regenerate V3 golden reference (if applicable)

Remove all references to: `train.py`, `evaluate.py`, `default.yaml`, `small_test.yaml`,
`smoke_test.yaml`, `training_data_cropped_st/`, `training_data_st/`, `deni_v1.denimodel`,
`preprocess_temporal.py` (replaced by `prepare_temporal.py`).

---

## Phase 3 — C++ Denoiser Core

### 12. Modify `denoise/src/vulkan/MlInference.h`

**In the `ModelVersion` enum:**
Remove `kV1_SingleFrame`. The enum becomes:
```cpp
enum class ModelVersion {
    kV3_Temporal,  // 2-level U-Net, depthwise separable, 26ch temporal input
};
```
Or remove the enum entirely if all callers can be simplified to not need it —
follow minimal-changes principle (keep with one value rather than restructuring callers).

**Remove constants:**
```cpp
static constexpr uint32_t kV1InputChannels = 19;
static constexpr uint32_t kV1OutputChannels = 6;
```

**Remove V1-only feature buffer members** (in the private section):
- `buf0_b_`, `buf1_a_`, `buf1_b_`, `buf2_a_`, `buf2_b_` — V1 U-Net level buffers
- `skip0_`, `skip1_` — V1 skip connection buffers
- `concat1_` — V1 concatenation buffer at level 1
- Level 2 buffers (whatever they are named for the 3rd U-Net level)

**Remove V1-only pipeline members:**
- `conv_pipelines_` (map keyed by `{in_ch, out_ch}`)
- `encoder_input_pipeline_`
- `output_conv_pipeline_`
- V1-specific `upsample_concat_pipelines_` entries (if stored separately from V3)

**Remove V1-only method declarations:**
- `CreateEncoderInputConvPipeline()`
- `CreateConvPipeline()` (or the overload for standard conv)
- `DispatchConv()`
- `DispatchEncoderInputConv()`
- `InferV1()`

### 13. Modify `denoise/src/vulkan/MlInference.cpp`

**`InferArchitectureFromWeights()` function:**
Remove the V1 detection branch (`"down0.conv1.conv.weight"` check).
Simplify to unconditionally set `model_version_ = ModelVersion::kV3_Temporal`
(or omit entirely if `ModelVersion` enum is removed).

**`Resize()` function:**
Remove V1 buffer allocation block:
- Level 2 buffer allocations (`buf2_a_`, `buf2_b_`)
- `concat1_` allocation
- Any VMA allocation calls for V1-only buffers

**`CreatePipelines()` function (or equivalent):**
Remove V1 pipeline creation block (roughly lines 936–990):
- All `conv_pipelines_` creation
- `CreateEncoderInputConvPipeline()` call
- 3-level U-Net pipeline setup (level0, level1, level2)

**`Infer()` function:**
Remove the `if (model_version_ == kV1_SingleFrame)` branch.
Call `InferV3Temporal()` unconditionally.

**Delete function implementations:**
- `InferV1()` — entire function body
- `DispatchConv()` — entire function body
- `DispatchEncoderInputConv()` — entire function body

### 14. Modify `denoise/src/vulkan/Denoiser.cpp`

**Model discovery (around lines 96–115):**
Replace the V1 fallback loop:
```cpp
// Old (remove):
for (auto name : {"deni_v3.denimodel", "deni_v1.denimodel"}) {
    std::string auto_path = std::string(DENI_MODEL_DIR) + "/" + name;
    if (std::filesystem::exists(auto_path)) {
        resolved_model_path = std::move(auto_path);
        break;
    }
}
```
With a direct path to V3 only:
```cpp
// New:
resolved_model_path = std::string(DENI_MODEL_DIR) + "/deni_v3.denimodel";
```
(Keep existing error handling for the case where the model file is missing.)

---

## Phase 4 — Shaders & Build

### 15. Delete V1-only GLSL shaders

- `denoise/src/vulkan/shaders/conv.comp` — V1 standard 3×3 convolution.
  V3 uses `depthwise_conv.comp` + `pointwise_conv.comp` instead.
- `denoise/src/vulkan/shaders/encoder_input_conv.comp` — V1 G-buffer encoder input (19→c0 channels).
  V3 uses `temporal_input_gather.comp` (10 images → 26ch flat buffer).
- `denoise/src/vulkan/shaders/output_conv.comp` — V1 output with remodulation.
  V3 uses `temporal_output_conv.comp`.

Shaders to **keep** (V3 or shared):
`passthrough_denoise.comp`, `group_norm_reduce.comp`, `group_norm_apply.comp`,
`downsample.comp`, `upsample_concat.comp`, `reproject.comp`,
`depthwise_conv.comp`, `pointwise_conv.comp`,
`temporal_input_gather.comp`, `temporal_output_conv.comp`

### 16. Delete `denoise/models/deni_v1.denimodel`

The V1 binary weights file. V3 uses `deni_v3.denimodel`.

### 17. Modify `CMakeLists.txt`

**In `DENI_SHADER_SOURCES` list:**
Remove these three entries:
```cmake
conv.comp
encoder_input_conv.comp
output_conv.comp
```

**In the `deni_model` target:**
Remove the V1 model copy rule:
```cmake
# Remove these lines:
set(DENI_MODEL_SOURCE "${CMAKE_SOURCE_DIR}/denoise/models/deni_v1.denimodel")
# ... the add_custom_target / add_custom_command block that copies deni_v1.denimodel
```
Keep the V3 copy rule (`deni_v3.denimodel` → `build/deni_models/deni_v3.denimodel`).

---

## Phase 5 — Tests

### 18. Modify `tests/ml_inference_numerical_test.cpp`

**Remove the V1 `GoldenTestFixture` class entirely:**
This is the fixture for version=1 binary format. It loads `golden_ref.bin`, sets up 7 G-buffer
images, and does NOT include temporal images.

**Remove V1 test cases (no `[v3]` tag):**
- `"MlInference: GPU output matches PyTorch golden reference"` — `[deni][numerical][golden]`
- `"MlInference: output is deterministic across multiple runs"` — `[deni][numerical][determinism]`

**Remove version=1 parsing code:**
The binary format reader has a `switch (ref.version)` or `if (ref.version == 1)` block that
reads 7 G-buffer images. Remove that branch and any associated structs.

**Keep everything with `[v3]` tag:**
- `GoldenTestFixtureV3` class (version=2 format: 7 G-buffer + 3 temporal images)
- `"MlInference V3: GPU output matches PyTorch golden reference (first frame)"` — `[v3]`
- `"MlInference V3: output is deterministic across multiple runs"` — `[v3]`
- `"MlInference V3: GPU matches PyTorch golden ref (base_channels=12)"` — `[v3][12ch]`
- All `[deni][temporal][reproject_*]` tests
- All integration allocation/resize/upload tests

### 19. Delete `tests/data/golden_ref.bin`

V1 binary golden reference (version=1 format). Confirm the path is
`tests/data/golden_ref.bin` and that `tests/data/golden_ref_v3.bin` (version=2) exists before
deleting.

### 20. Inspect and update `tests/generate_golden_reference.py`

Grep: `grep -n "version\|V1\|DeniUNet\|golden_ref\.bin" tests/generate_golden_reference.py`

- If the script only generates version=1 (`golden_ref.bin`), **delete the entire script**.
- If it generates both version=1 and version=2, remove the version=1 code path and retain
  only the version=2 output to `golden_ref_v3.bin`.

---

## File-by-File Summary

| File | Action |
|------|--------|
| `training/deni_train/train.py` | **Delete** |
| `training/deni_train/evaluate.py` | **Delete** |
| `training/deni_train/models/unet.py` | **Delete** |
| `training/deni_train/models/blocks.py` | Inspect: remove V1-only blocks; keep shared blocks |
| `training/deni_train/models/__init__.py` | Remove `DeniUNet` import |
| `training/deni_train/data/safetensors_dataset.py` | **Delete** |
| `training/deni_train/data/exr_dataset.py` | Inspect; delete if V1-only |
| `training/deni_train/data/splits.py` | Inspect; delete if V1-only |
| `training/deni_train/data/__init__.py` | Remove deleted module imports |
| `training/configs/default.yaml` | **Delete** |
| `training/configs/small_test.yaml` | **Delete** |
| `training/configs/smoke_test.yaml` | **Delete** |
| `training/scripts/preprocess_temporal.py` | **Rename** → `prepare_temporal.py`; remove `--window 1` path |
| `training/scripts/run_training_pipeline.py` | **Rewrite** for temporal pipeline (see Phase 2) |
| `training/scripts/generate_reference_output.py` | **Delete** (V1 DeniUNet shader validation) |
| `training/scripts/generate_training_data.py.bak` | **Delete** |
| `training/scripts/compare_denoisers.py` | Inspect; update V1 model references if present |
| `training/README.md` | **Rewrite** to single temporal pipeline |
| `denoise/src/vulkan/MlInference.h` | Remove V1 enum, constants, members, method decls |
| `denoise/src/vulkan/MlInference.cpp` | Delete `InferV1()`, V1 pipeline creation, V1 `Resize()` blocks |
| `denoise/src/vulkan/Denoiser.cpp` | Replace V1 fallback loop with direct V3 path |
| `denoise/src/vulkan/shaders/conv.comp` | **Delete** |
| `denoise/src/vulkan/shaders/encoder_input_conv.comp` | **Delete** |
| `denoise/src/vulkan/shaders/output_conv.comp` | **Delete** |
| `denoise/models/deni_v1.denimodel` | **Delete** |
| `CMakeLists.txt` | Remove V1 shaders from `DENI_SHADER_SOURCES`; remove V1 model copy rule |
| `tests/ml_inference_numerical_test.cpp` | Remove `GoldenTestFixture` (V1) class and test cases |
| `tests/data/golden_ref.bin` | **Delete** |
| `tests/generate_golden_reference.py` | Inspect; remove V1 generation path or delete |

---

## Verification Checklist

Run these checks after all phases are complete:

1. **CMake configure** — no errors about missing shader sources or missing model files:
   ```
   cmake -B build -DMONTI_BUILD_APPS=ON
   ```

2. **Build** — no compiler errors:
   ```
   cmake --build build --target deni_vulkan monti_tests
   ```

3. **No V1 identifiers remain** — this grep must return zero results:
   ```
   grep -rn "kV1_SingleFrame\|InferV1\|deni_v1\|DeniUNet\|conv\.comp\|encoder_input_conv\|output_conv\.comp" denoise/ tests/ training/
   ```
   (Note: `output_conv` appears inside `temporal_output_conv` — use the full pattern `output_conv\.comp` or `\boutput_conv\b`)

4. **V3 golden tests pass**:
   ```
   .\build\Release\monti_tests.exe "[deni][numerical][golden][v3]" --reporter compact
   .\build\Release\monti_tests.exe "[deni][numerical][determinism][v3]" --reporter compact
   ```

5. **Reprojection tests pass**:
   ```
   .\build\Release\monti_tests.exe "[deni][temporal]" --reporter compact
   ```

6. **Dry-run pipeline — no static references**:
   ```
   python training/scripts/run_training_pipeline.py --dry-run
   ```
   Output must not mention: `deni_v1`, `training_data_cropped_st`, `train.py`, `evaluate.py`, `default.yaml`

7. **README renders correctly** — only temporal pipeline steps visible, no dual-path sections.

---

## Key Decisions

- **Backwards compatibility:** Non-goal.
- **`ModelVersion` enum:** Remove `kV1_SingleFrame`. Keep `kV3_Temporal` with a single-value enum
  rather than restructuring all callers (minimal-changes principle).
- **Rename `preprocess_temporal.py` → `prepare_temporal.py`:** The desired final pipeline spec
  names this script `prepare_temporal.py`.
- **`evaluate_temporal` opt-in:** The `--evaluate` flag in `run_training_pipeline.py` uses
  `action="store_true"` (disabled by default). Pass `--evaluate` explicitly to run evaluation.
- **`training_data_st/` directory:** Contains pre-cropped static safetensors at runtime
  (not source-controlled). Not actionable as a code change — document as deprecated in README.
