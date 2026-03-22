# Monti Directory Reorganization Plan

> Goal: Consolidate the top-level directory structure into three primary modules — `app/`, `denoise/`, and `renderer/` — with a shared `scenes/` asset directory and top-level `tests/`.

## Current Top-Level Structure

```
monti/
    app/            # Application executables (monti_view, monti_datagen)
    capture/        # EXR writer + GPU readback (used only by datagen and tests)
    cmake/          # CMake helper scripts
    denoise/        # Deni denoiser library (deni_vulkan)
    renderer/       # Monti path tracer library (monti_vulkan)
    scene/          # Platform-agnostic scene data model (monti_scene)
    shaders/        # Renderer ray-tracing shaders (raygen, closesthit, miss, anyhit)
    tests/          # C++ test suite (Catch2)
    test_output/    # ML denoiser test output (active)
    test_output2/   # Unused, deprecated
    training/       # Python ML denoiser training pipeline
    ai/             # Design docs and plans
```

Scene assets are currently scattered:
- `tests/assets/` — Khronos glTF models (auto-downloaded by CMake)
- `tests/assets/extended/` — Cauldron-Media scenes (opt-in CMake download)
- `training/scenes/` — Training scene collection (manually curated .glb files)

## Target Structure

```
monti/
    app/
        capture/        ← from capture/
        core/
        datagen/
        shaders/
        view/
    denoise/
        include/
        src/
        training/       ← from training/
    renderer/
        include/
        scene/          ← from scene/
        src/
            vulkan/
                shaders/    ← from shaders/ (Vulkan-specific GLSL)
    scenes/             ← unified scene assets
        khronos/        ← from tests/assets/ (auto-downloaded)
        extended/       ← from tests/assets/extended/ (Cauldron-Media, opt-in)
        training/       ← from training/scenes/ (curated .glb files)
    tests/              ← stays top-level
    test_output/        ← stays (active use by ML tests)
    cmake/
    ai/
```

## Dependency Map

```
monti_scene     depends on: glm
monti_vulkan    depends on: monti_scene, Vulkan, VMA
deni_vulkan     depends on: Vulkan, VMA
monti_capture   depends on: tinyexr, Vulkan, VMA

monti_view      depends on: monti_vulkan, deni_vulkan, monti_scene, app/core
monti_datagen   depends on: monti_vulkan, monti_capture, monti_scene, app/core
monti_tests     depends on: all of the above

training/       depends on: monti_datagen (subprocess), .denimodel format (data contract)
```

Key constraints:
- `monti_scene` is used by renderer, app/core, app/view, app/datagen, and tests
- `monti_capture` is used only by app/datagen and tests (NOT monti_view)
- `deni_vulkan` has zero dependencies on monti_scene or renderer
- `training/` has zero code imports from C++ — only invokes monti_datagen as a subprocess

## Phases

### Phase A: Unify Scene Assets → `scenes/` ✅ Complete

**Risk: Low | Value: High (unblocks shared Cauldron-Media scenes)**

Create top-level `scenes/` and consolidate all scene asset locations:

1. `git mv tests/assets scenes/khronos`
2. Move Cauldron-Media download target: `tests/assets/extended/` → `scenes/extended/`
3. `git mv training/scenes scenes/training`
4. Update CMakeLists.txt:
   - `MONTI_TEST_ASSETS_DIR` → `"${CMAKE_SOURCE_DIR}/scenes/khronos"`
   - `MONTI_EXTENDED_SCENES_DIR` → `"${CMAKE_SOURCE_DIR}/scenes/extended/Cauldron-Media"`
   - Khronos download target → `scenes/khronos/`
   - Cauldron-Media sparse checkout target → `scenes/extended/`
5. Update training Python scripts: default `--scenes` path from `scenes/` to `../scenes/training/` (or make relative to project root)
6. Update training viewpoints paths if they reference `scenes/`
7. Update `.gitignore` entries for new paths

Files to modify:
- `CMakeLists.txt` (download paths, define paths)
- `cmake/FetchDependencies.cmake` (extended scenes checkout target)
- `training/scripts/generate_training_data.py` (default scenes path)
- `training/scripts/generate_viewpoints.py` (if it has default paths)
- `.gitignore`

### Phase B: Move `shaders/` → `renderer/src/vulkan/shaders/` ✅ Complete

**Risk: Medium | Touches: CMake shader compilation paths**

Placed under `renderer/src/vulkan/` to match the denoiser pattern (`denoise/src/vulkan/shaders/`)
and to support future platform backends (e.g., `renderer/src/webgpu/shaders/` for WGSL).

1. `git mv shaders/* renderer/src/vulkan/shaders/`
2. Update CMakeLists.txt:
   - `MONTI_SHADER_DIR` → `"${CMAKE_SOURCE_DIR}/renderer/src/vulkan/shaders"`
3. Verify all `-I ${SHADER_DIR}` include paths in glslc invocations still resolve

Files to modify:
- `CMakeLists.txt` (MONTI_SHADER_DIR)

### Phase C: Move `scene/` → `renderer/scene/`

**Risk: Medium | Touches: include paths, CMake target**

1. `git mv scene renderer/scene`
2. Update CMakeLists.txt:
   - `monti_scene` source paths: `scene/src/` → `renderer/scene/src/`
   - `monti_scene` public include: `scene/include` → `renderer/scene/include`
   - `monti_scene` private include: `scene/src` → `renderer/scene/src`
3. Update all `target_include_directories` that reference `scene/src` (used by monti_view, monti_datagen, monti_tests)
4. No C++ `#include` changes needed — headers use `monti/scene/...` which resolves via include dirs

Files to modify:
- `CMakeLists.txt` (monti_scene paths, target_include_directories for view/datagen/tests)

### Phase D: Move `capture/` → `app/capture/`

**Risk: Low | Few references**

1. `git mv capture app/capture`
2. Update CMakeLists.txt:
   - `monti_capture` source paths: `capture/src/` → `app/capture/src/`
   - `monti_capture` public include: `capture/include` → `app/capture/include`
   - `monti_capture` private include: `capture/src` → `app/capture/src`
   - `CAPTURE_SHADER_DIR` → `"${CMAKE_SOURCE_DIR}/app/capture/shaders"`
3. No C++ `#include` changes needed — headers use `monti/capture/...` which resolves via include dirs

Files to modify:
- `CMakeLists.txt` (monti_capture paths, CAPTURE_SHADER_DIR)

### Phase E: Delete `test_output2/`

**Risk: None**

1. Verify zero references (confirmed: none in codebase)
2. `rm -rf test_output2/`

### Phase F: Move `training/` → `denoise/training/`

**Risk: Higher | Touches: Python paths, venv, CMake model path**

1. `git mv training denoise/training`
2. Update CMakeLists.txt:
   - `DENI_MODEL_SOURCE` → `"${CMAKE_SOURCE_DIR}/denoise/training/models/deni_v1.denimodel"`
3. Update Python infrastructure:
   - `pyproject.toml` paths (if any are absolute)
   - `requirements.txt` location
   - Script shebang/relative paths in `scripts/`
   - Default `--scenes` path adjustments (now `../../scenes/training/`)
   - `.venv` will need recreation at new location
4. Update `run_coverage.py` if it references training paths
5. Update `.gitignore` entries
6. Update VS Code task definitions (`.vscode/tasks.json`) if they reference `training/`
7. Update any `ai/*.md` docs that reference `training/` paths

Files to modify:
- `CMakeLists.txt` (DENI_MODEL_SOURCE)
- `denoise/training/scripts/*.py` (relative path defaults)
- `denoise/training/pyproject.toml`
- `.gitignore`
- `.vscode/tasks.json`
- `run_coverage.py`
- Various `ai/*.md` docs (path references)

## Execution Strategy

- Execute phases A→F in order, one commit per phase
- Use `git mv` to preserve file history
- Full CMake reconfigure + rebuild after each phase
- Run `monti_tests` after each phase to catch breakage
- Run training Python tests after phases A and F

## Risk Notes

- **CMake cache invalidation:** Every phase requires `cmake --fresh` or deleting `build/`
- **Python venv:** Phase F will require recreating `.venv` at the new location
- **CI/CD:** If any CI scripts reference old paths, they'll need updating
- **Absolute paths in test code:** Some tests use relative paths like `"test_output/ml_weights"` — these are relative to CWD (build dir) and won't be affected by the moves
- **`.gitignore`:** Must be audited after each phase for stale entries and new coverage
