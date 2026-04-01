# Monti

Monti is a Vulkan path tracer with an ML denoiser, interactive viewer, and headless training data generator.

## Prerequisites

- **CMake** 3.22+
- **C++20 compiler** — MSVC 2022 (v17.x), GCC 12+, or Clang 15+
- **Vulkan SDK** — install from <https://vulkan.lunarg.com/sdk/home> and ensure `VULKAN_SDK` is set (provides `vulkan.h` and `glslc`)
- **Git** — required for FetchContent dependency downloads; also for extended scene sparse checkout
- **GPU** — NVIDIA RTX series recommended (ray tracing + ML denoiser). Any Vulkan 1.2 GPU with `VK_KHR_ray_tracing_pipeline` works for the path tracer

All C++ dependencies are fetched automatically by CMake (VMA, GLM, SDL3, Dear ImGui, FreeType, Catch2, CLI11, volk, tinyexr, stb, FLIP, MikkTSpace, nlohmann/json).

## Configure & Build

```bash
cmake -B build
cmake --build build --config Release
```

This builds everything: `monti_view` (interactive viewer), `monti_datagen` (headless data generator), `monti_tests`, and all core libraries. Khronos glTF test assets, extended Cauldron-Media scenes, and benchmark scenes are downloaded automatically during configuration. All C++ dependencies are fetched via CMake FetchContent.

### Minimal build (libraries + tests only)

To skip the applications and heavy scene downloads:

```bash
cmake -B build -DMONTI_BUILD_APPS=OFF -DMONTI_DOWNLOAD_EXTENDED_SCENES=OFF -DMONTI_DOWNLOAD_BENCHMARK_SCENES=OFF
cmake --build build --config Release
```

### Reconfigure after changing CMake options

CMake caches option values. To pick up new defaults after editing `CMakeLists.txt`:

```bash
Remove-Item build/CMakeCache.txt
cmake -B build
cmake --build build --config Release
```

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `MONTI_BUILD_APPS` | `ON` | Build `monti_view` and `monti_datagen` executables |
| `MONTI_DOWNLOAD_TEST_ASSETS` | `ON` | Download Khronos glTF sample models at configure time |
| `MONTI_DOWNLOAD_EXTENDED_SCENES` | `ON` | Download Cauldron-Media extended scenes (multi-GB) via Git sparse checkout |
| `MONTI_DOWNLOAD_BENCHMARK_SCENES` | `ON` | Download heavy benchmark scenes (Bistro, Sponza, San Miguel) |

## Scene Assets

### Khronos test scenes (automatic)

Downloaded automatically when `MONTI_DOWNLOAD_TEST_ASSETS=ON` (the default) into `scenes/khronos/`. Includes 12 GLB models (ABeautifulGame, AntiqueCamera, BoomBox, ClearCoatTest, DamagedHelmet, DragonAttenuation, Lantern, MaterialsVariantsShoe, MosquitoInAmber, SheenChair, ToyCar, WaterBottle) and 2 multi-file glTF models (FlightHelmet, Sponza) fetched via Git sparse checkout.

### Extended scenes (Cauldron-Media)

Downloaded automatically into `scenes/extended/Cauldron-Media/` via Git sparse checkout of [GPUOpen-LibrariesAndSDKs/Cauldron-Media](https://github.com/GPUOpen-LibrariesAndSDKs/Cauldron-Media). Fetches AbandonedWarehouse, BistroInterior, and Brutalism. These are multi-GB downloads. Disable with `-DMONTI_DOWNLOAD_EXTENDED_SCENES=OFF` if bandwidth is limited.

### Environment maps

Environment maps (HDR/EXR) go in `scenes/environments/`. This directory is gitignored. For standalone object scenes that have no embedded lighting, an environment map is required.

Download CC0 HDRIs from [Poly Haven](https://polyhaven.com/hdris) and place them in `scenes/environments/`:

```bash
mkdir scenes\environments
# Example: download studio_small_09_2k.exr from Poly Haven
```

## ML Denoiser Model

The trained model `deni_v1.denimodel` lives in `denoise/models/`. CMake copies it to `build/deni_models/` at build time. The denoiser auto-discovers the model at runtime — no path configuration needed.

If you need to install a model from another source (e.g., a sibling repo where training was performed):

```bash
copy <source-repo>\denoise\models\deni_v1.denimodel denoise\models\deni_v1.denimodel
cmake --build build --config Release --target deni_model
```

In `monti_view`, switch the denoiser mode from **Passthrough** to **ML** in the UI to use the model. The denoiser timing is displayed in the UI panel.

## Running

### monti_view (interactive viewer)

```bash
build\Release\monti_view.exe scenes\khronos\DamagedHelmet.glb
build\Release\monti_view.exe scenes\khronos\DamagedHelmet.glb --env scenes\environments\studio_small_09_2k.exr
```

### monti_datagen (headless data generator)

```bash
build\Release\monti_datagen.exe --scene scenes\khronos\DamagedHelmet.glb --output training_output
```

## Testing

### First-time setup: generate golden references

After a fresh clone and build, golden reference images must be generated before tests pass. These are high-SPP renders stored in `tests/golden/` and used by the golden validation tests:

```bash
build\Release\monti_tests.exe "[golden_gen]"
```

This renders each test scene at 1024 SPP and writes reference PNGs. The generation tests are skipped by default — only the validation tests run automatically. This step only needs to be repeated if the rendering pipeline changes.

### Run all tests (batched)

Tests are split into batches to avoid GPU resource exhaustion:

```bash
cmake --build build --config Release --target run_all_tests
```

### Run individual test batches

```bash
cmake --build build --config Release --target run_tests_core      # Core tests
cmake --build build --config Release --target run_tests_golden    # Golden reference tests
cmake --build build --config Release --target run_tests_extended  # Extended scene tests
cmake --build build --config Release --target run_tests_deni      # Denoiser + numerical tests
```

### Run tests directly with Catch2 filters

```bash
build\Release\monti_tests.exe                          # All tests
build\Release\monti_tests.exe "[deni]"                 # Denoiser tests only
build\Release\monti_tests.exe "~[extended]~[golden]"   # Skip heavy tests
```

### CTest

```bash
cd build
ctest -C Release
```
