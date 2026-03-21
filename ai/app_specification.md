# Monti Application Specification

## 1. Overview

The Monti project ships two separate executables built on the Monti renderer and Deni denoiser libraries:

| Executable | Purpose | Window | Swapchain | Key Dependencies |
|---|---|---|---|---|
| **`monti_view`** | Interactive glTF viewer — load scenes, navigate with a camera, view path-traced + denoised output | SDL3 window + ImGui | Yes | SDL3, ImGui, FreeType |
| **`monti_datagen`** | Headless training-data generator — render multi-channel EXR training sets from glTF scenes using the GPU, exit | None | No (offscreen render only) | tinyexr, nlohmann/json |

Both apps share the same core library stack (`monti_scene`, `monti_vulkan`, `deni_vulkan`) and Vulkan initialization code (`core/`). They differ in what they layer on top: `monti_view` adds windowing, presentation, and UI; `monti_datagen` adds `monti_capture`, batch automation, and EXR writing with no display dependencies.

**Why two executables?** `monti_datagen` runs on headless GPU servers (CI farms, cloud training pipelines) where SDL3 and ImGui are dead weight and a display server may not exist. A dedicated binary that links only the libraries, VMA, nlohmann/json, and tinyexr is leaner and more deployable. Conversely, `monti_view` carries interactive-mode dependencies that a batch pipeline never needs.

Both apps reuse proven patterns from the `rtx-chessboard` project: volk Vulkan loading, VMA memory allocation, and the same frame-in-flight architecture (for `monti_view`). All chess/physics/network-specific code is stripped. volk is an **app-level dependency only** — the libraries (`deni_vulkan`, `monti_vulkan`) are loader-agnostic and resolve Vulkan functions via `PFN_vkGetDeviceProcAddr` passed by the host (see design spec §4.4). This means Deni can be integrated into any Vulkan application regardless of which Vulkan loader it uses.

---

## 2. Roadmap Context

These apps support the following development arc:

1. **Implement Monti + Deni** — `monti_view` is the primary integration vehicle for both libraries. ✅ Complete (Phases 1–10B, 9A–9D, 11A–11B).
2. **Integrate DLSS-RR in monti_view** — app-level NVIDIA denoiser for interactive viewing and quality reference during development (leveraging rtx-chessboard integration). Pending (F1).
3. **Generate training data** — `monti_datagen` produces multi-channel EXR training sets from 14 scenes with viewpoints, lighting rigs, and HDRIs. ✅ Complete (F9-4, F9-6a–e).
4. **Train denoiser** — PyTorch pipeline trains a ~120K-parameter U-Net on ~2,240 frames with L1 + VGG perceptual loss in ACES-tonemapped space. ✅ Complete (F9-5, F9-7).
5. **Deploy trained model** — Deni loads `.denimodel` weights and performs GPU inference via custom GLSL compute shaders (7 dispatches per frame). F11-1 (weight loading) ✅ complete; F11-2 (inference shaders) in progress.
6. **Port to Vulkan mobile** — `monti_view` validates mobile path tracing and denoising on Android. Planned (F6).

The training data pipeline has been significantly expanded since the initial plan: manual viewpoint capture in `monti_view` (press `P`), automated viewpoint variation generation, per-viewpoint environment maps and lighting rigs, transparent-background rendering by default, GPU-side reference accumulation, uncompressed EXR output, safetensors conversion for fast training I/O, and invalid viewpoint pruning. See the completed plans: [datagen_performance_plan.md](datagen_performance_plan.md), [prune_dark_viewpoints_plan.md](prune_dark_viewpoints_plan.md), [safetensors_conversion_plan.md](safetensors_conversion_plan.md), [training_viewpoints_and_background_plan.md](training_viewpoints_and_background_plan.md).

Later phases (ReSTIR, WebGPU/WASM) will extend the renderer and denoiser libraries. NRD ReLAX may be added to Deni later if cross-vendor denoising is needed before the ML denoiser is ready. The apps will gain features to exercise new capabilities, but the initial scope is desktop Vulkan. Mobile Vulkan support (F6) and ReSTIR on mobile HW RT devices are planned follow-ups.

---

## 3. Dependencies

Fetched via CMake `FetchContent`, matching `rtx-chessboard` patterns:

### Shared (both apps)

| Dependency | Version | Purpose |
|---|---|---|
| **volk** | 1.4.304 | Vulkan function pointer loader (`VK_NO_PROTOTYPES`). **App-level only** — not linked by `deni_vulkan` or `monti_vulkan`. The app calls `volkInitialize()`, `volkLoadInstance()`, `volkLoadDevice()` and passes `vkGetDeviceProcAddr` to library desc structs. |
| **VMA** | v3.2.1 | GPU memory allocation |
| **GLM** | 1.0.1 | Math (vectors, matrices, quaternions) |
| **cgltf** | v1.14 | glTF 2.0 loading (used by `monti_scene`) |
| **tinyexr** | v1.0.9 | EXR read/write (used by `monti_capture` and environment loader) |
| **stb** | master | Image loading (PNG, JPG, TGA for textures) |
| **nlohmann/json** | v3.11.3 | Camera path files and configuration |

### `monti_datagen` only

| Dependency | Version | Purpose |
|---|---|---|
| **CLI11** | v2.4.2 | Command-line argument parsing |

### `monti_view` only

| Dependency | Version | Purpose |
|---|---|---|
| **SDL3** | release-3.2.8 | Window creation, input, Vulkan surface |
| **Dear ImGui** | v1.91.8 | Immediate-mode UI |
| **FreeType** | VER-2-13-3 | TrueType font rasterization for ImGui |

### Library targets

Both apps link against: `monti_scene`, `monti_vulkan`, and `deni_vulkan`. Additionally, `monti_datagen` links `monti_capture` (EXR writer).

Not carried over from `rtx-chessboard`: Jolt Physics, libdatachannel, MbedTLS, cpp-httplib, miniaudio, stduuid, NVIDIA NGX SDK.

---

## 4. Command-Line Interface

### `monti_view`

```
monti_view [options] [scene.glb]

Options:
  --help                          Show help and exit
  --width <px>                    Window width (default: 1280)
  --height <px>                   Window height (default: 720)
  --spp <n>                       Samples per pixel (default: 4)
  --env <file.exr>                Environment map (default: scene's environment, if any)
  --exposure <ev100>              Exposure override (default: 0.0)
```

Opens a window, loads the scene file (if provided, otherwise shows an empty scene), and enters the render loop. The user can also drag-and-drop a `.glb`/`.gltf` file onto the window to load it.

### `monti_datagen`

```
monti_datagen [options] <scene.glb>

Options:
  --help                          Show help and exit
  --output <dir>                  Output directory (default: ./capture/)
  --width <px>                    Render width (default: 960)
  --height <px>                   Render height (default: 540)
  --spp <n>                       Noisy samples per pixel (default: 4)
  --ref-frames <n>                Frames to accumulate for reference (default: 64)
  --exposure <ev100>              Default exposure EV100 (overridden by per-viewpoint values)
  --position <X Y Z>              Single camera position (mutually exclusive with --viewpoints)
  --target <X Y Z>                Single camera look-at target (requires --position)
  --fov <degrees>                 Vertical FOV in degrees (default: 60)
  --viewpoints <file.json>        JSON viewpoints file (mutually exclusive with --position/--target)
  --exr-compression <mode>        EXR compression: none (default), zip
```

Requires a scene file. If `--viewpoints` is provided, iterates through all viewpoints in the JSON file. If `--position`/`--target` are provided, renders a single viewpoint. If neither is specified, auto-fits the camera to the scene bounding box.

Each viewpoint is rendered as a noisy G-buffer (at `--spp`) plus a high-SPP reference accumulated over `--ref-frames` frames (effective reference SPP = `ref-frames × spp`). The reference is accumulated entirely on the GPU via a compute shader (no CPU readback per frame). Output is two EXR files per viewpoint (input + target).

Per-viewpoint JSON entries can override: exposure, environment map path, environment blur level, environment intensity, and light rig path. Background pixels render as transparent black (RGBA 0,0,0,0) by default; per-viewpoint `"environmentBlur"` enables environment map background with configurable blur.

Designed to be invoked by the Python orchestration scripts:

```bash
# Example: batch data generation with viewpoints JSON
python scripts/generate_training_data.py \
    --scenes scenes/ \
    --viewpoints viewpoints/ \
    --output training_data/ \
    --spp 4 --ref-frames 64
```

---

## 5. Viewpoints JSON File Format

Viewpoint files are JSON arrays that define camera positions and per-viewpoint rendering parameters for `monti_datagen`. Each entry produces one frame of training data.

```json
[
    {
        "id": "a3f1c0b2",
        "position": [0.0, 1.5, 3.0],
        "target": [0.0, 0.0, 0.0],
        "fov": 60.0,
        "exposure": 1.0,
        "environment": "hdris/autumn_field_puresky_1k.exr",
        "environmentIntensity": 1.5,
        "environmentBlur": 3.5,
        "lights": "light_rigs/overhead.json"
    },
    {
        "id": "f7e2d1a9",
        "position": [3.0, 1.5, 0.0],
        "target": [0.0, 0.0, 0.0],
        "fov": 60.0
    }
]
```

**Required fields:** `position` (vec3), `target` (vec3).

**Optional fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | string | — | 8-hex-char unique identifier (generated by `monti_view` and `generate_viewpoints.py`) |
| `fov` | float | 60.0 | Vertical FOV in degrees |
| `exposure` | float | CLI `--exposure` | Per-viewpoint exposure EV100 override |
| `environment` | string | scene default | Path to HDRI environment map (`.exr`) |
| `environmentIntensity` | float | 1.0 | Environment light intensity multiplier |
| `environmentBlur` | float | — | If present, enables environment map background with this blur mip level (otherwise transparent black) |
| `lights` | string | — | Path to light rig JSON file (area lights to add to the scene) |

**Viewpoint authoring:**
- **Manual capture:** In `monti_view`, press `P` to save the current camera position, target, FOV, and exposure to a viewpoints JSON file. The file is auto-named from the scene (e.g., `DamagedHelmet.glb` → `damaged_helmet.json`).
- **Automated generation:** `training/scripts/generate_viewpoints.py` computes orbit/hemisphere viewpoints and generates variations from seed files with position jitter, target jitter, and orbit perturbation.
- **Variation generation:** Seed viewpoints (hand-authored) can be expanded with `--variations-per-seed` and `--seed-jitter` options in `generate_viewpoints.py`.

---

## 6. Interactive Viewer (`monti_view`)

### 6.1 Window and Initialization

1. Parse CLI arguments.
2. Initialize SDL3 (`SDL_Init(SDL_INIT_VIDEO)`).
3. Create SDL3 window (1280×720 default, resizable, `SDL_WINDOW_VULKAN`).
4. Initialize Vulkan via volk: instance, physical device, device, queue.
5. Create VMA allocator.
6. Create swapchain (3 frames in flight, FIFO present mode).
7. Create Monti renderer (`monti::vulkan::Renderer::Create()`), passing `vkGetDeviceProcAddr` in `RendererDesc`.
8. Create Deni denoiser (`deni::vulkan::Denoiser::Create()`), passing `vkGetDeviceProcAddr` in `DenoiserDesc`.
9. Detect NVIDIA GPU; if present, initialize DLSS-RR (app-level, not part of Deni). Follows rtx-chessboard Volk integration pattern.
10. Allocate G-buffer images and tone-mapping output image.
11. Initialize ImGui (Vulkan + SDL3 backends, FreeType font rendering).
12. If a scene file was provided, load it (see §6.3).

### 6.2 Main Loop

Each frame:

1. **Poll events** — SDL3 events forwarded to ImGui, then to camera controller.
2. **Update camera** — apply controller input, update `scene.SetActiveCamera()`.
3. **Render** — record command buffer:
   - `renderer->RenderFrame(cmd, gbuffer, frame_index)` — path trace.
   - Denoise (user-selected mode):
     - **Passthrough:** `denoiser->Denoise(cmd, denoiser_input)` — Deni passthrough (no-op copy).
     - **DLSS-RR:** app-level DLSS-RR evaluation (NVIDIA only, follows rtx-chessboard pattern).
   - Tone map denoised output → swapchain image.
   - ImGui draw commands → swapchain image.
4. **Present** — `vkQueuePresentKHR`.
5. **Advance frame** — cycle frame-in-flight index.

Swapchain recreation on `VK_ERROR_OUT_OF_DATE_KHR` and `SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED` (via `SDL_AddEventWatch` for resize during drag), matching the `rtx-chessboard` pattern.

### 6.3 Scene Loading

When a glTF file is loaded (via CLI argument or drag-and-drop):

1. `monti::gltf::LoadGltf(scene, path)` — populates the `Scene` with nodes, meshes, materials, textures.
2. Upload mesh data to GPU via `monti::vulkan::UploadMeshToGpu()` and register with `GpuScene`.
3. Load textures to GPU.
4. Auto-fit camera: compute scene bounding box, position camera to see the entire model.
5. Reset accumulation (`denoiser_input.reset_accumulation = true`).

### 6.4 Camera Controller

**Fly camera** as the primary navigation mode (suitable for arbitrary scenes, unlike orbit which assumes a single focal point):

| Input | Action |
|---|---|
| Right-click + mouse drag | Look (yaw / pitch) |
| WASD | Move forward / left / backward / right |
| Q / E | Move down / up |
| Mouse wheel | Adjust movement speed |
| Shift | Fast movement multiplier |
| F | Focus on scene center (reset camera to auto-fit) |

Movement speed scales with the scene bounding box diagonal so it feels consistent across differently-sized models.

An **orbit mode** toggle (hotkey `O`) is also available for inspecting objects from fixed distances:

| Input | Action |
|---|---|
| Left-click + drag | Orbit (yaw / pitch around target) |
| Middle-click + drag | Pan target |
| Mouse wheel | Zoom (adjust distance) |

Camera input is suppressed when ImGui captures the mouse (`ImGui::GetIO().WantCaptureMouse`).

### 6.5 ImGui Panels

The UI is minimal — just enough to control the renderer and inspect the scene.

**Top Bar (always visible):**
- FPS / frame time (CPU-measured)
- Scene file name (or "No scene loaded")
- Mode indicator: "Fly" / "Orbit"

*GPU timestamp-based per-pass timing (renderer ms / denoiser ms) is deferred to a future phase.*

**Settings Panel (toggle `Tab`):**
- **Render:** SPP slider (1–64), exposure EV100, environment rotation
- **Debug visualization:** Off / Normals / Albedo / Depth / Motion Vectors / Noisy (undenoised)

*Denoiser selection UI (Passthrough / DLSS-RR) is added in F1 when DLSS-RR is implemented.*
- **Camera:** FOV slider, near/far planes, current position/rotation (read-only)
- **Scene info:** Node count, mesh count, material count, triangle count

**Font:**
- Single TrueType font loaded at initialization (e.g., Inter or Roboto, bundled in `app/assets/fonts/`)
- Sized for readability at 1080p+ resolutions (16px default)

---

## 7. Data Generator (`monti_datagen`)

### 7.1 Initialization (Headless)

1. Parse CLI arguments; validate scene file is provided.
2. Initialize Vulkan via volk: instance, physical device, device, queue. **No surface, no swapchain, no window.** Pass `vkGetDeviceProcAddr` to library desc structs.
3. Create VMA allocator.
4. Load scene (`monti::gltf::LoadGltf`).
5. Build viewpoints list from one of:
   - `--viewpoints` JSON file → parse all entries with per-viewpoint overrides
   - `--position`/`--target`/`--fov` → single viewpoint
   - Neither → auto-fit camera from scene bounding box
6. Set up environment map (from first viewpoint's `environment` field, scene default, or mid-gray fallback).
7. Load area lights (from first viewpoint's `lights` field, if present).
8. Create Monti renderer with `width`/`height`.
9. Upload meshes to GPU, build BLAS/TLAS.
10. Allocate G-buffer images.
11. Create GPU accumulator (compute shader for reference accumulation).
12. Create capture `Writer` with dimensions and compression mode.
13. Print configuration summary and start generation.

### 7.2 Generation Loop

For each viewpoint in the list:

1. Set camera from viewpoint entry: `scene.SetActiveCamera(position, target, fov, aspect)`.
2. Apply per-viewpoint exposure (from entry or CLI default).
3. If viewpoint has per-viewpoint environment/lights overrides, reload those resources.
4. Set renderer background mode (transparent black by default; environment background if viewpoint has `environmentBlur`).
5. Record command buffer:
   - `renderer->RenderFrame(cmd, gbuffer, frame_index)` — noisy G-buffer at render resolution.
6. Read back all G-buffer channels in a **single submission** (batched readback, not 7 separate submissions).
7. Accumulate reference on GPU:
   - Reset GPU accumulator.
   - Loop `ref_frames` times: render + accumulate via compute shader (one GPU sync per frame, no CPU readback).
   - Finalize: divide by frame count, read back once.
8. `writer->WriteFrame(input_frame, target_frame, viewpoint_index)` — writes two EXR files.
9. Print progress to stdout: `[42/256] vp_42 written (1.23s)`.

### 7.3 Exit

- Print summary: total viewpoints, total time, output directory.
- Write `timing.json` with per-viewpoint and overall timing data.
- Exit code 0 on success, 1 on any error (file I/O, Vulkan failure, missing scene).
- Errors printed to stderr.

---

## 8. File Structure

App code lives in three directories — `core/` (shared), `view/` (`monti_view`-only), and `datagen/` (`monti_datagen`-only):

```
app/
├── core/                               # Shared by both executables
│   ├── vulkan_context.h / .cpp         # Instance, device, queue, VMA (headless or windowed)
│   ├── frame_resources.h / .cpp        # Per-frame command buffer, sync objects
│   ├── GBufferImages.h / .cpp          # G-buffer + reference image allocation
│   ├── CameraSetup.h / .cpp            # Auto-fit camera from scene bounding box
│   ├── EnvironmentLoader.h / .cpp      # HDR environment map loading (tinyexr)
│   └── tone_mapper.h / .cpp            # HDR → LDR tone mapping compute shader
├── view/                               # monti_view only
│   ├── main.cpp                        # Entry point, CLI parsing, render loop, viewpoint capture (P key)
│   ├── CameraController.h / .cpp       # Fly + orbit camera from input
│   ├── swapchain.h / .cpp              # Swapchain management
│   ├── UiRenderer.h / .cpp             # ImGui init, frame recording, font loading
│   └── Panels.h / .cpp                 # Settings and info panels
├── datagen/                            # monti_datagen only
│   ├── main.cpp                        # Entry point, CLI parsing (CLI11)
│   ├── GenerationSession.h / .cpp      # Headless generation loop with per-viewpoint overrides
│   └── ViewpointEntry.h               # ViewpointEntry struct (position, target, fov, exposure, env, lights)
├── shaders/
│   └── tonemapping.comp                # Tone mapping compute shader
└── assets/
    └── fonts/
        └── Inter-Regular.ttf           # UI font (monti_view only)
```

### Shared Code with rtx-chessboard

The following modules are ported from `rtx-chessboard` with modifications (used by `monti_view` unless noted):

| Module | Source in rtx-chessboard | Changes for Monti |
|---|---|---|
| Vulkan context | `src/core/vulkan_context.*` | Headless mode for `monti_datagen` (skip surface/swapchain). Shared by both apps. |
| Swapchain | `src/core/swapchain.*` | Unchanged pattern (`monti_view` only) |
| Frame sync | `src/core/sync_objects.*`, `command_pool.*` | Combined into `frame_resources`. Shared by both apps. |
| Tone mapper | `src/render/tone_mapper.*` | Unchanged (app-local, not in Monti library). Shared by both apps. |
| ImGui setup | `src/ui/ui_renderer.*` | Remove game panel; add settings/info panels, viewpoint capture indicator (`monti_view` only) |
| Camera controller | `src/input/camera_controller.*` | Replace orbit-only with fly + orbit toggle. Added `CurrentViewpoint()` for viewpoint capture (`monti_view` only) |
| Environment loader | `src/loaders/environment_loader.*` | HDR pixel loading remains in app (tinyexr); CDF computation + GPU resources in `monti_vulkan` (internal `EnvironmentMap` class). Shared by both apps. Supports per-viewpoint environment swaps in `monti_datagen`. |

---

## 9. CMake Integration

Both app targets are defined in the root `CMakeLists.txt` alongside the library targets. They share a common source set (`CORE_SOURCES`) and link a common set of libraries:

```cmake
# --- Shared app core ---
set(CORE_SOURCES
    app/core/vulkan_context.cpp
    app/core/frame_resources.cpp
    app/core/gbuffer_images.cpp
    app/core/scene_loader.cpp
    app/core/tone_mapper.cpp
)

set(CORE_LIBS
    monti_scene
    monti_vulkan
    deni_vulkan
    volk                            # App-level only — libraries are loader-agnostic
    GPUOpen::VulkanMemoryAllocator
    glm::glm
    Vulkan::Headers
    nlohmann_json::nlohmann_json
)

set(CORE_DEFS
    VK_NO_PROTOTYPES
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    _CRT_SECURE_NO_WARNINGS
)

# --- monti_view (interactive viewer) ---
set(VIEW_SOURCES
    app/view/main.cpp
    app/view/CameraController.cpp
    app/view/swapchain.cpp
    app/view/UiRenderer.cpp
    app/view/Panels.cpp
    ${IMGUI_SOURCES}
)

add_executable(monti_view ${CORE_SOURCES} ${VIEW_SOURCES})
add_dependencies(monti_view app_shaders)

target_link_libraries(monti_view PRIVATE
    ${CORE_LIBS}
    SDL3::SDL3-static
    freetype
)

target_compile_definitions(monti_view PRIVATE
    ${CORE_DEFS}
    IMGUI_ENABLE_FREETYPE
)

# --- monti_datagen (headless training data generator) ---
set(DATAGEN_SOURCES
    app/datagen/main.cpp
    app/datagen/GenerationSession.cpp
)

add_executable(monti_datagen ${CORE_SOURCES} ${DATAGEN_SOURCES})
add_dependencies(monti_datagen app_shaders)

target_link_libraries(monti_datagen PRIVATE
    ${CORE_LIBS}
    monti_capture
    CLI11::CLI11
)

target_compile_definitions(monti_datagen PRIVATE
    ${CORE_DEFS}
)
```

---

## 10. Design Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | Two separate executables | `monti_datagen` deploys on headless GPU servers without SDL3/ImGui/display dependencies; `monti_view` carries interactive-mode dependencies that batch pipelines never need |
| 2 | Fly camera as default (`monti_view`) | General-purpose glTF scenes aren't always object-centric; fly cam works everywhere |
| 3 | GPU-side reference accumulation (`monti_datagen`) | Eliminates 768→257 fence waits per viewpoint and 512 staging buffer allocs; FP32 accumulation compute shader runs entirely on GPU |
| 4 | Viewpoints JSON replaces camera paths | Per-viewpoint overrides (exposure, environment, lights, background blur) enable richer training data without additional CLI complexity |
| 5 | Headless Vulkan for `monti_datagen` | No window/swapchain overhead; enables running on headless servers or in CI |
| 6 | Tone mapper lives in app, not library | Tone mapping is a display concern; the libraries produce linear HDR output only |
| 7 | Single font, single size (`monti_view`) | Keeps UI simple; can add font size options later if needed |
| 8 | Drag-and-drop scene loading (`monti_view`) | Fast iteration during development; avoids file-picker dialogs |
| 9 | Manual viewpoint capture via `P` key (`monti_view`) | Enables hand-authoring high-quality training viewpoints for scenes where auto-orbit fails (interiors, small objects, directional scenes) |
| 10 | Progress to stdout (`monti_datagen`) | Script-friendly; parseable for progress bars (`[N/M]` format) |
| 11 | nlohmann/json for viewpoints + CLI11 for arguments | Already dependencies in the ecosystem; simple and well-tested |
| 12 | Shared `core/` source set | Avoids duplicating Vulkan init, G-buffer allocation, and scene loading between executables |
| 13 | volk confined to app layer | Libraries (`deni_vulkan`, `monti_vulkan`) are loader-agnostic — they resolve Vulkan functions via `PFN_vkGetDeviceProcAddr` passed in their desc structs. Apps pass `vkGetDeviceProcAddr` after `volkLoadDevice()`. This enables Deni integration into any Vulkan engine regardless of loader choice. |
| 14 | Transparent black background by default (`monti_datagen`) | Prevents environment map samples from biasing training; geometry coverage can be computed from alpha=0 pixels |
| 15 | Uncompressed EXR default (`monti_datagen`) | ZIP compression achieves only ~1.3× on noisy FP16 radiance at significant CPU cost; uncompressed also benefits training DataLoader |
| 16 | Timing JSON output (`monti_datagen`) | Machine-readable per-viewpoint timing enables performance analysis and regression detection across datagen runs |

---

## 11. Future Extensions (Out of Scope for v1)

- **GPU frame timing** — `VkQueryPool` timestamp queries for per-pass timing (renderer ms, denoiser ms, tonemap ms) displayed in the top bar (`monti_view`).
- **Window resize** — Propagate swapchain recreation to renderer, denoiser, G-buffer, and tone mapper (`monti_view`).
- **Animation playback** — glTF skeletal/morph animations for temporal training data (`monti_datagen`).
- **Material editor** — tweak PBR parameters in the UI for experimentation (`monti_view`).
- **Multi-GPU capture** — distribute frames across GPUs for faster batch generation (`monti_datagen`).
- **Remote rendering** — stream the rendered output over a network (`monti_view`).
- **Android viewer** — SDL3 supports Android; `monti_view` could be ported with minimal changes.
- **Benchmark mode** — fixed camera path with timing output for renderer performance regression testing (either app).
- **Denoiser selection UI** — "Passthrough" / "ML" / "DLSS-RR" toggle in settings panel, added when F11-3 or F1 lands (`monti_view`).
