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

1. **Implement Monti + Deni** — `monti_view` is the primary integration vehicle for both libraries.
2. **Integrate DLSS-RR in monti_view** — app-level NVIDIA denoiser for interactive viewing and quality reference during development (leveraging rtx-chessboard integration).
3. **Generate initial training data** — `monti_datagen` produces multi-channel EXR training sets from integration test scenes.
4. **Train denoiser** — external PyTorch (or similar) pipeline consumes the EXR training data (outside these apps).
5. **Deploy trained model** — Deni loads the trained weights and performs GPU inference (Vulkan compute, implementation TBD).
6. **Port to Vulkan mobile** — `monti_view` validates mobile path tracing and denoising on Android.

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
  --width <px>                    Input (render) width (default: 960)
  --height <px>                   Input (render) height (default: 540)
  --target-scale <mode>           Target resolution scale factor: native, quality, performance
                                  (default: performance = 2× input resolution)
                                  Maps to ScaleMode enum: native=1×, quality=1.5×, performance=2×
  --spp <n>                       Samples per pixel for noisy render (default: 4)
  --ref-spp <n>                   Samples per pixel for reference render (default: 256)
  --camera-path <file.json>       Camera path file (required)
  --env <file.exr>                Environment map (default: scene's environment, if any)
  --exposure <ev100>              Exposure override (default: 0.0)
```

Requires a scene file and `--camera-path`. Loads the scene, iterates through camera positions defined in the path file, renders noisy G-buffer at input resolution + high-SPP reference at target resolution, writes two EXR files per frame (input + target), and exits with code 0 on success or non-zero on failure. No window is created. The target resolution is computed as `floor(input_dim × scale_factor / 2) × 2` using the same formula as `ScaleMode` (§4.11 of the design spec). Designed to be invoked by scripts:

```bash
# Example: batch training data generation (2× super-resolution)
for model in models/*.glb; do
    monti_datagen \
        --camera-path paths/orbit_64.json \
        --output "training_data/$(basename $model .glb)/" \
        --width 960 --height 540 \
        --target-scale performance \
        --spp 4 --ref-spp 256 \
        "$model"
done
```

---

## 5. Camera Path File Format

Camera paths are JSON files that define a sequence of camera positions for capture. They can also be recorded from interactive mode and saved.

```json
{
    "frames": [
        {
            "position": [0.0, 1.5, 3.0],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_degrees": 60.0
        },
        {
            "position": [3.0, 1.5, 0.0],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_degrees": 60.0
        }
    ]
}
```

Each entry produces one frame of training data. The `target` and `up` fields define the look-at orientation. The `fov_degrees` field is optional (defaults to 60°).

**Built-in generators:** The app can also generate camera paths programmatically for common patterns. These are invoked via the CLI:

```
  --camera-path orbit:64            # 64-frame orbit at default elevation
  --camera-path orbit:128:30        # 128 frames, 30° elevation
  --camera-path random:256          # 256 random viewpoints on a sphere
```

Built-in generators auto-fit the camera distance to the scene's bounding box so the model fills the frame.

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
- FPS / frame time / renderer ms / denoiser ms
- Scene file name (or "No scene loaded")
- Mode indicator: "Fly" / "Orbit"

**Settings Panel (toggle `Tab`):**
- **Render:** SPP slider (1–64), exposure EV100, environment rotation
- **Denoiser:** Denoiser selection (Passthrough / DLSS-RR when on NVIDIA hardware). DLSS-RR is app-level only, not part of Deni.
- **Debug visualization:** Off / Normals / Albedo / Depth / Motion Vectors / Noisy (undenoised)
- **Camera:** FOV slider, near/far planes, current position/rotation (read-only)
- **Scene info:** Node count, mesh count, material count, triangle count

**Camera Path Panel (toggle `C`):**
- "Record Path" toggle — records camera positions as the user navigates
- "Save Path" button — saves recorded positions to a JSON camera path file
- Camera path: file picker or built-in generator selector (for preview/playback)
- Path preview: visualize the camera path in the viewport

**Font:**
- Single TrueType font loaded at initialization (e.g., Inter or Roboto, bundled in `app/assets/fonts/`)
- Sized for readability at 1080p+ resolutions (16px default)

---

## 7. Data Generator (`monti_datagen`)

### 7.1 Initialization (Headless)

1. Parse CLI arguments; validate `--camera-path` and scene file are provided.
2. Compute target resolution from `--width`/`--height` and `--target-scale`:
   - `target_dim = floor(input_dim × scale_factor / 2) × 2` (same formula as `ScaleMode` §4.11)
   - `native` → 1.0×, `quality` → 1.5×, `performance` → 2.0×
   - Print both resolutions at startup: `Input: 960×540, Target: 1920×1080 (performance 2.0×)`
3. Initialize Vulkan via volk: instance, physical device, device, queue. **No surface, no swapchain, no window.** Pass `vkGetDeviceProcAddr` to library desc structs.
4. Create VMA allocator.
5. Create Monti renderer with `width`/`height` set to the **target** (larger) resolution.
6. Allocate **two** G-buffer sets:
   - Input G-buffer at input resolution (compact formats)
   - Reference G-buffer at target resolution (`VK_FORMAT_R32G32B32A32_SFLOAT` for radiance, `VK_FORMAT_R16G16B16A16_SFLOAT` for aux)
7. Load scene (same as §6.3 but without auto-fit camera or UI).
8. Create capture `Writer` with input dimensions and `ScaleMode`; query `TargetWidth()`/`TargetHeight()` for G-buffer allocation in step 6.

### 7.2 Generation Loop

For each camera position in the path:

1. Set `scene.SetActiveCamera(camera_params)` from the path entry.
2. Record command buffer:
   - `renderer->SetSamplesPerPixel(spp)`.
   - `renderer->RenderFrame(cmd, input_gbuffer, frame_index)` — noisy G-buffer at input resolution.
   - `renderer->SetSamplesPerPixel(ref_spp)`.
   - `renderer->RenderFrame(cmd, ref_gbuffer, frame_index)` — high-SPP reference at target resolution.
3. Submit and wait (synchronous — throughput doesn't matter for batch capture).
4. Read back input G-buffer channels (input resolution) + reference radiance (target resolution) to CPU via staging buffers.
5. `writer->WriteFrame(input_frame, target_frame, frame_index)` — writes two EXR files.
6. Print progress to stdout: `[42/256] frame_0042 written (1.23s)`.

### 7.3 Exit

- Print summary: total frames, total time, output directory.
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
│   ├── gbuffer_images.h / .cpp         # G-buffer + reference image allocation
│   ├── scene_loader.h / .cpp           # glTF load + GPU upload
│   └── tone_mapper.h / .cpp            # HDR → LDR tone mapping compute shader
├── view/                               # monti_view only
│   ├── main.cpp                        # Entry point, CLI parsing
│   ├── app.h / app.cpp                 # Interactive mode: init, loop, shutdown
│   ├── camera_controller.h / .cpp      # Fly + orbit camera from input
│   ├── swapchain.h / .cpp              # Swapchain management
│   ├── ui_renderer.h / .cpp            # ImGui init, frame recording, font loading
│   └── panels.h / .cpp                 # Settings, capture, and info panels
├── datagen/                            # monti_datagen only
│   ├── main.cpp                        # Entry point, CLI parsing
│   ├── generation_session.h / .cpp     # Headless generation loop
│   └── camera_path.h / .cpp            # Camera path loading, built-in generators
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
| Vulkan context | `src/core/vulkan_context.*` | Keep DLSS-RR extensions for `monti_view` (NVIDIA-only, app-level denoiser); add headless mode for `monti_datagen` (skip surface/swapchain). Shared by both apps. |
| Swapchain | `src/core/swapchain.*` | Unchanged pattern (`monti_view` only) |
| Frame sync | `src/core/sync_objects.*`, `command_pool.*` | Combined into `frame_resources`. Shared by both apps. |
| Tone mapper | `src/render/tone_mapper.*` | Unchanged (app-local, not in Monti library). Shared by both apps. |
| ImGui setup | `src/ui/ui_renderer.*` | Remove game panel; add settings/camera-path panels (`monti_view` only) |
| Camera controller | `src/input/camera_controller.*` | Replace orbit-only with fly + orbit toggle (`monti_view` only) |
| Environment loader | `src/loaders/environment_loader.*` | HDR pixel loading remains in app (tinyexr); CDF computation + GPU resources in `monti_vulkan` (internal `EnvironmentMap` class). Shared by both apps. |

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
    app/view/app.cpp
    app/view/camera_controller.cpp
    app/view/swapchain.cpp
    app/view/ui_renderer.cpp
    app/view/panels.cpp
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
    app/datagen/generation_session.cpp
    app/datagen/camera_path.cpp
)

add_executable(monti_datagen ${CORE_SOURCES} ${DATAGEN_SOURCES})
add_dependencies(monti_datagen app_shaders)

target_link_libraries(monti_datagen PRIVATE
    ${CORE_LIBS}
    monti_capture
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
| 3 | Synchronous generation loop (`monti_datagen`) | Batch capture is I/O-bound (EXR writing); pipelining adds complexity for marginal gain |
| 4 | Built-in camera path generators | Eliminates need to hand-author JSON for common patterns (orbit, random sampling) |
| 5 | Headless Vulkan for `monti_datagen` | No window/swapchain overhead; enables running on headless servers or in CI |
| 6 | Tone mapper lives in app, not library | Tone mapping is a display concern; the libraries produce linear HDR output only |
| 7 | Single font, single size (`monti_view`) | Keeps UI simple; can add font size options later if needed |
| 8 | Drag-and-drop scene loading (`monti_view`) | Fast iteration during development; avoids file-picker dialogs |
| 9 | Camera path recording in `monti_view` | Makes it easy to set up training viewpoints by hand for important scenes |
| 10 | Progress to stdout (`monti_datagen`) | Script-friendly; parseable for progress bars (`[N/M]` format) |
| 11 | nlohmann/json for camera paths | Already a dependency in the ecosystem; simple and well-tested |
| 12 | Shared `core/` source set | Avoids duplicating Vulkan init, G-buffer allocation, and scene loading between executables |
| 13 | volk confined to app layer | Libraries (`deni_vulkan`, `monti_vulkan`) are loader-agnostic — they resolve Vulkan functions via `PFN_vkGetDeviceProcAddr` passed in their desc structs. Apps pass `vkGetDeviceProcAddr` after `volkLoadDevice()`. This enables Deni integration into any Vulkan engine regardless of loader choice. |

---

## 11. Future Extensions (Out of Scope for v1)

- **Animation playback** — glTF skeletal/morph animations for temporal training data (`monti_datagen`).
- **Material editor** — tweak PBR parameters in the UI for experimentation (`monti_view`).
- **Multi-GPU capture** — distribute frames across GPUs for faster batch generation (`monti_datagen`).
- **Remote rendering** — stream the rendered output over a network (`monti_view`).
- **Android viewer** — SDL3 supports Android; `monti_view` could be ported with minimal changes.
- **Benchmark mode** — fixed camera path with timing output for renderer performance regression testing (either app).
