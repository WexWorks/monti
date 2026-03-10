# Monti App Specification

## 1. Overview

The Monti app (`monti_app`) is a glTF viewer and training-data generator built on the Monti renderer and Deni denoiser libraries. It operates in two modes:

| Mode | Purpose | Window | GPU Present |
|---|---|---|---|
| **Interactive** | Load a glTF scene, navigate with a camera, view path-traced + denoised output | SDL3 window + ImGui | Yes |
| **Capture** | Load a glTF scene, render training data to multi-channel EXR files, exit | Headless (no window) | No |

The app reuses proven patterns from the `rtx-chessboard` project: SDL3 windowing, volk Vulkan loading, VMA memory allocation, ImGui with FreeType font rendering, and the same frame-in-flight architecture. It strips everything chess/physics/network-specific and adds headless rendering and capture automation.

---

## 2. Roadmap Context

This app supports the following development arc:

1. **Implement Monti + Deni** — the app is the primary integration vehicle for both libraries.
2. **Generate initial training data** — capture mode produces multi-channel EXR training sets from integration test scenes.
3. **Train denoiser** — external PyTorch (or similar) pipeline consumes the EXR training data (outside this app).
4. **Deploy trained model** — Deni loads the trained weights and performs GPU inference (Vulkan compute, implementation TBD).
5. **Port to Vulkan mobile** — the app validates mobile path tracing and denoising on Android.

Later phases (ReSTIR, ReLAX, WebGPU/WASM) will extend the renderer and denoiser libraries. The app will gain features to exercise them, but the initial scope is desktop Vulkan only.

---

## 3. Dependencies

Fetched via CMake `FetchContent`, matching `rtx-chessboard` patterns:

| Dependency | Version | Purpose |
|---|---|---|
| **volk** | 1.4.304 | Vulkan function pointer loader (`VK_NO_PROTOTYPES`) |
| **VMA** | v3.2.1 | GPU memory allocation |
| **SDL3** | release-3.2.8 | Window creation, input, Vulkan surface (interactive mode only) |
| **GLM** | 1.0.1 | Math (vectors, matrices, quaternions) |
| **Dear ImGui** | v1.91.8 | Immediate-mode UI |
| **FreeType** | VER-2-13-3 | TrueType font rasterization for ImGui |
| **cgltf** | v1.14 | glTF 2.0 loading (used by `monti_scene`) |
| **tinyexr** | v1.0.9 | EXR read/write (used by `monti_capture` and environment loader) |
| **stb** | master | Image loading (PNG, JPG, TGA for textures) |
| **nlohmann/json** | v3.11.3 | Camera path files and configuration |

The app links against the Monti library targets: `monti_scene`, `monti_vulkan`, `deni_vulkan`, and `monti_capture`.

Not carried over from `rtx-chessboard`: Jolt Physics, libdatachannel, MbedTLS, cpp-httplib, miniaudio, stduuid, NVIDIA NGX SDK.

---

## 4. Command-Line Interface

```
monti_app [options] [scene.glb]

Options:
  --help                          Show help and exit
  --capture                       Run in headless capture mode (no window)
  --output <dir>                  Capture output directory (default: ./capture/)
  --width <px>                    Render width (default: 1920)
  --height <px>                   Render height (default: 1080)
  --spp <n>                       Samples per pixel for noisy render (default: 4)
  --ref-spp <n>                   Samples per pixel for reference render (default: 256)
  --camera-path <file.json>       Camera path file for capture (required in capture mode)
  --env <file.exr>                Environment map override (default: scene's environment or built-in)
  --exposure <ev100>              Exposure override (default: 0.0)
```

**Interactive mode** (default): opens a window, loads the scene file (if provided, otherwise shows an empty scene with the environment map), and enters the render loop. The user can also drag-and-drop a `.glb`/`.gltf` file onto the window to load it.

**Capture mode** (`--capture`): requires `--camera-path`. Loads the scene, iterates through camera positions defined in the path file, renders noisy + reference frames, writes EXR files, and exits with code 0 on success or non-zero on failure. No window is created. Designed to be invoked by scripts:

```bash
# Example: batch training data generation
for model in models/*.glb; do
    monti_app --capture \
        --camera-path paths/orbit_64.json \
        --output "training_data/$(basename $model .glb)/" \
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

## 6. Interactive Mode

### 6.1 Window and Initialization

1. Parse CLI arguments.
2. Initialize SDL3 (`SDL_Init(SDL_INIT_VIDEO)`).
3. Create SDL3 window (1280×720 default, resizable, `SDL_WINDOW_VULKAN`).
4. Initialize Vulkan via volk: instance, physical device, device, queue.
5. Create VMA allocator.
6. Create swapchain (3 frames in flight, FIFO present mode).
7. Create Monti renderer (`monti::vulkan::Renderer::Create()`).
8. Create Deni denoiser (`deni::vulkan::Denoiser::Create()`).
9. Allocate G-buffer images and tone-mapping output image.
10. Initialize ImGui (Vulkan + SDL3 backends, FreeType font rendering).
11. If a scene file was provided, load it (see §6.3).

### 6.2 Main Loop

Each frame:

1. **Poll events** — SDL3 events forwarded to ImGui, then to camera controller.
2. **Update camera** — apply controller input, update `scene.SetActiveCamera()`.
3. **Render** — record command buffer:
   - `renderer->RenderFrame(cmd, gbuffer, frame_index)` — path trace.
   - `denoiser->Denoise(cmd, denoiser_input)` — denoise.
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
4. If the glTF contains no environment light, load a default built-in environment map.
5. Auto-fit camera: compute scene bounding box, position camera to see the entire model.
6. Reset accumulation (`denoiser_input.reset_accumulation = true`).

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
- **Denoiser:** Enable/disable toggle (passthrough vs. active denoiser when available)
- **Debug visualization:** Off / Normals / Albedo / Depth / Motion Vectors / Noisy (undenoised)
- **Camera:** FOV slider, near/far planes, current position/rotation (read-only)
- **Scene info:** Node count, mesh count, material count, triangle count

**Capture Panel (toggle `C`):**
- Camera path: file picker or built-in generator selector
- Output directory
- SPP / reference SPP
- Resolution override
- "Start Capture" button — runs capture in the current window (shows progress bar), writes EXRs
- "Record Path" toggle — records camera positions as the user navigates, saves to JSON

**Font:**
- Single TrueType font loaded at initialization (e.g., Inter or Roboto, bundled in `app/assets/fonts/`)
- Sized for readability at 1080p+ resolutions (16px default)

---

## 7. Capture Mode

### 7.1 Initialization (Headless)

1. Parse CLI arguments; validate `--camera-path` and scene file are provided.
2. Initialize Vulkan via volk: instance, physical device, device, queue. **No surface, no swapchain, no SDL window.**
3. Create VMA allocator.
4. Create Monti renderer.
5. Allocate G-buffer images and reference image (`VK_FORMAT_R32G32B32A32_SFLOAT`).
6. Load scene (same as §6.3 but without auto-fit camera or UI).
7. Create capture `Writer`.

### 7.2 Capture Loop

For each camera position in the path:

1. Set `scene.SetActiveCamera(camera_params)` from the path entry.
2. Record command buffer:
   - `renderer->SetSamplesPerPixel(spp)`.
   - `renderer->RenderFrame(cmd, gbuffer, frame_index)` — noisy G-buffer.
   - `renderer->SetSamplesPerPixel(ref_spp)`.
   - `renderer->RenderFrame(cmd, reference_target, frame_index)` — high-SPP reference.
3. Submit and wait (synchronous — throughput doesn't matter for batch capture).
4. Read back all G-buffer channels + reference image to CPU via staging buffers.
5. `writer->WriteFrame(capture_frame, frame_index)`.
6. Print progress to stdout: `[42/256] frame_0042.exr written (1.23s)`.

### 7.3 Exit

- Print summary: total frames, total time, output directory.
- Exit code 0 on success, 1 on any error (file I/O, Vulkan failure, missing scene).
- Errors printed to stderr.

---

## 8. File Structure

All app code lives in the `app/` directory within the Monti project:

```
app/
├── main.cpp                        # Entry point, CLI parsing, mode dispatch
├── cli.h / cli.cpp                 # CLI argument parsing
├── interactive/
│   ├── app.h / app.cpp             # Interactive mode: init, loop, shutdown
│   ├── camera_controller.h / .cpp  # Fly + orbit camera from input
│   ├── scene_loader.h / .cpp       # glTF load + GPU upload + auto-fit
│   ├── ui_renderer.h / .cpp        # ImGui init, frame recording, font loading
│   └── panels.h / .cpp             # Settings, capture, and info panels
├── capture/
│   ├── capture_session.h / .cpp    # Headless capture loop
│   └── camera_path.h / .cpp        # Camera path loading, built-in generators
├── core/
│   ├── vulkan_context.h / .cpp     # Instance, device, queue, VMA
│   ├── swapchain.h / .cpp          # Swapchain management (interactive only)
│   ├── frame_resources.h / .cpp    # Per-frame command buffer, sync objects
│   ├── gbuffer_images.h / .cpp     # G-buffer + reference image allocation
│   └── tone_mapper.h / .cpp        # HDR → LDR tone mapping compute shader
├── shaders/
│   └── tonemapping.comp            # Tone mapping compute shader
└── assets/
    ├── fonts/
    │   └── Inter-Regular.ttf       # UI font
    └── env/
        └── default_env.exr         # Fallback environment map
```

### Shared Code with rtx-chessboard

The following modules are ported from `rtx-chessboard` with modifications:

| Module | Source in rtx-chessboard | Changes for Monti app |
|---|---|---|
| Vulkan context | `src/core/vulkan_context.*` | Remove DLSS-RR extensions; add headless mode (skip surface/swapchain) |
| Swapchain | `src/core/swapchain.*` | Unchanged pattern |
| Frame sync | `src/core/sync_objects.*`, `command_pool.*` | Combined into `frame_resources` |
| Tone mapper | `src/render/tone_mapper.*` | Unchanged (app-local, not in Monti library) |
| ImGui setup | `src/ui/ui_renderer.*` | Remove game panel; add settings/capture panels |
| Camera controller | `src/input/camera_controller.*` | Replace orbit-only with fly + orbit toggle |
| Environment loader | `src/loaders/environment_loader.*` | HDR pixel loading remains in app (tinyexr); CDF computation + GPU resources in `monti_vulkan` (internal `EnvironmentMap` class) |

---

## 9. CMake Integration

The app target is defined in the root `CMakeLists.txt` alongside the library targets:

```cmake
# --- App executable ---
set(APP_SOURCES
    app/main.cpp
    app/cli.cpp
    app/interactive/app.cpp
    app/interactive/camera_controller.cpp
    app/interactive/scene_loader.cpp
    app/interactive/ui_renderer.cpp
    app/interactive/panels.cpp
    app/capture/capture_session.cpp
    app/capture/camera_path.cpp
    app/core/vulkan_context.cpp
    app/core/swapchain.cpp
    app/core/frame_resources.cpp
    app/core/gbuffer_images.cpp
    app/core/tone_mapper.cpp
    ${IMGUI_SOURCES}
)

add_executable(monti_app ${APP_SOURCES})
add_dependencies(monti_app app_shaders)

target_link_libraries(monti_app PRIVATE
    monti_scene
    monti_vulkan
    deni_vulkan
    monti_capture
    volk
    GPUOpen::VulkanMemoryAllocator
    SDL3::SDL3-static
    glm::glm
    Vulkan::Headers
    nlohmann_json::nlohmann_json
    freetype
)

target_compile_definitions(monti_app PRIVATE
    VK_NO_PROTOTYPES
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    IMGUI_ENABLE_FREETYPE
    _CRT_SECURE_NO_WARNINGS
)
```

---

## 10. Design Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | Fly camera as default | General-purpose glTF scenes aren't always object-centric; fly cam works everywhere |
| 2 | Synchronous capture loop | Batch capture is I/O-bound (EXR writing); pipelining adds complexity for marginal gain |
| 3 | Built-in camera path generators | Eliminates need to hand-author JSON for common patterns (orbit, random sampling) |
| 4 | Headless Vulkan for capture | No window/swapchain overhead; enables running on headless servers or in CI |
| 5 | Tone mapper lives in app, not library | Tone mapping is a display concern; the libraries produce linear HDR output only |
| 6 | Single font, single size | Keeps UI simple; can add font size options later if needed |
| 7 | Drag-and-drop scene loading | Fast iteration during development; avoids file-picker dialogs |
| 8 | Camera path recording in interactive mode | Makes it easy to set up training viewpoints by hand for important scenes |
| 9 | Progress to stdout in capture mode | Script-friendly; parseable for progress bars (`[N/M]` format) |
| 10 | nlohmann/json for camera paths | Already a dependency in the ecosystem; simple and well-tested |

---

## 11. Future Extensions (Out of Scope for v1)

- **Animation playback** — glTF skeletal/morph animations for temporal training data.
- **Material editor** — tweak PBR parameters in the UI for experimentation.
- **Multi-GPU capture** — distribute frames across GPUs for faster batch generation.
- **Remote rendering** — stream the rendered output over a network.
- **Android viewer** — SDL3 supports Android; the interactive mode could be ported with minimal changes.
- **Benchmark mode** — fixed camera path with timing output for renderer performance regression testing.
