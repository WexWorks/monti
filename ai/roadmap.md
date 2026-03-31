# Monti/Deni — Roadmap

> These features are **not in the initial release** (Vulkan desktop path tracer + passthrough/ML denoiser). They are documented here for planning visibility. See [monti_design_spec.md](monti_design_spec.md) for the current-release architecture.

---

## Recommended Implementation Order

Two orderings of remaining work are provided: one prioritizing raw impact, the other prioritizing return on effort. All rendering phases (8E–8N), the ML training pipeline (F9), and the code review remediation (R1–R5) are complete. The orderings below cover only **remaining** work.

### Ordering A — Maximum Impact (Quality, Performance, Visual Accuracy)

Ordered by the magnitude of user-visible improvement, regardless of effort.

| Priority | Phase | Effort | Impact | Rationale |
|---|---|---|---|---|
| 1 | **F2** — ReSTIR DI | High | **Very High** | Spatiotemporal reservoir resampling. Dramatic quality gain for many-light scenes (Bistro, interiors, emissive signage). Builds on 8K ✅. |
| 2 | **F18** — Albedo demodulation in ML denoiser | Medium | **High** | Denoise in albedo-divided space, remodulate after inference. Matches NRD/DLSS-RR approach. Major quality gain for textured surfaces. Requires F11 ✅. |
| 3 | **F1** — DLSS-RR in `monti_view` | Medium | **High** | NVIDIA-only quality ceiling reference. Transforms interactive development experience. Leverages existing rtx-chessboard integration. |
| 4 | **F3** — Emissive mesh ReSTIR | Medium | **High** | Full temporal/spatial resampling of emissive triangles. Unlocks convergence for neon-lit streets, complex interior lighting. Requires F2. |
| 5 | **T1–T8** — Temporal super-resolution denoiser | High | **Very High** | Texture features, depthwise separable convs, temporal residual denoising, super-res upscaling, mobile fragment backend. See [temporal_denoiser_plan.md](temporal_denoiser_plan.md). Requires F11 ✅. |
| 6 | **F15** — ReSTIR GI | High | **High** | Spatiotemporal reuse of indirect illumination. The primary technique for real-time GI quality at low SPP. Requires F2. |
| 7 | **DoF-1** — Core thin-lens DoF | Low | Medium | Cinematic depth-of-field effect. ~50 LOC thin-lens ray perturbation. No BRDF/MIS changes. |
| 8 | **F4** — Volume enhancements | High | Medium | Homogeneous + heterogeneous media (fog, smoke, subsurface). Needed for specific scene types only. |
| 9 | **F6** — Mobile Vulkan renderer | Very High | Medium | Hybrid rasterize + ray query pipeline for mobile GPUs. Unlocks an entirely new platform. |
| 10 | **F14** — GPU skinning + morph targets | Medium | Medium | Animated character support. Required when dynamic scenes are needed. |
| 11 | **F20** — Cloud training scripts | Medium | Medium | Multi-GPU DDP, hyperparameter sweeps. Unlocks faster iteration and larger models. Requires F9 ✅. |
| 12 | **F21** — Broader scene acquisition | Low | Medium | More training scenes + stress scene generation. Improves denoiser generalization. Requires F9 ✅. |
| 13 | **F17** — `diffuseTransmissionTexture` | Low | Low | Per-texel transmission modulation. No current scenes require it. |
| 14 | **DoF-2** — Polygonal bokeh | Very Low | Low | ~15 LOC. Shaped bokeh highlights. Requires DoF-1. |
| 15 | **Viewpoint validation heuristics** | Low | Low | Additional `remove_invalid_viewpoints.py` checks. Training data quality polish. |
| — | ~~**F19** — Transparency output in denoiser~~ | — | — | **Deferred indefinitely.** See [F19 deferral rationale](#f19-transparency-output-deferred). |

### Ordering B — Best Return on Effort (Impact per Session)

Ordered by the ratio of user-visible improvement to implementation effort. Quick wins first.

| Priority | Phase | Effort | Impact | Rationale |
|---|---|---|---|---|
| 1 | **DoF-1** — Core thin-lens DoF | ~50 LOC | Medium | Cinematic feature. Low integration depth, no MIS/BRDF changes. One short session. |
| 2 | **DoF-2** — Polygonal bokeh | ~15 LOC | Low | Trivial delta atop DoF-1. Shaped bokeh for free. |
| 3 | **F18** — Albedo demodulation in ML denoiser | Medium | **High** | Mostly training-side changes + one remodulation shader. Leverages existing albedo GBuffer outputs. Big quality win per session. |
| 4 | **F1** — DLSS-RR in `monti_view` | Medium | **High** | Reference implementation exists in rtx-chessboard. Mostly integration wiring — the hard design work is done. Transforms interactive dev experience. |
| 5 | **F17** — `diffuseTransmissionTexture` | ~30 LOC | Low | Mechanical: add texture index, sample in shader, parse in glTF loader. One short session if a test scene needs it. |
| 6 | **F21** — Broader scene acquisition | Low | Medium | Download more scenes, generate viewpoints. Follows existing patterns. Directly improves denoiser quality. |
| 7 | **Viewpoint validation heuristics** | Low | Low | Each heuristic follows the existing near-black pattern. ~50 LOC per check, independent of each other. |
| 8 | **F2** — ReSTIR DI | High | **Very High** | Major pipeline addition (temporal + spatial resampling). High impact but also high effort and integration risk. |
| 9 | **F3** — Emissive mesh ReSTIR | Medium | **High** | Incremental on F2 — emissive lights participate in existing ReSTIR pipeline. Good !/$ *after* F2 is done. |
| 10 | **T1–T8** — Temporal super-resolution denoiser | High | **Very High** | Texture features, depthwise separable convs, temporal residual denoising, super-res upscaling, mobile fragment backend. See [temporal_denoiser_plan.md](temporal_denoiser_plan.md). Requires F11 ✅. |
| 11 | **F14** — GPU skinning + morph targets | Medium | Medium | Compute shader pipeline + BLAS refit integration. Moderate complexity, situation-dependent value. |
| 12 | **F20** — Cloud training scripts | Medium | Medium | DDP setup, sweep configs. Moderate effort, value scales with future training needs. |
| 13 | **F15** — ReSTIR GI | High | **High** | Complex (Jacobian-corrected spatial resampling). Very high impact but significant R&D risk. |
| 14 | **F4** — Volume enhancements | High | Medium | Delta tracking, phase functions, 3D density textures. High integration depth, value only for specific scenes. |
| 15 | **F6** — Mobile Vulkan renderer | Very High | Medium | Entire new renderer (rasterize G-buffer + ray query compute). Multi-session effort with new shader pipelines, TBDR optimization, and mobile-specific constraints. |

### Completed Phases (Reference)

All initial-release rendering phases (8E–8N), material extensions (8L, 8M, 8N), light system (8G, 8J, 8K), ML training pipeline (F9-1 through F9-7), ML denoiser deployment (F11-1 through F11-3), code review remediation (R1–R5), and training infrastructure improvements (datagen performance, viewpoints, transparent backgrounds, dark viewpoint pruning, safetensors conversion S1/S2/S3/S4/S5) are done.

### Key Dependencies

```
Completed: 8E ✅ → 8F ✅ → 8H ✅ → 8I ✅ (Wave 1)
           8G ✅ → 8J ✅ → 8K ✅ (Wave 2)
           8D ✅ → 8L ✅, 8M ✅, 8N ✅ (Material extensions)
           F9-6a ✅ → F9-6b ✅ → F9-6c ✅ → F9-6d ✅ → F9-6e ✅ → F9-7 ✅ → F11 ✅ → F13 ✅ (Training)

Remaining: F11 ✅ → F18 (albedo demodulation) → T2 (depthwise, retrains on demodulated data)
           F18 → T4 (temporal training assumes demodulated inputs)
           F11 ✅ → T1, T3 (infrastructure, no model change — independent of F18)
           T1, T2, T3 → T4 → T5 → T6 → T7 → T8
           monti_view path tracking (S6) → T4 (temporal training data capture)
           10B ✅ → F1 (DLSS-RR)
           8K ✅ → F2 → F3 (ReSTIR)
           F2 → F15 (ReSTIR GI)
           8I ✅ → F4 (volumes)
           DoF-1 → DoF-2
           F9 ✅ → F20 (cloud training)
           F9 ✅ → F21 (broader scenes)
           ~~F19 (deferred indefinitely — no reference denoiser uses alpha)~~
           S3 ✅, viewpoint heuristics, F17 (independent, no blockers)
           ML E2E tests ✅ → F22 (RTXPT comparison)
           F1 → F23 (DLSS-RR comparison)
```

F11 is complete (F11-1 weight loading, F11-2 GLSL inference shaders, F11-3 end-to-end integration). All initial-release rendering phases are complete. The ML training pipeline and data generation infrastructure are complete: 14 training scenes, camera path recording in `monti_view` (tracking mode, P key), sequential path rendering in `monti_datagen`, lighting rigs, HDRIs, GPU-side reference accumulation, safetensors data format, and viewpoint validation are all functional. See [datagen_performance_plan.md](datagen_performance_plan.md), [prune_dark_viewpoints_plan.md](prune_dark_viewpoints_plan.md), [safetensors_conversion_plan.md](safetensors_conversion_plan.md), and [training_viewpoints_and_background_plan.md](training_viewpoints_and_background_plan.md).

---

## RTXPT Reference Test Scenes

The NVIDIA RTXPT project (and its companion [RTXPT-Assets](https://github.com/NVIDIA-RTX/RTXPT-Assets) repository) provides several glTF test scenes that are useful for validating Monti rendering quality. Since RTXPT and Monti both consume glTF 2.0 with the same PBR extensions, these scenes can serve as direct comparison targets.

### Recommended Test Scenes for Monti

| Scene | Tests | Monti Phases Exercised |
|---|---|---|
| **Amazon Lumberyard Bistro** | Many-light interiors, emissive signage, complex geometry, reflections | 8G, 8J, 8K, 8N, F1, F2 — sourced from [Cauldron-Media BistroInterior](https://github.com/GPUOpen-LibrariesAndSDKs/Cauldron-Media/tree/master/BistroInterior) (Phase 10A-2) |
| **Kitchen** | Interior lighting, glossy reflections, transparent objects (glass, liquid) | 8E, 8I, F1 |
| **DragonAttenuation** | Transmission, volume attenuation, IOR, material variants | 8C, 8H, 8I |
| **A Beautiful Game** | Chess set with transmission + volumetric effects, clearcoat | 8C, 8H, 8I (direct comparison with rtx-chessboard) |
| **Transparent Machines** | Complex transparency, nested dielectrics | 8I, 8H |
| **Mosquito In Amber** | Nested transmission + IOR + volume | 8I, F4 |
| **Toy Car** | Clearcoat, sheen, transmission, texture transform | 8C, 8H, 8L, 8M |
| **Sponza** | Architectural lighting benchmark, environment + area lights | 8E, 8F, 8G, F1 |
| **Khronos glTF 2.0 sample models** | Material feature conformance (50+ models) | All material phases |

### Integration Approach

1. **Conformance testing:** Use Khronos glTF sample models as unit test references. Render at reference SPP (4096) and compare against known-good images.
2. **Visual comparison:** Render Bistro and Kitchen scenes in both RTXPT and Monti at matched settings (same camera, same SPP, same bounce count). Use FLIP error metric for quantitative comparison.
3. **Performance benchmarking:** Use Sponza and Bistro as standard benchmarks. Track frame time across phases to measure performance impact of new features.
4. **Scene format:** All RTXPT scenes use `.scene.json` configuration files that reference `.glb` models. Monti's glTF loader already handles `.glb`; the `.scene.json` wrapper (camera positions, light overrides) can be parsed with a small utility or manually transcribed.

---

## Future Phase Summary

> **Deni shader loading:** Phase 9A loads compiled SPIR-V from disk (`build/deni_shaders/*.spv`). The ML inference shaders (F11-2) will also be loaded from disk. A future cleanup pass could embed SPIR-V as C++ byte arrays at build time to eliminate the runtime file dependency and make Deni fully self-contained.

| Phase | Feature | Prerequisite |
|---|---|---|
| F1 | DLSS-RR in `monti_view` (NVIDIA-only, app-level quality reference) + denoiser selection UI | Phase 10B (interactive viewer with ImGui) |
| F2 | ReSTIR Direct Illumination | Phase 8K (WRS foundation) |
| F3 | Emissive mesh ReSTIR importance sampling | F2 (needs ReSTIR for correct sampling) |
| F4 | Volume enhancements (homogeneous + heterogeneous) | Phase 8I (nested dielectrics) |
| F6 | Mobile Vulkan renderer (`monti_vulkan_mobile`) | Shared GpuScene/GeometryManager ready |
| F7 | Metal renderer (C API) | Desktop design patterns established |
| F8 | WebGPU renderer (C API → WASM) | Desktop design patterns established |
| F9 | ML denoiser training pipeline | ~~Capture writer complete~~ **Done** (F9-1 through F9-7). Full pipeline: data generation, U-Net training, weight export |
| F10 | Shader permutation cache | Multi-bounce MIS complete |
| ~~F11~~ | ~~ML denoiser deployment in Deni (desktop + mobile)~~ | **Done** (F11-1 weight loading ✅, F11-2 GLSL shaders ✅, F11-3 integration ✅) |
| T1–T8 | Temporal super-resolution denoiser (texture features, depthwise separable convs, motion reprojection, temporal residual training/inference, super-res, mobile fragment backend). See [temporal_denoiser_plan.md](temporal_denoiser_plan.md) | F11 complete |
| F14 | GPU skinning + morph targets | Phase 6 (GeometryManager) |
| F15 | ReSTIR GI (indirect illumination reuse) | F2 complete |
| F16 | NRD ReLAX denoiser in Deni (cross-vendor) | F11 complete (deferred until cross-vendor denoising needed) |
| F17 | `diffuseTransmissionTexture` support | Phase 8H (diffuse transmission). Per-texel modulation of `diffuse_transmission_factor` via texture. Requires adding a texture index to `PackedMaterial::transmission_ext`, sampling in the shader, and parsing `diffuseTransmissionTexture` in the glTF loader. Low priority — no current test scenes require it. |
| F18 | Albedo demodulation in ML denoiser | F11 complete. 19-ch input (demodulated irradiance + albedo as auxiliary), 6-ch output (separate diffuse/specular irradiance), remodulate after inference. Prerequisite for T2 and T4. Detailed plan in [ml_denoiser_plan.md](ml_denoiser_plan.md#phase-f18-albedo-demodulation). |
| ~~F19~~ | ~~Transparency output in denoiser~~ | **Deferred indefinitely.** See [F19 deferral rationale](#f19-transparency-output-deferred). |
| F20 | Cloud training scripts (multi-GPU DDP, hyperparameter sweeps) | F9 complete. Enables faster iteration and larger model experiments. |
| F21 | Broader scene acquisition + stress scene generation | F9-6d complete. More diverse training data improves denoiser generalization. |
| F22 | RTXPT comparison test suite | ML E2E tests complete. Render RTXPT reference scenes (Bistro, Sponza) at matched settings and compare against Monti output using FLIP for quantitative quality tracking. |
| F23 | DLSS-RR comparison test suite | F1 complete. Add ML E2E tests comparing Monti ML denoiser output against DLSS-RR denoised output on the same noisy input for quality benchmarking. |

---

### F19: Transparency Output (Deferred)

> **Status:** Deferred indefinitely. No concrete use case requiring continuous alpha compositing exists today, and no reference denoiser consumes per-pixel alpha from a path tracer.

**Original proposal:** Use `diffuse.A`/`specular.A` alpha channels as transparency masks, enabling the denoiser to output per-pixel opacity for compositing denoised results over custom backgrounds.

**Analysis (March 2026):**

1. **Monti's renderer outputs binary alpha only.** The raygen shader's `pixel_alpha` is 0.0 (miss) or 1.0 (first opaque hit). Transparent surfaces (alpha-blend, specular transmission, nested dielectrics) continue the ray without modifying `pixel_alpha`. There is no accumulation of partial transmittance along the primary path. Adding continuous alpha would require non-trivial renderer work (accumulating per-path transmittance through stochastic alpha-blend and specular transmission decisions).

2. **Reference denoisers don't use alpha.** RTXPT outputs `float4(pathRadiance, 1)` — always alpha 1.0. NRD receives RGB radiance + hit distance only. The rtx-chessboard DLSS-RR integration hardcodes alpha to 1.0 and does not populate any of the DLSS-RR API's optional transparency parameters (`pInTransparencyMask`, `pInTransparencyLayer`, etc.). Those API parameters appear designed for hybrid renderers that rasterize particles/transparent objects separately — not for path-traced transparency.

3. **Path tracing resolves transparency inline.** In a path tracer, transparent surfaces are handled by tracing through them (refraction, transmission, stochastic alpha). The final pixel radiance already accounts for all transparent surfaces the ray encountered. There is no separate "transparent layer" to composite — the path integral is the composite.

4. **Binary hit mask is sufficient for current needs.** F18's albedo demodulation uses the binary hit mask (`diffuse.A > 0.5`) to gate demodulation on hit vs. miss pixels. The datagen pipeline's transparent-background mode (`background_mode == 0`) uses the binary `pixel_alpha` to distinguish geometry from sky. Both work correctly with the existing binary approach.

5. **DLSS-RR transparency features are for hybrid renderers.** The DLSS-RR API's `pInTransparencyLayer` / `pInTransparencyLayerOpacity` are for games that rasterize particle effects and transparent objects in a separate pass and need the denoiser to composite them. This doesn't apply to Monti's fully path-traced pipeline.

**Prerequisites if revisited:** Implementing F19 would first require a new renderer feature to output continuous alpha (accumulating `(1 - opacity)` transmittance along primary paths), then propagating that through the denoiser as a pass-through channel. A shared single alpha (same for diffuse and specular) would be sufficient — per-lobe alpha is not standard in any reference implementation.

---

### Standalone — Training Infrastructure Improvements

Remaining items from completed training plans. All are low priority and independent of each other.

#### S3 — Safetensors auto-detection in `train.py` ✅

**Source:** safetensors_conversion_plan §S3 &nbsp;|&nbsp; **Status:** Complete &nbsp;|&nbsp; **Est. ~20 LOC**

`train.py` hardcodes `ExrDataset` at line 56 (`_build_dataloaders`). `evaluate.py` already implements safetensors auto-detection — adapt the same pattern:

**Files to change:**
- `training/deni_train/train.py` — `_build_dataloaders()` function

**Implementation (adapt from `evaluate.py` lines 216-230):**
1. Add imports at the top of `train.py`:
   ```python
   from .data.safetensors_dataset import SafetensorsDataset
   from .data.splits import detect_data_format, stratified_split, stratified_split_files
   ```
2. In `_build_dataloaders()`, replace the hardcoded `ExrDataset` construction with:
   ```python
   data_format = getattr(cfg.data, "data_format", "auto")
   if data_format == "auto":
       data_format = detect_data_format(cfg.data.data_dir)
   print(f"Data format: {data_format}")

   if data_format == "safetensors":
       dataset = SafetensorsDataset(cfg.data.data_dir, transform=transform)
       train_indices, val_indices = stratified_split_files(dataset.files)
   else:
       dataset = ExrDataset(cfg.data.data_dir, transform=transform)
       train_indices, val_indices = stratified_split(dataset.pairs)
   ```
3. Remove the now-redundant `n == 0` / "No EXR pairs" check — replace with a format-aware message.

**Key details:**
- `SafetensorsDataset` accepts `transform=` with the same interface as `ExrDataset`.
- `SafetensorsDataset` exposes `.files` (list of paths); `ExrDataset` exposes `.pairs` (list of (input, target) tuples). The split functions differ accordingly: `stratified_split(pairs)` vs `stratified_split_files(files)`.
- `default.yaml` already has `data_format: "auto"` — no config change needed.
- `detect_data_format(data_dir)` in `splits.py` globs for `*.safetensors` recursively; returns `"safetensors"` or `"exr"`.

**Acceptance criteria:**
1. `python -m deni_train.train` with safetensors data auto-selects `SafetensorsDataset`.
2. Same command with EXR-only data auto-selects `ExrDataset`.
3. Explicit `data_format: "exr"` in config forces EXR even when safetensors exist.
4. Epoch time on safetensors data significantly faster than EXR (~4× speedup expected).

---

#### Viewpoint validation heuristics

**Source:** prune_dark_viewpoints_plan §5 &nbsp;|&nbsp; **Status:** Remaining

`training/scripts/remove_invalid_viewpoints.py` currently implements near-black detection only. The script is structured to accommodate additional heuristics — each new check plugs into the existing `_check_image()` → `run()` pipeline.

**File to change:**
- `training/scripts/remove_invalid_viewpoints.py`

**Existing pattern (reference for new heuristics):**
- `_check_image(input_path, target_path, ...)` → returns a reason string (e.g., `"near_black"`) or `None`.
- `is_near_black(path, ...)` → loads diffuse+specular EXR channels, computes per-pixel luminance, checks if fraction of dark pixels exceeds threshold.
- `run()` → iterates viewpoints, calls `_check_image`, moves invalid files to `invalid_<reason>/` sibling directory, removes viewpoint from JSON, logs removals.

**Heuristics to add** (each ~50 LOC, following the same scan→classify→move→log pattern):

| Heuristic | Detection logic | Threshold concept |
|---|---|---|
| **Low-contrast** | Compute luminance min/max range across the image; flag if range < threshold | `--contrast-range` (e.g., 0.01) |
| **NaN/Inf** | Count `np.isnan` + `np.isinf` pixels in loaded EXR array; flag if fraction exceeds threshold | `--nan-fraction` (e.g., 0.001) |
| **Geometric degeneracy** | Analyse depth buffer: if nearly all pixels have depth ≈ 0 or depth ≈ sentinel, camera is likely inside geometry | `--degenerate-depth-fraction` (e.g., 0.95) |
| **Saturation** | Check fraction of pixels where max(R,G,B) luminance exceeds a high threshold (blown highlights) | `--saturation-threshold`, `--saturation-fraction` |

**Implementation approach:**
1. Add a check function per heuristic (e.g., `is_low_contrast(path, range_threshold)`).
2. Register each in `_check_image()` as an additional check after `is_near_black`.
3. Add corresponding CLI arguments to `main()`.
4. Each moves invalid files to `invalid_<reason>/` (e.g., `invalid_low_contrast/`).

No changes to the rest of the pipeline — the `run()` loop, JSON update, and logging are already generic.

---

#### S6 — Camera path recording in `monti_view` (replaces `generate_viewpoints.py`)

**Status:** Pending — prerequisite for T4 (temporal training data)

Replaces the manual-seed + random-variation workflow (`generate_viewpoints.py`) with direct camera path capture in `monti_view`. This is a prerequisite for temporal training data generation (T4), which requires ordered multi-frame sequences with realistic camera motion.

**Viewpoint format change:** The per-viewpoint `id` field is replaced by `path_id` (8-hex, shared across all frames in one path) + `frame` (0-indexed integer, render order). Environment fields (`environment`, `environmentRotation`, `environmentIntensity`) are now captured per frame rather than added by a post-processing script. `generate_viewpoints.py` is deleted.

**Files to change:**
- `app/view/main.cpp` — P key toggle, per-frame motion detection, `FlushPath()`, Backspace undo, UI indicator
- `app/view/Panels.h` — `PanelState`: add `env_path` field and `PathTrackingState` fields
- `app/datagen/GenerationSession.cpp` — path-grouped sequential rendering (group by `path_id`, sort by `frame`, render all frames maintaining temporal state)
- `training/scripts/generate_training_data.py` — path-aware orchestration (all frames rendered, no truncation)
- `training/scripts/preprocess_temporal.py` — **new**: offline temporal windowing (sliding 8-frame windows) + crop extraction (consistent spatial coordinates across all frames in window), writes pre-cropped `(8,19,H,W)` safetensors
- `training/deni_train/data/temporal_safetensors_dataset.py` — **new**: minimal dataset loader for pre-cropped temporal safetensors
- `training/deni_train/train.py` — instantiate `TemporalSafetensorsDataset` in temporal mode
- **DELETE:** `training/scripts/generate_viewpoints.py`

**Interaction design:**
- `P` toggles tracking mode globally. Individual paths auto-start when camera moves (delta-based) and auto-stop when idle for ~500ms.
- Multiple paths captured per session without any additional keypresses.
- All captured frames rendered by `monti_datagen`; `preprocess_temporal.py` splits into 8-frame sliding windows offline.
- `Backspace` removes the last flushed path from the JSON file.

**Data math (example — 2500 frames, Sponza):**
- ~575 sliding windows (stride=4, window=8)
- × 4 crops/window = ~2,300 training samples
- Each sample: `(8, 19, 384, 384)` input + `(8, 7, 384, 384)` target, pre-cropped safetensors

---

## F1: DLSS-RR in `monti_view` (NVIDIA Quality Reference) + Denoiser Selection UI

> **NVIDIA only, app-level integration.** DLSS-RR (Deep Learning Super Sampling — Ray Reconstruction) is NVIDIA's ML-based denoiser providing the highest quality denoising available for ray-traced content. It is integrated directly in `monti_view` as app-level code — not in the Deni library — because it is vendor-locked to NVIDIA GPUs and serves as a development-time quality reference, not a shipping product denoiser.
>
> **Purpose:** DLSS-RR serves as the quality ceiling comparison for ML denoiser development. During training (F9), DLSS-RR output provides the perceptual reference for evaluating ML denoiser quality. During development, it provides interactive denoised viewing in `monti_view` for scene authoring and camera placement.
>
> **Reference implementation:** The [rtx-chessboard](../../../rtx-chessboard/) project has a working DLSS-RR integration using Volk for Vulkan function loading. This implementation can be directly adapted for `monti_view`.
>
> **Required inputs:** DLSS-RR consumes the same G-buffer channels already output by Monti's path tracer: noisy diffuse/specular radiance, motion vectors, linear depth, world normals + roughness, diffuse/specular albedo. No additional renderer modifications needed.
>
> **Integration approach:**
> 1. Add NVIDIA DLSS SDK as a dependency (downloaded, not committed — proprietary license).
> 2. Implement `app/view/dlss_rr_denoiser.h/.cpp` as app-local code (not in Deni).
> 3. Wire into `monti_view` render loop: trace → DLSS-RR denoise → tonemap → present.
> 4. Detect NVIDIA GPU at startup; fall back to Deni passthrough on non-NVIDIA hardware.
> 5. Add denoiser selection UI to `monti_view` settings panel (Panels.cpp): "Passthrough" / "DLSS-RR" toggle. Greyed out on non-NVIDIA hardware. This is the first denoiser UI — Phase 10B intentionally deferred the toggle until a second denoiser option existed.
>
> **Not in Deni:** DLSS-RR is intentionally not part of the Deni library. Deni is cross-vendor by design. DLSS-RR is an NVIDIA-only development tool used in the apps (`monti_view`, `monti_datagen`) for quality comparison during ML denoiser development.
>
> Expected cost: ~2–4 ms at 1080p on RTX 4090.

---

## F2: ReSTIR DI + Emissive Mesh Lights

> ReSTIR Direct Illumination will be added to support efficient many-light scenes, building on the WRS (Weighted Reservoir Sampling) foundation established in Phase 8K and emissive mesh light extraction from Phase 8J. While Phase 8K provides single-frame O(1) light selection, ReSTIR adds **temporal and spatial resampling** to dramatically improve sample quality by reusing light samples across frames and neighboring pixels.
>
> **Mobile considerations:** ReSTIR's 3 additional full-screen compute passes (temporal, spatial, final shading) add significant bandwidth. On older mobile GPUs without hardware RT acceleration, this exceeds the frame budget. However, newer mobile SoCs with dedicated RT hardware (Snapdragon 8 Elite, Dimensity 9400, Immortalis-G925+) can handle ReSTIR at mobile render resolution (540p): the reservoir buffer is ~16 MB double-buffered, and the 3 passes add ~50 MB/frame bandwidth — within budget when combined with ML upscale. The initial implementation targets desktop; mobile enablement is a follow-up once the mobile renderer (F6) is in place.

### ReSTIR Pipeline Overview

ReSTIR DI extends the Phase 8K WRS reservoir with spatiotemporal resampling. The pipeline inserts between camera ray generation and the path trace bounce loop:

1. **Candidate generation** — For each pixel, generate M light candidates using the Phase 8K WRS sampler. Evaluate unshadowed target PDF `p̂(x) = Le × BSDF × G / pdf_source` for each candidate. Store the best candidate in a per-pixel reservoir.
2. **Temporal resampling** — Merge the current frame's reservoir with the previous frame's reservoir at the reprojected pixel location (using motion vectors). Apply M-capping to bound bias (`max_temporal_M = 20`). This gives each pixel an effective sample count of ~20× without additional shadow rays.
3. **Spatial resampling** — For each pixel, gather K neighbor reservoirs (K = 3–5) within a radius and merge. Neighbor selection uses a randomized spiral pattern. Accept/reject based on normal similarity and depth ratio thresholds to avoid light leaking across geometric discontinuities.
4. **Final shading** — Trace a single shadow ray for the reservoir's selected light sample and apply the final contribution with the combined MIS weight.

### Reservoir Buffer Layout

Each pixel stores a compact reservoir (16 bytes packed):

```
struct Reservoir {
    uint   selected_light;  // Light index (Phase 8G unified light buffer)
    float  weight_sum;      // Running weight sum (for streaming WRS)
    float  target_pdf;      // p̂(selected sample) — target PDF evaluation
    uint   M;               // Number of candidates seen (for bias correction)
};
```

Double-buffered: `reservoir_current` + `reservoir_previous` (ping-pong per frame). Total memory: 2 × 16 bytes × width × height = ~64 MB at 4K.

### NEE-AT (Adaptive Temporal Feedback)

RTXPT uses NEE-AT to learn which lights matter over time. A per-light importance weight is maintained and updated each frame based on the actual contribution of selected lights. High-contribution lights are sampled more frequently in subsequent frames. This is implemented as a feedback buffer written during final shading and read during candidate generation to bias the WRS weight computation.

### Screen-Space Cache (SSC)

An optional tile-based local light cache (16×16 pixel tiles) stores the top-K most important lights per screen tile. During candidate generation, some fraction of candidates are drawn from the local tile cache (nearby lights likely to contribute) and the remainder from the global light importance map. This spatial locality heuristic improves convergence for scenes with clustered area lights.

### Integration with Phase 8K

Phase 8K provides the core WRS algorithm and `light_sampling.glsl`. ReSTIR extends this with:
- `restir_temporal.comp` — Temporal resampling compute shader
- `restir_spatial.comp` — Spatial resampling compute shader
- `restir_final.comp` — Final shading with MIS weight application
- Reservoir buffer allocation and ping-pong management in `GpuScene`

---

## F3: Emissive Mesh Importance Sampling via ReSTIR

> **Prerequisite:** F2 (ReSTIR DI pipeline).
>
> **Relationship to Phase 8J:** Phase 8J provides *basic* emissive mesh light extraction — a compute shader scans materials, decomposes emissive mesh faces into individual `TriangleLight` primitives, and adds them to the unified light buffer for single-frame WRS sampling (Phase 8K). This works well for small numbers of emissive meshes. However, scenes with many emissive meshes (hundreds of emissive triangles) degrade because even WRS selects only one light per pixel per bounce — the probability of selecting the *correct* emissive triangle is low without spatiotemporal resampling.
>
> **F3 scope:** Integrate the Phase 8J-extracted emissive triangle lights into the F2 ReSTIR DI pipeline. This means:
> - Emissive triangle lights participate in ReSTIR candidate generation alongside quad, sphere, and environment lights.
> - Temporal resampling reuses successful emissive light samples across frames (critical for convergence — an emissive TriangleLight that contributed last frame is likely still relevant).
> - Spatial resampling shares emissive light discoveries between neighboring pixels.
> - NEE-AT (F2) learns per-emissive-light importance over time.
>
> **Without F3:** Phase 8J + 8K provide basic emissive NEE (one WRS sample per bounce, no temporal reuse). This is functional but noisy for complex emissive-heavy scenes.
> **With F3:** Full ReSTIR treatment of emissive lights — converges dramatically faster for scenes like neon-lit streets, emissive signage, and complex interior lighting.

---

## F4: Volume Enhancements (Desktop Only)

> **Prerequisite:** Phase 8I (nested dielectric priority stack). Volume enhancements build on the IOR priority stack to properly track which medium the ray is currently inside.

### Per-Bounce Volume Absorption (Phase 0)

Upgrade the current exit-only Beer-Lambert absorption to apply at every bounce while the ray travels inside a dielectric volume. Currently, absorption is computed once when the ray exits the volume using `exp(-σ_a × hit_t)`. This is incorrect for paths that bounce multiple times inside the medium (e.g., total internal reflections in thick glass or gemstones) — each interior segment should attenuate the throughput independently. Use the interior list from Phase 8I to track the active medium: when the ray is inside a volume, apply `exp(-σ_a × hit_t)` after every `traceRayEXT` call, not just on exit. This matches RTXPT's approach (`PathState::InteriorList` + per-bounce `cumulativeAbsorption`).

### Homogeneous Scattering (Phase 1)

Add Beer-Lambert distance tracking for homogeneous participating media. When a ray enters a dielectric volume, sample a free-flight distance from an exponential distribution `t = -ln(ξ) / σ_t` where `σ_t` is the extinction coefficient. If the sampled distance is less than the next surface hit, a scattering event occurs in the medium:

- **Absorption:** Attenuate throughput by `exp(-σ_a × t)` along the ray segment.
- **Scattering:** At the scatter point, sample a new direction from a Henyey-Greenstein phase function with asymmetry parameter `g` (stored in `MaterialDesc`).
- **Transmittance:** If the ray exits the volume without scattering, apply `exp(-σ_t × distance)` transmittance.

New material properties: `vec3 scattering_coefficient`, `float scattering_anisotropy` (Henyey-Greenstein `g ∈ [-1, 1]`). These map to the KHR_materials_volume extension's `attenuationColor` + `attenuationDistance` (already partially supported in Phase 8C) with an added scattering term.

### Heterogeneous Media (Phase 2)

Add ray marching through spatially varying density fields (3D textures or procedural volumes). This requires:

- **Delta tracking** (Woodcock tracking): Step through the volume with a majorant extinction coefficient, accept/reject scattering events based on the local-to-majorant ratio. Unbiased.
- **3D density texture binding:** Extend `MaterialDesc` with an optional density volume texture index.
- **Enter/exit medium tracking:** Use the Phase 8I IOR stack to know which volume the ray is currently traversing. The top of the stack determines the active medium's scattering properties.

Heterogeneous media is considerably more expensive (many ray march steps per volume traversal) and should only be applied to materials flagged with volumetric density textures.

---

## F5: Future Platform Denoisers

> **Denoising strategy:** The ML-trained denoiser (F11) is the product denoiser — deployed in Deni across all platforms. DLSS-RR (F1) is an NVIDIA-only, app-level quality reference used during development and ML training. NRD ReLAX (F16) is a deferred cross-vendor fallback for AMD/Intel GPUs if needed before the ML denoiser is ready. ReBLUR is not planned.

| Library | Platform | Implementation |
|---|---|---|
| `deni_vulkan` (ML) | Desktop + Mobile | ML-trained single-pass denoiser (the product denoiser, F11) |
| `deni_vulkan` (NRD ReLAX) | Desktop only | NRD ReLAX spatial-temporal filter (F16, deferred, cross-vendor fallback) |
| `deni_metal` | iOS / macOS | ML denoiser in Metal compute (C API for Swift) |
| `deni_webgpu` | Web | ML denoiser in WebGPU compute (C API for WASM/JS) |
| DLSS-RR (app-level) | NVIDIA only | DLSS 3.5 Ray Reconstruction in `monti_view` / `monti_datagen` (F1, quality reference) |

---

## F6: Mobile Vulkan Renderer (`monti_vulkan_mobile`)

Mobile GPUs (Qualcomm Adreno on Snapdragon 8 Gen 3+, ARM Mali Immortalis-G720+) support `VK_KHR_ray_query` and `VK_KHR_acceleration_structure` but **do not** support `VK_KHR_ray_tracing_pipeline`. There are no dedicated raygen/closesthit/miss shader stages on mobile — ray tracing is done via `rayQueryEXT` inside compute (or fragment) shaders.

**Approach:** The mobile renderer uses a **hybrid rasterization + ray query** pipeline to exploit TBDR tile memory for primary visibility while using ray tracing only for indirect effects:

1. **Rasterization G-buffer pass** (render pass, TBDR-friendly) — Rasterize the scene to produce primary visibility: depth, world normals, material ID, albedo, motion vectors. This goes through the tile pipeline at full TBDR efficiency. Multiple outputs stay in tile memory via subpass attachments or MRT (multi-render-target); only the final resolved images are written to DRAM.

2. **Ray trace compute pass** (compute, `rayQueryEXT`) — Reads the G-buffer, traces indirect bounces (diffuse GI, specular reflections, shadow rays) using `rayQueryEXT`. Writes noisy diffuse and noisy specular to storage images.

3. **Denoise pass** — ML denoiser (or passthrough) processes the noisy outputs.

This hybrid approach cuts ray tracing workload by ~40–60% (primary rays are cheapest to rasterize but most expensive to trace through the BVH) and gets full TBDR benefits for the G-buffer pass.

```cpp
// renderer/include/monti/vulkan/MobileRenderer.h
#pragma once
#include <monti/scene/Scene.h>
#include <vulkan/vulkan.h>
#include <memory>

namespace monti::vulkan::mobile {

struct RendererDesc {
    VkDevice         device;
    VkPhysicalDevice physical_device;
    VkQueue          queue;
    uint32_t         queue_family_index;
    VkPipelineCache  pipeline_cache = VK_NULL_HANDLE;  // Critical on mobile
    VmaAllocator     allocator      = VK_NULL_HANDLE;
    uint32_t         width             = 1920;
    uint32_t         height            = 1080;
    uint32_t         samples_per_pixel = 1;  // Lower default for mobile
};

class Renderer {
public:
    static std::unique_ptr<Renderer> Create(const RendererDesc& desc);
    ~Renderer();

    void SetScene(monti::Scene* scene);
    void RegisterMeshBuffers(MeshId mesh, const MeshBufferBinding& binding);
    void NotifyMeshDeformed(MeshId mesh, bool topology_changed = false);
    bool RenderFrame(VkCommandBuffer cmd, const GBuffer& output,
                     uint32_t frame_index);
    void SetSamplesPerPixel(uint32_t spp);
    uint32_t GetSamplesPerPixel() const;
    void Resize(uint32_t width, uint32_t height);
    float LastFrameTimeMs() const;
};

} // namespace monti::vulkan::mobile
```

**Shared code:** `GpuScene`, `GeometryManager`, `EnvironmentMap`, `BlueNoise`, and all GLSL include files (`brdf.glsl`, `sampling.glsl`, `mis.glsl`, etc.) are identical between desktop and mobile renderers. The GLSL material evaluation and BRDF code is shared between the desktop closesthit shader and the mobile ray trace compute shader.

**Key differences from desktop:**

| Aspect | Desktop (`monti_vulkan`) | Mobile (`monti_vulkan_mobile`) |
|--------|--------------------------|--------------------------------|
| Extension | `VK_KHR_ray_tracing_pipeline` | `VK_KHR_ray_query` + `VK_KHR_acceleration_structure` |
| Primary visibility | Ray traced (raygen shader) | **Rasterized** (render pass, TBDR tile memory) |
| Indirect bounces | `traceRayEXT` (SBT dispatch) | `rayQueryEXT` (compute) |
| Shader structure | raygen + closesthit + miss + anyhit | Rasterization vertex/fragment + uber-compute for indirect |
| Material branching | SBT per-geometry hit shader | Dynamic `switch` on material type |
| G-buffer formats | Compact formats (see GBuffer struct) | Same compact formats |
| Default SPP | 4 | 1 |
| Max bounces | 4 + 8 transparency | 2 + 4 transparency (tunable) |
| Denoiser | Passthrough → ML (DLSS-RR in monti_view for dev) | Passthrough → ML (no ReLAX) |

### G-Buffer Formats

G-buffer images are **format-agnostic at the API level** — the renderer and denoiser accept `VkImageView` and use `shaderStorageImageReadWithoutFormat` / `shaderStorageImageWriteWithoutFormat` to read and write images in whatever format the host allocated. This means the host is free to use either compact or RGBA16F formats for any channel. No shader permutations, no API changes, no format negotiation.

The **recommended compact formats** provide 32% bandwidth savings and are the default in our app:

| Image | Recommended Format | Bytes/pixel | RGBA16F Alternative | Notes |
|---|---|---|---|---|
| `noisy_diffuse` | `VK_FORMAT_R16G16B16A16_SFLOAT` | 8 | (same) | HDR radiance, needs full FP16 range |
| `noisy_specular` | `VK_FORMAT_R16G16B16A16_SFLOAT` | 8 | (same) | HDR radiance, needs full FP16 range |
| `motion_vectors` | `VK_FORMAT_R16G16_SFLOAT` | 4 | RGBA16F (8 B) | Only .xy used |
| `linear_depth` | `VK_FORMAT_R16G16_SFLOAT` | 4 | RGBA16F (8 B) | .r = depth, .g = hit distance (Phase 8E) |
| `world_normals` | `VK_FORMAT_R16G16B16A16_SFLOAT` | 8 | (same) | .xyz normal + .w roughness |
| `diffuse_albedo` | `VK_FORMAT_R16G16B16A16_SFLOAT` | 8 | (same) | Diffuse reflectance; zero-copy to denoiser |
| `specular_albedo` | `VK_FORMAT_R16G16B16A16_SFLOAT` | 8 | (same) | F0 reflectance; zero-copy to denoiser |
| **Total** | | **48** | | **Uniform FP16 — no format conversions** |

Both `shaderStorageImageReadWithoutFormat` and `shaderStorageImageWriteWithoutFormat` are universally supported on Vulkan 1.2+ desktop GPUs and on all mobile GPUs that support ray query (Adreno 740+, Mali-G720+). The renderer and denoiser require these features at device creation time.

### TBDR and Mobile Pipeline

Mobile GPUs use Tile-Based Deferred Rendering (TBDR), where render pass attachments stay in fast on-chip tile memory and only the final resolved output is written to DRAM. The compute-only ray tracing passes cannot use tile memory (storage images always go to DRAM). The hybrid approach maximizes TBDR benefits:

- **Rasterization G-buffer pass:** Depth, normals, material ID, albedo, and motion vectors are produced as render pass color/depth attachments. On TBDR hardware, these stay in tile memory during the pass and are resolved to DRAM only once at the end. Camera jitter is applied as a sub-pixel offset to the projection matrix (identical to TAA jitter in rasterization engines), providing temporal anti-aliasing accumulation across frames. At 1 SPP on mobile, per-frame projection jitter is equivalent to per-pixel ray jitter — both see exactly one sample per pixel per frame.
- **Ray trace compute pass:** Reads the resolved G-buffer from DRAM; traces indirect bounces (diffuse GI, specular reflections, shadow rays) using `rayQueryEXT`. Writes noisy radiance to storage images. This pass has no tile memory benefit.
- **Fragment shader denoiser:** The ML denoiser implementation uses a fullscreen fragment shader instead of compute on mobile. This reads G-buffer inputs as texture samples (through the texture cache) and writes denoised output as a render pass color attachment — keeping the output in tile memory for a subsequent tone-map subpass. The chain (denoise → tone map → present) executes as subpasses within a single render pass, with intermediate results never touching DRAM. This is invisible to the caller — `Denoise(cmd, input)` returns `DenoiserOutput` regardless of whether the implementation uses compute or fragment shaders internally. On desktop, the compute path is used (no TBDR benefit from fragment). On mobile, the fragment path is preferred. The `Denoiser` selects the implementation automatically based on device properties.

### Mobile Shader Best Practices

- **FP16 throughout.** Mobile denoiser and renderer shaders should use `float16_t` (via `GL_EXT_shader_explicit_arithmetic_types_float16`) or `mediump` for ALU operations. Mobile GPUs typically have 2× FP16 throughput vs FP32. Only subpixel motion reprojection needs FP32 precision.
- **Workgroup size.** Use 8×8 workgroups for mobile compute shaders (vs 16×16 on desktop). Mobile GPUs have smaller wavefronts (Adreno: 64, Mali: 16) and benefit from smaller workgroup tiles that fit shader register files.

### Mobile Complications

1. **Uber-shader complexity.** The ray trace compute shader combines all material evaluation, miss handling, and transparency logic. This is larger than the desktop pipeline's per-stage shaders (more register pressure, possible occupancy loss). Shader specialization constants can reduce this per-material if profiling shows it's needed.
2. **No SBT dispatch.** Per-geometry hit shader selection via SBT is unavailable. Material branching is dynamic — the compute shader loads the material index from the hit and branches to the correct BRDF evaluation path.
3. **Performance budget.** Mobile GPUs lack dedicated RT cores (or have minimal acceleration). Expect 1–2 SPP at interactive rates vs 4+ on desktop. The denoiser becomes even more critical on mobile.
4. **VkPipelineCache importance.** Pipeline compilation on mobile drivers is significantly slower than desktop (50–500ms per pipeline). On mobile, `pipeline_cache` should be treated as effectively required. The denoiser and renderer should log a warning if `pipeline_cache` is `VK_NULL_HANDLE` on a mobile device.
5. **Acceleration structure build cost on mobile.** BLAS building on mobile GPUs (Adreno, Mali) is significantly slower than on desktop NVIDIA/AMD hardware with dedicated RT cores. On desktop, building 50 BLAS for topology-changed meshes in a single frame is practical; on mobile, the same workload can consume the entire frame budget (~16ms). Mitigation strategies:
   - **Amortize across frames.** When many meshes change topology simultaneously (e.g., LOD transitions, batch spawning), spread BLAS rebuilds across multiple frames. `RenderFrame()` can prioritize: build N highest-priority BLAS per frame, reuse stale BLAS for the rest. Priority can be based on screen-space size, distance to camera, or frame count since last rebuild.
   - **Prefer refit over rebuild.** For continuous deformation (animation, cloth), BLAS refit is 2–5× faster than full rebuild on mobile. Accept slightly looser BVH quality in exchange for meeting frame budget.
   - **Async compute overlap.** On mobile GPUs that support async compute queues (some Adreno 700+), BLAS/TLAS builds can run on a compute queue overlapped with the previous frame's fragment work. This hides latency but requires careful synchronization.
   - **Budget parameter (future).** If profiling shows this is a real bottleneck, add an optional `max_blas_builds_per_frame` parameter to the mobile `RendererDesc`. `RenderFrame()` would then limit how many dirty BLAS it processes per call, deferring the rest to subsequent frames. This is not needed for the initial implementation — measure first.
6. **Render resolution.** Mobile should default to a lower internal render resolution (720p or 540p) with the ML denoiser providing super-resolution upscaling to the display resolution via `ScaleMode::kPerformance` (2×) or `ScaleMode::kQuality` (1.5×). Rendering at 540p and upscaling to 1080p cuts ray tracing cost by 4× and G-buffer memory by 4×.

---

## F7/F8: Future Renderers

| Renderer | Platform | Strategy |
|---|---|---|
| `monti_metal` | iOS / macOS | Metal RT + `MTLAccelerationStructure` (C API) |
| `monti_webgpu` | Web | Rasterize depth → Hi-Z → screen-space ray march (C API → WASM) |

---

## F14: GPU Skinning + Morph Targets

> **Prerequisite:** Phase 6 (GeometryManager with `NotifyMeshDeformed()` and BLAS refit support).

GPU-driven skeletal animation and morph target blending, removing the requirement for the host application to perform vertex deformation on the CPU before uploading.

### Joint Matrix Compute Shader

A compute shader transforms joint matrices from local bone space to world space using the skeleton hierarchy. Input: array of local joint transforms + parent indices. Output: array of world-space joint matrices in an SSBO. This runs once per skeleton per frame.

### Vertex Skinning Compute Shader

Reads the bind-pose vertex buffer and joint matrices SSBO. For each vertex, accumulates `Σ(weight_i × jointMatrix_i)` (up to 4 joints per vertex, matching glTF `JOINTS_0` / `WEIGHTS_0` attributes). Writes deformed position + normal + tangent to a staging vertex buffer. Workgroup size: 256 threads, one vertex per thread.

### Morph Target Blending

Before or combined with skinning, apply morph target deltas: `position += Σ(morph_weight_i × delta_position_i)`. Morph targets are stored as delta buffers (position + normal + tangent offsets). The blend weights are provided per frame by the host via `Scene::SetMorphWeights(MeshId, span<float>)`.

### BLAS Integration

After vertex deformation, call `GeometryManager::NotifyMeshDeformed(mesh_id, false)` to trigger a BLAS **refit** (not rebuild) for the deformed mesh. Refit is 2–5× faster than rebuild and acceptable for continuous animation where topology doesn't change. The existing Phase 6 BLAS refit path handles this.

### Host Interface

```cpp
// scene/include/monti/scene/Scene.h additions
void SetSkeleton(MeshId mesh, std::span<const glm::mat4> inverse_bind_matrices,
                 std::span<const int32_t> parent_indices);
void SetJointTransforms(MeshId mesh, std::span<const glm::mat4> local_transforms);
void SetMorphWeights(MeshId mesh, std::span<const float> weights);
```

The renderer detects dirty skeletons/morph weights per frame and dispatches the skinning compute shader before TLAS build.

---

## F15: ReSTIR GI (Indirect Illumination Reuse)

> **Prerequisite:** F2 (ReSTIR DI complete).

ReSTIR GI extends the reservoir framework to cache and reuse **secondary bounce** illumination. While ReSTIR DI resamples direct light sources, ReSTIR GI resamples indirect lighting paths:

- At each primary hit, trace a secondary ray and record the hit point + incoming radiance as a "GI sample."
- Store GI samples in per-pixel reservoirs (similar to F2's light reservoirs but storing path segments instead of light indices).
- Temporal resampling: merge with previous frame's GI reservoir at the reprojected location.
- Spatial resampling: gather neighbor GI reservoirs with Jacobian-corrected resampling weights.

ReSTIR GI dramatically improves indirect illumination quality at low SPP by amortizing expensive secondary bounces across frames and pixels. It is the primary technique for achieving real-time global illumination quality comparable to offline rendering.

Memory cost: ~32 bytes/pixel for GI reservoirs (double-buffered: ~128 MB at 4K). The temporal/spatial passes add ~2–3 ms at 1080p.
