# Depth of Field — Implementation Plan

> **Feature:** Thin-lens depth of field for the Monti path tracer.
> **Approach:** Distributed ray tracing (Cook et al., 1984) — jitter primary ray origin on the aperture disk, aim at the focus plane. Zero additional ray bounces, ~50 lines of shader/C++ code for the core feature.
> **Dependencies:** None — can be implemented at any point. Existing `CameraParams` already defines `aperture_radius` and `focus_distance` (unused).

---

## Camera Lens Model

**Thin-lens model** with photographic parameterization:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `f_stop` | `float` | `0.0` | f-number (e.g., 1.4, 2.8, 5.6). `0.0` = pinhole (DoF disabled). |
| `focus_distance` | `float` | `10.0` | Distance to the in-focus plane (meters). |
| `aperture_blades` | `uint32_t` | `0` | Iris blade count for bokeh shape. `0` = circular. Deferred to Phase DoF-2. |

**f-stop to aperture radius conversion:**

$$r_{\text{aperture}} = \frac{f}{2N}$$

where $f$ is the focal length derived from FoV: $f = \frac{h_{\text{sensor}}}{2 \tan(\theta_v / 2)}$ with $h_{\text{sensor}} = 36\text{mm}$ (full-frame equivalent) and $\theta_v$ the vertical FoV.

When `f_stop == 0.0`, the thin-lens code path is skipped entirely (branch-free: `aperture_radius` is zero, so the offset is zero).

---

## Algorithm

In `raygen.rgen`, after computing the pinhole ray:

```glsl
// 1. Pinhole ray (existing code, unchanged)
vec3 pinhole_dir = normalize((frame.inv_view * vec4(normalize(target.xyz / target.w), 0.0)).xyz);
vec3 pinhole_origin = (frame.inv_view * vec4(0.0, 0.0, 0.0, 1.0)).xyz;

// 2. Focus point on the focal plane
float ft = frame.focus_distance / dot(pinhole_dir, cam_forward);
vec3 focus_point = pinhole_origin + pinhole_dir * ft;

// 3. Sample aperture disk (concentric disk mapping, blue noise driven)
vec2 lens_uv = concentricDiskSample(bn_xi0, bn_xi1) * frame.aperture_radius;

// 4. Offset ray origin in camera-local space
vec3 cam_right = frame.inv_view[0].xyz;
vec3 cam_up    = frame.inv_view[1].xyz;
vec3 dof_origin = pinhole_origin + cam_right * lens_uv.x + cam_up * lens_uv.y;
vec3 dof_dir    = normalize(focus_point - dof_origin);
```

**G-buffer writes use the pinhole ray** — depth, normals, and motion vectors are computed from `pinhole_origin`/`pinhole_dir` so that denoiser reprojection and edge-stopping remain geometrically consistent. Radiance tracing uses `dof_origin`/`dof_dir`.

---

## Implementation Phases

### Phase DoF-1: Core Thin-Lens DoF (Circular Aperture)

**Scope:** Add thin-lens DoF to the renderer and expose f-stop / focus distance in `monti_view` UI.

#### Changes

**1. `CameraParams` (scene/include/monti/scene/Camera.h)**
- Replace `aperture_radius` with `f_stop` (float, default `0.0` = pinhole)
- Keep `focus_distance` (already present)
- Add `FocalLength()` helper: derives focal length from `vertical_fov_radians` and a 36mm sensor equivalent
- Add `ApertureRadius()` helper: computes `focal_length / (2 * f_stop)`, returns `0.0` when `f_stop == 0.0`

**2. `FrameUniforms` (renderer/src/vulkan/FrameUniforms.h + shaders/include/frame_uniforms.glsl)**
- Replace `pad0` with `float aperture_radius` (offset 228)
- Replace `pad1` with `float focus_distance` (offset 232)
- `pad2` remains for std140 16-byte alignment
- Total size stays 240 bytes
- Static assert unchanged

**3. `Renderer.cpp` — FrameUniforms population**
- Set `fu.aperture_radius = camera.ApertureRadius()`
- Set `fu.focus_distance = camera.focus_distance`

**4. `raygen.rgen` — Thin-lens ray perturbation**
- Add `concentricDiskSample()` utility (in `include/sampling.glsl` or inline — ~10 lines)
- After pinhole ray computation, add the thin-lens offset (~12 lines, see algorithm above)
- Use 2 blue noise dimensions for the aperture sample (consume from existing `bn_packed`)
- G-buffer primary writes (depth, normals, motion vectors) use pinhole ray
- Path tracing uses DoF-perturbed ray
- Ray cone spread should account for circle of confusion: add `aperture_radius` contribution to initial `cone_spread`

**5. `monti_view` UI (app/view/Panels.cpp + AppState)**
- Add to `AppState`: `float f_stop = 0.0f;` and `float focus_distance = 10.0f;`
- Add to Camera collapsing header:
  - `ImGui::SliderFloat("f-stop", &state.f_stop, 0.0f, 22.0f, "%.1f")` with tooltip: "0 = pinhole (no DoF)"
  - `ImGui::SliderFloat("Focus Distance", &state.focus_distance, 0.1f, 1000.0f, "%.1f m", ImGuiSliderFlags_Logarithmic)`
- Wire `AppState` f-stop/focus_distance into `CameraParams` before `RenderFrame()`

**6. `monti_datagen` — CLI arguments**
- Add `--f-stop` and `--focus-distance` CLI flags
- Default `0.0` (pinhole) preserves existing behavior
- When non-zero, populates `CameraParams` accordingly

#### Verification
- Pinhole behavior (f_stop = 0) is unchanged — bit-exact output
- f_stop > 0 produces visible defocus blur on out-of-focus objects
- Focus distance slider moves the sharp focal plane
- G-buffer depth/normals remain clean (pinhole-equivalent)
- Motion vectors remain correct for temporal reprojection
- Blue noise sampling produces well-distributed aperture patterns (no visible structure)

---

### Phase DoF-2: Polygonal Bokeh (Mechanical Aperture Blades)

**Scope:** Replace circular aperture sampling with regular polygon sampling for shaped bokeh highlights. Deferred until DoF-1 is validated.

**Prerequisite:** DoF-1 complete and tested.

#### Changes

**1. `CameraParams`**
- Add `uint32_t aperture_blades = 0;` (0 = circular, 5–9 = polygon)

**2. `FrameUniforms`**
- Replace remaining `pad2` with `uint aperture_blades` (offset 236)
- Size stays 240 bytes

**3. `raygen.rgen` — Polygon aperture sampling**
- Add `polygonalApertureSample(vec2 xi, uint blades)` function (~15 lines)
  - Maps uniform 2D random numbers to points uniformly distributed inside a regular N-gon
  - Falls back to concentric disk when `blades == 0`
  - Standard approach: pick a random triangle of the polygon fan, sample uniformly within it
- Replace `concentricDiskSample()` call with `polygonalApertureSample()`

**4. `monti_view` UI**
- Add `ImGui::SliderInt("Aperture Blades", &state.aperture_blades, 0, 18)` with tooltip: "0 = circular"
- Only visible when f_stop > 0 (collapsed otherwise)

#### Verification
- `aperture_blades = 0` produces identical output to DoF-1 (circular disk)
- `aperture_blades = 6` produces hexagonal bokeh highlights at high SPP (256+)
- `aperture_blades = 5` produces pentagonal bokeh
- Blade count smoothly affects bokeh shape from triangular (3) to nearly circular (18)

---

## Denoiser Interactions

### DLSS-RR (F1)
DLSS-RR handles DoF natively — it is trained on ray-traced content including thin-lens DoF. No special integration needed beyond providing the standard G-buffer (which uses pinhole-equivalent auxiliaries as described above). DoF regions converge well under DLSS-RR's temporal accumulation.

### ML Denoiser — Single-Frame (F9–F11-3)
The initial single-frame ML denoiser **cannot effectively denoise DoF regions at low SPP**. At 1–4 SPP, the spatially varying, large-radius noise from aperture sampling is indistinguishable from scene detail to a single-frame network. This is expected and documented:

- DoF at low SPP + single-frame ML denoiser = visible noise in bokeh regions
- Users should either increase SPP or use DLSS-RR for DoF content
- The single-frame network will not be specifically trained to handle DoF (it would need to learn to preserve blur while removing noise — conflicting objectives without temporal information)

### ML Denoiser — Temporal (F11-4/F11-5)
Temporal denoising is the correct solution for ML-denoised DoF. With 2–4 frames of history and motion vector warping, the temporal network accumulates aperture samples across frames, converging toward the true defocus distribution.

**No architecture changes required.** The temporal extension (F11-4) already plans to widen input channels with warped previous frames. DoF does not change the channel layout or network structure.

**Training data changes (F9-4/F9-6):**
- Include camera configurations with varying `f_stop` (1.4–16.0) and `focus_distance` in training scene generation
- Include both pinhole and DoF renders in the training set so the network generalizes to both
- Ground truth: high-SPP (256+) DoF renders, where aperture sampling has converged
- Input: low-SPP (1–4) DoF renders, where the bokeh is noisy
- No changes to model architecture, loss function, or training strategy

**Temporal training (F11-4) should explicitly include DoF sequences:**
- Camera motion with DoF enabled, varying focus distance
- Focus pulls (rack focus) between foreground and background objects
- This teaches the network temporal accumulation of aperture samples, not just spatial denoising

### Summary Table

| Denoiser | DoF Quality at 1–4 SPP | Notes |
|---|---|---|
| Passthrough | Noisy (no denoising) | Expected — high SPP required |
| Single-frame ML (F11-3) | Poor in bokeh regions | Conflicting blur/noise objectives |
| DLSS-RR (F1) | Good | Natively trained for DoF |
| Temporal ML (F11-5) | Good (expected) | Accumulates aperture samples over frames |

---

## Roadmap Placement

DoF is a standalone feature with no dependencies on Waves 1–4. It slots between the current wave structure as an independent item:

```
DoF-1 (core thin-lens)  →  DoF-2 (polygonal bokeh)
                                ↓
                        (training data includes DoF)
                                ↓
                        F9-4/F9-6 (data generation with DoF scenes)
```

DoF-1 can be implemented immediately. DoF-2 is deferred until DoF-1 is validated. Training data generation (F9-4/F9-6) should include DoF scenes after DoF-1 is merged.
