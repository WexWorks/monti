# Test Output Image Reference

This directory contains diagnostic images produced by the Monti path tracer test suite. Each image captures the output of a specific rendering test, enabling visual verification of features beyond what automated pixel checks can catch.

All tests render at 256×256 unless noted otherwise. Images are tone-mapped (Reinhard) from HDR for PNG output. Multi-frame tests accumulate samples across frames using Halton jitter for anti-aliasing.

---

## Running the Tests

Build and run from the project root:

```bash
# Build the test executable (Release)
cmake --build build --config Release --target monti_tests

# Run all tests directly (CWD must be the project root)
build\Release\monti_tests.exe

# Run tests via CTest (from the build directory)
cd build
ctest --output-on-failure

# Run a specific test phase
build\Release\monti_tests.exe "[phase8b]"

# Run a single named test
build\Release\monti_tests.exe "Phase 8B: Multi-bounce renderer produces non-trivial diffuse and specular channels"

# Run with coverage (Debug build + OpenCppCoverage)
python run_coverage.py
```

Test output images are written to `tests/output/` relative to the working directory. When running the executable directly from the project root, images land in this directory. When running via CTest (which sets `WORKING_DIRECTORY` to `build/`), images land in `build/tests/output/`.

---

## How Tests Verify Rendered Images

Most tests that produce output images do **not** compare against a gilded reference. Instead, they use a combination of statistical analysis, perceptual difference metrics, and structural assertions to verify correctness. Only the golden reference tests (at the end of this document) perform full perceptual comparison against stored reference images.

### Statistical Analysis (`AnalyzeRGBA16F`)

The primary verification tool is `AnalyzeRGBA16F()`, which scans every pixel in an RGBA16F readback buffer and returns a `PixelStats` struct:

| Field | Type | Meaning |
|-------|------|---------|
| `nan_count` | `uint32_t` | Number of pixels containing NaN in any RGB channel |
| `inf_count` | `uint32_t` | Number of pixels containing Inf in any RGB channel |
| `nonzero_count` | `uint32_t` | Number of pixels where R+G+B > 0 |
| `has_color_variation` | `bool` | True if any two adjacent pixels differ in R by more than 0.001 |
| `sum_r`, `sum_g`, `sum_b` | `double` | Accumulated channel sums (for mean luminance) |
| `valid_count` | `uint32_t` | Total non-NaN, non-Inf pixels |

Tests assert against these stats at varying levels of strictness:

- **Numerical validity** — Nearly every rendering test asserts `nan_count == 0` and `inf_count == 0`. This catches math errors in shaders (division by zero, negative square roots, uninitialized values).
- **Non-trivial content** — Tests assert `nonzero_count > kPixelCount / 4` (at least 25% of pixels are lit). This catches complete rendering failures like missing light sources, broken acceleration structures, or shaders that output all zeros.
- **Color variation** — `has_color_variation == true` ensures the output is not a flat uniform color. This catches cases where the renderer "works" but produces a constant image (e.g., a solid grey frame from incorrect tonemapping).
- **Channel-level luminance** — Some tests compute mean luminance from `sum_r/g/b` to verify brightness relationships (e.g., a lit scene is at least 2× brighter than a dark control, or doubling radiance approximately doubles output luminance).

### Perceptual Difference (FLIP)

For tests that compare two rendered images (e.g., low-SPP vs high-SPP, feature-on vs feature-off), the suite uses the **FLIP** perceptual image difference metric via `ComputeMeanFlip()`. FLIP measures perceived differences accounting for human spatial and color sensitivity. Both images are first tone-mapped (Reinhard) to [0,1] before comparison.

Typical assertions:
- **Convergence tests**: `mean_flip < 0.5` (low-SPP should not be wildly different from high-SPP)
- **Feature visibility**: `mean_flip > 0.02` (enabling a feature must produce a visible difference)
- **Golden references**: `mean_flip < 0.05–0.08` (rendered output must match stored reference)
- **Energy equivalence**: `mean_flip < 0.15` (two code paths for the same scene should agree)

### Structural and A/B Assertions

Beyond aggregate statistics, some tests perform targeted structural checks:

- **Hue preservation** (8E firefly clamping): Scans bright pixels (luminance > 10) and verifies the R > G > B ordering is preserved in ≥70% of them — confirms the luminance-weighted clamp preserves color ratios rather than flattening to white.
- **Variance comparison** (8G soft shadows): Renders the same scene with different sphere light radii and asserts `flip(large, small) > 0.02` — the images must be visibly different.
- **NEE noise reduction** (8J emissive extraction): Compares variance with and without NEE, asserting `variance_with_NEE < variance_without × 0.7`.
- **Region analysis** (8D normal maps, 8F LOD): Computes per-region pixel variance (e.g., left half vs right half, near vs far) and asserts the expected relationship (normal-mapped side has more variance than flat side, near region has more detail than far region).
- **Edge brightening** (8M sheen): Measures edge-region luminance variance and asserts the sheen version exceeds the baseline by a multiplier.

### What Is _Not_ Verified

- **No pixel-level comparison** — Individual pixel values are never compared against expected values. Stochastic path tracing produces different noise patterns each run, making per-pixel checks brittle.
- **No fixed tolerance thresholds per pixel** — All assertions are statistical (aggregates over the full image or sub-regions) or perceptual (FLIP over the full image pair).
- **Most output images are for human review only** — The PNG files written to this directory exist for visual inspection. The automated assertions catch broad failures (NaN, darkness, uniformity), but subtle rendering bugs (wrong color bleeding, incorrect refraction angle, missing texture detail) can only be caught by looking at the images.

---

## Phase 8B — Diffuse/Specular Channel Separation

### `cornell_box_8b_diffuse.png` / `cornell_box_8b_specular.png` / `cornell_box_8b_combined.png`
**Feature tested:** Multi-bounce diffuse and specular channel split rendering in the classic Cornell Box scene, accumulated over 1024 SPP (64 frames × 16 SPP each) using GPU compute accumulation.

**What it should look like:** The diffuse image shows a well-lit Cornell Box — white floor/ceiling/back wall, red left wall, green right wall, two white boxes, and color bleeding between surfaces. The specular image is mostly dark since all Cornell Box materials are fully diffuse (roughness 1.0). Any non-zero specular contribution comes from multi-bounce light transport. The combined image adds both channels for a complete view.

**If broken:** A completely black diffuse image indicates bounced light is not propagating. If the red/green wall colors appear in the specular channel instead of the diffuse channel, the channel routing is wrong. NaN/Inf values (visible as white or black holes) indicate math errors in the path tracing accumulation.

**Key visual indicator:** Warm-tinted illumination from the ceiling light (radiance 17,12,4 — warm white), visible color bleeding from the red and green walls onto the white boxes and floor. At 1024 SPP the image should be clean with minimal noise.

---

## Phase 8C — Transparency and Alpha Modes

### `cornell_box_8c_transparent_combined.png` / `cornell_box_8c_transparent_diffuse.png` / `cornell_box_8c_transparent_specular.png`
**Feature tested:** Transparent materials (glass transmission + alpha blending) inside the Cornell Box, accumulated over 1024 SPP.

**What it should look like:** The Cornell Box with two additional panels: a wide glass panel (IOR 1.5, subtle blue tint, spanning X=0.15–0.85 at Z=0.65) positioned in front of both boxes, and a semi-transparent blue blend panel. The glass panel should show refracted views of objects behind it with a slight blue color shift from the tint and attenuation. The blend panel should show partial see-through with blended colors. The combined image adds diffuse and specular channels together.

**If broken:** If transparency is not working, the panels appear as opaque surfaces. If IOR handling is wrong, refraction angles will be incorrect. If the glass panel is too small or positioned behind the boxes, it won't visibly intersect the scene.

**Key visual indicator:** The wide glass panel spans most of the scene width, with both boxes visible through it with refraction distortion and a subtle blue tint. Channel split shows Fresnel reflections in specular and transmitted light in diffuse.

---

## Phase 8D — PBR Materials and Textures

### `damaged_helmet_8d_combined.png` / `damaged_helmet_8d_diffuse.png` / `damaged_helmet_8d_specular.png`
**Feature tested:** Full PBR material pipeline with the Khronos DamagedHelmet model — base color, normal maps, metallic-roughness maps, and emissive maps all applied simultaneously.

**What it should look like:** The DamagedHelmet viewed from its default camera angle, showing battle-worn metallic surfaces with varying roughness. Surface details from normal maps should create the appearance of dents and scratches. Emissive visor elements should glow. The specular channel should show strong metallic reflections, while diffuse captures the non-metallic areas.

**If broken:** A smooth, featureless helmet means normal maps are not being applied. Uniform shininess means the metallic-roughness map is not being read. No glow on the visor means emissive textures are not contributing. If all surface detail is missing but the silhouette is correct, texture sampling is failing entirely.

**Key visual indicator:** Visible surface damage detail (bumps, scratches from normal maps), varying reflectivity across the surface (metallic-roughness), and glowing visor elements (emissive).

### `metallic_roughness_8d_combined.png`
**Feature tested:** Metallic-roughness material response in isolation, accumulated over 1024 SPP. Lit by a directional HDR environment map with a sky gradient and a warm sun spot at elevation ~55°.

**What it should look like:** An object with clearly varying metallic and roughness properties. The directional environment creates distinct highlights and shadows — metallic areas should show tinted reflections of the sky/sun, while rough areas scatter reflections into broad highlights. The sun spot provides a strong directional cue for evaluating material response.

**Key visual indicator:** Sharp reflections on smooth-metallic regions transitioning to blurry highlights on rough regions. The directional lighting from the sun spot creates visible specular highlights that differ between metallic and non-metallic areas.

### `normal_map_8d_combined.png`
**Feature tested:** Normal mapping applied to geometry, accumulated over 1024 SPP. Lit by a directional HDR environment map with a sky gradient and warm sun spot.

**What it should look like:** A surface where the lighting response reveals bumps and details that are not present in the actual geometry. The directional sun lighting makes the normal map perturbation clearly visible through shading variation — the bumped side should show more luminance variance than the flat side.

**If broken:** The surface appears perfectly smooth and flat — the normal map is being ignored or sampled incorrectly. With the directional environment, the normal-mapped side should show clearly more variation than the flat side.

**Key visual indicator:** Visible lighting variation across the normal-mapped surface, with the directional sun producing distinct per-bump highlights and shadows.

### `emissive_8d_combined.png`
**Feature tested:** Emissive material contribution to the rendered output, accumulated over 1024 SPP.

**What it should look like:** Two quads side by side — the left quad is emissive pure red (factor 10,0,0) and appears self-illuminated with a strong red glow; the right quad is dark red and is only lit by the environment. The left half of the image should be noticeably brighter and distinctly red.

**If broken:** Both halves appear equally bright (emissive factor is being ignored) or both are dark (emissive is not being added to the output at all). If the emissive quad appears orange or yellow instead of red, the emissive factor has a non-zero green component that should be zero.

**Key visual indicator:** The left (emissive) quad is significantly brighter than the right (non-emissive) quad, with a clear luminance difference. The emissive color should be pure red with no orange or yellow tint.

---

## Phase 8E — Firefly Clamping, Hit Distance, and Linear Depth

### `cornell_8e_combined.png`
**Feature tested:** Rendering stability under extreme emission values (emissive strength 10,000).

**What it should look like:** A Cornell Box scene that should appear reasonably normal despite an extremely bright light source. Firefly clamping should prevent isolated ultra-bright pixels from dominating the image.

**If broken:** Without clamping, the image contains scattered pure-white "firefly" pixels — extremely bright isolated samples that have not been averaged down by neighboring samples.

**Key visual indicator:** A clean image without stray white hot-pixels, even though the light source radiance is extreme.

### `cornell_8e_hit_distance.png`
**Feature tested:** Hit distance output stored in the linear depth buffer's green channel.

**What it should look like:** A visualization where pixel brightness corresponds to the distance from the camera to the first surface hit. Near surfaces (the front-facing box walls) appear darker, while the back wall and far surfaces appear brighter. The ceiling light area may show distinct distance values.

**If broken:** Uniform brightness across the image (hit distance is constant or zero), or chaotic noise (distance values are corrupted).

**Key visual indicator:** A smooth depth gradient that matches the scene geometry — near objects darker, far objects brighter, with sharp transitions at object edges.

### `cornell_8e_linear_depth.png`
**Feature tested:** Linear depth storage in the R channel of the RG16F buffer.

**What it should look like:** Similar to the hit distance but encoding the linearized camera-space depth. Objects closer to the camera appear darker, with smooth gradients across surfaces facing the camera.

**If broken:** All black (depth not being written) or all white (depth is saturated), or visible quantization bands (precision issues).

---

## Phase 8F — Ray Cone LOD (Mip Level Selection)

### `checker_close_far_8f_combined.png`
**Feature tested:** Ray cone-driven texture LOD comparison — a high-frequency emissive checkerboard rendered on two quads at different distances from the camera. The close quad is on the left, the far quad on the right.

**What it should look like:** The left (close) quad shows a sharp, high-contrast checkerboard pattern. The right (far) quad shows the same pattern but blurred/averaged by higher mip level selection due to the larger ray cone at distance. The close side should have noticeably higher pixel variance than the far side.

**If broken:** If LOD is always 0, both sides look equally sharp with aliasing artifacts on the far quad. If LOD is too high, both sides appear blurred. If the ray cone calculation is wrong, the LOD difference between near and far is not visible.

**Key visual indicator:** Clear difference in pattern sharpness between left (close, sharp) and right (far, blurred) quads.

### `ground_mip_8f_combined.png`
**Feature tested:** A ground plane textured with a fake MIP map where each mip level is a distinct solid color (red=mip0, green=mip1, blue=mip2, yellow=mip3, magenta=mip4, cyan=mip5, orange=mip6, white=mip7, gray=mip8). The ground plane extends from z=1 to z=-60, so the ray cone expands with distance.

**What it should look like:** Bands of distinct colors across the ground plane, transitioning from red (mip 0, near camera) through green, blue, yellow, etc. as the surface recedes. Each color band corresponds to a mip level, making the LOD selection directly visible.

**If broken:** If only mip 0 is used, the entire ground plane is red. If the mip chain is not loaded correctly, colors don't match the expected per-level assignment. If LOD jumps erratically, the color bands are not smooth transitions.

**Key visual indicator:** Distinct color bands on the ground plane showing progressive mip level transitions from near (red) to far (higher mip colors).

---

## Phase 8G — Area Lights (Sphere and Triangle Lights)

### `phase8g_sphere_light.png`
**Feature tested:** Sphere light illumination in the Cornell Box at 1024 SPP. A sphere emitter (center near ceiling, radius 0.1, warm radiance 50,40,25) replaces the standard area light.

**What it should look like:** The Cornell Box illuminated by a point-like spherical light near the ceiling. Compared to the standard ceiling quad light, the sphere light produces slightly different shadow patterns — rounder, more point-like shadows. The warm-toned radiance gives the scene a slightly golden appearance.

**If broken:** Complete darkness (sphere light NEE sampling failed) or extremely noisy (PDF is wrong, causing high-variance estimates).

**Key visual indicator:** The scene is well-lit with warm-tinted shadows that have soft, round penumbras consistent with a small spherical emitter.

### `phase8g_sphere_large.png` / `phase8g_sphere_small.png`
**Feature tested:** Soft shadow variation with sphere light radius at 1024 SPP. A medium sphere (radius 0.15, warm radiance 40,35,25, Z=0.35) produces wider, softer shadows than a tiny sphere (radius 0.01, warm radiance 200,180,120, Z=0.35). Both are positioned further back from the camera to avoid geometry intersections.

**What it should look like:** The large-radius image shows soft, diffused shadows. The small-radius image shows sharper, more defined shadows closer to hard point-light shadows. The small sphere has higher radiance to compensate for its smaller solid angle, so both scenes should be similarly well-lit overall.

**If broken:** If both images look identical, the sphere radius is not affecting shadow sampling. If the large sphere intersects scene geometry, rendering artifacts appear.

**Key visual indicator:** Visibly different shadow softness between the two images, especially around the box edges on the floor. Both images have warm-tinted lighting.

### `phase8g_triangle_light.png`
**Feature tested:** Triangle light illumination with two-sided emission at 1024 SPP. A single emissive triangle (radiance 12,9,3 — warm tint, reduced from earlier values to control fireflies) mounted slightly below the ceiling (Y=0.98, avoiding z-fighting) illuminates the Cornell Box.

**What it should look like:** The Cornell Box illuminated by a triangular emitter. The illumination pattern differs from both the quad and sphere lights — a triangle produces an asymmetric light pattern. The warm tint (heavier red, lighter blue) should be visible. The lower radiance compared to sphere lights reduces colored firefly artifacts.

**If broken:** If the triangle light doesn't contribute, the scene is dark. If the triangle is at Y=0.999 (ceiling level), z-fighting with the ceiling plane can cause visible artifacts.

**Key visual indicator:** Warm-tinted illumination with an asymmetric light distribution pattern. No z-fighting artifacts at the ceiling.

### `phase8g_quad_light_compat.png`
**Feature tested:** Backward compatibility — the original quad ceiling light (from `AddCornellBoxLight`) still works correctly after the area light system was extended to support sphere and triangle lights.

**What it should look like:** The standard Cornell Box rendering with the canonical ceiling quad light. This should look essentially identical to the Phase 8B Cornell Box renders — the same warm illumination (radiance 17,12,4), the same shadow patterns.

**If broken:** If the quad light path broke during the sphere/triangle light implementation, this image will be dark or look fundamentally different from the established Cornell Box baseline.

**Key visual indicator:** Warm ceiling light illumination, color bleeding on walls and boxes — should match the canonical Cornell Box appearance.

### `phase8g_mixed_lo.png` / `phase8g_mixed_hi.png`
**Feature tested:** Convergence with all three light types active simultaneously (quad + sphere + triangle) at 1024 SPP. Low SPP (64) vs high SPP (1024).

**What it should look like:** A Cornell Box lit by three different light sources. The low-SPP image is noisy. The high-SPP image should converge to a clean result. Multiple overlapping soft shadows from the different light sources should be visible. Triangle light is at Y=0.98 to avoid ceiling z-fighting.

**If broken:** If the multi-light sampling is biased, the high-SPP image will converge to the wrong brightness or color. If FLIP between low and high SPP is > 0.25, convergence is too slow.

**Key visual indicator:** The high-SPP image is a clean version of the noisy low-SPP image, with complex shadow interplay from the three light sources.

---

## Phase 8H — Diffuse Transmission and Thin Surfaces

### `phase8h_backlit_leaf_dt.png` / `phase8h_backlit_leaf_opaque.png`
**Feature tested:** Diffuse transmission — light passing through a thin surface and scattering on the back side, simulating a backlit leaf. The transmissive version has `dt_factor=0.8` and `dt_color=(0.2, 0.8, 0.1)` (green leaf color).

**What it should look like:** The `_dt` image shows a quad (leaf) between the camera and a light source. Light passes through the leaf and is tinted green by the diffuse transmission color — the leaf glows with a green backlit appearance. The `_opaque` image shows the same scene but with `dt_factor=0.0` — the leaf blocks all light and appears as a dark silhouette against the lit background.

**If broken:** If diffuse transmission is not working, both images look identical (both opaque). If the transmission color is ignored, light passes through but without green tinting.

**Key visual indicator:** The transmissive image has a distinctly green, glowing quality on the leaf surface, while the opaque version shows the leaf as a dark shadow.

### `phase8h_thin_surface.png` / `phase8h_thick_surface.png`
**Feature tested:** Thin-surface transmission vs thick-surface (volumetric) refraction. A glass panel is placed between the camera and colored walls.

**What it should look like:** With `thin_surface=true`, light passes straight through the glass panel without offset — objects behind the glass appear at their correct positions. With `thin_surface=false` (thick glass, IOR 1.5), light refracts at both entry and exit surfaces — objects behind the glass appear shifted/distorted.

**If broken:** If the thin-surface flag is ignored, both images look identical — either both refract or both pass straight through.

**Key visual indicator:** In the thick-surface image, objects seen through the glass appear offset or distorted compared to the thin-surface image where they are in their correct positions.

### `phase8h_red_tint.png` / `phase8h_blue_tint.png`
**Feature tested:** Diffuse transmission color tinting. Two renders of a backlit leaf with different `dt_color` values: red (1,0,0) and blue (0,0,1).

**What it should look like:** The red-tint image shows the backlit leaf glowing red. The blue-tint image shows the backlit leaf glowing blue. The transmitted light should take on the color specified by `dt_color`.

**If broken:** If color tinting is not applied, both images look the same neutral color. If the color channels are swapped, the red image appears blue and vice versa.

**Key visual indicator:** The dominant color of the backlit glow matches the specified `dt_color` — clearly red in one image and clearly blue in the other.

### `phase8h_convergence_high.png`
**Feature tested:** Convergence of diffuse transmission under multi-frame accumulation (1024 SPP total). Validates that the MIS (Multiple Importance Sampling) integration for transmission converges properly.

**What it should look like:** A clean, noise-free rendering of the backlit leaf scene. This high-SPP image is the converged reference.

**If broken:** If the transmission sampling PDF is wrong, the image remains noisy even at high SPP, or converges to an incorrect brightness.

### `phase8h_spec_only.png` / `phase8h_spec_plus_dt.png`
**Feature tested:** Specular contribution isolation with and without diffuse transmission.

**What it should look like:** `_spec_only` shows only the specular (reflective) component of the material. `_spec_plus_dt` adds diffuse transmission to show the combined effect. The transmission adds light behind the surface that the specular-only version lacks.

**Key visual indicator:** The combined image is brighter in the transmitted region than the specular-only image.

### `phase8h_no_nan.png`
**Feature tested:** Numerical stability — no NaN/Inf values produced by diffuse transmission math under edge conditions.

**What it should look like:** A valid rendered image without any corrupted pixels.

---

## Phase 8I — Nested Dielectrics

### `phase8i_glass_in_glass.png` / `phase8i_glass_alone.png`
**Feature tested:** Nested dielectric IOR mediation — an inner glass sphere (IOR 1.5) enclosed within an outer sphere (IOR 1.33, like water). The interior list tracks IOR transitions at each surface crossing.

**What it should look like:** The `_glass_in_glass` image shows stronger refraction effects because light transitions between three media (air→water→glass→water→air). The `_glass_alone` image shows only the inner sphere with simpler air→glass→air transitions. The double-refraction in the nested case produces more distorted views of the background.

**If broken:** If the interior list is not working, the nested case looks identical to the isolated case — the outer medium's IOR is ignored.

**Key visual indicator:** More pronounced refraction distortion in the nested case compared to the single-sphere case.

### `phase8i_false_intersection_both.png` / `phase8i_false_intersection_outer_only.png`
**Feature tested:** False intersection rejection in the nested dielectric system. An inner sphere with higher priority is enclosed within an outer sphere — the inner sphere should be invisible (rejected by the nesting rules).

**What it should look like:** Both images should look nearly identical — the inner sphere is a "false intersection" that gets skipped during IOR mediation. The output should show only the outer sphere's refraction.

**If broken:** If false intersection rejection is not working, the inner sphere is visible as a distinct object inside the outer sphere, making the images look different (FLIP > 0.03).

**Key visual indicator:** Both images appear the same — the inner sphere is invisible, demonstrating correct priority-based rejection.

### `phase8i_stack_overflow.png`
**Feature tested:** Graceful handling when the interior list overflows. Eight concentric transmissive spheres exceed the stack capacity (`kInteriorListSlots=2`).

**What it should look like:** The image should render without crashing or producing NaN/Inf. Visual quality may degrade (incorrect IOR transitions after overflow) but the renderer should handle it gracefully.

**If broken:** GPU hang, crash, or NaN/Inf artifacts indicate the overflow is not being handled.

**Key visual indicator:** A valid, non-corrupted image — visual accuracy is secondary, stability is the goal.

### `phase8i_thin_bypass_thick.png` / `phase8i_thin_bypass_thin.png`
**Feature tested:** Thin-surface materials bypass the interior list system. A thin-surface inner sphere inside a thick outer sphere should behave differently from a thick inner sphere.

**What it should look like:** The thin-surface version shows the inner sphere transmitting light without entering the IOR stack, while the thick version correctly participates in nested dielectric tracking.

**Key visual indicator:** Visible difference between the two renders (FLIP > 0.02), demonstrating that the thin-surface flag correctly bypasses nesting logic.

---

## Phase 8J — Emissive Mesh Extraction for NEE

### `phase8j_nee_extracted.png` / `phase8j_nee_no_extraction.png`
**Feature tested:** Automatic extraction of emissive mesh triangles as explicit lights for Next-Event Estimation (NEE). An emissive panel (quad = 2 triangles) is placed in the Cornell Box.

**What it should look like:** Both images show the same scene, but the extracted version (NEE enabled) is noticeably brighter and less noisy. With extraction, the renderer explicitly samples directions toward the emissive panel, dramatically reducing variance. Without extraction, the renderer only finds the emissive surface through random path hits.

**If broken:** If extraction doesn't work, both images look equally noisy and dim. If the extraction double-counts energy (adds NEE without compensation), the extracted version will be too bright.

**Key visual indicator:** The NEE-extracted image is brighter with significantly less noise than the non-extracted version, especially in areas illuminated by the emissive panel.

### `phase8j_flip_4spp.png` / `phase8j_flip_64spp.png`
**Feature tested:** FLIP convergence with emissive mesh extraction — comparing 16 SPP to 64 SPP. Both use extraction.

**What it should look like:** The 4-SPP image is noisy but captures the scene structure. The 64-SPP image is cleaner. The convergence rate should be good thanks to NEE sampling of the emissive mesh.

**Key visual indicator:** Clear noise reduction from low to high SPP, with FLIP < 0.3.

### `phase8j_mixed_lights.png`
**Feature tested:** Mixed explicit area lights and extracted emissive mesh lights coexisting in the same scene.

**What it should look like:** A Cornell Box lit by both the canonical ceiling quad light and an emissive panel. The scene should be well-illuminated from multiple sources without energy duplication or missing light contributions.

**If broken:** If the light lists conflict, some lights may be sampled twice (too bright) or skipped (too dark). NaN from conflicting light type handling would appear as corrupted pixels.

**Key visual indicator:** A well-lit scene combining warm ceiling light and emissive panel illumination, with no NaN artifacts.

---

## Phase 8K — Weighted Reservoir Sampling (Many-Light Rendering)

### `phase8k_direct_single.png`
**Feature tested:** Single-light direct sampling path unchanged after adding WRS support. One area light in the Cornell Box.

**What it should look like:** The standard Cornell Box lit by the ceiling light. Should be identical to earlier Cornell Box renders — this confirms the WRS code does not affect the direct-sample path for small light counts.

**Key visual indicator:** Standard Cornell Box illumination, warm tint, no regression.

### `phase8k_direct_few.png`
**Feature tested:** Direct sampling path with 3 lights (1 quad + 1 sphere + 1 triangle) — still below the WRS threshold.

**What it should look like:** A Cornell Box lit by three diverse light sources. Multiple overlapping shadow patterns should be visible.

**Key visual indicator:** Complex multi-light illumination similar to the Phase 8G mixed light test.

### `phase8k_wrs_4spp.png` / `phase8k_wrs_16spp.png` / `phase8k_wrs_64spp.png`
**Feature tested:** WRS convergence with 50 procedural sphere lights — well above the direct-sample threshold, triggering the WRS code path.

**What it should look like:** A Cornell Box lit by many small sphere lights distributed around the scene. The 4-SPP image is very noisy. The 16-SPP image shows the scene structure more clearly. The 64-SPP image should be relatively clean. The convergence from noisy to clean demonstrates that WRS is an unbiased estimator.

**If broken:** If WRS is biased, the high-SPP image converges to incorrect brightness. If the sampling PDF is poor, convergence is very slow (FLIP > 0.3 between low and high).

**Key visual indicator:** Progressive noise reduction with increasing SPP, converging to a scene lit by many small lights.

### `phase8k_wrs_10lights.png` / `phase8k_wrs_200lights.png`
**Feature tested:** Sublinear performance scaling of WRS. Rendering with 10 lights vs 200 lights should not take 20× longer — WRS samples a fixed number of candidates regardless of total light count.

**What it should look like:** Both images show lit Cornell Boxes. The 200-light version has denser, more uniform illumination. The key test is performance, not visual appearance.

**If broken:** If the renderer falls back to iterating all lights (O(N) instead of O(1) WRS), the 200-light render takes dramatically longer than the 10-light render (ratio > 3.0).

**Key visual indicator:** Both renders complete in reasonable time. The 200-light scene has more uniform, omnidirectional illumination.

### `phase8k_wrs_210lights.png`
**Feature tested:** Extreme light count tolerance — 210 lights (100 sphere + 10 quad + 100 triangle) running through the WRS path in a single-frame render.

**What it should look like:** A Cornell Box lit by a very large number of diverse light types. The single SPP makes it noisy, but the image should be valid (no NaN/Inf, many lit pixels).

**Key visual indicator:** A noisy but valid image with illumination from many sources. No corrupted pixels.

### `phase8k_equiv_direct.png` / `phase8k_equiv_wrs.png`
**Feature tested:** Energy equivalence between direct-sample and WRS code paths. Both render the same 4-light scene at high SPP (256 total), but the WRS path is forced by adding a zero-radiance dummy light to push the count above the threshold.

**What it should look like:** Both images should look nearly identical — same brightness, same color, same shadow patterns. The only difference is which internal code path computed the result.

**If broken:** If the WRS PDF compensation is wrong, the WRS image will be brighter or darker than the direct-sample image. A luminance ratio outside 0.85–1.15 indicates energy is being created or lost.

**Key visual indicator:** The two images are visually indistinguishable (FLIP < 0.15).

### `phase8k_wrs_energy.png`
**Feature tested:** WRS energy conservation — output included as part of the energy proportionality test.

### `phase8k_prop_1x.png` / `phase8k_prop_2x.png`
**Feature tested:** WRS energy proportionality — 10 lights at radiance (3,3,3) vs (6,6,6). Doubling radiance should double the output luminance.

**What it should look like:** The `_2x` image is approximately twice as bright as the `_1x` image. Both show the same scene geometry and light distribution.

**If broken:** If the WRS probability weights are not proportional to radiance, the luminance ratio deviates from 2.0 (test allows 1.7–2.3).

**Key visual indicator:** The `_2x` image is noticeably brighter than `_1x`, approximately double.

---

## Phase 8L — Texture Transforms

### `phase8l_tiling_1x.png` / `phase8l_tiling_4x.png`
**Feature tested:** UV scale transform — an emissive checkerboard texture rendered at 1× and 4× tiling.

**What it should look like:** The 1× image shows one full repetition of the checkerboard pattern. The 4× image shows 16 tiles (4×4) of the same pattern in the same screen area — higher spatial frequency, smaller individual squares.

**If broken:** If texture transforms are ignored, both images show the same 1× pattern. If the scale is applied incorrectly (e.g., only on one axis), the checkerboard becomes stretched rectangles.

**Key visual indicator:** The 4× image has visibly more pattern repetitions and smaller individual checker squares than the 1× image.

### `phase8l_identity.png`
**Feature tested:** Identity texture transform passthrough on the DamagedHelmet model — verifying that models without `KHR_texture_transform` render correctly through the early-out path.

**What it should look like:** The DamagedHelmet rendered normally, identical to the Phase 8D/8F helmet renders. No texture distortion or offset.

**If broken:** If the identity path is wrong, textures appear shifted, scaled, or rotated incorrectly on the helmet.

**Key visual indicator:** Standard DamagedHelmet appearance with all PBR textures correctly applied.

### `phase8l_rotation_none.png` / `phase8l_rotation_90.png`
**Feature tested:** UV rotation transform. A horizontal red-green gradient texture is rendered with no rotation and with 90° rotation.

**What it should look like:** The unrotated image shows a horizontal gradient (left-to-right color change). The rotated image shows a vertical gradient (top-to-bottom or bottom-to-top color change) — the same gradient pattern but rotated 90°.

**If broken:** If rotation is ignored, both images show the same horizontal gradient. If the rotation angle is wrong, the gradient direction is neither horizontal nor vertical.

**Key visual indicator:** The gradient direction changes by 90° between the two images.

### `phase8l_tiling_high_spp.png`
**Feature tested:** High-SPP rendering of the tiled checkerboard texture for convergence verification.

**What it should look like:** A clean, noise-free checkerboard pattern with sharp edges.

---

## Phase 8M — Sheen (KHR_materials_sheen)

### `phase8m_sheen_visible.png` / `phase8m_no_sheen.png`
**Feature tested:** Charlie sheen BSDF on a fabric-like sphere. The sheen version has `sheen_color=(0.8,0.8,0.8)` and `sheen_roughness=0.5`.

**What it should look like:** The sheen image shows a white sphere with a subtle bright rim/edge highlight — the "sheen" effect that gives fabrics their characteristic soft glow at grazing angles. The no-sheen image shows the same sphere with standard diffuse-only response, lacking the edge brightening.

**If broken:** If sheen is not implemented, both images look identical. If the sheen Charlie NDF has numerical issues, the edge region may show artifacts or excessive brightness.

**Key visual indicator:** The sheen image shows a distinctive edge/rim brightening around the sphere's silhouette that the plain version lacks.

### `phase8m_sheen_furnace.png`
**Feature tested:** Sheen energy conservation under a white furnace test (sphere in a uniform white environment). A perfectly energy-conserving material should not create energy (mean luminance ≤ 1.05).

**What it should look like:** A white sphere in a white environment. The sphere should be uniformly bright but not brighter than the environment — no glowing edges.

**If broken:** If sheen creates energy, the sphere's edges appear brighter than the surrounding environment (luminance > 1.05), indicating the BSDF is not properly energy-conserving.

**Key visual indicator:** The sphere blends smoothly with the white environment. No visible energy gain at edges.

### `phase8m_no_sheen_helmet.png`
**Feature tested:** The DamagedHelmet rendered with zero sheen values — verifying the zero-sheen code path produces correct output without NaN/Inf.

**What it should look like:** A standard DamagedHelmet render identical to previous helmet outputs.

**Key visual indicator:** Normal helmet appearance, no artifacts from the sheen code path.

### `phase8m_blue_sheen.png`
**Feature tested:** Sheen color tinting — blue sheen `(0,0,1)` on a sphere.

**What it should look like:** A sphere with a distinctly blue-tinted edge highlight. The center of the sphere appears normal, but the rim/grazing angles have a blue glow.

**If broken:** If sheen color is ignored, the edge looks white (neutral) instead of blue.

**Key visual indicator:** Blue-tinted edge brightening, with the blue/red ratio higher than a non-sheened baseline.

### `phase8m_sheen_extreme.png` / `phase8m_sheen_high_spp.png`
**Feature tested:** Numerical stability under extreme sheen roughness values (0.001 and 1.0) and high-SPP convergence.

**What it should look like:** Valid renders without NaN artifacts. The extreme roughness values are edge cases for the Charlie NDF that could produce singularities if not guarded.

**Key visual indicator:** Clean images without corrupted pixels, even at extreme parameter values.

---

## Phase 8N — DDS Texture Loading (Block Compression)

### `phase8n_bc7.png`
**Feature tested:** BC7-compressed DDS texture loading and rendering. A 64×64 BC7 texture with 7 mip levels is applied to a quad.

**What it should look like:** A uniform orange/amber colored quad (RGB ~255,128,0) in the standard studio environment (grey back wall, grey floor, area light). The procedural BC7 texture is a solid orange color at mip 0, so the quad appears as a flat uniform color with lighting shading across its surface. BC7 compression is nearly lossless, so no compression artifacts are visible.

**If broken:** If DDS loading fails, the quad appears as a solid default color (untextured). If BC7 block decoding is wrong, the quad shows garbled blocks or incorrect colors instead of uniform orange.

**Key visual indicator:** Uniform orange/amber quad surface with smooth lighting gradient — no texture pattern, just a solid color from the BC7 mip 0 data.

### `phase8n_bc1.png`
**Feature tested:** BC1-compressed DDS texture loading (4 bits per pixel, lossy compression).

**What it should look like:** A uniform solid red quad (RGB ~255,0,0) in the studio environment. The procedural BC1 texture is a solid red color at mip 0. Despite BC1's lower quality (4:1 compression), a uniform solid color encodes perfectly, so no compression artifacts should be visible. Lighting shading creates a gentle gradient across the surface.

**If broken:** If DDS loading fails, the quad appears as a default untextured surface. If BC1 decoding is wrong, blocks of incorrect colors or garbled data appear.

**Key visual indicator:** Uniform solid red quad with smooth lighting, distinct from the orange BC7 test.

### `phase8n_bc5_normal.png`
**Feature tested:** BC5-compressed DDS texture used as a normal map on a lit icosphere. BC5 stores two channels (RG) with the Z component reconstructed mathematically. Rendered at 64 SPP × 16 frames for clean convergence.

**What it should look like:** A lit grey sphere with a visible checkerboard-faceted surface. The procedural BC5 normal map alternates between two normal directions in a 4×4 block checkerboard pattern — blocks tilted one way and blocks tilted the other — creating a faceted, bumpy appearance under directional lighting. The sphere should show clear patches of lighter and darker shading corresponding to the alternating normal directions.

**If broken:** If BC5 decoding fails or Z-reconstruction is wrong, the sphere appears perfectly smooth (no normal map effect) or has chaotic/inverted lighting. If the checkerboard pattern is not visible, the normal map data may not be loading correctly.

**Key visual indicator:** Checkerboard-faceted surface detail on the sphere — alternating patches of different shading from the two normal directions. Not a smooth sphere.

### `phase8n_mip_chain.png`
**Feature tested:** Mip chain usage from a DDS texture with all 7 mip levels. A ground plane recedes into the distance, using progressively higher mip levels for farther regions.

**What it should look like:** A textured ground plane extending into the distance. Near regions show the mip 0 color (orange, RGB ~255,128,0), mid-distance transitions to the mip 1 color (teal, RGB ~0,128,255), and far regions show blended higher-mip colors (lime, magenta). The color transitions demonstrate automatic mip level selection based on distance. Each mip level in the procedural texture is a distinct solid color, making the transitions clearly visible.

**If broken:** If only mip 0 is used, the entire plane appears orange with aliasing artifacts in the distance. If the mip chain is loaded incorrectly, wrong colors appear at wrong distances or corrupt data is visible.

**Key visual indicator:** Clear color transitions from orange (near) through teal (mid) as the ground plane recedes — proving mip levels are loaded and selected correctly.

### `phase8n_bistro_cloth.png`
**Feature tested:** Real-world BC7 DDS texture (Cloth_BaseColor.dds, 2048×2048, 11 mips) loaded through the full LoadGltf → DecodeImage → DecodeDdsImage pipeline on a textured quad. Validates that production BC-compressed textures load and render correctly via glTF.

**What it should look like:** A quad textured with the Bistro cloth base color — a fabric pattern with subtle color and detail. Rendered at 16 SPP × 8 frames in the standard studio environment with a computed default camera. The texture should appear clean and properly colored.

**If broken:** If the DDS file fails to load or parse, the quad appears untextured or the test fails. If the block compression format is misidentified, the texture appears garbled.

**Key visual indicator:** A recognizable fabric/cloth texture pattern on the quad, correctly decoded from a real production DDS file.

### `phase8n_bistro_normal.png`
**Feature tested:** Real-world BC5 DDS normal map (Cloth_Normal.dds, 2048×2048, 11 mips) loaded through the full LoadGltf → DecodeImage → DecodeDdsImage pipeline on a UV sphere. Validates that production BC5 normal maps produce correct surface perturbation via glTF.

**What it should look like:** A grey sphere with visible surface detail from the cloth normal map — fabric weave or textile bumps that modulate the lighting across the sphere surface. Rendered at 64 SPP × 16 frames.

**If broken:** If the normal map fails to load, the sphere appears perfectly smooth. If the BC5 format or Z-reconstruction is wrong, the surface perturbation appears chaotic or inverted.

**Key visual indicator:** Visible fabric/textile normal-mapped detail on the sphere surface, producing realistic lighting variation.

---

## Phase 9B — Denoiser Integration

### `phase9b_raw_combined.png` / `phase9b_denoised.png`
**Feature tested:** Denoiser pipeline integration (passthrough mode). The raw image is the path tracer output (diffuse + specular combined). The denoised image has passed through the denoiser barrier and passthrough denoise step.

**What it should look like:** Both images should be **identical** — the passthrough denoiser performs no filtering, so the output should match the input within ±1 ULP (Unit of Least Precision). Both show a noisy Cornell Box at 4 SPP.

**If broken:** If the denoiser introduces artifacts, the denoised image will differ from the raw image. A FLIP > 0.001 between them indicates the passthrough path is modifying data.

**Key visual indicator:** The two images are pixel-identical. Any visible difference indicates a pipeline bug.

---

## Phase 10A — Tone Mapping (ACES)

### `phase10a_ldr_range.png`
**Feature tested:** Tone-mapped output is within LDR range [0,1]. The ACES filmic curve compresses HDR values into displayable range.

**What it should look like:** A Cornell Box that appears natural with no clipped highlights — the tone-mapping curve should gracefully roll off bright values rather than hard-clipping to white.

**If broken:** Hard-clipped white regions (no tone compression), or values outside [0,1] causing display artifacts.

**Key visual indicator:** Smooth highlight rolloff, natural-looking illumination without hard white patches.

### `phase10a_exposure_ev0.png` / `phase10a_exposure_ev2.png`
**Feature tested:** Exposure parameter affects tone-mapped output. EV=0 is baseline; EV=+2 brightens the image by 4× before tone mapping.

**What it should look like:** The EV+2 image is noticeably brighter than the EV0 image. Shadows that were dark in EV0 become visible in EV+2, while highlights in EV+2 are pushed more toward white (compressed by ACES).

**If broken:** If both images look identical, the exposure parameter is being ignored.

**Key visual indicator:** Clear brightness difference between the two exposures.

### `phase10a_hdr_clamp.png`
**Feature tested:** ACES highlight compression at extreme exposure (EV+4). Demonstrates that the filmic curve compresses highlights rather than hard-clipping.

**What it should look like:** A bright, overexposed-looking Cornell Box where the ACES curve rolls off highlights smoothly. Less than 20% of pixels should be fully saturated — the film curve should preserve some detail even in bright regions.

**If broken:** If hard-clipping instead of ACES, most bright areas are pure white (>20% saturated pixels).

**Key visual indicator:** Smooth, filmic highlight rolloff rather than hard white clipping.

### `phase10a_golden_cornell_box.png` / `phase10a_golden_damaged_helmet.png`
**Feature tested:** End-to-end golden reference — full pipeline (render → denoise → tonemap) at 256 SPP. These are the final quality reference images.

**What it should look like:** Clean, noise-free, properly tone-mapped renders. The Cornell Box should show the canonical scene with accurate color bleeding, warm light, and proper shadow detail. The DamagedHelmet should show full PBR detail with correct metallic reflections and emissive elements.

**Key visual indicator:** Publication-quality renders with no noise, NaN artifacts, or tone-mapping errors.

---

## Golden Reference Tests

### `golden_Box.png` / `golden_Box_test.png`
**Feature tested:** High-SPP (1024) golden reference for the Box.glb model with the test rendering compared against the stored reference.

**What it should look like:** A simple box lit by a grey environment. The `_test.png` should match the golden `Box.png` within a FLIP threshold of 0.05.

### `golden_CornellBox.png` / `golden_CornellBox_test.png`
**Feature tested:** Golden reference for the Cornell Box scene at 1024 SPP.

**What it should look like:** The definitive Cornell Box rendering — white floor/ceiling/back, red left wall, green right wall, two white boxes, warm ceiling light. Color bleeding visible on all surfaces. The test image should match the golden reference within FLIP 0.05.

### `golden_DamagedHelmet.png` / `golden_DamagedHelmet_test.png`
**Feature tested:** Golden reference for the Khronos DamagedHelmet at 1024 SPP. Tests the full PBR pipeline end-to-end.

**What it should look like:** The battle-worn helmet with full texture detail — normal-mapped dents, varying metallic-roughness, emissive visor glow. FLIP threshold: 0.08 (complex scene).

### `golden_DragonAttenuation.png` / `golden_DragonAttenuation_test.png`
**Feature tested:** Golden reference for the DragonAttenuation model — tests volumetric attenuation through a transmissive dragon model.

**What it should look like:** A dragon figure where light attenuates as it passes through the volume — thicker parts appear darker/more saturated, thinner parts are more transparent. This validates the attenuation distance and color absorption implementation.

**Key visual indicator:** Thickness-dependent color absorption — thin features like wings/edges are lighter/more transparent than the thick body.

### `golden_ClearCoatTest.png` / `golden_ClearCoatTest_test.png`
**Feature tested:** Golden reference for KHR_materials_clearcoat. Tests the clear-coat layer over a base material.

**What it should look like:** Spheres or objects with a glossy clear-coat layer over a base material. The clear-coat adds a sharp specular reflection layer on top of the underlying material's response. Roughness variations in the clear coat should be visible.

**Key visual indicator:** Dual-layer reflections — a sharp clear-coat highlight over a broader or tinted base reflection.

### `golden_MorphPrimitivesTest.png` / `golden_MorphPrimitivesTest_test.png`
**Feature tested:** Golden reference for glTF morph target (shape key) rendering.

**What it should look like:** Geometric primitives with morph targets applied — the shapes should be deformed from their base mesh according to the morph weights. FLIP threshold: 0.05 (simple geometry).

**Key visual indicator:** Correct deformed geometry that matches the morph target specification.

---

## Firefly Clamping Tests

### `firefly_clamp_diffuse.png` / `firefly_clamp_specular.png`
**Feature tested:** Firefly clamping effectiveness on diffuse and specular channels independently.

**What it should look like:** Clean images without isolated ultra-bright pixels. The clamping should suppress high-energy outlier samples while preserving the overall image appearance.

**If broken:** Scattered white "firefly" pixels throughout the image indicate the clamp is not engaged or the threshold is too high.

---

## Environment Map Test

### `box_glb_envmap.png`
**Feature tested:** Environment map lighting on a glTF model.

**What it should look like:** The Box.glb model lit entirely by the environment map, showing reflections and ambient illumination from the surrounding environment.

**Key visual indicator:** Visible environment lighting and reflections on the box surfaces.
