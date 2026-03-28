#ifndef LIGHT_SAMPLING_GLSL
#define LIGHT_SAMPLING_GLSL

#include "wrs.glsl"

// ── Select one light via weighted reservoir sampling ─────────────
// Uses Wang hash per light for decorrelated random values.
// Returns a reservoir with sample_count == 0 if light_count == 0.
//
// When light_count <= kMaxWRSLights the full O(N) WRS scan runs,
// weighting each light by its estimated contribution.
//
// When light_count > kMaxWRSLights a uniform-random light is
// selected in O(1).  This is unbiased — the PDF is simply
// 1/light_count — but noisier than importance-weighted WRS.
// ReSTIR (F2/F3) will replace this path with spatiotemporal
// resampling for high-quality many-light convergence.
Reservoir selectLight(vec3 shading_pos, vec3 N,
                      uint light_count, uint random_seed) {
    Reservoir r;
    initReservoir(r);

    if (light_count <= kMaxWRSLights) {
        // Full WRS scan — O(light_count)
        for (uint i = 0u; i < light_count; ++i) {
            uint base = i * kLightStride;
            vec4 d0 = lights.data[base + 0u];
            vec4 d1 = lights.data[base + 1u];
            vec4 d2 = lights.data[base + 2u];
            vec4 d3 = lights.data[base + 3u];
            float weight = estimateLightContribution(
                d0, d1, d2, d3, shading_pos, N);
            uint hash = wangHash(random_seed ^ i);
            float rand_i = float(hash) / 4294967295.0;
            updateReservoir(r, i, weight, rand_i);
        }
    } else {
        // Uniform random selection — O(1)
        uint hash = wangHash(random_seed);
        uint selected = hash % light_count;
        uint base = selected * kLightStride;
        vec4 d0 = lights.data[base + 0u];
        vec4 d1 = lights.data[base + 1u];
        vec4 d2 = lights.data[base + 2u];
        vec4 d3 = lights.data[base + 3u];
        float weight = estimateLightContribution(
            d0, d1, d2, d3, shading_pos, N);
        r.selected_light  = selected;
        r.selected_weight  = 1.0;
        r.weight_sum       = float(light_count);
        r.sample_count     = 1u;
    }

    return r;
}

#endif // LIGHT_SAMPLING_GLSL
