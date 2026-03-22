#ifndef LIGHT_SAMPLING_GLSL
#define LIGHT_SAMPLING_GLSL

#include "wrs.glsl"

// ── Select one light via weighted reservoir sampling ─────────────
// Uses Wang hash per light for decorrelated random values.
// Returns a reservoir with sample_count == 0 if light_count == 0.
Reservoir selectLight(vec3 shading_pos, vec3 N,
                      uint light_count, uint random_seed) {
    Reservoir r;
    initReservoir(r);
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
    return r;
}

#endif // LIGHT_SAMPLING_GLSL
