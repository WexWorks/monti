#ifndef WRS_GLSL
#define WRS_GLSL

// ── Weighted Reservoir Sampling (single reservoir) ───────────────
// Streaming WRS: iterate N items maintaining a reservoir of size 1.
// Each item is accepted with probability weight / weight_sum.
// After iterating all items, the selected item has probability
// proportional to its weight.

struct Reservoir {
    uint  selected_light;   // index of currently selected light
    float selected_weight;  // weight of the selected light
    float weight_sum;       // running sum of all weights
    uint  sample_count;     // number of non-zero-weight lights seen
};

void initReservoir(out Reservoir r) {
    r.selected_light  = 0u;
    r.selected_weight = 0.0;
    r.weight_sum      = 0.0;
    r.sample_count    = 0u;
}

// Streaming update: accept light_index with probability weight/weight_sum.
void updateReservoir(inout Reservoir r, uint light_index,
                     float weight, float random_val) {
    if (weight <= 0.0) return;
    r.weight_sum += weight;
    r.sample_count += 1u;
    if (random_val * r.weight_sum < weight) {
        r.selected_light  = light_index;
        r.selected_weight = weight;
    }
}

// WRS selection probability for the chosen light: weight_i / weight_sum.
float getReservoirPdf(Reservoir r) {
    if (r.weight_sum <= 0.0) return 0.0;
    return r.selected_weight / r.weight_sum;
}

#endif // WRS_GLSL
