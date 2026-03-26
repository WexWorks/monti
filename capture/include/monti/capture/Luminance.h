#pragma once

#include <cstdint>

namespace monti::capture {

struct LuminanceResult {
    float log_average;     // exp(mean(log(L + epsilon)))
    uint32_t nan_count;    // Number of NaN/Inf pixels
    uint32_t total_pixels; // Total pixel count
};

// Compute log-average luminance from separate diffuse and specular FP32 buffers.
// Each buffer has 4 floats per pixel (RGBA). Only RGB is used for luminance.
LuminanceResult ComputeLogAverageLuminance(
    const float* diffuse_f32,
    const float* specular_f32,
    uint32_t pixel_count);

}  // namespace monti::capture
