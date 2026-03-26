#include <monti/capture/Luminance.h>

#include <cmath>
#include <cstddef>

namespace monti::capture {

LuminanceResult ComputeLogAverageLuminance(
    const float* diffuse_f32, const float* specular_f32, uint32_t pixel_count) {
    constexpr float kEpsilon = 1e-6f;
    constexpr float kLumaR = 0.2126f;
    constexpr float kLumaG = 0.7152f;
    constexpr float kLumaB = 0.0722f;

    double log_sum = 0.0;
    uint32_t nan_count = 0;
    uint32_t valid_count = 0;

    for (uint32_t i = 0; i < pixel_count; ++i) {
        auto base = static_cast<size_t>(i) * 4;
        float r = diffuse_f32[base + 0] + specular_f32[base + 0];
        float g = diffuse_f32[base + 1] + specular_f32[base + 1];
        float b = diffuse_f32[base + 2] + specular_f32[base + 2];
        float L = kLumaR * r + kLumaG * g + kLumaB * b;

        if (!std::isfinite(L)) {
            ++nan_count;
            continue;
        }

        log_sum += std::log(std::max(L, kEpsilon));
        ++valid_count;
    }

    float log_average = (valid_count > 0)
        ? static_cast<float>(std::exp(log_sum / valid_count))
        : 0.0f;

    return {log_average, nan_count, pixel_count};
}

}  // namespace monti::capture
