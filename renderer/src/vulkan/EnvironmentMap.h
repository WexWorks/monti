#pragma once

#include "Buffer.h"
#include "Image.h"

#include <cstdint>
#include <vector>

namespace monti::vulkan {

// Statistics computed during CDF construction for the environment map.
// Used by push constants/uniforms for MIS weight computation in shaders.
struct EnvironmentStatistics {
    float average_luminance = 0.0f;
    float max_luminance = 0.0f;
    float max_raw_luminance = 0.0f;
    float luminance_variance = 0.0f;
    float solid_angle_weighted_luminance = 0.0f;
    float total_luminance = 0.0f;
};

// GPU resources for an HDR equirectangular environment map with
// importance sampling CDFs. Owns three VkImages:
//   - Environment map texture (RGBA16F, full mip chain, linear sampler)
//   - Marginal CDF (height×1, R32F, nearest sampler)
//   - Conditional CDF (width×height, R32F, nearest sampler)
//
// If no environment map is set, holds 1×1 black placeholders so
// descriptor sets remain valid without null descriptors.
class EnvironmentMap {
public:
    EnvironmentMap() = default;
    ~EnvironmentMap() = default;

    // Creates 1×1 black placeholders for env map, marginal/conditional CDFs.
    // Must be called at initialization before any rendering.
    // Staging buffers are appended to staging_out and must be kept alive until cmd completes.
    bool CreatePlaceholders(VmaAllocator allocator, VkDevice device,
                            VkCommandBuffer cmd,
                            std::vector<Buffer>& staging_out);

    // Load environment map from RGBA float pixel data (already decoded by the app).
    // Computes CDFs, generates mipmaps, and uploads all GPU resources.
    // Staging buffers are appended to staging_out and must be kept alive until cmd completes.
    bool Load(VmaAllocator allocator, VkDevice device,
              VkCommandBuffer cmd,
              const float* rgba_data, uint32_t width, uint32_t height,
              std::vector<Buffer>& staging_out);

    const Image& EnvTexture() const { return env_texture_; }
    const Image& MarginalCdfTexture() const { return marginal_cdf_texture_; }
    const Image& ConditionalCdfTexture() const { return conditional_cdf_texture_; }
    const EnvironmentStatistics& Statistics() const { return statistics_; }

    uint32_t Width() const { return width_; }
    uint32_t Height() const { return height_; }
    bool IsLoaded() const { return loaded_; }

private:
    Image env_texture_;
    Image marginal_cdf_texture_;
    Image conditional_cdf_texture_;
    EnvironmentStatistics statistics_;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    bool loaded_ = false;
};

}  // namespace monti::vulkan
