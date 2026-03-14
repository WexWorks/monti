#include "EnvironmentMap.h"

#include "DeviceDispatch.h"
#include "Upload.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

#include <glm/gtc/packing.hpp>

namespace monti::vulkan {

namespace {

uint32_t CalculateMipLevels(uint32_t width, uint32_t height) {
    return static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
}

// Logarithmic luminance compression for super-bright pixels.
// Linear below reference_level, logarithmic above. Prevents a single
// bright pixel from dominating the CDF distribution.
float CompressLuminance(float luminance) {
    if (luminance <= 0.0f) return 0.0f;

    constexpr float reference_level = 1.0f;
    constexpr float compression_strength = 10.0f;

    if (luminance <= reference_level) return luminance;

    float excess = luminance - reference_level;
    float compressed = std::log1p(excess / compression_strength) * compression_strength;
    return reference_level + compressed;
}

struct CdfResult {
    std::vector<float> marginal_cdf;
    std::vector<float> conditional_cdf;
    EnvironmentStatistics statistics;
};

CdfResult ComputeEnvironmentCdf(const float* rgba_data, uint32_t width, uint32_t height) {
    const uint32_t pixel_count = width * height;
    std::vector<float> luminance_data(pixel_count);

    float total_luminance = 0.0f;
    float total_solid_angle = 0.0f;
    float solid_angle_weighted_luminance = 0.0f;
    float max_luminance = 0.0f;
    float max_raw_luminance = 0.0f;
    float sum_luminance_squared = 0.0f;

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            uint32_t pixel_index = (y * width + x) * 4;
            float r = rgba_data[pixel_index + 0];
            float g = rgba_data[pixel_index + 1];
            float b = rgba_data[pixel_index + 2];

            float raw_luminance = 0.299f * r + 0.587f * g + 0.114f * b;
            float luminance = CompressLuminance(raw_luminance);

            // cos(θ) weighting for equirectangular solid angles
            constexpr float pi = std::numbers::pi_v<float>;
            float theta = ((static_cast<float>(y) + 0.5f) / static_cast<float>(height) - 0.5f) * pi;
            float cos_theta = std::cos(theta);
            float weighted_luminance = luminance * cos_theta;

            float solid_angle = cos_theta * (pi / static_cast<float>(height)) *
                                (2.0f * pi / static_cast<float>(width));

            luminance_data[y * width + x] = weighted_luminance;
            total_luminance += weighted_luminance;
            total_solid_angle += solid_angle;
            solid_angle_weighted_luminance += luminance * solid_angle;
            max_luminance = std::max(max_luminance, luminance);
            max_raw_luminance = std::max(max_raw_luminance, raw_luminance);
            sum_luminance_squared += luminance * luminance;
        }
    }

    float average_luminance = total_luminance / static_cast<float>(pixel_count);
    float luminance_variance = (sum_luminance_squared / static_cast<float>(pixel_count)) -
                               (average_luminance * average_luminance);

    if (total_solid_angle > 0.0f)
        solid_angle_weighted_luminance /= total_solid_angle;

    if (total_luminance == 0.0f) {
        total_luminance = 1.0f;
        std::fill(luminance_data.begin(), luminance_data.end(),
                  1.0f / static_cast<float>(pixel_count));
    }

    // Marginal distribution (row sums)
    std::vector<float> marginal_pdf(height);
    for (uint32_t y = 0; y < height; ++y) {
        float row_sum = 0.0f;
        for (uint32_t x = 0; x < width; ++x)
            row_sum += luminance_data[y * width + x];
        marginal_pdf[y] = row_sum / total_luminance;
    }

    // Marginal CDF
    std::vector<float> marginal_cdf(height);
    marginal_cdf[0] = marginal_pdf[0];
    for (uint32_t y = 1; y < height; ++y)
        marginal_cdf[y] = marginal_cdf[y - 1] + marginal_pdf[y];

    // Conditional CDF for each row
    std::vector<float> conditional_cdf(static_cast<size_t>(width) * height);
    for (uint32_t y = 0; y < height; ++y) {
        float row_total = 0.0f;
        for (uint32_t x = 0; x < width; ++x)
            row_total += luminance_data[y * width + x];

        if (row_total == 0.0f) {
            float uniform = 1.0f / static_cast<float>(width);
            float running = 0.0f;
            for (uint32_t x = 0; x < width; ++x) {
                running += uniform;
                conditional_cdf[y * width + x] = running;
            }
        } else {
            float running = 0.0f;
            for (uint32_t x = 0; x < width; ++x) {
                running += luminance_data[y * width + x] / row_total;
                conditional_cdf[y * width + x] = running;
            }
        }
    }

    CdfResult result;
    result.marginal_cdf = std::move(marginal_cdf);
    result.conditional_cdf = std::move(conditional_cdf);
    result.statistics.average_luminance = average_luminance;
    result.statistics.max_luminance = max_luminance;
    result.statistics.max_raw_luminance = max_raw_luminance;
    result.statistics.luminance_variance = luminance_variance;
    result.statistics.solid_angle_weighted_luminance = solid_angle_weighted_luminance;
    result.statistics.total_luminance = total_luminance;
    return result;
}

std::vector<uint16_t> ConvertToHalf(const float* rgba_data, uint32_t pixel_count) {
    size_t total = static_cast<size_t>(pixel_count) * 4;
    std::vector<uint16_t> half_data(total);
    for (size_t i = 0; i < total; ++i)
        half_data[i] = glm::packHalf1x16(rgba_data[i]);
    return half_data;
}

}  // anonymous namespace

bool EnvironmentMap::CreatePlaceholders(VmaAllocator allocator, VkDevice device,
                                        VkCommandBuffer cmd,
                                        std::vector<Buffer>& staging_out,
                                        const DeviceDispatch& dispatch) {
    width_ = 1;
    height_ = 1;
    loaded_ = false;

    // 1×1 black RGBA16F env map
    if (!env_texture_.Create(allocator, device, dispatch, 1, 1,
                             VK_FORMAT_R16G16B16A16_SFLOAT,
                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        return false;
    if (!env_texture_.CreateSampler(VK_FILTER_LINEAR, VK_FILTER_LINEAR,
                                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f))
        return false;

    // 1×1 black R32F marginal CDF
    if (!marginal_cdf_texture_.Create(allocator, device, dispatch, 1, 1,
                                      VK_FORMAT_R32_SFLOAT,
                                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        return false;
    if (!marginal_cdf_texture_.CreateSampler(VK_FILTER_NEAREST, VK_FILTER_NEAREST,
                                             VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                             VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f))
        return false;

    // 1×1 black R32F conditional CDF
    if (!conditional_cdf_texture_.Create(allocator, device, dispatch, 1, 1,
                                         VK_FORMAT_R32_SFLOAT,
                                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        return false;
    if (!conditional_cdf_texture_.CreateSampler(VK_FILTER_NEAREST, VK_FILTER_NEAREST,
                                                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f))
        return false;

    // Upload black pixels via staging
    uint16_t black_half[4] = {0, 0, 0, 0};
    Buffer staging_env = upload::ToImage(allocator, cmd, env_texture_,
                                         black_half, sizeof(black_half), dispatch);
    if (staging_env.Handle() == VK_NULL_HANDLE) return false;
    staging_out.push_back(std::move(staging_env));

    float zero = 0.0f;
    Buffer staging_marginal = upload::ToImage(allocator, cmd, marginal_cdf_texture_,
                                              &zero, sizeof(float), dispatch);
    if (staging_marginal.Handle() == VK_NULL_HANDLE) return false;
    staging_out.push_back(std::move(staging_marginal));

    Buffer staging_cond = upload::ToImage(allocator, cmd, conditional_cdf_texture_,
                                          &zero, sizeof(float), dispatch);
    if (staging_cond.Handle() == VK_NULL_HANDLE) return false;
    staging_out.push_back(std::move(staging_cond));

    return true;
}

bool EnvironmentMap::Load(VmaAllocator allocator, VkDevice device,
                          VkCommandBuffer cmd,
                          const float* rgba_data, uint32_t width, uint32_t height,
                          std::vector<Buffer>& staging_out,
                          const DeviceDispatch& dispatch) {
    // Destroy old resources
    env_texture_.Destroy();
    marginal_cdf_texture_.Destroy();
    conditional_cdf_texture_.Destroy();

    width_ = width;
    height_ = height;
    loaded_ = true;

    // Compute CDFs for importance sampling
    auto cdf = ComputeEnvironmentCdf(rgba_data, width, height);
    statistics_ = cdf.statistics;

    // Convert RGBA32F → RGBA16F for the env map (half the memory of RGBA32F)
    uint32_t pixel_count = width * height;
    auto half_data = ConvertToHalf(rgba_data, pixel_count);

    // Create env map texture with full mip chain
    uint32_t mip_levels = CalculateMipLevels(width, height);
    VkDeviceSize env_size = static_cast<VkDeviceSize>(half_data.size()) * sizeof(uint16_t);

    if (!env_texture_.Create(allocator, device, dispatch, width, height,
                             VK_FORMAT_R16G16B16A16_SFLOAT,
                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT, mip_levels))
        return false;

    // Env map sampler: bilinear + trilinear mips, REPEAT horizontal, CLAMP vertical
    if (!env_texture_.CreateSampler(VK_FILTER_LINEAR, VK_FILTER_LINEAR,
                                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f))
        return false;

    // Upload base level (mip generation via upload::ToImage's built-in blit cascade)
    Buffer staging_env = upload::ToImage(allocator, cmd, env_texture_,
                                         half_data.data(), env_size, dispatch);
    if (staging_env.Handle() == VK_NULL_HANDLE) return false;
    staging_out.push_back(std::move(staging_env));

    std::fprintf(stderr, "Environment map loaded: %ux%u, %u mip levels\n", width, height, mip_levels);

    // Marginal CDF texture (1D: height×1, R32F)
    if (!marginal_cdf_texture_.Create(allocator, device, dispatch, height, 1,
                                      VK_FORMAT_R32_SFLOAT,
                                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
        return false;
    if (!marginal_cdf_texture_.CreateSampler(VK_FILTER_NEAREST, VK_FILTER_NEAREST,
                                             VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                             VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f))
        return false;

    Buffer staging_marginal = upload::ToImage(allocator, cmd, marginal_cdf_texture_,
                                              cdf.marginal_cdf.data(),
                                              cdf.marginal_cdf.size() * sizeof(float),
                                              dispatch);
    if (staging_marginal.Handle() == VK_NULL_HANDLE) return false;
    staging_out.push_back(std::move(staging_marginal));

    // Conditional CDF texture (2D: width×height, R32F)
    if (!conditional_cdf_texture_.Create(allocator, device, dispatch, width, height,
                                         VK_FORMAT_R32_SFLOAT,
                                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
        return false;
    if (!conditional_cdf_texture_.CreateSampler(VK_FILTER_NEAREST, VK_FILTER_NEAREST,
                                                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f))
        return false;

    Buffer staging_cond = upload::ToImage(allocator, cmd, conditional_cdf_texture_,
                                          cdf.conditional_cdf.data(),
                                          cdf.conditional_cdf.size() * sizeof(float),
                                          dispatch);
    if (staging_cond.Handle() == VK_NULL_HANDLE) return false;
    staging_out.push_back(std::move(staging_cond));

    std::fprintf(stderr, "Environment CDF computed: %ux%u\n", width, height);
    std::fprintf(stderr, "  Total luminance: %.4f\n", statistics_.total_luminance);
    std::fprintf(stderr, "  Average luminance: %.4f\n", statistics_.average_luminance);
    std::fprintf(stderr, "  Max compressed luminance: %.4f\n", statistics_.max_luminance);
    std::fprintf(stderr, "  Max raw luminance: %.4f\n", statistics_.max_raw_luminance);
    std::fprintf(stderr, "  Luminance variance: %.4f\n", statistics_.luminance_variance);
    std::fprintf(stderr, "  Solid angle weighted: %.4f\n", statistics_.solid_angle_weighted_luminance);

    return true;
}

}  // namespace monti::vulkan
