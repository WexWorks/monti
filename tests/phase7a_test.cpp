#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <volk.h>

#include "../app/core/vulkan_context.h"
#include "../app/core/gbuffer_images.h"

#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>

// Internal headers for direct testing
#include "../renderer/src/vulkan/EnvironmentMap.h"
#include "../renderer/src/vulkan/BlueNoise.h"
#include "../renderer/src/vulkan/Buffer.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace monti;
using namespace monti::vulkan;

namespace {

struct TestContext {
    monti::app::VulkanContext ctx;

    bool Init() {
        if (!ctx.CreateInstance()) return false;
        if (!ctx.CreateDevice(std::nullopt)) return false;
        return true;
    }
};

// Generate a simple synthetic HDR environment: a gradient from dark
// at poles to bright at equator, with a bright spot for CDF validation.
std::vector<float> GenerateTestEnvMap(uint32_t width, uint32_t height) {
    std::vector<float> data(static_cast<size_t>(width) * height * 4);
    for (uint32_t y = 0; y < height; ++y) {
        float v = static_cast<float>(y) / static_cast<float>(height);
        float brightness = std::sin(v * 3.14159f);  // bright at equator
        for (uint32_t x = 0; x < width; ++x) {
            uint32_t idx = (y * width + x) * 4;
            data[idx + 0] = brightness * 0.8f;
            data[idx + 1] = brightness * 0.9f;
            data[idx + 2] = brightness * 1.0f;
            data[idx + 3] = 1.0f;
        }
    }
    // Add a bright spot at (width/4, height/2) to create a non-uniform CDF
    uint32_t bx = width / 4;
    uint32_t by = height / 2;
    uint32_t bright_idx = (by * width + bx) * 4;
    data[bright_idx + 0] = 100.0f;
    data[bright_idx + 1] = 100.0f;
    data[bright_idx + 2] = 100.0f;
    return data;
}

}  // anonymous namespace

TEST_CASE("GBufferImages: create at correct resolution and formats",
          "[gbuffer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;
    constexpr uint32_t kWidth = 128;
    constexpr uint32_t kHeight = 64;

    monti::app::GBufferImages gbuffer;

    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(cmd != VK_NULL_HANDLE);

    REQUIRE(gbuffer.Create(ctx.Allocator(), ctx.Device(), kWidth, kHeight, cmd));

    ctx.SubmitAndWait(cmd);

    REQUIRE(gbuffer.Width() == kWidth);
    REQUIRE(gbuffer.Height() == kHeight);

    // Verify all images and views are non-null
    REQUIRE(gbuffer.NoisyDiffuseImage() != VK_NULL_HANDLE);
    REQUIRE(gbuffer.NoisySpecularImage() != VK_NULL_HANDLE);
    REQUIRE(gbuffer.MotionVectorsImage() != VK_NULL_HANDLE);
    REQUIRE(gbuffer.LinearDepthImage() != VK_NULL_HANDLE);
    REQUIRE(gbuffer.WorldNormalsImage() != VK_NULL_HANDLE);
    REQUIRE(gbuffer.DiffuseAlbedoImage() != VK_NULL_HANDLE);
    REQUIRE(gbuffer.SpecularAlbedoImage() != VK_NULL_HANDLE);

    REQUIRE(gbuffer.NoisyDiffuseView() != VK_NULL_HANDLE);
    REQUIRE(gbuffer.NoisySpecularView() != VK_NULL_HANDLE);
    REQUIRE(gbuffer.MotionVectorsView() != VK_NULL_HANDLE);
    REQUIRE(gbuffer.LinearDepthView() != VK_NULL_HANDLE);
    REQUIRE(gbuffer.WorldNormalsView() != VK_NULL_HANDLE);
    REQUIRE(gbuffer.DiffuseAlbedoView() != VK_NULL_HANDLE);
    REQUIRE(gbuffer.SpecularAlbedoView() != VK_NULL_HANDLE);

    ctx.WaitIdle();
}

TEST_CASE("GBufferImages: resize recreates images without leaks",
          "[gbuffer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;

    monti::app::GBufferImages gbuffer;

    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer.Create(ctx.Allocator(), ctx.Device(), 64, 64, cmd));
    ctx.SubmitAndWait(cmd);

    REQUIRE(gbuffer.Width() == 64);
    REQUIRE(gbuffer.Height() == 64);

    // Resize to different dimensions
    cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer.Resize(256, 128, cmd));
    ctx.SubmitAndWait(cmd);

    REQUIRE(gbuffer.Width() == 256);
    REQUIRE(gbuffer.Height() == 128);
    REQUIRE(gbuffer.NoisyDiffuseImage() != VK_NULL_HANDLE);

    // Resize to same dimensions should be a no-op
    VkImage prev_image = gbuffer.NoisyDiffuseImage();
    cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer.Resize(256, 128, cmd));
    ctx.SubmitAndWait(cmd);
    REQUIRE(gbuffer.NoisyDiffuseImage() == prev_image);

    ctx.WaitIdle();
}

TEST_CASE("EnvironmentMap: placeholder creation",
          "[environment_map][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;
    EnvironmentMap env_map;
    std::vector<Buffer> staging;

    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(env_map.CreatePlaceholders(ctx.Allocator(), ctx.Device(), cmd, staging));
    ctx.SubmitAndWait(cmd);

    REQUIRE(env_map.EnvTexture().Handle() != VK_NULL_HANDLE);
    REQUIRE(env_map.EnvTexture().View() != VK_NULL_HANDLE);
    REQUIRE(env_map.EnvTexture().Sampler() != VK_NULL_HANDLE);

    REQUIRE(env_map.MarginalCdfTexture().Handle() != VK_NULL_HANDLE);
    REQUIRE(env_map.MarginalCdfTexture().View() != VK_NULL_HANDLE);
    REQUIRE(env_map.MarginalCdfTexture().Sampler() != VK_NULL_HANDLE);

    REQUIRE(env_map.ConditionalCdfTexture().Handle() != VK_NULL_HANDLE);
    REQUIRE(env_map.ConditionalCdfTexture().View() != VK_NULL_HANDLE);
    REQUIRE(env_map.ConditionalCdfTexture().Sampler() != VK_NULL_HANDLE);

    REQUIRE(env_map.Width() == 1);
    REQUIRE(env_map.Height() == 1);
    REQUIRE_FALSE(env_map.IsLoaded());

    ctx.WaitIdle();
}

TEST_CASE("EnvironmentMap: load synthetic HDR, validate CDF and mipmaps",
          "[environment_map][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;

    constexpr uint32_t kEnvWidth = 64;
    constexpr uint32_t kEnvHeight = 32;
    auto env_data = GenerateTestEnvMap(kEnvWidth, kEnvHeight);

    EnvironmentMap env_map;
    std::vector<Buffer> staging;

    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(env_map.Load(ctx.Allocator(), ctx.Device(), cmd,
                         env_data.data(), kEnvWidth, kEnvHeight, staging));
    ctx.SubmitAndWait(cmd);

    REQUIRE(env_map.IsLoaded());
    REQUIRE(env_map.Width() == kEnvWidth);
    REQUIRE(env_map.Height() == kEnvHeight);

    // Verify env map texture has correct mip chain
    uint32_t expected_mips = static_cast<uint32_t>(
        std::floor(std::log2(std::max(kEnvWidth, kEnvHeight)))) + 1;
    REQUIRE(env_map.EnvTexture().MipLevels() == expected_mips);

    // Verify CDF textures are non-null and have samplers
    REQUIRE(env_map.MarginalCdfTexture().Handle() != VK_NULL_HANDLE);
    REQUIRE(env_map.MarginalCdfTexture().Sampler() != VK_NULL_HANDLE);
    REQUIRE(env_map.ConditionalCdfTexture().Handle() != VK_NULL_HANDLE);
    REQUIRE(env_map.ConditionalCdfTexture().Sampler() != VK_NULL_HANDLE);

    // Marginal CDF: width = height (one entry per row), height = 1
    REQUIRE(env_map.MarginalCdfTexture().Width() == kEnvHeight);
    REQUIRE(env_map.MarginalCdfTexture().Height() == 1);

    // Conditional CDF: width × height
    REQUIRE(env_map.ConditionalCdfTexture().Width() == kEnvWidth);
    REQUIRE(env_map.ConditionalCdfTexture().Height() == kEnvHeight);

    // Verify statistics are non-zero
    const auto& stats = env_map.Statistics();
    REQUIRE(stats.total_luminance > 0.0f);
    REQUIRE(stats.average_luminance > 0.0f);
    REQUIRE(stats.max_luminance > 0.0f);
    REQUIRE(stats.max_raw_luminance >= stats.max_luminance);

    // Readback marginal CDF last entry — should be ≈ 1.0
    // Create a readback buffer
    Buffer readback;
    REQUIRE(readback.Create(ctx.Allocator(),
                            kEnvHeight * sizeof(float),
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VMA_MEMORY_USAGE_CPU_ONLY));

    cmd = ctx.BeginOneShot();

    // Transition marginal CDF from SHADER_READ_ONLY to TRANSFER_SRC
    VkImageMemoryBarrier2 to_src{};
    to_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_src.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    to_src.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    to_src.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    to_src.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    to_src.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_src.image = env_map.MarginalCdfTexture().Handle();
    to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &to_src;
    vkCmdPipelineBarrier2(cmd, &dep);

    // Copy image to buffer
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {kEnvHeight, 1, 1};
    vkCmdCopyImageToBuffer(cmd, env_map.MarginalCdfTexture().Handle(),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.Handle(), 1, &region);

    ctx.SubmitAndWait(cmd);

    // Read back and verify last entry ≈ 1.0
    auto* cdf_data = static_cast<float*>(readback.Map());
    REQUIRE(cdf_data != nullptr);

    float last_cdf_value = cdf_data[kEnvHeight - 1];
    REQUIRE_THAT(last_cdf_value, Catch::Matchers::WithinAbs(1.0, 0.01));

    // Verify CDF is monotonically non-decreasing
    for (uint32_t i = 1; i < kEnvHeight; ++i)
        REQUIRE(cdf_data[i] >= cdf_data[i - 1]);

    readback.Unmap();

    ctx.WaitIdle();
}

TEST_CASE("BlueNoise: generation and buffer size",
          "[blue_noise][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;
    BlueNoise blue_noise;
    Buffer staging;

    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(blue_noise.Generate(ctx.Allocator(), cmd, staging));
    ctx.SubmitAndWait(cmd);

    // Verify buffer size: 16384 entries × 4 components × 4 bytes = 256 KB
    constexpr VkDeviceSize kExpectedSize =
        BlueNoise::kTableSize * BlueNoise::kComponentsPerEntry * sizeof(uint32_t);
    REQUIRE(blue_noise.BufferSize() == kExpectedSize);
    REQUIRE(blue_noise.TableBuffer().Handle() != VK_NULL_HANDLE);

    ctx.WaitIdle();
}

TEST_CASE("GBufferImages: datagen extra usage flags",
          "[gbuffer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;

    monti::app::GBufferImages gbuffer;

    VkCommandBuffer cmd = ctx.BeginOneShot();
    // Create with TRANSFER_SRC for datagen readback
    REQUIRE(gbuffer.Create(ctx.Allocator(), ctx.Device(), 32, 32, cmd,
                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
    ctx.SubmitAndWait(cmd);

    // Just verify creation doesn't fail and images are valid
    REQUIRE(gbuffer.NoisyDiffuseImage() != VK_NULL_HANDLE);
    REQUIRE(gbuffer.Width() == 32);

    ctx.WaitIdle();
}
