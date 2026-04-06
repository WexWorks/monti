#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <volk.h>

#include "../app/core/AutoExposure.h"
#include "../app/core/vulkan_context.h"
#include "shared_context.h"

#include <monti/capture/GpuReadback.h>
#include <monti/vulkan/VulkanBarriers.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using monti::capture::FloatToHalf;

// ============================================================================
// Test 1: Fixed-point encoding round-trip (GPU integration)
//
// Feeds known uniform-luminance images through the actual AutoExposure GPU
// shader pipeline. The shader uses fixed-point encoding internally. We verify
// the adapted luminance output matches the expected geometric mean, which
// implicitly validates the encoding/decoding round-trip in production code.
// ============================================================================

TEST_CASE("Fixed-point encoding round-trip via GPU", "[auto_exposure][vulkan]") {
    auto& ctx = monti::test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    // Test multiple luminance values through the actual GPU pipeline.
    // The GPU encodes log-luminance as uint((log(L)+BIAS)*SCALE), introducing
    // floor-truncation quantization. We compare against the analytically
    // expected quantized result rather than the input luminance.
    constexpr float kFixedScale = 4.0f;
    constexpr float kFixedBias = 14.0f;

    // Compute the expected adapted luminance after fixed-point quantization:
    //   encoded = uint((log(L) + BIAS) * SCALE)
    //   decoded = exp(encoded / SCALE - BIAS)
    auto ExpectedQuantized = [&](float L) -> float {
        auto encoded = static_cast<uint32_t>((std::log(L) + kFixedBias) * kFixedScale);
        return std::exp(static_cast<float>(encoded) / kFixedScale - kFixedBias);
    };

    std::array<float, 5> luminances = {0.01f, 0.1f, 0.18f, 1.0f, 10.0f};

    for (float L : luminances) {
        // Create a 4x4 uniform image at luminance L (gray: R=G=B=L)
        constexpr uint32_t kW = 4, kH = 4;
        constexpr uint32_t kPixelCount = kW * kH;
        size_t buffer_size = kPixelCount * 4 * sizeof(uint16_t);

        VkImageCreateInfo image_ci{};
        image_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_ci.imageType = VK_IMAGE_TYPE_2D;
        image_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        image_ci.extent = {kW, kH, 1};
        image_ci.mipLevels = 1;
        image_ci.arrayLayers = 1;
        image_ci.samples = VK_SAMPLE_COUNT_1_BIT;
        image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        image_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo alloc_ci{};
        alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        REQUIRE(vmaCreateImage(ctx.Allocator(), &image_ci, &alloc_ci,
                               &image, &allocation, nullptr) == VK_SUCCESS);

        VkImageViewCreateInfo view_ci{};
        view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_ci.image = image;
        view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkImageView view = VK_NULL_HANDLE;
        REQUIRE(vkCreateImageView(ctx.Device(), &view_ci, nullptr, &view) == VK_SUCCESS);

        VkBufferCreateInfo buf_ci{};
        buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_ci.size = buffer_size;
        buf_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo staging_alloc_ci{};
        staging_alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
        staging_alloc_ci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                 VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo staging_info{};
        VkBuffer staging = VK_NULL_HANDLE;
        VmaAllocation staging_alloc = VK_NULL_HANDLE;
        REQUIRE(vmaCreateBuffer(ctx.Allocator(), &buf_ci, &staging_alloc_ci,
                                &staging, &staging_alloc, &staging_info) == VK_SUCCESS);

        auto* pixels = static_cast<uint16_t*>(staging_info.pMappedData);
        uint16_t hv = FloatToHalf(L);
        uint16_t ha = FloatToHalf(1.0f);
        for (uint32_t i = 0; i < kPixelCount; ++i) {
            pixels[i * 4 + 0] = hv;
            pixels[i * 4 + 1] = hv;
            pixels[i * 4 + 2] = hv;
            pixels[i * 4 + 3] = ha;
        }
        vmaFlushAllocation(ctx.Allocator(), staging_alloc, 0, VK_WHOLE_SIZE);

        VkCommandBuffer cmd = ctx.BeginOneShot();
        auto barrier = monti::vulkan::MakeImageBarrier(
            image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        monti::vulkan::CmdPipelineBarrier(cmd, {&barrier, 1}, vkCmdPipelineBarrier2);

        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {kW, kH, 1};
        vkCmdCopyBufferToImage(cmd, staging, image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        auto barrier2 = monti::vulkan::MakeImageBarrier(
            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        monti::vulkan::CmdPipelineBarrier(cmd, {&barrier2, 1}, vkCmdPipelineBarrier2);
        ctx.SubmitAndWait(cmd);

        monti::app::AutoExposure auto_exp;
        REQUIRE(auto_exp.Create(ctx.Device(), ctx.Allocator(), APP_SHADER_SPV_DIR,
                                kW, kH, view));
        auto_exp.SetAdaptationSpeed(100.0f);

        for (int i = 0; i < 10; ++i) {
            VkCommandBuffer frame_cmd = ctx.BeginOneShot();
            auto_exp.Compute(frame_cmd, image, 0.1f);
            ctx.SubmitAndWait(frame_cmd);
        }

        float adapted = auto_exp.AdaptedLuminance();
        float expected = ExpectedQuantized(L);
        INFO("Luminance=" << L << " expected_quantized=" << expected
                          << " adapted=" << adapted);
        REQUIRE_THAT(adapted, WithinRel(expected, 0.01f));

        auto_exp.Destroy();
        vkDestroyImageView(ctx.Device(), view, nullptr);
        vmaDestroyImage(ctx.Allocator(), image, allocation);
        vmaDestroyBuffer(ctx.Allocator(), staging, staging_alloc);
    }
}

// ============================================================================
// Helper: Create a test RGBA16F image, fill it with a uniform color,
// run AutoExposure for several frames, and return the adapted luminance.
// ============================================================================

namespace {

struct TestImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation staging_alloc = VK_NULL_HANDLE;
};

TestImage CreateTestImage(monti::app::VulkanContext& ctx, uint32_t width, uint32_t height,
                          float r, float g, float b) {
    TestImage result{};

    // Create the image
    VkImageCreateInfo image_ci{};
    image_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_ci.imageType = VK_IMAGE_TYPE_2D;
    image_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    image_ci.extent = {width, height, 1};
    image_ci.mipLevels = 1;
    image_ci.arrayLayers = 1;
    image_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult vr = vmaCreateImage(ctx.Allocator(), &image_ci, &alloc_ci,
                                 &result.image, &result.allocation, nullptr);
    REQUIRE(vr == VK_SUCCESS);

    // Create image view
    VkImageViewCreateInfo view_ci{};
    view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image = result.image;
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    vr = vkCreateImageView(ctx.Device(), &view_ci, nullptr, &result.view);
    REQUIRE(vr == VK_SUCCESS);

    // Create staging buffer with pixel data
    auto pixel_count = static_cast<size_t>(width) * height;
    size_t buffer_size = pixel_count * 4 * sizeof(uint16_t);  // RGBA16F

    VkBufferCreateInfo buf_ci{};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = buffer_size;
    buf_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo staging_alloc_ci{};
    staging_alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
    staging_alloc_ci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                             VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo staging_info{};
    vr = vmaCreateBuffer(ctx.Allocator(), &buf_ci, &staging_alloc_ci,
                         &result.staging, &result.staging_alloc, &staging_info);
    REQUIRE(vr == VK_SUCCESS);

    // Fill staging buffer
    auto* pixels = static_cast<uint16_t*>(staging_info.pMappedData);
    uint16_t hr = FloatToHalf(r);
    uint16_t hg = FloatToHalf(g);
    uint16_t hb = FloatToHalf(b);
    uint16_t ha = FloatToHalf(1.0f);

    for (size_t i = 0; i < pixel_count; ++i) {
        pixels[i * 4 + 0] = hr;
        pixels[i * 4 + 1] = hg;
        pixels[i * 4 + 2] = hb;
        pixels[i * 4 + 3] = ha;
    }
    vmaFlushAllocation(ctx.Allocator(), result.staging_alloc, 0, VK_WHOLE_SIZE);

    // Upload: transition image → copy from staging → transition to GENERAL
    VkCommandBuffer cmd = ctx.BeginOneShot();

    auto barrier = monti::vulkan::MakeImageBarrier(
        result.image,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT);
    monti::vulkan::CmdPipelineBarrier(cmd, {&barrier, 1}, vkCmdPipelineBarrier2);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(cmd, result.staging, result.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    auto barrier2 = monti::vulkan::MakeImageBarrier(
        result.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    monti::vulkan::CmdPipelineBarrier(cmd, {&barrier2, 1}, vkCmdPipelineBarrier2);

    ctx.SubmitAndWait(cmd);
    return result;
}

void DestroyTestImage(monti::app::VulkanContext& ctx, TestImage& img) {
    if (img.view != VK_NULL_HANDLE)
        vkDestroyImageView(ctx.Device(), img.view, nullptr);
    if (img.image != VK_NULL_HANDLE)
        vmaDestroyImage(ctx.Allocator(), img.image, img.allocation);
    if (img.staging != VK_NULL_HANDLE)
        vmaDestroyBuffer(ctx.Allocator(), img.staging, img.staging_alloc);
    img = {};
}

// Run auto-exposure for N frames with the given delta_time, return adapted luminance.
float RunAutoExposureFrames(monti::app::VulkanContext& ctx,
                            monti::app::AutoExposure& auto_exp,
                            VkImage hdr_image,
                            int num_frames, float delta_time) {
    for (int i = 0; i < num_frames; ++i) {
        VkCommandBuffer cmd = ctx.BeginOneShot();
        auto_exp.Compute(cmd, hdr_image, delta_time);
        ctx.SubmitAndWait(cmd);
    }
    return auto_exp.AdaptedLuminance();
}

}  // namespace

// ============================================================================
// Test 2: Known mid-gray image → multiplier ≈ 1.0
// ============================================================================

TEST_CASE("AutoExposure — uniform mid-gray image", "[auto_exposure][vulkan]") {
    auto& ctx = monti::test::SharedVulkanContext();
    constexpr uint32_t kWidth = 4;
    constexpr uint32_t kHeight = 4;

    // All pixels = (0.18, 0.18, 0.18) → BT.709 luminance = 0.18
    auto test_img = CreateTestImage(ctx, kWidth, kHeight, 0.18f, 0.18f, 0.18f);

    monti::app::AutoExposure auto_exp;
    REQUIRE(auto_exp.Create(ctx.Device(), ctx.Allocator(), APP_SHADER_SPV_DIR,
                            kWidth, kHeight, test_img.view));

    // Run enough frames for convergence (high adaptation speed + generous dt)
    auto_exp.SetAdaptationSpeed(100.0f);
    float adapted = RunAutoExposureFrames(ctx, auto_exp, test_img.image, 10, 0.1f);

    INFO("Adapted luminance: " << adapted);
    REQUIRE_THAT(adapted, WithinRel(0.18f, 0.05f));

    float multiplier = auto_exp.ExposureMultiplier();
    INFO("Exposure multiplier: " << multiplier);
    REQUIRE_THAT(multiplier, WithinRel(1.0f, 0.05f));

    auto_exp.Destroy();
    DestroyTestImage(ctx, test_img);
}

// ============================================================================
// Test 3: Uniform bright image → multiplier ≈ 0.18
// ============================================================================

TEST_CASE("AutoExposure — uniform bright image", "[auto_exposure][vulkan]") {
    auto& ctx = monti::test::SharedVulkanContext();
    constexpr uint32_t kWidth = 4;
    constexpr uint32_t kHeight = 4;

    // All pixels = (1.0, 1.0, 1.0) → BT.709 luminance = 1.0
    auto test_img = CreateTestImage(ctx, kWidth, kHeight, 1.0f, 1.0f, 1.0f);

    monti::app::AutoExposure auto_exp;
    REQUIRE(auto_exp.Create(ctx.Device(), ctx.Allocator(), APP_SHADER_SPV_DIR,
                            kWidth, kHeight, test_img.view));

    auto_exp.SetAdaptationSpeed(100.0f);
    float adapted = RunAutoExposureFrames(ctx, auto_exp, test_img.image, 10, 0.1f);

    INFO("Adapted luminance: " << adapted);
    REQUIRE_THAT(adapted, WithinRel(1.0f, 0.05f));

    float multiplier = auto_exp.ExposureMultiplier();
    INFO("Exposure multiplier: " << multiplier);
    REQUIRE_THAT(multiplier, WithinRel(0.18f, 0.05f));

    auto_exp.Destroy();
    DestroyTestImage(ctx, test_img);
}

// ============================================================================
// Test 4: Mixed bright/dark image → geometric mean ≈ 0.1
// ============================================================================

TEST_CASE("AutoExposure — mixed bright/dark image", "[auto_exposure][vulkan]") {
    auto& ctx = monti::test::SharedVulkanContext();
    // Use a larger image to get 50/50 split: top half dark, bottom half bright
    constexpr uint32_t kWidth = 4;
    constexpr uint32_t kHeight = 4;

    // Create a custom mixed image: half at 0.01, half at 1.0
    // Expected L_avg = exp(0.5 * log(0.01) + 0.5 * log(1.0)) = exp(-2.302..) ≈ 0.1
    TestImage result{};

    VkImageCreateInfo image_ci{};
    image_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_ci.imageType = VK_IMAGE_TYPE_2D;
    image_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    image_ci.extent = {kWidth, kHeight, 1};
    image_ci.mipLevels = 1;
    image_ci.arrayLayers = 1;
    image_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult vr = vmaCreateImage(ctx.Allocator(), &image_ci, &alloc_ci,
                                 &result.image, &result.allocation, nullptr);
    REQUIRE(vr == VK_SUCCESS);

    VkImageViewCreateInfo view_ci{};
    view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image = result.image;
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    vr = vkCreateImageView(ctx.Device(), &view_ci, nullptr, &result.view);
    REQUIRE(vr == VK_SUCCESS);

    constexpr uint32_t kPixelCount = kWidth * kHeight;
    size_t buffer_size = kPixelCount * 4 * sizeof(uint16_t);

    VkBufferCreateInfo buf_ci{};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = buffer_size;
    buf_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo staging_alloc_ci{};
    staging_alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
    staging_alloc_ci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                             VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo staging_info{};
    vr = vmaCreateBuffer(ctx.Allocator(), &buf_ci, &staging_alloc_ci,
                         &result.staging, &result.staging_alloc, &staging_info);
    REQUIRE(vr == VK_SUCCESS);

    auto* pixels = static_cast<uint16_t*>(staging_info.pMappedData);
    uint16_t h_alpha = FloatToHalf(1.0f);
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        float val = (i < kPixelCount / 2) ? 0.01f : 1.0f;
        uint16_t hv = FloatToHalf(val);
        pixels[i * 4 + 0] = hv;
        pixels[i * 4 + 1] = hv;
        pixels[i * 4 + 2] = hv;
        pixels[i * 4 + 3] = h_alpha;
    }
    vmaFlushAllocation(ctx.Allocator(), result.staging_alloc, 0, VK_WHOLE_SIZE);

    VkCommandBuffer cmd = ctx.BeginOneShot();

    auto barrier = monti::vulkan::MakeImageBarrier(
        result.image,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT);
    monti::vulkan::CmdPipelineBarrier(cmd, {&barrier, 1}, vkCmdPipelineBarrier2);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {kWidth, kHeight, 1};
    vkCmdCopyBufferToImage(cmd, result.staging, result.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    auto barrier2 = monti::vulkan::MakeImageBarrier(
        result.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    monti::vulkan::CmdPipelineBarrier(cmd, {&barrier2, 1}, vkCmdPipelineBarrier2);

    ctx.SubmitAndWait(cmd);

    // Run auto-exposure
    monti::app::AutoExposure auto_exp;
    REQUIRE(auto_exp.Create(ctx.Device(), ctx.Allocator(), APP_SHADER_SPV_DIR,
                            kWidth, kHeight, result.view));

    auto_exp.SetAdaptationSpeed(100.0f);
    float adapted = RunAutoExposureFrames(ctx, auto_exp, result.image, 10, 0.1f);

    // Expected: geometric mean of half at 0.01, half at 1.0
    // L_avg = exp(0.5 * ln(0.01) + 0.5 * ln(1.0)) = exp(-2.3026) ≈ 0.1
    float expected_L = std::exp(0.5f * std::log(0.01f) + 0.5f * std::log(1.0f));
    INFO("Adapted luminance: " << adapted << ", expected: " << expected_L);
    REQUIRE_THAT(adapted, WithinRel(expected_L, 0.10f));

    float multiplier = auto_exp.ExposureMultiplier();
    float expected_mul = 0.18f / expected_L;
    INFO("Exposure multiplier: " << multiplier << ", expected: " << expected_mul);
    REQUIRE_THAT(multiplier, WithinRel(expected_mul, 0.10f));

    auto_exp.Destroy();
    DestroyTestImage(ctx, result);
}

// ============================================================================
// Test 5: Temporal smoothing — adaptation speed affects convergence
// ============================================================================

TEST_CASE("AutoExposure — temporal smoothing convergence", "[auto_exposure][vulkan]") {
    auto& ctx = monti::test::SharedVulkanContext();
    constexpr uint32_t kWidth = 4;
    constexpr uint32_t kHeight = 4;

    // Start with a bright image
    auto bright_img = CreateTestImage(ctx, kWidth, kHeight, 1.0f, 1.0f, 1.0f);
    // Also create a dark image
    auto dark_img = CreateTestImage(ctx, kWidth, kHeight, 0.01f, 0.01f, 0.01f);

    monti::app::AutoExposure auto_exp;
    REQUIRE(auto_exp.Create(ctx.Device(), ctx.Allocator(), APP_SHADER_SPV_DIR,
                            kWidth, kHeight, bright_img.view));

    // Converge on bright image
    auto_exp.SetAdaptationSpeed(100.0f);
    RunAutoExposureFrames(ctx, auto_exp, bright_img.image, 10, 0.1f);
    float bright_adapted = auto_exp.AdaptedLuminance();
    REQUIRE_THAT(bright_adapted, WithinRel(1.0f, 0.05f));

    // Switch to dark image — update descriptor set
    REQUIRE(auto_exp.Resize(kWidth, kHeight, dark_img.view));

    // Run 1 frame with slow adaptation — should NOT have converged to dark yet
    auto_exp.SetAdaptationSpeed(1.0f);
    {
        VkCommandBuffer cmd = ctx.BeginOneShot();
        auto_exp.Compute(cmd, dark_img.image, 0.016f);  // ~60fps
        ctx.SubmitAndWait(cmd);
    }
    float partial_adapted = auto_exp.AdaptedLuminance();

    // Should still be much closer to bright than dark
    INFO("After 1 slow frame, adapted: " << partial_adapted);
    REQUIRE(partial_adapted > 0.1f);  // Still above the dark target of 0.01

    // Now converge with fast adaptation
    auto_exp.SetAdaptationSpeed(100.0f);
    RunAutoExposureFrames(ctx, auto_exp, dark_img.image, 20, 0.1f);
    float dark_adapted = auto_exp.AdaptedLuminance();

    INFO("After convergence on dark image, adapted: " << dark_adapted);
    REQUIRE_THAT(dark_adapted, WithinRel(0.01f, 0.15f));

    auto_exp.Destroy();
    DestroyTestImage(ctx, bright_img);
    DestroyTestImage(ctx, dark_img);
}

// ============================================================================
// Test 6: Create/Destroy lifecycle — no leaks or crashes
// ============================================================================

TEST_CASE("AutoExposure — create and destroy", "[auto_exposure][vulkan]") {
    auto& ctx = monti::test::SharedVulkanContext();
    constexpr uint32_t kWidth = 16;
    constexpr uint32_t kHeight = 16;

    auto test_img = CreateTestImage(ctx, kWidth, kHeight, 0.5f, 0.5f, 0.5f);

    monti::app::AutoExposure auto_exp;
    REQUIRE(auto_exp.Create(ctx.Device(), ctx.Allocator(), APP_SHADER_SPV_DIR,
                            kWidth, kHeight, test_img.view));

    // Initial state before any compute
    REQUIRE(auto_exp.ExposureMultiplier() == 1.0f);  // result_mapped_ starts at 0
    REQUIRE(auto_exp.AdaptedLuminance() == 0.0f);

    auto_exp.Destroy();
    DestroyTestImage(ctx, test_img);
    ctx.WaitIdle();
}

// ============================================================================
// Test 7: Resize — works without recreating pipelines
// ============================================================================

TEST_CASE("AutoExposure — resize", "[auto_exposure][vulkan]") {
    auto& ctx = monti::test::SharedVulkanContext();

    auto img_small = CreateTestImage(ctx, 4, 4, 0.5f, 0.5f, 0.5f);
    auto img_large = CreateTestImage(ctx, 16, 16, 0.5f, 0.5f, 0.5f);

    monti::app::AutoExposure auto_exp;
    REQUIRE(auto_exp.Create(ctx.Device(), ctx.Allocator(), APP_SHADER_SPV_DIR,
                            4, 4, img_small.view));

    // Run a frame at small size
    auto_exp.SetAdaptationSpeed(100.0f);
    RunAutoExposureFrames(ctx, auto_exp, img_small.image, 5, 0.1f);

    // Resize to large
    REQUIRE(auto_exp.Resize(16, 16, img_large.view));

    // Run frames at new size — should still converge correctly
    float adapted = RunAutoExposureFrames(ctx, auto_exp, img_large.image, 10, 0.1f);

    // 0.5 uniform → luminance = 0.5
    INFO("Adapted after resize: " << adapted);
    REQUIRE_THAT(adapted, WithinRel(0.5f, 0.10f));

    auto_exp.Destroy();
    DestroyTestImage(ctx, img_small);
    DestroyTestImage(ctx, img_large);
}
