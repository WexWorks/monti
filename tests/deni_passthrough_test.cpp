#include <catch2/catch_test_macros.hpp>

#include <volk.h>

#include "../app/core/vulkan_context.h"
#include "shared_context.h"

#include <deni/vulkan/Denoiser.h>

#include <monti/capture/GpuReadback.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#ifndef DENI_SHADER_SPV_DIR
#define DENI_SHADER_SPV_DIR "build/deni_shaders"
#endif

namespace {

using monti::capture::HalfToFloat;
using monti::capture::FloatToHalf;

constexpr uint32_t kTestWidth = 64;
constexpr uint32_t kTestHeight = 64;
constexpr uint32_t kPixelCount = kTestWidth * kTestHeight;

struct TestImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
};

TestImage CreateTestImage(VmaAllocator allocator, VkDevice device,
                          uint32_t width, uint32_t height, VkFormat format) {
    TestImage img;

    VkImageCreateInfo image_ci{};
    image_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_ci.imageType = VK_IMAGE_TYPE_2D;
    image_ci.format = format;
    image_ci.extent = {width, height, 1};
    image_ci.mipLevels = 1;
    image_ci.arrayLayers = 1;
    image_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult result = vmaCreateImage(allocator, &image_ci, &alloc_ci,
                                     &img.image, &img.allocation, nullptr);
    REQUIRE(result == VK_SUCCESS);

    VkImageViewCreateInfo view_ci{};
    view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image = img.image;
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format = format;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    result = vkCreateImageView(device, &view_ci, nullptr, &img.view);
    REQUIRE(result == VK_SUCCESS);

    return img;
}

void DestroyTestImage(VmaAllocator allocator, VkDevice device, TestImage& img) {
    if (img.view != VK_NULL_HANDLE)
        vkDestroyImageView(device, img.view, nullptr);
    if (img.image != VK_NULL_HANDLE)
        vmaDestroyImage(allocator, img.image, img.allocation);
    img = {};
}

struct StagingBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
};

StagingBuffer CreateStagingBuffer(VmaAllocator allocator, VkDeviceSize size,
                                  VkBufferUsageFlags usage) {
    StagingBuffer buf;

    VkBufferCreateInfo buf_ci{};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = size;
    buf_ci.usage = usage;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    VkResult result = vmaCreateBuffer(allocator, &buf_ci, &alloc_ci,
                                      &buf.buffer, &buf.allocation, nullptr);
    REQUIRE(result == VK_SUCCESS);
    return buf;
}

void DestroyStagingBuffer(VmaAllocator allocator, StagingBuffer& buf) {
    if (buf.buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(allocator, buf.buffer, buf.allocation);
    buf = {};
}

void UploadRGBA16F(monti::app::VulkanContext& ctx, VkImage dst_image,
                   const uint16_t* data, uint32_t width, uint32_t height) {
    VkDeviceSize size = width * height * 4 * sizeof(uint16_t);

    auto staging = CreateStagingBuffer(ctx.Allocator(), size,
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    void* mapped = nullptr;
    VkResult result = vmaMapMemory(ctx.Allocator(), staging.allocation, &mapped);
    REQUIRE(result == VK_SUCCESS);
    REQUIRE(mapped != nullptr);
    std::memcpy(mapped, data, size);
    vmaUnmapMemory(ctx.Allocator(), staging.allocation);

    VkCommandBuffer cmd = ctx.BeginOneShot();

    // Transition to TRANSFER_DST
    VkImageMemoryBarrier2 to_dst{};
    to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_dst.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    to_dst.srcAccessMask = 0;
    to_dst.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    to_dst.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_dst.image = dst_image;
    to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &to_dst;
    vkCmdPipelineBarrier2(cmd, &dep);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(cmd, staging.buffer, dst_image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition to GENERAL for compute read
    VkImageMemoryBarrier2 to_general{};
    to_general.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_general.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    to_general.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    to_general.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    to_general.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    to_general.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    to_general.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_general.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_general.image = dst_image;
    to_general.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    dep.pImageMemoryBarriers = &to_general;
    vkCmdPipelineBarrier2(cmd, &dep);

    ctx.SubmitAndWait(cmd);

    DestroyStagingBuffer(ctx.Allocator(), staging);
}

}  // namespace

TEST_CASE("Deni passthrough denoiser: diffuse + specular", "[deni][integration]") {
    auto& ctx = monti::test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    // Fill diffuse = {0.3, 0.1, 0.2, 1.0} per pixel
    std::vector<uint16_t> diffuse_data(kPixelCount * 4);
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        diffuse_data[i * 4 + 0] = FloatToHalf(0.3f);
        diffuse_data[i * 4 + 1] = FloatToHalf(0.1f);
        diffuse_data[i * 4 + 2] = FloatToHalf(0.2f);
        diffuse_data[i * 4 + 3] = FloatToHalf(1.0f);
    }

    // Fill specular = {0.1, 0.4, 0.05, 1.0} per pixel
    std::vector<uint16_t> specular_data(kPixelCount * 4);
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        specular_data[i * 4 + 0] = FloatToHalf(0.1f);
        specular_data[i * 4 + 1] = FloatToHalf(0.4f);
        specular_data[i * 4 + 2] = FloatToHalf(0.05f);
        specular_data[i * 4 + 3] = FloatToHalf(1.0f);
    }

    // Create input images
    auto diffuse_img = CreateTestImage(ctx.Allocator(), ctx.Device(),
                                       kTestWidth, kTestHeight,
                                       VK_FORMAT_R16G16B16A16_SFLOAT);
    auto specular_img = CreateTestImage(ctx.Allocator(), ctx.Device(),
                                        kTestWidth, kTestHeight,
                                        VK_FORMAT_R16G16B16A16_SFLOAT);

    // Placeholder images for unused DenoiserInput fields
    auto motion_img = CreateTestImage(ctx.Allocator(), ctx.Device(),
                                      kTestWidth, kTestHeight,
                                      VK_FORMAT_R16G16B16A16_SFLOAT);
    auto depth_img = CreateTestImage(ctx.Allocator(), ctx.Device(),
                                     kTestWidth, kTestHeight,
                                     VK_FORMAT_R16G16B16A16_SFLOAT);
    auto normals_img = CreateTestImage(ctx.Allocator(), ctx.Device(),
                                       kTestWidth, kTestHeight,
                                       VK_FORMAT_R16G16B16A16_SFLOAT);
    auto diff_albedo_img = CreateTestImage(ctx.Allocator(), ctx.Device(),
                                           kTestWidth, kTestHeight,
                                           VK_FORMAT_R16G16B16A16_SFLOAT);
    auto spec_albedo_img = CreateTestImage(ctx.Allocator(), ctx.Device(),
                                           kTestWidth, kTestHeight,
                                           VK_FORMAT_R16G16B16A16_SFLOAT);

    // Upload input data
    UploadRGBA16F(ctx, diffuse_img.image, diffuse_data.data(), kTestWidth, kTestHeight);
    UploadRGBA16F(ctx, specular_img.image, specular_data.data(), kTestWidth, kTestHeight);

    // Transition placeholder images to GENERAL (they need valid layout for descriptor writes)
    {
        VkCommandBuffer cmd = ctx.BeginOneShot();

        std::array<VkImageMemoryBarrier2, 5> barriers{};
        VkImage placeholder_images[] = {motion_img.image, depth_img.image, normals_img.image,
                                        diff_albedo_img.image, spec_albedo_img.image};
        for (uint32_t i = 0; i < 5; ++i) {
            barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barriers[i].srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            barriers[i].srcAccessMask = 0;
            barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
            barriers[i].dstAccessMask = 0;
            barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].image = placeholder_images[i];
            barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
        dep.pImageMemoryBarriers = barriers.data();
        vkCmdPipelineBarrier2(cmd, &dep);

        ctx.SubmitAndWait(cmd);
    }

    // Create denoiser
    deni::vulkan::DenoiserDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.width = kTestWidth;
    desc.height = kTestHeight;
    desc.allocator = ctx.Allocator();
    desc.shader_dir = DENI_SHADER_SPV_DIR;
    desc.get_device_proc_addr = vkGetDeviceProcAddr;

    auto denoiser = deni::vulkan::Denoiser::Create(desc);
    REQUIRE(denoiser != nullptr);
    REQUIRE(denoiser->SetMode(deni::vulkan::DenoiserMode::kPassthrough));

    // Run denoise pass
    deni::vulkan::DenoiserInput input{};
    input.noisy_diffuse = diffuse_img.view;
    input.noisy_specular = specular_img.view;
    input.motion_vectors = motion_img.view;
    input.linear_depth = depth_img.view;
    input.world_normals = normals_img.view;
    input.diffuse_albedo = diff_albedo_img.view;
    input.specular_albedo = spec_albedo_img.view;
    input.render_width = kTestWidth;
    input.render_height = kTestHeight;
    input.reset_accumulation = false;

    VkCommandBuffer cmd = ctx.BeginOneShot();
    auto output = denoiser->Denoise(cmd, input);
    ctx.SubmitAndWait(cmd);

    REQUIRE(output.denoised_image != VK_NULL_HANDLE);
    REQUIRE(output.denoised_color != VK_NULL_HANDLE);

    // Read back output image
    VkDeviceSize readback_size = kPixelCount * 4 * sizeof(uint16_t);
    auto readback = CreateStagingBuffer(ctx.Allocator(), readback_size,
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    VkCommandBuffer copy_cmd = ctx.BeginOneShot();

    // Output is in GENERAL layout after Denoise(); transition to TRANSFER_SRC
    VkImageMemoryBarrier2 to_src{};
    to_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_src.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    to_src.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    to_src.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    to_src.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    to_src.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_src.image = output.denoised_image;
    to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &to_src;
    vkCmdPipelineBarrier2(copy_cmd, &dep);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {kTestWidth, kTestHeight, 1};
    vkCmdCopyImageToBuffer(copy_cmd, output.denoised_image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.buffer, 1, &region);

    ctx.SubmitAndWait(copy_cmd);

    // Verify output = diffuse + specular per pixel
    void* readback_mapped = nullptr;
    REQUIRE(vmaMapMemory(ctx.Allocator(), readback.allocation, &readback_mapped) == VK_SUCCESS);
    auto* result_raw = static_cast<const uint16_t*>(readback_mapped);

    constexpr float kExpectedR = 0.3f + 0.1f;   // 0.4
    constexpr float kExpectedG = 0.1f + 0.4f;   // 0.5
    constexpr float kExpectedB = 0.2f + 0.05f;   // 0.25
    constexpr float kExpectedA = 1.0f + 1.0f;   // 2.0
    constexpr float kTolerance = 0.002f;  // fp16 precision

    uint32_t mismatch_count = 0;
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        float r = HalfToFloat(result_raw[i * 4 + 0]);
        float g = HalfToFloat(result_raw[i * 4 + 1]);
        float b = HalfToFloat(result_raw[i * 4 + 2]);
        float a = HalfToFloat(result_raw[i * 4 + 3]);

        if (std::abs(r - kExpectedR) > kTolerance ||
            std::abs(g - kExpectedG) > kTolerance ||
            std::abs(b - kExpectedB) > kTolerance ||
            std::abs(a - kExpectedA) > kTolerance) {
            ++mismatch_count;
        }
    }

    vmaUnmapMemory(ctx.Allocator(), readback.allocation);
    REQUIRE(mismatch_count == 0);

    // Cleanup
    DestroyStagingBuffer(ctx.Allocator(), readback);
    denoiser.reset();
    DestroyTestImage(ctx.Allocator(), ctx.Device(), diffuse_img);
    DestroyTestImage(ctx.Allocator(), ctx.Device(), specular_img);
    DestroyTestImage(ctx.Allocator(), ctx.Device(), motion_img);
    DestroyTestImage(ctx.Allocator(), ctx.Device(), depth_img);
    DestroyTestImage(ctx.Allocator(), ctx.Device(), normals_img);
    DestroyTestImage(ctx.Allocator(), ctx.Device(), diff_albedo_img);
    DestroyTestImage(ctx.Allocator(), ctx.Device(), spec_albedo_img);
    ctx.WaitIdle();
}

TEST_CASE("Deni passthrough denoiser: Resize", "[deni][integration]") {
    auto& ctx = monti::test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    deni::vulkan::DenoiserDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.width = kTestWidth;
    desc.height = kTestHeight;
    desc.allocator = ctx.Allocator();
    desc.shader_dir = DENI_SHADER_SPV_DIR;
    desc.get_device_proc_addr = vkGetDeviceProcAddr;

    auto denoiser = deni::vulkan::Denoiser::Create(desc);
    REQUIRE(denoiser != nullptr);

    // Resize to a different resolution — should not crash or leak
    denoiser->Resize(128, 128);

    // Resize back to original
    denoiser->Resize(kTestWidth, kTestHeight);

    // Resize to same dimensions — should be a no-op
    denoiser->Resize(kTestWidth, kTestHeight);

    // Resize to zero — should be a no-op
    denoiser->Resize(0, 0);

    denoiser.reset();
    ctx.WaitIdle();
}

TEST_CASE("Deni passthrough denoiser: null allocator rejected", "[deni][unit]") {
    deni::vulkan::DenoiserDesc desc{};
    desc.device = VK_NULL_HANDLE;
    desc.physical_device = VK_NULL_HANDLE;
    desc.allocator = VK_NULL_HANDLE;
    desc.get_device_proc_addr = vkGetDeviceProcAddr;

    auto denoiser = deni::vulkan::Denoiser::Create(desc);
    REQUIRE(denoiser == nullptr);
}

TEST_CASE("Deni passthrough denoiser: null get_device_proc_addr rejected", "[deni][unit]") {
    deni::vulkan::DenoiserDesc desc{};
    desc.device = VK_NULL_HANDLE;
    desc.physical_device = VK_NULL_HANDLE;
    desc.allocator = VK_NULL_HANDLE;
    desc.get_device_proc_addr = nullptr;

    auto denoiser = deni::vulkan::Denoiser::Create(desc);
    REQUIRE(denoiser == nullptr);
}
