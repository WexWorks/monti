#include <catch2/catch_test_macros.hpp>

#include <volk.h>

#include "../app/core/vulkan_context.h"
#include "shared_context.h"

#include <monti/capture/GpuReadback.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#ifndef DENI_SHADER_SPV_DIR
#define DENI_SHADER_SPV_DIR "build/deni_shaders"
#endif

namespace {

using monti::capture::HalfToFloat;
using monti::capture::FloatToHalf;

constexpr uint32_t kTestWidth = 32;
constexpr uint32_t kTestHeight = 32;
constexpr uint32_t kPixelCount = kTestWidth * kTestHeight;
constexpr uint32_t kWorkgroupSize = 16;

// ---------------------------------------------------------------------------
// SPIR-V loader
// ---------------------------------------------------------------------------
std::vector<uint8_t> LoadShaderFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

// ---------------------------------------------------------------------------
// Test image helpers
// ---------------------------------------------------------------------------
struct TestImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
};

TestImage CreateTestImage(VmaAllocator allocator, VkDevice device,
                          uint32_t width, uint32_t height, VkFormat format,
                          VkImageUsageFlags extra_usage = 0) {
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
    image_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | extra_usage;
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

void UploadImageData(monti::app::VulkanContext& ctx, VkImage dst_image,
                     const uint16_t* data, uint32_t width, uint32_t height,
                     uint32_t channels) {
    VkDeviceSize size = width * height * channels * sizeof(uint16_t);

    auto staging = CreateStagingBuffer(ctx.Allocator(), size,
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    void* mapped = nullptr;
    VkResult result = vmaMapMemory(ctx.Allocator(), staging.allocation, &mapped);
    REQUIRE(result == VK_SUCCESS);
    std::memcpy(mapped, data, size);
    vmaUnmapMemory(ctx.Allocator(), staging.allocation);

    VkCommandBuffer cmd = ctx.BeginOneShot();

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

std::vector<uint16_t> ReadbackImageData(monti::app::VulkanContext& ctx,
                                        VkImage src_image,
                                        uint32_t width, uint32_t height,
                                        uint32_t channels) {
    VkDeviceSize size = width * height * channels * sizeof(uint16_t);

    auto staging = CreateStagingBuffer(ctx.Allocator(), size,
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    VkCommandBuffer cmd = ctx.BeginOneShot();

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
    to_src.image = src_image;
    to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &to_src;
    vkCmdPipelineBarrier2(cmd, &dep);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    vkCmdCopyImageToBuffer(cmd, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           staging.buffer, 1, &region);

    // Transition back to GENERAL for subsequent use
    VkImageMemoryBarrier2 to_general{};
    to_general.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_general.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    to_general.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    to_general.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    to_general.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    to_general.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    to_general.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_general.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_general.image = src_image;
    to_general.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    dep.pImageMemoryBarriers = &to_general;
    vkCmdPipelineBarrier2(cmd, &dep);

    ctx.SubmitAndWait(cmd);

    void* mapped = nullptr;
    VkResult result = vmaMapMemory(ctx.Allocator(), staging.allocation, &mapped);
    REQUIRE(result == VK_SUCCESS);

    std::vector<uint16_t> output(width * height * channels);
    std::memcpy(output.data(), mapped, size);
    vmaUnmapMemory(ctx.Allocator(), staging.allocation);

    DestroyStagingBuffer(ctx.Allocator(), staging);
    return output;
}

// ---------------------------------------------------------------------------
// Reproject pipeline test fixture
// ---------------------------------------------------------------------------
struct ReprojectFixture {
    monti::app::VulkanContext& ctx;

    // Reproject pipeline
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout ds_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;

    // Input images (readonly in shader)
    TestImage motion_vectors;   // RG16F
    TestImage prev_diffuse;     // RGBA16F
    TestImage prev_specular;    // RGBA16F
    TestImage prev_depth;       // RG16F
    TestImage curr_depth;       // RG16F

    // Output images (writeonly in shader)
    TestImage reprojected_d;    // RGBA16F
    TestImage reprojected_s;    // RGBA16F
    TestImage disocclusion;     // R16F

    ReprojectFixture() : ctx(monti::test::SharedVulkanContext()) {}

    void Create() {
        auto device = ctx.Device();
        auto allocator = ctx.Allocator();
        constexpr VkImageUsageFlags kReadback = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

        // Create input images (no readback needed)
        motion_vectors = CreateTestImage(allocator, device, kTestWidth, kTestHeight,
                                         VK_FORMAT_R16G16_SFLOAT);
        prev_diffuse = CreateTestImage(allocator, device, kTestWidth, kTestHeight,
                                       VK_FORMAT_R16G16B16A16_SFLOAT);
        prev_specular = CreateTestImage(allocator, device, kTestWidth, kTestHeight,
                                        VK_FORMAT_R16G16B16A16_SFLOAT);
        prev_depth = CreateTestImage(allocator, device, kTestWidth, kTestHeight,
                                     VK_FORMAT_R16G16_SFLOAT);
        curr_depth = CreateTestImage(allocator, device, kTestWidth, kTestHeight,
                                     VK_FORMAT_R16G16_SFLOAT);

        // Create output images (need readback)
        reprojected_d = CreateTestImage(allocator, device, kTestWidth, kTestHeight,
                                        VK_FORMAT_R16G16B16A16_SFLOAT, kReadback);
        reprojected_s = CreateTestImage(allocator, device, kTestWidth, kTestHeight,
                                        VK_FORMAT_R16G16B16A16_SFLOAT, kReadback);
        disocclusion = CreateTestImage(allocator, device, kTestWidth, kTestHeight,
                                       VK_FORMAT_R16_SFLOAT, kReadback);

        // Transition output images to GENERAL
        VkCommandBuffer cmd = ctx.BeginOneShot();
        std::array<VkImageMemoryBarrier2, 3> barriers{};
        for (auto& b : barriers) {
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        barriers[0].image = reprojected_d.image;
        barriers[1].image = reprojected_s.image;
        barriers[2].image = disocclusion.image;

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
        dep.pImageMemoryBarriers = barriers.data();
        vkCmdPipelineBarrier2(cmd, &dep);
        ctx.SubmitAndWait(cmd);

        // Create reproject pipeline
        CreatePipeline();
    }

    void CreatePipeline() {
        auto device = ctx.Device();

        // Load SPIR-V
        std::string shader_path = std::string(DENI_SHADER_SPV_DIR) + "/reproject.comp.spv";
        auto spirv = LoadShaderFile(shader_path);
        REQUIRE(!spirv.empty());

        VkShaderModuleCreateInfo module_ci{};
        module_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        module_ci.codeSize = spirv.size();
        module_ci.pCode = reinterpret_cast<const uint32_t*>(spirv.data());
        REQUIRE(vkCreateShaderModule(device, &module_ci, nullptr, &shader_module) == VK_SUCCESS);

        // 8 storage image bindings matching reproject.comp
        std::array<VkDescriptorSetLayoutBinding, 8> bindings{};
        for (uint32_t i = 0; i < 8; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo ds_layout_ci{};
        ds_layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ds_layout_ci.bindingCount = static_cast<uint32_t>(bindings.size());
        ds_layout_ci.pBindings = bindings.data();
        REQUIRE(vkCreateDescriptorSetLayout(device, &ds_layout_ci, nullptr, &ds_layout) == VK_SUCCESS);

        // Push constants: width, height
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 2 * sizeof(uint32_t);

        VkPipelineLayoutCreateInfo layout_ci{};
        layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_ci.setLayoutCount = 1;
        layout_ci.pSetLayouts = &ds_layout;
        layout_ci.pushConstantRangeCount = 1;
        layout_ci.pPushConstantRanges = &push_range;
        REQUIRE(vkCreatePipelineLayout(device, &layout_ci, nullptr, &pipeline_layout) == VK_SUCCESS);

        VkComputePipelineCreateInfo pipeline_ci{};
        pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_ci.stage.module = shader_module;
        pipeline_ci.stage.pName = "main";
        pipeline_ci.layout = pipeline_layout;
        REQUIRE(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_ci,
                                         nullptr, &pipeline) == VK_SUCCESS);

        // Descriptor pool
        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        pool_size.descriptorCount = 8;

        VkDescriptorPoolCreateInfo pool_ci{};
        pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_ci.maxSets = 1;
        pool_ci.poolSizeCount = 1;
        pool_ci.pPoolSizes = &pool_size;
        REQUIRE(vkCreateDescriptorPool(device, &pool_ci, nullptr, &pool) == VK_SUCCESS);
    }

    void Dispatch() {
        auto device = ctx.Device();

        // Allocate descriptor set
        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &ds_layout;

        VkDescriptorSet ds = VK_NULL_HANDLE;
        REQUIRE(vkAllocateDescriptorSets(device, &alloc_info, &ds) == VK_SUCCESS);

        // Bind all 8 images
        std::array<VkDescriptorImageInfo, 8> image_infos{};
        image_infos[0] = {VK_NULL_HANDLE, motion_vectors.view, VK_IMAGE_LAYOUT_GENERAL};
        image_infos[1] = {VK_NULL_HANDLE, prev_diffuse.view, VK_IMAGE_LAYOUT_GENERAL};
        image_infos[2] = {VK_NULL_HANDLE, prev_specular.view, VK_IMAGE_LAYOUT_GENERAL};
        image_infos[3] = {VK_NULL_HANDLE, prev_depth.view, VK_IMAGE_LAYOUT_GENERAL};
        image_infos[4] = {VK_NULL_HANDLE, curr_depth.view, VK_IMAGE_LAYOUT_GENERAL};
        image_infos[5] = {VK_NULL_HANDLE, reprojected_d.view, VK_IMAGE_LAYOUT_GENERAL};
        image_infos[6] = {VK_NULL_HANDLE, reprojected_s.view, VK_IMAGE_LAYOUT_GENERAL};
        image_infos[7] = {VK_NULL_HANDLE, disocclusion.view, VK_IMAGE_LAYOUT_GENERAL};

        std::array<VkWriteDescriptorSet, 8> writes{};
        for (uint32_t i = 0; i < 8; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = ds;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i].pImageInfo = &image_infos[i];
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);

        // Record and dispatch
        VkCommandBuffer cmd = ctx.BeginOneShot();

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout, 0, 1, &ds, 0, nullptr);

        uint32_t pc[2] = {kTestWidth, kTestHeight};
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), pc);

        uint32_t groups_x = (kTestWidth + kWorkgroupSize - 1) / kWorkgroupSize;
        uint32_t groups_y = (kTestHeight + kWorkgroupSize - 1) / kWorkgroupSize;
        vkCmdDispatch(cmd, groups_x, groups_y, 1);

        ctx.SubmitAndWait(cmd);
    }

    void Destroy() {
        auto device = ctx.Device();
        auto allocator = ctx.Allocator();

        if (pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, pool, nullptr);
        if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
        if (pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        if (ds_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, ds_layout, nullptr);
        if (shader_module != VK_NULL_HANDLE) vkDestroyShaderModule(device, shader_module, nullptr);

        DestroyTestImage(allocator, device, motion_vectors);
        DestroyTestImage(allocator, device, prev_diffuse);
        DestroyTestImage(allocator, device, prev_specular);
        DestroyTestImage(allocator, device, prev_depth);
        DestroyTestImage(allocator, device, curr_depth);
        DestroyTestImage(allocator, device, reprojected_d);
        DestroyTestImage(allocator, device, reprojected_s);
        DestroyTestImage(allocator, device, disocclusion);
    }
};

}  // namespace

// ===========================================================================
// Test: Zero motion vectors produce identity reprojection
// ===========================================================================
TEST_CASE("Temporal reproject: zero MV produces identity reprojection",
          "[deni][temporal][reproject_identity]") {
    ReprojectFixture fix;
    fix.Create();

    // Fill prev_diffuse with a horizontal gradient
    std::vector<uint16_t> diffuse_data(kPixelCount * 4);
    for (uint32_t y = 0; y < kTestHeight; ++y) {
        for (uint32_t x = 0; x < kTestWidth; ++x) {
            uint32_t idx = (y * kTestWidth + x) * 4;
            float val = static_cast<float>(x) / static_cast<float>(kTestWidth - 1);
            diffuse_data[idx + 0] = FloatToHalf(val);
            diffuse_data[idx + 1] = FloatToHalf(0.0f);
            diffuse_data[idx + 2] = FloatToHalf(0.0f);
            diffuse_data[idx + 3] = FloatToHalf(1.0f);
        }
    }
    UploadImageData(fix.ctx, fix.prev_diffuse.image, diffuse_data.data(),
                    kTestWidth, kTestHeight, 4);

    // Fill prev_specular with constant
    std::vector<uint16_t> specular_data(kPixelCount * 4);
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        specular_data[i * 4 + 0] = FloatToHalf(0.5f);
        specular_data[i * 4 + 1] = FloatToHalf(0.5f);
        specular_data[i * 4 + 2] = FloatToHalf(0.5f);
        specular_data[i * 4 + 3] = FloatToHalf(1.0f);
    }
    UploadImageData(fix.ctx, fix.prev_specular.image, specular_data.data(),
                    kTestWidth, kTestHeight, 4);

    // Zero motion vectors (already zero from allocation, but be explicit)
    std::vector<uint16_t> mv_data(kPixelCount * 2, FloatToHalf(0.0f));
    UploadImageData(fix.ctx, fix.motion_vectors.image, mv_data.data(),
                    kTestWidth, kTestHeight, 2);

    // Matching depth: both prev and curr at 1.0
    std::vector<uint16_t> depth_data(kPixelCount * 2);
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        depth_data[i * 2 + 0] = FloatToHalf(1.0f);
        depth_data[i * 2 + 1] = FloatToHalf(0.0f);
    }
    UploadImageData(fix.ctx, fix.prev_depth.image, depth_data.data(),
                    kTestWidth, kTestHeight, 2);
    UploadImageData(fix.ctx, fix.curr_depth.image, depth_data.data(),
                    kTestWidth, kTestHeight, 2);

    fix.Dispatch();

    // Read back reprojected diffuse
    auto reproj_d = ReadbackImageData(fix.ctx, fix.reprojected_d.image,
                                      kTestWidth, kTestHeight, 4);

    // Read back disocclusion mask
    auto disocc = ReadbackImageData(fix.ctx, fix.disocclusion.image,
                                    kTestWidth, kTestHeight, 1);

    // Verify: reprojected diffuse should exactly match the gradient input
    uint32_t mismatches = 0;
    for (uint32_t y = 0; y < kTestHeight; ++y) {
        for (uint32_t x = 0; x < kTestWidth; ++x) {
            uint32_t idx = (y * kTestWidth + x) * 4;
            float expected = static_cast<float>(x) / static_cast<float>(kTestWidth - 1);
            float actual = HalfToFloat(reproj_d[idx + 0]);
            if (std::abs(actual - expected) > 0.001f) mismatches++;
        }
    }
    CHECK(mismatches == 0);

    // Verify: disocclusion mask should be 1.0 everywhere (no disocclusion)
    uint32_t disocc_mismatches = 0;
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        float val = HalfToFloat(disocc[i]);
        if (std::abs(val - 1.0f) > 0.001f) disocc_mismatches++;
    }
    CHECK(disocc_mismatches == 0);

    // Read back reprojected specular — should match constant 0.5
    auto reproj_s = ReadbackImageData(fix.ctx, fix.reprojected_s.image,
                                      kTestWidth, kTestHeight, 4);
    uint32_t spec_mismatches = 0;
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        float r = HalfToFloat(reproj_s[i * 4 + 0]);
        if (std::abs(r - 0.5f) > 0.001f) spec_mismatches++;
    }
    CHECK(spec_mismatches == 0);

    fix.Destroy();
}

// ===========================================================================
// Test: Known motion produces correct warp
// ===========================================================================
TEST_CASE("Temporal reproject: known MV produces correct warp",
          "[deni][temporal][reproject_shift]") {
    ReprojectFixture fix;
    fix.Create();

    // Fill prev_diffuse with a checkerboard: 1.0 at (x+y)%2==0, 0.0 otherwise
    std::vector<uint16_t> diffuse_data(kPixelCount * 4);
    for (uint32_t y = 0; y < kTestHeight; ++y) {
        for (uint32_t x = 0; x < kTestWidth; ++x) {
            uint32_t idx = (y * kTestWidth + x) * 4;
            float val = ((x + y) % 2 == 0) ? 1.0f : 0.0f;
            diffuse_data[idx + 0] = FloatToHalf(val);
            diffuse_data[idx + 1] = FloatToHalf(val);
            diffuse_data[idx + 2] = FloatToHalf(val);
            diffuse_data[idx + 3] = FloatToHalf(1.0f);
        }
    }
    UploadImageData(fix.ctx, fix.prev_diffuse.image, diffuse_data.data(),
                    kTestWidth, kTestHeight, 4);

    // prev_specular: all zeros
    std::vector<uint16_t> spec_data(kPixelCount * 4, FloatToHalf(0.0f));
    UploadImageData(fix.ctx, fix.prev_specular.image, spec_data.data(),
                    kTestWidth, kTestHeight, 4);

    // Motion vector: uniform (2/width, 0) — means current pixel was at (x-2, y) in prev frame
    // In normalized screen-space: mv.x = 2/width, mv.y = 0
    // Reprojection: prev_coord = curr_coord - mv_pixels = curr_coord - (mv * dims)
    constexpr int kShiftX = 2;
    constexpr int kShiftY = 0;
    float mv_x = static_cast<float>(kShiftX) / static_cast<float>(kTestWidth);
    float mv_y = static_cast<float>(kShiftY) / static_cast<float>(kTestHeight);
    std::vector<uint16_t> mv_data(kPixelCount * 2);
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        mv_data[i * 2 + 0] = FloatToHalf(mv_x);
        mv_data[i * 2 + 1] = FloatToHalf(mv_y);
    }
    UploadImageData(fix.ctx, fix.motion_vectors.image, mv_data.data(),
                    kTestWidth, kTestHeight, 2);

    // Matching depth
    std::vector<uint16_t> depth_data(kPixelCount * 2);
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        depth_data[i * 2 + 0] = FloatToHalf(1.0f);
        depth_data[i * 2 + 1] = FloatToHalf(0.0f);
    }
    UploadImageData(fix.ctx, fix.prev_depth.image, depth_data.data(),
                    kTestWidth, kTestHeight, 2);
    UploadImageData(fix.ctx, fix.curr_depth.image, depth_data.data(),
                    kTestWidth, kTestHeight, 2);

    fix.Dispatch();

    auto reproj_d = ReadbackImageData(fix.ctx, fix.reprojected_d.image,
                                      kTestWidth, kTestHeight, 4);
    auto disocc = ReadbackImageData(fix.ctx, fix.disocclusion.image,
                                    kTestWidth, kTestHeight, 1);

    // Interior pixels (x >= kShiftX) should sample from prev at (x - kShiftX, y)
    uint32_t interior_mismatches = 0;
    uint32_t interior_checked = 0;
    for (uint32_t y = 0; y < kTestHeight; ++y) {
        for (uint32_t x = static_cast<uint32_t>(kShiftX); x < kTestWidth; ++x) {
            uint32_t idx = (y * kTestWidth + x) * 4;
            uint32_t prev_x = x - kShiftX;
            float expected = ((prev_x + y) % 2 == 0) ? 1.0f : 0.0f;
            float actual = HalfToFloat(reproj_d[idx + 0]);
            if (std::abs(actual - expected) > 0.001f) interior_mismatches++;
            interior_checked++;
        }
    }
    CHECK(interior_mismatches == 0);
    CHECK(interior_checked > 0);

    // Border pixels (x < kShiftX) should be disoccluded (mask = 0.0)
    uint32_t border_disocc_wrong = 0;
    for (uint32_t y = 0; y < kTestHeight; ++y) {
        for (uint32_t x = 0; x < static_cast<uint32_t>(kShiftX); ++x) {
            float mask = HalfToFloat(disocc[y * kTestWidth + x]);
            if (std::abs(mask - 0.0f) > 0.001f) border_disocc_wrong++;
        }
    }
    CHECK(border_disocc_wrong == 0);

    // Interior pixels should have disocclusion = 1.0 (valid)
    uint32_t interior_disocc_wrong = 0;
    for (uint32_t y = 0; y < kTestHeight; ++y) {
        for (uint32_t x = static_cast<uint32_t>(kShiftX); x < kTestWidth; ++x) {
            float mask = HalfToFloat(disocc[y * kTestWidth + x]);
            if (std::abs(mask - 1.0f) > 0.001f) interior_disocc_wrong++;
        }
    }
    CHECK(interior_disocc_wrong == 0);

    fix.Destroy();
}

// ===========================================================================
// Test: Depth discontinuity produces disocclusion
// ===========================================================================
TEST_CASE("Temporal reproject: depth discontinuity detected",
          "[deni][temporal][reproject_disocclusion]") {
    ReprojectFixture fix;
    fix.Create();

    // prev_diffuse: all 1.0
    std::vector<uint16_t> diffuse_data(kPixelCount * 4);
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        diffuse_data[i * 4 + 0] = FloatToHalf(1.0f);
        diffuse_data[i * 4 + 1] = FloatToHalf(1.0f);
        diffuse_data[i * 4 + 2] = FloatToHalf(1.0f);
        diffuse_data[i * 4 + 3] = FloatToHalf(1.0f);
    }
    UploadImageData(fix.ctx, fix.prev_diffuse.image, diffuse_data.data(),
                    kTestWidth, kTestHeight, 4);

    // prev_specular: all 0.0
    std::vector<uint16_t> spec_data(kPixelCount * 4, FloatToHalf(0.0f));
    UploadImageData(fix.ctx, fix.prev_specular.image, spec_data.data(),
                    kTestWidth, kTestHeight, 4);

    // Zero motion vectors
    std::vector<uint16_t> mv_data(kPixelCount * 2, FloatToHalf(0.0f));
    UploadImageData(fix.ctx, fix.motion_vectors.image, mv_data.data(),
                    kTestWidth, kTestHeight, 2);

    // Previous depth: 1.0 everywhere
    std::vector<uint16_t> prev_depth_data(kPixelCount * 2);
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        prev_depth_data[i * 2 + 0] = FloatToHalf(1.0f);
        prev_depth_data[i * 2 + 1] = FloatToHalf(0.0f);
    }
    UploadImageData(fix.ctx, fix.prev_depth.image, prev_depth_data.data(),
                    kTestWidth, kTestHeight, 2);

    // Current depth: left half at 1.0 (matches prev), right half at 2.0 (>10% different)
    std::vector<uint16_t> curr_depth_data(kPixelCount * 2);
    for (uint32_t y = 0; y < kTestHeight; ++y) {
        for (uint32_t x = 0; x < kTestWidth; ++x) {
            uint32_t idx = (y * kTestWidth + x) * 2;
            float depth = (x < kTestWidth / 2) ? 1.0f : 2.0f;
            curr_depth_data[idx + 0] = FloatToHalf(depth);
            curr_depth_data[idx + 1] = FloatToHalf(0.0f);
        }
    }
    UploadImageData(fix.ctx, fix.curr_depth.image, curr_depth_data.data(),
                    kTestWidth, kTestHeight, 2);

    fix.Dispatch();

    auto disocc = ReadbackImageData(fix.ctx, fix.disocclusion.image,
                                    kTestWidth, kTestHeight, 1);

    // Left half: matching depth → disocclusion = 1.0
    uint32_t left_wrong = 0;
    for (uint32_t y = 0; y < kTestHeight; ++y) {
        for (uint32_t x = 0; x < kTestWidth / 2; ++x) {
            float mask = HalfToFloat(disocc[y * kTestWidth + x]);
            if (std::abs(mask - 1.0f) > 0.001f) left_wrong++;
        }
    }
    CHECK(left_wrong == 0);

    // Right half: depth ratio = 2.0/1.0 = 2.0, well above 10% → disocclusion = 0.0
    uint32_t right_wrong = 0;
    for (uint32_t y = 0; y < kTestHeight; ++y) {
        for (uint32_t x = kTestWidth / 2; x < kTestWidth; ++x) {
            float mask = HalfToFloat(disocc[y * kTestWidth + x]);
            if (std::abs(mask - 0.0f) > 0.001f) right_wrong++;
        }
    }
    CHECK(right_wrong == 0);

    fix.Destroy();
}

// ===========================================================================
// Test: Both lobes warped identically
// ===========================================================================
TEST_CASE("Temporal reproject: both lobes warped identically",
          "[deni][temporal][reproject_dual_lobe]") {
    ReprojectFixture fix;
    fix.Create();

    // prev_diffuse: all 1.0
    std::vector<uint16_t> diffuse_data(kPixelCount * 4);
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        diffuse_data[i * 4 + 0] = FloatToHalf(1.0f);
        diffuse_data[i * 4 + 1] = FloatToHalf(1.0f);
        diffuse_data[i * 4 + 2] = FloatToHalf(1.0f);
        diffuse_data[i * 4 + 3] = FloatToHalf(1.0f);
    }
    UploadImageData(fix.ctx, fix.prev_diffuse.image, diffuse_data.data(),
                    kTestWidth, kTestHeight, 4);

    // prev_specular: all 0.5
    std::vector<uint16_t> specular_data(kPixelCount * 4);
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        specular_data[i * 4 + 0] = FloatToHalf(0.5f);
        specular_data[i * 4 + 1] = FloatToHalf(0.5f);
        specular_data[i * 4 + 2] = FloatToHalf(0.5f);
        specular_data[i * 4 + 3] = FloatToHalf(1.0f);
    }
    UploadImageData(fix.ctx, fix.prev_specular.image, specular_data.data(),
                    kTestWidth, kTestHeight, 4);

    // MV: uniform (2/width, 0) shift
    constexpr int kShiftX = 2;
    float mv_x = static_cast<float>(kShiftX) / static_cast<float>(kTestWidth);
    std::vector<uint16_t> mv_data(kPixelCount * 2);
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        mv_data[i * 2 + 0] = FloatToHalf(mv_x);
        mv_data[i * 2 + 1] = FloatToHalf(0.0f);
    }
    UploadImageData(fix.ctx, fix.motion_vectors.image, mv_data.data(),
                    kTestWidth, kTestHeight, 2);

    // Matching depth
    std::vector<uint16_t> depth_data(kPixelCount * 2);
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        depth_data[i * 2 + 0] = FloatToHalf(1.0f);
        depth_data[i * 2 + 1] = FloatToHalf(0.0f);
    }
    UploadImageData(fix.ctx, fix.prev_depth.image, depth_data.data(),
                    kTestWidth, kTestHeight, 2);
    UploadImageData(fix.ctx, fix.curr_depth.image, depth_data.data(),
                    kTestWidth, kTestHeight, 2);

    fix.Dispatch();

    auto reproj_d = ReadbackImageData(fix.ctx, fix.reprojected_d.image,
                                      kTestWidth, kTestHeight, 4);
    auto reproj_s = ReadbackImageData(fix.ctx, fix.reprojected_s.image,
                                      kTestWidth, kTestHeight, 4);

    // Interior pixels (x >= kShiftX): diffuse should be 1.0, specular should be 0.5
    // Ratio specular/diffuse = 0.5 everywhere
    uint32_t ratio_mismatches = 0;
    uint32_t diffuse_wrong = 0;
    uint32_t specular_wrong = 0;
    for (uint32_t y = 0; y < kTestHeight; ++y) {
        for (uint32_t x = static_cast<uint32_t>(kShiftX); x < kTestWidth; ++x) {
            uint32_t idx = (y * kTestWidth + x) * 4;
            float d = HalfToFloat(reproj_d[idx + 0]);
            float s = HalfToFloat(reproj_s[idx + 0]);
            if (std::abs(d - 1.0f) > 0.001f) diffuse_wrong++;
            if (std::abs(s - 0.5f) > 0.001f) specular_wrong++;
            if (d > 0.01f) {
                float ratio = s / d;
                if (std::abs(ratio - 0.5f) > 0.01f) ratio_mismatches++;
            }
        }
    }
    CHECK(diffuse_wrong == 0);
    CHECK(specular_wrong == 0);
    CHECK(ratio_mismatches == 0);

    fix.Destroy();
}
