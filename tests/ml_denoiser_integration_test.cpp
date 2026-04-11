#include <catch2/catch_test_macros.hpp>

#include <volk.h>

#include "../app/core/vulkan_context.h"
#include "shared_context.h"
#include "../denoise/src/vulkan/WeightLoader.h"

#include <deni/vulkan/Denoiser.h>

#include <monti/capture/GpuReadback.h>

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

namespace fs = std::filesystem;

using monti::capture::HalfToFloat;
using monti::capture::FloatToHalf;

constexpr uint32_t kTestWidth = 64;
constexpr uint32_t kTestHeight = 64;
constexpr uint32_t kPixelCount = kTestWidth * kTestHeight;

const std::string kTestDir = "test_output/ml_denoiser_integration";

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
    image_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
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
                   const uint16_t* data, uint32_t width, uint32_t height,
                   uint32_t channels = 4) {
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

void WriteDeniModel(const std::string& path,
                    const std::vector<deni::vulkan::LayerWeights>& layers) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream file(path, std::ios::binary);
    REQUIRE(file.is_open());

    auto write_u32 = [&](uint32_t val) {
        file.write(reinterpret_cast<const char*>(&val), sizeof(uint32_t));
    };

    uint32_t total_bytes = 0;
    for (const auto& layer : layers)
        total_bytes += static_cast<uint32_t>(layer.data.size() * sizeof(float));

    file.write("DENI", 4);
    write_u32(1);
    write_u32(static_cast<uint32_t>(layers.size()));
    write_u32(total_bytes);

    for (const auto& layer : layers) {
        write_u32(static_cast<uint32_t>(layer.name.size()));
        file.write(layer.name.data(), layer.name.size());
        write_u32(static_cast<uint32_t>(layer.shape.size()));
        for (auto d : layer.shape) write_u32(d);
        file.write(reinterpret_cast<const char*>(layer.data.data()),
                   layer.data.size() * sizeof(float));
    }
}

std::vector<deni::vulkan::LayerWeights> MakeTestLayers() {
    std::vector<deni::vulkan::LayerWeights> layers;
    // V3 temporal: 2-level depthwise-separable U-Net.
    // base_channels=8, c0=8, c1=16. 26-ch input, 7-ch output.
    constexpr uint32_t c0 = 8;
    constexpr uint32_t c1 = 16;
    constexpr uint32_t in_ch = 26;
    constexpr uint32_t out_ch = 7;

    auto add_depthwise_separable = [&](const char* prefix, uint32_t ic, uint32_t oc) {
        // Depthwise conv: [ic, 1, 3, 3]
        deni::vulkan::LayerWeights dw;
        dw.name = std::string(prefix) + ".depthwise.weight";
        dw.shape = {ic, 1, 3, 3};
        dw.data.resize(ic * 9, 0.01f);
        layers.push_back(std::move(dw));

        // Pointwise conv: [oc, ic, 1, 1]
        deni::vulkan::LayerWeights pw;
        pw.name = std::string(prefix) + ".pointwise.weight";
        pw.shape = {oc, ic, 1, 1};
        pw.data.resize(oc * ic, 0.01f);
        layers.push_back(std::move(pw));

        deni::vulkan::LayerWeights pb;
        pb.name = std::string(prefix) + ".pointwise.bias";
        pb.shape = {oc};
        pb.data.resize(oc, 0.0f);
        layers.push_back(std::move(pb));
    };

    auto add_norm = [&](const char* name, uint32_t ch) {
        deni::vulkan::LayerWeights gamma;
        gamma.name = std::string(name) + ".weight";
        gamma.shape = {ch};
        gamma.data.resize(ch, 1.0f);
        layers.push_back(std::move(gamma));

        deni::vulkan::LayerWeights beta;
        beta.name = std::string(name) + ".bias";
        beta.shape = {ch};
        beta.data.resize(ch, 0.0f);
        layers.push_back(std::move(beta));
    };

    // Encoder level 0
    add_depthwise_separable("down0.conv1", in_ch, c0);
    add_norm("down0.conv1.norm", c0);
    add_depthwise_separable("down0.conv2", c0, c0);
    add_norm("down0.conv2.norm", c0);

    // Bottleneck (at half resolution)
    add_depthwise_separable("bottleneck1", c0, c1);
    add_norm("bottleneck1.norm", c1);
    add_depthwise_separable("bottleneck2", c1, c1);
    add_norm("bottleneck2.norm", c1);

    // Decoder level 0
    add_depthwise_separable("up0.conv1", c1 + c0, c0);
    add_norm("up0.conv1.norm", c0);
    add_depthwise_separable("up0.conv2", c0, c0);
    add_norm("up0.conv2.norm", c0);

    // Output 1x1 conv: [out_ch, c0, 1, 1]
    {
        deni::vulkan::LayerWeights w;
        w.name = "out_conv.weight";
        w.shape = {out_ch, c0, 1, 1};
        w.data.resize(out_ch * c0, 0.1f);
        layers.push_back(std::move(w));

        deni::vulkan::LayerWeights b;
        b.name = "out_conv.bias";
        b.shape = {out_ch};
        b.data.resize(out_ch, 0.0f);
        layers.push_back(std::move(b));
    }

    return layers;
}

struct TestGBuffer {
    TestImage diffuse;
    TestImage specular;
    TestImage motion;
    TestImage depth;
    TestImage normals;
    TestImage diff_albedo;
    TestImage spec_albedo;
    monti::app::VulkanContext* ctx_ = nullptr;

    ~TestGBuffer() {
        if (ctx_) Destroy(*ctx_);
    }

    void Create(monti::app::VulkanContext& ctx) {
        ctx_ = &ctx;
        auto format = VK_FORMAT_R16G16B16A16_SFLOAT;
        diffuse = CreateTestImage(ctx.Allocator(), ctx.Device(), kTestWidth, kTestHeight, format);
        specular = CreateTestImage(ctx.Allocator(), ctx.Device(), kTestWidth, kTestHeight, format);
        motion = CreateTestImage(ctx.Allocator(), ctx.Device(), kTestWidth, kTestHeight,
                                 VK_FORMAT_R16G16_SFLOAT);
        depth = CreateTestImage(ctx.Allocator(), ctx.Device(), kTestWidth, kTestHeight,
                                VK_FORMAT_R16G16_SFLOAT);
        normals = CreateTestImage(ctx.Allocator(), ctx.Device(), kTestWidth, kTestHeight, format);
        diff_albedo = CreateTestImage(ctx.Allocator(), ctx.Device(), kTestWidth, kTestHeight, format);
        spec_albedo = CreateTestImage(ctx.Allocator(), ctx.Device(), kTestWidth, kTestHeight, format);

        // Upload known data to diffuse and specular
        std::vector<uint16_t> diffuse_data(kPixelCount * 4);
        for (uint32_t i = 0; i < kPixelCount; ++i) {
            diffuse_data[i * 4 + 0] = FloatToHalf(0.3f);
            diffuse_data[i * 4 + 1] = FloatToHalf(0.1f);
            diffuse_data[i * 4 + 2] = FloatToHalf(0.2f);
            diffuse_data[i * 4 + 3] = FloatToHalf(1.0f);
        }
        std::vector<uint16_t> specular_data(kPixelCount * 4);
        for (uint32_t i = 0; i < kPixelCount; ++i) {
            specular_data[i * 4 + 0] = FloatToHalf(0.1f);
            specular_data[i * 4 + 1] = FloatToHalf(0.4f);
            specular_data[i * 4 + 2] = FloatToHalf(0.05f);
            specular_data[i * 4 + 3] = FloatToHalf(1.0f);
        }
        UploadRGBA16F(ctx, diffuse.image, diffuse_data.data(), kTestWidth, kTestHeight);
        UploadRGBA16F(ctx, specular.image, specular_data.data(), kTestWidth, kTestHeight);

        // Upload zeroed data for auxiliary images to avoid uninitialized reads
        std::vector<uint16_t> zeros(kPixelCount * 4, 0);
        UploadRGBA16F(ctx, normals.image, zeros.data(), kTestWidth, kTestHeight);
        UploadRGBA16F(ctx, depth.image, zeros.data(), kTestWidth, kTestHeight, 2);
        UploadRGBA16F(ctx, motion.image, zeros.data(), kTestWidth, kTestHeight, 2);
        UploadRGBA16F(ctx, diff_albedo.image, zeros.data(), kTestWidth, kTestHeight);
        UploadRGBA16F(ctx, spec_albedo.image, zeros.data(), kTestWidth, kTestHeight);
    }

    deni::vulkan::DenoiserInput ToInput() const {
        deni::vulkan::DenoiserInput input{};
        input.noisy_diffuse = diffuse.view;
        input.noisy_specular = specular.view;
        input.motion_vectors = motion.view;
        input.linear_depth = depth.view;
        input.linear_depth_image = depth.image;
        input.world_normals = normals.view;
        input.diffuse_albedo = diff_albedo.view;
        input.specular_albedo = spec_albedo.view;
        input.render_width = kTestWidth;
        input.render_height = kTestHeight;
        input.reset_accumulation = true;
        return input;
    }

    void Destroy(monti::app::VulkanContext& ctx) {
        DestroyTestImage(ctx.Allocator(), ctx.Device(), diffuse);
        DestroyTestImage(ctx.Allocator(), ctx.Device(), specular);
        DestroyTestImage(ctx.Allocator(), ctx.Device(), motion);
        DestroyTestImage(ctx.Allocator(), ctx.Device(), depth);
        DestroyTestImage(ctx.Allocator(), ctx.Device(), normals);
        DestroyTestImage(ctx.Allocator(), ctx.Device(), diff_albedo);
        DestroyTestImage(ctx.Allocator(), ctx.Device(), spec_albedo);
        ctx_ = nullptr;
    }
};

std::vector<uint16_t> ReadbackOutput(monti::app::VulkanContext& ctx,
                                     const deni::vulkan::DenoiserOutput& output) {
    VkDeviceSize readback_size = kPixelCount * 4 * sizeof(uint16_t);
    auto readback = CreateStagingBuffer(ctx.Allocator(), readback_size,
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
    to_src.image = output.denoised_image;
    to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &to_src;
    vkCmdPipelineBarrier2(cmd, &dep);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {kTestWidth, kTestHeight, 1};
    vkCmdCopyImageToBuffer(cmd, output.denoised_image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.buffer, 1, &region);

    ctx.SubmitAndWait(cmd);

    void* mapped = nullptr;
    REQUIRE(vmaMapMemory(ctx.Allocator(), readback.allocation, &mapped) == VK_SUCCESS);

    std::vector<uint16_t> pixels(kPixelCount * 4);
    std::memcpy(pixels.data(), mapped, kPixelCount * 4 * sizeof(uint16_t));
    vmaUnmapMemory(ctx.Allocator(), readback.allocation);

    DestroyStagingBuffer(ctx.Allocator(), readback);
    return pixels;
}

}  // namespace

TEST_CASE("ML denoiser integration: produces non-zero output", "[deni][integration]") {
    auto& ctx = monti::test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    auto layers = MakeTestLayers();
    std::string model_path = kTestDir + "/test_model.denimodel";
    WriteDeniModel(model_path, layers);

    TestGBuffer gbuf;
    gbuf.Create(ctx);

    deni::vulkan::DenoiserDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.width = kTestWidth;
    desc.height = kTestHeight;
    desc.allocator = ctx.Allocator();
    desc.shader_dir = DENI_SHADER_SPV_DIR;
    desc.get_device_proc_addr = vkGetDeviceProcAddr;
    desc.model_path = model_path;

    auto denoiser = deni::vulkan::Denoiser::Create(desc);
    REQUIRE(denoiser != nullptr);
    CHECK(denoiser->HasMlModel());
    CHECK(denoiser->Mode() == deni::vulkan::DenoiserMode::kMl);

    auto input = gbuf.ToInput();
    VkCommandBuffer cmd = ctx.BeginOneShot();
    auto output = denoiser->Denoise(cmd, input);
    ctx.SubmitAndWait(cmd);

    REQUIRE(output.denoised_image != VK_NULL_HANDLE);

    auto pixels = ReadbackOutput(ctx, output);

    // Verify non-zero and no NaN/Inf
    bool has_nonzero = false;
    bool has_nan_inf = false;
    for (uint32_t i = 0; i < kPixelCount * 4; ++i) {
        float val = HalfToFloat(pixels[i]);
        if (val != 0.0f) has_nonzero = true;
        if (std::isnan(val) || std::isinf(val)) has_nan_inf = true;
    }
    CHECK(has_nonzero);
    CHECK_FALSE(has_nan_inf);

    // Timing should be populated
    CHECK(denoiser->LastPassTimeMs() >= 0.0f);

    gbuf.Destroy(ctx);
    denoiser.reset();
    ctx.WaitIdle();
    fs::remove(model_path);
}

TEST_CASE("ML denoiser integration: mode switching", "[deni][integration]") {
    auto& ctx = monti::test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    auto layers = MakeTestLayers();
    std::string model_path = kTestDir + "/test_model_mode.denimodel";
    WriteDeniModel(model_path, layers);

    TestGBuffer gbuf;
    gbuf.Create(ctx);

    deni::vulkan::DenoiserDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.width = kTestWidth;
    desc.height = kTestHeight;
    desc.allocator = ctx.Allocator();
    desc.shader_dir = DENI_SHADER_SPV_DIR;
    desc.get_device_proc_addr = vkGetDeviceProcAddr;
    desc.model_path = model_path;

    auto denoiser = deni::vulkan::Denoiser::Create(desc);
    REQUIRE(denoiser != nullptr);
    CHECK(denoiser->Mode() == deni::vulkan::DenoiserMode::kMl);

    auto input = gbuf.ToInput();

    // Run ML denoise
    VkCommandBuffer cmd1 = ctx.BeginOneShot();
    auto output_ml = denoiser->Denoise(cmd1, input);
    ctx.SubmitAndWait(cmd1);
    auto pixels_ml = ReadbackOutput(ctx, output_ml);

    // Switch to passthrough
    denoiser->SetMode(deni::vulkan::DenoiserMode::kPassthrough);
    CHECK(denoiser->Mode() == deni::vulkan::DenoiserMode::kPassthrough);

    VkCommandBuffer cmd2 = ctx.BeginOneShot();
    auto output_pt = denoiser->Denoise(cmd2, input);
    ctx.SubmitAndWait(cmd2);
    auto pixels_pt = ReadbackOutput(ctx, output_pt);

    // Passthrough should be diffuse+specular, ML output should differ
    // (random weights produce different result than simple addition)
    bool outputs_differ = false;
    for (uint32_t i = 0; i < kPixelCount * 4; ++i) {
        if (pixels_ml[i] != pixels_pt[i]) {
            outputs_differ = true;
            break;
        }
    }
    CHECK(outputs_differ);

    // Switch back to ML
    denoiser->SetMode(deni::vulkan::DenoiserMode::kMl);
    CHECK(denoiser->Mode() == deni::vulkan::DenoiserMode::kMl);

    gbuf.Destroy(ctx);
    denoiser.reset();
    ctx.WaitIdle();
    fs::remove(model_path);
}

TEST_CASE("ML denoiser integration: graceful fallback without model", "[deni][integration]") {
    auto& ctx = monti::test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    TestGBuffer gbuf;
    gbuf.Create(ctx);

    deni::vulkan::DenoiserDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.width = kTestWidth;
    desc.height = kTestHeight;
    desc.allocator = ctx.Allocator();
    desc.shader_dir = DENI_SHADER_SPV_DIR;
    desc.get_device_proc_addr = vkGetDeviceProcAddr;
    // No model_path — passthrough only

    auto denoiser = deni::vulkan::Denoiser::Create(desc);
    REQUIRE(denoiser != nullptr);
    // Force passthrough mode (auto-discovery may have found a model)
    REQUIRE(denoiser->SetMode(deni::vulkan::DenoiserMode::kPassthrough));
    CHECK(denoiser->Mode() == deni::vulkan::DenoiserMode::kPassthrough);

    auto input = gbuf.ToInput();
    VkCommandBuffer cmd = ctx.BeginOneShot();
    auto output = denoiser->Denoise(cmd, input);
    ctx.SubmitAndWait(cmd);

    REQUIRE(output.denoised_image != VK_NULL_HANDLE);

    auto pixels = ReadbackOutput(ctx, output);

    // Passthrough sums diffuse + specular
    constexpr float kExpectedR = 0.3f + 0.1f;
    constexpr float kExpectedG = 0.1f + 0.4f;
    constexpr float kExpectedB = 0.2f + 0.05f;
    constexpr float kTolerance = 0.002f;

    float r = HalfToFloat(pixels[0]);
    float g = HalfToFloat(pixels[1]);
    float b = HalfToFloat(pixels[2]);
    CHECK(std::abs(r - kExpectedR) < kTolerance);
    CHECK(std::abs(g - kExpectedG) < kTolerance);
    CHECK(std::abs(b - kExpectedB) < kTolerance);

    gbuf.Destroy(ctx);
    denoiser.reset();
    ctx.WaitIdle();
}
