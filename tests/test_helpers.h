#pragma once

#include "shared_context.h"

#include "../app/core/vulkan_context.h"
#include "../app/core/GBufferImages.h"
#include "../renderer/src/vulkan/Buffer.h"

#include <monti/capture/GpuAccumulator.h>
#include <monti/capture/GpuReadback.h>
#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>
#include <monti/vulkan/ProcAddrHelpers.h>
#include <deni/vulkan/Denoiser.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <FLIP.h>
#include <stb_image_write.h>

#ifndef MONTI_SHADER_SPV_DIR
#define MONTI_SHADER_SPV_DIR "build/shaders"
#endif

#ifndef CAPTURE_SHADER_SPV_DIR
#define CAPTURE_SHADER_SPV_DIR "build/capture_shaders"
#endif

// Fallback only — CMake normally sets this to an absolute path.
// The relative default assumes CWD is the project root.
#ifndef MONTI_TEST_ASSETS_DIR
#define MONTI_TEST_ASSETS_DIR "scenes/khronos"
#endif

#ifndef MONTI_DEBUG_SCENES_DIR
#define MONTI_DEBUG_SCENES_DIR "scenes/debug"
#endif

namespace monti::test {

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

constexpr uint32_t kTestWidth = 256;
constexpr uint32_t kTestHeight = 256;
constexpr uint32_t kPixelCount = kTestWidth * kTestHeight;

// ═══════════════════════════════════════════════════════════════════════════
// Proc address helpers — thin wrappers around monti::vulkan::ProcAddrHelpers
// ═══════════════════════════════════════════════════════════════════════════

inline void FillRendererProcAddrs(vulkan::RendererDesc& desc, const monti::app::VulkanContext& ctx) {
    vulkan::FillRendererProcAddrs(desc, ctx.Instance(),
                                  ctx.GetDeviceProcAddr(), ctx.GetInstanceProcAddr());
}

inline vulkan::GpuBufferProcs MakeGpuBufferProcs() {
    return vulkan::MakeGpuBufferProcs(vkGetBufferDeviceAddress, vkCmdPipelineBarrier2);
}

inline void FillDenoiserProcAddrs(deni::vulkan::DenoiserDesc& desc,
                                  const monti::app::VulkanContext& ctx) {
    vulkan::FillDenoiserProcAddrs(desc, ctx.GetDeviceProcAddr());
}

// ═══════════════════════════════════════════════════════════════════════════
// G-buffer helpers
// ═══════════════════════════════════════════════════════════════════════════

// Populate a GBuffer struct from GBufferImages (views + images).
inline vulkan::GBuffer MakeGBuffer(const monti::app::GBufferImages& images) {
    vulkan::GBuffer gb{};
    gb.noisy_diffuse   = images.NoisyDiffuseView();
    gb.noisy_specular  = images.NoisySpecularView();
    gb.motion_vectors  = images.MotionVectorsView();
    gb.linear_depth    = images.LinearDepthView();
    gb.world_normals   = images.WorldNormalsView();
    gb.diffuse_albedo  = images.DiffuseAlbedoView();
    gb.specular_albedo = images.SpecularAlbedoView();
    gb.noisy_diffuse_image   = images.NoisyDiffuseImage();
    gb.noisy_specular_image  = images.NoisySpecularImage();
    gb.motion_vectors_image  = images.MotionVectorsImage();
    gb.linear_depth_image    = images.LinearDepthImage();
    gb.world_normals_image   = images.WorldNormalsImage();
    gb.diffuse_albedo_image  = images.DiffuseAlbedoImage();
    gb.specular_albedo_image = images.SpecularAlbedoImage();
    return gb;
}

// ═══════════════════════════════════════════════════════════════════════════
// Half-float conversion — single definition in capture::, reused here
// ═══════════════════════════════════════════════════════════════════════════

using capture::HalfToFloat;
using capture::FloatToHalf;

// ═══════════════════════════════════════════════════════════════════════════
// GPU image readback
// ═══════════════════════════════════════════════════════════════════════════

// Read back an RGBA16F G-buffer image into a CPU staging buffer.
inline vulkan::Buffer ReadbackImage(monti::app::VulkanContext& ctx,
                                    VkImage image,
                                    VkDeviceSize pixel_size = 8,
                                    VkPipelineStageFlags2 src_stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                    uint32_t width = kTestWidth,
                                    uint32_t height = kTestHeight) {
    VkDeviceSize readback_size = width * height * pixel_size;

    vulkan::Buffer readback;
    readback.Create(ctx.Allocator(), readback_size,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_CPU_ONLY);

    VkCommandBuffer copy_cmd = ctx.BeginOneShot();

    VkImageMemoryBarrier2 to_src{};
    to_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_src.srcStageMask = src_stage;
    to_src.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    to_src.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    to_src.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    to_src.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_src.image = image;
    to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &to_src;
    vkCmdPipelineBarrier2(copy_cmd, &dep);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    vkCmdCopyImageToBuffer(copy_cmd, image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.Handle(), 1, &region);

    ctx.SubmitAndWait(copy_cmd);
    return readback;
}

// ═══════════════════════════════════════════════════════════════════════════
// Pixel analysis
// ═══════════════════════════════════════════════════════════════════════════

struct PixelStats {
    uint32_t nan_count = 0;
    uint32_t inf_count = 0;
    uint32_t nonzero_count = 0;
    bool has_color_variation = false;
    double sum_r = 0, sum_g = 0, sum_b = 0;
    uint32_t valid_count = 0;
};

// Analyze an RGBA16F readback buffer.
inline PixelStats AnalyzeRGBA16F(const uint16_t* raw, uint32_t pixel_count) {
    PixelStats stats{};
    float prev_r = -1.0f;
    for (uint32_t i = 0; i < pixel_count; ++i) {
        float r = HalfToFloat(raw[i * 4 + 0]);
        float g = HalfToFloat(raw[i * 4 + 1]);
        float b = HalfToFloat(raw[i * 4 + 2]);

        if (std::isnan(r) || std::isnan(g) || std::isnan(b)) { ++stats.nan_count; continue; }
        if (std::isinf(r) || std::isinf(g) || std::isinf(b)) { ++stats.inf_count; continue; }

        stats.sum_r += r;
        stats.sum_g += g;
        stats.sum_b += b;
        ++stats.valid_count;

        if (r + g + b > 0.0f) ++stats.nonzero_count;
        if (prev_r >= 0.0f && std::abs(r - prev_r) > 0.001f)
            stats.has_color_variation = true;
        prev_r = r;
    }
    return stats;
}

// ═══════════════════════════════════════════════════════════════════════════
// Tone-mapping and FLIP
// ═══════════════════════════════════════════════════════════════════════════

// Convert RGBA16F diffuse + specular to interleaved linear RGB floats for FLIP.
// Applies Reinhard tone-mapping so values fall in [0,1].
inline std::vector<float> TonemappedRGB(const uint16_t* diffuse_raw,
                                        const uint16_t* specular_raw,
                                        uint32_t pixel_count) {
    std::vector<float> rgb(pixel_count * 3);
    for (uint32_t i = 0; i < pixel_count; ++i) {
        float r = HalfToFloat(diffuse_raw[i * 4 + 0])
                + HalfToFloat(specular_raw[i * 4 + 0]);
        float g = HalfToFloat(diffuse_raw[i * 4 + 1])
                + HalfToFloat(specular_raw[i * 4 + 1]);
        float b = HalfToFloat(diffuse_raw[i * 4 + 2])
                + HalfToFloat(specular_raw[i * 4 + 2]);
        if (std::isnan(r) || std::isinf(r)) r = 0.0f;
        if (std::isnan(g) || std::isinf(g)) g = 0.0f;
        if (std::isnan(b) || std::isinf(b)) b = 0.0f;
        r = std::max(r, 0.0f) / (1.0f + std::max(r, 0.0f));
        g = std::max(g, 0.0f) / (1.0f + std::max(g, 0.0f));
        b = std::max(b, 0.0f) / (1.0f + std::max(b, 0.0f));
        rgb[i * 3 + 0] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
    return rgb;
}

// Convert a single RGBA16F buffer to Reinhard-tonemapped linear RGB for FLIP.
inline std::vector<float> TonemappedRGB(const uint16_t* rgba16f_raw,
                                        uint32_t pixel_count) {
    std::vector<float> rgb(pixel_count * 3);
    for (uint32_t i = 0; i < pixel_count; ++i) {
        float r = HalfToFloat(rgba16f_raw[i * 4 + 0]);
        float g = HalfToFloat(rgba16f_raw[i * 4 + 1]);
        float b = HalfToFloat(rgba16f_raw[i * 4 + 2]);
        if (std::isnan(r) || std::isinf(r)) r = 0.0f;
        if (std::isnan(g) || std::isinf(g)) g = 0.0f;
        if (std::isnan(b) || std::isinf(b)) b = 0.0f;
        r = std::max(r, 0.0f) / (1.0f + std::max(r, 0.0f));
        g = std::max(g, 0.0f) / (1.0f + std::max(g, 0.0f));
        b = std::max(b, 0.0f) / (1.0f + std::max(b, 0.0f));
        rgb[i * 3 + 0] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
    return rgb;
}

// ═══════════════════════════════════════════════════════════════════════════
// Gamma conversion utilities
// ═══════════════════════════════════════════════════════════════════════════

// BT.709 luminance (matches luminance.comp shader).
inline float Luminance(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

inline float LinearToSRGB(float linear) {
    return std::pow(std::max(linear, 0.0f), 1.0f / 2.2f);
}

inline float SRGBToLinear(float srgb) {
    return std::pow(std::max(srgb, 0.0f), 2.2f);
}

// ═══════════════════════════════════════════════════════════════════════════
// Environment map helper
// ═══════════════════════════════════════════════════════════════════════════

// Create a solid-color 4x2 RGBA32F environment map
inline TextureDesc MakeEnvMap(float r, float g, float b) {
    constexpr uint32_t kW = 4, kH = 2;
    std::vector<float> pixels(kW * kH * 4);
    for (uint32_t i = 0; i < kW * kH; ++i) {
        pixels[i * 4 + 0] = r;
        pixels[i * 4 + 1] = g;
        pixels[i * 4 + 2] = b;
        pixels[i * 4 + 3] = 1.0f;
    }
    TextureDesc tex;
    tex.width = kW;
    tex.height = kH;
    tex.format = PixelFormat::kRGBA32F;
    tex.data.resize(pixels.size() * sizeof(float));
    std::memcpy(tex.data.data(), pixels.data(), tex.data.size());
    return tex;
}

inline float ComputeMeanFlip(const std::vector<float>& reference_rgb,
                              const std::vector<float>& test_rgb,
                              int width, int height) {
    FLIP::image<FLIP::color3> ref_img(width, height);
    FLIP::image<FLIP::color3> test_img(width, height);
    FLIP::image<float> error_map(width, height, 0.0f);

    ref_img.setPixels(reference_rgb.data(), width, height);
    test_img.setPixels(test_rgb.data(), width, height);

    FLIP::Parameters params;
    FLIP::evaluate(ref_img, test_img, false, params, error_map);

    float sum = 0.0f;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            sum += error_map.get(x, y);
    return sum / static_cast<float>(width * height);
}

// ═══════════════════════════════════════════════════════════════════════════
// PNG output
// ═══════════════════════════════════════════════════════════════════════════

// Write an RGBA16F readback buffer to a PNG file for visual inspection.
inline bool WritePNG(std::string_view path, const uint16_t* raw,
                     uint32_t width, uint32_t height) {
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());
    std::vector<uint8_t> pixels(width * height * 3);
    for (uint32_t i = 0; i < width * height; ++i) {
        float r = HalfToFloat(raw[i * 4 + 0]);
        float g = HalfToFloat(raw[i * 4 + 1]);
        float b = HalfToFloat(raw[i * 4 + 2]);
        r = r / (1.0f + r);
        g = g / (1.0f + g);
        b = b / (1.0f + b);
        r = LinearToSRGB(r);
        g = LinearToSRGB(g);
        b = LinearToSRGB(b);
        pixels[i * 3 + 0] = static_cast<uint8_t>(std::clamp(r * 255.0f + 0.5f, 0.0f, 255.0f));
        pixels[i * 3 + 1] = static_cast<uint8_t>(std::clamp(g * 255.0f + 0.5f, 0.0f, 255.0f));
        pixels[i * 3 + 2] = static_cast<uint8_t>(std::clamp(b * 255.0f + 0.5f, 0.0f, 255.0f));
    }
    std::string path_str(path);
    return stbi_write_png(path_str.c_str(), static_cast<int>(width),
                          static_cast<int>(height), 3, pixels.data(),
                          static_cast<int>(width * 3)) != 0;
}

// Write combined diffuse+specular RGBA16F to a PNG file.
inline bool WriteCombinedPNG(std::string_view path,
                             const uint16_t* diffuse_raw,
                             const uint16_t* specular_raw,
                             uint32_t width, uint32_t height) {
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());
    std::vector<uint8_t> pixels(width * height * 3);
    for (uint32_t i = 0; i < width * height; ++i) {
        float r = HalfToFloat(diffuse_raw[i * 4 + 0])
                + HalfToFloat(specular_raw[i * 4 + 0]);
        float g = HalfToFloat(diffuse_raw[i * 4 + 1])
                + HalfToFloat(specular_raw[i * 4 + 1]);
        float b = HalfToFloat(diffuse_raw[i * 4 + 2])
                + HalfToFloat(specular_raw[i * 4 + 2]);
        r = r / (1.0f + r);
        g = g / (1.0f + g);
        b = b / (1.0f + b);
        r = LinearToSRGB(r);
        g = LinearToSRGB(g);
        b = LinearToSRGB(b);
        pixels[i * 3 + 0] = static_cast<uint8_t>(std::clamp(r * 255.0f + 0.5f, 0.0f, 255.0f));
        pixels[i * 3 + 1] = static_cast<uint8_t>(std::clamp(g * 255.0f + 0.5f, 0.0f, 255.0f));
        pixels[i * 3 + 2] = static_cast<uint8_t>(std::clamp(b * 255.0f + 0.5f, 0.0f, 255.0f));
    }
    std::string path_str(path);
    return stbi_write_png(path_str.c_str(), static_cast<int>(width),
                          static_cast<int>(height), 3, pixels.data(),
                          static_cast<int>(width * 3)) != 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// Multi-frame rendering
// ═══════════════════════════════════════════════════════════════════════════

struct MultiFrameResult {
    std::vector<uint16_t> diffuse;   // RGBA16F, 4 half-floats per pixel
    std::vector<uint16_t> specular;  // RGBA16F, 4 half-floats per pixel
    std::vector<vulkan::GpuBuffer> gpu_buffers;
};

// Render a scene across multiple frames with different frame indices,
// accumulating results CPU-side in float. Each frame uses a different
// Halton sub-pixel jitter and blue noise temporal hash, producing
// proper anti-aliasing and decorrelated noise that a single high-SPP
// frame cannot achieve.
//
// Total samples = num_frames * spp_per_frame. For best quality, prefer
// many frames with low SPP (e.g., 16 frames x 4 SPP) over few frames
// with high SPP (e.g., 1 frame x 64 SPP).
//
// Uses GPU-side compute accumulation (same path as monti_datagen) to avoid
// per-frame readback overhead. Renders + accumulates in the same command
// buffer, with a single readback after all frames complete.
inline MultiFrameResult RenderSceneMultiFrame(
    monti::app::VulkanContext& ctx,
    monti::Scene& scene,
    std::span<const MeshData> mesh_data,
    uint32_t num_frames,
    uint32_t spp_per_frame,
    uint32_t width,
    uint32_t height)
{
    vulkan::RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = width;
    desc.height = height;
    desc.samples_per_pixel = spp_per_frame;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;
    FillRendererProcAddrs(desc, ctx);

    auto renderer = vulkan::Renderer::Create(desc);
    renderer->SetScene(&scene);

    // Synthesize visible geometry for rectangular area lights so they
    // appear when hit by camera/path rays (area lights are otherwise
    // virtual — sampled via NEE but invisible to ray intersections).
    auto light_meshes = SynthesizeAreaLightGeometry(scene);
    std::vector<MeshData> all_mesh_data(mesh_data.begin(), mesh_data.end());
    all_mesh_data.insert(all_mesh_data.end(),
        std::make_move_iterator(light_meshes.begin()),
        std::make_move_iterator(light_meshes.end()));

    auto procs = MakeGpuBufferProcs();
    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = vulkan::UploadAndRegisterMeshes(
        *renderer, ctx.Allocator(), ctx.Device(), upload_cmd, all_mesh_data, procs);
    ctx.SubmitAndWait(upload_cmd);

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                          width, height, gbuf_cmd,
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = MakeGBuffer(gbuffer_images);

    // Create GPU accumulator
    capture::GpuAccumulatorDesc acc_desc{};
    acc_desc.device = ctx.Device();
    acc_desc.allocator = ctx.Allocator();
    acc_desc.width = width;
    acc_desc.height = height;
    acc_desc.shader_dir = CAPTURE_SHADER_SPV_DIR;
    acc_desc.noisy_diffuse = gbuffer_images.NoisyDiffuseImage();
    acc_desc.noisy_specular = gbuffer_images.NoisySpecularImage();
    acc_desc.procs.pfn_vkCreateDescriptorSetLayout  = vkCreateDescriptorSetLayout;
    acc_desc.procs.pfn_vkDestroyDescriptorSetLayout = vkDestroyDescriptorSetLayout;
    acc_desc.procs.pfn_vkCreateDescriptorPool       = vkCreateDescriptorPool;
    acc_desc.procs.pfn_vkDestroyDescriptorPool      = vkDestroyDescriptorPool;
    acc_desc.procs.pfn_vkAllocateDescriptorSets     = vkAllocateDescriptorSets;
    acc_desc.procs.pfn_vkUpdateDescriptorSets       = vkUpdateDescriptorSets;
    acc_desc.procs.pfn_vkCreateShaderModule         = vkCreateShaderModule;
    acc_desc.procs.pfn_vkDestroyShaderModule        = vkDestroyShaderModule;
    acc_desc.procs.pfn_vkCreatePipelineLayout       = vkCreatePipelineLayout;
    acc_desc.procs.pfn_vkDestroyPipelineLayout      = vkDestroyPipelineLayout;
    acc_desc.procs.pfn_vkCreateComputePipelines     = vkCreateComputePipelines;
    acc_desc.procs.pfn_vkDestroyPipeline            = vkDestroyPipeline;
    acc_desc.procs.pfn_vkCreateImageView            = vkCreateImageView;
    acc_desc.procs.pfn_vkDestroyImageView           = vkDestroyImageView;
    acc_desc.procs.pfn_vkCmdPipelineBarrier2        = vkCmdPipelineBarrier2;
    acc_desc.procs.pfn_vkCmdBindPipeline            = vkCmdBindPipeline;
    acc_desc.procs.pfn_vkCmdBindDescriptorSets      = vkCmdBindDescriptorSets;
    acc_desc.procs.pfn_vkCmdPushConstants           = vkCmdPushConstants;
    acc_desc.procs.pfn_vkCmdDispatch                = vkCmdDispatch;
    acc_desc.procs.pfn_vkCmdClearColorImage         = vkCmdClearColorImage;

    auto accumulator = capture::GpuAccumulator::Create(acc_desc);
    if (!accumulator)
        throw std::runtime_error("Failed to create GPU accumulator");

    // Render + accumulate on GPU
    for (uint32_t frame = 0; frame < num_frames; ++frame) {
        VkCommandBuffer cmd = ctx.BeginOneShot();

        if (frame == 0) accumulator->Reset(cmd);

        renderer->RenderFrame(cmd, gbuffer, frame);

        // Barrier: RT output → compute read
        std::array<VkImageMemoryBarrier2, 2> rt_to_compute{};
        for (uint32_t i = 0; i < 2; ++i) {
            rt_to_compute[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            rt_to_compute[i].srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            rt_to_compute[i].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            rt_to_compute[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            rt_to_compute[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            rt_to_compute[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            rt_to_compute[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            rt_to_compute[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            rt_to_compute[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            rt_to_compute[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        rt_to_compute[0].image = gbuffer_images.NoisyDiffuseImage();
        rt_to_compute[1].image = gbuffer_images.NoisySpecularImage();

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 2;
        dep.pImageMemoryBarriers = rt_to_compute.data();
        vkCmdPipelineBarrier2(cmd, &dep);

        accumulator->Accumulate(cmd);

        ctx.SubmitAndWait(cmd);
    }

    // Single readback of accumulated result
    VkCommandPool readback_pool;
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = ctx.QueueFamilyIndex();
    vkCreateCommandPool(ctx.Device(), &pool_info, nullptr, &readback_pool);

    capture::ReadbackContext rctx{};
    rctx.device = ctx.Device();
    rctx.queue = ctx.GraphicsQueue();
    rctx.queue_family_index = ctx.QueueFamilyIndex();
    rctx.allocator = ctx.Allocator();
    rctx.command_pool = readback_pool;
    rctx.pfn_vkAllocateCommandBuffers = vkAllocateCommandBuffers;
    rctx.pfn_vkBeginCommandBuffer     = vkBeginCommandBuffer;
    rctx.pfn_vkEndCommandBuffer       = vkEndCommandBuffer;
    rctx.pfn_vkCmdPipelineBarrier2    = vkCmdPipelineBarrier2;
    rctx.pfn_vkCmdCopyImageToBuffer   = vkCmdCopyImageToBuffer;
    rctx.pfn_vkQueueSubmit            = vkQueueSubmit;
    rctx.pfn_vkCreateFence            = vkCreateFence;
    rctx.pfn_vkWaitForFences          = vkWaitForFences;
    rctx.pfn_vkDestroyFence           = vkDestroyFence;
    rctx.pfn_vkFreeCommandBuffers     = vkFreeCommandBuffers;

    auto result = accumulator->FinalizeNormalized(rctx);

    vkDestroyCommandPool(ctx.Device(), readback_pool, nullptr);

    // Ensure all GPU work is complete before RAII destroys renderer/accumulator
    // resources. Without this, VMA allocations freed during destruction may
    // still be referenced by in-flight work, corrupting the allocator state
    // and causing VK_ERROR_DEVICE_LOST on subsequent tests.
    ctx.WaitIdle();

    // Convert FP32 accumulated result back to FP16 for test consumption
    const uint32_t pixel_count = width * height;
    constexpr uint32_t kChannels = 4;
    std::vector<uint16_t> out_diffuse(pixel_count * kChannels);
    std::vector<uint16_t> out_specular(pixel_count * kChannels);
    for (uint32_t i = 0; i < pixel_count * kChannels; ++i) {
        out_diffuse[i] = FloatToHalf(result.diffuse_f32[i]);
        out_specular[i] = FloatToHalf(result.specular_f32[i]);
    }

    return MultiFrameResult{
        std::move(out_diffuse),
        std::move(out_specular),
        std::move(gpu_buffers)};
}

// Convenience overload using the default test resolution.
inline MultiFrameResult RenderSceneMultiFrame(
    monti::app::VulkanContext& ctx,
    monti::Scene& scene,
    std::span<const MeshData> mesh_data,
    uint32_t num_frames,
    uint32_t spp_per_frame)
{
    return RenderSceneMultiFrame(ctx, scene, mesh_data, num_frames,
                                spp_per_frame, kTestWidth, kTestHeight);
}

inline void CleanupMultiFrameResult(VmaAllocator allocator,
                                     MultiFrameResult& result) {
    for (auto& buf : result.gpu_buffers)
        vulkan::DestroyGpuBuffer(allocator, buf);
}

}  // namespace monti::test
