#pragma once

#include "../app/core/vulkan_context.h"
#include "../app/core/GBufferImages.h"
#include "../renderer/src/vulkan/Buffer.h"

#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <FLIP.h>
#include <glm/gtc/packing.hpp>
#include <stb_image_write.h>

#ifndef MONTI_SHADER_SPV_DIR
#define MONTI_SHADER_SPV_DIR "build/shaders"
#endif

#ifndef MONTI_TEST_ASSETS_DIR
#define MONTI_TEST_ASSETS_DIR "tests/assets"
#endif

namespace monti::test {

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

constexpr uint32_t kTestWidth = 256;
constexpr uint32_t kTestHeight = 256;
constexpr uint32_t kPixelCount = kTestWidth * kTestHeight;

// ═══════════════════════════════════════════════════════════════════════════
// Proc address helpers (volk-based, for tests only)
// ═══════════════════════════════════════════════════════════════════════════

inline void FillRendererProcAddrs(vulkan::RendererDesc& desc, const monti::app::VulkanContext& ctx) {
    desc.instance = ctx.Instance();
    desc.get_device_proc_addr = ctx.GetDeviceProcAddr();
    desc.get_instance_proc_addr = ctx.GetInstanceProcAddr();
}

inline vulkan::GpuBufferProcs MakeGpuBufferProcs() {
    return {vkGetBufferDeviceAddress, vkCmdPipelineBarrier2};
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
// Half-float conversion (delegates to GLM)
// ═══════════════════════════════════════════════════════════════════════════

inline float HalfToFloat(uint16_t h) { return glm::unpackHalf1x16(h); }
inline uint16_t FloatToHalf(float f) { return glm::packHalf1x16(f); }

// ═══════════════════════════════════════════════════════════════════════════
// GPU image readback
// ═══════════════════════════════════════════════════════════════════════════

// Read back an RGBA16F G-buffer image into a CPU staging buffer.
inline vulkan::Buffer ReadbackImage(monti::app::VulkanContext& ctx,
                                    VkImage image,
                                    VkDeviceSize pixel_size = 8,
                                    VkPipelineStageFlags2 src_stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR) {
    VkDeviceSize readback_size = kTestWidth * kTestHeight * pixel_size;

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
    region.imageExtent = {kTestWidth, kTestHeight, 1};
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
        r = std::pow(std::max(r, 0.0f), 1.0f / 2.2f);
        g = std::pow(std::max(g, 0.0f), 1.0f / 2.2f);
        b = std::pow(std::max(b, 0.0f), 1.0f / 2.2f);
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
        r = std::pow(std::max(r, 0.0f), 1.0f / 2.2f);
        g = std::pow(std::max(g, 0.0f), 1.0f / 2.2f);
        b = std::pow(std::max(b, 0.0f), 1.0f / 2.2f);
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
inline MultiFrameResult RenderSceneMultiFrame(
    monti::app::VulkanContext& ctx,
    monti::Scene& scene,
    std::span<const MeshData> mesh_data,
    uint32_t num_frames,
    uint32_t spp_per_frame)
{
    vulkan::RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = kTestWidth;
    desc.height = kTestHeight;
    desc.samples_per_pixel = spp_per_frame;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;
    FillRendererProcAddrs(desc, ctx);

    auto renderer = vulkan::Renderer::Create(desc);
    renderer->SetScene(&scene);

    auto procs = MakeGpuBufferProcs();
    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = vulkan::UploadAndRegisterMeshes(
        *renderer, ctx.Allocator(), ctx.Device(), upload_cmd, mesh_data, procs);
    ctx.SubmitAndWait(upload_cmd);

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                          kTestWidth, kTestHeight, gbuf_cmd,
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = MakeGBuffer(gbuffer_images);

    constexpr uint32_t kChannels = 4;
    std::vector<float> accum_diffuse(kPixelCount * kChannels, 0.0f);
    std::vector<float> accum_specular(kPixelCount * kChannels, 0.0f);

    for (uint32_t frame = 0; frame < num_frames; ++frame) {
        VkCommandBuffer render_cmd = ctx.BeginOneShot();
        renderer->RenderFrame(render_cmd, gbuffer, frame);
        ctx.SubmitAndWait(render_cmd);

        auto diffuse_rb = ReadbackImage(ctx, gbuffer_images.NoisyDiffuseImage());
        auto specular_rb = ReadbackImage(ctx, gbuffer_images.NoisySpecularImage());
        auto* d_raw = static_cast<uint16_t*>(diffuse_rb.Map());
        auto* s_raw = static_cast<uint16_t*>(specular_rb.Map());

        for (uint32_t i = 0; i < kPixelCount * kChannels; ++i) {
            accum_diffuse[i] += HalfToFloat(d_raw[i]);
            accum_specular[i] += HalfToFloat(s_raw[i]);
        }

        diffuse_rb.Unmap();
        specular_rb.Unmap();
    }

    // Average and convert back to fp16
    float inv_frames = 1.0f / static_cast<float>(num_frames);
    std::vector<uint16_t> out_diffuse(kPixelCount * kChannels);
    std::vector<uint16_t> out_specular(kPixelCount * kChannels);
    for (uint32_t i = 0; i < kPixelCount * kChannels; ++i) {
        out_diffuse[i] = FloatToHalf(accum_diffuse[i] * inv_frames);
        out_specular[i] = FloatToHalf(accum_specular[i] * inv_frames);
    }

    return MultiFrameResult{
        std::move(out_diffuse),
        std::move(out_specular),
        std::move(gpu_buffers)};
}

inline void CleanupMultiFrameResult(VmaAllocator allocator,
                                     MultiFrameResult& result) {
    for (auto& buf : result.gpu_buffers)
        vulkan::DestroyGpuBuffer(allocator, buf);
}

}  // namespace monti::test
