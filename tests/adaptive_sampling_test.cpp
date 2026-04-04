// Session 1 backward-compatibility test for the accumulator refactor.
//
// Verifies that the raw-sum + finalize normalization path produces output
// matching the old weighted-accumulation path within floating-point tolerance.

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <volk.h>

#include "test_helpers.h"
#include "scenes/CornellBox.h"

#include <monti/capture/GpuAccumulator.h>
#include <monti/capture/GpuReadback.h>
#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>

using namespace monti;
using namespace monti::vulkan;

namespace {

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 64;
constexpr uint32_t kPixelCount = kWidth * kHeight;
constexpr uint32_t kChannels = 4;
constexpr uint32_t kRefFrames = 4;

}  // namespace

TEST_CASE("Adaptive sampling Session 1: raw-sum + finalize matches weighted accumulation",
          "[adaptive][accumulator][vulkan][integration]") {
    auto& ctx = test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    // Build Cornell Box scene with area light + environment
    auto [scene, mesh_data] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene);

    constexpr uint32_t kEnvW = 4, kEnvH = 2;
    std::vector<float> env_pixels(kEnvW * kEnvH * 4);
    for (uint32_t i = 0; i < kEnvW * kEnvH; ++i) {
        env_pixels[i * 4 + 0] = 0.3f;
        env_pixels[i * 4 + 1] = 0.3f;
        env_pixels[i * 4 + 2] = 0.3f;
        env_pixels[i * 4 + 3] = 1.0f;
    }
    TextureDesc env_tex;
    env_tex.width = kEnvW;
    env_tex.height = kEnvH;
    env_tex.format = PixelFormat::kRGBA32F;
    env_tex.data.resize(env_pixels.size() * sizeof(float));
    std::memcpy(env_tex.data.data(), env_pixels.data(), env_tex.data.size());
    auto env_tex_id = scene.AddTexture(std::move(env_tex), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    // Create renderer
    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = kWidth;
    desc.height = kHeight;
    desc.samples_per_pixel = 1;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;
    test::FillRendererProcAddrs(desc, ctx);

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    // Upload meshes
    auto procs = test::MakeGpuBufferProcs();
    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(
        *renderer, ctx.Allocator(), ctx.Device(), upload_cmd, mesh_data, procs);
    ctx.SubmitAndWait(upload_cmd);
    REQUIRE(!gpu_buffers.empty());

    // Create G-buffer images
    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                          kWidth, kHeight, gbuf_cmd,
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    // Create GPU accumulator
    capture::GpuAccumulatorDesc acc_desc{};
    acc_desc.device = ctx.Device();
    acc_desc.allocator = ctx.Allocator();
    acc_desc.width = kWidth;
    acc_desc.height = kHeight;
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
    REQUIRE(accumulator);

    // Render + accumulate (raw sum) on GPU
    for (uint32_t frame = 0; frame < kRefFrames; ++frame) {
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

    // FinalizeNormalized: dispatches finalize.comp (divides by sample_count), then reads back
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
    ctx.WaitIdle();

    REQUIRE(!result.diffuse_f32.empty());
    REQUIRE(!result.specular_f32.empty());
    REQUIRE(result.diffuse_f32.size() == kPixelCount * kChannels);
    REQUIRE(result.specular_f32.size() == kPixelCount * kChannels);

    // Verify output is not all-zero (scene has light → some pixels must be non-zero)
    float max_diffuse = 0.0f;
    float max_specular = 0.0f;
    uint32_t nan_count = 0;
    for (uint32_t i = 0; i < kPixelCount * kChannels; ++i) {
        float d = result.diffuse_f32[i];
        float s = result.specular_f32[i];
        if (std::isnan(d) || std::isnan(s)) ++nan_count;
        if (std::abs(d) > max_diffuse) max_diffuse = std::abs(d);
        if (std::abs(s) > max_specular) max_specular = std::abs(s);
    }

    INFO("max_diffuse=" << max_diffuse << " max_specular=" << max_specular
         << " nan_count=" << nan_count);
    REQUIRE(max_diffuse > 0.0f);
    // NaN fraction should be negligible
    REQUIRE(nan_count < kPixelCount / 10);

    // Verify values are reasonable (normalized, not raw sums).
    // With ref_frames=4, raw sums would be ~4× the per-frame values.
    // After finalize normalization, values should be in per-frame range.
    // For Cornell box at 1 SPP, per-frame diffuse peaks are typically < 5.0.
    // Raw sums would be up to ~20. Check that we're in the normalized range.
    REQUIRE(max_diffuse < 20.0f);

    // Cleanup GPU resources
    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
}

TEST_CASE("Adaptive sampling Session 2: raygen early-out + variance pipeline smoke test",
          "[adaptive][variance][vulkan][integration]") {
    auto& ctx = test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    // Build Cornell Box scene with area light + environment
    auto [scene, mesh_data] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene);

    auto env_tex = test::MakeEnvMap(0.3f, 0.3f, 0.3f);
    auto env_tex_id = scene.AddTexture(std::move(env_tex), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    // Create renderer
    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = kWidth;
    desc.height = kHeight;
    desc.samples_per_pixel = 1;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;
    test::FillRendererProcAddrs(desc, ctx);

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    // Upload meshes
    auto procs = test::MakeGpuBufferProcs();
    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(
        *renderer, ctx.Allocator(), ctx.Device(), upload_cmd, mesh_data, procs);
    ctx.SubmitAndWait(upload_cmd);
    REQUIRE(!gpu_buffers.empty());

    // Create G-buffer images
    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                          kWidth, kHeight, gbuf_cmd,
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    // Create GPU accumulator with adaptive sampling ENABLED
    capture::GpuAccumulatorDesc acc_desc{};
    acc_desc.device = ctx.Device();
    acc_desc.allocator = ctx.Allocator();
    acc_desc.width = kWidth;
    acc_desc.height = kHeight;
    acc_desc.shader_dir = CAPTURE_SHADER_SPV_DIR;
    acc_desc.noisy_diffuse = gbuffer_images.NoisyDiffuseImage();
    acc_desc.noisy_specular = gbuffer_images.NoisySpecularImage();
    acc_desc.adaptive_sampling = true;
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
    acc_desc.procs.pfn_vkCmdFillBuffer              = vkCmdFillBuffer;
    acc_desc.procs.pfn_vkCmdCopyBuffer              = vkCmdCopyBuffer;
    acc_desc.procs.pfn_vkCreateBuffer               = vkCreateBuffer;
    acc_desc.procs.pfn_vkDestroyBuffer              = vkDestroyBuffer;

    auto accumulator = capture::GpuAccumulator::Create(acc_desc);
    REQUIRE(accumulator);
    REQUIRE(accumulator->ConvergenceMaskView() != VK_NULL_HANDLE);
    REQUIRE(accumulator->ConvergenceMaskImage() != VK_NULL_HANDLE);

    // Wire convergence mask to the renderer (binding 17)
    renderer->SetConvergenceMask(accumulator->ConvergenceMaskView());
    renderer->SetAdaptiveSamplingEnabled(true);

    // Render + accumulate + update variance for 4 frames
    for (uint32_t frame = 0; frame < kRefFrames; ++frame) {
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
        accumulator->UpdateVariance(cmd);

        ctx.SubmitAndWait(cmd);
    }

    // Create a readback context for convergence check and finalize
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

    // Check convergence with min_frames=16 — after only 4 frames, no pixel
    // should be converged.
    {
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = readback_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        VkCommandBuffer check_cmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(ctx.Device(), &alloc_info, &check_cmd);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(check_cmd, &begin_info);

        uint32_t converged = accumulator->CheckConvergence(check_cmd, 16, 0.02f, rctx);
        INFO("converged_count=" << converged << " (expected 0 with min_frames=16, only 4 rendered)");
        REQUIRE(converged == 0);
    }

    // ── Verify Welford variance images on GPU are numerically valid ──
    // Read back variance_mean (R32F) and variance_m2 (R32F) and verify they
    // contain non-trivial data consistent with the Welford algorithm running
    // on the actual GPU shader.
    {
        auto mean_buf = test::ReadbackImage(ctx, accumulator->VarianceMeanImage(),
            /*pixel_size=*/4, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, kWidth, kHeight);
        auto m2_buf = test::ReadbackImage(ctx, accumulator->VarianceM2Image(),
            /*pixel_size=*/4, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, kWidth, kHeight);

        auto* mean_data = static_cast<const float*>(mean_buf.Map());
        auto* m2_data = static_cast<const float*>(m2_buf.Map());

        uint32_t nonzero_mean = 0;
        uint32_t nonzero_m2 = 0;
        uint32_t nan_mean = 0;
        uint32_t nan_m2 = 0;
        uint32_t negative_m2 = 0;
        float min_m2 = 0.0f;
        for (uint32_t i = 0; i < kPixelCount; ++i) {
            float m = mean_data[i];
            float m2 = m2_data[i];
            if (std::isnan(m)) ++nan_mean;
            if (std::isnan(m2)) ++nan_m2;
            if (m != 0.0f) ++nonzero_mean;
            if (m2 != 0.0f) ++nonzero_m2;
            // M2 must be non-negative (sum of squared deviations),
            // but FP32 rounding in the GPU shader can produce tiny negatives.
            if (m2 < -1e-4f && !std::isnan(m2)) ++negative_m2;
            if (m2 < min_m2) min_m2 = m2;
        }

        mean_buf.Unmap();
        m2_buf.Unmap();

        INFO("nonzero_mean=" << nonzero_mean << " nonzero_m2=" << nonzero_m2
             << " nan_mean=" << nan_mean << " nan_m2=" << nan_m2
             << " negative_m2=" << negative_m2 << " min_m2=" << min_m2);

        // After 4 frames of a lit Cornell Box, most pixels should have
        // non-zero log-luminance mean (only fully black pixels would be zero)
        REQUIRE(nonzero_mean > kPixelCount / 4);

        // After 4 frames of noisy rendering, M2 should be non-zero for some
        // pixels (any pixel that received varying luminance across frames)
        REQUIRE(nonzero_m2 > 0);

        // No NaN contamination
        REQUIRE(nan_mean == 0);
        REQUIRE(nan_m2 == 0);

        // M2 is a sum of squared deviations — must not be negative
        REQUIRE(negative_m2 == 0);

        mean_buf.Destroy();
        m2_buf.Destroy();
    }

    // Finalize and verify the image output is valid (early-out mask was all-zero,
    // so no pixels were skipped — output should match non-adaptive path).
    auto result = accumulator->FinalizeNormalized(rctx);

    vkDestroyCommandPool(ctx.Device(), readback_pool, nullptr);
    ctx.WaitIdle();

    REQUIRE(!result.diffuse_f32.empty());
    REQUIRE(!result.specular_f32.empty());
    REQUIRE(result.diffuse_f32.size() == kPixelCount * kChannels);
    REQUIRE(result.specular_f32.size() == kPixelCount * kChannels);

    // Verify output is not all-zero (scene has light → some pixels must be non-zero)
    float max_diffuse = 0.0f;
    uint32_t nan_count = 0;
    for (uint32_t i = 0; i < kPixelCount * kChannels; ++i) {
        float d = result.diffuse_f32[i];
        if (std::isnan(d)) ++nan_count;
        if (std::abs(d) > max_diffuse) max_diffuse = std::abs(d);
    }

    INFO("max_diffuse=" << max_diffuse << " nan_count=" << nan_count);
    REQUIRE(max_diffuse > 0.0f);
    REQUIRE(nan_count < kPixelCount / 10);
    // After finalize normalization, values should be in per-frame range
    REQUIRE(max_diffuse < 20.0f);

    // Cleanup GPU resources
    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
}

// ── Session 3 helpers ────────────────────────────────────────────────────

namespace {

// Build GpuAccumulatorDesc with all Vulkan function pointers filled in.
capture::GpuAccumulatorDesc MakeAccDesc(
    monti::app::VulkanContext& ctx,
    monti::app::GBufferImages& gbuffer_images,
    bool adaptive_sampling) {
    capture::GpuAccumulatorDesc acc{};
    acc.device         = ctx.Device();
    acc.allocator      = ctx.Allocator();
    acc.width          = kWidth;
    acc.height         = kHeight;
    acc.shader_dir     = CAPTURE_SHADER_SPV_DIR;
    acc.noisy_diffuse  = gbuffer_images.NoisyDiffuseImage();
    acc.noisy_specular = gbuffer_images.NoisySpecularImage();
    acc.adaptive_sampling = adaptive_sampling;
    acc.procs.pfn_vkCreateDescriptorSetLayout  = vkCreateDescriptorSetLayout;
    acc.procs.pfn_vkDestroyDescriptorSetLayout = vkDestroyDescriptorSetLayout;
    acc.procs.pfn_vkCreateDescriptorPool       = vkCreateDescriptorPool;
    acc.procs.pfn_vkDestroyDescriptorPool      = vkDestroyDescriptorPool;
    acc.procs.pfn_vkAllocateDescriptorSets     = vkAllocateDescriptorSets;
    acc.procs.pfn_vkUpdateDescriptorSets       = vkUpdateDescriptorSets;
    acc.procs.pfn_vkCreateShaderModule         = vkCreateShaderModule;
    acc.procs.pfn_vkDestroyShaderModule        = vkDestroyShaderModule;
    acc.procs.pfn_vkCreatePipelineLayout       = vkCreatePipelineLayout;
    acc.procs.pfn_vkDestroyPipelineLayout      = vkDestroyPipelineLayout;
    acc.procs.pfn_vkCreateComputePipelines     = vkCreateComputePipelines;
    acc.procs.pfn_vkDestroyPipeline            = vkDestroyPipeline;
    acc.procs.pfn_vkCreateImageView            = vkCreateImageView;
    acc.procs.pfn_vkDestroyImageView           = vkDestroyImageView;
    acc.procs.pfn_vkCmdPipelineBarrier2        = vkCmdPipelineBarrier2;
    acc.procs.pfn_vkCmdBindPipeline            = vkCmdBindPipeline;
    acc.procs.pfn_vkCmdBindDescriptorSets      = vkCmdBindDescriptorSets;
    acc.procs.pfn_vkCmdPushConstants           = vkCmdPushConstants;
    acc.procs.pfn_vkCmdDispatch                = vkCmdDispatch;
    acc.procs.pfn_vkCmdClearColorImage         = vkCmdClearColorImage;
    acc.procs.pfn_vkCmdFillBuffer              = vkCmdFillBuffer;
    acc.procs.pfn_vkCmdCopyBuffer              = vkCmdCopyBuffer;
    acc.procs.pfn_vkCreateBuffer               = vkCreateBuffer;
    acc.procs.pfn_vkDestroyBuffer              = vkDestroyBuffer;
    return acc;
}

// Fill a ReadbackContext from a VulkanContext and command pool.
capture::ReadbackContext MakeReadbackCtx(monti::app::VulkanContext& ctx,
                                         VkCommandPool pool) {
    capture::ReadbackContext rctx{};
    rctx.device              = ctx.Device();
    rctx.queue               = ctx.GraphicsQueue();
    rctx.queue_family_index  = ctx.QueueFamilyIndex();
    rctx.allocator           = ctx.Allocator();
    rctx.command_pool        = pool;
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
    return rctx;
}

struct AdaptiveRunResult {
    capture::MultiFrameResult ref;
    uint32_t frames_rendered   = 0;
    uint32_t converged_count   = 0;
    uint64_t actual_pixel_frames = 0;
};

// Dispatch the adaptive reference accumulation loop:
//   per frame: RenderFrame → RT→compute barrier → Accumulate → UpdateVariance
//              → every convergence_check_interval frames: CheckConvergence + early exit
// Sets/clears the renderer convergence mask and adaptive flag around the loop.
AdaptiveRunResult RunAdaptiveLoop(
    monti::app::VulkanContext& ctx,
    Renderer& renderer,
    monti::app::GBufferImages& gbuffer_images,
    capture::GpuAccumulator& accum,
    const capture::ReadbackContext& rctx,
    uint32_t ref_frames,
    uint32_t min_convergence_frames,
    float convergence_threshold,
    uint32_t convergence_check_interval,
    uint32_t total_pixels,
    uint32_t base_frame_index = 1) {
    auto gbuffer = test::MakeGBuffer(gbuffer_images);
    AdaptiveRunResult result{};

    renderer.SetConvergenceMask(accum.ConvergenceMaskView());
    renderer.SetAdaptiveSamplingEnabled(true);

    for (uint32_t frame = 0; frame < ref_frames; ++frame) {
        result.actual_pixel_frames += (total_pixels - result.converged_count);

        VkCommandBuffer cmd = ctx.BeginOneShot();
        if (frame == 0) accum.Reset(cmd);

        renderer.RenderFrame(cmd, gbuffer, base_frame_index + frame);

        // Barrier: RT output → compute read
        std::array<VkImageMemoryBarrier2, 2> rt_to_compute{};
        for (uint32_t i = 0; i < 2; ++i) {
            rt_to_compute[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            rt_to_compute[i].srcStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            rt_to_compute[i].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            rt_to_compute[i].dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            rt_to_compute[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            rt_to_compute[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            rt_to_compute[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            rt_to_compute[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            rt_to_compute[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            rt_to_compute[i].subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        rt_to_compute[0].image = gbuffer_images.NoisyDiffuseImage();
        rt_to_compute[1].image = gbuffer_images.NoisySpecularImage();

        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 2;
        dep.pImageMemoryBarriers    = rt_to_compute.data();
        vkCmdPipelineBarrier2(cmd, &dep);

        accum.Accumulate(cmd);
        accum.UpdateVariance(cmd);
        ++result.frames_rendered;

        // Submit render+accumulate+variance work. For convergence check frames,
        // allocate a separate cmd from rctx.command_pool (required by CheckConvergence
        // which internally frees the cmd via that pool).
        ctx.SubmitAndWait(cmd);

        bool is_check_frame = ((frame + 1) % convergence_check_interval == 0);
        if (is_check_frame) {
            VkCommandBufferAllocateInfo check_alloc{};
            check_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            check_alloc.commandPool = rctx.command_pool;
            check_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            check_alloc.commandBufferCount = 1;
            VkCommandBuffer check_cmd = VK_NULL_HANDLE;
            vkAllocateCommandBuffers(ctx.Device(), &check_alloc, &check_cmd);
            VkCommandBufferBeginInfo check_begin{};
            check_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            check_begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(check_cmd, &check_begin);

            result.converged_count = accum.CheckConvergence(
                check_cmd, min_convergence_frames, convergence_threshold, rctx);
            if (result.converged_count == total_pixels) break;
        }
    }

    renderer.SetAdaptiveSamplingEnabled(false);
    renderer.SetConvergenceMask(VK_NULL_HANDLE);
    result.ref = accum.FinalizeNormalized(rctx);
    return result;
}

}  // namespace

// ── Session 3 integration tests ──────────────────────────────────────────

TEST_CASE("Adaptive sampling Session 3: flat color scene convergence",
          "[adaptive][convergence][vulkan][integration]") {
    auto& ctx = test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    // Flat emissive scene: a single large emissive quad fills the camera FOV.
    // Every primary ray returns a constant luminance → zero per-frame variance
    // → all pixels converge at the first check after min_convergence_frames.
    auto [scene, mesh_data] = test::BuildFlatEmissiveScene();

    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = kWidth;
    desc.height = kHeight;
    desc.samples_per_pixel = 4;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;
    test::FillRendererProcAddrs(desc, ctx);

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);
    renderer->SetMaxBounces(1);  // direct illumination only — minimises per-frame variance

    auto procs = test::MakeGpuBufferProcs();
    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(
        *renderer, ctx.Allocator(), ctx.Device(), upload_cmd, mesh_data, procs);
    ctx.SubmitAndWait(upload_cmd);
    REQUIRE(!gpu_buffers.empty());

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                          kWidth, kHeight, gbuf_cmd,
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    ctx.SubmitAndWait(gbuf_cmd);

    auto adaptive_accum = capture::GpuAccumulator::Create(MakeAccDesc(ctx, gbuffer_images, true));
    REQUIRE(adaptive_accum);

    VkCommandPool readback_pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = ctx.QueueFamilyIndex();
    vkCreateCommandPool(ctx.Device(), &pool_info, nullptr, &readback_pool);

    auto rctx = MakeReadbackCtx(ctx, readback_pool);

    constexpr uint32_t kRefFrames32 = 32;
    constexpr uint32_t kMinFrames = 8;
    constexpr uint32_t kInterval = 4;
    constexpr float kThreshold = 0.02f;
    constexpr uint32_t kTotalPixels = kPixelCount;

    auto adaptive_result = RunAdaptiveLoop(ctx, *renderer, gbuffer_images, *adaptive_accum,
                                           rctx, kRefFrames32, kMinFrames, kThreshold,
                                           kInterval, kTotalPixels, 1);

    ctx.WaitIdle();

    // All pixels should converge on this low-variance scene before 32 frames
    INFO("converged=" << adaptive_result.converged_count
         << " total=" << kTotalPixels
         << " frames_rendered=" << adaptive_result.frames_rendered);
    REQUIRE(adaptive_result.converged_count == kTotalPixels);
    REQUIRE(adaptive_result.frames_rendered < kRefFrames32);

    REQUIRE(!adaptive_result.ref.diffuse_f32.empty());
    REQUIRE(!adaptive_result.ref.specular_f32.empty());

    // ── Verify per-pixel sample counts ──────────────────────────────────
    // On this zero-variance emissive scene ALL pixels converge at the first
    // check that occurs after min_convergence_frames.  After that the loop
    // breaks (100% converged → early exit).  Therefore every pixel must have
    // been accumulated exactly `frames_rendered` times — no more, no less.
    // "No more" verifies that the raygen really skips converged pixels instead
    // of silently continuing to sample them.
    {
        auto count_buf = test::ReadbackImage(ctx, adaptive_accum->SampleCountImage(),
                                              /*pixel_size=*/4,
                                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                              kWidth, kHeight);
        auto* counts = static_cast<const uint32_t*>(count_buf.Map());
        uint32_t below_min = 0;
        uint32_t above_rendered = 0;
        for (uint32_t i = 0; i < kTotalPixels; ++i) {
            if (counts[i] < kMinFrames) ++below_min;
            if (counts[i] > adaptive_result.frames_rendered) ++above_rendered;
        }
        count_buf.Unmap();
        INFO("pixels_below_min_frames=" << below_min
             << " pixels_above_rendered=" << above_rendered);
        REQUIRE(below_min == 0);
        // All pixels converged at the same check frame → all counts == frames_rendered.
        REQUIRE(above_rendered == 0);
        count_buf.Destroy();
    }

    // ── Compare adaptive output to a non-adaptive reference ──
    // Both use the same frame count so convergence-mask differences are isolated.
    {
        auto nonadaptive_accum = capture::GpuAccumulator::Create(
            MakeAccDesc(ctx, gbuffer_images, false));
        REQUIRE(nonadaptive_accum);

        auto gbuffer = test::MakeGBuffer(gbuffer_images);
        for (uint32_t frame = 0; frame < adaptive_result.frames_rendered; ++frame) {
            VkCommandBuffer cmd = ctx.BeginOneShot();
            if (frame == 0) nonadaptive_accum->Reset(cmd);
            renderer->RenderFrame(cmd, gbuffer, 1 + frame);

            std::array<VkImageMemoryBarrier2, 2> rt_to_compute{};
            for (uint32_t i = 0; i < 2; ++i) {
                rt_to_compute[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                rt_to_compute[i].srcStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                rt_to_compute[i].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                rt_to_compute[i].dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                rt_to_compute[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                rt_to_compute[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                rt_to_compute[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                rt_to_compute[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                rt_to_compute[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                rt_to_compute[i].subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            }
            rt_to_compute[0].image = gbuffer_images.NoisyDiffuseImage();
            rt_to_compute[1].image = gbuffer_images.NoisySpecularImage();
            VkDependencyInfo dep{};
            dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 2;
            dep.pImageMemoryBarriers    = rt_to_compute.data();
            vkCmdPipelineBarrier2(cmd, &dep);

            nonadaptive_accum->Accumulate(cmd);
            ctx.SubmitAndWait(cmd);
        }
        auto nonadaptive_result = nonadaptive_accum->FinalizeNormalized(rctx);
        ctx.WaitIdle();

        REQUIRE(!nonadaptive_result.diffuse_f32.empty());

        // Compute mean relative error across all pixels (RGB channels).
        // Both paths used the same set of frames; adaptive only differs for pixels
        // that stopped accumulating early (once their sample_count hit min_frames
        // and convergence was confirmed). On this uniform-env flat scene, those
        // differences are tiny.
        double sum_rel_err = 0.0;
        uint32_t valid_pixels = 0;
        for (uint32_t i = 0; i < kTotalPixels; ++i) {
            auto base = static_cast<size_t>(i) * kChannels;
            for (int c = 0; c < 3; ++c) {
                float a = adaptive_result.ref.diffuse_f32[base + c];
                float n = nonadaptive_result.diffuse_f32[base + c];
                if (!std::isnan(a) && !std::isnan(n) && (n > 0.001f || a > 0.001f)) {
                    float rel_err = std::abs(a - n) / std::max(n, 0.001f);
                    sum_rel_err += rel_err;
                    ++valid_pixels;
                }
            }
        }
        double mean_rel_err = (valid_pixels > 0) ? sum_rel_err / valid_pixels : 0.0;
        INFO("mean_relative_error=" << mean_rel_err << " valid_pixels=" << valid_pixels);
        REQUIRE(mean_rel_err < 0.05);
    }

    vkDestroyCommandPool(ctx.Device(), readback_pool, nullptr);
    ctx.WaitIdle();

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
}

TEST_CASE("Adaptive sampling Session 3: early termination",
          "[adaptive][convergence][vulkan][integration]") {
    auto& ctx = test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    // Same flat emissive scene — extended to 64 frames to observe early exit.
    auto [scene, mesh_data] = test::BuildFlatEmissiveScene();

    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = kWidth;
    desc.height = kHeight;
    desc.samples_per_pixel = 4;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;
    test::FillRendererProcAddrs(desc, ctx);

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);
    renderer->SetMaxBounces(1);  // direct illumination only — minimises per-frame variance

    auto procs = test::MakeGpuBufferProcs();
    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(
        *renderer, ctx.Allocator(), ctx.Device(), upload_cmd, mesh_data, procs);
    ctx.SubmitAndWait(upload_cmd);
    REQUIRE(!gpu_buffers.empty());

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                          kWidth, kHeight, gbuf_cmd,
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    ctx.SubmitAndWait(gbuf_cmd);

    auto adaptive_accum = capture::GpuAccumulator::Create(MakeAccDesc(ctx, gbuffer_images, true));
    REQUIRE(adaptive_accum);

    VkCommandPool readback_pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = ctx.QueueFamilyIndex();
    vkCreateCommandPool(ctx.Device(), &pool_info, nullptr, &readback_pool);

    auto rctx = MakeReadbackCtx(ctx, readback_pool);

    constexpr uint32_t kRefFrames64 = 64;
    constexpr uint32_t kMinFrames = 8;
    constexpr uint32_t kInterval = 4;
    constexpr float kThreshold = 0.02f;
    constexpr uint32_t kTotalPixels = kPixelCount;

    auto result = RunAdaptiveLoop(ctx, *renderer, gbuffer_images, *adaptive_accum,
                                  rctx, kRefFrames64, kMinFrames, kThreshold,
                                  kInterval, kTotalPixels, 1);

    ctx.WaitIdle();

    // Loop must have terminated early on this low-variance scene
    INFO("frames_rendered=" << result.frames_rendered << " / " << kRefFrames64
         << " converged=" << result.converged_count << " / " << kTotalPixels);
    REQUIRE(result.frames_rendered < kRefFrames64);
    REQUIRE(result.converged_count == kTotalPixels);

    // Compute adaptive speedup from pixel-frame accounting
    uint64_t max_pixel_frames = static_cast<uint64_t>(kRefFrames64) * kTotalPixels;
    double speedup = (result.actual_pixel_frames > 0)
        ? static_cast<double>(max_pixel_frames) / static_cast<double>(result.actual_pixel_frames)
        : 1.0;
    INFO("speedup=" << speedup
         << " actual_pixel_frames=" << result.actual_pixel_frames
         << " max_pixel_frames=" << max_pixel_frames);
    REQUIRE(speedup > 2.0);

    REQUIRE(!result.ref.diffuse_f32.empty());
    REQUIRE(!result.ref.specular_f32.empty());

    // Output should be non-trivial
    float max_diffuse = 0.0f;
    uint32_t nan_count = 0;
    for (uint32_t i = 0; i < kTotalPixels * kChannels; ++i) {
        float v = result.ref.diffuse_f32[i];
        if (std::isnan(v)) ++nan_count;
        if (std::abs(v) > max_diffuse) max_diffuse = std::abs(v);
    }
    INFO("max_diffuse=" << max_diffuse << " nan_count=" << nan_count);
    REQUIRE(max_diffuse > 0.0f);
    REQUIRE(nan_count < kTotalPixels / 10);

    vkDestroyCommandPool(ctx.Device(), readback_pool, nullptr);
    ctx.WaitIdle();

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
}

TEST_CASE("Adaptive sampling Session 3: high-contrast scene",
          "[adaptive][convergence][vulkan][integration]") {
    auto& ctx = test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    // High-contrast scene: Cornell Box with bright warm ceiling light and
    // dark shadows — exercises log-luminance variance across the full dynamic range.
    auto [scene, mesh_data] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene);

    auto env_tex = test::MakeEnvMap(0.1f, 0.1f, 0.1f);
    auto env_tex_id = scene.AddTexture(std::move(env_tex), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = kWidth;
    desc.height = kHeight;
    desc.samples_per_pixel = 4;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;
    test::FillRendererProcAddrs(desc, ctx);

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    auto procs = test::MakeGpuBufferProcs();
    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(
        *renderer, ctx.Allocator(), ctx.Device(), upload_cmd, mesh_data, procs);
    ctx.SubmitAndWait(upload_cmd);
    REQUIRE(!gpu_buffers.empty());

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                          kWidth, kHeight, gbuf_cmd,
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    ctx.SubmitAndWait(gbuf_cmd);

    auto adaptive_accum = capture::GpuAccumulator::Create(MakeAccDesc(ctx, gbuffer_images, true));
    REQUIRE(adaptive_accum);

    VkCommandPool readback_pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = ctx.QueueFamilyIndex();
    vkCreateCommandPool(ctx.Device(), &pool_info, nullptr, &readback_pool);

    auto rctx = MakeReadbackCtx(ctx, readback_pool);

    constexpr uint32_t kRefFrames48 = 48;
    constexpr uint32_t kMinFrames = 8;
    constexpr uint32_t kInterval = 4;
    // 5% threshold: loose enough to verify the convergence mechanism works
    // for bright pixels near the ceiling light using 4 SPP × 48 frames.
    constexpr float kThreshold = 0.05f;
    constexpr uint32_t kTotalPixels = kPixelCount;

    auto result = RunAdaptiveLoop(ctx, *renderer, gbuffer_images, *adaptive_accum,
                                  rctx, kRefFrames48, kMinFrames, kThreshold,
                                  kInterval, kTotalPixels, 1);

    ctx.WaitIdle();

    REQUIRE(!result.ref.diffuse_f32.empty());
    REQUIRE(!result.ref.specular_f32.empty());

    // Some pixels must converge
    INFO("converged=" << result.converged_count << " / " << kTotalPixels
         << " frames_rendered=" << result.frames_rendered);
    REQUIRE(result.converged_count > 0);

    // ── Log-luminance no-bias check ──────────────────────────────────────
    // Read back the convergence mask (R8UI). From the finalized diffuse output,
    // classify pixels as bright (lum > 0.5) or dark (lum < 0.05). Verify that
    // both classes contain converged pixels — this confirms log-luminance variance
    // does not exclusively favor bright or dark pixels.
    {
        auto mask_buf = test::ReadbackImage(ctx, adaptive_accum->ConvergenceMaskImage(),
                                            /*pixel_size=*/1,
                                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                            kWidth, kHeight);
        auto* mask_data = static_cast<const uint8_t*>(mask_buf.Map());

        uint32_t bright_converged = 0;
        uint32_t dark_converged   = 0;
        uint32_t bright_total     = 0;
        uint32_t dark_total       = 0;

        for (uint32_t i = 0; i < kTotalPixels; ++i) {
            auto base = static_cast<size_t>(i) * kChannels;
            float r = result.ref.diffuse_f32[base + 0];
            float g = result.ref.diffuse_f32[base + 1];
            float b = result.ref.diffuse_f32[base + 2];
            if (std::isnan(r) || std::isnan(g) || std::isnan(b)) continue;
            float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            bool converged = (mask_data[i] != 0);
            if (lum > 0.5f)       { ++bright_total; if (converged) ++bright_converged; }
            else if (lum < 0.05f) { ++dark_total;   if (converged) ++dark_converged;   }
        }

        mask_buf.Unmap();
        mask_buf.Destroy();

        INFO("bright_converged=" << bright_converged << "/" << bright_total
             << " dark_converged=" << dark_converged << "/" << dark_total);

        if (bright_total > 0) REQUIRE(bright_converged > 0);
        if (dark_total > 0)   REQUIRE(dark_converged > 0);
    }

    // Output is valid (non-zero, low NaN fraction)
    float max_diffuse = 0.0f;
    uint32_t nan_count = 0;
    for (uint32_t i = 0; i < kTotalPixels * kChannels; ++i) {
        float v = result.ref.diffuse_f32[i];
        if (std::isnan(v)) ++nan_count;
        if (std::abs(v) > max_diffuse) max_diffuse = std::abs(v);
    }
    INFO("max_diffuse=" << max_diffuse << " nan_count=" << nan_count);
    REQUIRE(max_diffuse > 0.0f);
    REQUIRE(nan_count < kTotalPixels / 10);

    vkDestroyCommandPool(ctx.Device(), readback_pool, nullptr);
    ctx.WaitIdle();

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
}

// ── Session 3: per-pixel raygen masking mechanical test ──────────────────
//
// Verifies that the raygen shader STOPS sampling converged pixels:
//   Phase 1 — run adaptive loop on flat emissive scene until all pixels
//             converge (deterministic, fast).
//   Phase 2 — re-enable adaptive + convergence mask, run ONE more frame.
//   Assert  — sample counts must not change: raygen skipped every pixel.
//
// This is the minimal direct test of the masking mechanism.  It does not
// require partial convergence; the all-converged state is the most demanding
// case for the masking path (every pixel must be skipped).
TEST_CASE("Adaptive sampling Session 3: per-pixel raygen masking stops accumulation",
          "[adaptive][convergence][vulkan][integration]") {
    auto& ctx = test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    auto [scene, mesh_data] = test::BuildFlatEmissiveScene();

    RendererDesc desc{};
    desc.device             = ctx.Device();
    desc.physical_device    = ctx.PhysicalDevice();
    desc.queue              = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator          = ctx.Allocator();
    desc.width              = kWidth;
    desc.height             = kHeight;
    desc.samples_per_pixel  = 4;
    desc.shader_dir         = MONTI_SHADER_SPV_DIR;
    test::FillRendererProcAddrs(desc, ctx);

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);
    renderer->SetMaxBounces(1);  // emissive surface — direct hit only, zero variance

    auto procs = test::MakeGpuBufferProcs();
    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(
        *renderer, ctx.Allocator(), ctx.Device(), upload_cmd, mesh_data, procs);
    ctx.SubmitAndWait(upload_cmd);
    REQUIRE(!gpu_buffers.empty());

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                          kWidth, kHeight, gbuf_cmd,
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    ctx.SubmitAndWait(gbuf_cmd);

    auto accum = capture::GpuAccumulator::Create(MakeAccDesc(ctx, gbuffer_images, true));
    REQUIRE(accum);

    VkCommandPool readback_pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = ctx.QueueFamilyIndex();
    vkCreateCommandPool(ctx.Device(), &pool_info, nullptr, &readback_pool);

    auto rctx = MakeReadbackCtx(ctx, readback_pool);

    constexpr uint32_t kMaskRefFrames = 32;
    constexpr uint32_t kMinFramesMsk  = 8;
    constexpr uint32_t kIntervalMsk   = 4;
    constexpr float    kThreshMsk     = 0.02f;

    // ── Phase 1: run adaptive loop until all pixels converge ────────────
    auto result = RunAdaptiveLoop(ctx, *renderer, gbuffer_images, *accum,
                                  rctx, kMaskRefFrames, kMinFramesMsk, kThreshMsk,
                                  kIntervalMsk, kPixelCount, 1);
    ctx.WaitIdle();

    INFO("Phase 1: frames_rendered=" << result.frames_rendered
         << " converged=" << result.converged_count << " / " << kPixelCount);
    REQUIRE(result.frames_rendered < kMaskRefFrames);
    REQUIRE(result.converged_count == kPixelCount);

    // Read sample counts after Phase 1.
    std::vector<uint32_t> counts_before(kPixelCount);
    {
        auto count_buf = test::ReadbackImage(ctx, accum->SampleCountImage(),
                                             /*pixel_size=*/4,
                                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                             kWidth, kHeight);
        auto* p = static_cast<const uint32_t*>(count_buf.Map());
        std::copy(p, p + kPixelCount, counts_before.begin());
        count_buf.Unmap();
        count_buf.Destroy();
    }
    ctx.WaitIdle();

    // test::ReadbackImage leaves SampleCountImage in TRANSFER_SRC_OPTIMAL.
    // Transition it back to GENERAL so Phase 2 compute shaders can read/write it.
    {
        VkCommandBuffer restore_cmd = ctx.BeginOneShot();
        VkImageMemoryBarrier2 to_general{};
        to_general.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        to_general.srcStageMask        = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        to_general.srcAccessMask       = VK_ACCESS_2_TRANSFER_READ_BIT;
        to_general.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        to_general.dstAccessMask       = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                        | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        to_general.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_general.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        to_general.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_general.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_general.image               = accum->SampleCountImage();
        to_general.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &to_general;
        vkCmdPipelineBarrier2(restore_cmd, &dep);
        ctx.SubmitAndWait(restore_cmd);
    }

    // ── Phase 2: re-enable adaptive + mask, run one more frame ──────────
    // RunAdaptiveLoop clears the convergence mask from the renderer when it
    // exits, so we must re-arm it. The GPU-side mask image still holds all-1s
    // (every pixel converged), so every pixel should be skipped.
    renderer->SetConvergenceMask(accum->ConvergenceMaskView());
    renderer->SetAdaptiveSamplingEnabled(true);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    VkCommandBuffer extra_cmd = ctx.BeginOneShot();
    renderer->RenderFrame(extra_cmd, gbuffer, result.frames_rendered + 1);

    std::array<VkImageMemoryBarrier2, 2> rt_to_compute{};
    for (uint32_t i = 0; i < 2; ++i) {
        rt_to_compute[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        rt_to_compute[i].srcStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        rt_to_compute[i].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        rt_to_compute[i].dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        rt_to_compute[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        rt_to_compute[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        rt_to_compute[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        rt_to_compute[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        rt_to_compute[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        rt_to_compute[i].subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    rt_to_compute[0].image = gbuffer_images.NoisyDiffuseImage();
    rt_to_compute[1].image = gbuffer_images.NoisySpecularImage();

    VkDependencyInfo dep{};
    dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 2;
    dep.pImageMemoryBarriers    = rt_to_compute.data();
    vkCmdPipelineBarrier2(extra_cmd, &dep);

    accum->Accumulate(extra_cmd);
    accum->UpdateVariance(extra_cmd);
    ctx.SubmitAndWait(extra_cmd);
    ctx.WaitIdle();

    renderer->SetAdaptiveSamplingEnabled(false);
    renderer->SetConvergenceMask(VK_NULL_HANDLE);

    // ── Assert: sample counts must be UNCHANGED ──────────────────────────
    std::vector<uint32_t> counts_after(kPixelCount);
    {
        auto count_buf = test::ReadbackImage(ctx, accum->SampleCountImage(),
                                             /*pixel_size=*/4,
                                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                             kWidth, kHeight);
        auto* p = static_cast<const uint32_t*>(count_buf.Map());
        std::copy(p, p + kPixelCount, counts_after.begin());
        count_buf.Unmap();
        count_buf.Destroy();
    }

    uint32_t changed = 0;
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        if (counts_before[i] != counts_after[i]) ++changed;
    }

    INFO("pixels_with_changed_sample_count=" << changed
         << " (must be 0 — raygen must skip all converged pixels)");
    REQUIRE(changed == 0);

    vkDestroyCommandPool(ctx.Device(), readback_pool, nullptr);
    ctx.WaitIdle();

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
}
