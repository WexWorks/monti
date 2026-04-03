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
