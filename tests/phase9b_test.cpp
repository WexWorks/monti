#include <catch2/catch_test_macros.hpp>

#include <volk.h>

#include "test_helpers.h"
#include "scenes/CornellBox.h"

#include <monti/scene/Scene.h>

#include <cstdio>
#include <vector>

using namespace monti;
using namespace monti::vulkan;

namespace {

struct TestContext {
    monti::app::VulkanContext& ctx = test::SharedVulkanContext();
    bool Init() { return ctx.Device() != VK_NULL_HANDLE; }
};

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Test: End-to-end denoiser integration — passthrough produces exact sum
//
// Renders a Cornell box through the renderer, then wires the G-buffer
// outputs into the Deni passthrough denoiser via a single memory barrier.
// Reads back the denoised output and the raw diffuse/specular channels,
// computes CPU-side diffuse + specular sum, and verifies the denoised output
// matches within ±1 ULP (GPU float→half rounding may differ from GLM's
// packHalf1x16).  FLIP is checked as a secondary gate (< 0.001).
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 9B: Denoiser integration - passthrough exact match",
          "[phase9b][denoiser][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // ── Build and upload Cornell box scene ──
    auto [scene, mesh_data] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene);

    RendererDesc renderer_desc{};
    renderer_desc.device = ctx.Device();
    renderer_desc.physical_device = ctx.PhysicalDevice();
    renderer_desc.queue = ctx.GraphicsQueue();
    renderer_desc.queue_family_index = ctx.QueueFamilyIndex();
    renderer_desc.allocator = ctx.Allocator();
    renderer_desc.width = test::kTestWidth;
    renderer_desc.height = test::kTestHeight;
    renderer_desc.samples_per_pixel = 4;
    renderer_desc.shader_dir = MONTI_SHADER_SPV_DIR;
    test::FillRendererProcAddrs(renderer_desc, ctx);

    auto renderer = Renderer::Create(renderer_desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    auto procs = test::MakeGpuBufferProcs();
    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(
        *renderer, ctx.Allocator(), ctx.Device(), upload_cmd, mesh_data, procs);
    REQUIRE_FALSE(gpu_buffers.empty());
    ctx.SubmitAndWait(upload_cmd);

    // ── Create G-buffer images ──
    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                                  test::kTestWidth, test::kTestHeight, gbuf_cmd,
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    // ── Create denoiser ──
    deni::vulkan::DenoiserDesc denoiser_desc{};
    denoiser_desc.device = ctx.Device();
    denoiser_desc.physical_device = ctx.PhysicalDevice();
    denoiser_desc.width = test::kTestWidth;
    denoiser_desc.height = test::kTestHeight;
    denoiser_desc.allocator = ctx.Allocator();
    denoiser_desc.shader_dir = DENI_SHADER_SPV_DIR;
    test::FillDenoiserProcAddrs(denoiser_desc, ctx);

    auto denoiser = deni::vulkan::Denoiser::Create(denoiser_desc);
    REQUIRE(denoiser != nullptr);

    // ── Record: render → barrier → denoise ──
    VkCommandBuffer cmd = ctx.BeginOneShot();

    REQUIRE(renderer->RenderFrame(cmd, gbuffer, 0));

    // Single memory barrier: ray tracing writes → compute reads
    VkMemoryBarrier2 rt_to_compute{};
    rt_to_compute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    rt_to_compute.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    rt_to_compute.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    rt_to_compute.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    rt_to_compute.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo barrier_dep{};
    barrier_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    barrier_dep.memoryBarrierCount = 1;
    barrier_dep.pMemoryBarriers = &rt_to_compute;
    vkCmdPipelineBarrier2(cmd, &barrier_dep);

    // Populate DenoiserInput from GBuffer views
    deni::vulkan::DenoiserInput input{};
    input.noisy_diffuse = gbuffer.noisy_diffuse;
    input.noisy_specular = gbuffer.noisy_specular;
    input.motion_vectors = gbuffer.motion_vectors;
    input.linear_depth = gbuffer.linear_depth;
    input.world_normals = gbuffer.world_normals;
    input.diffuse_albedo = gbuffer.diffuse_albedo;
    input.specular_albedo = gbuffer.specular_albedo;
    input.render_width = test::kTestWidth;
    input.render_height = test::kTestHeight;
    input.reset_accumulation = true;

    auto output = denoiser->Denoise(cmd, input);

    ctx.SubmitAndWait(cmd);

    REQUIRE(output.denoised_image != VK_NULL_HANDLE);
    REQUIRE(output.denoised_color != VK_NULL_HANDLE);

    // ── Readback denoised output (compute stage) ──
    auto denoised_rb = test::ReadbackImage(
        ctx, output.denoised_image, 8,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    auto* denoised_raw = static_cast<uint16_t*>(denoised_rb.Map());
    REQUIRE(denoised_raw != nullptr);

    // ── Readback raw diffuse and specular (ray tracing stage already done) ──
    // The images are still in GENERAL layout after the render+denoise pass.
    // ReadbackImage transitions from GENERAL → TRANSFER_SRC.
    // Use COMPUTE_SHADER stage since the denoiser read them last.
    auto diffuse_rb = test::ReadbackImage(
        ctx, gbuffer_images.NoisyDiffuseImage(), 8,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    auto specular_rb = test::ReadbackImage(
        ctx, gbuffer_images.NoisySpecularImage(), 8,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    auto* diffuse_raw = static_cast<uint16_t*>(diffuse_rb.Map());
    auto* specular_raw = static_cast<uint16_t*>(specular_rb.Map());
    REQUIRE(diffuse_raw != nullptr);
    REQUIRE(specular_raw != nullptr);

    // ── CPU reference: diffuse + specular in RGBA16F ──
    // Compute the sum in half-float precision to match the GPU passthrough
    // shader, which operates on RGBA16F storage images.
    constexpr uint32_t kChannels = 4;
    std::vector<uint16_t> cpu_sum(test::kPixelCount * kChannels);
    for (uint32_t i = 0; i < test::kPixelCount * kChannels; ++i) {
        float d = test::HalfToFloat(diffuse_raw[i]);
        float s = test::HalfToFloat(specular_raw[i]);
        cpu_sum[i] = test::FloatToHalf(d + s);
    }

    // ── Primary check: pixel-level half-float match ──
    // GPU float→half rounding when writing to rgba16f storage may differ from
    // GLM's packHalf1x16 by ±1 ULP. Allow 1-ULP tolerance.
    uint32_t mismatch_count = 0;
    for (uint32_t i = 0; i < test::kPixelCount * kChannels; ++i) {
        int diff = static_cast<int>(denoised_raw[i]) - static_cast<int>(cpu_sum[i]);
        if (diff < -1 || diff > 1) {
            if (mismatch_count < 5) {
                uint32_t pixel = i / kChannels;
                uint32_t chan = i % kChannels;
                float gpu_val = test::HalfToFloat(denoised_raw[i]);
                float cpu_val = test::HalfToFloat(cpu_sum[i]);
                std::printf("  Mismatch pixel=%u ch=%u: gpu=%.6f (0x%04x) "
                            "cpu=%.6f (0x%04x) diff=%d ULP\n",
                            pixel, chan, gpu_val, denoised_raw[i],
                            cpu_val, cpu_sum[i], diff);
            }
            ++mismatch_count;
        }
    }

    if (mismatch_count > 0) {
        std::printf("Phase 9B: %u / %u half-float mismatches\n",
                    mismatch_count, test::kPixelCount * kChannels);
    }
    REQUIRE(mismatch_count == 0);

    // ── Secondary check: FLIP comparison ──
    // Convert both to tone-mapped RGB for FLIP. The passthrough denoiser
    // output should be identical to diffuse + specular.
    // Both buffers are already combined (no separate specular component),
    // so use a zero specular buffer for the TonemappedRGB helper.
    std::vector<uint16_t> zero_specular(test::kPixelCount * kChannels, 0);

    auto denoised_rgb = test::TonemappedRGB(
        denoised_raw, zero_specular.data(), test::kPixelCount);

    auto cpu_rgb = test::TonemappedRGB(
        cpu_sum.data(), zero_specular.data(), test::kPixelCount);

    float mean_flip = test::ComputeMeanFlip(
        cpu_rgb, denoised_rgb,
        static_cast<int>(test::kTestWidth),
        static_cast<int>(test::kTestHeight));

    std::printf("Phase 9B: Denoiser integration FLIP = %.6f\n", mean_flip);

    // With exact half-float match (above), FLIP should be 0.0.
    // Use a tiny epsilon to guard against FP rounding in the FLIP pipeline.
    REQUIRE(mean_flip < 0.001f);

    // ── Write diagnostic PNGs ──
    test::WriteCombinedPNG("tests/output/phase9b_raw_combined.png",
                           diffuse_raw, specular_raw,
                           test::kTestWidth, test::kTestHeight);
    test::WritePNG("tests/output/phase9b_denoised.png",
                   denoised_raw, test::kTestWidth, test::kTestHeight);

    // ── Cleanup ──
    denoised_rb.Unmap();
    diffuse_rb.Unmap();
    specular_rb.Unmap();

    denoiser.reset();
    renderer.reset();

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);

    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: No validation errors through full pipeline
//
// Runs the same render → barrier → denoise pipeline and relies on Vulkan
// validation layers (enabled by default in debug builds) to catch any
// synchronization, layout, or descriptor errors. The test passes if no
// assertions or validation failures occur.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 9B: Full pipeline runs without validation errors",
          "[phase9b][denoiser][renderer][vulkan][integration][validation]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = test::BuildCornellBox();

    RendererDesc renderer_desc{};
    renderer_desc.device = ctx.Device();
    renderer_desc.physical_device = ctx.PhysicalDevice();
    renderer_desc.queue = ctx.GraphicsQueue();
    renderer_desc.queue_family_index = ctx.QueueFamilyIndex();
    renderer_desc.allocator = ctx.Allocator();
    renderer_desc.width = test::kTestWidth;
    renderer_desc.height = test::kTestHeight;
    renderer_desc.samples_per_pixel = 4;
    renderer_desc.shader_dir = MONTI_SHADER_SPV_DIR;
    test::FillRendererProcAddrs(renderer_desc, ctx);

    auto renderer = Renderer::Create(renderer_desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    auto procs = test::MakeGpuBufferProcs();
    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(
        *renderer, ctx.Allocator(), ctx.Device(), upload_cmd, mesh_data, procs);
    REQUIRE_FALSE(gpu_buffers.empty());
    ctx.SubmitAndWait(upload_cmd);

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                                  test::kTestWidth, test::kTestHeight, gbuf_cmd,
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    deni::vulkan::DenoiserDesc denoiser_desc{};
    denoiser_desc.device = ctx.Device();
    denoiser_desc.physical_device = ctx.PhysicalDevice();
    denoiser_desc.width = test::kTestWidth;
    denoiser_desc.height = test::kTestHeight;
    denoiser_desc.allocator = ctx.Allocator();
    denoiser_desc.shader_dir = DENI_SHADER_SPV_DIR;
    test::FillDenoiserProcAddrs(denoiser_desc, ctx);

    auto denoiser = deni::vulkan::Denoiser::Create(denoiser_desc);
    REQUIRE(denoiser != nullptr);

    // Run multiple frames to exercise temporal state (reset on first frame)
    for (uint32_t frame = 0; frame < 3; ++frame) {
        VkCommandBuffer cmd = ctx.BeginOneShot();

        REQUIRE(renderer->RenderFrame(cmd, gbuffer, frame));

        VkMemoryBarrier2 rt_to_compute{};
        rt_to_compute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        rt_to_compute.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        rt_to_compute.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        rt_to_compute.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        rt_to_compute.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        VkDependencyInfo barrier_dep{};
        barrier_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        barrier_dep.memoryBarrierCount = 1;
        barrier_dep.pMemoryBarriers = &rt_to_compute;
        vkCmdPipelineBarrier2(cmd, &barrier_dep);

        deni::vulkan::DenoiserInput input{};
        input.noisy_diffuse = gbuffer.noisy_diffuse;
        input.noisy_specular = gbuffer.noisy_specular;
        input.motion_vectors = gbuffer.motion_vectors;
        input.linear_depth = gbuffer.linear_depth;
        input.world_normals = gbuffer.world_normals;
        input.diffuse_albedo = gbuffer.diffuse_albedo;
        input.specular_albedo = gbuffer.specular_albedo;
        input.render_width = test::kTestWidth;
        input.render_height = test::kTestHeight;
        input.reset_accumulation = (frame == 0);

        auto output = denoiser->Denoise(cmd, input);
        ctx.SubmitAndWait(cmd);

        REQUIRE(output.denoised_image != VK_NULL_HANDLE);
        REQUIRE(output.denoised_color != VK_NULL_HANDLE);
    }

    denoiser.reset();
    renderer.reset();

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);

    ctx.WaitIdle();
}
