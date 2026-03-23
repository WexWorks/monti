#include <catch2/catch_test_macros.hpp>

#include <volk.h>

#include "test_helpers.h"

#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>
#include <monti/scene/Scene.h>

#include "../renderer/src/vulkan/Buffer.h"

#include "scenes/CornellBox.h"

#include <cmath>
#include <cstring>
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
// Test 1: Cornell box multi-bounce — color bleeding requires 2+ bounces
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8B: Cornell box multi-bounce renders with no NaN/Inf",
          "[phase8b][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene);

    // 64 frames × 16 SPP = 1024 total samples for clean visual output
    auto result = test::RenderSceneMultiFrame(ctx, scene, mesh_data, 64, 16);

    auto* diffuse_raw = result.diffuse.data();
    auto* specular_raw = result.specular.data();

    test::WritePNG("tests/output/phase8b_cornell_box_diffuse.png",
                   diffuse_raw, test::kTestWidth, test::kTestHeight);
    test::WritePNG("tests/output/phase8b_cornell_box_specular.png",
                   specular_raw, test::kTestWidth, test::kTestHeight);
    test::WriteCombinedPNG("tests/output/phase8b_cornell_box_combined.png",
                           diffuse_raw, specular_raw,
                           test::kTestWidth, test::kTestHeight);

    auto diffuse_stats = test::AnalyzeRGBA16F(diffuse_raw, test::kPixelCount);
    auto specular_stats = test::AnalyzeRGBA16F(specular_raw, test::kPixelCount);

    // No NaN or Inf in either channel
    REQUIRE(diffuse_stats.nan_count == 0);
    REQUIRE(diffuse_stats.inf_count == 0);
    REQUIRE(specular_stats.nan_count == 0);
    REQUIRE(specular_stats.inf_count == 0);

    // Diffuse channel should have non-trivial content (Cornell box is mostly diffuse)
    REQUIRE(diffuse_stats.nonzero_count > test::kPixelCount / 4);
    REQUIRE(diffuse_stats.has_color_variation);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Multi-frame — validates no validation errors across 2 frames
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8B: Multi-frame multi-bounce produces no validation errors",
          "[phase8b][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene);

    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = test::kTestWidth;
    desc.height = test::kTestHeight;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;
    test::FillRendererProcAddrs(desc, ctx);

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(*renderer, ctx.Allocator(),
                                               ctx.Device(), upload_cmd, mesh_data,
                                               test::MakeGpuBufferProcs());
    ctx.SubmitAndWait(upload_cmd);

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                                  test::kTestWidth, test::kTestHeight, gbuf_cmd));
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    VkCommandBuffer cmd1 = ctx.BeginOneShot();
    REQUIRE(renderer->RenderFrame(cmd1, gbuffer, 0));
    ctx.SubmitAndWait(cmd1);

    VkCommandBuffer cmd2 = ctx.BeginOneShot();
    REQUIRE(renderer->RenderFrame(cmd2, gbuffer, 1));
    ctx.SubmitAndWait(cmd2);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}
