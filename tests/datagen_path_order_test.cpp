// Integration test for Session 3: Sequential Path Rendering.
//
// Uses the production Renderer directly (no code duplication) to verify:
//   1. ResetTemporalState() causes zero motion vectors on the next frame
//   2. Consecutive renders WITHOUT a reset produce non-zero motion vectors
//   3. ResetTemporalState() after rendering clears stale temporal state
//
// This exercises the exact same has_prev_view_proj_ / prev_view_proj_ code
// path that GenerationSession::Run() relies on for path boundary resets.

#include <catch2/catch_test_macros.hpp>

#include <volk.h>

#include "test_helpers.h"
#include "scenes/CornellBox.h"

#include <monti/scene/Scene.h>
#include <monti/vulkan/Renderer.h>

#include "../app/core/CameraSetup.h"

#include <cmath>
#include <cstdint>
#include <vector>

using namespace monti;
using namespace monti::vulkan;

namespace {

constexpr uint32_t kWidth = 128;
constexpr uint32_t kHeight = 128;
constexpr uint32_t kPixelCount = kWidth * kHeight;

// Render a single frame at the given camera position and read back motion vectors.
// Returns RG16F data (2 half-floats per pixel).
std::vector<uint16_t> RenderAndReadbackMV(
    monti::app::VulkanContext& ctx,
    Renderer& renderer,
    monti::app::GBufferImages& gbuffer_images,
    GBuffer& gbuffer,
    Scene& scene,
    glm::vec3 position,
    glm::vec3 target) {

    CameraParams camera{};
    camera.position = position;
    camera.target = target;
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.vertical_fov_radians = glm::radians(60.0f);
    camera.near_plane = app::kDefaultNearPlane;
    camera.far_plane = app::kDefaultFarPlane;
    scene.SetActiveCamera(camera);

    VkCommandBuffer cmd = ctx.BeginOneShot();
    renderer.RenderFrame(cmd, gbuffer, 0);
    ctx.SubmitAndWait(cmd);

    // Read back motion vectors (RG16F = 4 bytes per pixel)
    auto readback = test::ReadbackImage(ctx, gbuffer_images.MotionVectorsImage(),
                                        4, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                        kWidth, kHeight);

    auto* mapped = static_cast<uint16_t*>(readback.Map());
    std::vector<uint16_t> mv(kPixelCount * 2);
    std::memcpy(mv.data(), mapped, kPixelCount * 2 * sizeof(uint16_t));
    readback.Unmap();
    readback.Destroy();

    return mv;
}

// Compute max motion vector magnitude from RG16F data.
float MaxMVMagnitude(const std::vector<uint16_t>& mv) {
    float max_mag = 0.0f;
    for (uint32_t i = 0; i < mv.size() / 2; ++i) {
        float x = capture::HalfToFloat(mv[i * 2 + 0]);
        float y = capture::HalfToFloat(mv[i * 2 + 1]);
        if (std::isnan(x) || std::isnan(y)) continue;
        float mag = std::sqrt(x * x + y * y);
        if (mag > max_mag) max_mag = mag;
    }
    return max_mag;
}

// Count pixels with motion vector magnitude above threshold.
uint32_t CountNonZeroMV(const std::vector<uint16_t>& mv, float threshold = 1e-5f) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < mv.size() / 2; ++i) {
        float x = capture::HalfToFloat(mv[i * 2 + 0]);
        float y = capture::HalfToFloat(mv[i * 2 + 1]);
        if (std::isnan(x) || std::isnan(y)) continue;
        if (std::sqrt(x * x + y * y) > threshold) ++count;
    }
    return count;
}

}  // namespace

TEST_CASE("ResetTemporalState produces zero MVs on first render",
          "[datagen][temporal][vulkan][integration]") {
    auto& ctx = test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    // Build scene
    auto [scene, mesh_data] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene);

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

    auto procs = test::MakeGpuBufferProcs();
    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(
        *renderer, ctx.Allocator(), ctx.Device(), upload_cmd, mesh_data, procs);
    ctx.SubmitAndWait(upload_cmd);

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                          kWidth, kHeight, gbuf_cmd,
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    ctx.SubmitAndWait(gbuf_cmd);
    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    // Camera positions — small lateral offset to produce detectable MVs
    glm::vec3 pos_A = {0.5f, 0.5f, 1.5f};
    glm::vec3 pos_B = {0.6f, 0.5f, 1.5f};  // 0.1 units to the right
    glm::vec3 look  = {0.5f, 0.5f, 0.0f};

    // ── Frame 1: First render after construction → zero MVs ──
    // (Renderer starts with has_prev_view_proj_=false)
    auto mv0 = RenderAndReadbackMV(ctx, *renderer, gbuffer_images, gbuffer,
                                   scene, pos_A, look);
    REQUIRE(MaxMVMagnitude(mv0) < 1e-5f);

    // ── Frame 2: Same camera → zero MVs (no motion) ──
    auto mv1 = RenderAndReadbackMV(ctx, *renderer, gbuffer_images, gbuffer,
                                   scene, pos_A, look);
    REQUIRE(MaxMVMagnitude(mv1) < 1e-5f);

    // ── Frame 3: Different camera → non-zero MVs (camera moved) ──
    auto mv2 = RenderAndReadbackMV(ctx, *renderer, gbuffer_images, gbuffer,
                                   scene, pos_B, look);
    REQUIRE(MaxMVMagnitude(mv2) > 0.001f);
    REQUIRE(CountNonZeroMV(mv2) > kPixelCount / 10);  // >10% of pixels should show motion

    // ── Frame 4: ResetTemporalState then render → zero MVs ──
    // This is the critical path boundary test: simulates what GenerationSession
    // does when switching to a new path_id.
    renderer->ResetTemporalState();
    auto mv3 = RenderAndReadbackMV(ctx, *renderer, gbuffer_images, gbuffer,
                                   scene, pos_A, look);
    REQUIRE(MaxMVMagnitude(mv3) < 1e-5f);

    // ── Frame 5: Move again without reset → non-zero MVs return ──
    auto mv4 = RenderAndReadbackMV(ctx, *renderer, gbuffer_images, gbuffer,
                                   scene, pos_B, look);
    REQUIRE(MaxMVMagnitude(mv4) > 0.001f);
    REQUIRE(CountNonZeroMV(mv4) > kPixelCount / 10);

    // ── Frame 6: Reset again (simulates second path boundary) → zero MVs ──
    renderer->ResetTemporalState();
    auto mv5 = RenderAndReadbackMV(ctx, *renderer, gbuffer_images, gbuffer,
                                   scene, pos_B, look);
    REQUIRE(MaxMVMagnitude(mv5) < 1e-5f);

    // Cleanup
    ctx.WaitIdle();
    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
}
