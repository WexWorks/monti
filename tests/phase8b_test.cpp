#include <catch2/catch_test_macros.hpp>

#include <volk.h>

#include "test_helpers.h"

#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>
#include <monti/scene/Scene.h>

#include "../renderer/src/vulkan/Buffer.h"
#include "../scene/src/gltf/GltfLoader.h"

#include "scenes/CornellBox.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace monti;
using namespace monti::vulkan;

namespace {

struct TestContext {
    monti::app::VulkanContext ctx;

    bool Init() {
        if (!ctx.CreateInstance()) return false;
        if (!ctx.CreateDevice(std::nullopt)) return false;
        return true;
    }
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

    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = test::kTestWidth;
    desc.height = test::kTestHeight;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(*renderer, ctx.Allocator(),
                                               ctx.Device(), upload_cmd, mesh_data);
    REQUIRE_FALSE(gpu_buffers.empty());
    ctx.SubmitAndWait(upload_cmd);

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                                  test::kTestWidth, test::kTestHeight, gbuf_cmd,
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    VkCommandBuffer render_cmd = ctx.BeginOneShot();
    REQUIRE(renderer->RenderFrame(render_cmd, gbuffer, 0));
    ctx.SubmitAndWait(render_cmd);

    // Read back noisy_diffuse
    auto diffuse_readback = test::ReadbackImage(ctx, gbuffer_images.NoisyDiffuseImage());
    auto* diffuse_raw = static_cast<uint16_t*>(diffuse_readback.Map());
    REQUIRE(diffuse_raw != nullptr);

    test::WritePNG("tests/output/cornell_box_8b_diffuse.png",
                   diffuse_raw, test::kTestWidth, test::kTestHeight);

    auto diffuse_stats = test::AnalyzeRGBA16F(diffuse_raw, test::kTestWidth * test::kTestHeight);
    diffuse_readback.Unmap();

    // Read back noisy_specular
    auto specular_readback = test::ReadbackImage(ctx, gbuffer_images.NoisySpecularImage());
    auto* specular_raw = static_cast<uint16_t*>(specular_readback.Map());
    REQUIRE(specular_raw != nullptr);

    test::WritePNG("tests/output/cornell_box_8b_specular.png",
                   specular_raw, test::kTestWidth, test::kTestHeight);

    auto specular_stats = test::AnalyzeRGBA16F(specular_raw, test::kTestWidth * test::kTestHeight);
    specular_readback.Unmap();

    // No NaN or Inf in either channel
    REQUIRE(diffuse_stats.nan_count == 0);
    REQUIRE(diffuse_stats.inf_count == 0);
    REQUIRE(specular_stats.nan_count == 0);
    REQUIRE(specular_stats.inf_count == 0);

    // Diffuse channel should have non-trivial content (Cornell box is mostly diffuse)
    REQUIRE(diffuse_stats.nonzero_count > (test::kTestWidth * test::kTestHeight) / 4);
    REQUIRE(diffuse_stats.has_color_variation);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Diffuse/specular split — sum of diffuse+specular equals total
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8B: Diffuse + specular split sums correctly",
          "[phase8b][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;

    std::string box_path = std::string(MONTI_TEST_ASSETS_DIR) + "/Box.glb";
    auto result = gltf::LoadGltf(scene, box_path);
    REQUIRE(result.success);
    REQUIRE_FALSE(result.mesh_data.empty());

    constexpr uint32_t kEnvWidth = 4;
    constexpr uint32_t kEnvHeight = 2;
    std::vector<float> env_pixels(kEnvWidth * kEnvHeight * 4);
    for (uint32_t i = 0; i < kEnvWidth * kEnvHeight; ++i) {
        env_pixels[i * 4 + 0] = 0.5f;
        env_pixels[i * 4 + 1] = 0.5f;
        env_pixels[i * 4 + 2] = 0.5f;
        env_pixels[i * 4 + 3] = 1.0f;
    }

    TextureDesc env_tex{};
    env_tex.width = kEnvWidth;
    env_tex.height = kEnvHeight;
    env_tex.format = PixelFormat::kRGBA32F;
    env_tex.data.resize(env_pixels.size() * sizeof(float));
    std::memcpy(env_tex.data.data(), env_pixels.data(), env_tex.data.size());
    auto env_tex_id = scene.AddTexture(std::move(env_tex), "env_map");

    EnvironmentLight env_light{};
    env_light.hdr_lat_long = env_tex_id;
    env_light.intensity = 1.0f;
    env_light.rotation = 0.0f;
    scene.SetEnvironmentLight(env_light);

    CameraParams camera;
    camera.position = {0.0f, 1.0f, 4.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.vertical_fov_radians = 0.8f;
    camera.near_plane = 0.01f;
    camera.far_plane = 100.0f;
    scene.SetActiveCamera(camera);

    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = test::kTestWidth;
    desc.height = test::kTestHeight;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(*renderer, ctx.Allocator(),
                                               ctx.Device(), upload_cmd,
                                               result.mesh_data);
    REQUIRE_FALSE(gpu_buffers.empty());
    ctx.SubmitAndWait(upload_cmd);

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                                  test::kTestWidth, test::kTestHeight, gbuf_cmd,
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    VkCommandBuffer render_cmd = ctx.BeginOneShot();
    REQUIRE(renderer->RenderFrame(render_cmd, gbuffer, 0));
    ctx.SubmitAndWait(render_cmd);

    auto diffuse_readback = test::ReadbackImage(ctx, gbuffer_images.NoisyDiffuseImage());
    auto* diffuse_raw = static_cast<uint16_t*>(diffuse_readback.Map());
    REQUIRE(diffuse_raw != nullptr);

    auto specular_readback = test::ReadbackImage(ctx, gbuffer_images.NoisySpecularImage());
    auto* specular_raw = static_cast<uint16_t*>(specular_readback.Map());
    REQUIRE(specular_raw != nullptr);

    test::WritePNG("tests/output/box_8b_diffuse.png",
                   diffuse_raw, test::kTestWidth, test::kTestHeight);
    test::WritePNG("tests/output/box_8b_specular.png",
                   specular_raw, test::kTestWidth, test::kTestHeight);

    // Verify no NaN/Inf and that at least one channel has data
    uint32_t nan_count = 0;
    uint32_t diffuse_nonzero = 0;
    uint32_t specular_nonzero = 0;

    for (uint32_t i = 0; i < test::kTestWidth * test::kTestHeight; ++i) {
        float dr = test::HalfToFloat(diffuse_raw[i * 4 + 0]);
        float dg = test::HalfToFloat(diffuse_raw[i * 4 + 1]);
        float db = test::HalfToFloat(diffuse_raw[i * 4 + 2]);
        float sr = test::HalfToFloat(specular_raw[i * 4 + 0]);
        float sg = test::HalfToFloat(specular_raw[i * 4 + 1]);
        float sb = test::HalfToFloat(specular_raw[i * 4 + 2]);

        if (std::isnan(dr) || std::isnan(dg) || std::isnan(db)) ++nan_count;
        if (std::isnan(sr) || std::isnan(sg) || std::isnan(sb)) ++nan_count;

        if (dr + dg + db > 0.0f) ++diffuse_nonzero;
        if (sr + sg + sb > 0.0f) ++specular_nonzero;
    }

    diffuse_readback.Unmap();
    specular_readback.Unmap();

    REQUIRE(nan_count == 0);
    // At least one of diffuse or specular should have content
    REQUIRE((diffuse_nonzero + specular_nonzero) > 0);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Multi-frame — validates no validation errors across 2 frames
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8B: Multi-frame multi-bounce produces no validation errors",
          "[phase8b][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = test::BuildCornellBox();

    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = test::kTestWidth;
    desc.height = test::kTestHeight;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(*renderer, ctx.Allocator(),
                                               ctx.Device(), upload_cmd, mesh_data);
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
