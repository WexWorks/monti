#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <volk.h>

#include "test_helpers.h"
#include "scenes/CornellBox.h"

#include <monti/scene/Scene.h>

#include "../renderer/src/vulkan/GpuScene.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace monti;
using namespace monti::vulkan;
using Catch::Matchers::WithinAbs;

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
// Test 1: FLIP convergence under extreme emission — firefly clamp stability
//
// Renders a Cornell box with extreme emission (10000) at 1x4 SPP (low) and
// 16x64 SPP multi-frame accumulated (high).  The firefly clamp bounds
// per-path luminance, so increasing SPP refines noise but doesn't change
// the clamped brightness.  A working clamp produces structurally similar
// images at different SPP (low FLIP score).  A broken clamp (NaN, unclamped
// spikes) causes wild divergence (high FLIP score).
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8E: FLIP convergence under extreme emission",
          "[phase8e][renderer][vulkan][integration][flip][firefly]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto build_extreme_cornell = []() {
        auto [scene, mesh_data] = test::BuildCornellBox();

        constexpr float kExtremeEmission = 10000.0f;
        auto* light_mat = scene.GetMaterial(MaterialId{3});
        REQUIRE(light_mat != nullptr);
        light_mat->emissive_factor = {kExtremeEmission, kExtremeEmission, kExtremeEmission};
        light_mat->emissive_strength = 1.0f;

        AreaLight bright_light;
        bright_light.corner = {0.35f, 0.999f, 0.35f};
        bright_light.edge_a = {0.3f, 0.0f, 0.0f};
        bright_light.edge_b = {0.0f, 0.0f, 0.3f};
        bright_light.radiance = {kExtremeEmission, kExtremeEmission, kExtremeEmission};
        bright_light.two_sided = false;
        scene.AddAreaLight(bright_light);

        return std::make_pair(std::move(scene), std::move(mesh_data));
    };

    // ── Low quality: single frame, 4 SPP ──
    auto [scene_low, mesh_data_low] = build_extreme_cornell();
    auto result_low = test::RenderSceneMultiFrame(
        ctx, scene_low, mesh_data_low, 1, 4);

    constexpr uint32_t kPixelCount = test::kTestWidth * test::kTestHeight;
    auto low_rgb = test::TonemappedRGB(
        result_low.diffuse.data(), result_low.specular.data(), kPixelCount);

    // ── High quality: 16 frames x 64 SPP = 1024 total samples ──
    auto [scene_high, mesh_data_high] = build_extreme_cornell();
    auto result_high = test::RenderSceneMultiFrame(
        ctx, scene_high, mesh_data_high, 16, 64);

    auto high_rgb = test::TonemappedRGB(
        result_high.diffuse.data(), result_high.specular.data(), kPixelCount);

    float mean_flip = test::ComputeMeanFlip(high_rgb, low_rgb,
                                      static_cast<int>(test::kTestWidth),
                                      static_cast<int>(test::kTestHeight));

    std::printf("Phase 8E FLIP convergence (extreme emission): mean=%.4f\n",
                mean_flip);

    // With firefly clamping active, 4 vs 1024 SPP should differ mainly by
    // MC noise in the bounded range.  Broken clamping would push this
    // well above 0.5.
    REQUIRE(mean_flip < 0.5f);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_low);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_high);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Firefly clamp preserves hue — GPU integration test
//
// Renders a Cornell box with strongly-colored extreme emission (warm orange:
// R=10000, G=5000, B=1000).  Reads back pixels that hit the emissive surface
// and verifies the clamped output preserves the original R > G > B ordering
// (hue).  A per-component min clamp would flatten all channels to the same
// threshold, destroying the hue.  A luminance-proportional clamp preserves
// the ratios.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8E: Firefly clamp preserves hue under extreme emission",
          "[phase8e][renderer][vulkan][integration][firefly]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = test::BuildCornellBox();

    // Strongly colored extreme emission — warm orange (R >> G >> B)
    auto* light_mat = scene.GetMaterial(MaterialId{3});
    REQUIRE(light_mat != nullptr);
    light_mat->emissive_factor = {10000.0f, 5000.0f, 1000.0f};
    light_mat->emissive_strength = 1.0f;

    AreaLight colored_light;
    colored_light.corner = {0.35f, 0.999f, 0.35f};
    colored_light.edge_a = {0.3f, 0.0f, 0.0f};
    colored_light.edge_b = {0.0f, 0.0f, 0.3f};
    colored_light.radiance = {10000.0f, 5000.0f, 1000.0f};
    colored_light.two_sided = false;
    scene.AddAreaLight(colored_light);

    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = test::kTestWidth;
    desc.height = test::kTestHeight;
    desc.samples_per_pixel = 16;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(*renderer, ctx.Allocator(),
                                               ctx.Device(), upload_cmd,
                                               mesh_data);
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

    auto diffuse_rb = test::ReadbackImage(ctx, gbuffer_images.NoisyDiffuseImage());
    auto* diffuse_raw = static_cast<uint16_t*>(diffuse_rb.Map());
    REQUIRE(diffuse_raw != nullptr);

    // Find bright pixels (clamped emissive hits) and verify hue preservation.
    // The original emission has R:G:B ratio of 10:5:1, so after proportional
    // clamping we expect R > G > B with similar ratios.
    constexpr float kFireflyClampDiffuse = 20.0f;
    uint32_t hue_preserved_count = 0;
    uint32_t bright_pixel_count = 0;

    for (uint32_t i = 0; i < test::kTestWidth * test::kTestHeight; ++i) {
        float r = test::HalfToFloat(diffuse_raw[i * 4 + 0]);
        float g = test::HalfToFloat(diffuse_raw[i * 4 + 1]);
        float b = test::HalfToFloat(diffuse_raw[i * 4 + 2]);

        if (std::isnan(r) || std::isnan(g) || std::isnan(b)) continue;
        if (std::isinf(r) || std::isinf(g) || std::isinf(b)) continue;

        float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        // Check pixels near the clamp threshold (bright, clamped pixels)
        if (lum > kFireflyClampDiffuse * 0.5f) {
            ++bright_pixel_count;
            // Proportional clamp preserves R > G > B ordering
            if (r > g && g > b) ++hue_preserved_count;
        }
    }

    diffuse_rb.Unmap();

    // Should have bright pixels (camera sees the light directly)
    REQUIRE(bright_pixel_count > 50);

    // Majority of bright clamped pixels should preserve R > G > B ordering.
    // Allow some tolerance for stochastic noise at pixel boundaries.
    float hue_ratio = static_cast<float>(hue_preserved_count)
                    / static_cast<float>(bright_pixel_count);
    INFO("Hue preserved: " << hue_preserved_count << " / "
         << bright_pixel_count << " (" << hue_ratio * 100.0f << "%)");
    REQUIRE(hue_ratio > 0.7f);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Firefly clamp passthrough — dim scene pixels are unmodified
//
// Renders a Cornell box with a dim light (emission below the firefly
// threshold).  Verifies that dim pixels are not clipped or distorted by
// the clamp.  At 64 SPP, the brightest pixel should remain well below
// the clamp threshold, confirming the clamp is a no-op for low-energy paths.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8E: Firefly clamp passthrough for dim scene",
          "[phase8e][renderer][vulkan][integration][firefly]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = test::BuildCornellBox();

    // Reduce emission well below the firefly clamp threshold (20.0)
    auto* light_mat = scene.GetMaterial(MaterialId{3});
    REQUIRE(light_mat != nullptr);
    light_mat->emissive_factor = {3.0f, 2.0f, 1.0f};
    light_mat->emissive_strength = 1.0f;

    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = test::kTestWidth;
    desc.height = test::kTestHeight;
    desc.samples_per_pixel = 64;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(*renderer, ctx.Allocator(),
                                               ctx.Device(), upload_cmd,
                                               mesh_data);
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

    auto diffuse_rb = test::ReadbackImage(ctx, gbuffer_images.NoisyDiffuseImage());
    auto* diffuse_raw = static_cast<uint16_t*>(diffuse_rb.Map());
    REQUIRE(diffuse_raw != nullptr);

    constexpr float kFireflyClampDiffuse = 20.0f;
    float max_lum = 0.0f;
    uint32_t nonzero_count = 0;
    uint32_t nan_count = 0;

    for (uint32_t i = 0; i < test::kTestWidth * test::kTestHeight; ++i) {
        float r = test::HalfToFloat(diffuse_raw[i * 4 + 0]);
        float g = test::HalfToFloat(diffuse_raw[i * 4 + 1]);
        float b = test::HalfToFloat(diffuse_raw[i * 4 + 2]);

        if (std::isnan(r) || std::isnan(g) || std::isnan(b)) { ++nan_count; continue; }
        if (std::isinf(r) || std::isinf(g) || std::isinf(b)) continue;

        float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        max_lum = std::max(max_lum, lum);
        if (r + g + b > 0.0f) ++nonzero_count;
    }

    diffuse_rb.Unmap();

    REQUIRE(nan_count == 0);
    // Scene should have non-trivial content
    REQUIRE(nonzero_count > 100);
    // All pixels should be well below the clamp threshold,
    // confirming the clamp is passthrough for low-energy paths
    REQUIRE(max_lum < kFireflyClampDiffuse);
    // But the scene should still be non-trivially lit
    REQUIRE(max_lum > 0.1f);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: Firefly clamp no NaN/Inf under zero and extreme inputs
//
// Renders two scenes with edge-case emission: one with zero emission
// (dark room) and one with extreme emission (10000).  Both must produce
// no NaN or Inf in the output, proving the shader clamp handles edge cases.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8E: Firefly clamp no NaN/Inf edge cases",
          "[phase8e][renderer][vulkan][integration][firefly]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // Helper lambda to render and check for NaN/Inf
    auto render_and_check = [&](float emission, std::string_view label) {
        auto [scene, mesh_data] = test::BuildCornellBox();

        auto* light_mat = scene.GetMaterial(MaterialId{3});
        REQUIRE(light_mat != nullptr);
        light_mat->emissive_factor = {emission, emission, emission};
        light_mat->emissive_strength = 1.0f;

        RendererDesc desc{};
        desc.device = ctx.Device();
        desc.physical_device = ctx.PhysicalDevice();
        desc.queue = ctx.GraphicsQueue();
        desc.queue_family_index = ctx.QueueFamilyIndex();
        desc.allocator = ctx.Allocator();
        desc.width = test::kTestWidth;
        desc.height = test::kTestHeight;
        desc.samples_per_pixel = 1;
        desc.shader_dir = MONTI_SHADER_SPV_DIR;

        auto renderer = Renderer::Create(desc);
        REQUIRE(renderer);
        renderer->SetScene(&scene);

        VkCommandBuffer upload_cmd = ctx.BeginOneShot();
        auto gpu_buffers = UploadAndRegisterMeshes(*renderer, ctx.Allocator(),
                                                   ctx.Device(), upload_cmd,
                                                   mesh_data);
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

        auto diffuse_rb = test::ReadbackImage(ctx, gbuffer_images.NoisyDiffuseImage());
        auto specular_rb = test::ReadbackImage(ctx, gbuffer_images.NoisySpecularImage());
        auto* d = static_cast<uint16_t*>(diffuse_rb.Map());
        auto* s = static_cast<uint16_t*>(specular_rb.Map());
        REQUIRE(d != nullptr);
        REQUIRE(s != nullptr);

        uint32_t nan_count = 0;
        uint32_t inf_count = 0;
        for (uint32_t i = 0; i < test::kTestWidth * test::kTestHeight; ++i) {
            for (int c = 0; c < 3; ++c) {
                float dv = test::HalfToFloat(d[i * 4 + c]);
                float sv = test::HalfToFloat(s[i * 4 + c]);
                if (std::isnan(dv) || std::isnan(sv)) ++nan_count;
                if (std::isinf(dv) || std::isinf(sv)) ++inf_count;
            }
        }

        diffuse_rb.Unmap();
        specular_rb.Unmap();

        INFO("Emission=" << emission << " (" << label << ")");
        REQUIRE(nan_count == 0);
        REQUIRE(inf_count == 0);

        for (auto& buf : gpu_buffers)
            DestroyGpuBuffer(ctx.Allocator(), buf);
        ctx.WaitIdle();
    };

    render_and_check(0.0f, "zero emission");
    render_and_check(10000.0f, "extreme emission");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: LinearDepth format is RG16F
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8E: LinearDepth format is RG16F",
          "[phase8e][gbuffer]") {
    VkFormat fmt = monti::app::GBufferImages::FormatFor(
        monti::app::GBufferImages::Index::kLinearDepth);
    REQUIRE(fmt == VK_FORMAT_R16G16_SFLOAT);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: Hit distance output — integration test with Cornell box
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8E: Hit distance output in linear_depth.g",
          "[phase8e][renderer][vulkan][integration]") {
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
                                               ctx.Device(), upload_cmd,
                                               mesh_data);
    REQUIRE_FALSE(gpu_buffers.empty());
    ctx.SubmitAndWait(upload_cmd);

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                                  test::kTestWidth, test::kTestHeight, gbuf_cmd,
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    // Render one frame at 1 spp
    VkCommandBuffer render_cmd = ctx.BeginOneShot();
    REQUIRE(renderer->RenderFrame(render_cmd, gbuffer, 0));
    ctx.SubmitAndWait(render_cmd);

    // Readback linear_depth (RG16F = 4 bytes per pixel)
    auto depth_rb = test::ReadbackImage(ctx, gbuffer_images.LinearDepthImage(), 4);
    auto* raw = static_cast<uint16_t*>(depth_rb.Map());
    REQUIRE(raw != nullptr);

    // Write diagnostic images for manual verification
    {
        // Readback and write combined diffuse+specular render
        auto diffuse_rb = test::ReadbackImage(ctx, gbuffer_images.NoisyDiffuseImage());
        auto specular_rb = test::ReadbackImage(ctx, gbuffer_images.NoisySpecularImage());
        auto* diffuse_raw = static_cast<uint16_t*>(diffuse_rb.Map());
        auto* specular_raw = static_cast<uint16_t*>(specular_rb.Map());

        // Combined render (diffuse + specular, tone-mapped)
        std::vector<uint8_t> combined(test::kTestWidth * test::kTestHeight * 3);
        for (uint32_t i = 0; i < test::kTestWidth * test::kTestHeight; ++i) {
            float r = test::HalfToFloat(diffuse_raw[i * 4 + 0])
                    + test::HalfToFloat(specular_raw[i * 4 + 0]);
            float g = test::HalfToFloat(diffuse_raw[i * 4 + 1])
                    + test::HalfToFloat(specular_raw[i * 4 + 1]);
            float b = test::HalfToFloat(diffuse_raw[i * 4 + 2])
                    + test::HalfToFloat(specular_raw[i * 4 + 2]);
            r = std::pow(std::max(r / (1.0f + r), 0.0f), 1.0f / 2.2f);
            g = std::pow(std::max(g / (1.0f + g), 0.0f), 1.0f / 2.2f);
            b = std::pow(std::max(b / (1.0f + b), 0.0f), 1.0f / 2.2f);
            combined[i * 3 + 0] = static_cast<uint8_t>(std::clamp(r * 255.0f + 0.5f, 0.0f, 255.0f));
            combined[i * 3 + 1] = static_cast<uint8_t>(std::clamp(g * 255.0f + 0.5f, 0.0f, 255.0f));
            combined[i * 3 + 2] = static_cast<uint8_t>(std::clamp(b * 255.0f + 0.5f, 0.0f, 255.0f));
        }
        std::filesystem::create_directories("tests/output");
        stbi_write_png("tests/output/cornell_8e_combined.png",
                       test::kTestWidth, test::kTestHeight, 3, combined.data(), test::kTestWidth * 3);

        // Linear depth channel (normalized to visible range)
        // Hit distance channel (normalized to visible range)
        std::vector<uint8_t> depth_vis(test::kTestWidth * test::kTestHeight * 3);
        std::vector<uint8_t> hit_t_vis(test::kTestWidth * test::kTestHeight * 3);
        float max_depth = 0.0f;
        float max_hit_t = 0.0f;
        for (uint32_t i = 0; i < test::kTestWidth * test::kTestHeight; ++i) {
            float d = test::HalfToFloat(raw[i * 2 + 0]);
            float t = test::HalfToFloat(raw[i * 2 + 1]);
            if (!std::isnan(d) && !std::isinf(d) && d < 1000.0f)
                max_depth = std::max(max_depth, std::abs(d));
            if (!std::isnan(t) && !std::isinf(t) && t < 1000.0f)
                max_hit_t = std::max(max_hit_t, t);
        }
        for (uint32_t i = 0; i < test::kTestWidth * test::kTestHeight; ++i) {
            float d = test::HalfToFloat(raw[i * 2 + 0]);
            float t = test::HalfToFloat(raw[i * 2 + 1]);
            uint8_t dv = (d < 1000.0f && max_depth > 0.0f)
                ? static_cast<uint8_t>(std::clamp(std::abs(d) / max_depth * 255.0f, 0.0f, 255.0f))
                : 0;
            uint8_t tv = (t < 1000.0f && max_hit_t > 0.0f)
                ? static_cast<uint8_t>(std::clamp(t / max_hit_t * 255.0f, 0.0f, 255.0f))
                : 0;
            depth_vis[i * 3 + 0] = depth_vis[i * 3 + 1] = depth_vis[i * 3 + 2] = dv;
            hit_t_vis[i * 3 + 0] = hit_t_vis[i * 3 + 1] = hit_t_vis[i * 3 + 2] = tv;
        }
        stbi_write_png("tests/output/cornell_8e_linear_depth.png",
                       test::kTestWidth, test::kTestHeight, 3, depth_vis.data(), test::kTestWidth * 3);
        stbi_write_png("tests/output/cornell_8e_hit_distance.png",
                       test::kTestWidth, test::kTestHeight, 3, hit_t_vis.data(), test::kTestWidth * 3);

        diffuse_rb.Unmap();
        specular_rb.Unmap();
    }

    constexpr uint32_t kPixelCount = test::kTestWidth * test::kTestHeight;
    constexpr float kSentinelDepth = 1e4f;
    constexpr float kSceneDiagonal = 2.0f;  // Cornell box is unit-scale

    uint32_t hit_count = 0;
    uint32_t miss_count = 0;
    uint32_t nan_count = 0;
    uint32_t inf_count = 0;
    uint32_t negative_hit_t_count = 0;
    uint32_t excessive_hit_t_count = 0;

    for (uint32_t i = 0; i < kPixelCount; ++i) {
        float depth = test::HalfToFloat(raw[i * 2 + 0]);
        float hit_t = test::HalfToFloat(raw[i * 2 + 1]);

        if (std::isnan(depth) || std::isnan(hit_t)) { ++nan_count; continue; }
        if (std::isinf(depth) || std::isinf(hit_t)) { ++inf_count; continue; }

        bool is_miss = (depth >= kSentinelDepth * 0.9f);
        if (is_miss) {
            // Miss pixels: both channels should be sentinel
            REQUIRE(hit_t >= kSentinelDepth * 0.9f);
            ++miss_count;
        } else {
            // Hit pixels: depth is signed, hit_t is positive raw distance
            REQUIRE(hit_t > 0.0f);
            if (hit_t < 0.0f) ++negative_hit_t_count;
            // hit_t should be within reasonable scene bounds
            // (camera is ~2 units from the back wall, scene diagonal ~sqrt(3))
            if (hit_t > kSceneDiagonal * 3.0f) ++excessive_hit_t_count;
            ++hit_count;
        }
    }

    depth_rb.Unmap();

    // No NaN or Inf in either channel
    REQUIRE(nan_count == 0);
    REQUIRE(inf_count == 0);

    // Cornell box should have both hit and miss pixels
    REQUIRE(hit_count > 100);

    // No negative hit distances
    REQUIRE(negative_hit_t_count == 0);

    // No excessive hit distances (beyond 3x scene diagonal)
    REQUIRE(excessive_hit_t_count == 0);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 7: GPU firefly clamp validation — proves the shader clamp is active
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8E: GPU firefly clamp limits pixel luminance",
          "[phase8e][renderer][vulkan][integration][firefly]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = test::BuildCornellBox();

    // Crank the light material emission to far exceed the firefly threshold.
    // The light material is the 4th material added (index 3) in BuildCornellBox.
    // Direct rays hitting this emissive surface produce path_radiance = emission
    // with throughput = 1, so unclamped luminance would be ~10000.
    constexpr float kExtremeEmission = 10000.0f;
    auto* light_mat = scene.GetMaterial(MaterialId{3});
    REQUIRE(light_mat != nullptr);
    light_mat->emissive_factor = {kExtremeEmission, kExtremeEmission, kExtremeEmission};
    light_mat->emissive_strength = 1.0f;

    // Also update the area light radiance to match, so direct-light sampling
    // contributions are also extreme.
    AreaLight bright_light;
    bright_light.corner = {0.35f, 0.999f, 0.35f};
    bright_light.edge_a = {0.3f, 0.0f, 0.0f};
    bright_light.edge_b = {0.0f, 0.0f, 0.3f};
    bright_light.radiance = {kExtremeEmission, kExtremeEmission, kExtremeEmission};
    bright_light.two_sided = false;
    scene.AddAreaLight(bright_light);

    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = test::kTestWidth;
    desc.height = test::kTestHeight;
    desc.samples_per_pixel = 1;  // 1 spp so pixel value = single path value
    desc.shader_dir = MONTI_SHADER_SPV_DIR;

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(*renderer, ctx.Allocator(),
                                               ctx.Device(), upload_cmd,
                                               mesh_data);
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

    // Readback diffuse (RGBA16F = 8 bytes/pixel) and specular
    auto diffuse_rb = test::ReadbackImage(ctx, gbuffer_images.NoisyDiffuseImage());
    auto specular_rb = test::ReadbackImage(ctx, gbuffer_images.NoisySpecularImage());
    auto* diffuse_raw = static_cast<uint16_t*>(diffuse_rb.Map());
    auto* specular_raw = static_cast<uint16_t*>(specular_rb.Map());
    REQUIRE(diffuse_raw != nullptr);
    REQUIRE(specular_raw != nullptr);

    constexpr float kFireflyClampDiffuse = 20.0f;
    constexpr float kFireflyClampSpecular = 80.0f;
    // FP16 has limited precision; allow a small tolerance
    constexpr float kTolerance = 0.5f;

    float max_diffuse_lum = 0.0f;
    float max_specular_lum = 0.0f;
    uint32_t diffuse_violations = 0;
    uint32_t specular_violations = 0;

    for (uint32_t i = 0; i < test::kTestWidth * test::kTestHeight; ++i) {
        float dr = test::HalfToFloat(diffuse_raw[i * 4 + 0]);
        float dg = test::HalfToFloat(diffuse_raw[i * 4 + 1]);
        float db = test::HalfToFloat(diffuse_raw[i * 4 + 2]);
        float d_lum = 0.2126f * dr + 0.7152f * dg + 0.0722f * db;

        float sr = test::HalfToFloat(specular_raw[i * 4 + 0]);
        float sg = test::HalfToFloat(specular_raw[i * 4 + 1]);
        float sb = test::HalfToFloat(specular_raw[i * 4 + 2]);
        float s_lum = 0.2126f * sr + 0.7152f * sg + 0.0722f * sb;

        if (!std::isnan(d_lum) && !std::isinf(d_lum)) {
            max_diffuse_lum = std::max(max_diffuse_lum, d_lum);
            if (d_lum > kFireflyClampDiffuse + kTolerance) ++diffuse_violations;
        }
        if (!std::isnan(s_lum) && !std::isinf(s_lum)) {
            max_specular_lum = std::max(max_specular_lum, s_lum);
            if (s_lum > kFireflyClampSpecular + kTolerance) ++specular_violations;
        }
    }

    // Write diagnostic images showing clamped output
    {
        std::vector<uint8_t> diffuse_vis(test::kTestWidth * test::kTestHeight * 3);
        std::vector<uint8_t> specular_vis(test::kTestWidth * test::kTestHeight * 3);
        for (uint32_t i = 0; i < test::kTestWidth * test::kTestHeight; ++i) {
            // Tone-map diffuse (normalize to diffuse threshold for visibility)
            float dr = test::HalfToFloat(diffuse_raw[i * 4 + 0]);
            float dg = test::HalfToFloat(diffuse_raw[i * 4 + 1]);
            float db = test::HalfToFloat(diffuse_raw[i * 4 + 2]);
            dr = std::pow(std::clamp(dr / kFireflyClampDiffuse, 0.0f, 1.0f), 1.0f / 2.2f);
            dg = std::pow(std::clamp(dg / kFireflyClampDiffuse, 0.0f, 1.0f), 1.0f / 2.2f);
            db = std::pow(std::clamp(db / kFireflyClampDiffuse, 0.0f, 1.0f), 1.0f / 2.2f);
            diffuse_vis[i * 3 + 0] = static_cast<uint8_t>(dr * 255.0f + 0.5f);
            diffuse_vis[i * 3 + 1] = static_cast<uint8_t>(dg * 255.0f + 0.5f);
            diffuse_vis[i * 3 + 2] = static_cast<uint8_t>(db * 255.0f + 0.5f);
            // Tone-map specular
            float sr = test::HalfToFloat(specular_raw[i * 4 + 0]);
            float sg = test::HalfToFloat(specular_raw[i * 4 + 1]);
            float sb = test::HalfToFloat(specular_raw[i * 4 + 2]);
            sr = std::pow(std::clamp(sr / kFireflyClampSpecular, 0.0f, 1.0f), 1.0f / 2.2f);
            sg = std::pow(std::clamp(sg / kFireflyClampSpecular, 0.0f, 1.0f), 1.0f / 2.2f);
            sb = std::pow(std::clamp(sb / kFireflyClampSpecular, 0.0f, 1.0f), 1.0f / 2.2f);
            specular_vis[i * 3 + 0] = static_cast<uint8_t>(sr * 255.0f + 0.5f);
            specular_vis[i * 3 + 1] = static_cast<uint8_t>(sg * 255.0f + 0.5f);
            specular_vis[i * 3 + 2] = static_cast<uint8_t>(sb * 255.0f + 0.5f);
        }
        std::filesystem::create_directories("tests/output");
        stbi_write_png("tests/output/firefly_clamp_diffuse.png",
                       test::kTestWidth, test::kTestHeight, 3, diffuse_vis.data(), test::kTestWidth * 3);
        stbi_write_png("tests/output/firefly_clamp_specular.png",
                       test::kTestWidth, test::kTestHeight, 3, specular_vis.data(), test::kTestWidth * 3);
    }

    diffuse_rb.Unmap();
    specular_rb.Unmap();

    INFO("Max diffuse luminance: " << max_diffuse_lum
         << " (threshold: " << kFireflyClampDiffuse << ")");
    INFO("Max specular luminance: " << max_specular_lum
         << " (threshold: " << kFireflyClampSpecular << ")");

    // No pixels should exceed the firefly clamp thresholds
    REQUIRE(diffuse_violations == 0);
    REQUIRE(specular_violations == 0);
    REQUIRE(max_diffuse_lum <= kFireflyClampDiffuse + kTolerance);
    REQUIRE(max_specular_lum <= kFireflyClampSpecular + kTolerance);

    // With emission=10000 and the camera seeing the light directly,
    // many pixels should be clamped TO the threshold, proving the
    // clamp is active (not just that the scene is dim).
    REQUIRE(max_diffuse_lum >= kFireflyClampDiffuse * 0.5f);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}
