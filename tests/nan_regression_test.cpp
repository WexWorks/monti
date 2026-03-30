#include <catch2/catch_test_macros.hpp>

#include <volk.h>

#include "test_helpers.h"

#include <monti/scene/Scene.h>

#include "../app/core/CameraSetup.h"
#include "gltf/GltfLoader.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using namespace monti;
using namespace monti::vulkan;

// ═══════════════════════════════════════════════════════════════════════════
//
// GGX NaN Regression Tests
//
// Renders confirmed-failing viewpoints extracted from training data skipped
// logs (training/training_data/skipped-*.json). Each viewpoint was skipped
// during data generation due to excessive NaN in the rendered output.
//
// Root cause: sampleGGX() in sampling.glsl computed
//   sqrt(1.0 - cos_theta * cos_theta)
// where cos_theta can slightly exceed 1.0 due to float32 rounding when
// alpha2 is near kMinGGXAlpha2 = 4e-8, producing NaN in sin_theta.
//
// Fix: sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
//
// These tests must pass (nan_count == 0) after the fix is applied.
//
// ═══════════════════════════════════════════════════════════════════════════

namespace {

struct TestContext {
    monti::app::VulkanContext& ctx = test::SharedVulkanContext();
    bool Init() { return ctx.Device() != VK_NULL_HANDLE; }
};

TextureDesc MakeEnvMap(float r, float g, float b) {
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

static std::string AssetPath(const char* filename) {
    return std::string(MONTI_TEST_ASSETS_DIR) + "/" + filename;
}

// Render 16 frames × 4 SPP and assert zero NaN pixels in diffuse and specular.
void CheckNoNaN(monti::app::VulkanContext& ctx, monti::Scene& scene,
                std::span<const MeshData> mesh_data) {
    auto mf = test::RenderSceneMultiFrame(ctx, scene, mesh_data, 16, 4);
    auto stats_d = test::AnalyzeRGBA16F(mf.diffuse.data(), test::kPixelCount);
    auto stats_s = test::AnalyzeRGBA16F(mf.specular.data(), test::kPixelCount);
    CHECK(stats_d.nan_count == 0);
    CHECK(stats_s.nan_count == 0);
    test::CleanupMultiFrameResult(ctx.Allocator(), mf);
}

static monti::CameraParams MakeCamera(glm::vec3 pos, glm::vec3 tgt, float fov_deg) {
    monti::CameraParams cam{};
    cam.position = pos;
    cam.target   = tgt;
    cam.up       = {0.0f, 1.0f, 0.0f};
    cam.vertical_fov_radians = glm::radians(fov_deg);
    cam.near_plane = monti::app::kDefaultNearPlane;
    cam.far_plane  = monti::app::kDefaultFarPlane;
    return cam;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// DamagedHelmet — pure PBR metallic/roughness, known NaN-failing viewpoints
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NaN regression: DamagedHelmet",
          "[nan_regression][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("DamagedHelmet.glb"));
    REQUIRE(result.success);

    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.3f, 0.3f, 0.3f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    SECTION("viewpoint 1472b02d") {
        scene.SetActiveCamera(MakeCamera(
            {0.7936f, 0.0074f, 1.47f},
            {-0.6859f, 0.1014f, -1.2905f},
            60.0f));
        CheckNoNaN(tc.ctx, scene, result.mesh_data);
    }

    SECTION("viewpoint f4f41ba0") {
        scene.SetActiveCamera(MakeCamera(
            {-0.9125f, 0.675f, -0.2446f},
            {0.7669f, -0.3636f, -0.5102f},
            60.0f));
        CheckNoNaN(tc.ctx, scene, result.mesh_data);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AntiqueCamera — pure PBR metallic/roughness, known NaN-failing viewpoints
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NaN regression: AntiqueCamera",
          "[nan_regression][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("AntiqueCamera.glb"));
    REQUIRE(result.success);

    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.3f, 0.3f, 0.3f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    SECTION("viewpoint 6031d3f0") {
        scene.SetActiveCamera(MakeCamera(
            {-0.258f, 5.182f, 1.0875f},
            {0.5223f, 7.9385f, -5.2298f},
            60.0f));
        CheckNoNaN(tc.ctx, scene, result.mesh_data);
    }

    SECTION("viewpoint 1d35f11b") {
        scene.SetActiveCamera(MakeCamera(
            {0.7658f, 5.9903f, 1.3352f},
            {-2.4035f, 7.6242f, -5.3909f},
            60.0f));
        CheckNoNaN(tc.ctx, scene, result.mesh_data);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FlightHelmet — multi-file glTF, known NaN-failing viewpoints
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NaN regression: FlightHelmet",
          "[nan_regression][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(
        scene, std::string(MONTI_TEST_ASSETS_DIR) + "/FlightHelmet/FlightHelmet.gltf");
    REQUIRE(result.success);

    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.3f, 0.3f, 0.3f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    SECTION("viewpoint 7513b79d") {
        scene.SetActiveCamera(MakeCamera(
            {0.1699f, 0.6183f, 0.2203f},
            {-0.5434f, 0.6373f, -0.2219f},
            61.5f));
        CheckNoNaN(tc.ctx, scene, result.mesh_data);
    }

    SECTION("viewpoint dc3a59a7") {
        scene.SetActiveCamera(MakeCamera(
            {-0.1885f, 0.6293f, 0.0098f},
            {0.5283f, 0.1889f, -0.059f},
            60.0f));
        CheckNoNaN(tc.ctx, scene, result.mesh_data);
    }
}
