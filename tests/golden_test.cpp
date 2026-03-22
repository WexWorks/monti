#include <catch2/catch_test_macros.hpp>

#include <volk.h>

#include "test_helpers.h"
#include "scenes/CornellBox.h"

#include <monti/scene/Scene.h>
#include <deni/vulkan/Denoiser.h>

#include "../app/core/CameraSetup.h"
#include "../app/core/ToneMapper.h"
#include "gltf/GltfLoader.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <stb_image.h>

using namespace monti;
using namespace monti::vulkan;

#ifndef APP_SHADER_SPV_DIR
#define APP_SHADER_SPV_DIR "build/app_shaders"
#endif

// ═══════════════════════════════════════════════════════════════════════════
//
// Phase 10A-2: Golden Reference Test Expansion
//
// Renders core scenes at 1024×1024 with 1024 total SPP (256 frames × 4 SPP),
// tone-maps, and compares against stored golden reference PNGs using FLIP.
// Extended scenes are tested when MONTI_DOWNLOAD_EXTENDED_SCENES=ON.
//
// Golden references are generated locally by the "generate golden"
// tests and stored in tests/golden/. Core scene references are committed
// to the repo; extended scene references are gitignored.
//
// ═══════════════════════════════════════════════════════════════════════════

namespace {

constexpr uint32_t kGoldenWidth = 1024;
constexpr uint32_t kGoldenHeight = 1024;
constexpr uint32_t kGoldenPixelCount = kGoldenWidth * kGoldenHeight;
constexpr uint32_t kGoldenFrames = 256;
constexpr uint32_t kGoldenSppPerFrame = 4;
constexpr float kSimpleSceneFlipThreshold = 0.05f;
constexpr float kComplexSceneFlipThreshold = 0.08f;

struct TestContext {
    monti::app::VulkanContext ctx;

    bool Init() {
        if (!ctx.CreateInstance()) return false;
        if (!ctx.CreateDevice(std::nullopt)) return false;
        return true;
    }
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

static std::string GoldenPath(const std::string& name) {
    return "tests/golden/" + name + ".png";
}

// Read a PNG file as interleaved linear RGB floats [0,1] for FLIP comparison.
// Returns empty vector if the file doesn't exist.
std::vector<float> ReadPngAsLinearRGB(const std::string& path, int& out_w, int& out_h) {
    out_w = out_h = 0;
    if (!std::filesystem::exists(path)) return {};

    int w, h, channels;
    auto* pixels = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (!pixels) return {};

    out_w = w;
    out_h = h;

    // Convert sRGB uint8 to linear float (inverse gamma 2.2)
    std::vector<float> rgb(static_cast<size_t>(w * h * 3));
    for (int i = 0; i < w * h * 3; ++i) {
        float srgb = static_cast<float>(pixels[i]) / 255.0f;
        rgb[static_cast<size_t>(i)] = std::pow(srgb, 2.2f);
    }

    stbi_image_free(pixels);
    return rgb;
}

// Render a scene at high SPP and return tonemapped linear RGB for FLIP.
// Also writes a diagnostic PNG to the golden directory.
std::vector<float> RenderGoldenRGB(monti::app::VulkanContext& ctx,
                                   Scene& scene,
                                   std::span<const MeshData> mesh_data,
                                   const std::string& name) {
    auto mf = test::RenderSceneMultiFrame(ctx, scene, mesh_data,
                                          kGoldenFrames, kGoldenSppPerFrame,
                                          kGoldenWidth, kGoldenHeight);

    auto rgb = test::TonemappedRGB(mf.diffuse.data(), mf.specular.data(),
                                   kGoldenPixelCount);

    // Write diagnostic PNG
    std::string png_path = "tests/output/golden_" + name + ".png";
    test::WriteCombinedPNG(png_path, mf.diffuse.data(), mf.specular.data(),
                           kGoldenWidth, kGoldenHeight);

    test::CleanupMultiFrameResult(ctx.Allocator(), mf);
    return rgb;
}

// Write tonemapped linear RGB as a sRGB PNG file (for golden reference storage).
bool WriteGoldenPNG(const std::string& path, const std::vector<float>& rgb,
                    uint32_t width, uint32_t height) {
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());
    std::vector<uint8_t> pixels(width * height * 3);
    for (uint32_t i = 0; i < width * height * 3; ++i) {
        float linear = std::max(rgb[i], 0.0f);
        float srgb = std::pow(linear, 1.0f / 2.2f);
        pixels[i] = static_cast<uint8_t>(std::clamp(srgb * 255.0f + 0.5f, 0.0f, 255.0f));
    }
    return stbi_write_png(path.c_str(), static_cast<int>(width),
                          static_cast<int>(height), 3, pixels.data(),
                          static_cast<int>(width * 3)) != 0;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Golden reference generation — run these manually to (re)create references
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Generate golden: CornellBox",
          "[golden_gen][.][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto [scene, mesh_data] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene);

    auto rgb = RenderGoldenRGB(tc.ctx, scene, mesh_data, "CornellBox");
    REQUIRE(WriteGoldenPNG(GoldenPath("CornellBox"), rgb,
                           kGoldenWidth, kGoldenHeight));
}

TEST_CASE("Generate golden: Box",
          "[golden_gen][.][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("Box.glb"));
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.5f, 0.5f, 0.5f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto rgb = RenderGoldenRGB(tc.ctx, scene, result.mesh_data, "Box");
    REQUIRE(WriteGoldenPNG(GoldenPath("Box"), rgb,
                           kGoldenWidth, kGoldenHeight));
}

TEST_CASE("Generate golden: DamagedHelmet",
          "[golden_gen][.][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("DamagedHelmet.glb"));
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.3f, 0.3f, 0.3f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto rgb = RenderGoldenRGB(tc.ctx, scene, result.mesh_data, "DamagedHelmet");
    REQUIRE(WriteGoldenPNG(GoldenPath("DamagedHelmet"), rgb,
                           kGoldenWidth, kGoldenHeight));
}

TEST_CASE("Generate golden: DragonAttenuation",
          "[golden_gen][.][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("DragonAttenuation.glb"));
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.5f, 0.5f, 0.5f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto rgb = RenderGoldenRGB(tc.ctx, scene, result.mesh_data, "DragonAttenuation");
    REQUIRE(WriteGoldenPNG(GoldenPath("DragonAttenuation"), rgb,
                           kGoldenWidth, kGoldenHeight));
}

TEST_CASE("Generate golden: ClearCoatTest",
          "[golden_gen][.][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("ClearCoatTest.glb"));
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.4f, 0.4f, 0.4f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto rgb = RenderGoldenRGB(tc.ctx, scene, result.mesh_data, "ClearCoatTest");
    REQUIRE(WriteGoldenPNG(GoldenPath("ClearCoatTest"), rgb,
                           kGoldenWidth, kGoldenHeight));
}

TEST_CASE("Generate golden: MorphPrimitivesTest",
          "[golden_gen][.][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("MorphPrimitivesTest.glb"));
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.4f, 0.4f, 0.4f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto rgb = RenderGoldenRGB(tc.ctx, scene, result.mesh_data, "MorphPrimitivesTest");
    REQUIRE(WriteGoldenPNG(GoldenPath("MorphPrimitivesTest"), rgb,
                           kGoldenWidth, kGoldenHeight));
}

// ═══════════════════════════════════════════════════════════════════════════
// Golden comparison tests — core scenes (always run)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Golden test: CornellBox",
          "[golden][pipeline][vulkan][integration]") {
    int ref_w, ref_h;
    auto ref_rgb = ReadPngAsLinearRGB(GoldenPath("CornellBox"), ref_w, ref_h);
    if (ref_rgb.empty()) {
        WARN("Golden reference not found: " + GoldenPath("CornellBox")
             + " — run [golden_gen] tests first");
        SKIP();
    }
    REQUIRE(ref_w == kGoldenWidth);
    REQUIRE(ref_h == kGoldenHeight);

    TestContext tc;
    REQUIRE(tc.Init());

    auto [scene, mesh_data] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene);

    auto test_rgb = RenderGoldenRGB(tc.ctx, scene, mesh_data, "CornellBox_test");

    float flip = test::ComputeMeanFlip(ref_rgb, test_rgb,
                                       kGoldenWidth, kGoldenHeight);
    INFO("CornellBox FLIP: " << flip);
    CHECK(flip < kSimpleSceneFlipThreshold);
}

TEST_CASE("Golden test: Box",
          "[golden][pipeline][vulkan][integration]") {
    int ref_w, ref_h;
    auto ref_rgb = ReadPngAsLinearRGB(GoldenPath("Box"), ref_w, ref_h);
    if (ref_rgb.empty()) {
        WARN("Golden reference not found — run [golden_gen] tests first");
        SKIP();
    }
    REQUIRE(ref_w == kGoldenWidth);
    REQUIRE(ref_h == kGoldenHeight);

    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("Box.glb"));
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.5f, 0.5f, 0.5f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto test_rgb = RenderGoldenRGB(tc.ctx, scene, result.mesh_data, "Box_test");

    float flip = test::ComputeMeanFlip(ref_rgb, test_rgb,
                                       kGoldenWidth, kGoldenHeight);
    INFO("Box FLIP: " << flip);
    CHECK(flip < kSimpleSceneFlipThreshold);
}

TEST_CASE("Golden test: DamagedHelmet",
          "[golden][pipeline][vulkan][integration]") {
    int ref_w, ref_h;
    auto ref_rgb = ReadPngAsLinearRGB(GoldenPath("DamagedHelmet"), ref_w, ref_h);
    if (ref_rgb.empty()) {
        WARN("Golden reference not found — run [golden_gen] tests first");
        SKIP();
    }
    REQUIRE(ref_w == kGoldenWidth);
    REQUIRE(ref_h == kGoldenHeight);

    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("DamagedHelmet.glb"));
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.3f, 0.3f, 0.3f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto test_rgb = RenderGoldenRGB(tc.ctx, scene, result.mesh_data, "DamagedHelmet_test");

    float flip = test::ComputeMeanFlip(ref_rgb, test_rgb,
                                       kGoldenWidth, kGoldenHeight);
    INFO("DamagedHelmet FLIP: " << flip);
    CHECK(flip < kComplexSceneFlipThreshold);
}

TEST_CASE("Golden test: DragonAttenuation",
          "[golden][pipeline][vulkan][integration]") {
    int ref_w, ref_h;
    auto ref_rgb = ReadPngAsLinearRGB(GoldenPath("DragonAttenuation"), ref_w, ref_h);
    if (ref_rgb.empty()) {
        WARN("Golden reference not found — run [golden_gen] tests first");
        SKIP();
    }
    REQUIRE(ref_w == kGoldenWidth);
    REQUIRE(ref_h == kGoldenHeight);

    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("DragonAttenuation.glb"));
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.5f, 0.5f, 0.5f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto test_rgb = RenderGoldenRGB(tc.ctx, scene, result.mesh_data, "DragonAttenuation_test");

    float flip = test::ComputeMeanFlip(ref_rgb, test_rgb,
                                       kGoldenWidth, kGoldenHeight);
    INFO("DragonAttenuation FLIP: " << flip);
    CHECK(flip < kComplexSceneFlipThreshold);
}

TEST_CASE("Golden test: ClearCoatTest",
          "[golden][pipeline][vulkan][integration]") {
    int ref_w, ref_h;
    auto ref_rgb = ReadPngAsLinearRGB(GoldenPath("ClearCoatTest"), ref_w, ref_h);
    if (ref_rgb.empty()) {
        WARN("Golden reference not found — run [golden_gen] tests first");
        SKIP();
    }
    REQUIRE(ref_w == kGoldenWidth);
    REQUIRE(ref_h == kGoldenHeight);

    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("ClearCoatTest.glb"));
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.4f, 0.4f, 0.4f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto test_rgb = RenderGoldenRGB(tc.ctx, scene, result.mesh_data, "ClearCoatTest_test");

    float flip = test::ComputeMeanFlip(ref_rgb, test_rgb,
                                       kGoldenWidth, kGoldenHeight);
    INFO("ClearCoatTest FLIP: " << flip);
    CHECK(flip < kComplexSceneFlipThreshold);
}

TEST_CASE("Golden test: MorphPrimitivesTest",
          "[golden][pipeline][vulkan][integration]") {
    int ref_w, ref_h;
    auto ref_rgb = ReadPngAsLinearRGB(GoldenPath("MorphPrimitivesTest"), ref_w, ref_h);
    if (ref_rgb.empty()) {
        WARN("Golden reference not found — run [golden_gen] tests first");
        SKIP();
    }
    REQUIRE(ref_w == kGoldenWidth);
    REQUIRE(ref_h == kGoldenHeight);

    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("MorphPrimitivesTest.glb"));
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.4f, 0.4f, 0.4f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto test_rgb = RenderGoldenRGB(tc.ctx, scene, result.mesh_data, "MorphPrimitivesTest_test");

    float flip = test::ComputeMeanFlip(ref_rgb, test_rgb,
                                       kGoldenWidth, kGoldenHeight);
    INFO("MorphPrimitivesTest FLIP: " << flip);
    CHECK(flip < kSimpleSceneFlipThreshold);
}

// ═══════════════════════════════════════════════════════════════════════════
// Golden tests — extended scenes (only when MONTI_DOWNLOAD_EXTENDED_SCENES)
// ═══════════════════════════════════════════════════════════════════════════

#ifdef MONTI_EXTENDED_SCENES_DIR

TEST_CASE("Generate golden: BistroInterior",
          "[golden_gen][.][extended][vulkan][integration]") {
    std::string gltf_path =
        std::string(MONTI_EXTENDED_SCENES_DIR) + "/BistroInterior/scene.gltf";
    if (!std::filesystem::exists(gltf_path)) SKIP("Scene not downloaded");

    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, gltf_path);
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.2f, 0.2f, 0.2f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto rgb = RenderGoldenRGB(tc.ctx, scene, result.mesh_data, "BistroInterior");
    REQUIRE(WriteGoldenPNG(GoldenPath("BistroInterior"), rgb,
                           kGoldenWidth, kGoldenHeight));
}

TEST_CASE("Generate golden: AbandonedWarehouse",
          "[golden_gen][.][extended][vulkan][integration]") {
    std::string gltf_path =
        std::string(MONTI_EXTENDED_SCENES_DIR) + "/AbandonedWarehouse/AbandonedWarehouse.gltf";
    if (!std::filesystem::exists(gltf_path)) SKIP("Scene not downloaded");

    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, gltf_path);
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.3f, 0.3f, 0.3f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto rgb = RenderGoldenRGB(tc.ctx, scene, result.mesh_data, "AbandonedWarehouse");
    REQUIRE(WriteGoldenPNG(GoldenPath("AbandonedWarehouse"), rgb,
                           kGoldenWidth, kGoldenHeight));
}

TEST_CASE("Generate golden: Brutalism",
          "[golden_gen][.][extended][vulkan][integration]") {
    std::string gltf_path =
        std::string(MONTI_EXTENDED_SCENES_DIR) + "/Brutalism/BrutalistHall.gltf";
    if (!std::filesystem::exists(gltf_path)) SKIP("Scene not downloaded");

    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, gltf_path);
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.3f, 0.3f, 0.3f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto rgb = RenderGoldenRGB(tc.ctx, scene, result.mesh_data, "Brutalism");
    REQUIRE(WriteGoldenPNG(GoldenPath("Brutalism"), rgb,
                           kGoldenWidth, kGoldenHeight));
}

TEST_CASE("Golden test: BistroInterior",
          "[golden][extended][pipeline][vulkan][integration]") {
    std::string gltf_path =
        std::string(MONTI_EXTENDED_SCENES_DIR) + "/BistroInterior/scene.gltf";
    if (!std::filesystem::exists(gltf_path)) SKIP("Scene not downloaded");

    int ref_w, ref_h;
    auto ref_rgb = ReadPngAsLinearRGB(GoldenPath("BistroInterior"), ref_w, ref_h);
    if (ref_rgb.empty()) {
        WARN("Golden reference not found — run [golden_gen] tests first");
        SKIP();
    }
    REQUIRE(ref_w == kGoldenWidth);
    REQUIRE(ref_h == kGoldenHeight);

    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, gltf_path);
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.2f, 0.2f, 0.2f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto test_rgb = RenderGoldenRGB(tc.ctx, scene, result.mesh_data, "BistroInterior_test");

    float flip = test::ComputeMeanFlip(ref_rgb, test_rgb,
                                       kGoldenWidth, kGoldenHeight);
    INFO("BistroInterior FLIP: " << flip);
    CHECK(flip < kComplexSceneFlipThreshold);
}

TEST_CASE("Golden test: AbandonedWarehouse",
          "[golden][extended][pipeline][vulkan][integration]") {
    std::string gltf_path =
        std::string(MONTI_EXTENDED_SCENES_DIR) + "/AbandonedWarehouse/AbandonedWarehouse.gltf";
    if (!std::filesystem::exists(gltf_path)) SKIP("Scene not downloaded");

    int ref_w, ref_h;
    auto ref_rgb = ReadPngAsLinearRGB(GoldenPath("AbandonedWarehouse"), ref_w, ref_h);
    if (ref_rgb.empty()) {
        WARN("Golden reference not found — run [golden_gen] tests first");
        SKIP();
    }
    REQUIRE(ref_w == kGoldenWidth);
    REQUIRE(ref_h == kGoldenHeight);

    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, gltf_path);
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.3f, 0.3f, 0.3f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto test_rgb = RenderGoldenRGB(tc.ctx, scene, result.mesh_data, "AbandonedWarehouse_test");

    float flip = test::ComputeMeanFlip(ref_rgb, test_rgb,
                                       kGoldenWidth, kGoldenHeight);
    INFO("AbandonedWarehouse FLIP: " << flip);
    CHECK(flip < kComplexSceneFlipThreshold);
}

TEST_CASE("Golden test: Brutalism",
          "[golden][extended][pipeline][vulkan][integration]") {
    std::string gltf_path =
        std::string(MONTI_EXTENDED_SCENES_DIR) + "/Brutalism/BrutalistHall.gltf";
    if (!std::filesystem::exists(gltf_path)) SKIP("Scene not downloaded");

    int ref_w, ref_h;
    auto ref_rgb = ReadPngAsLinearRGB(GoldenPath("Brutalism"), ref_w, ref_h);
    if (ref_rgb.empty()) {
        WARN("Golden reference not found — run [golden_gen] tests first");
        SKIP();
    }
    REQUIRE(ref_w == kGoldenWidth);
    REQUIRE(ref_h == kGoldenHeight);

    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, gltf_path);
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.3f, 0.3f, 0.3f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto test_rgb = RenderGoldenRGB(tc.ctx, scene, result.mesh_data, "Brutalism_test");

    float flip = test::ComputeMeanFlip(ref_rgb, test_rgb,
                                       kGoldenWidth, kGoldenHeight);
    INFO("Brutalism FLIP: " << flip);
    CHECK(flip < kComplexSceneFlipThreshold);
}

#endif  // MONTI_EXTENDED_SCENES_DIR
