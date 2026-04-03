#include <catch2/catch_test_macros.hpp>

#include <volk.h>

#include "test_helpers.h"

#include <monti/scene/Scene.h>

#include "../app/core/CameraSetup.h"
#include "gltf/GltfLoader.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace monti;
using namespace monti::vulkan;

// ═══════════════════════════════════════════════════════════════════════════
//
// Phase 10A-2: Extended Scene Download — Basic Render Verification
//
// Tests load Cauldron-Media scenes (BistroInterior, AbandonedWarehouse,
// Brutalism) via the glTF loader and verify basic rendering correctness:
// no NaN/Inf, non-zero pixel output, no Vulkan validation errors.
//
// Guarded by MONTI_EXTENDED_SCENES_DIR — only compiled when
// MONTI_DOWNLOAD_EXTENDED_SCENES=ON.
//
// ═══════════════════════════════════════════════════════════════════════════

#ifdef MONTI_EXTENDED_SCENES_DIR

namespace {

struct TestContext {
    monti::app::VulkanContext& ctx = test::SharedVulkanContext();
    bool Init() { return ctx.Device() != VK_NULL_HANDLE; }
};

struct ExtendedScene {
    std::string name;
    std::string gltf_path;
};

static std::vector<ExtendedScene> GetExtendedScenes() {
    return {
        {"BistroInterior",
         std::string(MONTI_EXTENDED_SCENES_DIR) + "/BistroInterior/scene.gltf"},
        {"AbandonedWarehouse",
         std::string(MONTI_EXTENDED_SCENES_DIR) + "/AbandonedWarehouse/AbandonedWarehouse.gltf"},
        {"Brutalism",
         std::string(MONTI_EXTENDED_SCENES_DIR) + "/Brutalism/BrutalistHall.gltf"},
    };
}

// ── Test: Extended scenes load without errors ────────────────────────────

TEST_CASE("Phase 10A-2: Extended scene loads via glTF loader",
          "[phase10a2][extended][gltf]") {
    auto scenes = GetExtendedScenes();

    for (const auto& es : scenes) {
        SECTION(es.name) {
            if (!std::filesystem::exists(es.gltf_path)) {
                WARN("Skipping " + es.name + " — file not found: " + es.gltf_path);
                continue;
            }

            Scene scene;
            auto result = gltf::LoadGltf(scene, es.gltf_path);
            REQUIRE(result.success);
            REQUIRE_FALSE(result.mesh_data.empty());
            CHECK(scene.Materials().size() > 0);
            CHECK(scene.Nodes().size() > 0);

            INFO(es.name << ": " << result.mesh_data.size() << " meshes, "
                         << scene.Materials().size() << " materials, "
                         << scene.Nodes().size() << " nodes");
        }
    }
}

// ── Test: Extended scenes render 1 SPP — no NaN/Inf ─────────────────────

TEST_CASE("Phase 10A-2: Extended scene renders 1 SPP without NaN/Inf",
          "[phase10a2][extended][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto scenes = GetExtendedScenes();

    for (const auto& es : scenes) {
        SECTION(es.name) {
            if (!std::filesystem::exists(es.gltf_path)) {
                WARN("Skipping " + es.name + " — file not found: " + es.gltf_path);
                continue;
            }

            Scene scene;
            auto result = gltf::LoadGltf(scene, es.gltf_path);
            REQUIRE(result.success);

            // Auto-fit camera
            auto camera = monti::app::ComputeDefaultCamera(scene);
            scene.SetActiveCamera(camera);

            // Grey environment for basic illumination
            auto env_tex_id = scene.AddTexture(test::MakeEnvMap(0.3f, 0.3f, 0.3f), "env_map");
            EnvironmentLight env{};
            env.hdr_lat_long = env_tex_id;
            env.intensity = 1.0f;
            scene.SetEnvironmentLight(env);

            // Render 1 SPP single frame
            auto mf = test::RenderSceneMultiFrame(ctx, scene, result.mesh_data, 1, 1);

            auto stats_d = test::AnalyzeRGBA16F(mf.diffuse.data(), test::kPixelCount);
            auto stats_s = test::AnalyzeRGBA16F(mf.specular.data(), test::kPixelCount);

            CHECK(stats_d.nan_count == 0);
            CHECK(stats_d.inf_count == 0);
            CHECK(stats_s.nan_count == 0);
            CHECK(stats_s.inf_count == 0);

            // Write diagnostic PNG
            std::string png_path = "tests/output/extended_" + es.name + "_1spp.png";
            test::WriteCombinedPNG(png_path, mf.diffuse.data(), mf.specular.data(),
                                   test::kTestWidth, test::kTestHeight);

            test::CleanupMultiFrameResult(ctx.Allocator(), mf);
        }
    }
}

// ── Test: Extended scenes render 64 SPP — meaningful output ──────────────

TEST_CASE("Phase 10A-2: Extended scene renders 64 SPP with non-zero output",
          "[phase10a2][extended][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto scenes = GetExtendedScenes();

    for (const auto& es : scenes) {
        SECTION(es.name) {
            if (!std::filesystem::exists(es.gltf_path)) {
                WARN("Skipping " + es.name + " — file not found: " + es.gltf_path);
                continue;
            }

            Scene scene;
            auto result = gltf::LoadGltf(scene, es.gltf_path);
            REQUIRE(result.success);

            // Auto-fit camera
            auto camera = monti::app::ComputeDefaultCamera(scene);
            scene.SetActiveCamera(camera);

            // Grey environment
            auto env_tex_id = scene.AddTexture(test::MakeEnvMap(0.3f, 0.3f, 0.3f), "env_map");
            EnvironmentLight env{};
            env.hdr_lat_long = env_tex_id;
            env.intensity = 1.0f;
            scene.SetEnvironmentLight(env);

            // Render 16 frames x 4 SPP = 64 total samples
            auto mf = test::RenderSceneMultiFrame(ctx, scene, result.mesh_data, 16, 4);

            auto stats_d = test::AnalyzeRGBA16F(mf.diffuse.data(), test::kPixelCount);
            auto stats_s = test::AnalyzeRGBA16F(mf.specular.data(), test::kPixelCount);

            CHECK(stats_d.nan_count == 0);
            CHECK(stats_d.inf_count == 0);
            CHECK(stats_s.nan_count == 0);
            CHECK(stats_s.inf_count == 0);

            // At least 10% of pixels should have non-zero diffuse or specular
            uint32_t total_nonzero = stats_d.nonzero_count + stats_s.nonzero_count;
            CHECK(total_nonzero > test::kPixelCount / 10);

            // Write diagnostic PNG
            std::string png_path = "tests/output/extended_" + es.name + "_64spp.png";
            test::WriteCombinedPNG(png_path, mf.diffuse.data(), mf.specular.data(),
                                   test::kTestWidth, test::kTestHeight);

            test::CleanupMultiFrameResult(ctx.Allocator(), mf);
        }
    }
}

}  // namespace

#endif  // MONTI_EXTENDED_SCENES_DIR
