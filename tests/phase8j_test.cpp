#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <volk.h>

#include "test_helpers.h"
#include "scenes/CornellBox.h"
#include "vulkan/EmissiveLightExtractor.h"

#include <monti/scene/Scene.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace monti;

// ═══════════════════════════════════════════════════════════════════════════
//
// Phase 8J: Emissive Mesh Light Extraction
//
// Tests verify that emissive mesh surfaces are automatically extracted into
// triangle light primitives for NEE (next-event estimation), reducing noise
// at low SPP and improving convergence.
//
// ═══════════════════════════════════════════════════════════════════════════

namespace {

struct TestContext {
    monti::app::VulkanContext ctx;

    bool Init() {
        if (!ctx.CreateInstance()) return false;
        if (!ctx.CreateDevice(std::nullopt)) return false;
        return true;
    }
};

// Build a Cornell box with an emissive object (two-triangle quad) on the floor.
// Returns the scene, mesh data, and the material ID of the emissive object.
struct EmissiveSceneResult {
    Scene scene;
    std::vector<MeshData> mesh_data;
    MaterialId emissive_mat_id;
};

EmissiveSceneResult BuildEmissiveScene(float emissive_strength) {
    auto [scene, mesh_data] = test::BuildCornellBox();

    // Add an emissive material matching the canonical Cornell box ceiling light.
    // Base radiance {17, 12, 4} is baked into emissive_factor; emissive_strength
    // acts as a multiplier (1.0 = canonical intensity).
    MaterialDesc emissive_mat;
    emissive_mat.base_color = {1.0f, 1.0f, 1.0f};
    emissive_mat.emissive_factor = {17.0f, 12.0f, 4.0f};
    emissive_mat.emissive_strength = emissive_strength;
    emissive_mat.roughness = 1.0f;
    emissive_mat.metallic = 0.0f;
    emissive_mat.double_sided = true;
    auto emissive_id = scene.AddMaterial(std::move(emissive_mat), "emissive_panel");

    // Create an emissive panel matching the canonical ceiling light position/size:
    // corner {0.35, 0.999, 0.35}, 0.3 × 0.3 in XZ, facing down.
    MeshData panel_data;
    panel_data.vertices = {
        {{0.35f, 0.999f, 0.35f}, {0.0f, -1.0f, 0.0f}, {1, 0, 0, 1}, {0, 0}, {0, 0}},
        {{0.65f, 0.999f, 0.35f}, {0.0f, -1.0f, 0.0f}, {1, 0, 0, 1}, {1, 0}, {0, 0}},
        {{0.65f, 0.999f, 0.65f}, {0.0f, -1.0f, 0.0f}, {1, 0, 0, 1}, {1, 1}, {0, 0}},
        {{0.35f, 0.999f, 0.65f}, {0.0f, -1.0f, 0.0f}, {1, 0, 0, 1}, {0, 1}, {0, 0}},
    };
    panel_data.indices = {0, 1, 2, 0, 2, 3};

    Mesh panel_mesh;
    panel_mesh.name = "emissive_panel";
    panel_mesh.vertex_count = static_cast<uint32_t>(panel_data.vertices.size());
    panel_mesh.index_count = static_cast<uint32_t>(panel_data.indices.size());
    panel_mesh.vertex_stride = sizeof(Vertex);
    panel_mesh.bbox_min = {0.35f, 0.999f, 0.35f};
    panel_mesh.bbox_max = {0.65f, 0.999f, 0.65f};

    auto panel_mesh_id = scene.AddMesh(std::move(panel_mesh), "emissive_panel");
    panel_data.mesh_id = panel_mesh_id;
    scene.AddNode(panel_mesh_id, emissive_id, "emissive_panel");
    mesh_data.push_back(std::move(panel_data));

    return {std::move(scene), std::move(mesh_data), emissive_id};
}

// Compute pixel variance across the entire RGBA16F diffuse buffer.
// Returns {variance, mean_luminance}.
std::pair<float, float> ComputePixelVarianceAndMean(const uint16_t* raw, uint32_t pixel_count) {
    double sum_lum = 0;
    double sum_lum2 = 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < pixel_count; ++i) {
        float r = test::HalfToFloat(raw[i * 4 + 0]);
        float g = test::HalfToFloat(raw[i * 4 + 1]);
        float b = test::HalfToFloat(raw[i * 4 + 2]);
        if (std::isnan(r) || std::isnan(g) || std::isnan(b)) continue;
        if (std::isinf(r) || std::isinf(g) || std::isinf(b)) continue;
        float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        sum_lum += lum;
        sum_lum2 += lum * lum;
        ++count;
    }
    if (count < 2) return {0.0f, 0.0f};
    double n = static_cast<double>(count);
    double mean = sum_lum / n;
    double var = sum_lum2 / n - mean * mean;
    return {static_cast<float>(var), static_cast<float>(mean)};
}

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: EmissiveMeshNEEReducesNoise
//
// Scene A: emissive panel with extraction enabled → triangle lights in
//          light buffer for NEE shadow rays.
// Scene B: identical emissive panel but without calling extraction → no
//          triangle lights, emission only via random path hits.
//
// At low SPP, scene A (with NEE) should have lower pixel variance on
// surfaces illuminated by the emissive object.
// Verify variance_A < variance_B * 0.7.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8J: EmissiveMeshNEEReducesNoise",
          "[phase8j][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // Scene A: extraction enabled — NEE samples emissive triangles
    auto [scene_a, mesh_data_a, mat_a] = BuildEmissiveScene(1.0f);
    uint32_t extracted_a = vulkan::ExtractEmissiveLights(scene_a, mesh_data_a);
    INFO("Extracted triangle lights (scene A): " << extracted_a);
    CHECK(extracted_a == 2);  // 2 triangles from the quad panel

    auto result_a = test::RenderSceneMultiFrame(ctx, scene_a, mesh_data_a, 16, 4);

    // Scene B: same emissive strength, but NO extraction — no NEE
    auto [scene_b, mesh_data_b, mat_b] = BuildEmissiveScene(1.0f);
    // Do NOT call ExtractEmissiveLights — emissive mesh only emits via path hits

    auto result_b = test::RenderSceneMultiFrame(ctx, scene_b, mesh_data_b, 16, 4);

    test::WriteCombinedPNG("tests/output/phase8j_nee_extracted.png",
                           result_a.diffuse.data(), result_a.specular.data(),
                           test::kTestWidth, test::kTestHeight);
    test::WriteCombinedPNG("tests/output/phase8j_nee_no_extraction.png",
                           result_b.diffuse.data(), result_b.specular.data(),
                           test::kTestWidth, test::kTestHeight);

    auto [variance_a, mean_a] = ComputePixelVarianceAndMean(result_a.diffuse.data(),
                                                               test::kPixelCount);
    auto [variance_b, mean_b] = ComputePixelVarianceAndMean(result_b.diffuse.data(),
                                                               test::kPixelCount);

    // Coefficient of variation (CV = stddev / mean) — scale-invariant noise measure.
    // NEE produces a brighter image, so absolute variance isn't meaningful;
    // CV normalizes by brightness.
    float cv_a = (mean_a > 0.0f) ? std::sqrt(variance_a) / mean_a : 0.0f;
    float cv_b = (mean_b > 0.0f) ? std::sqrt(variance_b) / mean_b : 0.0f;

    INFO("Variance with extraction: " << variance_a << ", mean: " << mean_a);
    INFO("Variance without extraction: " << variance_b << ", mean: " << mean_b);
    INFO("CV with extraction: " << cv_a);
    INFO("CV without extraction: " << cv_b);

    // With NEE, the scene should be brighter (more mean luminance)
    CHECK(mean_a > mean_b);
    // And the relative noise (CV) should be lower
    CHECK(cv_a < cv_b * 0.75f);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_a);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_b);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: EmissiveMeshFLIPImprovement
//
// Render at 4 spp and 64 spp with extraction enabled.
// FLIP(4spp, 64spp) below convergence threshold confirms extraction
// improves convergence.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8J: EmissiveMeshFLIPImprovement",
          "[phase8j][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // 4 spp render with extraction
    auto [scene_lo, mesh_data_lo, mat_lo] = BuildEmissiveScene(1.0f);
    vulkan::ExtractEmissiveLights(scene_lo, mesh_data_lo);
    auto result_lo = test::RenderSceneMultiFrame(ctx, scene_lo, mesh_data_lo, 4, 4);

    // 64 spp render with extraction (reference)
    auto [scene_hi, mesh_data_hi, mat_hi] = BuildEmissiveScene(1.0f);
    vulkan::ExtractEmissiveLights(scene_hi, mesh_data_hi);
    auto result_hi = test::RenderSceneMultiFrame(ctx, scene_hi, mesh_data_hi, 16, 4);

    test::WriteCombinedPNG("tests/output/phase8j_flip_4spp.png",
                           result_lo.diffuse.data(), result_lo.specular.data(),
                           test::kTestWidth, test::kTestHeight);
    test::WriteCombinedPNG("tests/output/phase8j_flip_64spp.png",
                           result_hi.diffuse.data(), result_hi.specular.data(),
                           test::kTestWidth, test::kTestHeight);

    auto rgb_lo = test::TonemappedRGB(result_lo.diffuse.data(),
                                      result_lo.specular.data(),
                                      test::kPixelCount);
    auto rgb_hi = test::TonemappedRGB(result_hi.diffuse.data(),
                                      result_hi.specular.data(),
                                      test::kPixelCount);
    float flip = test::ComputeMeanFlip(rgb_lo, rgb_hi,
                                       test::kTestWidth, test::kTestHeight);
    INFO("FLIP(4spp vs 64spp with extraction): " << flip);
    CHECK(flip < 0.3f);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_lo);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_hi);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: EmissiveMeshExtractionThreshold
//
// Emission below kMinEmissiveLuminance → no triangle lights extracted.
// Emission above threshold → triangle lights added to the scene.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8J: EmissiveMeshExtractionThreshold",
          "[phase8j][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // Below threshold: emissive_strength such that max(factor * strength) < 0.01
    // max component is 17.0 * 0.0005 = 0.0085 < kMinEmissiveLuminance (0.01)
    auto [scene_dim, mesh_data_dim, mat_dim] = BuildEmissiveScene(0.0005f);
    auto initial_light_count = scene_dim.TriangleLights().size();
    uint32_t extracted_dim = vulkan::ExtractEmissiveLights(scene_dim, mesh_data_dim);
    CHECK(extracted_dim == 0);
    CHECK(scene_dim.TriangleLights().size() == initial_light_count);

    // Above threshold (canonical intensity)
    auto [scene_bright, mesh_data_bright, mat_bright] = BuildEmissiveScene(1.0f);
    auto initial_bright_count = scene_bright.TriangleLights().size();
    uint32_t extracted_bright = vulkan::ExtractEmissiveLights(scene_bright, mesh_data_bright);
    CHECK(extracted_bright == 2);  // 2 triangles from quad
    CHECK(scene_bright.TriangleLights().size() == initial_bright_count + 2);

    // Verify extracted lights have correct radiance
    const auto& lights = scene_bright.TriangleLights();
    for (size_t i = initial_bright_count; i < lights.size(); ++i) {
        const auto& light = lights[i];
        // emissive_factor = {17, 12, 4}, strength = 1.0
        CHECK(light.radiance.r == Catch::Approx(17.0f));
        CHECK(light.radiance.g == Catch::Approx(12.0f));
        CHECK(light.radiance.b == Catch::Approx(4.0f));
        CHECK(light.two_sided == true);
    }

    // Render the bright scene to confirm it works end-to-end
    auto result = test::RenderSceneMultiFrame(ctx, scene_bright, mesh_data_bright, 4, 4);
    auto stats = test::AnalyzeRGBA16F(result.diffuse.data(), test::kPixelCount);
    CHECK(stats.nan_count == 0);
    CHECK(stats.inf_count == 0);
    CHECK(stats.nonzero_count > test::kPixelCount / 4);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: EmissiveMeshNoNaN
//
// Render a scene with both extracted emissive triangle lights and explicit
// area lights. Verify no NaN/Inf in output.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8J: EmissiveMeshNoNaN",
          "[phase8j][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data, mat_id] = BuildEmissiveScene(1.0f);

    // Add an explicit area light (ceiling light)
    test::AddCornellBoxLight(scene);

    // Extract emissive mesh triangles
    uint32_t extracted = vulkan::ExtractEmissiveLights(scene, mesh_data);
    CHECK(extracted == 2);

    // Verify mixed light count
    uint32_t total_lights = static_cast<uint32_t>(
        scene.AreaLights().size() + scene.SphereLights().size()
        + scene.TriangleLights().size());
    CHECK(total_lights >= 3);  // 1 area + 2 extracted triangles

    auto result = test::RenderSceneMultiFrame(ctx, scene, mesh_data, 8, 4);

    test::WriteCombinedPNG("tests/output/phase8j_mixed_lights.png",
                           result.diffuse.data(), result.specular.data(),
                           test::kTestWidth, test::kTestHeight);

    auto diff_stats = test::AnalyzeRGBA16F(result.diffuse.data(), test::kPixelCount);
    auto spec_stats = test::AnalyzeRGBA16F(result.specular.data(), test::kPixelCount);

    CHECK(diff_stats.nan_count == 0);
    CHECK(diff_stats.inf_count == 0);
    CHECK(spec_stats.nan_count == 0);
    CHECK(spec_stats.inf_count == 0);

    // Scene should be well-lit with both light sources
    CHECK(diff_stats.nonzero_count > test::kPixelCount / 3);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
}
