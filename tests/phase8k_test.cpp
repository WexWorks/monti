#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <volk.h>

#include "test_helpers.h"
#include "scenes/CornellBox.h"
#include "vulkan/EmissiveLightExtractor.h"

#include <monti/scene/Scene.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

using namespace monti;

// ═══════════════════════════════════════════════════════════════════════════
//
// Phase 8K: Weighted Reservoir Sampling for NEE
//
// Tests verify that the hybrid NEE strategy works correctly:
// - Direct-sample path (≤ kMaxDirectSampleLights) produces identical
//   results to the pre-8K implementation.
// - WRS path (> kMaxDirectSampleLights) converges correctly and scales
//   sublinearly with light count.
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

// Build a Cornell box with N procedural sphere lights on the ceiling.
// Used to exercise the WRS path (light_count > kMaxDirectSampleLights).
void AddProceduralSphereLights(Scene& scene, uint32_t count,
                               float radius = 0.02f,
                               glm::vec3 radiance = {5.0f, 5.0f, 5.0f}) {
    // Distribute lights in a grid on the ceiling (y = 0.95)
    uint32_t grid = static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<float>(count))));
    uint32_t placed = 0;
    for (uint32_t gx = 0; gx < grid && placed < count; ++gx) {
        for (uint32_t gz = 0; gz < grid && placed < count; ++gz) {
            float x = 0.1f + 0.8f * (static_cast<float>(gx) + 0.5f) / static_cast<float>(grid);
            float z = 0.1f + 0.8f * (static_cast<float>(gz) + 0.5f) / static_cast<float>(grid);
            SphereLight sl;
            sl.center = {x, 0.95f, z};
            sl.radius = radius;
            sl.radiance = radiance;
            scene.AddSphereLight(sl);
            ++placed;
        }
    }
}

// Compute pixel variance across the entire RGBA16F diffuse buffer.
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
// Test 1: DirectSampleSingleLightUnchanged
//
// Render Cornell box with exactly 1 area light (direct-sample path).
// Confirm output is valid and well-lit. This exercises the verbatim
// copy of the old O(N) loop in the direct-sample branch.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8K: DirectSampleSingleLightUnchanged",
          "[phase8k][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene);

    auto result = test::RenderSceneMultiFrame(ctx, scene, mesh_data, 16, 4);

    test::WriteCombinedPNG("tests/output/phase8k_direct_single.png",
                           result.diffuse.data(), result.specular.data(),
                           test::kTestWidth, test::kTestHeight);

    auto stats = test::AnalyzeRGBA16F(result.diffuse.data(), test::kPixelCount);
    CHECK(stats.nan_count == 0);
    CHECK(stats.inf_count == 0);
    CHECK(stats.nonzero_count > test::kPixelCount / 4);
    CHECK(stats.has_color_variation);

    auto [variance, mean] = ComputePixelVarianceAndMean(
        result.diffuse.data(), test::kPixelCount);
    INFO("Single light direct-sample: variance=" << variance << " mean=" << mean);
    CHECK(mean > 0.01f);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: DirectSampleFewLightsUnchanged
//
// Render Cornell box with 3 mixed lights (quad + sphere + triangle).
// Exercises the direct-sample path at the threshold boundary.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8K: DirectSampleFewLightsUnchanged",
          "[phase8k][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = test::BuildCornellBox();

    // 1 quad light (ceiling)
    test::AddCornellBoxLight(scene);

    // 1 sphere light
    SphereLight sl;
    sl.center = {0.7f, 0.85f, 0.3f};
    sl.radius = 0.05f;
    sl.radiance = {8.0f, 6.0f, 4.0f};
    scene.AddSphereLight(sl);

    // 1 triangle light
    TriangleLight tl;
    tl.v0 = {0.2f, 0.999f, 0.6f};
    tl.v1 = {0.4f, 0.999f, 0.6f};
    tl.v2 = {0.3f, 0.999f, 0.8f};
    tl.radiance = {5.0f, 10.0f, 5.0f};
    tl.two_sided = false;
    scene.AddTriangleLight(tl);

    auto result = test::RenderSceneMultiFrame(ctx, scene, mesh_data, 16, 4);

    test::WriteCombinedPNG("tests/output/phase8k_direct_few.png",
                           result.diffuse.data(), result.specular.data(),
                           test::kTestWidth, test::kTestHeight);

    auto stats = test::AnalyzeRGBA16F(result.diffuse.data(), test::kPixelCount);
    CHECK(stats.nan_count == 0);
    CHECK(stats.inf_count == 0);
    CHECK(stats.nonzero_count > test::kPixelCount / 4);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: WRSManyLightsConvergence
//
// Build a procedural scene with ~50 sphere lights (WRS path active).
// Render at 16 spp and 64 spp. FLIP between the two should be below
// the convergence threshold, confirming WRS produces unbiased results.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8K: WRSManyLightsConvergence",
          "[phase8k][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // Low SPP render
    auto [scene_lo, mesh_data_lo] = test::BuildCornellBox();
    AddProceduralSphereLights(scene_lo, 50);
    auto result_lo = test::RenderSceneMultiFrame(ctx, scene_lo, mesh_data_lo, 4, 4);

    // High SPP render (reference)
    auto [scene_hi, mesh_data_hi] = test::BuildCornellBox();
    AddProceduralSphereLights(scene_hi, 50);
    auto result_hi = test::RenderSceneMultiFrame(ctx, scene_hi, mesh_data_hi, 16, 4);

    test::WriteCombinedPNG("tests/output/phase8k_wrs_16spp.png",
                           result_lo.diffuse.data(), result_lo.specular.data(),
                           test::kTestWidth, test::kTestHeight);
    test::WriteCombinedPNG("tests/output/phase8k_wrs_64spp.png",
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
    INFO("FLIP(16spp vs 64spp, 50 lights WRS): " << flip);
    CHECK(flip < 0.3f);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_lo);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_hi);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: WRSManyLightsSublinearScaling
//
// Render two configurations: 10 lights and 200 lights at identical SPP.
// Verify frame time ratio (200-light / 10-light) < 3.0.
// WRS reduces shadow ray traces from O(N) to O(1) per hit, so the cost
// difference should be modest (only the WRS selection loop iterates N).
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8K: WRSManyLightsSublinearScaling",
          "[phase8k][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // 10-light scene (above threshold to force WRS)
    auto [scene_10, mesh_data_10] = test::BuildCornellBox();
    AddProceduralSphereLights(scene_10, 10);

    auto start_10 = std::chrono::high_resolution_clock::now();
    auto result_10 = test::RenderSceneMultiFrame(ctx, scene_10, mesh_data_10, 4, 4);
    auto end_10 = std::chrono::high_resolution_clock::now();
    double time_10 = std::chrono::duration<double, std::milli>(end_10 - start_10).count();

    // 200-light scene
    auto [scene_200, mesh_data_200] = test::BuildCornellBox();
    AddProceduralSphereLights(scene_200, 200);

    auto start_200 = std::chrono::high_resolution_clock::now();
    auto result_200 = test::RenderSceneMultiFrame(ctx, scene_200, mesh_data_200, 4, 4);
    auto end_200 = std::chrono::high_resolution_clock::now();
    double time_200 = std::chrono::duration<double, std::milli>(end_200 - start_200).count();

    double ratio = time_200 / std::max(time_10, 0.001);
    INFO("10-light time: " << time_10 << " ms");
    INFO("200-light time: " << time_200 << " ms");
    INFO("Scaling ratio (200/10): " << ratio);

    // With WRS, 200 lights should not be 20x slower than 10 lights.
    // Allow generous 3x margin for WRS selection loop overhead.
    CHECK(ratio < 3.0);

    // Both should be valid
    auto stats_10 = test::AnalyzeRGBA16F(result_10.diffuse.data(), test::kPixelCount);
    auto stats_200 = test::AnalyzeRGBA16F(result_200.diffuse.data(), test::kPixelCount);
    CHECK(stats_10.nan_count == 0);
    CHECK(stats_200.nan_count == 0);

    test::WriteCombinedPNG("tests/output/phase8k_wrs_10lights.png",
                           result_10.diffuse.data(), result_10.specular.data(),
                           test::kTestWidth, test::kTestHeight);
    test::WriteCombinedPNG("tests/output/phase8k_wrs_200lights.png",
                           result_200.diffuse.data(), result_200.specular.data(),
                           test::kTestWidth, test::kTestHeight);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_10);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_200);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: WRSManyLightsNoNaN
//
// Render a scene with 210 lights at 1 spp. Verify no NaN/Inf.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8K: WRSManyLightsNoNaN",
          "[phase8k][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = test::BuildCornellBox();

    // 100 sphere lights on ceiling
    AddProceduralSphereLights(scene, 100);

    // 10 quad lights along walls
    for (int i = 0; i < 10; ++i) {
        AreaLight al;
        float y = 0.1f + 0.8f * static_cast<float>(i) / 10.0f;
        al.corner = {0.001f, y, 0.3f};
        al.edge_a = {0.0f, 0.08f, 0.0f};
        al.edge_b = {0.0f, 0.0f, 0.08f};
        al.radiance = {3.0f, 3.0f, 3.0f};
        al.two_sided = true;
        scene.AddAreaLight(al);
    }

    // ~100 small triangle lights
    for (int i = 0; i < 100; ++i) {
        float x = 0.1f + 0.8f * static_cast<float>(i % 10) / 10.0f;
        float z = 0.1f + 0.8f * static_cast<float>(i / 10) / 10.0f;
        TriangleLight tl;
        tl.v0 = {x, 0.999f, z};
        tl.v1 = {x + 0.02f, 0.999f, z};
        tl.v2 = {x + 0.01f, 0.999f, z + 0.02f};
        tl.radiance = {2.0f, 2.0f, 2.0f};
        tl.two_sided = false;
        scene.AddTriangleLight(tl);
    }

    auto result = test::RenderSceneMultiFrame(ctx, scene, mesh_data, 1, 4);

    test::WriteCombinedPNG("tests/output/phase8k_wrs_210lights.png",
                           result.diffuse.data(), result.specular.data(),
                           test::kTestWidth, test::kTestHeight);

    auto diff_stats = test::AnalyzeRGBA16F(result.diffuse.data(), test::kPixelCount);
    auto spec_stats = test::AnalyzeRGBA16F(result.specular.data(), test::kPixelCount);

    CHECK(diff_stats.nan_count == 0);
    CHECK(diff_stats.inf_count == 0);
    CHECK(spec_stats.nan_count == 0);
    CHECK(spec_stats.inf_count == 0);
    CHECK(diff_stats.nonzero_count > test::kPixelCount / 4);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: WRSDirectSampleEquivalence
//
// Render identical lighting via both code paths:
//   A) 4 sphere lights → direct-sample path (light_count <= threshold)
//   B) same 4 lights + 1 near-zero-radiance dummy → WRS path (5 > threshold)
// The dummy light contributes negligibly, so both scenes should produce
// nearly identical images. This directly verifies that the WRS one-sample
// estimator converges to the same answer as the direct-sample sum.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8K: WRSDirectSampleEquivalence",
          "[phase8k][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // Direct-sample path: exactly 4 lights (at kMaxDirectSampleLights)
    auto [scene_ds, mesh_data_ds] = test::BuildCornellBox();
    AddProceduralSphereLights(scene_ds, 4, 0.03f, {8.0f, 8.0f, 8.0f});
    auto result_ds = test::RenderSceneMultiFrame(ctx, scene_ds, mesh_data_ds, 64, 4);

    // WRS path: same 4 lights + 1 near-zero-radiance dummy to exceed threshold
    auto [scene_wrs, mesh_data_wrs] = test::BuildCornellBox();
    AddProceduralSphereLights(scene_wrs, 4, 0.03f, {8.0f, 8.0f, 8.0f});
    SphereLight dummy;
    dummy.center = {0.5f, 0.95f, 0.5f};
    dummy.radius = 0.001f;
    dummy.radiance = {0.0001f, 0.0001f, 0.0001f};
    scene_wrs.AddSphereLight(dummy);
    auto result_wrs = test::RenderSceneMultiFrame(ctx, scene_wrs, mesh_data_wrs, 64, 4);

    test::WriteCombinedPNG("tests/output/phase8k_equiv_direct.png",
                           result_ds.diffuse.data(), result_ds.specular.data(),
                           test::kTestWidth, test::kTestHeight);
    test::WriteCombinedPNG("tests/output/phase8k_equiv_wrs.png",
                           result_wrs.diffuse.data(), result_wrs.specular.data(),
                           test::kTestWidth, test::kTestHeight);

    // FLIP comparison: both paths should converge to the same image
    auto rgb_ds = test::TonemappedRGB(result_ds.diffuse.data(),
                                      result_ds.specular.data(),
                                      test::kPixelCount);
    auto rgb_wrs = test::TonemappedRGB(result_wrs.diffuse.data(),
                                       result_wrs.specular.data(),
                                       test::kPixelCount);
    float flip = test::ComputeMeanFlip(rgb_ds, rgb_wrs,
                                       test::kTestWidth, test::kTestHeight);
    INFO("FLIP(direct-sample vs WRS, 4+1 lights): " << flip);
    CHECK(flip < 0.15f);

    // Mean luminance should be nearly identical (energy equivalence)
    auto [var_ds, mean_ds] = ComputePixelVarianceAndMean(
        result_ds.diffuse.data(), test::kPixelCount);
    auto [var_wrs, mean_wrs] = ComputePixelVarianceAndMean(
        result_wrs.diffuse.data(), test::kPixelCount);
    float energy_ratio = mean_wrs / std::max(mean_ds, 0.001f);
    INFO("Mean luminance: direct=" << mean_ds
         << " WRS=" << mean_wrs << " ratio=" << energy_ratio);
    CHECK(energy_ratio > 0.85f);
    CHECK(energy_ratio < 1.15f);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_ds);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_wrs);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 7: WRSEnergyProportionality
//
// Render the same geometry and light positions with radiance R and 2R.
// Mean luminance should scale ~2x. This catches WRS PDF errors: if
// p_wrs or the 1/nee_pdf compensation is wrong, the ratio deviates.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8K: WRSEnergyProportionality",
          "[phase8k][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // Scene A: 10 lights, radiance = (3, 3, 3) → WRS path
    auto [scene_a, mesh_data_a] = test::BuildCornellBox();
    AddProceduralSphereLights(scene_a, 10, 0.02f, {3.0f, 3.0f, 3.0f});
    auto result_a = test::RenderSceneMultiFrame(ctx, scene_a, mesh_data_a, 64, 4);

    // Scene B: same 10 lights, radiance = (6, 6, 6) → WRS path
    auto [scene_b, mesh_data_b] = test::BuildCornellBox();
    AddProceduralSphereLights(scene_b, 10, 0.02f, {6.0f, 6.0f, 6.0f});
    auto result_b = test::RenderSceneMultiFrame(ctx, scene_b, mesh_data_b, 64, 4);

    test::WriteCombinedPNG("tests/output/phase8k_prop_1x.png",
                           result_a.diffuse.data(), result_a.specular.data(),
                           test::kTestWidth, test::kTestHeight);
    test::WriteCombinedPNG("tests/output/phase8k_prop_2x.png",
                           result_b.diffuse.data(), result_b.specular.data(),
                           test::kTestWidth, test::kTestHeight);

    auto [var_a, mean_a] = ComputePixelVarianceAndMean(
        result_a.diffuse.data(), test::kPixelCount);
    auto [var_b, mean_b] = ComputePixelVarianceAndMean(
        result_b.diffuse.data(), test::kPixelCount);

    float ratio = mean_b / std::max(mean_a, 0.001f);
    INFO("Mean luminance: 1x=" << mean_a << " 2x=" << mean_b
         << " ratio=" << ratio);

    // Doubling radiance should roughly double mean luminance.
    // Tolerance accounts for Monte Carlo noise and firefly clamping.
    CHECK(ratio > 1.7f);
    CHECK(ratio < 2.3f);
    CHECK(mean_a > 0.01f);
    CHECK(mean_b > 0.01f);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_a);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_b);
}
