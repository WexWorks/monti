#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <volk.h>

#include "test_helpers.h"
#include "scenes/CornellBox.h"

#include <monti/scene/Scene.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace monti;

// ═══════════════════════════════════════════════════════════════════════════
//
// Phase 8G: Spherical Area Lights + Triangle Light Primitives
//
// Tests verify sphere light illumination, soft shadow variation from radius,
// triangle light illumination (including two-sided), quad light backward
// compatibility after the PackedAreaLight → PackedLight migration, mixed
// light convergence, and degenerate light rejection.
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

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: Sphere light illumination
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8G: SphereLightIllumination",
          "[phase8g][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // Render with sphere light
    auto [scene, mesh_data] = test::BuildCornellBox();
    SphereLight sphere{};
    sphere.center = {0.5f, 0.8f, 0.5f};
    sphere.radius = 0.1f;
    sphere.radiance = {50.0f, 50.0f, 50.0f};
    scene.AddSphereLight(sphere);

    auto result = test::RenderSceneMultiFrame(ctx, scene, mesh_data, 16, 4);

    auto* diffuse_raw = result.diffuse.data();
    auto* specular_raw = result.specular.data();

    test::WriteCombinedPNG("tests/output/phase8g_sphere_light.png",
                           diffuse_raw, specular_raw,
                           test::kTestWidth, test::kTestHeight);

    auto stats = test::AnalyzeRGBA16F(diffuse_raw, test::kPixelCount);
    CHECK(stats.nan_count == 0);
    CHECK(stats.inf_count == 0);
    CHECK(stats.nonzero_count > test::kPixelCount / 4);

    // Render without any light (control)
    auto [scene_dark, mesh_data_dark] = test::BuildCornellBox();
    auto result_dark = test::RenderSceneMultiFrame(ctx, scene_dark, mesh_data_dark, 16, 4);

    auto stats_dark = test::AnalyzeRGBA16F(result_dark.diffuse.data(), test::kPixelCount);

    // Sphere-lit render should have significantly more luminance
    double lit_luminance = stats.sum_r + stats.sum_g + stats.sum_b;
    double dark_luminance = stats_dark.sum_r + stats_dark.sum_g + stats_dark.sum_b;
    CHECK(lit_luminance > dark_luminance * 2.0);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_dark);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Sphere light soft shadow — large vs small radius
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8G: SphereLightSoftShadow",
          "[phase8g][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // Large sphere light (soft shadows)
    auto [scene_large, mesh_data_large] = test::BuildCornellBox();
    SphereLight large_sphere{};
    large_sphere.center = {0.5f, 0.8f, 0.5f};
    large_sphere.radius = 0.3f;
    large_sphere.radiance = {30.0f, 30.0f, 30.0f};
    scene_large.AddSphereLight(large_sphere);

    auto result_large = test::RenderSceneMultiFrame(ctx, scene_large, mesh_data_large, 16, 4);

    // Small sphere light (hard shadows)
    auto [scene_small, mesh_data_small] = test::BuildCornellBox();
    SphereLight small_sphere{};
    small_sphere.center = {0.5f, 0.8f, 0.5f};
    small_sphere.radius = 0.02f;
    small_sphere.radiance = {30.0f, 30.0f, 30.0f};
    scene_small.AddSphereLight(small_sphere);

    auto result_small = test::RenderSceneMultiFrame(ctx, scene_small, mesh_data_small, 16, 4);

    test::WriteCombinedPNG("tests/output/phase8g_sphere_large.png",
                           result_large.diffuse.data(), result_large.specular.data(),
                           test::kTestWidth, test::kTestHeight);
    test::WriteCombinedPNG("tests/output/phase8g_sphere_small.png",
                           result_small.diffuse.data(), result_small.specular.data(),
                           test::kTestWidth, test::kTestHeight);

    auto rgb_large = test::TonemappedRGB(result_large.diffuse.data(),
                                         result_large.specular.data(),
                                         test::kPixelCount);
    auto rgb_small = test::TonemappedRGB(result_small.diffuse.data(),
                                         result_small.specular.data(),
                                         test::kPixelCount);
    float flip = test::ComputeMeanFlip(rgb_large, rgb_small,
                                       test::kTestWidth, test::kTestHeight);
    INFO("FLIP(large vs small sphere): " << flip);
    CHECK(flip > 0.02f);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_large);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_small);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Triangle light illumination (including two-sided)
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8G: TriangleLightIllumination",
          "[phase8g][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = test::BuildCornellBox();
    TriangleLight tri{};
    tri.v0 = {0.35f, 0.999f, 0.35f};
    tri.v1 = {0.65f, 0.999f, 0.35f};
    tri.v2 = {0.5f, 0.999f, 0.65f};
    tri.radiance = {20.0f, 15.0f, 5.0f};
    tri.two_sided = true;
    scene.AddTriangleLight(tri);

    auto result = test::RenderSceneMultiFrame(ctx, scene, mesh_data, 16, 4);

    auto* diffuse_raw = result.diffuse.data();
    auto* specular_raw = result.specular.data();

    test::WriteCombinedPNG("tests/output/phase8g_triangle_light.png",
                           diffuse_raw, specular_raw,
                           test::kTestWidth, test::kTestHeight);

    auto stats = test::AnalyzeRGBA16F(diffuse_raw, test::kPixelCount);
    CHECK(stats.nan_count == 0);
    CHECK(stats.inf_count == 0);
    CHECK(stats.nonzero_count > test::kPixelCount / 4);

    auto spec_stats = test::AnalyzeRGBA16F(specular_raw, test::kPixelCount);
    CHECK(spec_stats.nan_count == 0);
    CHECK(spec_stats.inf_count == 0);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: Quad light backward compatibility after PackedLight migration
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8G: QuadLightBackwardCompatibility",
          "[phase8g][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene);

    auto result = test::RenderSceneMultiFrame(ctx, scene, mesh_data, 16, 4);

    auto* diffuse_raw = result.diffuse.data();
    auto* specular_raw = result.specular.data();

    test::WriteCombinedPNG("tests/output/phase8g_quad_light_compat.png",
                           diffuse_raw, specular_raw,
                           test::kTestWidth, test::kTestHeight);

    auto stats = test::AnalyzeRGBA16F(diffuse_raw, test::kPixelCount);
    CHECK(stats.nan_count == 0);
    CHECK(stats.inf_count == 0);
    CHECK(stats.nonzero_count > test::kPixelCount / 4);
    CHECK(stats.has_color_variation);

    // Verify the scene produces warm illumination (the canonical light has
    // reddish-warm radiance {17, 12, 4}).
    CHECK(stats.sum_r / stats.valid_count > stats.sum_b / stats.valid_count);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: Mixed light convergence (quad + sphere + triangle)
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8G: MixedLightConvergence",
          "[phase8g][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // Build scene with all 3 light types
    auto [scene_lo, mesh_data_lo] = test::BuildCornellBox();

    // Quad light — canonical ceiling light
    test::AddCornellBoxLight(scene_lo);

    // Sphere light
    SphereLight sphere{};
    sphere.center = {0.2f, 0.6f, 0.5f};
    sphere.radius = 0.08f;
    sphere.radiance = {10.0f, 10.0f, 30.0f};
    scene_lo.AddSphereLight(sphere);

    // Triangle light
    TriangleLight tri{};
    tri.v0 = {0.7f, 0.999f, 0.4f};
    tri.v1 = {0.9f, 0.999f, 0.4f};
    tri.v2 = {0.8f, 0.999f, 0.6f};
    tri.radiance = {5.0f, 15.0f, 5.0f};
    tri.two_sided = false;
    scene_lo.AddTriangleLight(tri);

    // Low SPP render
    auto result_lo = test::RenderSceneMultiFrame(ctx, scene_lo, mesh_data_lo, 1, 4);

    // High SPP render — rebuild scene (same geometry and lights)
    auto [scene_hi, mesh_data_hi] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene_hi);
    scene_hi.AddSphereLight(sphere);
    scene_hi.AddTriangleLight(tri);

    auto result_hi = test::RenderSceneMultiFrame(ctx, scene_hi, mesh_data_hi, 16, 4);

    test::WriteCombinedPNG("tests/output/phase8g_mixed_lo.png",
                           result_lo.diffuse.data(), result_lo.specular.data(),
                           test::kTestWidth, test::kTestHeight);
    test::WriteCombinedPNG("tests/output/phase8g_mixed_hi.png",
                           result_hi.diffuse.data(), result_hi.specular.data(),
                           test::kTestWidth, test::kTestHeight);

    auto rgb_lo = test::TonemappedRGB(result_lo.diffuse.data(),
                                      result_lo.specular.data(),
                                      test::kPixelCount);
    auto rgb_hi = test::TonemappedRGB(result_hi.diffuse.data(),
                                      result_hi.specular.data(),
                                      test::kPixelCount);
    float flip = test::ComputeMeanFlip(rgb_hi, rgb_lo,
                                       test::kTestWidth, test::kTestHeight);
    INFO("FLIP(4spp vs 64spp mixed lights): " << flip);
    CHECK(flip < 0.25f);

    // No NaN/Inf in high-SPP result
    auto stats = test::AnalyzeRGBA16F(result_hi.diffuse.data(), test::kPixelCount);
    CHECK(stats.nan_count == 0);
    CHECK(stats.inf_count == 0);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_lo);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_hi);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: Degenerate light rejection (unit test — no GPU required)
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8G: DegenerateLightRejection",
          "[phase8g][scene][unit]") {
    Scene scene;

    // Sphere with zero radius — rejected
    SphereLight zero_radius{};
    zero_radius.center = {0, 1, 0};
    zero_radius.radius = 0.0f;
    zero_radius.radiance = {10, 10, 10};
    scene.AddSphereLight(zero_radius);
    CHECK(scene.SphereLights().empty());

    // Sphere with negative radius — rejected
    SphereLight neg_radius{};
    neg_radius.center = {0, 1, 0};
    neg_radius.radius = -0.5f;
    neg_radius.radiance = {10, 10, 10};
    scene.AddSphereLight(neg_radius);
    CHECK(scene.SphereLights().empty());

    // Valid sphere — accepted
    SphereLight valid_sphere{};
    valid_sphere.center = {0, 1, 0};
    valid_sphere.radius = 0.1f;
    valid_sphere.radiance = {10, 10, 10};
    scene.AddSphereLight(valid_sphere);
    CHECK(scene.SphereLights().size() == 1);

    // Triangle with zero area (collinear vertices) — rejected
    TriangleLight zero_area{};
    zero_area.v0 = {0, 0, 0};
    zero_area.v1 = {1, 0, 0};
    zero_area.v2 = {2, 0, 0};  // collinear
    zero_area.radiance = {10, 10, 10};
    scene.AddTriangleLight(zero_area);
    CHECK(scene.TriangleLights().empty());

    // Triangle with duplicate vertices — rejected
    TriangleLight degenerate{};
    degenerate.v0 = {0, 0, 0};
    degenerate.v1 = {0, 0, 0};
    degenerate.v2 = {0, 0, 0};
    degenerate.radiance = {10, 10, 10};
    scene.AddTriangleLight(degenerate);
    CHECK(scene.TriangleLights().empty());

    // Valid triangle — accepted
    TriangleLight valid_tri{};
    valid_tri.v0 = {0, 0, 0};
    valid_tri.v1 = {1, 0, 0};
    valid_tri.v2 = {0, 1, 0};
    valid_tri.radiance = {10, 10, 10};
    scene.AddTriangleLight(valid_tri);
    CHECK(scene.TriangleLights().size() == 1);
}
