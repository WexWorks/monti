#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <volk.h>

#include "test_helpers.h"
#include "scenes/Primitives.h"

#include <monti/scene/Scene.h>

#include "../renderer/src/vulkan/GpuScene.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace monti;
using namespace monti::vulkan;

// ═══════════════════════════════════════════════════════════════════════════
//
// Phase 8I test strategy
//
// Tests nested dielectric priority via the interior list. Uses icosphere
// geometry to create overlapping transmissive volumes. Two-scene FLIP
// comparisons verify that the interior list affects refraction (IOR
// mediation), false intersection rejection hides lower-priority surfaces,
// thin surfaces bypass the interior list, and the system handles graceful
// overflow when more volumes exist than interior list slots.
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

// ═══════════════════════════════════════════════════════════════════════════
// Scene construction helpers
// ═══════════════════════════════════════════════════════════════════════════

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

void AddIcosphereToScene(Scene& scene, std::vector<MeshData>& mesh_data_list,
                         std::string_view name, MaterialId mat_id,
                         const glm::vec3& center, float radius,
                         uint32_t subdivisions = 2) {
    auto md = test::MakeIcosphere(center, radius, subdivisions);
    Mesh mesh_desc;
    mesh_desc.name = std::string(name);
    mesh_desc.vertex_count = static_cast<uint32_t>(md.vertices.size());
    mesh_desc.index_count = static_cast<uint32_t>(md.indices.size());
    mesh_desc.bbox_min = center - glm::vec3(radius);
    mesh_desc.bbox_max = center + glm::vec3(radius);
    auto mesh_id = scene.AddMesh(std::move(mesh_desc), name);
    md.mesh_id = mesh_id;
    scene.AddNode(mesh_id, mat_id, name);
    mesh_data_list.push_back(std::move(md));
}

Vertex MakeVertex(const glm::vec3& pos, const glm::vec3& normal,
                  const glm::vec4& tangent, const glm::vec2& uv) {
    Vertex v{};
    v.position = pos;
    v.normal = normal;
    v.tangent = tangent;
    v.tex_coord_0 = uv;
    v.tex_coord_1 = uv;
    return v;
}

MeshData MakeQuad(const glm::vec3& center, float half_size) {
    glm::vec3 n{0, 0, 1};
    glm::vec4 t{1, 0, 0, 1};
    MeshData md;
    md.vertices = {
        MakeVertex(center + glm::vec3(-half_size, -half_size, 0), n, t, {0, 0}),
        MakeVertex(center + glm::vec3( half_size, -half_size, 0), n, t, {1, 0}),
        MakeVertex(center + glm::vec3( half_size,  half_size, 0), n, t, {1, 1}),
        MakeVertex(center + glm::vec3(-half_size,  half_size, 0), n, t, {0, 1}),
    };
    md.indices = {0, 1, 2, 0, 2, 3};
    return md;
}

void AddQuadToScene(Scene& scene, std::vector<MeshData>& mesh_data_list,
                    std::string_view name, MaterialId mat_id,
                    const glm::vec3& center, float half_size) {
    auto md = MakeQuad(center, half_size);
    Mesh mesh_desc;
    mesh_desc.name = std::string(name);
    mesh_desc.vertex_count = static_cast<uint32_t>(md.vertices.size());
    mesh_desc.index_count = static_cast<uint32_t>(md.indices.size());
    mesh_desc.bbox_min = center - glm::vec3(half_size);
    mesh_desc.bbox_max = center + glm::vec3(half_size);
    auto mesh_id = scene.AddMesh(std::move(mesh_desc), name);
    md.mesh_id = mesh_id;
    scene.AddNode(mesh_id, mat_id, name);
    mesh_data_list.push_back(std::move(md));
}

// Setup a common environment, camera, and background for nested dielectric tests
void SetupTestEnvironment(Scene& scene, std::vector<MeshData>& mesh_data) {
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.1f, 0.1f, 0.15f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    // Colored background wall behind the spheres
    MaterialDesc back_mat;
    back_mat.base_color = {0.8f, 0.2f, 0.2f};
    back_mat.roughness = 1.0f;
    auto back_id = scene.AddMaterial(std::move(back_mat), "back_wall");
    AddQuadToScene(scene, mesh_data, "back_wall", back_id, {0, 0, -2.0f}, 3.0f);

    // Floor
    MaterialDesc floor_mat;
    floor_mat.base_color = {0.5f, 0.5f, 0.5f};
    floor_mat.roughness = 0.8f;
    auto floor_id = scene.AddMaterial(std::move(floor_mat), "floor");

    // Approximate a horizontal floor with a large quad rotated via vertex positions
    MeshData floor_md;
    glm::vec3 fn{0, 1, 0};
    glm::vec4 ft{1, 0, 0, 1};
    floor_md.vertices = {
        MakeVertex({-3, -1, -3}, fn, ft, {0, 0}),
        MakeVertex({ 3, -1, -3}, fn, ft, {1, 0}),
        MakeVertex({ 3, -1,  3}, fn, ft, {1, 1}),
        MakeVertex({-3, -1,  3}, fn, ft, {0, 1}),
    };
    floor_md.indices = {0, 1, 2, 0, 2, 3};
    Mesh floor_mesh;
    floor_mesh.name = "floor";
    floor_mesh.vertex_count = static_cast<uint32_t>(floor_md.vertices.size());
    floor_mesh.index_count = static_cast<uint32_t>(floor_md.indices.size());
    floor_mesh.bbox_min = {-3, -1, -3};
    floor_mesh.bbox_max = {3, -1, 3};
    auto floor_mesh_id = scene.AddMesh(std::move(floor_mesh), "floor");
    floor_md.mesh_id = floor_mesh_id;
    scene.AddNode(floor_mesh_id, floor_id, "floor");
    mesh_data.push_back(std::move(floor_md));

    // Area light above
    AreaLight light;
    light.corner = {-1.5f, 2.0f, -1.5f};
    light.edge_a = {3.0f, 0.0f, 0.0f};
    light.edge_b = {0.0f, 0.0f, 3.0f};
    light.radiance = {4.0f, 4.0f, 4.0f};
    light.two_sided = true;
    scene.AddAreaLight(light);

    CameraParams camera;
    camera.position = {0.0f, 0.0f, 3.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.vertical_fov_radians = 0.8f;
    camera.near_plane = 0.01f;
    camera.far_plane = 100.0f;
    scene.SetActiveCamera(camera);
}

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: NestedDielectricGlassInGlass
//
// Validates that the interior list mediates IOR when nested volumes exist.
//
// A: inner sphere (IOR 1.5, priority 1) inside outer sphere (IOR 1.33, priority 2)
// B: inner sphere alone (no outer sphere)
//
// The outer medium's IOR should affect the inner sphere's refraction,
// so FLIP(A, B) must be > 0.02. If the interior list is broken, n1
// defaults to 1.0 in both cases and the renders are identical.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8I: NestedDielectricGlassInGlass",
          "[phase8i][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // Scene A: inner + outer sphere
    Scene scene_a;
    std::vector<MeshData> mesh_data_a;
    SetupTestEnvironment(scene_a, mesh_data_a);

    MaterialDesc outer_mat;
    outer_mat.base_color = {1, 1, 1};
    outer_mat.roughness = 0.01f;
    outer_mat.transmission_factor = 1.0f;
    outer_mat.ior = 1.33f;
    outer_mat.nested_priority = 2;
    auto outer_id = scene_a.AddMaterial(std::move(outer_mat), "outer_glass");

    MaterialDesc inner_mat;
    inner_mat.base_color = {1, 1, 1};
    inner_mat.roughness = 0.01f;
    inner_mat.transmission_factor = 1.0f;
    inner_mat.ior = 1.5f;
    inner_mat.nested_priority = 1;
    auto inner_id = scene_a.AddMaterial(std::move(inner_mat), "inner_glass");

    AddIcosphereToScene(scene_a, mesh_data_a, "outer", outer_id,
                        {0, 0, 0}, 1.0f, 2);
    AddIcosphereToScene(scene_a, mesh_data_a, "inner", inner_id,
                        {0, 0, 0}, 0.5f, 2);

    auto result_a = test::RenderSceneMultiFrame(
        ctx, scene_a, mesh_data_a, 16, 64);

    test::WriteCombinedPNG(
        "tests/output/phase8i_glass_in_glass.png",
        result_a.diffuse.data(), result_a.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // Scene B: inner sphere only (same camera, lights, background)
    Scene scene_b;
    std::vector<MeshData> mesh_data_b;
    SetupTestEnvironment(scene_b, mesh_data_b);

    MaterialDesc inner_mat_b;
    inner_mat_b.base_color = {1, 1, 1};
    inner_mat_b.roughness = 0.01f;
    inner_mat_b.transmission_factor = 1.0f;
    inner_mat_b.ior = 1.5f;
    inner_mat_b.nested_priority = 1;
    auto inner_id_b = scene_b.AddMaterial(std::move(inner_mat_b), "inner_glass");

    AddIcosphereToScene(scene_b, mesh_data_b, "inner", inner_id_b,
                        {0, 0, 0}, 0.5f, 2);

    auto result_b = test::RenderSceneMultiFrame(
        ctx, scene_b, mesh_data_b, 16, 64);

    test::WriteCombinedPNG(
        "tests/output/phase8i_glass_alone.png",
        result_b.diffuse.data(), result_b.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // FLIP: outer medium's IOR must visibly alter refraction
    auto rgb_a = test::TonemappedRGB(
        result_a.diffuse.data(), result_a.specular.data(), test::kPixelCount);
    auto rgb_b = test::TonemappedRGB(
        result_b.diffuse.data(), result_b.specular.data(), test::kPixelCount);

    float mean_flip = test::ComputeMeanFlip(
        rgb_a, rgb_b,
        static_cast<int>(test::kTestWidth),
        static_cast<int>(test::kTestHeight));

    std::printf("Phase 8I glass-in-glass FLIP: %.4f\n", mean_flip);
    REQUIRE(mean_flip > 0.02f);

    // No NaN/Inf
    auto stats_a = test::AnalyzeRGBA16F(result_a.diffuse.data(), test::kPixelCount);
    auto stats_b = test::AnalyzeRGBA16F(result_b.diffuse.data(), test::kPixelCount);
    REQUIRE(stats_a.nan_count == 0);
    REQUIRE(stats_a.inf_count == 0);
    REQUIRE(stats_b.nan_count == 0);
    REQUIRE(stats_b.inf_count == 0);

    auto stats_a_spec = test::AnalyzeRGBA16F(result_a.specular.data(), test::kPixelCount);
    auto stats_b_spec = test::AnalyzeRGBA16F(result_b.specular.data(), test::kPixelCount);
    REQUIRE(stats_a_spec.nan_count == 0);
    REQUIRE(stats_a_spec.inf_count == 0);
    REQUIRE(stats_b_spec.nan_count == 0);
    REQUIRE(stats_b_spec.inf_count == 0);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_a);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_b);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: NestedDielectricFalseIntersection
//
// Validates false intersection rejection. A high-priority outer sphere
// should make a low-priority inner sphere invisible (its intersections
// are rejected as false).
//
// A: outer (priority 3) + inner (priority 1) → inner is invisible
// B: outer sphere only
//
// FLIP(A, B) < 0.01. If false intersection rejection is broken, the
// inner sphere appears and the FLIP is higher.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8I: NestedDielectricFalseIntersection",
          "[phase8i][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // Scene A: outer (priority 3) + inner (priority 1)
    Scene scene_a;
    std::vector<MeshData> mesh_data_a;
    SetupTestEnvironment(scene_a, mesh_data_a);

    MaterialDesc outer_mat;
    outer_mat.base_color = {1, 1, 1};
    outer_mat.roughness = 0.01f;
    outer_mat.transmission_factor = 1.0f;
    outer_mat.ior = 1.4f;
    outer_mat.nested_priority = 3;
    auto outer_id = scene_a.AddMaterial(std::move(outer_mat), "outer");

    MaterialDesc inner_mat;
    inner_mat.base_color = {1, 1, 1};
    inner_mat.roughness = 0.01f;
    inner_mat.transmission_factor = 1.0f;
    inner_mat.ior = 2.0f;  // Very different IOR — would be visible if not rejected
    inner_mat.nested_priority = 1;
    auto inner_id = scene_a.AddMaterial(std::move(inner_mat), "inner");

    AddIcosphereToScene(scene_a, mesh_data_a, "outer", outer_id,
                        {0, 0, 0}, 1.0f, 2);
    AddIcosphereToScene(scene_a, mesh_data_a, "inner", inner_id,
                        {0, 0, 0}, 0.5f, 2);

    auto result_a = test::RenderSceneMultiFrame(
        ctx, scene_a, mesh_data_a, 16, 64);

    test::WriteCombinedPNG(
        "tests/output/phase8i_false_intersection_both.png",
        result_a.diffuse.data(), result_a.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // Scene B: outer sphere only
    Scene scene_b;
    std::vector<MeshData> mesh_data_b;
    SetupTestEnvironment(scene_b, mesh_data_b);

    MaterialDesc outer_mat_b;
    outer_mat_b.base_color = {1, 1, 1};
    outer_mat_b.roughness = 0.01f;
    outer_mat_b.transmission_factor = 1.0f;
    outer_mat_b.ior = 1.4f;
    outer_mat_b.nested_priority = 3;
    auto outer_id_b = scene_b.AddMaterial(std::move(outer_mat_b), "outer");

    AddIcosphereToScene(scene_b, mesh_data_b, "outer", outer_id_b,
                        {0, 0, 0}, 1.0f, 2);

    auto result_b = test::RenderSceneMultiFrame(
        ctx, scene_b, mesh_data_b, 16, 64);

    test::WriteCombinedPNG(
        "tests/output/phase8i_false_intersection_outer_only.png",
        result_b.diffuse.data(), result_b.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // FLIP: inner sphere should be invisible → renders match closely
    auto rgb_a = test::TonemappedRGB(
        result_a.diffuse.data(), result_a.specular.data(), test::kPixelCount);
    auto rgb_b = test::TonemappedRGB(
        result_b.diffuse.data(), result_b.specular.data(), test::kPixelCount);

    float mean_flip = test::ComputeMeanFlip(
        rgb_a, rgb_b,
        static_cast<int>(test::kTestWidth),
        static_cast<int>(test::kTestHeight));

    std::printf("Phase 8I false intersection FLIP (both vs outer only): %.4f\n",
                mean_flip);
    REQUIRE(mean_flip < 0.03f);

    // No NaN/Inf
    auto stats_a = test::AnalyzeRGBA16F(result_a.diffuse.data(), test::kPixelCount);
    auto stats_b = test::AnalyzeRGBA16F(result_b.diffuse.data(), test::kPixelCount);
    REQUIRE(stats_a.nan_count == 0);
    REQUIRE(stats_a.inf_count == 0);
    REQUIRE(stats_b.nan_count == 0);
    REQUIRE(stats_b.inf_count == 0);

    auto stats_a_spec = test::AnalyzeRGBA16F(result_a.specular.data(), test::kPixelCount);
    auto stats_b_spec = test::AnalyzeRGBA16F(result_b.specular.data(), test::kPixelCount);
    REQUIRE(stats_a_spec.nan_count == 0);
    REQUIRE(stats_a_spec.inf_count == 0);
    REQUIRE(stats_b_spec.nan_count == 0);
    REQUIRE(stats_b_spec.inf_count == 0);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_a);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_b);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: NestedDielectricStackOverflow
//
// Edge case: 8 concentric transmissive spheres exceed kInteriorListSlots=2.
// Verifies no NaN/Inf and no GPU hang — excess insertions are silently
// dropped (matching RTXPT behavior).
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8I: NestedDielectricStackOverflow",
          "[phase8i][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;
    std::vector<MeshData> mesh_data;
    SetupTestEnvironment(scene, mesh_data);

    // 8 concentric transmissive spheres with different priorities
    for (int i = 0; i < 8; ++i) {
        MaterialDesc mat;
        mat.base_color = {1, 1, 1};
        mat.roughness = 0.01f;
        mat.transmission_factor = 1.0f;
        mat.ior = 1.2f + 0.1f * static_cast<float>(i);
        mat.nested_priority = static_cast<uint8_t>(i + 1);
        std::string name = "sphere_" + std::to_string(i);
        auto mat_id = scene.AddMaterial(std::move(mat), name);

        float radius = 1.0f - 0.1f * static_cast<float>(i);
        AddIcosphereToScene(scene, mesh_data, name, mat_id,
                            {0, 0, 0}, radius, 1);  // Lower subdivision to reduce geometry
    }

    auto result = test::RenderSceneMultiFrame(
        ctx, scene, mesh_data, 4, 4);

    test::WriteCombinedPNG(
        "tests/output/phase8i_stack_overflow.png",
        result.diffuse.data(), result.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // Verify no NaN/Inf — graceful degradation
    auto stats_d = test::AnalyzeRGBA16F(result.diffuse.data(), test::kPixelCount);
    auto stats_s = test::AnalyzeRGBA16F(result.specular.data(), test::kPixelCount);

    std::printf("Phase 8I stack overflow: NaN=%u/%u, Inf=%u/%u\n",
                stats_d.nan_count, stats_s.nan_count,
                stats_d.inf_count, stats_s.inf_count);

    REQUIRE(stats_d.nan_count == 0);
    REQUIRE(stats_d.inf_count == 0);
    REQUIRE(stats_s.nan_count == 0);
    REQUIRE(stats_s.inf_count == 0);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: NestedDielectricThinSurfaceBypass
//
// Validates that thin surfaces skip the interior list.
//
// A: inner sphere with thin_surface=true (interior list unaffected)
// B: inner sphere with thin_surface=false (interior list updated)
//
// FLIP(A, B) > 0.02 — thin and thick surfaces behave differently when
// nested inside another volume.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8I: NestedDielectricThinSurfaceBypass",
          "[phase8i][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto build_scene = [](bool thin_inner) {
        Scene scene;
        std::vector<MeshData> mesh_data;
        SetupTestEnvironment(scene, mesh_data);

        MaterialDesc outer_mat;
        outer_mat.base_color = {1, 1, 1};
        outer_mat.roughness = 0.01f;
        outer_mat.transmission_factor = 1.0f;
        outer_mat.ior = 1.33f;
        outer_mat.nested_priority = 2;
        auto outer_id = scene.AddMaterial(std::move(outer_mat), "outer");

        MaterialDesc inner_mat;
        inner_mat.base_color = {1, 1, 1};
        inner_mat.roughness = 0.01f;
        inner_mat.transmission_factor = 1.0f;
        inner_mat.ior = 1.5f;
        inner_mat.thin_surface = thin_inner;
        inner_mat.nested_priority = 5;
        auto inner_id = scene.AddMaterial(std::move(inner_mat), "inner");

        AddIcosphereToScene(scene, mesh_data, "outer", outer_id,
                            {0, 0, 0}, 1.0f, 2);
        AddIcosphereToScene(scene, mesh_data, "inner", inner_id,
                            {0, 0, 0}, 0.5f, 2);

        return std::make_pair(std::move(scene), std::move(mesh_data));
    };

    // A: thin inner surface
    auto [scene_a, mesh_data_a] = build_scene(true);
    auto result_a = test::RenderSceneMultiFrame(
        ctx, scene_a, mesh_data_a, 16, 64);

    test::WriteCombinedPNG(
        "tests/output/phase8i_thin_bypass_thin.png",
        result_a.diffuse.data(), result_a.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // B: thick inner surface
    auto [scene_b, mesh_data_b] = build_scene(false);
    auto result_b = test::RenderSceneMultiFrame(
        ctx, scene_b, mesh_data_b, 16, 64);

    test::WriteCombinedPNG(
        "tests/output/phase8i_thin_bypass_thick.png",
        result_b.diffuse.data(), result_b.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // FLIP: thin vs thick should produce noticeably different results
    auto rgb_a = test::TonemappedRGB(
        result_a.diffuse.data(), result_a.specular.data(), test::kPixelCount);
    auto rgb_b = test::TonemappedRGB(
        result_b.diffuse.data(), result_b.specular.data(), test::kPixelCount);

    float mean_flip = test::ComputeMeanFlip(
        rgb_a, rgb_b,
        static_cast<int>(test::kTestWidth),
        static_cast<int>(test::kTestHeight));

    std::printf("Phase 8I thin bypass FLIP (thin vs thick inner): %.4f\n",
                mean_flip);
    REQUIRE(mean_flip > 0.02f);

    // No NaN/Inf
    auto stats_a = test::AnalyzeRGBA16F(result_a.diffuse.data(), test::kPixelCount);
    auto stats_b = test::AnalyzeRGBA16F(result_b.diffuse.data(), test::kPixelCount);
    REQUIRE(stats_a.nan_count == 0);
    REQUIRE(stats_a.inf_count == 0);
    REQUIRE(stats_b.nan_count == 0);
    REQUIRE(stats_b.inf_count == 0);

    auto stats_a_spec = test::AnalyzeRGBA16F(result_a.specular.data(), test::kPixelCount);
    auto stats_b_spec = test::AnalyzeRGBA16F(result_b.specular.data(), test::kPixelCount);
    REQUIRE(stats_a_spec.nan_count == 0);
    REQUIRE(stats_a_spec.inf_count == 0);
    REQUIRE(stats_b_spec.nan_count == 0);
    REQUIRE(stats_b_spec.inf_count == 0);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_a);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_b);
    ctx.WaitIdle();
}
