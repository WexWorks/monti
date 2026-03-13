#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <volk.h>

#include "test_helpers.h"

#include <monti/scene/Scene.h>

#include "../renderer/src/vulkan/GpuScene.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace monti;
using namespace monti::vulkan;

// ═══════════════════════════════════════════════════════════════════════════
//
// Phase 8H test strategy
//
// Uses multi-frame accumulation (RenderSceneMultiFrame) for proper
// sub-pixel jitter and noise decorrelation. Each frame gets a different
// Halton jitter and blue noise temporal hash, producing anti-aliased
// reference images that single-frame rendering cannot achieve.
//
// A/B FLIP tests are reinforced with channel-analysis assertions that
// verify *why* images differ (color dominance, energy ratios), not just
// *that* they differ.
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

MeshData& AddQuadToScene(Scene& scene, std::vector<MeshData>& mesh_data_list,
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
    return mesh_data_list.back();
}

// ═══════════════════════════════════════════════════════════════════════════
// Scene builders
// ═══════════════════════════════════════════════════════════════════════════

// Backlit-leaf scene: thin quad at z=0 with diffuse transmission, lit from
// behind by an emissive quad at z=-1. Camera at z=+2 looking at origin.
struct BacklitScene {
    Scene scene;
    std::vector<MeshData> mesh_data;

    void Build(float dt_factor, glm::vec3 dt_color,
               glm::vec3 base_color = {1, 1, 1},
               float transmission_factor = 0.0f,
               bool thin_surface = true) {
        auto env_tex_id = scene.AddTexture(MakeEnvMap(0.0f, 0.0f, 0.0f), "env_map");
        EnvironmentLight env{};
        env.hdr_lat_long = env_tex_id;
        env.intensity = 0.0f;
        scene.SetEnvironmentLight(env);

        MaterialDesc mat;
        mat.base_color = base_color;
        mat.roughness = 1.0f;
        mat.metallic = 0.0f;
        mat.diffuse_transmission_factor = dt_factor;
        mat.diffuse_transmission_color = dt_color;
        mat.thin_surface = thin_surface;
        mat.transmission_factor = transmission_factor;
        mat.double_sided = true;
        auto mat_id = scene.AddMaterial(std::move(mat), "leaf");

        MaterialDesc light_mat;
        light_mat.base_color = {0, 0, 0};
        light_mat.emissive_factor = {5.0f, 5.0f, 5.0f};
        light_mat.emissive_strength = 1.0f;
        auto light_mat_id = scene.AddMaterial(std::move(light_mat), "light");

        AddQuadToScene(scene, mesh_data, "leaf", mat_id, {0, 0, 0}, 1.0f);
        AddQuadToScene(scene, mesh_data, "light", light_mat_id, {0, 0, -1.0f}, 1.0f);

        AreaLight light;
        light.corner = {-1.0f, -1.0f, -1.0f};
        light.edge_a = {2.0f, 0.0f, 0.0f};
        light.edge_b = {0.0f, 2.0f, 0.0f};
        light.radiance = {5.0f, 5.0f, 5.0f};
        light.two_sided = true;
        scene.AddAreaLight(light);

        CameraParams camera;
        camera.position = {0.0f, 0.0f, 2.0f};
        camera.target = {0.0f, 0.0f, 0.0f};
        camera.up = {0.0f, 1.0f, 0.0f};
        camera.vertical_fov_radians = 1.0f;
        camera.near_plane = 0.01f;
        camera.far_plane = 100.0f;
        scene.SetActiveCamera(camera);
    }
};

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: DiffuseTransmissionBacklitLeaf
//
// Validates that diffuse transmission produces visible backlit illumination.
//
// A: dt_factor=0.8, green dt_color (feature under test)
// B: dt_factor=0.0 (opaque reference — no DT)
//
// FLIP verifies perceptual difference. Channel analysis verifies the DT
// path transmits through the green dt_color: avg_g must dominate in A's
// diffuse channel. Also verifies A is brighter than B overall (DT adds
// energy from the backlight that the opaque surface blocks).
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8H: DiffuseTransmissionBacklitLeaf",
          "[phase8h][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // A: diffuse transmission enabled (16 frames x 64 SPP = 1024 total samples)
    BacklitScene scene_a;
    scene_a.Build(0.8f, {0.2f, 0.8f, 0.1f});
    auto result_a = test::RenderSceneMultiFrame(
        ctx, scene_a.scene, scene_a.mesh_data, 16, 64);

    test::WriteCombinedPNG(
        "tests/output/phase8h_backlit_leaf_dt.png",
        result_a.diffuse.data(), result_a.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // B: opaque (no diffuse transmission)
    BacklitScene scene_b;
    scene_b.Build(0.0f, {0.2f, 0.8f, 0.1f});
    auto result_b = test::RenderSceneMultiFrame(
        ctx, scene_b.scene, scene_b.mesh_data, 16, 64);

    test::WriteCombinedPNG(
        "tests/output/phase8h_backlit_leaf_opaque.png",
        result_b.diffuse.data(), result_b.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // FLIP comparison
    auto rgb_a = test::TonemappedRGB(
        result_a.diffuse.data(), result_a.specular.data(), test::kPixelCount);
    auto rgb_b = test::TonemappedRGB(
        result_b.diffuse.data(), result_b.specular.data(), test::kPixelCount);

    float mean_flip = test::ComputeMeanFlip(
        rgb_a, rgb_b,
        static_cast<int>(test::kTestWidth),
        static_cast<int>(test::kTestHeight));

    std::printf("Phase 8H backlit leaf FLIP (dt=0.8 vs dt=0.0): %.4f\n", mean_flip);
    REQUIRE(mean_flip > 0.05f);

    // No NaN/Inf
    auto stats_a = test::AnalyzeRGBA16F(result_a.diffuse.data(), test::kPixelCount);
    auto stats_b = test::AnalyzeRGBA16F(result_b.diffuse.data(), test::kPixelCount);
    REQUIRE(stats_a.nan_count == 0);
    REQUIRE(stats_a.inf_count == 0);
    REQUIRE(stats_b.nan_count == 0);
    REQUIRE(stats_b.inf_count == 0);

    // Channel analysis: DT with green dt_color makes green dominant in A
    REQUIRE(stats_a.valid_count > 0);
    double a_avg_r = stats_a.sum_r / stats_a.valid_count;
    double a_avg_g = stats_a.sum_g / stats_a.valid_count;
    double a_avg_b = stats_a.sum_b / stats_a.valid_count;
    std::printf("  DT render avg channels: R=%.4f G=%.4f B=%.4f\n",
                a_avg_r, a_avg_g, a_avg_b);
    REQUIRE(a_avg_g > a_avg_r);
    REQUIRE(a_avg_g > a_avg_b);

    // A must produce non-zero illumination from transmission
    REQUIRE(stats_a.nonzero_count > 100);

    // A (translucent) should have more total diffuse energy than B (opaque)
    double a_energy = stats_a.sum_r + stats_a.sum_g + stats_a.sum_b;
    double b_energy = stats_b.sum_r + stats_b.sum_g + stats_b.sum_b;
    std::printf("  DT total diffuse energy: %.2f, opaque: %.2f\n", a_energy, b_energy);
    REQUIRE(a_energy > b_energy);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_a);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_b);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: ThinSurfaceNoRefraction
//
// Validates that thin_surface=true bypasses IOR refraction bending.
//
// A: thin_surface=true  (straight-through transmission)
// B: thin_surface=false (IOR=1.5 refraction bends geometry)
//
// FLIP verifies the two renders differ perceptually.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8H: ThinSurfaceNoRefraction",
          "[phase8h][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto build_glass_scene = [](bool thin) {
        BacklitScene s;

        auto env_tex_id = s.scene.AddTexture(MakeEnvMap(0.0f, 0.0f, 0.0f), "env_map");
        EnvironmentLight env{};
        env.hdr_lat_long = env_tex_id;
        env.intensity = 0.0f;
        s.scene.SetEnvironmentLight(env);

        MaterialDesc glass;
        glass.base_color = {1, 1, 1};
        glass.roughness = 0.01f;
        glass.metallic = 0.0f;
        glass.transmission_factor = 1.0f;
        glass.ior = 1.5f;
        glass.thin_surface = thin;
        glass.double_sided = true;
        auto glass_id = s.scene.AddMaterial(std::move(glass), "glass");

        MaterialDesc red;
        red.base_color = {0.9f, 0.1f, 0.1f};
        red.roughness = 1.0f;
        red.metallic = 0.0f;
        auto red_id = s.scene.AddMaterial(std::move(red), "red_wall");

        MaterialDesc green;
        green.base_color = {0.1f, 0.9f, 0.1f};
        green.roughness = 1.0f;
        green.metallic = 0.0f;
        auto green_id = s.scene.AddMaterial(std::move(green), "green_wall");

        AddQuadToScene(s.scene, s.mesh_data, "glass", glass_id, {0, 0, 0}, 1.0f);
        AddQuadToScene(s.scene, s.mesh_data, "red_wall", red_id, {0, 0, -1.5f}, 2.0f);
        AddQuadToScene(s.scene, s.mesh_data, "green_wall", green_id, {2.0f, 0, -0.75f}, 1.0f);

        MaterialDesc light_mat;
        light_mat.base_color = {0, 0, 0};
        light_mat.emissive_factor = {3.0f, 3.0f, 3.0f};
        light_mat.emissive_strength = 1.0f;
        auto light_id = s.scene.AddMaterial(std::move(light_mat), "light");
        AddQuadToScene(s.scene, s.mesh_data, "light", light_id, {0, 2.0f, -0.75f}, 1.0f);

        AreaLight light;
        light.corner = {-1.0f, 2.0f, -1.75f};
        light.edge_a = {2.0f, 0.0f, 0.0f};
        light.edge_b = {0.0f, 0.0f, 2.0f};
        light.radiance = {3.0f, 3.0f, 3.0f};
        light.two_sided = true;
        s.scene.AddAreaLight(light);

        CameraParams camera;
        camera.position = {0.0f, 0.0f, 2.0f};
        camera.target = {0.0f, 0.0f, 0.0f};
        camera.up = {0.0f, 1.0f, 0.0f};
        camera.vertical_fov_radians = 1.0f;
        camera.near_plane = 0.01f;
        camera.far_plane = 100.0f;
        s.scene.SetActiveCamera(camera);

        return s;
    };

    // A: thin surface (no refraction) — 16 frames x 64 SPP
    auto scene_a = build_glass_scene(true);
    auto result_a = test::RenderSceneMultiFrame(
        ctx, scene_a.scene, scene_a.mesh_data, 16, 64);

    test::WriteCombinedPNG(
        "tests/output/phase8h_thin_surface.png",
        result_a.diffuse.data(), result_a.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // B: thick surface (IOR refraction) — 16 frames x 64 SPP
    auto scene_b = build_glass_scene(false);
    auto result_b = test::RenderSceneMultiFrame(
        ctx, scene_b.scene, scene_b.mesh_data, 16, 64);

    test::WriteCombinedPNG(
        "tests/output/phase8h_thick_surface.png",
        result_b.diffuse.data(), result_b.specular.data(),
        test::kTestWidth, test::kTestHeight);

    auto rgb_a = test::TonemappedRGB(
        result_a.diffuse.data(), result_a.specular.data(), test::kPixelCount);
    auto rgb_b = test::TonemappedRGB(
        result_b.diffuse.data(), result_b.specular.data(), test::kPixelCount);

    float mean_flip = test::ComputeMeanFlip(
        rgb_a, rgb_b,
        static_cast<int>(test::kTestWidth),
        static_cast<int>(test::kTestHeight));

    std::printf("Phase 8H thin vs thick surface FLIP: %.4f\n", mean_flip);
    REQUIRE(mean_flip > 0.02f);

    // No NaN/Inf in either render
    auto stats_a = test::AnalyzeRGBA16F(result_a.diffuse.data(), test::kPixelCount);
    auto stats_b = test::AnalyzeRGBA16F(result_b.diffuse.data(), test::kPixelCount);
    REQUIRE(stats_a.nan_count == 0);
    REQUIRE(stats_a.inf_count == 0);
    REQUIRE(stats_b.nan_count == 0);
    REQUIRE(stats_b.inf_count == 0);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_a);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_b);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: DiffuseTransmissionColorTinting
//
// Validates that dt_color correctly tints transmitted light.
//
// Two renders with different dt_colors, same dt_factor:
//   A: dt_color={1,0,0} (red) — transmitted light is red
//   B: dt_color={0,0,1} (blue) — transmitted light is blue
//
// Channel analysis on both: A has avg_r >> avg_b, B has avg_b >> avg_r.
// Cross-check: A's avg_r > B's avg_r, and B's avg_b > A's avg_b.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8H: DiffuseTransmissionColorTinting",
          "[phase8h][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // A: red dt_color — 16 frames x 64 SPP
    BacklitScene scene_r;
    scene_r.Build(0.8f, {1.0f, 0.0f, 0.0f});
    auto result_r = test::RenderSceneMultiFrame(
        ctx, scene_r.scene, scene_r.mesh_data, 16, 64);

    test::WriteCombinedPNG(
        "tests/output/phase8h_red_tint.png",
        result_r.diffuse.data(), result_r.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // B: blue dt_color — 16 frames x 64 SPP
    BacklitScene scene_b;
    scene_b.Build(0.8f, {0.0f, 0.0f, 1.0f});
    auto result_b = test::RenderSceneMultiFrame(
        ctx, scene_b.scene, scene_b.mesh_data, 16, 64);

    test::WriteCombinedPNG(
        "tests/output/phase8h_blue_tint.png",
        result_b.diffuse.data(), result_b.specular.data(),
        test::kTestWidth, test::kTestHeight);

    auto stats_r = test::AnalyzeRGBA16F(result_r.diffuse.data(), test::kPixelCount);
    auto stats_b = test::AnalyzeRGBA16F(result_b.diffuse.data(), test::kPixelCount);

    // No NaN/Inf
    REQUIRE(stats_r.nan_count == 0);
    REQUIRE(stats_r.inf_count == 0);
    REQUIRE(stats_b.nan_count == 0);
    REQUIRE(stats_b.inf_count == 0);

    // Both renders must produce non-zero illumination
    REQUIRE(stats_r.nonzero_count > 100);
    REQUIRE(stats_b.nonzero_count > 100);
    REQUIRE(stats_r.valid_count > 0);
    REQUIRE(stats_b.valid_count > 0);

    double r_avg_r = stats_r.sum_r / stats_r.valid_count;
    double r_avg_g = stats_r.sum_g / stats_r.valid_count;
    double r_avg_b = stats_r.sum_b / stats_r.valid_count;

    double b_avg_r = stats_b.sum_r / stats_b.valid_count;
    double b_avg_g = stats_b.sum_g / stats_b.valid_count;
    double b_avg_b = stats_b.sum_b / stats_b.valid_count;

    std::printf("Phase 8H red-tint diffuse avg: R=%.4f G=%.4f B=%.4f\n",
                r_avg_r, r_avg_g, r_avg_b);
    std::printf("Phase 8H blue-tint diffuse avg: R=%.4f G=%.4f B=%.4f\n",
                b_avg_r, b_avg_g, b_avg_b);

    // Red-tinted render: red dominant
    REQUIRE(r_avg_r > r_avg_g * 1.5);
    REQUIRE(r_avg_r > r_avg_b * 1.5);

    // Blue-tinted render: blue dominant
    REQUIRE(b_avg_b > b_avg_r * 1.5);
    REQUIRE(b_avg_b > b_avg_g * 1.5);

    // Cross-check: red render's R > blue render's R
    REQUIRE(r_avg_r > b_avg_r * 1.5);
    // Cross-check: blue render's B > red render's B
    REQUIRE(b_avg_b > r_avg_b * 1.5);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_r);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_b);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: DiffuseTransmissionConvergence
//
// Validates MIS integration converges with multi-frame accumulation.
//
// Low quality: 2 frames x 2 SPP = 4 total samples
// High quality: 16 frames x 64 SPP = 1024 total samples
//
// Multi-frame rendering exercises the Halton jitter sequence and blue noise
// temporal decorrelation, producing higher-quality reference images than
// single-frame high-SPP.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8H: DiffuseTransmissionConvergence",
          "[phase8h][renderer][vulkan][integration][flip]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // Low quality render
    BacklitScene scene_low;
    scene_low.Build(0.8f, {0.2f, 0.8f, 0.1f});
    auto result_low = test::RenderSceneMultiFrame(
        ctx, scene_low.scene, scene_low.mesh_data, 2, 2);

    // High quality reference
    BacklitScene scene_high;
    scene_high.Build(0.8f, {0.2f, 0.8f, 0.1f});
    auto result_high = test::RenderSceneMultiFrame(
        ctx, scene_high.scene, scene_high.mesh_data, 16, 64);

    test::WriteCombinedPNG(
        "tests/output/phase8h_convergence_high.png",
        result_high.diffuse.data(), result_high.specular.data(),
        test::kTestWidth, test::kTestHeight);

    auto rgb_low = test::TonemappedRGB(
        result_low.diffuse.data(), result_low.specular.data(), test::kPixelCount);
    auto rgb_high = test::TonemappedRGB(
        result_high.diffuse.data(), result_high.specular.data(), test::kPixelCount);

    float convergence_flip = test::ComputeMeanFlip(
        rgb_low, rgb_high,
        static_cast<int>(test::kTestWidth),
        static_cast<int>(test::kTestHeight));

    std::printf("Phase 8H convergence FLIP (4 vs 1024 total samples): %.4f\n",
                convergence_flip);

    // Low-vs-high should differ mainly by MC noise, not structural bias
    REQUIRE(convergence_flip < 0.5f);

    // No NaN/Inf at low quality (most numerically stressed)
    auto stats_low = test::AnalyzeRGBA16F(result_low.diffuse.data(), test::kPixelCount);
    REQUIRE(stats_low.nan_count == 0);
    REQUIRE(stats_low.inf_count == 0);

    // Both quality levels should agree on dominant channel (green)
    auto stats_high = test::AnalyzeRGBA16F(result_high.diffuse.data(), test::kPixelCount);
    REQUIRE(stats_low.valid_count > 0);
    REQUIRE(stats_high.valid_count > 0);

    double low_avg_r = stats_low.sum_r / stats_low.valid_count;
    double low_avg_g = stats_low.sum_g / stats_low.valid_count;
    double low_avg_b = stats_low.sum_b / stats_low.valid_count;
    double high_avg_r = stats_high.sum_r / stats_high.valid_count;
    double high_avg_g = stats_high.sum_g / stats_high.valid_count;
    double high_avg_b = stats_high.sum_b / stats_high.valid_count;

    std::printf("  Low quality avg: R=%.4f G=%.4f B=%.4f\n",
                low_avg_r, low_avg_g, low_avg_b);
    std::printf("  High quality avg: R=%.4f G=%.4f B=%.4f\n",
                high_avg_r, high_avg_g, high_avg_b);

    // Green dominant at both quality levels
    REQUIRE(low_avg_g > low_avg_r);
    REQUIRE(low_avg_g > low_avg_b);
    REQUIRE(high_avg_g > high_avg_r);
    REQUIRE(high_avg_g > high_avg_b);

    // Per-channel means should converge within 50%
    REQUIRE(low_avg_g > high_avg_g * 0.5);
    REQUIRE(low_avg_g < high_avg_g * 2.0);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_low);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_high);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: DiffuseTransmissionNoNaN
//
// Edge case: dt_factor=1.0 (100% transmission, 0% diffuse reflection).
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8H: DiffuseTransmissionNoNaN",
          "[phase8h][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    BacklitScene scene;
    scene.Build(1.0f, {1.0f, 1.0f, 1.0f});

    auto result = test::RenderSceneMultiFrame(
        ctx, scene.scene, scene.mesh_data, 16, 64);

    test::WriteCombinedPNG(
        "tests/output/phase8h_no_nan.png",
        result.diffuse.data(), result.specular.data(),
        test::kTestWidth, test::kTestHeight);

    auto d_stats = test::AnalyzeRGBA16F(result.diffuse.data(), test::kPixelCount);
    auto s_stats = test::AnalyzeRGBA16F(result.specular.data(), test::kPixelCount);

    REQUIRE(d_stats.nan_count == 0);
    REQUIRE(d_stats.inf_count == 0);
    REQUIRE(s_stats.nan_count == 0);
    REQUIRE(s_stats.inf_count == 0);

    // With dt_factor=1.0, all diffuse energy goes to transmission.
    // The front face should still receive transmitted light from behind.
    REQUIRE(d_stats.nonzero_count > 100);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: SpecularPlusDiffuseTransmission
//
// Validates that specular and diffuse transmission contribute independently.
//
// A: transmission_factor=0.5 + dt_factor=0.6 (both lobes)
// B: transmission_factor=0.5 + dt_factor=0.0 (specular only)
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8H: SpecularPlusDiffuseTransmission",
          "[phase8h][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // A: both specular + diffuse transmission — 16 frames x 64 SPP
    BacklitScene scene_a;
    scene_a.Build(0.6f, {0.5f, 0.8f, 0.3f}, {1, 1, 1}, 0.5f, true);
    auto result_a = test::RenderSceneMultiFrame(
        ctx, scene_a.scene, scene_a.mesh_data, 16, 64);

    test::WriteCombinedPNG(
        "tests/output/phase8h_spec_plus_dt.png",
        result_a.diffuse.data(), result_a.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // B: only specular transmission — 16 frames x 64 SPP
    BacklitScene scene_b;
    scene_b.Build(0.0f, {0.5f, 0.8f, 0.3f}, {1, 1, 1}, 0.5f, true);
    auto result_b = test::RenderSceneMultiFrame(
        ctx, scene_b.scene, scene_b.mesh_data, 16, 64);

    test::WriteCombinedPNG(
        "tests/output/phase8h_spec_only.png",
        result_b.diffuse.data(), result_b.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // No NaN/Inf in either render
    auto stats_a = test::AnalyzeRGBA16F(result_a.diffuse.data(), test::kPixelCount);
    auto stats_b = test::AnalyzeRGBA16F(result_b.diffuse.data(), test::kPixelCount);
    REQUIRE(stats_a.nan_count == 0);
    REQUIRE(stats_a.inf_count == 0);
    REQUIRE(stats_b.nan_count == 0);
    REQUIRE(stats_b.inf_count == 0);

    // FLIP confirms visible difference
    auto rgb_a = test::TonemappedRGB(
        result_a.diffuse.data(), result_a.specular.data(), test::kPixelCount);
    auto rgb_b = test::TonemappedRGB(
        result_b.diffuse.data(), result_b.specular.data(), test::kPixelCount);

    float mean_flip = test::ComputeMeanFlip(
        rgb_a, rgb_b,
        static_cast<int>(test::kTestWidth),
        static_cast<int>(test::kTestHeight));

    std::printf("Phase 8H specular+dt vs specular-only FLIP: %.4f\n", mean_flip);
    REQUIRE(mean_flip > 0.02f);

    // A (both lobes) should have more diffuse energy than B (specular only)
    REQUIRE(stats_a.valid_count > 0);
    REQUIRE(stats_b.valid_count > 0);
    double a_diffuse_energy = stats_a.sum_r + stats_a.sum_g + stats_a.sum_b;
    double b_diffuse_energy = stats_b.sum_r + stats_b.sum_g + stats_b.sum_b;
    std::printf("  A total diffuse energy: %.2f, B total diffuse energy: %.2f\n",
                a_diffuse_energy, b_diffuse_energy);
    REQUIRE(a_diffuse_energy > b_diffuse_energy);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_a);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_b);
    ctx.WaitIdle();
}
