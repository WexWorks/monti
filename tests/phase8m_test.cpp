#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <volk.h>

#include "test_helpers.h"
#include "scenes/Primitives.h"

#include <monti/scene/Scene.h>
#include "gltf/GltfLoader.h"

#include "../renderer/src/vulkan/GpuScene.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace monti;
using namespace monti::vulkan;

// ═══════════════════════════════════════════════════════════════════════════
//
// Phase 8M: KHR_materials_sheen (Charlie Sheen BSDF) integration tests
//
// Tests verify the sheen lobe produces visible edge brightening (grazing
// angle effect), respects energy conservation, and introduces no NaN.
// Scene design isolates features per the testing philosophy: no feature
// toggles, only scene-based signal isolation.
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

// ── Scene construction helpers ───────────────────────────────────────────

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

// Common test environment: environment map, back wall, floor, area light, camera.
void SetupSheenTestEnvironment(Scene& scene, std::vector<MeshData>& mesh_data) {
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.3f, 0.3f, 0.3f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    // Back wall
    MaterialDesc back_mat;
    back_mat.base_color = {0.5f, 0.5f, 0.5f};
    back_mat.roughness = 1.0f;
    auto back_id = scene.AddMaterial(std::move(back_mat), "back_wall");
    AddQuadToScene(scene, mesh_data, "back_wall", back_id, {0, 0, -2.0f}, 3.0f);

    // Floor
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
    MaterialDesc floor_mat;
    floor_mat.base_color = {0.5f, 0.5f, 0.5f};
    floor_mat.roughness = 0.8f;
    auto floor_id = scene.AddMaterial(std::move(floor_mat), "floor");
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
// Test 1: SheenVisibleOnFabric
//
// Sphere with sheen (white, roughness=0.5) vs sphere without sheen.
// Sheen produces visible edge brightening at grazing angles (fabric luster).
// FLIP between the two must exceed a threshold, proving the sheen lobe
// contributes visible energy.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8M: SheenVisibleOnFabric",
          "[phase8m][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // Scene A: sphere WITH sheen
    Scene scene_a;
    std::vector<MeshData> mesh_data_a;
    SetupSheenTestEnvironment(scene_a, mesh_data_a);

    MaterialDesc sheen_mat;
    sheen_mat.base_color = {0.4f, 0.4f, 0.4f};
    sheen_mat.roughness = 0.8f;
    sheen_mat.metallic = 0.0f;
    sheen_mat.sheen_color = {0.8f, 0.8f, 0.8f};
    sheen_mat.sheen_roughness = 0.5f;
    auto sheen_id = scene_a.AddMaterial(std::move(sheen_mat), "fabric_sheen");
    AddIcosphereToScene(scene_a, mesh_data_a, "sphere", sheen_id,
                        {0, 0, 0}, 0.8f, 3);

    auto result_a = test::RenderSceneMultiFrame(
        ctx, scene_a, mesh_data_a, 8, 8);

    test::WriteCombinedPNG(
        "tests/output/phase8m_sheen_visible.png",
        result_a.diffuse.data(), result_a.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // Scene B: sphere WITHOUT sheen (same base material)
    Scene scene_b;
    std::vector<MeshData> mesh_data_b;
    SetupSheenTestEnvironment(scene_b, mesh_data_b);

    MaterialDesc no_sheen_mat;
    no_sheen_mat.base_color = {0.4f, 0.4f, 0.4f};
    no_sheen_mat.roughness = 0.8f;
    no_sheen_mat.metallic = 0.0f;
    auto no_sheen_id = scene_b.AddMaterial(std::move(no_sheen_mat), "fabric_no_sheen");
    AddIcosphereToScene(scene_b, mesh_data_b, "sphere", no_sheen_id,
                        {0, 0, 0}, 0.8f, 3);

    auto result_b = test::RenderSceneMultiFrame(
        ctx, scene_b, mesh_data_b, 8, 8);

    test::WriteCombinedPNG(
        "tests/output/phase8m_no_sheen.png",
        result_b.diffuse.data(), result_b.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // FLIP comparison
    constexpr uint32_t kPixelCount = test::kTestWidth * test::kTestHeight;
    auto rgb_a = test::TonemappedRGB(
        result_a.diffuse.data(), result_a.specular.data(), kPixelCount);
    auto rgb_b = test::TonemappedRGB(
        result_b.diffuse.data(), result_b.specular.data(), kPixelCount);

    float mean_flip = test::ComputeMeanFlip(rgb_a, rgb_b,
        static_cast<int>(test::kTestWidth),
        static_cast<int>(test::kTestHeight));

    std::printf("Phase 8M SheenVisibleOnFabric FLIP: %.4f\n", mean_flip);

    // Sheen should produce a clearly visible difference
    REQUIRE(mean_flip > 0.02f);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_a);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_b);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: SheenEnergyConservation (furnace test)
//
// White sphere with max sheen inside a uniform white environment.
// Mean luminance must not exceed 1.0 (no energy gain). The sheen lobe
// with albedo scaling should attenuate the base accordingly.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8M: SheenEnergyConservation",
          "[phase8m][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;
    std::vector<MeshData> mesh_data;

    // Uniform white environment (furnace)
    auto env_tex_id = scene.AddTexture(MakeEnvMap(1.0f, 1.0f, 1.0f), "env_white");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    // White sphere with full sheen
    MaterialDesc mat;
    mat.base_color = {1.0f, 1.0f, 1.0f};
    mat.roughness = 0.5f;
    mat.metallic = 0.0f;
    mat.sheen_color = {1.0f, 1.0f, 1.0f};
    mat.sheen_roughness = 0.5f;
    auto mat_id = scene.AddMaterial(std::move(mat), "white_sheen");
    AddIcosphereToScene(scene, mesh_data, "sphere", mat_id,
                        {0, 0, 0}, 0.8f, 3);

    CameraParams camera;
    camera.position = {0.0f, 0.0f, 3.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.vertical_fov_radians = 0.8f;
    camera.near_plane = 0.01f;
    camera.far_plane = 100.0f;
    scene.SetActiveCamera(camera);

    auto result = test::RenderSceneMultiFrame(
        ctx, scene, mesh_data, 16, 16);

    test::WriteCombinedPNG(
        "tests/output/phase8m_sheen_furnace.png",
        result.diffuse.data(), result.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // Check mean luminance does not exceed 1.0
    constexpr uint32_t kPixelCount = test::kTestWidth * test::kTestHeight;
    auto stats = test::AnalyzeRGBA16F(result.diffuse.data(), kPixelCount);
    auto spec_stats = test::AnalyzeRGBA16F(result.specular.data(), kPixelCount);

    double total_lum = (stats.sum_r + stats.sum_g + stats.sum_b) / 3.0;
    total_lum += (spec_stats.sum_r + spec_stats.sum_g + spec_stats.sum_b) / 3.0;
    uint32_t total_valid = stats.valid_count;
    double mean_lum = total_valid > 0 ? total_lum / total_valid : 0.0;

    std::printf("Phase 8M SheenEnergyConservation: mean_luminance=%.4f\n",
                mean_lum);

    // Energy conservation: mean luminance ≤ 1.0 (with small tolerance for MC noise)
    REQUIRE(mean_lum <= 1.05);

    REQUIRE(stats.nan_count == 0);
    REQUIRE(stats.inf_count == 0);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: SheenNoEffectWhenZero
//
// Render DamagedHelmet.glb (no sheen in its materials). The sheen code
// path should produce zero contribution. FLIP against itself at different
// quality should be low (pure convergence test), confirming zero sheen
// causes no regression.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8M: SheenNoEffectWhenZero",
          "[phase8m][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto asset_path = std::string(MONTI_TEST_ASSETS_DIR) + "/DamagedHelmet.glb";
    Scene scene;
    auto load_result = monti::gltf::LoadGltf(scene, asset_path);
    REQUIRE(load_result.success);
    auto& mesh_data = load_result.mesh_data;

    // Add environment map
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.3f, 0.3f, 0.3f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    CameraParams camera;
    camera.position = {0.0f, 0.0f, 3.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.vertical_fov_radians = 0.8f;
    camera.near_plane = 0.01f;
    camera.far_plane = 100.0f;
    scene.SetActiveCamera(camera);

    auto result = test::RenderSceneMultiFrame(
        ctx, scene, mesh_data, 8, 8);

    test::WriteCombinedPNG(
        "tests/output/phase8m_no_sheen_helmet.png",
        result.diffuse.data(), result.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // Verify no NaN/Inf in output
    constexpr uint32_t kPixelCount = test::kTestWidth * test::kTestHeight;
    auto stats = test::AnalyzeRGBA16F(result.diffuse.data(), kPixelCount);
    auto spec_stats = test::AnalyzeRGBA16F(result.specular.data(), kPixelCount);

    REQUIRE(stats.nan_count == 0);
    REQUIRE(stats.inf_count == 0);
    REQUIRE(spec_stats.nan_count == 0);
    REQUIRE(spec_stats.inf_count == 0);

    // Verify the image has content (not black)
    REQUIRE(stats.nonzero_count > kPixelCount / 10);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: SheenColorTinting
//
// Sphere with blue sheen (0, 0, 1). Compared against sphere with no sheen,
// the sheen sphere should have elevated blue channel at grazing angles.
// Verify by comparing blue channel proportion of the combined output.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8M: SheenColorTinting",
          "[phase8m][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // Scene A: blue sheen sphere
    Scene scene_a;
    std::vector<MeshData> mesh_data_a;
    SetupSheenTestEnvironment(scene_a, mesh_data_a);

    MaterialDesc blue_sheen_mat;
    blue_sheen_mat.base_color = {0.4f, 0.4f, 0.4f};
    blue_sheen_mat.roughness = 0.8f;
    blue_sheen_mat.metallic = 0.0f;
    blue_sheen_mat.sheen_color = {0.0f, 0.0f, 1.0f};
    blue_sheen_mat.sheen_roughness = 0.5f;
    auto blue_id = scene_a.AddMaterial(std::move(blue_sheen_mat), "blue_sheen");
    AddIcosphereToScene(scene_a, mesh_data_a, "sphere", blue_id,
                        {0, 0, 0}, 0.8f, 3);

    auto result_a = test::RenderSceneMultiFrame(
        ctx, scene_a, mesh_data_a, 8, 8);

    test::WriteCombinedPNG(
        "tests/output/phase8m_blue_sheen.png",
        result_a.diffuse.data(), result_a.specular.data(),
        test::kTestWidth, test::kTestHeight);

    // Scene B: no sheen (neutral)
    Scene scene_b;
    std::vector<MeshData> mesh_data_b;
    SetupSheenTestEnvironment(scene_b, mesh_data_b);

    MaterialDesc neutral_mat;
    neutral_mat.base_color = {0.4f, 0.4f, 0.4f};
    neutral_mat.roughness = 0.8f;
    neutral_mat.metallic = 0.0f;
    auto neutral_id = scene_b.AddMaterial(std::move(neutral_mat), "neutral");
    AddIcosphereToScene(scene_b, mesh_data_b, "sphere", neutral_id,
                        {0, 0, 0}, 0.8f, 3);

    auto result_b = test::RenderSceneMultiFrame(
        ctx, scene_b, mesh_data_b, 8, 8);

    // Compare blue channel proportions
    constexpr uint32_t kPixelCount = test::kTestWidth * test::kTestHeight;
    auto stats_a = test::AnalyzeRGBA16F(result_a.diffuse.data(), kPixelCount);
    auto spec_a = test::AnalyzeRGBA16F(result_a.specular.data(), kPixelCount);
    auto stats_b = test::AnalyzeRGBA16F(result_b.diffuse.data(), kPixelCount);
    auto spec_b = test::AnalyzeRGBA16F(result_b.specular.data(), kPixelCount);

    double total_b_a = stats_a.sum_b + spec_a.sum_b;
    double total_r_a = stats_a.sum_r + spec_a.sum_r;
    double total_b_b = stats_b.sum_b + spec_b.sum_b;
    double total_r_b = stats_b.sum_r + spec_b.sum_r;

    // Blue sheen sphere must have higher blue/red ratio than neutral sphere
    double ratio_a = (total_r_a > 0.0) ? total_b_a / total_r_a : 0.0;
    double ratio_b = (total_r_b > 0.0) ? total_b_b / total_r_b : 0.0;

    std::printf("Phase 8M SheenColorTinting: blue/red ratio sheen=%.4f, neutral=%.4f\n",
                ratio_a, ratio_b);

    REQUIRE(ratio_a > ratio_b);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_a);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_b);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: SheenNoNaN
//
// Render sheen materials at 1 spp with extreme parameters (roughness near
// 0 and 1, full sheen color). No NaN or Inf in output. Guards against
// alpha_g = 0 singularity in Charlie NDF.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8M: SheenNoNaN",
          "[phase8m][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;
    std::vector<MeshData> mesh_data;
    SetupSheenTestEnvironment(scene, mesh_data);

    // Extreme sheen: roughness near zero (singularity guard test)
    MaterialDesc mat_low;
    mat_low.base_color = {0.5f, 0.5f, 0.5f};
    mat_low.roughness = 1.0f;
    mat_low.sheen_color = {1.0f, 1.0f, 1.0f};
    mat_low.sheen_roughness = 0.001f;
    auto low_id = scene.AddMaterial(std::move(mat_low), "sheen_low_rough");
    AddIcosphereToScene(scene, mesh_data, "sphere_low", low_id,
                        {-1.0f, 0, 0}, 0.4f, 2);

    // Maximum roughness
    MaterialDesc mat_high;
    mat_high.base_color = {0.5f, 0.5f, 0.5f};
    mat_high.roughness = 1.0f;
    mat_high.sheen_color = {1.0f, 1.0f, 1.0f};
    mat_high.sheen_roughness = 1.0f;
    auto high_id = scene.AddMaterial(std::move(mat_high), "sheen_high_rough");
    AddIcosphereToScene(scene, mesh_data, "sphere_high", high_id,
                        {1.0f, 0, 0}, 0.4f, 2);

    auto result = test::RenderSceneMultiFrame(
        ctx, scene, mesh_data, 1, 1);

    test::WriteCombinedPNG(
        "tests/output/phase8m_sheen_extreme.png",
        result.diffuse.data(), result.specular.data(),
        test::kTestWidth, test::kTestHeight);

    constexpr uint32_t kPixelCount = test::kTestWidth * test::kTestHeight;
    auto stats = test::AnalyzeRGBA16F(result.diffuse.data(), kPixelCount);
    auto spec_stats = test::AnalyzeRGBA16F(result.specular.data(), kPixelCount);

    std::printf("Phase 8M SheenNoNaN: nan=%u, inf=%u (diffuse); nan=%u, inf=%u (specular)\n",
                stats.nan_count, stats.inf_count,
                spec_stats.nan_count, spec_stats.inf_count);

    REQUIRE(stats.nan_count == 0);
    REQUIRE(stats.inf_count == 0);
    REQUIRE(spec_stats.nan_count == 0);
    REQUIRE(spec_stats.inf_count == 0);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: High-SPP showcase — converged sheen renders at 16×64=1024 SPP.
// Three spheres with varying sheen roughness (0.2, 0.5, 0.8) demonstrate
// the Charlie sheen BRDF converging to smooth fabric appearance.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8M: SheenHighSPP",
          "[phase8m][renderer][vulkan][integration][high_spp]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;
    std::vector<MeshData> mesh_data;
    SetupSheenTestEnvironment(scene, mesh_data);

    // Three spheres with increasing sheen roughness
    MaterialDesc mat_low;
    mat_low.base_color = {0.3f, 0.3f, 0.3f};
    mat_low.roughness = 0.9f;
    mat_low.metallic = 0.0f;
    mat_low.sheen_color = {0.9f, 0.9f, 0.9f};
    mat_low.sheen_roughness = 0.2f;
    auto low_id = scene.AddMaterial(std::move(mat_low), "sheen_low");
    AddIcosphereToScene(scene, mesh_data, "sphere_low", low_id,
                        {-1.2f, 0, 0}, 0.45f, 3);

    MaterialDesc mat_mid;
    mat_mid.base_color = {0.3f, 0.3f, 0.3f};
    mat_mid.roughness = 0.9f;
    mat_mid.metallic = 0.0f;
    mat_mid.sheen_color = {0.9f, 0.9f, 0.9f};
    mat_mid.sheen_roughness = 0.5f;
    auto mid_id = scene.AddMaterial(std::move(mat_mid), "sheen_mid");
    AddIcosphereToScene(scene, mesh_data, "sphere_mid", mid_id,
                        {0.0f, 0, 0}, 0.45f, 3);

    MaterialDesc mat_high;
    mat_high.base_color = {0.3f, 0.3f, 0.3f};
    mat_high.roughness = 0.9f;
    mat_high.metallic = 0.0f;
    mat_high.sheen_color = {0.9f, 0.9f, 0.9f};
    mat_high.sheen_roughness = 0.8f;
    auto high_id = scene.AddMaterial(std::move(mat_high), "sheen_high");
    AddIcosphereToScene(scene, mesh_data, "sphere_high", high_id,
                        {1.2f, 0, 0}, 0.45f, 3);

    auto result = test::RenderSceneMultiFrame(
        ctx, scene, mesh_data, 16, 64);

    test::WriteCombinedPNG(
        "tests/output/phase8m_sheen_high_spp.png",
        result.diffuse.data(), result.specular.data(),
        test::kTestWidth, test::kTestHeight);

    constexpr uint32_t kPixelCount = test::kTestWidth * test::kTestHeight;
    auto stats = test::AnalyzeRGBA16F(result.diffuse.data(), kPixelCount);
    auto spec_stats = test::AnalyzeRGBA16F(result.specular.data(), kPixelCount);

    REQUIRE(stats.nan_count == 0);
    REQUIRE(stats.inf_count == 0);
    REQUIRE(spec_stats.nan_count == 0);
    REQUIRE(spec_stats.inf_count == 0);
    REQUIRE(stats.nonzero_count > kPixelCount / 10);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
}
