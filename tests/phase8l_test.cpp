#include <catch2/catch_test_macros.hpp>

#include <volk.h>

#include "test_helpers.h"

#include <monti/scene/Scene.h>
#include "gltf/GltfLoader.h"

#include "../renderer/src/vulkan/GpuScene.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace monti;
using namespace monti::vulkan;

namespace {

static std::string AssetPath(const char* filename) {
    return std::string(MONTI_TEST_ASSETS_DIR) + "/" + filename;
}

struct TestContext {
    monti::app::VulkanContext& ctx = test::SharedVulkanContext();
    bool Init() { return ctx.Device() != VK_NULL_HANDLE; }
};

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

TextureDesc MakeCheckerboard256(std::array<uint8_t, 4> color_a,
                                std::array<uint8_t, 4> color_b,
                                uint32_t cell_size = 4) {
    constexpr uint32_t kSize = 256;
    TextureDesc tex;
    tex.width = kSize;
    tex.height = kSize;
    tex.format = PixelFormat::kRGBA8_UNORM;
    tex.mip_levels = 9;
    tex.data.resize(kSize * kSize * 4);
    for (uint32_t y = 0; y < kSize; ++y) {
        for (uint32_t x = 0; x < kSize; ++x) {
            bool even = ((x / cell_size) + (y / cell_size)) % 2 == 0;
            auto& c = even ? color_a : color_b;
            uint32_t idx = (y * kSize + x) * 4;
            std::memcpy(&tex.data[idx], c.data(), 4);
        }
    }
    return tex;
}

// Create a 256x256 horizontal gradient: red on left, green on right.
TextureDesc MakeGradientTexture() {
    constexpr uint32_t kSize = 256;
    TextureDesc tex;
    tex.width = kSize;
    tex.height = kSize;
    tex.format = PixelFormat::kRGBA8_UNORM;
    tex.mip_levels = 1;
    tex.data.resize(kSize * kSize * 4);
    for (uint32_t y = 0; y < kSize; ++y) {
        for (uint32_t x = 0; x < kSize; ++x) {
            float t = static_cast<float>(x) / static_cast<float>(kSize - 1);
            uint32_t idx = (y * kSize + x) * 4;
            tex.data[idx + 0] = static_cast<uint8_t>((1.0f - t) * 255.0f);
            tex.data[idx + 1] = static_cast<uint8_t>(t * 255.0f);
            tex.data[idx + 2] = 0;
            tex.data[idx + 3] = 255;
        }
    }
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

float RegionColorVariance(const uint16_t* raw, uint32_t stride,
                          uint32_t x0, uint32_t y0,
                          uint32_t x1, uint32_t y1) {
    double sum_r = 0, sum_g = 0, sum_b = 0;
    double sum_r2 = 0, sum_g2 = 0, sum_b2 = 0;
    uint32_t count = 0;
    for (uint32_t y = y0; y < y1; ++y) {
        for (uint32_t x = x0; x < x1; ++x) {
            uint32_t i = y * stride + x;
            float r = test::HalfToFloat(raw[i * 4 + 0]);
            float g = test::HalfToFloat(raw[i * 4 + 1]);
            float b = test::HalfToFloat(raw[i * 4 + 2]);
            if (std::isnan(r) || std::isnan(g) || std::isnan(b)) continue;
            if (std::isinf(r) || std::isinf(g) || std::isinf(b)) continue;
            sum_r += r; sum_g += g; sum_b += b;
            sum_r2 += r * r; sum_g2 += g * g; sum_b2 += b * b;
            ++count;
        }
    }
    if (count < 2) return 0.0f;
    double n = static_cast<double>(count);
    double var_r = sum_r2 / n - (sum_r / n) * (sum_r / n);
    double var_g = sum_g2 / n - (sum_g / n) * (sum_g / n);
    double var_b = sum_b2 / n - (sum_b / n) * (sum_b / n);
    return static_cast<float>((var_r + var_g + var_b) / 3.0);
}

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: UV scale tiling — checkerboard with uv_scale = (4,4) shows
// higher spatial frequency than uv_scale = (1,1).
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8L: TextureTransformTiling",
          "[phase8l][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // ── Scene A: uv_scale = (1,1) baseline ──
    Scene scene_a;
    std::vector<MeshData> mesh_data_a;

    auto env_id_a = scene_a.AddTexture(MakeEnvMap(0.0f, 0.0f, 0.0f), "env");
    EnvironmentLight env_a{};
    env_a.hdr_lat_long = env_id_a;
    env_a.intensity = 0.0f;
    scene_a.SetEnvironmentLight(env_a);

    auto checker_id_a = scene_a.AddTexture(
        MakeCheckerboard256({255, 0, 0, 255}, {0, 255, 0, 255}), "checker");

    MaterialDesc mat_a;
    mat_a.base_color = {0.0f, 0.0f, 0.0f};
    mat_a.roughness = 1.0f;
    mat_a.metallic = 0.0f;
    mat_a.emissive_factor = {1.0f, 1.0f, 1.0f};
    mat_a.emissive_strength = 1.0f;
    mat_a.emissive_map = checker_id_a;
    // Identity UV transform (default)
    auto mat_id_a = scene_a.AddMaterial(std::move(mat_a), "checker_1x");

    AddQuadToScene(scene_a, mesh_data_a, "quad", mat_id_a, {0, 0, 0}, 1.0f);

    CameraParams cam;
    cam.position = {0.0f, 0.0f, 2.0f};
    cam.target = {0.0f, 0.0f, 0.0f};
    cam.up = {0.0f, 1.0f, 0.0f};
    cam.vertical_fov_radians = 0.8f;
    cam.near_plane = 0.01f;
    cam.far_plane = 100.0f;
    scene_a.SetActiveCamera(cam);

    auto result_a = test::RenderSceneMultiFrame(ctx, scene_a, mesh_data_a, 16, 4);

    // ── Scene B: uv_scale = (4,4) ──
    Scene scene_b;
    std::vector<MeshData> mesh_data_b;

    auto env_id_b = scene_b.AddTexture(MakeEnvMap(0.0f, 0.0f, 0.0f), "env");
    EnvironmentLight env_b{};
    env_b.hdr_lat_long = env_id_b;
    env_b.intensity = 0.0f;
    scene_b.SetEnvironmentLight(env_b);

    auto checker_id_b = scene_b.AddTexture(
        MakeCheckerboard256({255, 0, 0, 255}, {0, 255, 0, 255}), "checker");

    MaterialDesc mat_b;
    mat_b.base_color = {0.0f, 0.0f, 0.0f};
    mat_b.roughness = 1.0f;
    mat_b.metallic = 0.0f;
    mat_b.emissive_factor = {1.0f, 1.0f, 1.0f};
    mat_b.emissive_strength = 1.0f;
    mat_b.emissive_map = checker_id_b;
    mat_b.uv_scale = {4.0f, 4.0f};
    auto mat_id_b = scene_b.AddMaterial(std::move(mat_b), "checker_4x");

    AddQuadToScene(scene_b, mesh_data_b, "quad", mat_id_b, {0, 0, 0}, 1.0f);
    scene_b.SetActiveCamera(cam);

    auto result_b = test::RenderSceneMultiFrame(ctx, scene_b, mesh_data_b, 16, 4);

    // Compare via FLIP — tiled version should be visibly different
    auto rgb_a = test::TonemappedRGB(result_a.diffuse.data(), result_a.specular.data(),
                                     test::kPixelCount);
    auto rgb_b = test::TonemappedRGB(result_b.diffuse.data(), result_b.specular.data(),
                                     test::kPixelCount);
    float flip = test::ComputeMeanFlip(rgb_a, rgb_b,
                                       test::kTestWidth, test::kTestHeight);
    INFO("FLIP (1x vs 4x tiling): " << flip);
    REQUIRE(flip > 0.1f);

    // Variances should differ — the UV transform changes the pattern
    float var_a = RegionColorVariance(result_a.diffuse.data(), test::kTestWidth,
                                      64, 64, 192, 192);
    float var_b = RegionColorVariance(result_b.diffuse.data(), test::kTestWidth,
                                      64, 64, 192, 192);
    INFO("Variance baseline: " << var_a << " tiled: " << var_b);
    REQUIRE(var_a != var_b);

    test::WritePNG("tests/output/phase8l_tiling_1x.png",
                   result_a.diffuse.data(), test::kTestWidth, test::kTestHeight);
    test::WritePNG("tests/output/phase8l_tiling_4x.png",
                   result_b.diffuse.data(), test::kTestWidth, test::kTestHeight);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_a);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_b);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Identity transform — DamagedHelmet (no KHR_texture_transform)
// renders identically to baseline. Confirms the identity early-out path.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8L: TextureTransformIdentity",
          "[phase8l][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("DamagedHelmet.glb"));
    REQUIRE(result.success);
    REQUIRE_FALSE(result.mesh_data.empty());

    // Verify identity transform on all materials
    for (const auto& mat : scene.Materials()) {
        REQUIRE(mat.uv_offset == glm::vec2(0, 0));
        REQUIRE(mat.uv_scale == glm::vec2(1, 1));
        REQUIRE(mat.uv_rotation == 0.0f);
    }

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

    auto mf_result = test::RenderSceneMultiFrame(ctx, scene, result.mesh_data, 16, 4);

    // No NaN or Inf
    auto diff_stats = test::AnalyzeRGBA16F(mf_result.diffuse.data(), test::kPixelCount);
    auto spec_stats = test::AnalyzeRGBA16F(mf_result.specular.data(), test::kPixelCount);
    REQUIRE(diff_stats.nan_count == 0);
    REQUIRE(diff_stats.inf_count == 0);
    REQUIRE(spec_stats.nan_count == 0);
    REQUIRE(spec_stats.inf_count == 0);

    // Should have non-trivial output
    REQUIRE(diff_stats.nonzero_count > 100);
    REQUIRE(diff_stats.has_color_variation);

    test::WriteCombinedPNG("tests/output/phase8l_identity.png",
                           mf_result.diffuse.data(), mf_result.specular.data(),
                           test::kTestWidth, test::kTestHeight);

    test::CleanupMultiFrameResult(ctx.Allocator(), mf_result);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: UV rotation — gradient texture rotated 90° should swap axes.
// Horizontal gradient becomes vertical.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8L: TextureTransformRotation",
          "[phase8l][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    constexpr float kPiOver2 = 1.5707963267948966f;

    // ── Scene A: no rotation ──
    Scene scene_a;
    std::vector<MeshData> mesh_data_a;

    auto env_id_a = scene_a.AddTexture(MakeEnvMap(0.0f, 0.0f, 0.0f), "env");
    EnvironmentLight env_a{};
    env_a.hdr_lat_long = env_id_a;
    env_a.intensity = 0.0f;
    scene_a.SetEnvironmentLight(env_a);

    auto grad_id_a = scene_a.AddTexture(MakeGradientTexture(), "gradient");

    MaterialDesc mat_a;
    mat_a.base_color = {0.0f, 0.0f, 0.0f};
    mat_a.roughness = 1.0f;
    mat_a.metallic = 0.0f;
    mat_a.emissive_factor = {1.0f, 1.0f, 1.0f};
    mat_a.emissive_strength = 1.0f;
    mat_a.emissive_map = grad_id_a;
    auto mat_id_a = scene_a.AddMaterial(std::move(mat_a), "gradient_norot");

    AddQuadToScene(scene_a, mesh_data_a, "quad", mat_id_a, {0, 0, 0}, 1.0f);

    CameraParams cam;
    cam.position = {0.0f, 0.0f, 2.0f};
    cam.target = {0.0f, 0.0f, 0.0f};
    cam.up = {0.0f, 1.0f, 0.0f};
    cam.vertical_fov_radians = 0.8f;
    cam.near_plane = 0.01f;
    cam.far_plane = 100.0f;
    scene_a.SetActiveCamera(cam);

    auto result_a = test::RenderSceneMultiFrame(ctx, scene_a, mesh_data_a, 16, 4);

    // ── Scene B: rotation = π/2 ──
    Scene scene_b;
    std::vector<MeshData> mesh_data_b;

    auto env_id_b = scene_b.AddTexture(MakeEnvMap(0.0f, 0.0f, 0.0f), "env");
    EnvironmentLight env_b{};
    env_b.hdr_lat_long = env_id_b;
    env_b.intensity = 0.0f;
    scene_b.SetEnvironmentLight(env_b);

    auto grad_id_b = scene_b.AddTexture(MakeGradientTexture(), "gradient");

    MaterialDesc mat_b;
    mat_b.base_color = {0.0f, 0.0f, 0.0f};
    mat_b.roughness = 1.0f;
    mat_b.metallic = 0.0f;
    mat_b.emissive_factor = {1.0f, 1.0f, 1.0f};
    mat_b.emissive_strength = 1.0f;
    mat_b.emissive_map = grad_id_b;
    mat_b.uv_rotation = kPiOver2;
    auto mat_id_b = scene_b.AddMaterial(std::move(mat_b), "gradient_rot90");

    AddQuadToScene(scene_b, mesh_data_b, "quad", mat_id_b, {0, 0, 0}, 1.0f);
    scene_b.SetActiveCamera(cam);

    auto result_b = test::RenderSceneMultiFrame(ctx, scene_b, mesh_data_b, 16, 4);

    // Rotated gradient should look different from unrotated
    auto rgb_a = test::TonemappedRGB(result_a.diffuse.data(), result_a.specular.data(),
                                     test::kPixelCount);
    auto rgb_b = test::TonemappedRGB(result_b.diffuse.data(), result_b.specular.data(),
                                     test::kPixelCount);
    float flip = test::ComputeMeanFlip(rgb_a, rgb_b,
                                       test::kTestWidth, test::kTestHeight);
    INFO("FLIP (unrotated vs rotated): " << flip);
    REQUIRE(flip > 0.05f);

    test::WritePNG("tests/output/phase8l_rotation_none.png",
                   result_a.diffuse.data(), test::kTestWidth, test::kTestHeight);
    test::WritePNG("tests/output/phase8l_rotation_90.png",
                   result_b.diffuse.data(), test::kTestWidth, test::kTestHeight);

    test::CleanupMultiFrameResult(ctx.Allocator(), result_a);
    test::CleanupMultiFrameResult(ctx.Allocator(), result_b);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: No NaN/Inf with transformed UVs at 1 spp across all test scenes.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8L: TextureTransformNoNaN",
          "[phase8l][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // Test with extreme UV transform parameters
    Scene scene;
    std::vector<MeshData> mesh_data;

    auto env_id = scene.AddTexture(MakeEnvMap(0.2f, 0.2f, 0.2f), "env");
    EnvironmentLight env{};
    env.hdr_lat_long = env_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto checker_id = scene.AddTexture(
        MakeCheckerboard256({255, 128, 0, 255}, {0, 128, 255, 255}), "checker");

    // Extreme scale + rotation + offset
    MaterialDesc mat;
    mat.base_color = {1.0f, 1.0f, 1.0f};
    mat.roughness = 0.5f;
    mat.metallic = 0.0f;
    mat.emissive_factor = {1.0f, 1.0f, 1.0f};
    mat.emissive_strength = 0.5f;
    mat.emissive_map = checker_id;
    mat.base_color_map = checker_id;
    mat.uv_scale = {100.0f, 100.0f};
    mat.uv_offset = {-50.0f, -50.0f};
    mat.uv_rotation = 2.35619f;  // 3π/4
    auto mat_id = scene.AddMaterial(std::move(mat), "extreme_transform");

    AddQuadToScene(scene, mesh_data, "quad", mat_id, {0, 0, 0}, 1.0f);

    CameraParams cam;
    cam.position = {0.0f, 0.0f, 2.0f};
    cam.target = {0.0f, 0.0f, 0.0f};
    cam.up = {0.0f, 1.0f, 0.0f};
    cam.vertical_fov_radians = 0.8f;
    cam.near_plane = 0.01f;
    cam.far_plane = 100.0f;
    scene.SetActiveCamera(cam);

    // Single frame, 1 spp — minimal cost, checks for NaN/Inf
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
    test::FillRendererProcAddrs(desc, ctx);

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(*renderer, ctx.Allocator(),
                                               ctx.Device(), upload_cmd,
                                               mesh_data,
                                               test::MakeGpuBufferProcs());
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

    auto specular_rb = test::ReadbackImage(ctx, gbuffer_images.NoisySpecularImage());
    auto* specular_raw = static_cast<uint16_t*>(specular_rb.Map());
    REQUIRE(specular_raw != nullptr);

    auto diff_stats = test::AnalyzeRGBA16F(diffuse_raw, test::kPixelCount);
    auto spec_stats = test::AnalyzeRGBA16F(specular_raw, test::kPixelCount);

    diffuse_rb.Unmap();
    specular_rb.Unmap();

    REQUIRE(diff_stats.nan_count == 0);
    REQUIRE(diff_stats.inf_count == 0);
    REQUIRE(spec_stats.nan_count == 0);
    REQUIRE(spec_stats.inf_count == 0);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: High-SPP showcase — converged PBR render with tiled checkerboard.
// Demonstrates UV tiling with full path-traced lighting at 16×64=1024 SPP.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8L: TextureTransformHighSPP",
          "[phase8l][renderer][vulkan][integration][high_spp]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;
    std::vector<MeshData> mesh_data;

    auto env_id = scene.AddTexture(MakeEnvMap(0.3f, 0.3f, 0.3f), "env");
    EnvironmentLight env{};
    env.hdr_lat_long = env_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto checker_id = scene.AddTexture(
        MakeCheckerboard256({200, 40, 40, 255}, {240, 230, 200, 255}), "checker");

    // Tiled checkerboard on PBR quad with full lighting
    MaterialDesc mat;
    mat.base_color = {1.0f, 1.0f, 1.0f};
    mat.roughness = 0.4f;
    mat.metallic = 0.0f;
    mat.base_color_map = checker_id;
    mat.uv_scale = {4.0f, 4.0f};
    auto mat_id = scene.AddMaterial(std::move(mat), "tiled_pbr");

    AddQuadToScene(scene, mesh_data, "quad", mat_id, {0, 0, 0}, 1.0f);

    // Gray back wall
    MaterialDesc back_mat;
    back_mat.base_color = {0.5f, 0.5f, 0.5f};
    back_mat.roughness = 1.0f;
    auto back_id = scene.AddMaterial(std::move(back_mat), "back");
    AddQuadToScene(scene, mesh_data, "back", back_id, {0, 0, -1.5f}, 2.0f);

    AreaLight light;
    light.corner = {-1.0f, 1.5f, -0.5f};
    light.edge_a = {2.0f, 0.0f, 0.0f};
    light.edge_b = {0.0f, 0.0f, 2.0f};
    light.radiance = {3.0f, 3.0f, 3.0f};
    light.two_sided = true;
    scene.AddAreaLight(light);

    CameraParams cam;
    cam.position = {0.0f, 0.0f, 2.0f};
    cam.target = {0.0f, 0.0f, 0.0f};
    cam.up = {0.0f, 1.0f, 0.0f};
    cam.vertical_fov_radians = 0.8f;
    cam.near_plane = 0.01f;
    cam.far_plane = 100.0f;
    scene.SetActiveCamera(cam);

    auto result = test::RenderSceneMultiFrame(ctx, scene, mesh_data, 16, 64);

    test::WriteCombinedPNG("tests/output/phase8l_tiling_high_spp.png",
                           result.diffuse.data(), result.specular.data(),
                           test::kTestWidth, test::kTestHeight);

    auto stats = test::AnalyzeRGBA16F(result.diffuse.data(), test::kPixelCount);
    REQUIRE(stats.nan_count == 0);
    REQUIRE(stats.inf_count == 0);
    REQUIRE(stats.nonzero_count > 100);
    REQUIRE(stats.has_color_variation);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
    ctx.WaitIdle();
}
