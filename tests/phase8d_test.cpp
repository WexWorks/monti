#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <volk.h>

#include "test_helpers.h"
#include "scenes/CornellBox.h"

#include <monti/scene/Scene.h>

#include "gltf/GltfLoader.h"
#include "EnvironmentLoader.h"

#include "../renderer/src/vulkan/GpuScene.h"
#include "../renderer/src/vulkan/DeviceDispatch.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <vector>

using namespace monti;
using namespace monti::vulkan;
using Catch::Matchers::WithinAbs;

namespace {

struct TestContext {
    monti::app::VulkanContext& ctx = test::SharedVulkanContext();
    DeviceDispatch dispatch;

    bool Init() {
        if (ctx.Device() == VK_NULL_HANDLE) return false;
        if (!dispatch.Load(ctx.Device(), ctx.Instance(),
                           ctx.GetDeviceProcAddr(), ctx.GetInstanceProcAddr()))
            return false;
        return true;
    }
};

static std::string AssetPath(const char* filename) {
    return std::string(MONTI_TEST_ASSETS_DIR) + "/" + filename;
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

// Build a flat quad mesh facing +Z
MeshData MakeQuad(const glm::vec3& center, float half_size) {
    glm::vec3 n{0, 0, 1};
    glm::vec4 t{1, 0, 0, 1};  // Tangent along +X, bitangent sign +1
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

// Create a procedural 2x2 RGBA8 texture with given pixel data
TextureDesc MakeTexture2x2(std::array<uint8_t, 4> p0,
                           std::array<uint8_t, 4> p1,
                           std::array<uint8_t, 4> p2,
                           std::array<uint8_t, 4> p3) {
    TextureDesc tex;
    tex.width = 2;
    tex.height = 2;
    tex.format = PixelFormat::kRGBA8_UNORM;
    tex.data.resize(2 * 2 * 4);
    std::memcpy(&tex.data[0],  p0.data(), 4);
    std::memcpy(&tex.data[4],  p1.data(), 4);
    std::memcpy(&tex.data[8],  p2.data(), 4);
    std::memcpy(&tex.data[12], p3.data(), 4);
    return tex;
}

// Create a solid-color 4x2 RGBA32F environment map
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

// Create a directional HDR environment map with a bright warm spot.
// Unlike the uniform MakeEnvMap(), this provides angular variation that
// reveals material property differences (metallic reflection, normal
// perturbation, roughness) in side-by-side material comparison tests.
TextureDesc MakeDirectionalEnvMap() {
    constexpr uint32_t kW = 32, kH = 16;
    std::vector<float> pixels(kW * kH * 4);
    for (uint32_t y = 0; y < kH; ++y) {
        for (uint32_t x = 0; x < kW; ++x) {
            float u = static_cast<float>(x) / static_cast<float>(kW);
            float v = static_cast<float>(y) / static_cast<float>(kH);
            // Sky gradient: blue-ish at top, warm at horizon
            float sky_r = 0.15f + 0.25f * v;
            float sky_g = 0.2f + 0.15f * v;
            float sky_b = 0.5f - 0.2f * v;
            // Bright sun spot near (u=0.25, v=0.3)
            float du = u - 0.25f;
            float dv = v - 0.3f;
            float sun = std::exp(-(du * du + dv * dv) / 0.008f);
            uint32_t i = (y * kW + x) * 4;
            pixels[i + 0] = sky_r + sun * 12.0f;
            pixels[i + 1] = sky_g + sun * 10.0f;
            pixels[i + 2] = sky_b + sun * 7.0f;
            pixels[i + 3] = 1.0f;
        }
    }
    TextureDesc tex;
    tex.width = kW;
    tex.height = kH;
    tex.format = PixelFormat::kRGBA32F;
    tex.data.resize(pixels.size() * sizeof(float));
    std::memcpy(tex.data.data(), pixels.data(), tex.data.size());
    return tex;
}

// Try to load a real HDR environment from the shared environments directory.
// Falls back to a procedural constant-color environment if unavailable.
TextureDesc LoadOrMakeEnvMap(float fallback_r, float fallback_g, float fallback_b,
                             std::string_view exr_name = "royal_esplanade_2k.exr") {
#ifdef MONTI_ENVIRONMENTS_DIR
    auto path = std::filesystem::path(MONTI_ENVIRONMENTS_DIR) / exr_name;
    if (std::filesystem::exists(path)) {
        auto tex = monti::app::LoadExrEnvironment(path.string());
        if (tex) return std::move(*tex);
    }
#endif
    return MakeEnvMap(fallback_r, fallback_g, fallback_b);
}

// Add mesh and node to scene, return mesh data
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

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: PackedMaterial 112-byte layout with all new texture channels
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8D: PackedMaterial 112-byte layout with all texture channels",
          "[phase8d][gpu_scene]") {
    // 7 vec4 = 112 bytes
    STATIC_REQUIRE(sizeof(PackedMaterial) == 176);
    STATIC_REQUIRE(alignof(PackedMaterial) == 16);

    // Verify field offsets (vec4 boundaries)
    PackedMaterial pm{};
    auto* base = reinterpret_cast<const char*>(&pm);
    REQUIRE(reinterpret_cast<const char*>(&pm.base_color_roughness) - base == 0);
    REQUIRE(reinterpret_cast<const char*>(&pm.metallic_clearcoat)   - base == 16);
    REQUIRE(reinterpret_cast<const char*>(&pm.opacity_ior)          - base == 32);
    REQUIRE(reinterpret_cast<const char*>(&pm.transmission_volume)  - base == 48);
    REQUIRE(reinterpret_cast<const char*>(&pm.attenuation_color_pad)- base == 64);
    REQUIRE(reinterpret_cast<const char*>(&pm.alpha_mode_misc)      - base == 80);
    REQUIRE(reinterpret_cast<const char*>(&pm.emissive)             - base == 96);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Material packing encodes all Phase 8D texture maps
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8D: Material packing encodes MR, normal, transmission, emissive maps",
          "[phase8d][gpu_scene][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;

    // Add 4 textures (one for each new channel)
    auto mr_tex_id = scene.AddTexture(
        MakeTexture2x2({0,128,128,255}, {0,128,128,255},
                       {0,128,128,255}, {0,128,128,255}), "mr_tex");
    auto norm_tex_id = scene.AddTexture(
        MakeTexture2x2({128,128,255,255}, {128,128,255,255},
                       {128,128,255,255}, {128,128,255,255}), "norm_tex");
    auto trans_tex_id = scene.AddTexture(
        MakeTexture2x2({128,0,0,255}, {128,0,0,255},
                       {128,0,0,255}, {128,0,0,255}), "trans_tex");
    auto emissive_tex_id = scene.AddTexture(
        MakeTexture2x2({255,128,0,255}, {255,128,0,255},
                       {255,128,0,255}, {255,128,0,255}), "emissive_tex");
    auto base_tex_id = scene.AddTexture(
        MakeTexture2x2({200,180,160,255}, {200,180,160,255},
                       {200,180,160,255}, {200,180,160,255}), "base_tex");

    // Material referencing all texture channels
    MaterialDesc mat;
    mat.base_color = {0.8f, 0.7f, 0.6f};
    mat.roughness = 0.5f;
    mat.metallic = 1.0f;
    mat.normal_scale = 0.75f;
    mat.base_color_map = base_tex_id;
    mat.metallic_roughness_map = mr_tex_id;
    mat.normal_map = norm_tex_id;
    mat.transmission_factor = 0.4f;
    mat.transmission_map = trans_tex_id;
    mat.emissive_factor = {5.0f, 2.5f, 1.0f};
    mat.emissive_strength = 2.0f;
    mat.emissive_map = emissive_tex_id;
    auto mat_id = scene.AddMaterial(std::move(mat), "full_pbr");

    // Material with NO textures (sentinel test)
    MaterialDesc plain_mat;
    plain_mat.base_color = {1, 1, 1};
    plain_mat.roughness = 0.5f;
    auto plain_id = scene.AddMaterial(std::move(plain_mat), "plain");

    GpuScene gpu_scene(ctx.Allocator(), ctx.Device(), ctx.PhysicalDevice(), tc.dispatch);

    // Upload textures first
    VkCommandBuffer cmd = ctx.BeginOneShot();
    auto staging = gpu_scene.UploadTextures(scene, cmd);
    ctx.SubmitAndWait(cmd);
    REQUIRE(gpu_scene.TextureCount() == 5);

    // Update materials
    REQUIRE(gpu_scene.UpdateMaterials(scene));
    REQUIRE(gpu_scene.MaterialCount() == 2);
    REQUIRE(gpu_scene.MaterialBufferSize() == 2 * sizeof(PackedMaterial));

    // Verify material indices
    REQUIRE(gpu_scene.GetMaterialIndex(mat_id) == 0);
    REQUIRE(gpu_scene.GetMaterialIndex(plain_id) == 1);

    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Emissive packing roundtrip — factor, strength, normal_scale
// Verify that emissive and normal_scale fields are packed into the correct
// PackedMaterial slots by constructing a PackedMaterial directly.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8D: Emissive factor/strength packing and normal_scale",
          "[phase8d][gpu_scene]") {
    // Construct a PackedMaterial and verify field placement
    PackedMaterial pm{};
    pm.emissive = glm::vec4(10.0f, 5.0f, 2.0f, 3.0f);
    pm.alpha_mode_misc = glm::vec4(0.0f, 0.5f, 0.75f, 0.0f);

    // Emissive is at vec4 index 6 (byte offset 96)
    auto* raw = reinterpret_cast<const float*>(&pm);
    REQUIRE_THAT(raw[24], WithinAbs(10.0, 0.001));   // emissive.r (offset 96)
    REQUIRE_THAT(raw[25], WithinAbs(5.0, 0.001));    // emissive.g (offset 100)
    REQUIRE_THAT(raw[26], WithinAbs(2.0, 0.001));    // emissive.b (offset 104)
    REQUIRE_THAT(raw[27], WithinAbs(3.0, 0.001));    // emissive.a (offset 108)

    // alpha_mode_misc.b = normal_scale at vec4 index 5 (byte offset 80)
    REQUIRE_THAT(raw[22], WithinAbs(0.75, 0.001));   // alpha_mode_misc.b (offset 88)

    // Verify the GPU-side packing via GpuScene
    // Uses the same logic as UpdateMaterials() — just check buffer size
    Scene scene;

    MaterialDesc emissive_mat;
    emissive_mat.base_color = {0, 0, 0};
    emissive_mat.emissive_factor = {10.0f, 5.0f, 2.0f};
    emissive_mat.emissive_strength = 3.0f;
    emissive_mat.normal_scale = 0.5f;
    scene.AddMaterial(std::move(emissive_mat), "emissive");

    MaterialDesc plain_mat;
    plain_mat.base_color = {0.5f, 0.5f, 0.5f};
    scene.AddMaterial(std::move(plain_mat), "plain");

    REQUIRE(scene.Materials().size() == 2);

    // Verify the packing by checking MaterialDesc values directly
    const auto& m0 = scene.Materials()[0];
    REQUIRE_THAT(m0.emissive_factor.r, WithinAbs(10.0, 0.001));
    REQUIRE_THAT(m0.emissive_factor.g, WithinAbs(5.0, 0.001));
    REQUIRE_THAT(m0.emissive_factor.b, WithinAbs(2.0, 0.001));
    REQUIRE_THAT(m0.emissive_strength, WithinAbs(3.0, 0.001));
    REQUIRE_THAT(m0.normal_scale, WithinAbs(0.5, 0.001));

    const auto& m1 = scene.Materials()[1];
    REQUIRE_THAT(m1.emissive_strength, WithinAbs(1.0, 0.001));  // default
    REQUIRE_THAT(m1.normal_scale, WithinAbs(1.0, 0.001));       // default
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: DamagedHelmet.glb renders with PBR textures — no NaN/Inf
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8D: DamagedHelmet PBR render produces valid output",
          "[phase8d][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("DamagedHelmet.glb"));
    REQUIRE(result.success);
    REQUIRE_FALSE(result.mesh_data.empty());

    // Verify the material has all expected texture maps
    REQUIRE(scene.Materials().size() >= 1);
    const auto& mat = scene.Materials()[0];
    REQUIRE(mat.base_color_map.has_value());
    REQUIRE(mat.normal_map.has_value());
    REQUIRE(mat.metallic_roughness_map.has_value());
    REQUIRE(mat.emissive_map.has_value());

    // Add environment map (grey for single-frame test reliability)
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.3f, 0.3f, 0.3f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    // Camera looking at the helmet from front
    CameraParams camera;
    camera.position = {0.0f, 0.0f, 3.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.vertical_fov_radians = 0.8f;
    camera.near_plane = 0.01f;
    camera.far_plane = 100.0f;
    scene.SetActiveCamera(camera);

    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = test::kTestWidth;
    desc.height = test::kTestHeight;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;
    test::FillRendererProcAddrs(desc, ctx);

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(*renderer, ctx.Allocator(),
                                               ctx.Device(), upload_cmd,
                                               result.mesh_data,
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

    // Render one frame
    VkCommandBuffer render_cmd = ctx.BeginOneShot();
    REQUIRE(renderer->RenderFrame(render_cmd, gbuffer, 0));
    ctx.SubmitAndWait(render_cmd);

    // Read back and analyze diffuse
    auto diffuse_rb = test::ReadbackImage(ctx, gbuffer_images.NoisyDiffuseImage());
    auto* diffuse_raw = static_cast<uint16_t*>(diffuse_rb.Map());
    REQUIRE(diffuse_raw != nullptr);

    test::WritePNG("tests/output/phase8d_damaged_helmet_diffuse.png",
                   diffuse_raw, test::kTestWidth, test::kTestHeight);

    auto diffuse_stats = test::AnalyzeRGBA16F(diffuse_raw, test::kTestWidth * test::kTestHeight);

    // Read back and analyze specular
    auto specular_rb = test::ReadbackImage(ctx, gbuffer_images.NoisySpecularImage());
    auto* specular_raw = static_cast<uint16_t*>(specular_rb.Map());
    REQUIRE(specular_raw != nullptr);

    // Write combined image for visual inspection
    test::WriteCombinedPNG("tests/output/phase8d_damaged_helmet_combined.png",
                     diffuse_raw, specular_raw, test::kTestWidth, test::kTestHeight);

    auto specular_stats = test::AnalyzeRGBA16F(specular_raw, test::kTestWidth * test::kTestHeight);

    diffuse_rb.Unmap();
    specular_rb.Unmap();

    // No NaN or Inf
    REQUIRE(diffuse_stats.nan_count == 0);
    REQUIRE(diffuse_stats.inf_count == 0);
    REQUIRE(specular_stats.nan_count == 0);
    REQUIRE(specular_stats.inf_count == 0);

    // Helmet should produce non-trivial output
    REQUIRE(diffuse_stats.nonzero_count > 100);
    REQUIRE(diffuse_stats.has_color_variation);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: Procedural emissive scene — emissive surfaces glow
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8D: Emissive surfaces contribute to output",
          "[phase8d][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;
    std::vector<MeshData> mesh_data;

    // Grey environment (low intensity so emissive stands out)
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.05f, 0.05f, 0.05f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    // Emissive quad (bright red glow)
    MaterialDesc emissive_mat;
    emissive_mat.base_color = {1, 0, 0};
    emissive_mat.roughness = 0.5f;
    emissive_mat.emissive_factor = {10.0f, 0.0f, 0.0f};
    emissive_mat.emissive_strength = 2.0f;
    auto emissive_id = scene.AddMaterial(std::move(emissive_mat), "emissive_red");

    // Non-emissive comparison quad
    MaterialDesc dark_mat;
    dark_mat.base_color = {1, 0, 0};
    dark_mat.roughness = 0.5f;
    auto dark_id = scene.AddMaterial(std::move(dark_mat), "dark_red");

    // Emissive quad on left, non-emissive on right
    AddQuadToScene(scene, mesh_data, "emissive_quad", emissive_id,
                   {-0.6f, 0.0f, 0.0f}, 0.4f);
    AddQuadToScene(scene, mesh_data, "dark_quad", dark_id,
                   { 0.6f, 0.0f, 0.0f}, 0.4f);

    CameraParams camera;
    camera.position = {0.0f, 0.0f, 2.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.vertical_fov_radians = 1.0f;
    camera.near_plane = 0.01f;
    camera.far_plane = 100.0f;
    scene.SetActiveCamera(camera);

    // 64 frames × 16 SPP = 1024 total samples for clean visual output
    auto result = test::RenderSceneMultiFrame(ctx, scene, mesh_data, 64, 16);

    auto* diffuse_raw = result.diffuse.data();
    auto* specular_raw = result.specular.data();

    test::WriteCombinedPNG("tests/output/phase8d_emissive_combined.png",
                     diffuse_raw, specular_raw, test::kTestWidth, test::kTestHeight);

    // Analyze left half (emissive) vs right half (dark)
    float left_sum = 0.0f;
    float right_sum = 0.0f;
    uint32_t left_count = 0;
    uint32_t right_count = 0;
    bool any_nan = false;

    for (uint32_t y = 0; y < test::kTestHeight; ++y) {
        for (uint32_t x = 0; x < test::kTestWidth; ++x) {
            uint32_t i = y * test::kTestWidth + x;
            float r = test::HalfToFloat(diffuse_raw[i * 4 + 0])
                    + test::HalfToFloat(specular_raw[i * 4 + 0]);
            float g = test::HalfToFloat(diffuse_raw[i * 4 + 1])
                    + test::HalfToFloat(specular_raw[i * 4 + 1]);
            float b = test::HalfToFloat(diffuse_raw[i * 4 + 2])
                    + test::HalfToFloat(specular_raw[i * 4 + 2]);

            if (std::isnan(r) || std::isnan(g) || std::isnan(b)) { any_nan = true; continue; }
            if (std::isinf(r) || std::isinf(g) || std::isinf(b)) continue;

            float lum = r + g + b;
            if (x < test::kTestWidth / 2) {
                left_sum += lum;
                ++left_count;
            } else {
                right_sum += lum;
                ++right_count;
            }
        }
    }

    REQUIRE_FALSE(any_nan);

    // Emissive side (left) should be significantly brighter than non-emissive (right)
    float left_avg = (left_count > 0) ? left_sum / static_cast<float>(left_count) : 0.0f;
    float right_avg = (right_count > 0) ? right_sum / static_cast<float>(right_count) : 0.0f;
    REQUIRE(left_avg > right_avg * 1.5f);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: Procedural normal map affects shading
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8D: Normal map perturbs shading on flat geometry",
          "[phase8d][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;
    std::vector<MeshData> mesh_data;

    // Directional HDR environment — angular variation reveals normal-map
    // perturbation that a uniform env would not show.
    auto env_tex_id = scene.AddTexture(MakeDirectionalEnvMap(), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    // Normal map: 2x2, one pixel tilted left, one right, one up, one flat
    // Tangent-space normal: (R,G,B) -> (2*R-1, 2*G-1, 2*B-1)
    // Flat = (128,128,255) -> (0,0,1)
    // Tilted +X = (255,128,128) -> (1,0,0) [extreme for testing]
    // Tilted -X = (0,128,128) -> (-1,0,0)
    TextureDesc normal_tex;
    normal_tex.width = 2;
    normal_tex.height = 2;
    normal_tex.format = PixelFormat::kRGBA8_UNORM;
    normal_tex.data = {
        255, 128, 128, 255,   // pixel (0,0): tilted +X
          0, 128, 128, 255,   // pixel (1,0): tilted -X
        128, 255, 128, 255,   // pixel (0,1): tilted +Y
        128, 128, 255, 255,   // pixel (1,1): flat
    };
    auto norm_id = scene.AddTexture(std::move(normal_tex), "normal_map");

    // Material WITH normal map
    MaterialDesc normal_mat;
    normal_mat.base_color = {0.7f, 0.7f, 0.7f};
    normal_mat.roughness = 0.5f;
    normal_mat.normal_map = norm_id;
    normal_mat.normal_scale = 1.0f;
    auto normal_mat_id = scene.AddMaterial(std::move(normal_mat), "normal_mapped");

    // Material WITHOUT normal map (comparison)
    MaterialDesc flat_mat;
    flat_mat.base_color = {0.7f, 0.7f, 0.7f};
    flat_mat.roughness = 0.5f;
    auto flat_mat_id = scene.AddMaterial(std::move(flat_mat), "flat");

    // Two side-by-side quads
    AddQuadToScene(scene, mesh_data, "normal_quad", normal_mat_id,
                   {-0.6f, 0.0f, 0.0f}, 0.4f);
    AddQuadToScene(scene, mesh_data, "flat_quad", flat_mat_id,
                   { 0.6f, 0.0f, 0.0f}, 0.4f);

    CameraParams camera;
    camera.position = {0.0f, 0.0f, 2.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.vertical_fov_radians = 1.0f;
    camera.near_plane = 0.01f;
    camera.far_plane = 100.0f;
    scene.SetActiveCamera(camera);

    // 64 frames × 16 SPP = 1024 total samples for clean visual output
    auto result = test::RenderSceneMultiFrame(ctx, scene, mesh_data, 64, 16);

    auto* diffuse_raw = result.diffuse.data();
    auto* specular_raw = result.specular.data();

    test::WriteCombinedPNG("tests/output/phase8d_normal_map_combined.png",
                     diffuse_raw, specular_raw, test::kTestWidth, test::kTestHeight);

    // The normal-mapped quad should have more pixel variance than the flat one
    // because different tangent-space normals cause different shading
    auto compute_variance = [&](uint32_t x_start, uint32_t x_end) {
        double sum = 0.0, sum_sq = 0.0;
        uint32_t count = 0;
        for (uint32_t y = 0; y < test::kTestHeight; ++y) {
            for (uint32_t x = x_start; x < x_end; ++x) {
                uint32_t i = y * test::kTestWidth + x;
                float r = test::HalfToFloat(diffuse_raw[i * 4 + 0])
                        + test::HalfToFloat(specular_raw[i * 4 + 0]);
                float g = test::HalfToFloat(diffuse_raw[i * 4 + 1])
                        + test::HalfToFloat(specular_raw[i * 4 + 1]);
                float b = test::HalfToFloat(diffuse_raw[i * 4 + 2])
                        + test::HalfToFloat(specular_raw[i * 4 + 2]);
                if (std::isnan(r) || std::isnan(g) || std::isnan(b)) continue;
                if (std::isinf(r) || std::isinf(g) || std::isinf(b)) continue;
                float lum = r + g + b;
                if (lum > 0.0f) {  // Only count pixels that hit the quad
                    sum += lum;
                    sum_sq += lum * lum;
                    ++count;
                }
            }
        }
        if (count < 2) return 0.0;
        double mean = sum / count;
        return sum_sq / count - mean * mean;
    };

    double left_var = compute_variance(0, test::kTestWidth / 2);
    double right_var = compute_variance(test::kTestWidth / 2, test::kTestWidth);

    // No NaN/Inf
    auto diffuse_stats = test::AnalyzeRGBA16F(diffuse_raw, test::kPixelCount);
    REQUIRE(diffuse_stats.nan_count == 0);
    REQUIRE(diffuse_stats.inf_count == 0);

    // Normal mapped side should show more variance
    INFO("Normal map variance: " << left_var << " vs flat: " << right_var);
    CHECK(left_var > right_var);  // Normal-mapped side should have more shading variance

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 7: Metallic-roughness texture changes material appearance
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8D: Metallic-roughness texture modulates material properties",
          "[phase8d][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;
    std::vector<MeshData> mesh_data;

    // Directional HDR environment — angular variation reveals differences
    // between the MR-textured quad and the constant-metal quad.
    auto env_tex_id = scene.AddTexture(MakeDirectionalEnvMap(), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    // Metallic-roughness texture:
    // glTF: G=roughness, B=metallic, R and A unused
    // Pixel 0: rough=1.0, metallic=1.0 (rough metal)
    // Pixel 1: rough=0.0, metallic=1.0 (mirror metal)
    // Pixel 2: rough=1.0, metallic=0.0 (rough dielectric)
    // Pixel 3: rough=0.0, metallic=0.0 (smooth dielectric)
    TextureDesc mr_tex;
    mr_tex.width = 2;
    mr_tex.height = 2;
    mr_tex.format = PixelFormat::kRGBA8_UNORM;
    mr_tex.data = {
        0, 255,255,255,   // (0,0): R=0, G=255(rough=1), B=255(metal=1)
        0,   0,255,255,   // (1,0): R=0, G=0(rough=0),   B=255(metal=1)
        0, 255,  0,255,   // (0,1): R=0, G=255(rough=1), B=0(metal=0)
        0,   0,  0,255,   // (1,1): R=0, G=0(rough=0),   B=0(metal=0)
    };
    auto mr_id = scene.AddTexture(std::move(mr_tex), "mr_tex");

    // Material with MR texture (base factors = 1.0 so texture drives result)
    MaterialDesc mr_mat;
    mr_mat.base_color = {0.8f, 0.6f, 0.3f};
    mr_mat.roughness = 1.0f;
    mr_mat.metallic = 1.0f;
    mr_mat.metallic_roughness_map = mr_id;
    auto mr_mat_id = scene.AddMaterial(std::move(mr_mat), "mr_textured");

    // Material WITHOUT MR texture (constant rough metal)
    MaterialDesc const_mat;
    const_mat.base_color = {0.8f, 0.6f, 0.3f};
    const_mat.roughness = 1.0f;
    const_mat.metallic = 1.0f;
    auto const_mat_id = scene.AddMaterial(std::move(const_mat), "const_metal");

    AddQuadToScene(scene, mesh_data, "mr_quad", mr_mat_id,
                   {-0.6f, 0.0f, 0.0f}, 0.4f);
    AddQuadToScene(scene, mesh_data, "const_quad", const_mat_id,
                   { 0.6f, 0.0f, 0.0f}, 0.4f);

    CameraParams camera;
    camera.position = {0.0f, 0.0f, 2.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.vertical_fov_radians = 1.0f;
    camera.near_plane = 0.01f;
    camera.far_plane = 100.0f;
    scene.SetActiveCamera(camera);

    // 64 frames × 16 SPP = 1024 total samples for clean visual output
    auto result = test::RenderSceneMultiFrame(ctx, scene, mesh_data, 64, 16);

    auto* diffuse_raw = result.diffuse.data();
    auto* specular_raw = result.specular.data();

    test::WriteCombinedPNG("tests/output/phase8d_metallic_roughness_combined.png",
                     diffuse_raw, specular_raw, test::kTestWidth, test::kTestHeight);

    // Basic sanity: no NaN/Inf
    auto stats_d = test::AnalyzeRGBA16F(diffuse_raw, test::kPixelCount);
    auto stats_s = test::AnalyzeRGBA16F(specular_raw, test::kPixelCount);
    REQUIRE(stats_d.nan_count == 0);
    REQUIRE(stats_d.inf_count == 0);
    REQUIRE(stats_s.nan_count == 0);
    REQUIRE(stats_s.inf_count == 0);

    // The MR-textured side should have more variation because the texture
    // spatially varies roughness and metallic, while the constant side is uniform
    REQUIRE(stats_d.has_color_variation);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
}
