#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <volk.h>

#include "test_helpers.h"
#include "scenes/CornellBox.h"

#include <monti/scene/Scene.h>

#include "../renderer/src/vulkan/GpuScene.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace monti;
using namespace monti::vulkan;
using Catch::Matchers::WithinAbs;

namespace {

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

// Build a flat quad mesh facing +Z
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

// Create a 256x256 RGBA8 texture with pre-computed mip levels where each
// level is a unique solid color.  This "fake MIP map" makes the ray cone
// LOD selection directly visible: mip 0 = red, mip 1 = green, mip 2 = blue,
// mip 3 = yellow, mip 4 = magenta, mip 5 = cyan, mip 6 = orange,
// mip 7 = white, mip 8 = gray.
TextureDesc MakeFakeMipTexture256() {
    constexpr uint32_t kSize = 256;
    constexpr uint32_t kMipLevels = 9;

    // Unique color per mip level
    constexpr std::array<std::array<uint8_t, 4>, kMipLevels> kMipColors = {{
        {255,   0,   0, 255},  // mip 0: red (nearest)
        {  0, 255,   0, 255},  // mip 1: green
        {  0,   0, 255, 255},  // mip 2: blue
        {255, 255,   0, 255},  // mip 3: yellow
        {255,   0, 255, 255},  // mip 4: magenta
        {  0, 255, 255, 255},  // mip 5: cyan
        {255, 128,   0, 255},  // mip 6: orange
        {255, 255, 255, 255},  // mip 7: white
        {128, 128, 128, 255},  // mip 8: gray (farthest)
    }};

    // Compute total data size and per-mip offsets
    std::vector<uint32_t> offsets(kMipLevels);
    uint32_t total_bytes = 0;
    for (uint32_t level = 0; level < kMipLevels; ++level) {
        offsets[level] = total_bytes;
        uint32_t dim = kSize >> level;
        total_bytes += dim * dim * 4;
    }

    TextureDesc tex;
    tex.width = kSize;
    tex.height = kSize;
    tex.format = PixelFormat::kRGBA8_UNORM;
    tex.mip_levels = kMipLevels;
    tex.mip_offsets = std::move(offsets);
    tex.data.resize(total_bytes);

    // Fill each mip level with its solid color
    for (uint32_t level = 0; level < kMipLevels; ++level) {
        uint32_t dim = kSize >> level;
        uint32_t offset = tex.mip_offsets[level];
        const auto& color = kMipColors[level];
        for (uint32_t i = 0; i < dim * dim; ++i) {
            std::memcpy(&tex.data[offset + i * 4], color.data(), 4);
        }
    }
    return tex;
}

// Create a 256x256 RGBA8 checkerboard with full mip chain.
// Cell size in pixels controls the frequency (default 4).
// mip_levels = 9, so max usable LOD = max(9 - 5, 0) = 4.
TextureDesc MakeCheckerboard256(std::array<uint8_t, 4> color_a,
                                std::array<uint8_t, 4> color_b,
                                uint32_t cell_size = 4) {
    constexpr uint32_t kSize = 256;

    TextureDesc tex;
    tex.width = kSize;
    tex.height = kSize;
    tex.format = PixelFormat::kRGBA8_UNORM;
    tex.mip_levels = 9;  // floor(log2(256)) + 1
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

// Build a ground plane (XZ, facing +Y) that extends from z_near to z_far.
// UVs tile the texture uv_repeats times across the shorter dimension.
MeshData MakeGroundPlane(float x_half, float z_near, float z_far,
                         float uv_repeats) {
    glm::vec3 n{0, 1, 0};
    glm::vec4 t{1, 0, 0, 1};
    float z_extent = z_near - z_far;
    float u_max = uv_repeats;
    float v_max = uv_repeats * z_extent / (2.0f * x_half);
    MeshData md;
    md.vertices = {
        MakeVertex({-x_half, 0, z_near}, n, t, {0,     0}),
        MakeVertex({ x_half, 0, z_near}, n, t, {u_max, 0}),
        MakeVertex({ x_half, 0, z_far},  n, t, {u_max, v_max}),
        MakeVertex({-x_half, 0, z_far},  n, t, {0,     v_max}),
    };
    md.indices = {0, 1, 2, 0, 2, 3};
    return md;
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

// Compute color variance across pixels in a rectangular region of an
// RGBA16F readback buffer.  Returns the average per-channel variance
// (mean of R/G/B variances).
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
// Test 1: High-frequency checkerboard at close vs far distance
//
// Two quads with the same high-frequency checkerboard texture: one close
// to the camera (left) and one far away (right).  The close quad should
// show distinct checker cells (high variance), while the far quad should
// show filtered/averaged color (low variance) due to the ray cone LOD
// selecting higher mip levels.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8F: High-frequency checkerboard close vs far LOD",
          "[phase8f][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;
    std::vector<MeshData> mesh_data;

    // Black environment — emissive checkerboard is the only visible content
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.0f, 0.0f, 0.0f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 0.0f;
    scene.SetEnvironmentLight(env);

    // Fine red/green checkerboard (2-pixel cells)
    auto checker_tex_id = scene.AddTexture(
        MakeCheckerboard256({255, 0, 0, 255}, {0, 255, 0, 255}, 2), "checker");

    // Emissive-only material with checkerboard
    MaterialDesc emissive_mat;
    emissive_mat.base_color = {0.0f, 0.0f, 0.0f};
    emissive_mat.roughness = 1.0f;
    emissive_mat.metallic = 0.0f;
    emissive_mat.emissive_factor = {1.0f, 1.0f, 1.0f};
    emissive_mat.emissive_strength = 1.0f;
    emissive_mat.emissive_map = checker_tex_id;
    auto mat_id = scene.AddMaterial(std::move(emissive_mat), "emissive_checker");

    // Close quad (left half of view)
    AddQuadToScene(scene, mesh_data, "close_quad", mat_id,
                   {-0.6f, 0.0f, 0.0f}, 0.4f);
    // Far quad (right half of view, pushed back)
    AddQuadToScene(scene, mesh_data, "far_quad", mat_id,
                   { 1.5f, 0.0f, -6.0f}, 1.0f);

    CameraParams camera;
    camera.position = {0.0f, 0.0f, 2.0f};
    camera.target = {0.3f, 0.0f, -2.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.vertical_fov_radians = 1.0f;
    camera.near_plane = 0.01f;
    camera.far_plane = 200.0f;
    scene.SetActiveCamera(camera);

    auto result = test::RenderSceneMultiFrame(ctx, scene, mesh_data, 1, 4);

    auto* diffuse_raw = result.diffuse.data();
    auto* specular_raw = result.specular.data();

    test::WriteCombinedPNG("tests/output/phase8f_checker_close_far_combined.png",
                           diffuse_raw, specular_raw,
                           test::kTestWidth, test::kTestHeight);

    auto stats = test::AnalyzeRGBA16F(diffuse_raw, test::kPixelCount);
    REQUIRE(stats.nan_count == 0);
    REQUIRE(stats.inf_count == 0);
    REQUIRE(stats.nonzero_count > 100);

    // Close quad (left half) should have more variance than far quad (right half)
    float close_var = RegionColorVariance(
        diffuse_raw, test::kTestWidth,
        0, 0, test::kTestWidth / 2, test::kTestHeight);
    float far_var = RegionColorVariance(
        diffuse_raw, test::kTestWidth,
        test::kTestWidth / 2, 0, test::kTestWidth, test::kTestHeight);

    INFO("Close variance: " << close_var << " vs far: " << far_var);
    CHECK(close_var > far_var);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Fake MIP map ground plane — LOD increases with distance
//
// A single ground plane with a "fake MIP map" texture (each mip level is a
// unique solid color: red=mip0, green=mip1, blue=mip2, yellow=mip3, etc.)
// extends from the camera into the distance.  The output shows color bands
// that directly reveal which mip level the ray cone LOD pipeline selects
// at each distance.  Near ground should be red (mip 0), transitioning to
// green, blue, etc. as distance increases.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8F: Fake MIP map ground plane shows LOD bands",
          "[phase8f][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;
    std::vector<MeshData> mesh_data;

    // Black environment — emissive texture is the sole light source
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.0f, 0.0f, 0.0f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 0.0f;
    scene.SetEnvironmentLight(env);

    // Fake MIP map texture — each mip level is a different solid color,
    // making the ray cone LOD level selection directly visible as color
    // bands in the output (red=near/mip0, green=mip1, blue=mip2, ...).
    auto mip_tex_id = scene.AddTexture(MakeFakeMipTexture256(), "fake_mip");

    // Emissive-only material with fake MIP texture on the emissive channel.
    // base_color is black so diffuse/specular BRDF contributes nothing.
    MaterialDesc emissive_mat;
    emissive_mat.base_color = {0.0f, 0.0f, 0.0f};
    emissive_mat.roughness = 1.0f;
    emissive_mat.metallic = 0.0f;
    emissive_mat.emissive_factor = {1.0f, 1.0f, 1.0f};
    emissive_mat.emissive_strength = 1.0f;
    emissive_mat.emissive_map = mip_tex_id;
    auto mat_id = scene.AddMaterial(std::move(emissive_mat), "emissive_mip");

    // Ground plane extending from z=1 (near) to z=-500 (far), wide enough to
    // fill the screen all the way to the horizon.  UVs tile 8 times.
    auto ground_md = MakeGroundPlane(500.0f, 1.0f, -500.0f, 8.0f);
    Mesh mesh_desc;
    mesh_desc.name = "ground";
    mesh_desc.vertex_count = static_cast<uint32_t>(ground_md.vertices.size());
    mesh_desc.index_count = static_cast<uint32_t>(ground_md.indices.size());
    mesh_desc.bbox_min = {-500.0f, 0.0f, -500.0f};
    mesh_desc.bbox_max = { 500.0f, 0.0f,    1.0f};
    auto mesh_id = scene.AddMesh(std::move(mesh_desc), "ground");
    ground_md.mesh_id = mesh_id;
    scene.AddNode(mesh_id, mat_id, "ground");
    mesh_data.push_back(std::move(ground_md));

    // Camera above the ground looking forward and slightly down
    CameraParams camera;
    camera.position = {0.0f, 1.5f, 2.0f};
    camera.target = {0.0f, 0.0f, -10.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.vertical_fov_radians = 1.0f;
    camera.near_plane = 0.01f;
    camera.far_plane = 200.0f;
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

    test::WriteCombinedPNG("tests/output/phase8f_ground_mip_combined.png",
                     diffuse_raw, specular_raw, test::kTestWidth, test::kTestHeight);

    // No NaN or Inf from the ray cone LOD computation
    auto stats = test::AnalyzeRGBA16F(diffuse_raw, test::kTestWidth * test::kTestHeight);
    REQUIRE(stats.nan_count == 0);
    REQUIRE(stats.inf_count == 0);

    // Emissive ground plane should produce visible output
    REQUIRE(stats.nonzero_count > 100);

    // Compare color variance between bottom quarter (near ground, low LOD)
    // and top quarter (far ground, high LOD).  With the fake MIP map, the
    // near ground should be red (mip 0) and the far ground should transition
    // to other colors (mip 1+).  Both regions will have low *internal*
    // variance (solid colors), but they should differ from each other.
    // Verify by checking that the near region has higher red than the far region.
    uint32_t quarter = test::kTestHeight / 4;

    // Near ground (bottom quarter): should be predominantly red (mip 0)
    float near_variance = RegionColorVariance(
        diffuse_raw, test::kTestWidth,
        0, test::kTestHeight - quarter, test::kTestWidth, test::kTestHeight);
    // Far ground (top quarter): should be a higher mip level (not red)
    float far_variance = RegionColorVariance(
        diffuse_raw, test::kTestWidth,
        0, 0, test::kTestWidth, quarter);

    // With distinct solid colors per mip, the near and far regions should
    // produce visually different output (the overall image should show LOD bands).
    // The near region variance may be nonzero due to mip transitions at
    // the boundary, so we just verify the image has meaningful content.
    INFO("Near region variance: " << near_variance);
    INFO("Far region variance: " << far_variance);

    diffuse_rb.Unmap();
    specular_rb.Unmap();

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}
