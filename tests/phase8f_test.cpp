#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <volk.h>

#include "test_helpers.h"
#include "scenes/CornellBox.h"

#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>
#include <monti/scene/Scene.h>

#include "gltf/GltfLoader.h"

#include "../renderer/src/vulkan/Buffer.h"
#include "../renderer/src/vulkan/GpuScene.h"

#include <FLIP.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace monti;
using namespace monti::vulkan;
using Catch::Matchers::WithinAbs;

namespace {

struct TestContext {
    monti::app::VulkanContext ctx;

    bool Init() {
        if (!ctx.CreateInstance()) return false;
        if (!ctx.CreateDevice(std::nullopt)) return false;
        return true;
    }
};

#ifndef MONTI_SHADER_SPV_DIR
#define MONTI_SHADER_SPV_DIR "build/shaders"
#endif

#ifndef MONTI_TEST_ASSETS_DIR
#define MONTI_TEST_ASSETS_DIR "tests/assets"
#endif

constexpr uint32_t kTestWidth = 256;
constexpr uint32_t kTestHeight = 256;

static std::string AssetPath(const char* filename) {
    return std::string(MONTI_TEST_ASSETS_DIR) + "/" + filename;
}

Buffer ReadbackImage(monti::app::VulkanContext& ctx, VkImage image,
                     VkDeviceSize pixel_size = 8) {
    VkDeviceSize readback_size = kTestWidth * kTestHeight * pixel_size;

    Buffer readback;
    readback.Create(ctx.Allocator(), readback_size,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_CPU_ONLY);

    VkCommandBuffer copy_cmd = ctx.BeginOneShot();

    VkImageMemoryBarrier2 to_src{};
    to_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_src.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    to_src.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    to_src.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    to_src.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    to_src.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_src.image = image;
    to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &to_src;
    vkCmdPipelineBarrier2(copy_cmd, &dep);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {kTestWidth, kTestHeight, 1};
    vkCmdCopyImageToBuffer(copy_cmd, image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.Handle(), 1, &region);

    ctx.SubmitAndWait(copy_cmd);
    return readback;
}

struct PixelStats {
    uint32_t nan_count = 0;
    uint32_t inf_count = 0;
    uint32_t nonzero_count = 0;
    bool has_color_variation = false;
};

PixelStats AnalyzeRGBA16F(const uint16_t* raw, uint32_t pixel_count) {
    PixelStats stats{};
    float prev_r = -1.0f;
    for (uint32_t i = 0; i < pixel_count; ++i) {
        float r = test::HalfToFloat(raw[i * 4 + 0]);
        float g = test::HalfToFloat(raw[i * 4 + 1]);
        float b = test::HalfToFloat(raw[i * 4 + 2]);

        if (std::isnan(r) || std::isnan(g) || std::isnan(b)) { ++stats.nan_count; continue; }
        if (std::isinf(r) || std::isinf(g) || std::isinf(b)) { ++stats.inf_count; continue; }

        if (r + g + b > 0.0f) ++stats.nonzero_count;
        if (prev_r >= 0.0f && std::abs(r - prev_r) > 0.001f)
            stats.has_color_variation = true;
        prev_r = r;
    }
    return stats;
}

bool WriteCombinedPNG(std::string_view path,
                      const uint16_t* diffuse_raw, const uint16_t* specular_raw,
                      uint32_t width, uint32_t height) {
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());
    std::vector<uint8_t> pixels(width * height * 3);
    for (uint32_t i = 0; i < width * height; ++i) {
        float r = test::HalfToFloat(diffuse_raw[i * 4 + 0])
                + test::HalfToFloat(specular_raw[i * 4 + 0]);
        float g = test::HalfToFloat(diffuse_raw[i * 4 + 1])
                + test::HalfToFloat(specular_raw[i * 4 + 1]);
        float b = test::HalfToFloat(diffuse_raw[i * 4 + 2])
                + test::HalfToFloat(specular_raw[i * 4 + 2]);
        r = r / (1.0f + r);
        g = g / (1.0f + g);
        b = b / (1.0f + b);
        r = std::pow(std::max(r, 0.0f), 1.0f / 2.2f);
        g = std::pow(std::max(g, 0.0f), 1.0f / 2.2f);
        b = std::pow(std::max(b, 0.0f), 1.0f / 2.2f);
        pixels[i * 3 + 0] = static_cast<uint8_t>(std::clamp(r * 255.0f + 0.5f, 0.0f, 255.0f));
        pixels[i * 3 + 1] = static_cast<uint8_t>(std::clamp(g * 255.0f + 0.5f, 0.0f, 255.0f));
        pixels[i * 3 + 2] = static_cast<uint8_t>(std::clamp(b * 255.0f + 0.5f, 0.0f, 255.0f));
    }
    std::string path_str(path);
    return stbi_write_png(path_str.c_str(), static_cast<int>(width),
                          static_cast<int>(height), 3, pixels.data(),
                          static_cast<int>(width * 3)) != 0;
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

// Convert RGBA16F diffuse + specular readback to interleaved linear RGB floats
// for FLIP.  Applies Reinhard tone-mapping so values fall in [0,1].
std::vector<float> TonemappedRGB(const uint16_t* diffuse_raw,
                                 const uint16_t* specular_raw,
                                 uint32_t pixel_count) {
    std::vector<float> rgb(pixel_count * 3);
    for (uint32_t i = 0; i < pixel_count; ++i) {
        float r = test::HalfToFloat(diffuse_raw[i * 4 + 0])
                + test::HalfToFloat(specular_raw[i * 4 + 0]);
        float g = test::HalfToFloat(diffuse_raw[i * 4 + 1])
                + test::HalfToFloat(specular_raw[i * 4 + 1]);
        float b = test::HalfToFloat(diffuse_raw[i * 4 + 2])
                + test::HalfToFloat(specular_raw[i * 4 + 2]);
        // Clamp NaN/Inf to zero
        if (std::isnan(r) || std::isinf(r)) r = 0.0f;
        if (std::isnan(g) || std::isinf(g)) g = 0.0f;
        if (std::isnan(b) || std::isinf(b)) b = 0.0f;
        // Reinhard tone-map to [0,1]
        r = std::max(r, 0.0f) / (1.0f + std::max(r, 0.0f));
        g = std::max(g, 0.0f) / (1.0f + std::max(g, 0.0f));
        b = std::max(b, 0.0f) / (1.0f + std::max(b, 0.0f));
        rgb[i * 3 + 0] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
    return rgb;
}

// Compute mean FLIP score between two linear RGB float images in [0,1].
float ComputeMeanFlip(const std::vector<float>& reference_rgb,
                      const std::vector<float>& test_rgb,
                      int width, int height) {
    FLIP::image<FLIP::color3> ref_img(width, height);
    FLIP::image<FLIP::color3> test_img(width, height);
    FLIP::image<float> error_map(width, height, 0.0f);

    ref_img.setPixels(reference_rgb.data(), width, height);
    test_img.setPixels(test_rgb.data(), width, height);

    FLIP::Parameters params;
    FLIP::evaluate(ref_img, test_img, false, params, error_map);

    float sum = 0.0f;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            sum += error_map.get(x, y);
    return sum / static_cast<float>(width * height);
}

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: DamagedHelmet renders with valid output through textureLod
//
// Exercises all 5 textureLod sampling sites (base color, metallic-roughness,
// normal map, transmission, emissive) via the ray cone LOD computation.
// The LOD math runs on real geometry; any NaN or invalid LOD would propagate
// through textureLod and corrupt the output.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8F: DamagedHelmet renders valid output via textureLod",
          "[phase8f][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("DamagedHelmet.glb"));
    REQUIRE(result.success);
    REQUIRE_FALSE(result.mesh_data.empty());

    // Verify the material has texture maps that exercise textureLod
    REQUIRE(scene.Materials().size() >= 1);
    const auto& mat = scene.Materials()[0];
    REQUIRE(mat.base_color_map.has_value());
    REQUIRE(mat.normal_map.has_value());
    REQUIRE(mat.metallic_roughness_map.has_value());
    REQUIRE(mat.emissive_map.has_value());

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

    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = kTestWidth;
    desc.height = kTestHeight;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(*renderer, ctx.Allocator(),
                                               ctx.Device(), upload_cmd,
                                               result.mesh_data);
    REQUIRE_FALSE(gpu_buffers.empty());
    ctx.SubmitAndWait(upload_cmd);

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                                  kTestWidth, kTestHeight, gbuf_cmd,
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    VkCommandBuffer render_cmd = ctx.BeginOneShot();
    REQUIRE(renderer->RenderFrame(render_cmd, gbuffer, 0));
    ctx.SubmitAndWait(render_cmd);

    auto diffuse_rb = ReadbackImage(ctx, gbuffer_images.NoisyDiffuseImage());
    auto* diffuse_raw = static_cast<uint16_t*>(diffuse_rb.Map());
    REQUIRE(diffuse_raw != nullptr);

    auto specular_rb = ReadbackImage(ctx, gbuffer_images.NoisySpecularImage());
    auto* specular_raw = static_cast<uint16_t*>(specular_rb.Map());
    REQUIRE(specular_raw != nullptr);

    WriteCombinedPNG("tests/output/damaged_helmet_8f_combined.png",
                     diffuse_raw, specular_raw, kTestWidth, kTestHeight);

    auto diffuse_stats = AnalyzeRGBA16F(diffuse_raw, kTestWidth * kTestHeight);
    auto specular_stats = AnalyzeRGBA16F(specular_raw, kTestWidth * kTestHeight);

    diffuse_rb.Unmap();
    specular_rb.Unmap();

    // Ray cone LOD math must not produce NaN or Inf
    REQUIRE(diffuse_stats.nan_count == 0);
    REQUIRE(diffuse_stats.inf_count == 0);
    REQUIRE(specular_stats.nan_count == 0);
    REQUIRE(specular_stats.inf_count == 0);

    // Textured helmet should produce non-trivial, colorful output
    REQUIRE(diffuse_stats.nonzero_count > 100);
    REQUIRE(diffuse_stats.has_color_variation);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Emissive ground plane — LOD increases with distance
//
// A single ground plane with a high-contrast checkerboard emissive texture
// extends from the camera into the distance.  Emissive at primary hit is
// deterministic (no stochastic BRDF sampling), so the output cleanly shows
// the texture LOD effect.  The bottom half of the image hits nearby ground
// (small cone_width → low LOD → sharp checkerboard → high color variance)
// while the top half hits distant ground (large cone_width + grazing angle
// → high LOD → averaged mip → low variance).
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8F: Emissive ground plane LOD increases with distance",
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

    // 256x256 red/green checkerboard with full mip chain (9 levels, max LOD 4)
    auto checker_tex_id = scene.AddTexture(
        MakeCheckerboard256({255, 0, 0, 255}, {0, 255, 0, 255}), "checker");

    // Emissive-only material with checkerboard texture on the emissive channel.
    // base_color is black so diffuse/specular BRDF contributes nothing.
    MaterialDesc emissive_mat;
    emissive_mat.base_color = {0.0f, 0.0f, 0.0f};
    emissive_mat.roughness = 1.0f;
    emissive_mat.metallic = 0.0f;
    emissive_mat.emissive_factor = {1.0f, 1.0f, 1.0f};
    emissive_mat.emissive_strength = 1.0f;
    emissive_mat.emissive_map = checker_tex_id;
    auto mat_id = scene.AddMaterial(std::move(emissive_mat), "emissive_checker");

    // Ground plane extending from z=1 (near) to z=-60 (far), wide enough to
    // fill the screen.  UVs tile 8 times so the checkerboard repeats.
    auto ground_md = MakeGroundPlane(30.0f, 1.0f, -60.0f, 8.0f);
    Mesh mesh_desc;
    mesh_desc.name = "ground";
    mesh_desc.vertex_count = static_cast<uint32_t>(ground_md.vertices.size());
    mesh_desc.index_count = static_cast<uint32_t>(ground_md.indices.size());
    mesh_desc.bbox_min = {-30.0f, 0.0f, -60.0f};
    mesh_desc.bbox_max = { 30.0f, 0.0f,   1.0f};
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
    desc.width = kTestWidth;
    desc.height = kTestHeight;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(*renderer, ctx.Allocator(),
                                               ctx.Device(), upload_cmd,
                                               mesh_data);
    REQUIRE_FALSE(gpu_buffers.empty());
    ctx.SubmitAndWait(upload_cmd);

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                                  kTestWidth, kTestHeight, gbuf_cmd,
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    VkCommandBuffer render_cmd = ctx.BeginOneShot();
    REQUIRE(renderer->RenderFrame(render_cmd, gbuffer, 0));
    ctx.SubmitAndWait(render_cmd);

    auto diffuse_rb = ReadbackImage(ctx, gbuffer_images.NoisyDiffuseImage());
    auto* diffuse_raw = static_cast<uint16_t*>(diffuse_rb.Map());
    REQUIRE(diffuse_raw != nullptr);

    auto specular_rb = ReadbackImage(ctx, gbuffer_images.NoisySpecularImage());
    auto* specular_raw = static_cast<uint16_t*>(specular_rb.Map());
    REQUIRE(specular_raw != nullptr);

    WriteCombinedPNG("tests/output/ground_lod_8f_combined.png",
                     diffuse_raw, specular_raw, kTestWidth, kTestHeight);

    // No NaN or Inf from the ray cone LOD computation
    auto stats = AnalyzeRGBA16F(diffuse_raw, kTestWidth * kTestHeight);
    REQUIRE(stats.nan_count == 0);
    REQUIRE(stats.inf_count == 0);

    // Emissive ground plane should produce visible output
    REQUIRE(stats.nonzero_count > 100);

    // Compare color variance between bottom quarter (near ground, low LOD)
    // and top quarter (far ground, high LOD).  The bottom quarter hits the
    // ground close to the camera where the cone is narrow → low mip → sharp
    // checkerboard pattern → high pixel-to-pixel variance.  The top quarter
    // hits distant ground where the cone is wide → high mip → averaged
    // texture → low variance.
    uint32_t quarter = kTestHeight / 4;
    float near_variance = RegionColorVariance(
        diffuse_raw, kTestWidth,
        0, kTestHeight - quarter, kTestWidth, kTestHeight);
    float far_variance = RegionColorVariance(
        diffuse_raw, kTestWidth,
        0, 0, kTestWidth, quarter);

    // Near ground should show more texture detail than far ground
    REQUIRE(near_variance > far_variance);

    diffuse_rb.Unmap();
    specular_rb.Unmap();

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: FLIP convergence — low SPP vs high SPP on DamagedHelmet
//
// Renders the DamagedHelmet at 4 SPP and 64 SPP (same frame_index, same
// jitter).  Both renders exercise the full textureLod / ray cone LOD
// pipeline on real PBR textures.  If the LOD math is correct, the high-SPP
// image is a less-noisy version of the same scene; FLIP between them should
// be moderate (dominated by MC noise, not structural errors).  If the LOD
// math is broken (NaN textureLod, wrong mip selection), the images diverge
// and FLIP exceeds the threshold.
//
// This is a self-consistency test: no stored reference images required.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8F: FLIP convergence DamagedHelmet 4 vs 64 SPP",
          "[phase8f][renderer][vulkan][integration][flip]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("DamagedHelmet.glb"));
    REQUIRE(result.success);
    REQUIRE_FALSE(result.mesh_data.empty());

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

    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = kTestWidth;
    desc.height = kTestHeight;
    desc.samples_per_pixel = 4;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(*renderer, ctx.Allocator(),
                                               ctx.Device(), upload_cmd,
                                               result.mesh_data);
    REQUIRE_FALSE(gpu_buffers.empty());
    ctx.SubmitAndWait(upload_cmd);

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                                  kTestWidth, kTestHeight, gbuf_cmd,
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    // ── Render at low SPP (4) ──
    renderer->SetSamplesPerPixel(4);
    VkCommandBuffer low_cmd = ctx.BeginOneShot();
    REQUIRE(renderer->RenderFrame(low_cmd, gbuffer, 0));
    ctx.SubmitAndWait(low_cmd);

    auto low_diffuse_rb = ReadbackImage(ctx, gbuffer_images.NoisyDiffuseImage());
    auto low_specular_rb = ReadbackImage(ctx, gbuffer_images.NoisySpecularImage());
    auto* low_diffuse = static_cast<uint16_t*>(low_diffuse_rb.Map());
    auto* low_specular = static_cast<uint16_t*>(low_specular_rb.Map());
    REQUIRE(low_diffuse != nullptr);
    REQUIRE(low_specular != nullptr);

    constexpr uint32_t kPixelCount = kTestWidth * kTestHeight;
    auto low_rgb = TonemappedRGB(low_diffuse, low_specular, kPixelCount);

    WriteCombinedPNG("tests/output/helmet_8f_low_spp.png",
                     low_diffuse, low_specular, kTestWidth, kTestHeight);

    low_diffuse_rb.Unmap();
    low_specular_rb.Unmap();

    // ── Render at high SPP (64) ──
    renderer->SetSamplesPerPixel(64);
    VkCommandBuffer high_cmd = ctx.BeginOneShot();
    REQUIRE(renderer->RenderFrame(high_cmd, gbuffer, 0));
    ctx.SubmitAndWait(high_cmd);

    auto high_diffuse_rb = ReadbackImage(ctx, gbuffer_images.NoisyDiffuseImage());
    auto high_specular_rb = ReadbackImage(ctx, gbuffer_images.NoisySpecularImage());
    auto* high_diffuse = static_cast<uint16_t*>(high_diffuse_rb.Map());
    auto* high_specular = static_cast<uint16_t*>(high_specular_rb.Map());
    REQUIRE(high_diffuse != nullptr);
    REQUIRE(high_specular != nullptr);

    WriteCombinedPNG("tests/output/helmet_8f_high_spp.png",
                     high_diffuse, high_specular, kTestWidth, kTestHeight);

    auto high_rgb = TonemappedRGB(high_diffuse, high_specular, kPixelCount);

    high_diffuse_rb.Unmap();
    high_specular_rb.Unmap();

    // ── Compute FLIP ──
    float mean_flip = ComputeMeanFlip(high_rgb, low_rgb,
                                      static_cast<int>(kTestWidth),
                                      static_cast<int>(kTestHeight));

    std::printf("Phase 8F FLIP convergence: mean=%.4f (4 vs 64 SPP)\n",
                mean_flip);

    // Threshold: 4 vs 64 SPP differs mainly by MC noise.  A working renderer
    // with correct textureLod should score well below 0.5.  Broken LOD math
    // (NaN, black textures, wrong mips) would push this much higher.
    REQUIRE(mean_flip < 0.5f);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}
