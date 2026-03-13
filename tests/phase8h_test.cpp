#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <volk.h>

#include "test_helpers.h"

#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>
#include <monti/scene/Scene.h>

#include "../renderer/src/vulkan/Buffer.h"
#include "../renderer/src/vulkan/GpuScene.h"

#include <FLIP.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace monti;
using namespace monti::vulkan;

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

constexpr uint32_t kTestWidth = 256;
constexpr uint32_t kTestHeight = 256;

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
    double sum_r = 0, sum_g = 0, sum_b = 0;
    uint32_t valid_count = 0;
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

        stats.sum_r += r;
        stats.sum_g += g;
        stats.sum_b += b;
        ++stats.valid_count;

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
        if (std::isnan(r) || std::isinf(r)) r = 0.0f;
        if (std::isnan(g) || std::isinf(g)) g = 0.0f;
        if (std::isnan(b) || std::isinf(b)) b = 0.0f;
        r = std::max(r, 0.0f) / (1.0f + std::max(r, 0.0f));
        g = std::max(g, 0.0f) / (1.0f + std::max(g, 0.0f));
        b = std::max(b, 0.0f) / (1.0f + std::max(b, 0.0f));
        rgb[i * 3 + 0] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
    return rgb;
}

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

// Build a backlit-leaf scene: a thin quad with diffuse transmission, lit from behind.
// Camera at z=+2 looking at z=0, quad at z=0 facing +Z, area light at z=-1.
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

        // Light-emitting quad behind the translucent surface
        MaterialDesc light_mat;
        light_mat.base_color = {0, 0, 0};
        light_mat.emissive_factor = {5.0f, 5.0f, 5.0f};
        light_mat.emissive_strength = 1.0f;
        auto light_mat_id = scene.AddMaterial(std::move(light_mat), "light");

        AddQuadToScene(scene, mesh_data, "leaf", mat_id, {0, 0, 0}, 1.0f);
        AddQuadToScene(scene, mesh_data, "light", light_mat_id, {0, 0, -1.0f}, 1.0f);

        // Area light matching the emissive quad (for NEE)
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

// Render a scene and return diffuse + specular readback buffers.
struct RenderResult {
    Buffer diffuse_rb;
    Buffer specular_rb;
    std::vector<GpuBuffer> gpu_buffers;
};

RenderResult RenderScene(monti::app::VulkanContext& ctx, Scene& scene,
                         std::vector<MeshData>& mesh_data, uint32_t spp = 64) {
    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = kTestWidth;
    desc.height = kTestHeight;
    desc.samples_per_pixel = spp;
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

    RenderResult result;
    result.diffuse_rb = ReadbackImage(ctx, gbuffer_images.NoisyDiffuseImage());
    result.specular_rb = ReadbackImage(ctx, gbuffer_images.NoisySpecularImage());
    result.gpu_buffers = std::move(gpu_buffers);
    return result;
}

void CleanupRenderResult(VmaAllocator allocator, RenderResult& result) {
    result.diffuse_rb.Unmap();
    result.specular_rb.Unmap();
    for (auto& buf : result.gpu_buffers)
        DestroyGpuBuffer(allocator, buf);
}

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: DiffuseTransmissionBacklitLeaf
//
// A thin quad with dt_factor=0.8, green dt_color, lit from behind.
// Camera sees the front face. The front face should receive green
// illumination via diffuse transmission. Without dt_factor, the front
// face should be significantly darker. FLIP between the two > 0.05.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8H: DiffuseTransmissionBacklitLeaf",
          "[phase8h][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // Scene A: diffuse transmission enabled (leaf-like)
    BacklitScene scene_a;
    scene_a.Build(0.8f, {0.2f, 0.8f, 0.1f});

    auto result_a = RenderScene(ctx, scene_a.scene, scene_a.mesh_data, 64);
    auto* diffuse_a = static_cast<uint16_t*>(result_a.diffuse_rb.Map());
    auto* specular_a = static_cast<uint16_t*>(result_a.specular_rb.Map());
    REQUIRE(diffuse_a != nullptr);
    REQUIRE(specular_a != nullptr);

    WriteCombinedPNG("tests/output/phase8h_backlit_leaf_dt.png",
                     diffuse_a, specular_a, kTestWidth, kTestHeight);

    auto stats_a = AnalyzeRGBA16F(diffuse_a, kTestWidth * kTestHeight);

    // Scene B: no diffuse transmission (opaque leaf)
    BacklitScene scene_b;
    scene_b.Build(0.0f, {0.2f, 0.8f, 0.1f});

    auto result_b = RenderScene(ctx, scene_b.scene, scene_b.mesh_data, 64);
    auto* diffuse_b = static_cast<uint16_t*>(result_b.diffuse_rb.Map());
    auto* specular_b = static_cast<uint16_t*>(result_b.specular_rb.Map());
    REQUIRE(diffuse_b != nullptr);
    REQUIRE(specular_b != nullptr);

    WriteCombinedPNG("tests/output/phase8h_backlit_leaf_opaque.png",
                     diffuse_b, specular_b, kTestWidth, kTestHeight);

    constexpr uint32_t kPixelCount = kTestWidth * kTestHeight;
    auto rgb_a = TonemappedRGB(diffuse_a, specular_a, kPixelCount);
    auto rgb_b = TonemappedRGB(diffuse_b, specular_b, kPixelCount);

    float mean_flip = ComputeMeanFlip(rgb_a, rgb_b,
                                      static_cast<int>(kTestWidth),
                                      static_cast<int>(kTestHeight));

    std::printf("Phase 8H backlit leaf FLIP (dt=0.8 vs dt=0.0): %.4f\n", mean_flip);

    // No NaN/Inf
    REQUIRE(stats_a.nan_count == 0);
    REQUIRE(stats_a.inf_count == 0);

    // The translucent surface should produce non-zero green illumination
    REQUIRE(stats_a.nonzero_count > 100);

    // The two renders should differ significantly
    REQUIRE(mean_flip > 0.05f);

    CleanupRenderResult(ctx.Allocator(), result_a);
    CleanupRenderResult(ctx.Allocator(), result_b);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: ThinSurfaceNoRefraction
//
// A glass panel (transmission_factor=1.0, IOR=1.5) with thin_surface=true
// vs thin_surface=false. With thin_surface=true, there is no IOR refraction
// (straight-through). With thin_surface=false, refraction shifts geometry.
// FLIP between the two > 0.02 confirms the thin-surface flag has effect.
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

        // Glass panel
        MaterialDesc glass;
        glass.base_color = {1, 1, 1};
        glass.roughness = 0.01f;
        glass.metallic = 0.0f;
        glass.transmission_factor = 1.0f;
        glass.ior = 1.5f;
        glass.thin_surface = thin;
        glass.double_sided = true;
        auto glass_id = s.scene.AddMaterial(std::move(glass), "glass");

        // Red wall behind the glass
        MaterialDesc red;
        red.base_color = {0.9f, 0.1f, 0.1f};
        red.roughness = 1.0f;
        red.metallic = 0.0f;
        auto red_id = s.scene.AddMaterial(std::move(red), "red_wall");

        // Green wall beside the glass (visible through refraction shift)
        MaterialDesc green;
        green.base_color = {0.1f, 0.9f, 0.1f};
        green.roughness = 1.0f;
        green.metallic = 0.0f;
        auto green_id = s.scene.AddMaterial(std::move(green), "green_wall");

        AddQuadToScene(s.scene, s.mesh_data, "glass", glass_id, {0, 0, 0}, 1.0f);
        AddQuadToScene(s.scene, s.mesh_data, "red_wall", red_id, {0, 0, -1.5f}, 2.0f);
        AddQuadToScene(s.scene, s.mesh_data, "green_wall", green_id, {2.0f, 0, -0.75f}, 1.0f);

        // Light source
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

    auto scene_thin = build_glass_scene(true);
    auto result_thin = RenderScene(ctx, scene_thin.scene, scene_thin.mesh_data, 64);
    auto* d_thin = static_cast<uint16_t*>(result_thin.diffuse_rb.Map());
    auto* s_thin = static_cast<uint16_t*>(result_thin.specular_rb.Map());
    REQUIRE(d_thin != nullptr);

    WriteCombinedPNG("tests/output/phase8h_thin_surface.png",
                     d_thin, s_thin, kTestWidth, kTestHeight);

    auto scene_thick = build_glass_scene(false);
    auto result_thick = RenderScene(ctx, scene_thick.scene, scene_thick.mesh_data, 64);
    auto* d_thick = static_cast<uint16_t*>(result_thick.diffuse_rb.Map());
    auto* s_thick = static_cast<uint16_t*>(result_thick.specular_rb.Map());
    REQUIRE(d_thick != nullptr);

    WriteCombinedPNG("tests/output/phase8h_thick_surface.png",
                     d_thick, s_thick, kTestWidth, kTestHeight);

    constexpr uint32_t kPixelCount = kTestWidth * kTestHeight;
    auto rgb_thin = TonemappedRGB(d_thin, s_thin, kPixelCount);
    auto rgb_thick = TonemappedRGB(d_thick, s_thick, kPixelCount);

    float mean_flip = ComputeMeanFlip(rgb_thin, rgb_thick,
                                      static_cast<int>(kTestWidth),
                                      static_cast<int>(kTestHeight));

    std::printf("Phase 8H thin vs thick surface FLIP: %.4f\n", mean_flip);
    REQUIRE(mean_flip > 0.02f);

    // No NaN/Inf in either render
    auto stats_thin = AnalyzeRGBA16F(d_thin, kPixelCount);
    auto stats_thick = AnalyzeRGBA16F(d_thick, kPixelCount);
    REQUIRE(stats_thin.nan_count == 0);
    REQUIRE(stats_thin.inf_count == 0);
    REQUIRE(stats_thick.nan_count == 0);
    REQUIRE(stats_thick.inf_count == 0);

    CleanupRenderResult(ctx.Allocator(), result_thin);
    CleanupRenderResult(ctx.Allocator(), result_thick);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: DiffuseTransmissionColorTinting
//
// Quad with dt_factor=0.8, dt_color={1,0,0} (red), base_color white,
// lit from behind by a white area light. The transmitted light on the
// front face should be red-dominant (R >> G and B).
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8H: DiffuseTransmissionColorTinting",
          "[phase8h][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    BacklitScene scene;
    scene.Build(0.8f, {1.0f, 0.0f, 0.0f});

    auto result = RenderScene(ctx, scene.scene, scene.mesh_data, 64);
    auto* diffuse_raw = static_cast<uint16_t*>(result.diffuse_rb.Map());
    auto* specular_raw = static_cast<uint16_t*>(result.specular_rb.Map());
    REQUIRE(diffuse_raw != nullptr);

    WriteCombinedPNG("tests/output/phase8h_red_tint.png",
                     diffuse_raw, specular_raw, kTestWidth, kTestHeight);

    auto stats = AnalyzeRGBA16F(diffuse_raw, kTestWidth * kTestHeight);

    REQUIRE(stats.nan_count == 0);
    REQUIRE(stats.inf_count == 0);
    REQUIRE(stats.nonzero_count > 100);

    // Transmitted light should be red-dominant
    if (stats.valid_count > 0) {
        double avg_r = stats.sum_r / stats.valid_count;
        double avg_g = stats.sum_g / stats.valid_count;
        double avg_b = stats.sum_b / stats.valid_count;

        std::printf("Phase 8H red tint: avg R=%.4f G=%.4f B=%.4f\n",
                    avg_r, avg_g, avg_b);

        // Red channel should be significantly larger than green and blue
        REQUIRE(avg_r > avg_g * 1.5);
        REQUIRE(avg_r > avg_b * 1.5);
    }

    CleanupRenderResult(ctx.Allocator(), result);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: DiffuseTransmissionConvergence
//
// Translucent material at 4 spp vs 64 spp. FLIP below convergence threshold.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8H: DiffuseTransmissionConvergence",
          "[phase8h][renderer][vulkan][integration][flip]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    BacklitScene scene_low;
    scene_low.Build(0.8f, {0.2f, 0.8f, 0.1f});

    auto result_low = RenderScene(ctx, scene_low.scene, scene_low.mesh_data, 4);
    auto* d_low = static_cast<uint16_t*>(result_low.diffuse_rb.Map());
    auto* s_low = static_cast<uint16_t*>(result_low.specular_rb.Map());
    REQUIRE(d_low != nullptr);

    BacklitScene scene_high;
    scene_high.Build(0.8f, {0.2f, 0.8f, 0.1f});

    auto result_high = RenderScene(ctx, scene_high.scene, scene_high.mesh_data, 64);
    auto* d_high = static_cast<uint16_t*>(result_high.diffuse_rb.Map());
    auto* s_high = static_cast<uint16_t*>(result_high.specular_rb.Map());
    REQUIRE(d_high != nullptr);

    constexpr uint32_t kPixelCount = kTestWidth * kTestHeight;
    auto rgb_low = TonemappedRGB(d_low, s_low, kPixelCount);
    auto rgb_high = TonemappedRGB(d_high, s_high, kPixelCount);

    float mean_flip = ComputeMeanFlip(rgb_high, rgb_low,
                                      static_cast<int>(kTestWidth),
                                      static_cast<int>(kTestHeight));

    std::printf("Phase 8H convergence FLIP (4 vs 64 SPP): %.4f\n", mean_flip);

    // Low vs high SPP should differ mainly by MC noise, not structural errors
    REQUIRE(mean_flip < 0.5f);

    // No NaN/Inf at low SPP
    auto stats_low = AnalyzeRGBA16F(d_low, kPixelCount);
    REQUIRE(stats_low.nan_count == 0);
    REQUIRE(stats_low.inf_count == 0);

    CleanupRenderResult(ctx.Allocator(), result_low);
    CleanupRenderResult(ctx.Allocator(), result_high);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: DiffuseTransmissionNoNaN
//
// Render with dt_factor=1.0 (100% transmitted, 0% reflection) — edge case.
// Verify no NaN/Inf in output.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8H: DiffuseTransmissionNoNaN",
          "[phase8h][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    BacklitScene scene;
    scene.Build(1.0f, {1.0f, 1.0f, 1.0f});

    auto result = RenderScene(ctx, scene.scene, scene.mesh_data, 64);
    auto* diffuse_raw = static_cast<uint16_t*>(result.diffuse_rb.Map());
    auto* specular_raw = static_cast<uint16_t*>(result.specular_rb.Map());
    REQUIRE(diffuse_raw != nullptr);
    REQUIRE(specular_raw != nullptr);

    WriteCombinedPNG("tests/output/phase8h_no_nan.png",
                     diffuse_raw, specular_raw, kTestWidth, kTestHeight);

    constexpr uint32_t kPixelCount = kTestWidth * kTestHeight;
    auto d_stats = AnalyzeRGBA16F(diffuse_raw, kPixelCount);
    auto s_stats = AnalyzeRGBA16F(specular_raw, kPixelCount);

    REQUIRE(d_stats.nan_count == 0);
    REQUIRE(d_stats.inf_count == 0);
    REQUIRE(s_stats.nan_count == 0);
    REQUIRE(s_stats.inf_count == 0);

    CleanupRenderResult(ctx.Allocator(), result);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: SpecularPlusDiffuseTransmission
//
// Panel with both transmission_factor=0.5 (specular transmission) and
// dt_factor=0.6, thin_surface=true. Verify no NaN/Inf and that output
// differs from a panel with only specular transmission (dt_factor=0.0).
// FLIP > 0.02 confirms both lobes contribute independently.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8H: SpecularPlusDiffuseTransmission",
          "[phase8h][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    // Scene A: both specular + diffuse transmission
    BacklitScene scene_a;
    scene_a.Build(0.6f, {0.5f, 0.8f, 0.3f}, {1, 1, 1}, 0.5f, true);

    auto result_a = RenderScene(ctx, scene_a.scene, scene_a.mesh_data, 64);
    auto* d_a = static_cast<uint16_t*>(result_a.diffuse_rb.Map());
    auto* s_a = static_cast<uint16_t*>(result_a.specular_rb.Map());
    REQUIRE(d_a != nullptr);

    WriteCombinedPNG("tests/output/phase8h_spec_plus_dt.png",
                     d_a, s_a, kTestWidth, kTestHeight);

    // Scene B: only specular transmission (no diffuse transmission)
    BacklitScene scene_b;
    scene_b.Build(0.0f, {0.5f, 0.8f, 0.3f}, {1, 1, 1}, 0.5f, true);

    auto result_b = RenderScene(ctx, scene_b.scene, scene_b.mesh_data, 64);
    auto* d_b = static_cast<uint16_t*>(result_b.diffuse_rb.Map());
    auto* s_b = static_cast<uint16_t*>(result_b.specular_rb.Map());
    REQUIRE(d_b != nullptr);

    WriteCombinedPNG("tests/output/phase8h_spec_only.png",
                     d_b, s_b, kTestWidth, kTestHeight);

    constexpr uint32_t kPixelCount = kTestWidth * kTestHeight;

    // No NaN/Inf in either render
    auto stats_a = AnalyzeRGBA16F(d_a, kPixelCount);
    auto stats_b = AnalyzeRGBA16F(d_b, kPixelCount);
    REQUIRE(stats_a.nan_count == 0);
    REQUIRE(stats_a.inf_count == 0);
    REQUIRE(stats_b.nan_count == 0);
    REQUIRE(stats_b.inf_count == 0);

    auto rgb_a = TonemappedRGB(d_a, s_a, kPixelCount);
    auto rgb_b = TonemappedRGB(d_b, s_b, kPixelCount);

    float mean_flip = ComputeMeanFlip(rgb_a, rgb_b,
                                      static_cast<int>(kTestWidth),
                                      static_cast<int>(kTestHeight));

    std::printf("Phase 8H specular+dt vs specular-only FLIP: %.4f\n", mean_flip);
    REQUIRE(mean_flip > 0.02f);

    CleanupRenderResult(ctx.Allocator(), result_a);
    CleanupRenderResult(ctx.Allocator(), result_b);
    ctx.WaitIdle();
}
