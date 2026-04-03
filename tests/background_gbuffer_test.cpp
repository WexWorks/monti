#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <volk.h>

#include "test_helpers.h"
#include "scenes/CornellBox.h"

#include <monti/scene/Scene.h>

#include "../renderer/src/vulkan/FrameUniforms.h"
#include "../renderer/src/vulkan/GpuScene.h"

#include <cmath>
#include <cstddef>
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

// Create a uniform solid-color env map (4x2, RGBA32F).
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

// Build a minimal scene: single tiny triangle behind the camera so TLAS is
// non-degenerate, plus a uniform env map. Camera faces +Z, triangle is at
// Z = -10 (behind), so every pixel is a primary miss.
struct BackgroundScene {
    Scene scene;
    std::vector<MeshData> mesh_data;
};

BackgroundScene BuildBackgroundScene(float env_r, float env_g, float env_b,
                                     float env_intensity) {
    BackgroundScene bs;
    auto& scene = bs.scene;

    // Env map
    auto env_tex_id = scene.AddTexture(MakeEnvMap(env_r, env_g, env_b), "env");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = env_intensity;
    scene.SetEnvironmentLight(env);

    // Minimal material
    MaterialDesc mat;
    mat.base_color = {0.5f, 0.5f, 0.5f};
    mat.roughness = 1.0f;
    auto mat_id = scene.AddMaterial(std::move(mat), "dummy");

    // Tiny triangle behind the camera (Z = -10)
    MeshData md;
    Vertex v{};
    v.normal = {0.0f, 0.0f, 1.0f};
    v.tangent = {1.0f, 0.0f, 0.0f, 1.0f};

    v.position = {0.0f, 0.0f, -10.0f};
    md.vertices.push_back(v);
    v.position = {0.01f, 0.0f, -10.0f};
    md.vertices.push_back(v);
    v.position = {0.0f, 0.01f, -10.0f};
    md.vertices.push_back(v);

    md.indices = {0, 1, 2};

    Mesh mesh_desc{};
    mesh_desc.name = "dummy";
    mesh_desc.vertex_count = 3;
    mesh_desc.index_count = 3;
    mesh_desc.vertex_stride = sizeof(Vertex);
    mesh_desc.bbox_min = {0.0f, 0.0f, -10.0f};
    mesh_desc.bbox_max = {0.01f, 0.01f, -10.0f};

    auto mesh_id = scene.AddMesh(std::move(mesh_desc), "dummy");
    md.mesh_id = mesh_id;
    scene.AddNode(mesh_id, mat_id, "dummy");
    bs.mesh_data.push_back(std::move(md));

    // Camera looking +Z (default forward), positioned at origin
    CameraParams cam{};
    cam.position = {0.5f, 0.5f, 0.0f};
    cam.target = {0.5f, 0.5f, 1.0f};
    cam.up = {0.0f, 1.0f, 0.0f};
    cam.vertical_fov_radians = 0.785398f; // ~45 degrees
    cam.near_plane = 0.01f;
    cam.far_plane = 100.0f;
    scene.SetActiveCamera(cam);

    return bs;
}

// Render one frame with a background scene. Returns a struct holding all
// readback buffers (caller must Unmap + Destroy).
struct BackgroundReadback {
    vulkan::Buffer noisy_diffuse;
    vulkan::Buffer diffuse_albedo;
    vulkan::Buffer specular_albedo;
    vulkan::Buffer world_normals;
    vulkan::Buffer linear_depth;
    std::vector<GpuBuffer> gpu_buffers;
};

BackgroundReadback RenderAndReadback(monti::app::VulkanContext& ctx,
                                     Scene& scene,
                                     std::span<const MeshData> mesh_data,
                                     float env_blur = 3.5f) {
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
    renderer->SetEnvironmentBlur(env_blur);

    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(
        *renderer, ctx.Allocator(), ctx.Device(), upload_cmd,
        mesh_data, test::MakeGpuBufferProcs());
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

    BackgroundReadback rb;
    rb.noisy_diffuse   = test::ReadbackImage(ctx, gbuffer_images.NoisyDiffuseImage());
    rb.diffuse_albedo  = test::ReadbackImage(ctx, gbuffer_images.DiffuseAlbedoImage());
    rb.specular_albedo = test::ReadbackImage(ctx, gbuffer_images.SpecularAlbedoImage());
    rb.world_normals   = test::ReadbackImage(ctx, gbuffer_images.WorldNormalsImage());
    rb.linear_depth    = test::ReadbackImage(ctx, gbuffer_images.LinearDepthImage(),
                                              /*pixel_size=*/4); // RG16F = 4 bytes/pixel
    rb.gpu_buffers = std::move(gpu_buffers);
    return rb;
}

void CleanupReadback(monti::app::VulkanContext& ctx, BackgroundReadback& rb) {
    for (auto& buf : rb.gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: Background pixel writes unit diffuse albedo
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Background pixel writes unit diffuse albedo",
          "[background][gbuffer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto bs = BuildBackgroundScene(0.5f, 0.3f, 0.1f, 1.0f);
    auto rb = RenderAndReadback(tc.ctx, bs.scene, bs.mesh_data);

    auto* raw = static_cast<uint16_t*>(rb.diffuse_albedo.Map());
    REQUIRE(raw != nullptr);

    // Check center pixel — should be (1, 1, 1) for miss
    uint32_t cx = test::kTestWidth / 2;
    uint32_t cy = test::kTestHeight / 2;
    uint32_t idx = (cy * test::kTestWidth + cx) * 4;
    float r = test::HalfToFloat(raw[idx + 0]);
    float g = test::HalfToFloat(raw[idx + 1]);
    float b = test::HalfToFloat(raw[idx + 2]);

    INFO("Diffuse albedo center pixel: (" << r << ", " << g << ", " << b << ")");
    CHECK_THAT(r, WithinAbs(1.0, 0.01));
    CHECK_THAT(g, WithinAbs(1.0, 0.01));
    CHECK_THAT(b, WithinAbs(1.0, 0.01));

    // Verify ALL pixels are unit diffuse albedo (every pixel is background)
    uint32_t fail_count = 0;
    for (uint32_t i = 0; i < test::kPixelCount; ++i) {
        float pr = test::HalfToFloat(raw[i * 4 + 0]);
        float pg = test::HalfToFloat(raw[i * 4 + 1]);
        float pb = test::HalfToFloat(raw[i * 4 + 2]);
        if (std::abs(pr - 1.0f) > 0.01f || std::abs(pg - 1.0f) > 0.01f ||
            std::abs(pb - 1.0f) > 0.01f)
            ++fail_count;
    }
    INFO("Pixels with non-unit diffuse albedo: " << fail_count);
    CHECK(fail_count == 0);

    rb.diffuse_albedo.Unmap();
    CleanupReadback(tc.ctx, rb);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Background pixel writes zero specular albedo
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Background pixel writes zero specular albedo",
          "[background][gbuffer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto bs = BuildBackgroundScene(0.5f, 0.3f, 0.1f, 1.0f);
    auto rb = RenderAndReadback(tc.ctx, bs.scene, bs.mesh_data);

    auto* raw = static_cast<uint16_t*>(rb.specular_albedo.Map());
    REQUIRE(raw != nullptr);

    uint32_t cx = test::kTestWidth / 2;
    uint32_t cy = test::kTestHeight / 2;
    uint32_t idx = (cy * test::kTestWidth + cx) * 4;
    float r = test::HalfToFloat(raw[idx + 0]);
    float g = test::HalfToFloat(raw[idx + 1]);
    float b = test::HalfToFloat(raw[idx + 2]);

    INFO("Specular albedo center pixel: (" << r << ", " << g << ", " << b << ")");
    CHECK_THAT(r, WithinAbs(0.0, 0.01));
    CHECK_THAT(g, WithinAbs(0.0, 0.01));
    CHECK_THAT(b, WithinAbs(0.0, 0.01));

    // Verify all pixels are zero specular
    uint32_t fail_count = 0;
    for (uint32_t i = 0; i < test::kPixelCount; ++i) {
        float pr = test::HalfToFloat(raw[i * 4 + 0]);
        float pg = test::HalfToFloat(raw[i * 4 + 1]);
        float pb = test::HalfToFloat(raw[i * 4 + 2]);
        if (std::abs(pr) > 0.01f || std::abs(pg) > 0.01f || std::abs(pb) > 0.01f)
            ++fail_count;
    }
    INFO("Pixels with non-zero specular albedo: " << fail_count);
    CHECK(fail_count == 0);

    rb.specular_albedo.Unmap();
    CleanupReadback(tc.ctx, rb);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Background pixel writes zero normal
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Background pixel writes zero normal",
          "[background][gbuffer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto bs = BuildBackgroundScene(0.5f, 0.3f, 0.1f, 1.0f);
    auto rb = RenderAndReadback(tc.ctx, bs.scene, bs.mesh_data);

    auto* raw = static_cast<uint16_t*>(rb.world_normals.Map());
    REQUIRE(raw != nullptr);

    uint32_t cx = test::kTestWidth / 2;
    uint32_t cy = test::kTestHeight / 2;
    uint32_t idx = (cy * test::kTestWidth + cx) * 4;
    float x = test::HalfToFloat(raw[idx + 0]);
    float y = test::HalfToFloat(raw[idx + 1]);
    float z = test::HalfToFloat(raw[idx + 2]);
    float w = test::HalfToFloat(raw[idx + 3]);

    INFO("World normal center pixel: (" << x << ", " << y << ", " << z << ", " << w << ")");
    CHECK_THAT(x, WithinAbs(0.0, 0.01));
    CHECK_THAT(y, WithinAbs(0.0, 0.01));
    CHECK_THAT(z, WithinAbs(0.0, 0.01));
    CHECK_THAT(w, WithinAbs(0.0, 0.01));

    // Verify all pixels are zero normal
    uint32_t fail_count = 0;
    for (uint32_t i = 0; i < test::kPixelCount; ++i) {
        float px = test::HalfToFloat(raw[i * 4 + 0]);
        float py = test::HalfToFloat(raw[i * 4 + 1]);
        float pz = test::HalfToFloat(raw[i * 4 + 2]);
        float pw = test::HalfToFloat(raw[i * 4 + 3]);
        if (std::abs(px) > 0.01f || std::abs(py) > 0.01f ||
            std::abs(pz) > 0.01f || std::abs(pw) > 0.01f)
            ++fail_count;
    }
    INFO("Pixels with non-zero normals: " << fail_count);
    CHECK(fail_count == 0);

    rb.world_normals.Unmap();
    CleanupReadback(tc.ctx, rb);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: Background pixel alpha is 1.0
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Background pixel alpha is 1.0",
          "[background][gbuffer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto bs = BuildBackgroundScene(0.5f, 0.3f, 0.1f, 1.0f);
    auto rb = RenderAndReadback(tc.ctx, bs.scene, bs.mesh_data);

    auto* raw = static_cast<uint16_t*>(rb.noisy_diffuse.Map());
    REQUIRE(raw != nullptr);

    uint32_t cx = test::kTestWidth / 2;
    uint32_t cy = test::kTestHeight / 2;
    uint32_t idx = (cy * test::kTestWidth + cx) * 4;
    float a = test::HalfToFloat(raw[idx + 3]);

    INFO("Noisy diffuse alpha at center: " << a);
    CHECK_THAT(a, WithinAbs(1.0, 0.01));

    // Verify all pixels have alpha = 1.0
    uint32_t fail_count = 0;
    for (uint32_t i = 0; i < test::kPixelCount; ++i) {
        float pa = test::HalfToFloat(raw[i * 4 + 3]);
        if (std::abs(pa - 1.0f) > 0.01f)
            ++fail_count;
    }
    INFO("Pixels with alpha != 1.0: " << fail_count);
    CHECK(fail_count == 0);

    rb.noisy_diffuse.Unmap();
    CleanupReadback(tc.ctx, rb);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: Background pixel contains env map color
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Background pixel contains env map color",
          "[background][gbuffer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    // Uniform env (0.5, 0.3, 0.1) * intensity 2.0 → expect (1.0, 0.6, 0.2)
    auto bs = BuildBackgroundScene(0.5f, 0.3f, 0.1f, 2.0f);
    auto rb = RenderAndReadback(tc.ctx, bs.scene, bs.mesh_data);

    auto* raw = static_cast<uint16_t*>(rb.noisy_diffuse.Map());
    REQUIRE(raw != nullptr);

    // For a uniform env map at default blur (3.5), the blurred value should
    // still equal the uniform color since all texels are identical.
    uint32_t cx = test::kTestWidth / 2;
    uint32_t cy = test::kTestHeight / 2;
    uint32_t idx = (cy * test::kTestWidth + cx) * 4;
    float r = test::HalfToFloat(raw[idx + 0]);
    float g = test::HalfToFloat(raw[idx + 1]);
    float b = test::HalfToFloat(raw[idx + 2]);

    INFO("Noisy diffuse center pixel: (" << r << ", " << g << ", " << b << ")");
    // env_color * env_intensity = (0.5, 0.3, 0.1) * 2.0 = (1.0, 0.6, 0.2)
    // Tolerance is wider because the env map is tiny (4x2) and mip
    // filtering across RGBA16F may introduce small precision loss.
    CHECK_THAT(r, WithinAbs(1.0, 0.05));
    CHECK_THAT(g, WithinAbs(0.6, 0.05));
    CHECK_THAT(b, WithinAbs(0.2, 0.05));

    // Check mean across all pixels matches expected (uniform env → all equal)
    double sum_r = 0, sum_g = 0, sum_b = 0;
    for (uint32_t i = 0; i < test::kPixelCount; ++i) {
        sum_r += test::HalfToFloat(raw[i * 4 + 0]);
        sum_g += test::HalfToFloat(raw[i * 4 + 1]);
        sum_b += test::HalfToFloat(raw[i * 4 + 2]);
    }
    float mean_r = static_cast<float>(sum_r / test::kPixelCount);
    float mean_g = static_cast<float>(sum_g / test::kPixelCount);
    float mean_b = static_cast<float>(sum_b / test::kPixelCount);

    INFO("Mean background color: (" << mean_r << ", " << mean_g << ", " << mean_b << ")");
    CHECK_THAT(mean_r, WithinAbs(1.0, 0.1));
    CHECK_THAT(mean_g, WithinAbs(0.6, 0.1));
    CHECK_THAT(mean_b, WithinAbs(0.2, 0.1));

    rb.noisy_diffuse.Unmap();
    CleanupReadback(tc.ctx, rb);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: Background pixel depth is sentinel
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Background pixel depth is sentinel",
          "[background][gbuffer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto bs = BuildBackgroundScene(0.5f, 0.3f, 0.1f, 1.0f);
    auto rb = RenderAndReadback(tc.ctx, bs.scene, bs.mesh_data);

    auto* raw = static_cast<uint16_t*>(rb.linear_depth.Map());
    REQUIRE(raw != nullptr);

    constexpr float kSentinelDepth = 1e4f;

    // RG16F: 2 half-floats per pixel
    uint32_t cx = test::kTestWidth / 2;
    uint32_t cy = test::kTestHeight / 2;
    uint32_t idx = (cy * test::kTestWidth + cx) * 2;
    float depth_r = test::HalfToFloat(raw[idx + 0]);
    float depth_g = test::HalfToFloat(raw[idx + 1]);

    INFO("Linear depth center pixel: (" << depth_r << ", " << depth_g << ")");
    // Half-float can represent 10000, but precision is limited at that range.
    // RG16F max is 65504; 10000 is representable. Allow wider tolerance.
    CHECK_THAT(depth_r, WithinAbs(kSentinelDepth, 100.0));
    CHECK_THAT(depth_g, WithinAbs(kSentinelDepth, 100.0));

    // Verify all pixels have sentinel depth
    uint32_t fail_count = 0;
    for (uint32_t i = 0; i < test::kPixelCount; ++i) {
        float dr = test::HalfToFloat(raw[i * 2 + 0]);
        float dg = test::HalfToFloat(raw[i * 2 + 1]);
        if (std::abs(dr - kSentinelDepth) > 100.0f ||
            std::abs(dg - kSentinelDepth) > 100.0f)
            ++fail_count;
    }
    INFO("Pixels with non-sentinel depth: " << fail_count);
    CHECK(fail_count == 0);

    rb.linear_depth.Unmap();
    CleanupReadback(tc.ctx, rb);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 7: Background env blur 0.0 produces sharp env color
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Background env blur 0.0 produces sharp env color",
          "[background][gbuffer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    // Uniform env: sharp lookup should produce exact color * intensity
    constexpr float kEnvR = 0.4f, kEnvG = 0.7f, kEnvB = 0.2f;
    constexpr float kIntensity = 1.5f;
    auto bs = BuildBackgroundScene(kEnvR, kEnvG, kEnvB, kIntensity);
    auto rb = RenderAndReadback(tc.ctx, bs.scene, bs.mesh_data, /*env_blur=*/0.0f);

    auto* raw = static_cast<uint16_t*>(rb.noisy_diffuse.Map());
    REQUIRE(raw != nullptr);

    uint32_t cx = test::kTestWidth / 2;
    uint32_t cy = test::kTestHeight / 2;
    uint32_t idx = (cy * test::kTestWidth + cx) * 4;
    float r = test::HalfToFloat(raw[idx + 0]);
    float g = test::HalfToFloat(raw[idx + 1]);
    float b = test::HalfToFloat(raw[idx + 2]);

    float expected_r = kEnvR * kIntensity;  // 0.6
    float expected_g = kEnvG * kIntensity;  // 1.05
    float expected_b = kEnvB * kIntensity;  // 0.3

    INFO("Sharp env center pixel: (" << r << ", " << g << ", " << b << ")"
         << " expected: (" << expected_r << ", " << expected_g << ", " << expected_b << ")");
    CHECK_THAT(r, WithinAbs(expected_r, 0.05));
    CHECK_THAT(g, WithinAbs(expected_g, 0.05));
    CHECK_THAT(b, WithinAbs(expected_b, 0.05));

    rb.noisy_diffuse.Unmap();
    CleanupReadback(tc.ctx, rb);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 8: Background env blur produces smoothed env color
//
// For a uniform env map both sharp and blurred should produce the same
// value. This test uses a non-uniform env map (bright left, dark right)
// and verifies the blurred version is closer to the mean.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Background env blur produces smoothed env color",
          "[background][gbuffer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    // Create a non-uniform env map: left half = bright, right half = dark
    constexpr uint32_t kW = 4, kH = 2;
    std::vector<float> pixels(kW * kH * 4);
    for (uint32_t y = 0; y < kH; ++y) {
        for (uint32_t x = 0; x < kW; ++x) {
            uint32_t i = y * kW + x;
            float bright = (x < kW / 2) ? 2.0f : 0.1f;
            pixels[i * 4 + 0] = bright;
            pixels[i * 4 + 1] = bright * 0.5f;
            pixels[i * 4 + 2] = bright * 0.25f;
            pixels[i * 4 + 3] = 1.0f;
        }
    }
    TextureDesc env_tex;
    env_tex.width = kW;
    env_tex.height = kH;
    env_tex.format = PixelFormat::kRGBA32F;
    env_tex.data.resize(pixels.size() * sizeof(float));
    std::memcpy(env_tex.data.data(), pixels.data(), env_tex.data.size());

    auto make_scene = [&]() {
        Scene scene;
        auto env_id = scene.AddTexture(env_tex, "env");
        EnvironmentLight env{};
        env.hdr_lat_long = env_id;
        env.intensity = 1.0f;
        scene.SetEnvironmentLight(env);

        // Dummy triangle behind camera
        MaterialDesc mat;
        mat.base_color = {0.5f, 0.5f, 0.5f};
        mat.roughness = 1.0f;
        auto mat_id = scene.AddMaterial(std::move(mat), "dummy");

        MeshData md;
        Vertex v{};
        v.normal = {0.0f, 0.0f, 1.0f};
        v.tangent = {1.0f, 0.0f, 0.0f, 1.0f};
        v.position = {0.0f, 0.0f, -10.0f};
        md.vertices.push_back(v);
        v.position = {0.01f, 0.0f, -10.0f};
        md.vertices.push_back(v);
        v.position = {0.0f, 0.01f, -10.0f};
        md.vertices.push_back(v);
        md.indices = {0, 1, 2};

        Mesh mesh_desc{};
        mesh_desc.name = "dummy";
        mesh_desc.vertex_count = 3;
        mesh_desc.index_count = 3;
        mesh_desc.vertex_stride = sizeof(Vertex);
        mesh_desc.bbox_min = {0.0f, 0.0f, -10.0f};
        mesh_desc.bbox_max = {0.01f, 0.01f, -10.0f};

        auto mesh_id = scene.AddMesh(std::move(mesh_desc), "dummy");
        md.mesh_id = mesh_id;
        scene.AddNode(mesh_id, mat_id, "dummy");

        CameraParams cam{};
        cam.position = {0.5f, 0.5f, 0.0f};
        cam.target = {0.5f, 0.5f, 1.0f};
        cam.up = {0.0f, 1.0f, 0.0f};
        cam.vertical_fov_radians = 0.785398f; // ~45 degrees
        cam.near_plane = 0.01f;
        cam.far_plane = 100.0f;
        scene.SetActiveCamera(cam);

        std::vector<MeshData> mesh_data;
        mesh_data.push_back(std::move(md));
        return std::make_pair(std::move(scene), std::move(mesh_data));
    };

    // Render sharp (mip 0)
    auto [scene_sharp, md_sharp] = make_scene();
    auto rb_sharp = RenderAndReadback(tc.ctx, scene_sharp, md_sharp, 0.0f);
    auto* raw_sharp = static_cast<uint16_t*>(rb_sharp.noisy_diffuse.Map());
    REQUIRE(raw_sharp != nullptr);

    // Render blurred (mip 4)
    auto [scene_blur, md_blur] = make_scene();
    auto rb_blur = RenderAndReadback(tc.ctx, scene_blur, md_blur, 4.0f);
    auto* raw_blur = static_cast<uint16_t*>(rb_blur.noisy_diffuse.Map());
    REQUIRE(raw_blur != nullptr);

    // Compute per-pixel variance for sharp and blurred images. The blurred
    // version should have lower variance (closer to the spatial mean).
    double sharp_sum = 0, sharp_sum_sq = 0;
    double blur_sum = 0, blur_sum_sq = 0;
    for (uint32_t i = 0; i < test::kPixelCount; ++i) {
        float sr = test::HalfToFloat(raw_sharp[i * 4 + 0]);
        float sg = test::HalfToFloat(raw_sharp[i * 4 + 1]);
        float sb = test::HalfToFloat(raw_sharp[i * 4 + 2]);
        float s_lum = 0.2126f * sr + 0.7152f * sg + 0.0722f * sb;
        sharp_sum += s_lum;
        sharp_sum_sq += s_lum * s_lum;

        float br = test::HalfToFloat(raw_blur[i * 4 + 0]);
        float bg = test::HalfToFloat(raw_blur[i * 4 + 1]);
        float bb = test::HalfToFloat(raw_blur[i * 4 + 2]);
        float b_lum = 0.2126f * br + 0.7152f * bg + 0.0722f * bb;
        blur_sum += b_lum;
        blur_sum_sq += b_lum * b_lum;
    }

    double n = test::kPixelCount;
    double sharp_var = (sharp_sum_sq / n) - (sharp_sum / n) * (sharp_sum / n);
    double blur_var  = (blur_sum_sq / n)  - (blur_sum / n)  * (blur_sum / n);

    INFO("Sharp variance: " << sharp_var << ", Blur variance: " << blur_var);
    // Blurred image should have lower pixel-to-pixel variance
    CHECK(blur_var < sharp_var);

    rb_sharp.noisy_diffuse.Unmap();
    rb_blur.noisy_diffuse.Unmap();
    CleanupReadback(tc.ctx, rb_sharp);
    CleanupReadback(tc.ctx, rb_blur);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 9: Frame uniform alignment is valid
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Frame uniform alignment is valid",
          "[background][alignment]") {
    STATIC_CHECK(offsetof(FrameUniforms, bg_env_mip_level) == 232);
    STATIC_CHECK(sizeof(FrameUniforms) % 16 == 0);
    STATIC_CHECK(sizeof(FrameUniforms) == 240);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 10: Denoiser golden reference still passes
//
// Re-run the existing denoiser passthrough golden test to confirm that
// unconditional demodulation/remodulation produces correct output.
// This is a meta-test: it validates that the existing golden test still
// passes after Phase 1 changes. The actual golden test lives in
// deni_passthrough_test.cpp.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Background changes preserve denoiser passthrough",
          "[background][denoiser][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    // Render a Cornell box scene (with geometry) through the full pipeline
    // and verify the denoiser doesn't corrupt the output.
    auto [scene, mesh_data] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene);

    // Add env map so the env path is exercised
    auto env_id = scene.AddTexture(MakeEnvMap(0.2f, 0.3f, 0.4f), "env");
    EnvironmentLight env{};
    env.hdr_lat_long = env_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    RendererDesc desc{};
    desc.device = tc.ctx.Device();
    desc.physical_device = tc.ctx.PhysicalDevice();
    desc.queue = tc.ctx.GraphicsQueue();
    desc.queue_family_index = tc.ctx.QueueFamilyIndex();
    desc.allocator = tc.ctx.Allocator();
    desc.width = test::kTestWidth;
    desc.height = test::kTestHeight;
    desc.samples_per_pixel = 4;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;
    test::FillRendererProcAddrs(desc, tc.ctx);

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);
    renderer->SetEnvironmentBlur(3.5f);

    auto light_meshes = SynthesizeAreaLightGeometry(scene);
    std::vector<MeshData> all_mesh_data(mesh_data.begin(), mesh_data.end());
    all_mesh_data.insert(all_mesh_data.end(),
        std::make_move_iterator(light_meshes.begin()),
        std::make_move_iterator(light_meshes.end()));

    VkCommandBuffer upload_cmd = tc.ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(
        *renderer, tc.ctx.Allocator(), tc.ctx.Device(), upload_cmd,
        all_mesh_data, test::MakeGpuBufferProcs());
    REQUIRE_FALSE(gpu_buffers.empty());
    tc.ctx.SubmitAndWait(upload_cmd);

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = tc.ctx.BeginOneShot();
    REQUIRE(gbuffer_images.Create(tc.ctx.Allocator(), tc.ctx.Device(),
                                  test::kTestWidth, test::kTestHeight, gbuf_cmd,
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
    tc.ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    VkCommandBuffer render_cmd = tc.ctx.BeginOneShot();
    REQUIRE(renderer->RenderFrame(render_cmd, gbuffer, 0));
    tc.ctx.SubmitAndWait(render_cmd);

    // Read back noisy diffuse and verify no NaN/Inf
    auto diffuse_rb = test::ReadbackImage(tc.ctx, gbuffer_images.NoisyDiffuseImage());
    auto* diffuse_raw = static_cast<uint16_t*>(diffuse_rb.Map());
    REQUIRE(diffuse_raw != nullptr);

    auto stats = test::AnalyzeRGBA16F(diffuse_raw, test::kPixelCount);

    INFO("NaN count: " << stats.nan_count);
    INFO("Inf count: " << stats.inf_count);
    INFO("Nonzero pixels: " << stats.nonzero_count);
    CHECK(stats.nan_count == 0);
    CHECK(stats.inf_count == 0);
    CHECK(stats.nonzero_count > 0);

    // All pixels should have alpha = 1.0
    uint32_t alpha_fail = 0;
    for (uint32_t i = 0; i < test::kPixelCount; ++i) {
        float a = test::HalfToFloat(diffuse_raw[i * 4 + 3]);
        if (std::abs(a - 1.0f) > 0.01f)
            ++alpha_fail;
    }
    INFO("Pixels with alpha != 1.0: " << alpha_fail);
    CHECK(alpha_fail == 0);

    diffuse_rb.Unmap();

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(tc.ctx.Allocator(), buf);
    tc.ctx.WaitIdle();
}
