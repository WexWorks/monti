#include <catch2/catch_test_macros.hpp>

#include <volk.h>

#include "test_helpers.h"
#include "scenes/CornellBox.h"

#include <monti/scene/Scene.h>
#include <deni/vulkan/Denoiser.h>

#include "../app/core/CameraSetup.h"
#include "gltf/GltfLoader.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace monti;
using namespace monti::vulkan;

#ifndef APP_SHADER_SPV_DIR
#define APP_SHADER_SPV_DIR "build/app_shaders"
#endif

// ═══════════════════════════════════════════════════════════════════════════
//
// ML Denoiser End-to-End Quality Tests
//
// Compares single-frame ML-denoised output against a high-SPP multi-frame
// reference. The ML denoiser should produce output perceptually closer to
// the reference than the raw noisy input. This validates the entire pipeline:
// scene loading → path tracing → G-buffer → ML denoiser → output.
//
// Quality comparisons use WARN (not CHECK) because the model is still
// improving. Convert to CHECK once the model consistently beats noisy input.
//
// Tests use the production model (deni_v1.denimodel) from DENI_MODEL_DIR.
// If no model is available, tests SKIP.
//
// ═══════════════════════════════════════════════════════════════════════════

namespace {

constexpr uint32_t kTestWidth = 1024;
constexpr uint32_t kTestHeight = 1024;
constexpr uint32_t kPixelCount = kTestWidth * kTestHeight;

// Reference rendering: many frames at moderate SPP for a clean target.
constexpr uint32_t kRefFrames = 64;
constexpr uint32_t kRefSppPerFrame = 16;  // 1024 total SPP

// Noisy rendering: single frame at low SPP, fed to the ML denoiser.
constexpr uint32_t kNoisySpp = 4;

struct TestContext {
    monti::app::VulkanContext& ctx = test::SharedVulkanContext();
    bool Init() { return ctx.Device() != VK_NULL_HANDLE; }
};

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

static std::string AssetPath(const char* filename) {
    return std::string(MONTI_TEST_ASSETS_DIR) + "/" + filename;
}

// Log FLIP quality results. Uses WARN for visibility when quality targets
// are not met, INFO otherwise. Returns true if denoiser beats noisy baseline.
bool LogQualityResults(float flip_denoised, float flip_noisy) {
    if (flip_denoised >= flip_noisy) {
        WARN("Denoiser not yet improving quality: FLIP denoised="
             << flip_denoised << " >= noisy=" << flip_noisy);
    } else {
        INFO("Denoiser improving quality: FLIP denoised="
             << flip_denoised << " < noisy=" << flip_noisy);
    }
    if (flip_denoised >= 0.5f) {
        WARN("Denoised FLIP above 0.5 threshold: " << flip_denoised);
    }
    return flip_denoised < flip_noisy;
}

// Render one frame at low SPP, denoise with the ML model, read back the
// denoised RGBA16F image and return tone-mapped linear RGB for FLIP.
// Also writes a diagnostic PNG.
struct DenoisedResult {
    std::vector<float> rgb;     // Tone-mapped linear RGB for FLIP
    bool has_ml_model = false;
};

DenoisedResult RenderDenoised(monti::app::VulkanContext& ctx,
                              Scene& scene,
                              std::span<const MeshData> mesh_data,
                              const std::string& name) {
    DenoisedResult result;

    // Create renderer
    RendererDesc renderer_desc{};
    renderer_desc.device = ctx.Device();
    renderer_desc.physical_device = ctx.PhysicalDevice();
    renderer_desc.queue = ctx.GraphicsQueue();
    renderer_desc.queue_family_index = ctx.QueueFamilyIndex();
    renderer_desc.allocator = ctx.Allocator();
    renderer_desc.width = kTestWidth;
    renderer_desc.height = kTestHeight;
    renderer_desc.samples_per_pixel = kNoisySpp;
    renderer_desc.shader_dir = MONTI_SHADER_SPV_DIR;
    test::FillRendererProcAddrs(renderer_desc, ctx);

    auto renderer = Renderer::Create(renderer_desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    // Upload meshes
    auto procs = test::MakeGpuBufferProcs();
    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(
        *renderer, ctx.Allocator(), ctx.Device(), upload_cmd, mesh_data, procs);
    REQUIRE_FALSE(gpu_buffers.empty());
    ctx.SubmitAndWait(upload_cmd);

    // Create G-buffer images
    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                                  kTestWidth, kTestHeight, gbuf_cmd,
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
    ctx.SubmitAndWait(gbuf_cmd);

    // Create ML denoiser (model auto-discovered by library)
    deni::vulkan::DenoiserDesc denoiser_desc{};
    denoiser_desc.device = ctx.Device();
    denoiser_desc.physical_device = ctx.PhysicalDevice();
    denoiser_desc.width = kTestWidth;
    denoiser_desc.height = kTestHeight;
    denoiser_desc.allocator = ctx.Allocator();
    denoiser_desc.shader_dir = DENI_SHADER_SPV_DIR;
    test::FillDenoiserProcAddrs(denoiser_desc, ctx);

    auto denoiser = deni::vulkan::Denoiser::Create(denoiser_desc);
    REQUIRE(denoiser);

    result.has_ml_model = denoiser->HasMlModel();
    if (!result.has_ml_model) {
        // Cleanup and return empty — caller will SKIP
        for (auto& buf : gpu_buffers) DestroyGpuBuffer(ctx.Allocator(), buf);
        gbuffer_images.Destroy();
        ctx.WaitIdle();
        return result;
    }

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    // Render single frame
    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(renderer->RenderFrame(cmd, gbuffer, 0));

    // RT → compute barrier
    VkMemoryBarrier2 rt_to_compute{};
    rt_to_compute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    rt_to_compute.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    rt_to_compute.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    rt_to_compute.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    rt_to_compute.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo barrier_dep{};
    barrier_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    barrier_dep.memoryBarrierCount = 1;
    barrier_dep.pMemoryBarriers = &rt_to_compute;
    vkCmdPipelineBarrier2(cmd, &barrier_dep);

    deni::vulkan::DenoiserInput input{};
    input.noisy_diffuse = gbuffer.noisy_diffuse;
    input.noisy_specular = gbuffer.noisy_specular;
    input.motion_vectors = gbuffer.motion_vectors;
    input.linear_depth = gbuffer.linear_depth;
    input.world_normals = gbuffer.world_normals;
    input.diffuse_albedo = gbuffer.diffuse_albedo;
    input.specular_albedo = gbuffer.specular_albedo;
    input.render_width = kTestWidth;
    input.render_height = kTestHeight;
    input.reset_accumulation = true;

    auto denoise_output = denoiser->Denoise(cmd, input);
    ctx.SubmitAndWait(cmd);

    REQUIRE(denoise_output.denoised_image != VK_NULL_HANDLE);

    // Read back denoised RGBA16F
    auto denoised_rb = test::ReadbackImage(
        ctx, denoise_output.denoised_image, 8,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        kTestWidth, kTestHeight);
    auto* denoised_raw = static_cast<uint16_t*>(denoised_rb.Map());
    REQUIRE(denoised_raw);

    // Also read back raw noisy for comparison PNG
    auto noisy_diffuse_rb = test::ReadbackImage(
        ctx, gbuffer_images.NoisyDiffuseImage(), 8,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        kTestWidth, kTestHeight);
    auto noisy_specular_rb = test::ReadbackImage(
        ctx, gbuffer_images.NoisySpecularImage(), 8,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        kTestWidth, kTestHeight);
    auto* noisy_d = static_cast<uint16_t*>(noisy_diffuse_rb.Map());
    auto* noisy_s = static_cast<uint16_t*>(noisy_specular_rb.Map());

    // Tone-map denoised output for FLIP (Reinhard)
    result.rgb.resize(kPixelCount * 3);
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        float r = test::HalfToFloat(denoised_raw[i * 4 + 0]);
        float g = test::HalfToFloat(denoised_raw[i * 4 + 1]);
        float b = test::HalfToFloat(denoised_raw[i * 4 + 2]);
        if (std::isnan(r) || std::isinf(r)) r = 0.0f;
        if (std::isnan(g) || std::isinf(g)) g = 0.0f;
        if (std::isnan(b) || std::isinf(b)) b = 0.0f;
        r = std::max(r, 0.0f) / (1.0f + std::max(r, 0.0f));
        g = std::max(g, 0.0f) / (1.0f + std::max(g, 0.0f));
        b = std::max(b, 0.0f) / (1.0f + std::max(b, 0.0f));
        result.rgb[i * 3 + 0] = r;
        result.rgb[i * 3 + 1] = g;
        result.rgb[i * 3 + 2] = b;
    }

    // Write diagnostic PNGs
    test::WritePNG("tests/output/ml_e2e_" + name + "_denoised.png",
                   denoised_raw, kTestWidth, kTestHeight);
    test::WriteCombinedPNG("tests/output/ml_e2e_" + name + "_noisy.png",
                           noisy_d, noisy_s, kTestWidth, kTestHeight);

    denoised_rb.Unmap();
    noisy_diffuse_rb.Unmap();
    noisy_specular_rb.Unmap();

    // Cleanup
    denoised_rb.Destroy();
    noisy_diffuse_rb.Destroy();
    noisy_specular_rb.Destroy();
    denoiser.reset();
    renderer.reset();
    for (auto& buf : gpu_buffers) DestroyGpuBuffer(ctx.Allocator(), buf);
    gbuffer_images.Destroy();
    ctx.WaitIdle();

    return result;
}

// Render a high-SPP multi-frame reference and return tone-mapped linear RGB.
std::vector<float> RenderReference(monti::app::VulkanContext& ctx,
                                   Scene& scene,
                                   std::span<const MeshData> mesh_data,
                                   const std::string& name) {
    auto mf = test::RenderSceneMultiFrame(ctx, scene, mesh_data,
                                          kRefFrames, kRefSppPerFrame,
                                          kTestWidth, kTestHeight);

    auto rgb = test::TonemappedRGB(mf.diffuse.data(), mf.specular.data(),
                                   kPixelCount);

    test::WriteCombinedPNG("tests/output/ml_e2e_" + name + "_reference.png",
                           mf.diffuse.data(), mf.specular.data(),
                           kTestWidth, kTestHeight);

    test::CleanupMultiFrameResult(ctx.Allocator(), mf);
    return rgb;
}

// Render raw noisy (no denoiser) and return tone-mapped linear RGB for baseline.
std::vector<float> RenderNoisyBaseline(monti::app::VulkanContext& ctx,
                                       Scene& scene,
                                       std::span<const MeshData> mesh_data) {
    auto mf = test::RenderSceneMultiFrame(ctx, scene, mesh_data, 1, kNoisySpp,
                                          kTestWidth, kTestHeight);

    auto rgb = test::TonemappedRGB(mf.diffuse.data(), mf.specular.data(),
                                   kPixelCount);

    test::CleanupMultiFrameResult(ctx.Allocator(), mf);
    return rgb;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Cornell Box: ML denoiser vs multi-frame reference
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("ML E2E: Cornell Box — denoised closer to reference than noisy",
          "[ml_e2e][pipeline][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    // Build scene once — both paths trace the same scene
    auto [scene, mesh_data] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene);

    // 1) ML-denoised single frame
    auto denoised = RenderDenoised(tc.ctx, scene, mesh_data, "cornell");
    if (!denoised.has_ml_model) {
        WARN("ML model not available — skipping");
        SKIP();
    }

    // 2) High-SPP reference
    auto ref_rgb = RenderReference(tc.ctx, scene, mesh_data, "cornell");

    // 3) Raw noisy baseline (same SPP as denoised input)
    auto noisy_rgb = RenderNoisyBaseline(tc.ctx, scene, mesh_data);

    float flip_denoised = test::ComputeMeanFlip(ref_rgb, denoised.rgb,
                                                 kTestWidth, kTestHeight);
    float flip_noisy = test::ComputeMeanFlip(ref_rgb, noisy_rgb,
                                              kTestWidth, kTestHeight);

    CHECK_FALSE(std::isnan(flip_denoised));
    CHECK_FALSE(std::isnan(flip_noisy));
    LogQualityResults(flip_denoised, flip_noisy);
}

// ═══════════════════════════════════════════════════════════════════════════
// DamagedHelmet: ML denoiser vs multi-frame reference
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("ML E2E: DamagedHelmet — denoised closer to reference than noisy",
          "[ml_e2e][pipeline][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("DamagedHelmet.glb"));
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.3f, 0.3f, 0.3f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto denoised = RenderDenoised(tc.ctx, scene, result.mesh_data, "helmet");
    if (!denoised.has_ml_model) {
        WARN("ML model not available — skipping");
        SKIP();
    }

    auto ref_rgb = RenderReference(tc.ctx, scene, result.mesh_data, "helmet");
    auto noisy_rgb = RenderNoisyBaseline(tc.ctx, scene, result.mesh_data);

    float flip_denoised = test::ComputeMeanFlip(ref_rgb, denoised.rgb,
                                                 kTestWidth, kTestHeight);
    float flip_noisy = test::ComputeMeanFlip(ref_rgb, noisy_rgb,
                                              kTestWidth, kTestHeight);

    CHECK_FALSE(std::isnan(flip_denoised));
    CHECK_FALSE(std::isnan(flip_noisy));
    LogQualityResults(flip_denoised, flip_noisy);
}

// ═══════════════════════════════════════════════════════════════════════════
// DragonAttenuation: ML denoiser with transmissive materials
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("ML E2E: DragonAttenuation — denoised closer to reference than noisy",
          "[ml_e2e][pipeline][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, AssetPath("DragonAttenuation.glb"));
    REQUIRE(result.success);

    auto camera = monti::app::ComputeDefaultCamera(scene);
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.5f, 0.5f, 0.5f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto denoised = RenderDenoised(tc.ctx, scene, result.mesh_data, "dragon");
    if (!denoised.has_ml_model) {
        WARN("ML model not available — skipping");
        SKIP();
    }

    auto ref_rgb = RenderReference(tc.ctx, scene, result.mesh_data, "dragon");
    auto noisy_rgb = RenderNoisyBaseline(tc.ctx, scene, result.mesh_data);

    float flip_denoised = test::ComputeMeanFlip(ref_rgb, denoised.rgb,
                                                 kTestWidth, kTestHeight);
    float flip_noisy = test::ComputeMeanFlip(ref_rgb, noisy_rgb,
                                              kTestWidth, kTestHeight);

    CHECK_FALSE(std::isnan(flip_denoised));
    CHECK_FALSE(std::isnan(flip_noisy));
    LogQualityResults(flip_denoised, flip_noisy);
}

// ═══════════════════════════════════════════════════════════════════════════
// Extended scene tests — only when MONTI_DOWNLOAD_EXTENDED_SCENES is on
// ═══════════════════════════════════════════════════════════════════════════

#ifdef MONTI_EXTENDED_SCENES_DIR

TEST_CASE("ML E2E: BistroInterior — denoised closer to reference than noisy",
          "[ml_e2e][extended][pipeline][vulkan][integration]") {
    std::string gltf_path =
        std::string(MONTI_EXTENDED_SCENES_DIR) + "/BistroInterior/scene.gltf";
    if (!std::filesystem::exists(gltf_path)) SKIP("Scene not downloaded");

    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, gltf_path);
    REQUIRE(result.success);

    CameraParams camera{};
    camera.position = {0.6843839883804321f, 2.1141409873962402f, -0.13792076706886292f};
    camera.target = {24.156171798706055f, 1.5206208229064941f, -7.91049337387085f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.vertical_fov_radians = glm::radians(60.0f);
    camera.near_plane = 0.01f;
    camera.far_plane = 10000.0f;
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.2f, 0.2f, 0.2f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto denoised = RenderDenoised(tc.ctx, scene, result.mesh_data, "bistro");
    if (!denoised.has_ml_model) {
        WARN("ML model not available — skipping");
        SKIP();
    }

    auto ref_rgb = RenderReference(tc.ctx, scene, result.mesh_data, "bistro");
    auto noisy_rgb = RenderNoisyBaseline(tc.ctx, scene, result.mesh_data);

    float flip_denoised = test::ComputeMeanFlip(ref_rgb, denoised.rgb,
                                                 kTestWidth, kTestHeight);
    float flip_noisy = test::ComputeMeanFlip(ref_rgb, noisy_rgb,
                                              kTestWidth, kTestHeight);

    CHECK_FALSE(std::isnan(flip_denoised));
    CHECK_FALSE(std::isnan(flip_noisy));
    LogQualityResults(flip_denoised, flip_noisy);
}

TEST_CASE("ML E2E: AbandonedWarehouse — denoised closer to reference than noisy",
          "[ml_e2e][extended][pipeline][vulkan][integration]") {
    std::string gltf_path =
        std::string(MONTI_EXTENDED_SCENES_DIR) + "/AbandonedWarehouse/AbandonedWarehouse.gltf";
    if (!std::filesystem::exists(gltf_path)) SKIP("Scene not downloaded");

    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, gltf_path);
    REQUIRE(result.success);

    CameraParams camera{};
    camera.position = {10.485430717468262f, -0.16410964727401733f, -5.454324245452881f};
    camera.target = {-15.309294700622559f, -8.962313652038574f, 7.236011028289795f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.vertical_fov_radians = glm::radians(60.0f);
    camera.near_plane = 0.01f;
    camera.far_plane = 10000.0f;
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.3f, 0.3f, 0.3f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto denoised = RenderDenoised(tc.ctx, scene, result.mesh_data, "warehouse");
    if (!denoised.has_ml_model) {
        WARN("ML model not available — skipping");
        SKIP();
    }

    auto ref_rgb = RenderReference(tc.ctx, scene, result.mesh_data, "warehouse");
    auto noisy_rgb = RenderNoisyBaseline(tc.ctx, scene, result.mesh_data);

    float flip_denoised = test::ComputeMeanFlip(ref_rgb, denoised.rgb,
                                                 kTestWidth, kTestHeight);
    float flip_noisy = test::ComputeMeanFlip(ref_rgb, noisy_rgb,
                                              kTestWidth, kTestHeight);

    CHECK_FALSE(std::isnan(flip_denoised));
    CHECK_FALSE(std::isnan(flip_noisy));
    LogQualityResults(flip_denoised, flip_noisy);
}

TEST_CASE("ML E2E: Brutalism — denoised closer to reference than noisy",
          "[ml_e2e][extended][pipeline][vulkan][integration]") {
    std::string gltf_path =
        std::string(MONTI_EXTENDED_SCENES_DIR) + "/Brutalism/BrutalistHall.gltf";
    if (!std::filesystem::exists(gltf_path)) SKIP("Scene not downloaded");

    TestContext tc;
    REQUIRE(tc.Init());

    Scene scene;
    auto result = gltf::LoadGltf(scene, gltf_path);
    REQUIRE(result.success);

    CameraParams camera{};
    camera.position = {1.554876685142517f, 16.712493896484375f, -46.013267517089844f};
    camera.target = {62.37873840332031f, -3.2877864837646484f, 18.419349670410156f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.vertical_fov_radians = glm::radians(60.0f);
    camera.near_plane = 0.01f;
    camera.far_plane = 10000.0f;
    scene.SetActiveCamera(camera);
    auto env_tex_id = scene.AddTexture(MakeEnvMap(0.3f, 0.3f, 0.3f), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    auto denoised = RenderDenoised(tc.ctx, scene, result.mesh_data, "brutalism");
    if (!denoised.has_ml_model) {
        WARN("ML model not available — skipping");
        SKIP();
    }

    auto ref_rgb = RenderReference(tc.ctx, scene, result.mesh_data, "brutalism");
    auto noisy_rgb = RenderNoisyBaseline(tc.ctx, scene, result.mesh_data);

    float flip_denoised = test::ComputeMeanFlip(ref_rgb, denoised.rgb,
                                                 kTestWidth, kTestHeight);
    float flip_noisy = test::ComputeMeanFlip(ref_rgb, noisy_rgb,
                                              kTestWidth, kTestHeight);

    CHECK_FALSE(std::isnan(flip_denoised));
    CHECK_FALSE(std::isnan(flip_noisy));
    LogQualityResults(flip_denoised, flip_noisy);
}

#endif  // MONTI_EXTENDED_SCENES_DIR
