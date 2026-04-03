// End-to-end integration test for GenerationSession path rendering.
//
// Exercises the full production pipeline:
//   GenerationSession::Run() → Renderer → GBufferImages → Writer → EXR output
//
// Verifies that:
//   1. ViewpointEntries are grouped by (path_id, frame) in output
//   2. Frame 0 of each path has near-zero motion vectors (temporal state reset)
//   3. Frame 1+ of each path has non-zero motion vectors (camera moved)
//   4. Two separate paths each independently reset their temporal state

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <tinyexr.h>
#include <volk.h>

#include "test_helpers.h"
#include "scenes/CornellBox.h"

#include <monti/capture/Writer.h>
#include <monti/scene/Scene.h>
#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>

#include "../app/core/CameraSetup.h"
#include "../app/core/GBufferImages.h"
#include "../app/datagen/GenerationSession.h"

#ifndef CAPTURE_SHADER_SPV_DIR
#define CAPTURE_SHADER_SPV_DIR "build/capture_shaders"
#endif

using namespace monti;
using namespace monti::vulkan;

namespace {

constexpr uint32_t kWidth = 128;
constexpr uint32_t kHeight = 128;
const std::string kTestOutputDir = "tests/output/datagen_session_e2e";

// Load an EXR file via tinyexr. Returns channel count on success, -1 on failure.
int LoadExr(const std::string& path, EXRHeader& header, EXRImage& image) {
    EXRVersion version;
    if (ParseEXRVersionFromFile(&version, path.c_str()) != TINYEXR_SUCCESS)
        return -1;

    InitEXRHeader(&header);
    const char* err = nullptr;
    if (ParseEXRHeaderFromFile(&header, &version, path.c_str(), &err) != TINYEXR_SUCCESS) {
        if (err) FreeEXRErrorMessage(err);
        return -1;
    }

    for (int i = 0; i < header.num_channels; ++i)
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;

    InitEXRImage(&image);
    if (LoadEXRImageFromFile(&image, &header, path.c_str(), &err) != TINYEXR_SUCCESS) {
        if (err) FreeEXRErrorMessage(err);
        FreeEXRHeader(&header);
        return -1;
    }

    return header.num_channels;
}

int FindChannel(const EXRHeader& header, const char* name) {
    for (int i = 0; i < header.num_channels; ++i) {
        if (std::strcmp(header.channels[i].name, name) == 0) return i;
    }
    return -1;
}

// Compute max motion vector magnitude from loaded EXR motion.X and motion.Y channels.
float MaxMVMagnitude(const EXRImage& image, int mx_idx, int my_idx, uint32_t pixels) {
    auto* mx_data = reinterpret_cast<const float*>(image.images[mx_idx]);
    auto* my_data = reinterpret_cast<const float*>(image.images[my_idx]);
    float max_mag = 0.0f;
    for (uint32_t i = 0; i < pixels; ++i) {
        float x = mx_data[i];
        float y = my_data[i];
        if (std::isnan(x) || std::isnan(y)) continue;
        float mag = std::sqrt(x * x + y * y);
        if (mag > max_mag) max_mag = mag;
    }
    return max_mag;
}

// Count pixels with motion vector magnitude above threshold.
uint32_t CountNonZeroMV(const EXRImage& image, int mx_idx, int my_idx,
                        uint32_t pixels, float threshold = 1e-5f) {
    auto* mx_data = reinterpret_cast<const float*>(image.images[mx_idx]);
    auto* my_data = reinterpret_cast<const float*>(image.images[my_idx]);
    uint32_t count = 0;
    for (uint32_t i = 0; i < pixels; ++i) {
        float x = mx_data[i];
        float y = my_data[i];
        if (std::isnan(x) || std::isnan(y)) continue;
        if (std::sqrt(x * x + y * y) > threshold) ++count;
    }
    return count;
}

// RAII cleanup for test output directory.
struct ScopedCleanup {
    ~ScopedCleanup() {
        std::error_code ec;
        std::filesystem::remove_all(kTestOutputDir, ec);
    }
};

// Read motion vector stats from an EXR file's motion.X / motion.Y channels.
struct MVStats {
    float max_magnitude = 0.0f;
    uint32_t nonzero_count = 0;
    bool valid = false;
};

MVStats ReadMotionVectors(const std::string& exr_path) {
    EXRHeader header;
    EXRImage image;
    int num_ch = LoadExr(exr_path, header, image);
    if (num_ch < 0) return {};

    int mx_idx = FindChannel(header, "motion.X");
    int my_idx = FindChannel(header, "motion.Y");
    if (mx_idx < 0 || my_idx < 0) {
        FreeEXRImage(&image);
        FreeEXRHeader(&header);
        return {};
    }

    uint32_t pixels = static_cast<uint32_t>(image.width * image.height);
    MVStats stats;
    stats.max_magnitude = MaxMVMagnitude(image, mx_idx, my_idx, pixels);
    stats.nonzero_count = CountNonZeroMV(image, mx_idx, my_idx, pixels);
    stats.valid = true;

    FreeEXRImage(&image);
    FreeEXRHeader(&header);
    return stats;
}

}  // namespace

TEST_CASE("GenerationSession E2E: two paths produce correct motion vectors",
          "[datagen][e2e][vulkan][integration]") {
    ScopedCleanup cleanup;

    auto& ctx = test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    // ── Build Cornell Box scene with environment ──
    auto [scene, mesh_data] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene);

    // Add a minimal environment map (required by renderer)
    constexpr uint32_t kEnvW = 4, kEnvH = 2;
    std::vector<float> env_pixels(kEnvW * kEnvH * 4);
    for (uint32_t i = 0; i < kEnvW * kEnvH; ++i) {
        env_pixels[i * 4 + 0] = 0.3f;
        env_pixels[i * 4 + 1] = 0.3f;
        env_pixels[i * 4 + 2] = 0.3f;
        env_pixels[i * 4 + 3] = 1.0f;
    }
    TextureDesc env_tex;
    env_tex.width = kEnvW;
    env_tex.height = kEnvH;
    env_tex.format = PixelFormat::kRGBA32F;
    env_tex.data.resize(env_pixels.size() * sizeof(float));
    std::memcpy(env_tex.data.data(), env_pixels.data(), env_tex.data.size());
    auto env_tex_id = scene.AddTexture(std::move(env_tex), "env_map");
    EnvironmentLight env{};
    env.hdr_lat_long = env_tex_id;
    env.intensity = 1.0f;
    scene.SetEnvironmentLight(env);

    // ── Create renderer ──
    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = kWidth;
    desc.height = kHeight;
    desc.samples_per_pixel = 1;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;
    test::FillRendererProcAddrs(desc, ctx);

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    // Upload meshes
    auto procs = test::MakeGpuBufferProcs();
    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(
        *renderer, ctx.Allocator(), ctx.Device(), upload_cmd, mesh_data, procs);
    ctx.SubmitAndWait(upload_cmd);
    REQUIRE(!gpu_buffers.empty());

    // ── Create G-buffer images ──
    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                          kWidth, kHeight, gbuf_cmd,
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    ctx.SubmitAndWait(gbuf_cmd);

    // ── Create Writer ──
    capture::WriterDesc writer_desc{};
    writer_desc.output_dir = kTestOutputDir;
    writer_desc.input_width = kWidth;
    writer_desc.input_height = kHeight;
    writer_desc.scale_mode = capture::ScaleMode::kNative;
    writer_desc.compression = capture::ExrCompression::kNone;
    auto writer = capture::Writer::Create(writer_desc);
    REQUIRE(writer);

    // ── Define viewpoints: 2 paths × 2 frames, deliberately out of order ──
    // Path "pathAAAA": frames 0 and 1 with lateral camera movement
    // Path "pathBBBB": frames 0 and 1 with forward camera movement
    // Inserted out of order to verify sorting works.
    glm::vec3 look = {0.5f, 0.5f, 0.0f};

    app::datagen::ViewpointEntry vp_b1{};  // pathBBBB frame 1 (inserted first!)
    vp_b1.position = {0.5f, 0.5f, 1.3f};  // moved forward from b0
    vp_b1.target = look;
    vp_b1.id = "pathBBBB_0001";
    vp_b1.path_id = "pathBBBB";
    vp_b1.frame = 1;

    app::datagen::ViewpointEntry vp_a0{};  // pathAAAA frame 0
    vp_a0.position = {0.5f, 0.5f, 1.5f};
    vp_a0.target = look;
    vp_a0.id = "pathAAAA_0000";
    vp_a0.path_id = "pathAAAA";
    vp_a0.frame = 0;

    app::datagen::ViewpointEntry vp_a1{};  // pathAAAA frame 1
    vp_a1.position = {0.6f, 0.5f, 1.5f};  // moved laterally
    vp_a1.target = look;
    vp_a1.id = "pathAAAA_0001";
    vp_a1.path_id = "pathAAAA";
    vp_a1.frame = 1;

    app::datagen::ViewpointEntry vp_b0{};  // pathBBBB frame 0
    vp_b0.position = {0.5f, 0.5f, 1.5f};
    vp_b0.target = look;
    vp_b0.id = "pathBBBB_0000";
    vp_b0.path_id = "pathBBBB";
    vp_b0.frame = 0;

    // ── Build GenerationConfig ──
    app::datagen::GenerationConfig gen_config{};
    gen_config.width = kWidth;
    gen_config.height = kHeight;
    gen_config.spp = 1;
    gen_config.ref_frames = 2;  // Minimal for speed
    gen_config.output_dir = kTestOutputDir;
    gen_config.capture_shader_dir = CAPTURE_SHADER_SPV_DIR;
    gen_config.force_write = true;  // Bypass luminance/NaN validation
    gen_config.viewpoints = {vp_b1, vp_a0, vp_a1, vp_b0};  // Scrambled order

    // ── Run the full production pipeline ──
    app::datagen::GenerationSession session(ctx, *renderer, gbuffer_images,
                                            *writer, scene, gen_config);
    REQUIRE(session.Run());

    // ── Verify output files exist ──
    // GenerationSession uses the original index `i` for subdirectory names.
    // After sorting by (path_id, frame), the render order is:
    //   order[0]=1 (pathAAAA f0, original index 1) → vp_1/
    //   order[1]=2 (pathAAAA f1, original index 2) → vp_2/
    //   order[2]=3 (pathBBBB f0, original index 3) → vp_3/
    //   order[3]=0 (pathBBBB f1, original index 0) → vp_0/
    REQUIRE(std::filesystem::exists(kTestOutputDir + "/vp_1/input.exr"));
    REQUIRE(std::filesystem::exists(kTestOutputDir + "/vp_2/input.exr"));
    REQUIRE(std::filesystem::exists(kTestOutputDir + "/vp_3/input.exr"));
    REQUIRE(std::filesystem::exists(kTestOutputDir + "/vp_0/input.exr"));

    uint32_t pixels = kWidth * kHeight;

    // ── pathAAAA frame 0 (vp_1/): first frame of path → near-zero MVs ──
    auto mv_a0 = ReadMotionVectors(kTestOutputDir + "/vp_1/input.exr");
    REQUIRE(mv_a0.valid);
    REQUIRE(mv_a0.max_magnitude < 1e-4f);

    // ── pathAAAA frame 1 (vp_2/): camera moved → non-zero MVs ──
    auto mv_a1 = ReadMotionVectors(kTestOutputDir + "/vp_2/input.exr");
    REQUIRE(mv_a1.valid);
    REQUIRE(mv_a1.max_magnitude > 0.001f);
    REQUIRE(mv_a1.nonzero_count > pixels / 10);

    // ── pathBBBB frame 0 (vp_3/): NEW path → temporal reset → near-zero MVs ──
    auto mv_b0 = ReadMotionVectors(kTestOutputDir + "/vp_3/input.exr");
    REQUIRE(mv_b0.valid);
    REQUIRE(mv_b0.max_magnitude < 1e-4f);

    // ── pathBBBB frame 1 (vp_0/): camera moved forward → non-zero MVs ──
    auto mv_b1 = ReadMotionVectors(kTestOutputDir + "/vp_0/input.exr");
    REQUIRE(mv_b1.valid);
    REQUIRE(mv_b1.max_magnitude > 0.001f);
    REQUIRE(mv_b1.nonzero_count > pixels / 10);

    // Cleanup GPU resources
    ctx.WaitIdle();
    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
}
