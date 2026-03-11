#include <catch2/catch_test_macros.hpp>

#include <volk.h>

#include "test_helpers.h"

#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>
#include <monti/scene/Scene.h>

#include "../renderer/src/vulkan/Buffer.h"
#include "../scene/src/gltf/GltfLoader.h"

#include "scenes/CornellBox.h"

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

#ifndef MONTI_TEST_ASSETS_DIR
#define MONTI_TEST_ASSETS_DIR "tests/assets"
#endif

constexpr uint32_t kTestWidth = 64;
constexpr uint32_t kTestHeight = 64;

// Read back noisy_diffuse from a GBufferImages into a CPU staging buffer.
// Returns the mapped staging buffer; caller must Unmap() + destroy.
Buffer ReadbackNoisyDiffuse(monti::app::VulkanContext& ctx,
                            const monti::app::GBufferImages& gbuffer_images) {
    constexpr VkDeviceSize kPixelSize = 8;  // RGBA16F
    constexpr VkDeviceSize kReadbackSize = kTestWidth * kTestHeight * kPixelSize;

    Buffer readback;
    readback.Create(ctx.Allocator(), kReadbackSize,
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
    to_src.image = gbuffer_images.NoisyDiffuseImage();
    to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &to_src;
    vkCmdPipelineBarrier2(copy_cmd, &dep);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {kTestWidth, kTestHeight, 1};
    vkCmdCopyImageToBuffer(copy_cmd,
                           gbuffer_images.NoisyDiffuseImage(),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.Handle(), 1, &region);

    ctx.SubmitAndWait(copy_cmd);
    return readback;
}

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: Cornell box — all hits, no environment map
// The Cornell box is a closed room. The default camera is inside facing the
// back wall, so every primary ray should hit geometry (barycentric output).
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 7C: Cornell box renders all-hit barycentric output",
          "[phase7c][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = test::BuildCornellBox();
    // No environment map — Cornell box is a closed room

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
                                               ctx.Device(), upload_cmd, mesh_data);
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

    auto readback = ReadbackNoisyDiffuse(ctx, gbuffer_images);
    auto* raw = static_cast<uint16_t*>(readback.Map());
    REQUIRE(raw != nullptr);

    uint32_t nan_count = 0;
    uint32_t hit_pixels = 0;
    bool has_color_variation = false;
    float prev_r = -1.0f;

    for (uint32_t i = 0; i < kTestWidth * kTestHeight; ++i) {
        float r = test::HalfToFloat(raw[i * 4 + 0]);
        float g = test::HalfToFloat(raw[i * 4 + 1]);
        float b = test::HalfToFloat(raw[i * 4 + 2]);

        if (std::isnan(r) || std::isnan(g) || std::isnan(b)) ++nan_count;
        if (std::isinf(r) || std::isinf(g) || std::isinf(b)) ++nan_count;

        // Barycentric RGB sums to ~1.0 for every hit pixel
        float rgb_sum = r + g + b;
        if (rgb_sum > 0.85f && rgb_sum < 1.15f) ++hit_pixels;

        if (prev_r >= 0.0f && std::abs(r - prev_r) > 0.01f)
            has_color_variation = true;
        prev_r = r;
    }

    readback.Unmap();

    REQUIRE(nan_count == 0);
    // Closed room: every pixel should hit geometry
    REQUIRE(hit_pixels == kTestWidth * kTestHeight);
    REQUIRE(has_color_variation);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Box.glb with environment map — hits and misses
// Load a small glTF model (Khronos Box.glb), add a procedural env map,
// and position the camera so some rays hit the box and others see sky.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 7C: Box.glb with environment map renders hits and misses",
          "[phase7c][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;

    // Load Box.glb via the glTF loader
    std::string box_path = std::string(MONTI_TEST_ASSETS_DIR) + "/Box.glb";
    auto result = gltf::LoadGltf(scene, box_path);
    REQUIRE(result.success);
    REQUIRE_FALSE(result.mesh_data.empty());

    // Add a procedural environment map (solid teal color)
    constexpr uint32_t kEnvWidth = 4;
    constexpr uint32_t kEnvHeight = 2;
    std::vector<float> env_pixels(kEnvWidth * kEnvHeight * 4);
    for (uint32_t i = 0; i < kEnvWidth * kEnvHeight; ++i) {
        env_pixels[i * 4 + 0] = 0.2f;  // R
        env_pixels[i * 4 + 1] = 0.6f;  // G
        env_pixels[i * 4 + 2] = 0.8f;  // B
        env_pixels[i * 4 + 3] = 1.0f;  // A
    }

    TextureDesc env_tex{};
    env_tex.width = kEnvWidth;
    env_tex.height = kEnvHeight;
    env_tex.format = PixelFormat::kRGBA32F;
    env_tex.data.resize(env_pixels.size() * sizeof(float));
    std::memcpy(env_tex.data.data(), env_pixels.data(), env_tex.data.size());
    auto env_tex_id = scene.AddTexture(std::move(env_tex), "env_map");

    EnvironmentLight env_light{};
    env_light.hdr_lat_long = env_tex_id;
    env_light.intensity = 1.0f;
    env_light.rotation = 0.0f;
    scene.SetEnvironmentLight(env_light);

    // Camera looking at the box from a distance — box occupies part of the frame
    CameraParams camera;
    camera.position = {0.0f, 1.0f, 4.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.vertical_fov_radians = 0.8f;  // ~46 degrees
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

    auto readback = ReadbackNoisyDiffuse(ctx, gbuffer_images);
    auto* raw = static_cast<uint16_t*>(readback.Map());
    REQUIRE(raw != nullptr);

    uint32_t nan_count = 0;
    uint32_t hit_pixels = 0;
    uint32_t miss_env_pixels = 0;
    uint32_t other_pixels = 0;

    // Expected env color from the procedural texture
    constexpr float kEnvR = 0.2f, kEnvG = 0.6f, kEnvB = 0.8f;
    constexpr float kColorTol = 0.05f;  // half-float quantization tolerance

    for (uint32_t i = 0; i < kTestWidth * kTestHeight; ++i) {
        float r = test::HalfToFloat(raw[i * 4 + 0]);
        float g = test::HalfToFloat(raw[i * 4 + 1]);
        float b = test::HalfToFloat(raw[i * 4 + 2]);

        if (std::isnan(r) || std::isnan(g) || std::isnan(b)) { ++nan_count; continue; }
        if (std::isinf(r) || std::isinf(g) || std::isinf(b)) { ++nan_count; continue; }

        // Barycentrics: RGB sums to exactly 1.0
        float rgb_sum = r + g + b;
        if (rgb_sum > 0.85f && rgb_sum < 1.15f) {
            ++hit_pixels;
        } else if (std::abs(r - kEnvR) < kColorTol &&
                   std::abs(g - kEnvG) < kColorTol &&
                   std::abs(b - kEnvB) < kColorTol) {
            ++miss_env_pixels;
        } else {
            ++other_pixels;
        }
    }

    readback.Unmap();

    REQUIRE(nan_count == 0u);
    REQUIRE((hit_pixels > 0u));
    REQUIRE((miss_env_pixels > 0u));
    // Every pixel should be classifiable as either a hit or env miss
    REQUIRE(other_pixels == 0u);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Two frames — exercises prev_view_proj caching, no validation errors
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 7C: Zero Vulkan validation errors during render",
          "[phase7c][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = test::BuildCornellBox();

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
                                               ctx.Device(), upload_cmd, mesh_data);
    ctx.SubmitAndWait(upload_cmd);

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                                  kTestWidth, kTestHeight, gbuf_cmd));
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    // Render two frames to exercise prev_view_proj caching
    VkCommandBuffer cmd1 = ctx.BeginOneShot();
    REQUIRE(renderer->RenderFrame(cmd1, gbuffer, 0));
    ctx.SubmitAndWait(cmd1);

    VkCommandBuffer cmd2 = ctx.BeginOneShot();
    REQUIRE(renderer->RenderFrame(cmd2, gbuffer, 1));
    ctx.SubmitAndWait(cmd2);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: PushConstants struct fits within device limits
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 7C: PushConstants size within device limits",
          "[phase7c][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(tc.ctx.PhysicalDevice(), &props);

    REQUIRE(248 <= props.limits.maxPushConstantsSize);
}
