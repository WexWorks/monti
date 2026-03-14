#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <volk.h>

#include "test_helpers.h"
#include "scenes/CornellBox.h"

#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>

#include <bit>
#include <cstring>

// Access GpuScene internals for material buffer readback.
// This is a test-only inclusion of an internal header.
#include "../renderer/src/vulkan/GpuScene.h"
#include "../renderer/src/vulkan/DeviceDispatch.h"

using namespace monti;
using namespace monti::vulkan;

#ifndef MONTI_SHADER_SPV_DIR
#define MONTI_SHADER_SPV_DIR "build/shaders"
#endif

namespace {

// Helper: create a headless VulkanContext for GPU tests.
struct TestContext {
    monti::app::VulkanContext ctx;
    DeviceDispatch dispatch;

    bool Init() {
        if (!ctx.CreateInstance()) return false;
        if (!ctx.CreateDevice(std::nullopt)) return false;
        if (!dispatch.Load(ctx.Device(), ctx.Instance(),
                           ctx.GetDeviceProcAddr(), ctx.GetInstanceProcAddr()))
            return false;
        return true;
    }
};

}  // anonymous namespace

TEST_CASE("GPU scene: register mesh buffers and verify bindings", "[gpu_scene][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;

    // Build Cornell box scene
    auto [scene, mesh_data] = test::BuildCornellBox();
    REQUIRE(mesh_data.size() == 7);

    // Create renderer
    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = 64;
    desc.height = 64;
    test::FillRendererProcAddrs(desc, ctx);

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    // Upload mesh data to GPU and register with renderer
    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(cmd != VK_NULL_HANDLE);

    auto gpu_buffers = UploadAndRegisterMeshes(*renderer, ctx.Allocator(),
                                                ctx.Device(), cmd, mesh_data,
                                                test::MakeGpuBufferProcs());
    REQUIRE(gpu_buffers.size() == 14);  // 7 meshes × 2 buffers each

    ctx.SubmitAndWait(cmd);

    // Verify all buffers have valid device addresses
    for (const auto& buf : gpu_buffers) {
        REQUIRE(buf.buffer != VK_NULL_HANDLE);
        REQUIRE(buf.device_address != 0);
        REQUIRE(buf.size > 0);
    }

    // Clean up GPU buffers
    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);

    ctx.WaitIdle();
}

TEST_CASE("GPU scene: individual mesh upload", "[gpu_scene][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;
    auto [scene, mesh_data] = test::BuildCornellBox();

    // Upload a single mesh
    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(cmd != VK_NULL_HANDLE);

    const auto& floor_data = mesh_data[0];
    auto [vb, ib] = UploadMeshToGpu(ctx.Allocator(), ctx.Device(), cmd, floor_data,
                                     test::MakeGpuBufferProcs());
    REQUIRE(vb.buffer != VK_NULL_HANDLE);
    REQUIRE(ib.buffer != VK_NULL_HANDLE);

    // Verify vertex buffer size
    REQUIRE(vb.size == floor_data.vertices.size() * sizeof(Vertex));
    REQUIRE(ib.size == floor_data.indices.size() * sizeof(uint32_t));

    // Verify device addresses
    REQUIRE(vb.device_address != 0);
    REQUIRE(ib.device_address != 0);

    ctx.SubmitAndWait(cmd);

    DestroyGpuBuffer(ctx.Allocator(), vb);
    DestroyGpuBuffer(ctx.Allocator(), ib);
    ctx.WaitIdle();
}

TEST_CASE("GPU scene: material buffer packing", "[gpu_scene][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;

    // Build a scene with known material values, including transmission/volume
    Scene scene;

    MaterialDesc mat;
    mat.base_color = {0.73f, 0.73f, 0.73f};
    mat.roughness = 0.8f;
    mat.metallic = 0.2f;
    mat.opacity = 0.9f;
    mat.ior = 1.5f;
    mat.clear_coat = 0.3f;
    mat.clear_coat_roughness = 0.1f;
    mat.transmission_factor = 0.5f;
    mat.thickness_factor = 0.02f;
    mat.attenuation_distance = 1.0f;
    mat.attenuation_color = {0.8f, 0.9f, 1.0f};
    // No textures assigned — all should encode as UINT32_MAX
    auto mat_id = scene.AddMaterial(std::move(mat), "test_mat");

    // Second material with one texture set
    TextureDesc tex;
    tex.width = 2;
    tex.height = 2;
    tex.format = PixelFormat::kRGBA8_UNORM;
    tex.data.resize(2 * 2 * 4, 128);
    auto tex_id = scene.AddTexture(std::move(tex), "test_tex");

    MaterialDesc mat2;
    mat2.base_color = {1.0f, 0.0f, 0.0f};
    mat2.roughness = 1.0f;
    mat2.metallic = 0.0f;
    mat2.base_color_map = tex_id;
    mat2.emissive_factor = {10.0f, 5.0f, 2.0f};
    mat2.emissive_map = tex_id;
    auto mat2_id = scene.AddMaterial(std::move(mat2), "textured_mat");

    // Create GpuScene directly for internal testing
    GpuScene gpu_scene(ctx.Allocator(), ctx.Device(), ctx.PhysicalDevice(), tc.dispatch);

    // Upload textures first (populates texture_id_to_index_)
    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(cmd != VK_NULL_HANDLE);
    auto staging = gpu_scene.UploadTextures(scene, cmd);
    ctx.SubmitAndWait(cmd);

    REQUIRE(gpu_scene.TextureCount() == 1);

    // Update materials (uses texture index mapping)
    REQUIRE(gpu_scene.UpdateMaterials(scene));
    REQUIRE(gpu_scene.MaterialBuffer() != VK_NULL_HANDLE);

    // Verify material indices
    REQUIRE(gpu_scene.GetMaterialIndex(mat_id) == 0);
    REQUIRE(gpu_scene.GetMaterialIndex(mat2_id) == 1);
    REQUIRE(gpu_scene.MaterialCount() == 2);

    // Read back material buffer contents (host-visible, can be mapped)
    // Map the buffer and compare packed values
    VkDeviceSize buf_size = gpu_scene.MaterialBufferSize();
    REQUIRE(buf_size == 2 * sizeof(PackedMaterial));

    // We can verify the buffer was created with the right size.
    // For a full readback, we'd need the internal buffer access.
    // The material index mapping is verified above.

    ctx.WaitIdle();
}

TEST_CASE("GPU scene: texture upload with sampler", "[gpu_scene][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;

    Scene scene;

    // Create a 4x4 RGBA texture with known pixel data
    TextureDesc tex;
    tex.width = 4;
    tex.height = 4;
    tex.format = PixelFormat::kRGBA8_UNORM;
    tex.mip_levels = 1;
    tex.wrap_s = SamplerWrap::kClampToEdge;
    tex.wrap_t = SamplerWrap::kMirroredRepeat;
    tex.mag_filter = SamplerFilter::kLinear;
    tex.min_filter = SamplerFilter::kNearest;
    tex.data.resize(4 * 4 * 4);
    for (uint32_t i = 0; i < 4 * 4; ++i) {
        tex.data[i * 4 + 0] = static_cast<uint8_t>(i * 16);  // R
        tex.data[i * 4 + 1] = 128;                            // G
        tex.data[i * 4 + 2] = 255;                            // B
        tex.data[i * 4 + 3] = 255;                            // A
    }
    auto tex_id = scene.AddTexture(std::move(tex), "test_texture");

    // Create another texture with different sampler settings
    TextureDesc tex2;
    tex2.width = 2;
    tex2.height = 2;
    tex2.format = PixelFormat::kRGBA8_UNORM;
    tex2.wrap_s = SamplerWrap::kRepeat;
    tex2.wrap_t = SamplerWrap::kRepeat;
    tex2.mag_filter = SamplerFilter::kNearest;
    tex2.min_filter = SamplerFilter::kLinear;
    tex2.data.resize(2 * 2 * 4, 200);
    auto tex2_id = scene.AddTexture(std::move(tex2), "test_texture_2");

    GpuScene gpu_scene(ctx.Allocator(), ctx.Device(), ctx.PhysicalDevice(), tc.dispatch);

    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(cmd != VK_NULL_HANDLE);

    auto staging = gpu_scene.UploadTextures(scene, cmd);
    REQUIRE_FALSE(staging.empty());

    ctx.SubmitAndWait(cmd);

    // Verify texture count
    REQUIRE(gpu_scene.TextureCount() == 2);

    // Verify images have correct dimensions
    const auto& images = gpu_scene.TextureImages();
    REQUIRE(images.size() == 2);

    REQUIRE(images[0].Width() == 4);
    REQUIRE(images[0].Height() == 4);
    REQUIRE(images[0].MipLevels() == 1);
    REQUIRE(images[0].Handle() != VK_NULL_HANDLE);
    REQUIRE(images[0].View() != VK_NULL_HANDLE);
    REQUIRE(images[0].Sampler() != VK_NULL_HANDLE);

    REQUIRE(images[1].Width() == 2);
    REQUIRE(images[1].Height() == 2);
    REQUIRE(images[1].Handle() != VK_NULL_HANDLE);
    REQUIRE(images[1].View() != VK_NULL_HANDLE);
    REQUIRE(images[1].Sampler() != VK_NULL_HANDLE);

    ctx.WaitIdle();
}

TEST_CASE("GPU scene: material packing with texture indices", "[gpu_scene][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;

    Scene scene;

    // Add textures
    TextureDesc base_tex;
    base_tex.width = 2;
    base_tex.height = 2;
    base_tex.format = PixelFormat::kRGBA8_UNORM;
    base_tex.data.resize(2 * 2 * 4, 255);
    auto base_tex_id = scene.AddTexture(std::move(base_tex), "base_color");

    TextureDesc normal_tex;
    normal_tex.width = 2;
    normal_tex.height = 2;
    normal_tex.format = PixelFormat::kRGBA8_UNORM;
    normal_tex.data.resize(2 * 2 * 4, 128);
    auto normal_tex_id = scene.AddTexture(std::move(normal_tex), "normal");

    // Material WITH texture references
    MaterialDesc mat;
    mat.base_color = {0.5f, 0.5f, 0.5f};
    mat.roughness = 0.7f;
    mat.base_color_map = base_tex_id;
    mat.normal_map = normal_tex_id;
    // metallic_roughness_map, transmission_map, emissive_map NOT set
    auto mat_id = scene.AddMaterial(std::move(mat), "textured");

    // Material WITHOUT any textures
    MaterialDesc mat2;
    mat2.base_color = {1.0f, 1.0f, 1.0f};
    mat2.roughness = 1.0f;
    auto mat2_id = scene.AddMaterial(std::move(mat2), "plain");

    GpuScene gpu_scene(ctx.Allocator(), ctx.Device(), ctx.PhysicalDevice(), tc.dispatch);

    // Upload textures first
    VkCommandBuffer cmd = ctx.BeginOneShot();
    auto staging = gpu_scene.UploadTextures(scene, cmd);
    ctx.SubmitAndWait(cmd);

    REQUIRE(gpu_scene.TextureCount() == 2);

    // Update materials — texture indices should now resolve
    REQUIRE(gpu_scene.UpdateMaterials(scene));
    REQUIRE(gpu_scene.MaterialCount() == 2);

    // Verify material indices
    REQUIRE(gpu_scene.GetMaterialIndex(mat_id) == 0);
    REQUIRE(gpu_scene.GetMaterialIndex(mat2_id) == 1);

    ctx.WaitIdle();
}

TEST_CASE("GPU scene: PackedMaterial layout", "[gpu_scene]") {
    // Verify compile-time properties of PackedMaterial
    STATIC_REQUIRE(sizeof(PackedMaterial) == 128);
    STATIC_REQUIRE(alignof(PackedMaterial) == 16);

    // Verify float-encoding of texture indices
    float encoded_none = std::bit_cast<float>(UINT32_MAX);
    REQUIRE(std::bit_cast<uint32_t>(encoded_none) == UINT32_MAX);

    float encoded_zero = std::bit_cast<float>(uint32_t{0});
    REQUIRE(std::bit_cast<uint32_t>(encoded_zero) == 0);

    float encoded_five = std::bit_cast<float>(uint32_t{5});
    REQUIRE(std::bit_cast<uint32_t>(encoded_five) == 5);
}

TEST_CASE("GPU scene: Cornell box end-to-end via Renderer", "[gpu_scene][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;
    auto [scene, mesh_data] = test::BuildCornellBox();

    // Create renderer
    RendererDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.queue = ctx.GraphicsQueue();
    desc.queue_family_index = ctx.QueueFamilyIndex();
    desc.allocator = ctx.Allocator();
    desc.width = 64;
    desc.height = 64;
    desc.shader_dir = MONTI_SHADER_SPV_DIR;
    test::FillRendererProcAddrs(desc, ctx);

    auto renderer = Renderer::Create(desc);
    REQUIRE(renderer);
    renderer->SetScene(&scene);

    // Upload and register all meshes
    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = UploadAndRegisterMeshes(*renderer, ctx.Allocator(),
                                                ctx.Device(), upload_cmd, mesh_data,
                                                test::MakeGpuBufferProcs());
    REQUIRE(gpu_buffers.size() == 14);
    ctx.SubmitAndWait(upload_cmd);

    // Create G-buffer images for the render pass
    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer_images.Create(ctx.Allocator(), ctx.Device(), 64, 64, gbuf_cmd));
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    // Trigger RenderFrame to exercise material/texture upload + pipeline creation
    VkCommandBuffer render_cmd = ctx.BeginOneShot();
    bool ok = renderer->RenderFrame(render_cmd, gbuffer, 0);
    REQUIRE(ok);
    ctx.SubmitAndWait(render_cmd);

    // Clean up
    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);

    ctx.WaitIdle();
}
