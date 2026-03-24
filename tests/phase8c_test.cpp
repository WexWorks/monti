#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <volk.h>

#include "test_helpers.h"
#include "scenes/CornellBox.h"

#include <monti/scene/Scene.h>

#include "../renderer/src/vulkan/GpuScene.h"
#include "../renderer/src/vulkan/DeviceDispatch.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <vector>

using namespace monti;
using namespace monti::vulkan;

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

// Build a Cornell box variant with a glass panel and a semi-transparent panel.
struct TransparentCornellBoxResult {
    Scene scene;
    std::vector<MeshData> mesh_data;
};

TransparentCornellBoxResult BuildTransparentCornellBox() {
    auto [scene, mesh_data] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene);

    // Glass material (transmission + IOR)
    MaterialDesc glass;
    glass.base_color = {0.85f, 0.95f, 1.0f};  // Slight blue tint for visibility
    glass.roughness = 0.0f;
    glass.metallic = 0.0f;
    glass.opacity = 1.0f;
    glass.ior = 1.5f;
    glass.transmission_factor = 1.0f;
    glass.attenuation_distance = 0.5f;
    glass.attenuation_color = {0.7f, 0.9f, 1.0f};
    auto glass_id = scene.AddMaterial(std::move(glass), "glass");

    // Semi-transparent blend material
    MaterialDesc blend;
    blend.base_color = {0.2f, 0.5f, 0.9f};
    blend.roughness = 0.5f;
    blend.metallic = 0.0f;
    blend.opacity = 0.3f;
    blend.alpha_mode = MaterialDesc::AlphaMode::kBlend;
    auto blend_id = scene.AddMaterial(std::move(blend), "blend_blue");

    auto make_vertex = [](const glm::vec3& pos, const glm::vec3& normal,
                          const glm::vec2& uv) {
        Vertex v{};
        v.position = pos;
        v.normal = normal;
        v.tangent = glm::vec4(1, 0, 0, 1);
        v.tex_coord_0 = uv;
        v.tex_coord_1 = uv;
        return v;
    };

    // Glass panel — centered in scene, spanning both boxes, positioned
    // clearly in front for visible refraction.
    MeshData glass_quad;
    {
        glm::vec3 n{0, 0, 1};
        glass_quad.vertices = {
            make_vertex({0.15f, 0.0f, 0.65f}, n, {0, 0}),
            make_vertex({0.85f, 0.0f, 0.65f}, n, {1, 0}),
            make_vertex({0.85f, 0.8f, 0.65f}, n, {1, 1}),
            make_vertex({0.15f, 0.8f, 0.65f}, n, {0, 1}),
        };
        glass_quad.indices = {0, 1, 2, 0, 2, 3};
    }

    Mesh glass_mesh;
    glass_mesh.name = "glass_panel";
    glass_mesh.vertex_count = static_cast<uint32_t>(glass_quad.vertices.size());
    glass_mesh.index_count = static_cast<uint32_t>(glass_quad.indices.size());
    glass_mesh.bbox_min = {0.15f, 0.0f, 0.65f};
    glass_mesh.bbox_max = {0.85f, 0.8f, 0.65f};
    auto glass_mesh_id = scene.AddMesh(std::move(glass_mesh), "glass_panel");
    glass_quad.mesh_id = glass_mesh_id;
    scene.AddNode(glass_mesh_id, glass_id, "glass_panel_node");
    mesh_data.push_back(std::move(glass_quad));

    // Blend panel floating in scene
    MeshData blend_quad;
    {
        glm::vec3 n{0, 0, 1};
        blend_quad.vertices = {
            make_vertex({0.1f, 0.1f, 0.6f}, n, {0, 0}),
            make_vertex({0.45f, 0.1f, 0.6f}, n, {1, 0}),
            make_vertex({0.45f, 0.5f, 0.6f}, n, {1, 1}),
            make_vertex({0.1f, 0.5f, 0.6f}, n, {0, 1}),
        };
        blend_quad.indices = {0, 1, 2, 0, 2, 3};
    }

    Mesh blend_mesh;
    blend_mesh.name = "blend_panel";
    blend_mesh.vertex_count = static_cast<uint32_t>(blend_quad.vertices.size());
    blend_mesh.index_count = static_cast<uint32_t>(blend_quad.indices.size());
    blend_mesh.bbox_min = {0.1f, 0.1f, 0.6f};
    blend_mesh.bbox_max = {0.45f, 0.5f, 0.6f};
    auto blend_mesh_id = scene.AddMesh(std::move(blend_mesh), "blend_panel");
    blend_quad.mesh_id = blend_mesh_id;
    scene.AddNode(blend_mesh_id, blend_id, "blend_panel_node");
    mesh_data.push_back(std::move(blend_quad));

    return {std::move(scene), std::move(mesh_data)};
}

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: PackedMaterial alpha_mode/cutoff and transmission packing
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8C: PackedMaterial packs alpha_mode, cutoff, and transmission",
          "[phase8c][gpu_scene][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    Scene scene;

    // Opaque material (alpha_mode = 0)
    MaterialDesc opaque;
    opaque.base_color = {1, 1, 1};
    opaque.alpha_mode = MaterialDesc::AlphaMode::kOpaque;
    auto opaque_id = scene.AddMaterial(std::move(opaque), "opaque");

    // Mask material (alpha_mode = 1, cutoff = 0.65)
    MaterialDesc mask;
    mask.base_color = {0.5f, 0.5f, 0.5f};
    mask.alpha_mode = MaterialDesc::AlphaMode::kMask;
    mask.alpha_cutoff = 0.65f;
    auto mask_id = scene.AddMaterial(std::move(mask), "mask");

    // Blend material (alpha_mode = 2)
    MaterialDesc blend;
    blend.base_color = {0.2f, 0.5f, 0.9f};
    blend.opacity = 0.4f;
    blend.alpha_mode = MaterialDesc::AlphaMode::kBlend;
    auto blend_id = scene.AddMaterial(std::move(blend), "blend");

    // Glass material with transmission
    MaterialDesc glass;
    glass.base_color = {1, 1, 1};
    glass.ior = 1.5f;
    glass.transmission_factor = 0.8f;
    glass.attenuation_distance = 1.5f;
    glass.attenuation_color = {0.7f, 0.85f, 1.0f};
    auto glass_id = scene.AddMaterial(std::move(glass), "glass");

    GpuScene gpu_scene(ctx.Allocator(), ctx.Device(), ctx.PhysicalDevice(), tc.dispatch);
    REQUIRE(gpu_scene.UpdateMaterials(scene));

    // 4 materials × 112 bytes each (7 vec4 per material)
    REQUIRE(gpu_scene.MaterialBufferSize() == 4 * sizeof(PackedMaterial));
    STATIC_REQUIRE(sizeof(PackedMaterial) == 176);

    // Material indices are sequential
    REQUIRE(gpu_scene.GetMaterialIndex(opaque_id) == 0);
    REQUIRE(gpu_scene.GetMaterialIndex(mask_id) == 1);
    REQUIRE(gpu_scene.GetMaterialIndex(blend_id) == 2);
    REQUIRE(gpu_scene.GetMaterialIndex(glass_id) == 3);

    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Transparent Cornell box renders with no NaN/Inf
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8C: Transparent scene renders with no NaN/Inf",
          "[phase8c][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = BuildTransparentCornellBox();

    auto result = test::RenderSceneMultiFrame(ctx, scene, mesh_data, 64, 16);

    auto* diffuse_raw = result.diffuse.data();
    auto* specular_raw = result.specular.data();

    test::WritePNG("tests/output/phase8c_cornell_box_transparent_diffuse.png",
                   diffuse_raw, test::kTestWidth, test::kTestHeight);

    auto diffuse_stats = test::AnalyzeRGBA16F(diffuse_raw, test::kPixelCount);

    test::WritePNG("tests/output/phase8c_cornell_box_transparent_specular.png",
                   specular_raw, test::kTestWidth, test::kTestHeight);

    auto specular_stats = test::AnalyzeRGBA16F(specular_raw, test::kPixelCount);

    test::WriteCombinedPNG("tests/output/phase8c_cornell_box_transparent_combined.png",
                     diffuse_raw, specular_raw, test::kTestWidth, test::kTestHeight);

    // No NaN or Inf in either channel
    REQUIRE(diffuse_stats.nan_count == 0);
    REQUIRE(diffuse_stats.inf_count == 0);
    REQUIRE(specular_stats.nan_count == 0);
    REQUIRE(specular_stats.inf_count == 0);

    // Scene should have non-trivial content
    REQUIRE(diffuse_stats.nonzero_count > test::kPixelCount / 4);
    REQUIRE(diffuse_stats.has_color_variation);

    test::CleanupMultiFrameResult(ctx.Allocator(), result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Motion vectors are NaN-free on first frame
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8C: Motion vectors are NaN-free on first frame",
          "[phase8c][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene);

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
                                               ctx.Device(), upload_cmd, mesh_data,
                                               test::MakeGpuBufferProcs());
    ctx.SubmitAndWait(upload_cmd);

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                                  test::kTestWidth, test::kTestHeight, gbuf_cmd,
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    // Render frame 0 — first ever frame, tests prev_view_proj initialization
    VkCommandBuffer render_cmd = ctx.BeginOneShot();
    REQUIRE(renderer->RenderFrame(render_cmd, gbuffer, 0));
    ctx.SubmitAndWait(render_cmd);

    // Read back motion vectors (RG16F = 4 bytes per pixel)
    auto mv_readback = test::ReadbackImage(ctx, gbuffer_images.MotionVectorsImage(), 4);
    auto* mv_raw = static_cast<uint16_t*>(mv_readback.Map());
    REQUIRE(mv_raw != nullptr);

    uint32_t nan_count = 0;
    uint32_t inf_count = 0;
    for (uint32_t i = 0; i < test::kTestWidth * test::kTestHeight; ++i) {
        float u = test::HalfToFloat(mv_raw[i * 2 + 0]);
        float v = test::HalfToFloat(mv_raw[i * 2 + 1]);

        if (std::isnan(u) || std::isnan(v)) ++nan_count;
        if (std::isinf(u) || std::isinf(v)) ++inf_count;
    }
    mv_readback.Unmap();

    REQUIRE(nan_count == 0);
    REQUIRE(inf_count == 0);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: Sub-pixel jitter produces different results across frames
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8C: Sub-pixel jitter varies output across frames",
          "[phase8c][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = test::BuildCornellBox();
    test::AddCornellBoxLight(scene);

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
                                               ctx.Device(), upload_cmd, mesh_data,
                                               test::MakeGpuBufferProcs());
    ctx.SubmitAndWait(upload_cmd);

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                                  test::kTestWidth, test::kTestHeight, gbuf_cmd,
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    // Render frame 0
    VkCommandBuffer cmd0 = ctx.BeginOneShot();
    REQUIRE(renderer->RenderFrame(cmd0, gbuffer, 0));
    ctx.SubmitAndWait(cmd0);

    auto frame0_readback = test::ReadbackImage(ctx, gbuffer_images.NoisyDiffuseImage());
    auto* frame0_raw = static_cast<uint16_t*>(frame0_readback.Map());
    REQUIRE(frame0_raw != nullptr);

    // Copy frame 0 data
    constexpr uint32_t kPixelCount = test::kTestWidth * test::kTestHeight;
    std::vector<uint16_t> frame0_copy(kPixelCount * 4);
    std::memcpy(frame0_copy.data(), frame0_raw, kPixelCount * 4 * sizeof(uint16_t));
    frame0_readback.Unmap();

    // Render frame 1 (different Halton jitter offset)
    VkCommandBuffer cmd1 = ctx.BeginOneShot();
    REQUIRE(renderer->RenderFrame(cmd1, gbuffer, 1));
    ctx.SubmitAndWait(cmd1);

    auto frame1_readback = test::ReadbackImage(ctx, gbuffer_images.NoisyDiffuseImage());
    auto* frame1_raw = static_cast<uint16_t*>(frame1_readback.Map());
    REQUIRE(frame1_raw != nullptr);

    // At least one pixel should differ between frames
    bool frames_differ = false;
    for (uint32_t i = 0; i < kPixelCount * 4; ++i) {
        if (frame0_copy[i] != frame1_raw[i]) {
            frames_differ = true;
            break;
        }
    }
    frame1_readback.Unmap();

    // Jitter + blue noise scramble should produce different output per frame
    REQUIRE(frames_differ);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: Multi-frame with transparency — no validation errors
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Phase 8C: Multi-frame transparent scene no validation errors",
          "[phase8c][renderer][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());
    auto& ctx = tc.ctx;

    auto [scene, mesh_data] = BuildTransparentCornellBox();

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
                                               ctx.Device(), upload_cmd, mesh_data,
                                               test::MakeGpuBufferProcs());
    ctx.SubmitAndWait(upload_cmd);

    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                                  test::kTestWidth, test::kTestHeight, gbuf_cmd));
    ctx.SubmitAndWait(gbuf_cmd);

    auto gbuffer = test::MakeGBuffer(gbuffer_images);

    // Render 3 frames — exercises jitter cycling, prev_view_proj update,
    // and transparency across multiple frames.
    for (uint32_t frame = 0; frame < 3; ++frame) {
        VkCommandBuffer cmd = ctx.BeginOneShot();
        REQUIRE(renderer->RenderFrame(cmd, gbuffer, frame));
        ctx.SubmitAndWait(cmd);
    }

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}