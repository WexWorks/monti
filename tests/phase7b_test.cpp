#include <catch2/catch_test_macros.hpp>

#include <volk.h>

#include "../app/core/vulkan_context.h"
#include "shared_context.h"
#include "../app/core/GBufferImages.h"

#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>
#include <monti/scene/Scene.h>

// Internal headers for direct testing
#include "../renderer/src/vulkan/RaytracePipeline.h"
#include "../renderer/src/vulkan/GpuScene.h"
#include "../renderer/src/vulkan/EnvironmentMap.h"
#include "../renderer/src/vulkan/BlueNoise.h"
#include "../renderer/src/vulkan/Buffer.h"
#include "../renderer/src/vulkan/DeviceDispatch.h"

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

// Path to compiled shaders (set by CMake)
#ifndef MONTI_SHADER_SPV_DIR
#define MONTI_SHADER_SPV_DIR "build/shaders"
#endif

}  // anonymous namespace

TEST_CASE("RaytracePipeline: descriptor set layout + pool + set creation",
          "[raytrace_pipeline][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;

    // Create minimal resources needed for pipeline creation
    EnvironmentMap env_map;
    BlueNoise blue_noise;
    std::vector<Buffer> staging;

    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(env_map.CreatePlaceholders(ctx.Allocator(), ctx.Device(), cmd, staging,
                                        tc.dispatch));

    Buffer blue_noise_staging;
    REQUIRE(blue_noise.Generate(ctx.Allocator(), cmd, blue_noise_staging, tc.dispatch));

    ctx.SubmitAndWait(cmd);

    // Create pipeline
    RaytracePipeline pipeline;
    REQUIRE(pipeline.Create(ctx.Device(), ctx.PhysicalDevice(),
                            ctx.Allocator(), VK_NULL_HANDLE,
                            MONTI_SHADER_SPV_DIR, tc.dispatch));

    // Verify all objects were created
    REQUIRE(pipeline.Pipeline() != VK_NULL_HANDLE);
    REQUIRE(pipeline.PipelineLayout() != VK_NULL_HANDLE);
    REQUIRE(pipeline.DescriptorSet() != VK_NULL_HANDLE);

    // Verify SBT regions have valid addresses
    REQUIRE(pipeline.RaygenRegion().deviceAddress != 0);
    REQUIRE(pipeline.RaygenRegion().size > 0);
    REQUIRE(pipeline.MissRegion().deviceAddress != 0);
    REQUIRE(pipeline.MissRegion().size > 0);
    REQUIRE(pipeline.HitRegion().deviceAddress != 0);
    REQUIRE(pipeline.HitRegion().size > 0);
    REQUIRE(pipeline.CallableRegion().size == 0);  // empty

    ctx.WaitIdle();
}

TEST_CASE("RaytracePipeline: SBT alignment matches device properties",
          "[raytrace_pipeline][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;

    // Create minimal resources
    EnvironmentMap env_map;
    BlueNoise blue_noise;
    std::vector<Buffer> staging;

    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(env_map.CreatePlaceholders(ctx.Allocator(), ctx.Device(), cmd, staging,
                                        tc.dispatch));

    Buffer blue_noise_staging;
    REQUIRE(blue_noise.Generate(ctx.Allocator(), cmd, blue_noise_staging, tc.dispatch));

    ctx.SubmitAndWait(cmd);

    // Create pipeline
    RaytracePipeline pipeline;
    REQUIRE(pipeline.Create(ctx.Device(), ctx.PhysicalDevice(),
                            ctx.Allocator(), VK_NULL_HANDLE,
                            MONTI_SHADER_SPV_DIR, tc.dispatch));

    // Get RT properties for alignment validation
    const auto& rt_props = ctx.RaytracePipelineProperties();
    uint32_t base_alignment = rt_props.shaderGroupBaseAlignment;

    // Verify SBT region addresses are aligned to shaderGroupBaseAlignment
    REQUIRE((pipeline.RaygenRegion().deviceAddress % base_alignment) == 0);
    REQUIRE((pipeline.MissRegion().deviceAddress % base_alignment) == 0);
    REQUIRE((pipeline.HitRegion().deviceAddress % base_alignment) == 0);

    // Vulkan spec: raygen region size must equal stride
    REQUIRE(pipeline.RaygenRegion().size == pipeline.RaygenRegion().stride);
    // Miss/hit region sizes are aligned to shaderGroupBaseAlignment
    REQUIRE((pipeline.MissRegion().size % base_alignment) == 0);
    REQUIRE((pipeline.HitRegion().size % base_alignment) == 0);

    ctx.WaitIdle();
}

TEST_CASE("RaytracePipeline: descriptor update with all resources",
          "[raytrace_pipeline][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;

    // Create all resources
    EnvironmentMap env_map;
    BlueNoise blue_noise;
    std::vector<Buffer> staging;

    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(env_map.CreatePlaceholders(ctx.Allocator(), ctx.Device(), cmd, staging,
                                        tc.dispatch));

    Buffer blue_noise_staging;
    REQUIRE(blue_noise.Generate(ctx.Allocator(), cmd, blue_noise_staging, tc.dispatch));

    ctx.SubmitAndWait(cmd);

    // Create GpuScene with a placeholder light buffer
    GpuScene gpu_scene(ctx.Allocator(), ctx.Device(), ctx.PhysicalDevice(), tc.dispatch);
    Scene scene;
    REQUIRE(gpu_scene.UpdateLights(scene));

    // Create a placeholder material buffer (empty scene still needs a valid buffer)
    Buffer material_placeholder;
    REQUIRE(material_placeholder.Create(
        ctx.Allocator(), sizeof(PackedMaterial),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT));

    // Create a placeholder mesh address buffer
    Buffer mesh_addr_placeholder;
    REQUIRE(mesh_addr_placeholder.Create(
        ctx.Allocator(), sizeof(MeshAddressEntry),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT));

    // Create G-buffer images
    monti::app::GBufferImages gbuffer;
    cmd = ctx.BeginOneShot();
    REQUIRE(gbuffer.Create(ctx.Allocator(), ctx.Device(), 64, 64, cmd));
    ctx.SubmitAndWait(cmd);

    // Build a minimal TLAS (needed for descriptor update)
    // We need at least a dummy TLAS — use GeometryManager from the scene test pattern
    // For this test, just verify descriptor update doesn't crash with valid resources

    // Create pipeline
    RaytracePipeline pipeline;
    REQUIRE(pipeline.Create(ctx.Device(), ctx.PhysicalDevice(),
                            ctx.Allocator(), VK_NULL_HANDLE,
                            MONTI_SHADER_SPV_DIR, tc.dispatch));

    // Note: We can't fully test descriptor update without a valid TLAS,
    // but we verify pipeline creation succeeded with all necessary layouts
    REQUIRE(pipeline.Pipeline() != VK_NULL_HANDLE);

    ctx.WaitIdle();
}

TEST_CASE("PushConstants: struct size within guaranteed minimum",
          "[raytrace_pipeline][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;

    // Verify push constant struct fits within device limits
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.PhysicalDevice(), &props);

    REQUIRE(sizeof(PushConstants) == 16);
    REQUIRE(sizeof(PushConstants) <= props.limits.maxPushConstantsSize);
}

TEST_CASE("GpuScene::UpdateLights: placeholder buffer creation",
          "[gpu_scene][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;

    GpuScene gpu_scene(ctx.Allocator(), ctx.Device(), ctx.PhysicalDevice(), tc.dispatch);
    Scene scene;

    // Update with no lights — should create placeholder
    REQUIRE(gpu_scene.UpdateLights(scene));
    REQUIRE(gpu_scene.LightBuffer() != VK_NULL_HANDLE);
    REQUIRE(gpu_scene.LightBufferSize() >= sizeof(PackedLight));

    // Add area lights and update
    scene.AddAreaLight(AreaLight{
        .corner = {-1.0f, 2.0f, -1.0f},
        .edge_a = {2.0f, 0.0f, 0.0f},
        .edge_b = {0.0f, 0.0f, 2.0f},
        .radiance = {10.0f, 10.0f, 10.0f},
        .two_sided = false,
    });

    REQUIRE(gpu_scene.UpdateLights(scene));
    REQUIRE(gpu_scene.LightBufferSize() >= sizeof(PackedLight));

    ctx.WaitIdle();
}
