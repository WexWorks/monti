#include <catch2/catch_test_macros.hpp>

// Include volk before vulkan_context to ensure VK_NO_PROTOTYPES is active
#include <volk.h>

#include "../app/core/vulkan_context.h"
#include "shared_context.h"

TEST_CASE("Headless VulkanContext creation and command submission", "[vulkan][integration]") {
    auto& ctx = monti::test::SharedVulkanContext();

    // Verify device is not null
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);
    REQUIRE(ctx.PhysicalDevice() != VK_NULL_HANDLE);
    REQUIRE(ctx.GraphicsQueue() != VK_NULL_HANDLE);
    REQUIRE(ctx.Allocator() != VK_NULL_HANDLE);

    // Verify RT properties are populated
    REQUIRE(ctx.RaytracePipelineProperties().shaderGroupHandleSize > 0);
    REQUIRE(ctx.RaytracePipelineProperties().maxRayRecursionDepth > 0);
    REQUIRE(ctx.AccelStructProperties().maxGeometryCount > 0);

    // Record a trivial command buffer (pipeline barrier), submit, and wait
    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(cmd != VK_NULL_HANDLE);

    VkMemoryBarrier2 memory_barrier{};
    memory_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    memory_barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    memory_barrier.srcAccessMask = 0;
    memory_barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    memory_barrier.dstAccessMask = 0;

    VkDependencyInfo dep_info{};
    dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep_info.memoryBarrierCount = 1;
    dep_info.pMemoryBarriers = &memory_barrier;
    vkCmdPipelineBarrier2(cmd, &dep_info);

    ctx.SubmitAndWait(cmd);

    // Clean shutdown
    ctx.WaitIdle();

    // Destructor runs here — should complete with zero validation errors
}
