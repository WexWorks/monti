#include "../core/vulkan_context.h"

#include <cstdio>
#include <cstdlib>
#include <optional>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    monti::app::VulkanContext ctx;

    if (!ctx.CreateInstance()) return EXIT_FAILURE;
    if (!ctx.CreateDevice(std::nullopt)) return EXIT_FAILURE;

    // Print device name
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx.PhysicalDevice(), &props);
    std::printf("monti_datagen: device = %s\n", props.deviceName);

    ctx.WaitIdle();
    return EXIT_SUCCESS;
}
