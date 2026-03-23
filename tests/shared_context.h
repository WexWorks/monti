#pragma once

#include "../app/core/vulkan_context.h"

#include <memory>

namespace monti::test {

// Process-wide shared VulkanContext for tests.
// Creating and destroying hundreds of VkInstance/VkDevice pairs in a single
// process exhausts driver-internal resources on some NVIDIA configurations.
// This singleton creates one VulkanContext on first use and reuses it for
// the lifetime of the test process.
inline monti::app::VulkanContext& SharedVulkanContext() {
    static auto ctx = std::make_unique<monti::app::VulkanContext>();
    [[maybe_unused]] static bool ok = [&] {
        return ctx->CreateInstance() && ctx->CreateDevice(std::nullopt);
    }();
    return *ctx;
}

}  // namespace monti::test
