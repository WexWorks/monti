#pragma once

#include <volk.h>

#include <cstdint>
#include <vector>

namespace monti::app {

class VulkanContext;

// Per-frame command pool, command buffer, fence, and semaphores.
// Triple-buffered by default (kFramesInFlight = 3).
class FrameResources {
public:
    static constexpr uint32_t kFramesInFlight = 3;

    FrameResources() = default;
    ~FrameResources();

    FrameResources(const FrameResources&) = delete;
    FrameResources& operator=(const FrameResources&) = delete;
    FrameResources(FrameResources&&) = delete;
    FrameResources& operator=(FrameResources&&) = delete;

    bool Create(VulkanContext& ctx);
    void Destroy();

    // Recreate render-finished semaphores when the swapchain image count changes.
    bool RecreateRenderFinishedSemaphores(uint32_t swapchain_image_count);

    VkCommandBuffer CommandBuffer(uint32_t frame) const { return command_buffers_[frame]; }
    VkFence InFlightFence(uint32_t frame) const { return in_flight_fences_[frame]; }
    VkSemaphore ImageAvailableSemaphore(uint32_t frame) const { return image_available_[frame]; }
    VkSemaphore RenderFinishedSemaphore(uint32_t image_index) const { return render_finished_[image_index]; }

    void WaitForFence(uint32_t frame) const;
    void ResetFence(uint32_t frame) const;
    void ResetCommandBuffer(uint32_t frame) const;

private:
    VulkanContext* ctx_ = nullptr;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers_;
    std::vector<VkFence> in_flight_fences_;
    std::vector<VkSemaphore> image_available_;
    std::vector<VkSemaphore> render_finished_;
};

}  // namespace monti::app
