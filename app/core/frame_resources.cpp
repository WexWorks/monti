#include "frame_resources.h"

#include "vulkan_context.h"

#include <cstdio>

namespace monti::app {

FrameResources::~FrameResources() {
    Destroy();
}

bool FrameResources::Create(VulkanContext& ctx) {
    ctx_ = &ctx;

    // Command pool
    VkCommandPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_ci.queueFamilyIndex = ctx.QueueFamilyIndex();

    VkResult result = vkCreateCommandPool(ctx.Device(), &pool_ci, nullptr, &command_pool_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to create command pool (VkResult: %d)\n", result);
        return false;
    }

    // Command buffers
    command_buffers_.resize(kFramesInFlight);
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = kFramesInFlight;

    result = vkAllocateCommandBuffers(ctx.Device(), &alloc_info, command_buffers_.data());
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to allocate command buffers (VkResult: %d)\n", result);
        return false;
    }

    // Fences (start signaled so first frame doesn't block)
    VkFenceCreateInfo fence_ci{};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    in_flight_fences_.resize(kFramesInFlight);
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        result = vkCreateFence(ctx.Device(), &fence_ci, nullptr, &in_flight_fences_[i]);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr, "Failed to create fence %u (VkResult: %d)\n", i, result);
            return false;
        }
    }

    // Image-available semaphores (per frame in flight)
    VkSemaphoreCreateInfo sem_ci{};
    sem_ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    image_available_.resize(kFramesInFlight);
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        result = vkCreateSemaphore(ctx.Device(), &sem_ci, nullptr, &image_available_[i]);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr, "Failed to create image-available semaphore %u (VkResult: %d)\n", i, result);
            return false;
        }
    }

    return true;
}

bool FrameResources::RecreateRenderFinishedSemaphores(uint32_t swapchain_image_count) {
    // Destroy old render-finished semaphores
    for (auto sem : render_finished_)
        vkDestroySemaphore(ctx_->Device(), sem, nullptr);
    render_finished_.clear();

    VkSemaphoreCreateInfo sem_ci{};
    sem_ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    render_finished_.resize(swapchain_image_count);
    for (uint32_t i = 0; i < swapchain_image_count; ++i) {
        VkResult result = vkCreateSemaphore(ctx_->Device(), &sem_ci, nullptr, &render_finished_[i]);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr, "Failed to create render-finished semaphore %u (VkResult: %d)\n", i, result);
            return false;
        }
    }

    return true;
}

void FrameResources::WaitForFence(uint32_t frame) const {
    vkWaitForFences(ctx_->Device(), 1, &in_flight_fences_[frame], VK_TRUE, UINT64_MAX);
}

void FrameResources::ResetFence(uint32_t frame) const {
    vkResetFences(ctx_->Device(), 1, &in_flight_fences_[frame]);
}

void FrameResources::ResetCommandBuffer(uint32_t frame) const {
    vkResetCommandBuffer(command_buffers_[frame], 0);
}

void FrameResources::Destroy() {
    if (!ctx_ || ctx_->Device() == VK_NULL_HANDLE) return;

    for (auto sem : image_available_)
        vkDestroySemaphore(ctx_->Device(), sem, nullptr);
    image_available_.clear();

    for (auto sem : render_finished_)
        vkDestroySemaphore(ctx_->Device(), sem, nullptr);
    render_finished_.clear();

    for (auto fence : in_flight_fences_)
        vkDestroyFence(ctx_->Device(), fence, nullptr);
    in_flight_fences_.clear();

    command_buffers_.clear();

    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(ctx_->Device(), command_pool_, nullptr);
        command_pool_ = VK_NULL_HANDLE;
    }
}

}  // namespace monti::app
