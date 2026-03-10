#pragma once

#include <volk.h>

#include <cstdint>
#include <vector>

namespace monti::app {

class VulkanContext;

// Swapchain creation, recreation, and image management (monti_view only).
class Swapchain {
public:
    Swapchain() = default;
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&&) = delete;
    Swapchain& operator=(Swapchain&&) = delete;

    bool Create(VulkanContext& ctx, VkSurfaceKHR surface, uint32_t width, uint32_t height);
    void Destroy();

    VkSwapchainKHR Handle() const { return swapchain_; }
    VkFormat ImageFormat() const { return image_format_; }
    VkExtent2D Extent() const { return extent_; }
    uint32_t ImageCount() const { return static_cast<uint32_t>(images_.size()); }
    VkImageView ImageView(uint32_t index) const { return image_views_[index]; }
    VkImage Image(uint32_t index) const { return images_[index]; }

private:
    VulkanContext* ctx_ = nullptr;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat image_format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};

    std::vector<VkImage> images_;
    std::vector<VkImageView> image_views_;
};

}  // namespace monti::app
