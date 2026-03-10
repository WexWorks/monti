#include "swapchain.h"

#include "../core/vulkan_context.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace monti::app {

Swapchain::~Swapchain() {
    Destroy();
}

bool Swapchain::Create(VulkanContext& ctx, VkSurfaceKHR surface, uint32_t width, uint32_t height) {
    ctx_ = &ctx;
    surface_ = surface;

    // Query surface capabilities
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx.PhysicalDevice(), surface_, &capabilities);

    // Choose surface format (prefer BGRA8 SRGB)
    uint32_t format_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.PhysicalDevice(), surface_, &format_count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.PhysicalDevice(), surface_, &format_count, formats.data());

    VkSurfaceFormatKHR chosen_format = formats[0];
    for (const auto& fmt : formats) {
        if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB &&
            fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen_format = fmt;
            break;
        }
    }
    image_format_ = chosen_format.format;

    // Choose extent
    if (capabilities.currentExtent.width != UINT32_MAX) {
        extent_ = capabilities.currentExtent;
    } else {
        extent_.width = std::clamp(width, capabilities.minImageExtent.width,
                                   capabilities.maxImageExtent.width);
        extent_.height = std::clamp(height, capabilities.minImageExtent.height,
                                    capabilities.maxImageExtent.height);
    }

    // Triple buffering: request 3 images, clamped to supported range
    uint32_t image_count = std::max(3u, capabilities.minImageCount);
    if (capabilities.maxImageCount > 0)
        image_count = std::min(image_count, capabilities.maxImageCount);

    VkSwapchainCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = surface_;
    ci.minImageCount = image_count;
    ci.imageFormat = chosen_format.format;
    ci.imageColorSpace = chosen_format.colorSpace;
    ci.imageExtent = extent_;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = capabilities.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    // Prefer MAILBOX, fall back to FIFO
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    uint32_t mode_count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.PhysicalDevice(), surface_,
                                              &mode_count, nullptr);
    std::vector<VkPresentModeKHR> modes(mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.PhysicalDevice(), surface_,
                                              &mode_count, modes.data());
    for (auto mode : modes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            present_mode = mode;
            break;
        }
    }
    ci.presentMode = present_mode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = swapchain_;

    VkSwapchainKHR new_swapchain;
    VkResult result = vkCreateSwapchainKHR(ctx.Device(), &ci, nullptr, &new_swapchain);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to create swapchain (VkResult: %d)\n", result);
        return false;
    }

    // Destroy old swapchain resources
    for (auto view : image_views_)
        vkDestroyImageView(ctx.Device(), view, nullptr);
    image_views_.clear();
    images_.clear();

    if (swapchain_ != VK_NULL_HANDLE)
        vkDestroySwapchainKHR(ctx.Device(), swapchain_, nullptr);

    swapchain_ = new_swapchain;

    // Get swapchain images
    uint32_t actual_count = 0;
    vkGetSwapchainImagesKHR(ctx.Device(), swapchain_, &actual_count, nullptr);
    images_.resize(actual_count);
    vkGetSwapchainImagesKHR(ctx.Device(), swapchain_, &actual_count, images_.data());

    // Create image views
    image_views_.resize(actual_count);
    for (uint32_t i = 0; i < actual_count; ++i) {
        VkImageViewCreateInfo view_ci{};
        view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_ci.image = images_[i];
        view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_ci.format = image_format_;
        view_ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_ci.subresourceRange.baseMipLevel = 0;
        view_ci.subresourceRange.levelCount = 1;
        view_ci.subresourceRange.baseArrayLayer = 0;
        view_ci.subresourceRange.layerCount = 1;

        result = vkCreateImageView(ctx.Device(), &view_ci, nullptr, &image_views_[i]);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr, "Failed to create swapchain image view %u (VkResult: %d)\n", i, result);
            return false;
        }
    }

    std::printf("Swapchain created: %ux%u, %u images\n", extent_.width, extent_.height, actual_count);
    return true;
}

void Swapchain::Destroy() {
    if (!ctx_ || ctx_->Device() == VK_NULL_HANDLE) return;

    for (auto view : image_views_)
        vkDestroyImageView(ctx_->Device(), view, nullptr);
    image_views_.clear();
    images_.clear();

    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(ctx_->Device(), swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

}  // namespace monti::app
