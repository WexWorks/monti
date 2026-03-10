#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstdint>

namespace monti::vulkan {

// RAII wrapper for a VMA-allocated Vulkan image with view and optional sampler.
// Supports mip generation via vkCmdBlitImage and per-texture VkSampler creation.
class Image {
public:
    Image() = default;
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    bool Create(VmaAllocator allocator, VkDevice device,
                uint32_t width, uint32_t height, VkFormat format,
                VkImageUsageFlags usage,
                VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                uint32_t mip_levels = 1);

    // Create a VkSampler for this image from specified sampler parameters.
    // max_anisotropy should be capped to VkPhysicalDeviceLimits::maxSamplerAnisotropy.
    // Pass 0.0f to disable anisotropic filtering.
    bool CreateSampler(VkFilter mag_filter, VkFilter min_filter,
                       VkSamplerAddressMode wrap_u, VkSamplerAddressMode wrap_v,
                       float max_anisotropy = 16.0f);

    void Destroy();

    VkImage Handle() const { return image_; }
    VkImageView View() const { return view_; }
    VkSampler Sampler() const { return sampler_; }
    VkFormat Format() const { return format_; }
    uint32_t Width() const { return width_; }
    uint32_t Height() const { return height_; }
    uint32_t MipLevels() const { return mip_levels_; }

private:
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkImage image_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t mip_levels_ = 1;
};

}  // namespace monti::vulkan
