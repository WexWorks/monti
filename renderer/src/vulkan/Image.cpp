#include <volk.h>

#include "Image.h"

#include <cstdio>

namespace monti::vulkan {

Image::~Image() {
    Destroy();
}

Image::Image(Image&& other) noexcept
    : allocator_(other.allocator_),
      device_(other.device_),
      image_(other.image_),
      view_(other.view_),
      sampler_(other.sampler_),
      allocation_(other.allocation_),
      format_(other.format_),
      width_(other.width_),
      height_(other.height_),
      mip_levels_(other.mip_levels_) {
    other.allocator_ = VK_NULL_HANDLE;
    other.device_ = VK_NULL_HANDLE;
    other.image_ = VK_NULL_HANDLE;
    other.view_ = VK_NULL_HANDLE;
    other.sampler_ = VK_NULL_HANDLE;
    other.allocation_ = VK_NULL_HANDLE;
    other.format_ = VK_FORMAT_UNDEFINED;
    other.width_ = 0;
    other.height_ = 0;
    other.mip_levels_ = 1;
}

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        Destroy();
        allocator_ = other.allocator_;
        device_ = other.device_;
        image_ = other.image_;
        view_ = other.view_;
        sampler_ = other.sampler_;
        allocation_ = other.allocation_;
        format_ = other.format_;
        width_ = other.width_;
        height_ = other.height_;
        mip_levels_ = other.mip_levels_;
        other.allocator_ = VK_NULL_HANDLE;
        other.device_ = VK_NULL_HANDLE;
        other.image_ = VK_NULL_HANDLE;
        other.view_ = VK_NULL_HANDLE;
        other.sampler_ = VK_NULL_HANDLE;
        other.allocation_ = VK_NULL_HANDLE;
        other.format_ = VK_FORMAT_UNDEFINED;
        other.width_ = 0;
        other.height_ = 0;
        other.mip_levels_ = 1;
    }
    return *this;
}

bool Image::Create(VmaAllocator allocator, VkDevice device,
                   uint32_t width, uint32_t height, VkFormat format,
                   VkImageUsageFlags usage, VkImageAspectFlags aspect,
                   uint32_t mip_levels) {
    Destroy();

    allocator_ = allocator;
    device_ = device;
    format_ = format;
    width_ = width;
    height_ = height;
    mip_levels_ = mip_levels;

    VkImageCreateInfo image_ci{};
    image_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_ci.imageType = VK_IMAGE_TYPE_2D;
    image_ci.format = format;
    image_ci.extent = {width, height, 1};
    image_ci.mipLevels = mip_levels;
    image_ci.arrayLayers = 1;
    image_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_ci.usage = usage;
    image_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult result = vmaCreateImage(allocator_, &image_ci, &alloc_ci, &image_, &allocation_, nullptr);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "Image::Create failed (VkResult: %d)\n", result);
        return false;
    }

    VkImageViewCreateInfo view_ci{};
    view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image = image_;
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format = format;
    view_ci.subresourceRange.aspectMask = aspect;
    view_ci.subresourceRange.baseMipLevel = 0;
    view_ci.subresourceRange.levelCount = mip_levels;
    view_ci.subresourceRange.baseArrayLayer = 0;
    view_ci.subresourceRange.layerCount = 1;

    result = vkCreateImageView(device_, &view_ci, nullptr, &view_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "Image::Create view failed (VkResult: %d)\n", result);
        vmaDestroyImage(allocator_, image_, allocation_);
        image_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
        return false;
    }

    return true;
}

bool Image::CreateSampler(VkFilter mag_filter, VkFilter min_filter,
                          VkSamplerAddressMode wrap_u, VkSamplerAddressMode wrap_v,
                          float max_anisotropy) {
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_, sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }

    VkSamplerCreateInfo sampler_ci{};
    sampler_ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_ci.magFilter = mag_filter;
    sampler_ci.minFilter = min_filter;
    sampler_ci.mipmapMode = (min_filter == VK_FILTER_LINEAR)
                                ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler_ci.addressModeU = wrap_u;
    sampler_ci.addressModeV = wrap_v;
    sampler_ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_ci.anisotropyEnable = (max_anisotropy > 0.0f) ? VK_TRUE : VK_FALSE;
    sampler_ci.maxAnisotropy = max_anisotropy;
    sampler_ci.maxLod = static_cast<float>(mip_levels_);
    sampler_ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;

    VkResult result = vkCreateSampler(device_, &sampler_ci, nullptr, &sampler_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "Image::CreateSampler failed (VkResult: %d)\n", result);
        return false;
    }

    return true;
}

void Image::Destroy() {
    if (sampler_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_, sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }
    if (view_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_, view_, nullptr);
        view_ = VK_NULL_HANDLE;
    }
    if (image_ != VK_NULL_HANDLE && allocator_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, image_, allocation_);
        image_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
    }
}

}  // namespace monti::vulkan
