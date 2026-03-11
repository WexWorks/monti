#include <volk.h>

#include "gbuffer_images.h"

#include <array>
#include <cstdio>

namespace monti::app {

namespace {

// Format table indexed by GBufferImages::Index
constexpr std::array<VkFormat, GBufferImages::kImageCount> kFormats = {{
    VK_FORMAT_R16G16B16A16_SFLOAT,     // kNoisyDiffuse
    VK_FORMAT_R16G16B16A16_SFLOAT,     // kNoisySpecular
    VK_FORMAT_R16G16_SFLOAT,           // kMotionVectors
    VK_FORMAT_R16_SFLOAT,              // kLinearDepth
    VK_FORMAT_R16G16B16A16_SFLOAT,     // kWorldNormals
    VK_FORMAT_B10G11R11_UFLOAT_PACK32, // kDiffuseAlbedo
    VK_FORMAT_B10G11R11_UFLOAT_PACK32, // kSpecularAlbedo
}};

}  // anonymous namespace

GBufferImages::~GBufferImages() {
    Destroy();
}

bool GBufferImages::CreateImage(VkFormat format, VkImageUsageFlags usage,
                                ImageEntry& out) {
    VkImageCreateInfo image_ci{};
    image_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_ci.imageType = VK_IMAGE_TYPE_2D;
    image_ci.format = format;
    image_ci.extent = {width_, height_, 1};
    image_ci.mipLevels = 1;
    image_ci.arrayLayers = 1;
    image_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_ci.usage = usage;
    image_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult result = vmaCreateImage(allocator_, &image_ci, &alloc_ci,
                                     &out.image, &out.allocation, nullptr);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GBufferImages: image creation failed (VkResult: %d)\n", result);
        return false;
    }

    VkImageViewCreateInfo view_ci{};
    view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image = out.image;
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format = format;
    view_ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_ci.subresourceRange.baseMipLevel = 0;
    view_ci.subresourceRange.levelCount = 1;
    view_ci.subresourceRange.baseArrayLayer = 0;
    view_ci.subresourceRange.layerCount = 1;

    result = vkCreateImageView(device_, &view_ci, nullptr, &out.view);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GBufferImages: image view creation failed (VkResult: %d)\n", result);
        vmaDestroyImage(allocator_, out.image, out.allocation);
        out.image = VK_NULL_HANDLE;
        out.allocation = VK_NULL_HANDLE;
        return false;
    }

    return true;
}

void GBufferImages::TransitionToGeneral(VkCommandBuffer cmd) {
    std::array<VkImageMemoryBarrier2, kImageCount> barriers{};
    for (uint32_t i = 0; i < kImageCount; ++i) {
        auto& b = barriers[i];
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        b.srcAccessMask = 0;
        b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = entries_[i].image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = kImageCount;
    dep.pImageMemoryBarriers = barriers.data();
    vkCmdPipelineBarrier2(cmd, &dep);
}

bool GBufferImages::Create(VmaAllocator allocator, VkDevice device,
                           uint32_t width, uint32_t height,
                           VkCommandBuffer cmd,
                           VkImageUsageFlags datagen_extra_usage) {
    Destroy();

    allocator_ = allocator;
    device_ = device;
    width_ = width;
    height_ = height;
    datagen_extra_usage_ = datagen_extra_usage;

    VkImageUsageFlags base_usage = VK_IMAGE_USAGE_STORAGE_BIT | datagen_extra_usage;

    for (uint32_t i = 0; i < kImageCount; ++i) {
        if (!CreateImage(kFormats[i], base_usage, entries_[i]))
            return false;
    }

    TransitionToGeneral(cmd);
    return true;
}

bool GBufferImages::Resize(uint32_t width, uint32_t height, VkCommandBuffer cmd) {
    if (width == width_ && height == height_) return true;
    return Create(allocator_, device_, width, height, cmd, datagen_extra_usage_);
}

void GBufferImages::DestroyEntry(ImageEntry& entry) {
    if (entry.view != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_, entry.view, nullptr);
        entry.view = VK_NULL_HANDLE;
    }
    if (entry.image != VK_NULL_HANDLE && allocator_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, entry.image, entry.allocation);
        entry.image = VK_NULL_HANDLE;
        entry.allocation = VK_NULL_HANDLE;
    }
}

void GBufferImages::Destroy() {
    for (auto& entry : entries_)
        DestroyEntry(entry);

    width_ = 0;
    height_ = 0;
}

}  // namespace monti::app
