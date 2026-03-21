#include "MlInference.h"

#include <cstdio>
#include <cstring>

namespace deni::vulkan {

bool MlDeviceDispatch::Load(VkDevice device, PFN_vkGetDeviceProcAddr get_proc) {
    bool ok = true;
    auto resolve = [&](auto& fn_ptr, const char* name) {
        fn_ptr = reinterpret_cast<std::remove_reference_t<decltype(fn_ptr)>>(
            get_proc(device, name));
        if (!fn_ptr) {
            std::fprintf(stderr, "deni::MlInference: failed to resolve %s\n", name);
            ok = false;
        }
    };

    resolve(vkCreateBuffer,       "vkCreateBuffer");
    resolve(vkDestroyBuffer,      "vkDestroyBuffer");
    resolve(vkCreateImageView,    "vkCreateImageView");
    resolve(vkDestroyImageView,   "vkDestroyImageView");
    resolve(vkCmdCopyBuffer,      "vkCmdCopyBuffer");
    resolve(vkCmdPipelineBarrier2, "vkCmdPipelineBarrier2");

    return ok;
}

MlInference::MlInference(VkDevice device, VmaAllocator allocator,
                          PFN_vkGetDeviceProcAddr get_device_proc_addr,
                          uint32_t width, uint32_t height)
    : device_(device), allocator_(allocator), width_(width), height_(height) {
    dispatch_.Load(device, get_device_proc_addr);
    Resize(width, height);
}

MlInference::~MlInference() {
    DestroyFeatureMaps();
    DestroyWeightBuffers();

    // Free staging buffer if not yet freed
    if (staging_buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, staging_buffer_, staging_allocation_);
        staging_buffer_ = VK_NULL_HANDLE;
        staging_allocation_ = VK_NULL_HANDLE;
    }
}

bool MlInference::LoadWeights(const WeightData& weights, VkCommandBuffer cmd) {
    DestroyWeightBuffers();

    // Calculate total staging size
    VkDeviceSize total_size = 0;
    for (const auto& layer : weights.layers)
        total_size += layer.data.size() * sizeof(float);

    if (total_size == 0) {
        std::fprintf(stderr, "deni::MlInference: weight data is empty\n");
        return false;
    }

    // Create staging buffer
    VkBufferCreateInfo staging_ci{};
    staging_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    staging_ci.size = total_size;
    staging_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo staging_alloc_ci{};
    staging_alloc_ci.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    staging_alloc_ci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo staging_info{};
    VkResult result = vmaCreateBuffer(allocator_, &staging_ci, &staging_alloc_ci,
                                      &staging_buffer_, &staging_allocation_, &staging_info);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr,
                     "deni::MlInference: failed to create staging buffer (VkResult: %d)\n",
                     result);
        return false;
    }

    // Copy all layer data into staging buffer and create per-layer GPU buffers
    auto* staging_ptr = static_cast<char*>(staging_info.pMappedData);
    VkDeviceSize staging_offset = 0;

    weight_buffers_.reserve(weights.layers.size());

    for (const auto& layer : weights.layers) {
        VkDeviceSize layer_size = layer.data.size() * sizeof(float);

        // Copy to staging
        std::memcpy(staging_ptr + staging_offset, layer.data.data(), layer_size);

        // Create GPU storage buffer
        VkBufferCreateInfo buffer_ci{};
        buffer_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_ci.size = layer_size;
        buffer_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VmaAllocationCreateInfo alloc_ci{};
        alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        WeightBuffer wb;
        wb.name = layer.name;
        wb.size_bytes = layer_size;

        result = vmaCreateBuffer(allocator_, &buffer_ci, &alloc_ci,
                                 &wb.buffer, &wb.allocation, nullptr);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr,
                         "deni::MlInference: failed to create weight buffer for '%s' "
                         "(VkResult: %d)\n",
                         layer.name.c_str(), result);
            DestroyWeightBuffers();
            return false;
        }

        // Record copy from staging to GPU buffer
        VkBufferCopy copy_region{};
        copy_region.srcOffset = staging_offset;
        copy_region.dstOffset = 0;
        copy_region.size = layer_size;
        dispatch_.vkCmdCopyBuffer(cmd, staging_buffer_, wb.buffer, 1, &copy_region);

        weight_buffers_.push_back(std::move(wb));
        staging_offset += layer_size;
    }

    // Memory barrier: ensure all transfers complete before compute reads
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    dispatch_.vkCmdPipelineBarrier2(cmd, &dep);

    weights_loaded_ = true;
    return true;
}

void MlInference::FreeStagingBuffer() {
    if (staging_buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, staging_buffer_, staging_allocation_);
        staging_buffer_ = VK_NULL_HANDLE;
        staging_allocation_ = VK_NULL_HANDLE;
    }
}

bool MlInference::Resize(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) return false;
    if (width == width_ && height == height_ && features_allocated_) return true;

    DestroyFeatureMaps();

    width_ = width;
    height_ = height;

    // Level 0: full resolution, 16 channels
    if (!AllocateFeatureLevel(level0_, width, height, kLevel0Channels)) return false;
    if (!AllocateFeatureLevel(skip0_, width, height, kLevel0Channels)) return false;

    // Level 1: half resolution, 32 channels
    uint32_t w1 = (width + 1) / 2;
    uint32_t h1 = (height + 1) / 2;
    if (!AllocateFeatureLevel(level1_, w1, h1, kLevel1Channels)) return false;
    if (!AllocateFeatureLevel(skip1_, w1, h1, kLevel1Channels)) return false;

    // Level 2: quarter resolution, 64 channels
    uint32_t w2 = (w1 + 1) / 2;
    uint32_t h2 = (h1 + 1) / 2;
    if (!AllocateFeatureLevel(level2_, w2, h2, kLevel2Channels)) return false;

    features_allocated_ = true;
    return true;
}

bool MlInference::AllocateFeatureLevel(FeatureLevel& level, uint32_t width, uint32_t height,
                                        uint32_t channels) {
    level.width = width;
    level.height = height;
    level.channels = channels;
    level.image_count = channels / 4;  // 4 channels per RGBA16F image
    level.images_a.resize(level.image_count);
    level.images_b.resize(level.image_count);

    auto allocate_image = [&](FeatureImage& img) -> bool {
        VkImageCreateInfo image_ci{};
        image_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_ci.imageType = VK_IMAGE_TYPE_2D;
        image_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        image_ci.extent = {width, height, 1};
        image_ci.mipLevels = 1;
        image_ci.arrayLayers = 1;
        image_ci.samples = VK_SAMPLE_COUNT_1_BIT;
        image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT;
        image_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo alloc_ci{};
        alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        VkResult result = vmaCreateImage(allocator_, &image_ci, &alloc_ci,
                                         &img.image, &img.allocation, nullptr);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr,
                         "deni::MlInference: failed to create feature image %ux%u "
                         "(VkResult: %d)\n",
                         width, height, result);
            return false;
        }

        VkImageViewCreateInfo view_ci{};
        view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_ci.image = img.image;
        view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        result = dispatch_.vkCreateImageView(device_, &view_ci, nullptr, &img.view);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr,
                         "deni::MlInference: failed to create feature image view "
                         "(VkResult: %d)\n",
                         result);
            vmaDestroyImage(allocator_, img.image, img.allocation);
            img.image = VK_NULL_HANDLE;
            img.allocation = VK_NULL_HANDLE;
            return false;
        }

        return true;
    };

    for (uint32_t i = 0; i < level.image_count; ++i) {
        if (!allocate_image(level.images_a[i])) {
            DestroyFeatureLevel(level);
            return false;
        }
        if (!allocate_image(level.images_b[i])) {
            DestroyFeatureLevel(level);
            return false;
        }
    }

    return true;
}

void MlInference::DestroyWeightBuffers() {
    for (auto& wb : weight_buffers_) {
        if (wb.buffer != VK_NULL_HANDLE)
            vmaDestroyBuffer(allocator_, wb.buffer, wb.allocation);
    }
    weight_buffers_.clear();
    weights_loaded_ = false;
}

void MlInference::DestroyFeatureMaps() {
    DestroyFeatureLevel(level0_);
    DestroyFeatureLevel(level1_);
    DestroyFeatureLevel(level2_);
    DestroyFeatureLevel(skip0_);
    DestroyFeatureLevel(skip1_);
    features_allocated_ = false;
}

void MlInference::DestroyFeatureLevel(FeatureLevel& level) {
    auto destroy_image = [&](FeatureImage& img) {
        if (img.view != VK_NULL_HANDLE) {
            dispatch_.vkDestroyImageView(device_, img.view, nullptr);
            img.view = VK_NULL_HANDLE;
        }
        if (img.image != VK_NULL_HANDLE) {
            vmaDestroyImage(allocator_, img.image, img.allocation);
            img.image = VK_NULL_HANDLE;
            img.allocation = VK_NULL_HANDLE;
        }
    };

    for (auto& img : level.images_a) destroy_image(img);
    for (auto& img : level.images_b) destroy_image(img);
    level.images_a.clear();
    level.images_b.clear();
    level = {};
}

}  // namespace deni::vulkan
