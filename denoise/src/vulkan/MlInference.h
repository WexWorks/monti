#pragma once

#include "WeightLoader.h"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <string>
#include <vector>

namespace deni::vulkan {

// Forward-declared dispatch table (shared with Denoiser)
struct MlDeviceDispatch {
    PFN_vkCreateBuffer      vkCreateBuffer      = nullptr;
    PFN_vkDestroyBuffer     vkDestroyBuffer     = nullptr;
    PFN_vkCreateImageView   vkCreateImageView   = nullptr;
    PFN_vkDestroyImageView  vkDestroyImageView  = nullptr;
    PFN_vkCmdCopyBuffer     vkCmdCopyBuffer     = nullptr;
    PFN_vkCmdPipelineBarrier2 vkCmdPipelineBarrier2 = nullptr;

    bool Load(VkDevice device, PFN_vkGetDeviceProcAddr get_proc);
};

// GPU buffer holding weights for a single conv layer (kernel + bias concatenated)
struct WeightBuffer {
    std::string name;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkDeviceSize size_bytes = 0;
};

// Intermediate feature map image at one resolution level
struct FeatureImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
};

// Feature map set at a single resolution level (multiple RGBA16F images = 4 channels each)
struct FeatureLevel {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;        // Total channels (e.g. 16, 32, 64)
    uint32_t image_count = 0;     // channels / 4 (each RGBA16F image holds 4 channels)
    std::vector<FeatureImage> images_a;  // Ping buffer
    std::vector<FeatureImage> images_b;  // Pong buffer
};

// Manages GPU buffers and feature map images for ML inference.
// No inference dispatching yet — that is added in F11-2.
class MlInference {
public:
    MlInference(VkDevice device, VmaAllocator allocator,
                PFN_vkGetDeviceProcAddr get_device_proc_addr,
                uint32_t width, uint32_t height);
    ~MlInference();

    MlInference(const MlInference&) = delete;
    MlInference& operator=(const MlInference&) = delete;

    // Upload weight data to GPU storage buffers via staging buffer.
    // cmd must be a command buffer in the recording state. The caller must
    // submit and wait before the staging resources can be freed.
    bool LoadWeights(const WeightData& weights, VkCommandBuffer cmd);

    // (Re)allocate intermediate feature map images for the given resolution.
    bool Resize(uint32_t width, uint32_t height);

    // Free staging buffer after the transfer command buffer has completed.
    void FreeStagingBuffer();

    bool IsReady() const { return weights_loaded_ && features_allocated_; }
    uint32_t Width() const { return width_; }
    uint32_t Height() const { return height_; }
    uint32_t WeightBufferCount() const { return static_cast<uint32_t>(weight_buffers_.size()); }

    // U-Net architecture constants
    static constexpr uint32_t kLevel0Channels = 16;
    static constexpr uint32_t kLevel1Channels = 32;
    static constexpr uint32_t kLevel2Channels = 64;

private:
    void DestroyWeightBuffers();
    void DestroyFeatureMaps();
    bool AllocateFeatureLevel(FeatureLevel& level, uint32_t width, uint32_t height,
                              uint32_t channels);
    void DestroyFeatureLevel(FeatureLevel& level);

    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    MlDeviceDispatch dispatch_;

    // Weight storage — one buffer per layer
    std::vector<WeightBuffer> weight_buffers_;
    // Staging buffer for weight upload (freed after transfer completes)
    VkBuffer staging_buffer_ = VK_NULL_HANDLE;
    VmaAllocation staging_allocation_ = VK_NULL_HANDLE;

    // Intermediate feature maps at each U-Net level (ping-pong pairs)
    FeatureLevel level0_;  // Full resolution, 16 channels
    FeatureLevel level1_;  // Half resolution, 32 channels
    FeatureLevel level2_;  // Quarter resolution, 64 channels

    // Skip connection buffers (encoder outputs saved for decoder)
    FeatureLevel skip0_;   // Full resolution, 16 channels
    FeatureLevel skip1_;   // Half resolution, 32 channels

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    bool weights_loaded_ = false;
    bool features_allocated_ = false;
};

}  // namespace deni::vulkan
