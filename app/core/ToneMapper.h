#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <string_view>

namespace monti::app {

// Compute-based ACES filmic tone mapper with sRGB output.
// Reads an RGBA16F input (denoiser output in GENERAL layout),
// writes an RGBA16F output with sRGB-encoded values.
// After Apply(), the output image is in TRANSFER_SRC_OPTIMAL for blit to swapchain.
class ToneMapper {
public:
    ToneMapper() = default;
    ~ToneMapper();

    ToneMapper(const ToneMapper&) = delete;
    ToneMapper& operator=(const ToneMapper&) = delete;

    bool Create(VkDevice device, VmaAllocator allocator,
                std::string_view shader_dir,
                uint32_t width, uint32_t height,
                VkImageView hdr_input_view);
    void Destroy();

    bool Resize(uint32_t width, uint32_t height, VkImageView hdr_input_view);

    // Record tone mapping commands. Input image is expected in GENERAL layout.
    // Leaves output image in TRANSFER_SRC_OPTIMAL for blit to swapchain.
    void Apply(VkCommandBuffer cmd, VkImage hdr_input);

    VkImage OutputImage() const { return output_image_; }
    VkImageView OutputView() const { return output_view_; }

    void SetExposure(float exposure_ev) { exposure_ = exposure_ev; }
    float Exposure() const { return exposure_; }

    void SetAutoExposureMultiplier(float m) { auto_exposure_multiplier_ = m; }
    float AutoExposureMultiplier() const { return auto_exposure_multiplier_; }

private:
    bool CreateOutputImage(uint32_t width, uint32_t height);
    bool CreateDescriptorLayout();
    bool AllocateDescriptorSet();
    bool CreatePipeline(std::string_view shader_dir);
    void UpdateDescriptorSet(VkImageView hdr_input_view);
    void DestroyOutputImage();

    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;

    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;

    VkImage output_image_ = VK_NULL_HANDLE;
    VkImageView output_view_ = VK_NULL_HANDLE;
    VmaAllocation output_allocation_ = VK_NULL_HANDLE;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    float exposure_ = 0.0f;
    float auto_exposure_multiplier_ = 1.0f;
};

}  // namespace monti::app
