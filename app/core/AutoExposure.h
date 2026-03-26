#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <string_view>

namespace monti::app {

// GPU-based auto-exposure via log-average luminance with temporal smoothing.
// Computes scene luminance from the HDR input (denoiser output) and produces
// an exposure multiplier that maps the average luminance to mid-gray (0.18).
//
// The result SSBO is host-visible with persistent mapping — the CPU reads the
// previous frame's adapted luminance (N-1 latency), avoiding GPU-CPU sync
// stalls. This latency is invisible due to temporal smoothing.
class AutoExposure {
public:
    AutoExposure() = default;
    ~AutoExposure();

    AutoExposure(const AutoExposure&) = delete;
    AutoExposure& operator=(const AutoExposure&) = delete;

    bool Create(VkDevice device, VmaAllocator allocator,
                std::string_view shader_dir,
                uint32_t width, uint32_t height,
                VkImageView hdr_input_view);
    void Destroy();
    bool Resize(uint32_t width, uint32_t height, VkImageView hdr_input_view);

    // Record luminance compute dispatches into cmd. Must be called before
    // ToneMapper::Apply() in the same command buffer.
    void Compute(VkCommandBuffer cmd, VkImage hdr_input, float delta_time);

    // Returns 0.18 / adapted_luminance (the multiplier to apply to HDR).
    float ExposureMultiplier() const;

    // Read adapted luminance value directly (for UI display).
    float AdaptedLuminance() const;

    void SetAdaptationSpeed(float speed) { adaptation_speed_ = speed; }
    float AdaptationSpeed() const { return adaptation_speed_; }

private:
    bool CreateAccumBuffer();
    bool CreateResultBuffer();
    bool CreateDescriptorLayout();
    bool AllocateDescriptorSets();
    bool CreatePipelines(std::string_view shader_dir);
    void UpdateDescriptorSets(VkImageView hdr_input_view);

    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;

    // Luminance accumulation pipeline (16x16 workgroups)
    VkPipeline accum_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout accum_pipeline_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout accum_descriptor_layout_ = VK_NULL_HANDLE;
    VkDescriptorSet accum_descriptor_set_ = VK_NULL_HANDLE;

    // Luminance resolve pipeline (1x1x1)
    VkPipeline resolve_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout resolve_pipeline_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout resolve_descriptor_layout_ = VK_NULL_HANDLE;
    VkDescriptorSet resolve_descriptor_set_ = VK_NULL_HANDLE;

    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    // Accumulation SSBO: {uint log_sum_fixed, uint pixel_count} — device-local
    VkBuffer accum_buffer_ = VK_NULL_HANDLE;
    VmaAllocation accum_allocation_ = VK_NULL_HANDLE;

    // Result SSBO: {float adapted_luminance} — host-visible, persistently mapped
    VkBuffer result_buffer_ = VK_NULL_HANDLE;
    VmaAllocation result_allocation_ = VK_NULL_HANDLE;
    float* result_mapped_ = nullptr;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    float adaptation_speed_ = 5.0f;
    bool first_frame_ = true;
};

}  // namespace monti::app
