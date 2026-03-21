#pragma once

#include <monti/capture/GpuReadback.h>

#include <cstdint>
#include <memory>
#include <string_view>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace monti::capture {

// Vulkan function pointers needed by GpuAccumulator beyond those in ReadbackContext.
struct AccumulatorProcs {
    PFN_vkCreateDescriptorSetLayout  pfn_vkCreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout pfn_vkDestroyDescriptorSetLayout;
    PFN_vkCreateDescriptorPool       pfn_vkCreateDescriptorPool;
    PFN_vkDestroyDescriptorPool      pfn_vkDestroyDescriptorPool;
    PFN_vkAllocateDescriptorSets     pfn_vkAllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets       pfn_vkUpdateDescriptorSets;
    PFN_vkCreateShaderModule         pfn_vkCreateShaderModule;
    PFN_vkDestroyShaderModule        pfn_vkDestroyShaderModule;
    PFN_vkCreatePipelineLayout       pfn_vkCreatePipelineLayout;
    PFN_vkDestroyPipelineLayout      pfn_vkDestroyPipelineLayout;
    PFN_vkCreateComputePipelines     pfn_vkCreateComputePipelines;
    PFN_vkDestroyPipeline            pfn_vkDestroyPipeline;
    PFN_vkCreateImageView            pfn_vkCreateImageView;
    PFN_vkDestroyImageView           pfn_vkDestroyImageView;
    PFN_vkCmdPipelineBarrier2        pfn_vkCmdPipelineBarrier2;
    PFN_vkCmdBindPipeline            pfn_vkCmdBindPipeline;
    PFN_vkCmdBindDescriptorSets      pfn_vkCmdBindDescriptorSets;
    PFN_vkCmdPushConstants           pfn_vkCmdPushConstants;
    PFN_vkCmdDispatch                pfn_vkCmdDispatch;
    PFN_vkCmdClearColorImage         pfn_vkCmdClearColorImage;
};

struct GpuAccumulatorDesc {
    VkDevice device;
    VmaAllocator allocator;
    uint32_t width;
    uint32_t height;
    std::string_view shader_dir;

    // Source images (RGBA16F) to accumulate from — handles are stable for the
    // lifetime of the GBufferImages instance.
    VkImage noisy_diffuse;
    VkImage noisy_specular;

    AccumulatorProcs procs;
};

// GPU-side reference frame accumulator. Replaces the CPU-side AccumulateFrames()
// by accumulating RGBA16F render outputs directly into RGBA32F storage images
// on the GPU, then reading back once at the end.
//
// Usage:
//   1. Create with source image handles and resolution.
//   2. Reset() in a command buffer to clear accumulators.
//   3. After each RenderFrame, call Accumulate() in the same command buffer.
//   4. After all frames, call Finalize() to read back the averaged result.
class GpuAccumulator {
public:
    ~GpuAccumulator();

    GpuAccumulator(const GpuAccumulator&) = delete;
    GpuAccumulator& operator=(const GpuAccumulator&) = delete;

    static std::unique_ptr<GpuAccumulator> Create(const GpuAccumulatorDesc& desc);

    // Clear accumulation images to zero. Must be called within an active command buffer.
    void Reset(VkCommandBuffer cmd);

    // Add current source images (scaled by weight) to accumulators. Must be called
    // within an active command buffer, after the render pass that produced the source
    // images. Pass weight = 1.0/num_frames to produce an averaged result directly.
    // Caller is responsible for barriers between RT output and compute read.
    void Accumulate(VkCommandBuffer cmd, float weight);

    // Read back accumulated images directly (no division — caller controls weighting
    // via Accumulate's weight parameter).
    MultiFrameResult Finalize(const ReadbackContext& ctx);

private:
    GpuAccumulator() = default;

    bool Init(const GpuAccumulatorDesc& desc);
    bool CreateAccumulationImages();
    bool CreateImageViews(VkImage noisy_diffuse, VkImage noisy_specular);
    bool CreateDescriptorResources();
    bool CreatePipeline(std::string_view shader_dir);
    void DestroyAccumulationImages();

    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    AccumulatorProcs procs_{};

    // FP32 accumulation images
    VkImage accum_diffuse_ = VK_NULL_HANDLE;
    VkImage accum_specular_ = VK_NULL_HANDLE;
    VmaAllocation accum_diffuse_alloc_ = VK_NULL_HANDLE;
    VmaAllocation accum_specular_alloc_ = VK_NULL_HANDLE;

    // Image views for descriptor binding
    VkImageView noisy_diffuse_view_ = VK_NULL_HANDLE;
    VkImageView noisy_specular_view_ = VK_NULL_HANDLE;
    VkImageView accum_diffuse_view_ = VK_NULL_HANDLE;
    VkImageView accum_specular_view_ = VK_NULL_HANDLE;

    // Compute pipeline
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet desc_set_ = VK_NULL_HANDLE;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
};

}  // namespace monti::capture
