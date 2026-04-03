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
    PFN_vkCmdFillBuffer              pfn_vkCmdFillBuffer;
    PFN_vkCmdCopyBuffer              pfn_vkCmdCopyBuffer;
    PFN_vkCreateBuffer               pfn_vkCreateBuffer;
    PFN_vkDestroyBuffer              pfn_vkDestroyBuffer;
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

    bool adaptive_sampling = false;
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

    // Add current source images (raw sum) to accumulators and increment per-pixel
    // sample count. Must be called within an active command buffer, after the render
    // pass that produced the source images.
    // Caller is responsible for barriers between RT output and compute read.
    void Accumulate(VkCommandBuffer cmd);

    // Dispatch finalize.comp to divide accumulators by per-pixel sample count,
    // then read back the normalized result.
    MultiFrameResult FinalizeNormalized(const ReadbackContext& ctx);

    // Read back accumulated images directly (raw sums, no normalization).
    MultiFrameResult Finalize(const ReadbackContext& ctx);

    // Dispatch variance_update.comp to update Welford statistics from the
    // current noisy output. Only available when adaptive_sampling is enabled.
    void UpdateVariance(VkCommandBuffer cmd);

    // Dispatch convergence_check.comp to evaluate convergence and update the mask.
    // Returns the number of converged pixels (read back from the atomic counter).
    // Only available when adaptive_sampling is enabled.
    uint32_t CheckConvergence(VkCommandBuffer cmd, uint32_t min_frames, float threshold,
                              const ReadbackContext& ctx);

    // Access the convergence mask image view (R8UI). For binding to the renderer's
    // raygen shader (binding 17). Returns VK_NULL_HANDLE if adaptive is disabled.
    VkImageView ConvergenceMaskView() const { return convergence_mask_view_; }

    // Access the convergence mask image handle (for barriers).
    VkImage ConvergenceMaskImage() const { return convergence_mask_; }

    // Access Welford variance images (for readback in tests/diagnostics).
    // Returns VK_NULL_HANDLE if adaptive is disabled.
    VkImage VarianceMeanImage() const { return variance_mean_; }
    VkImage VarianceM2Image() const { return variance_m2_; }

private:
    GpuAccumulator() = default;

    bool Init(const GpuAccumulatorDesc& desc);
    bool CreateAccumulationImages();
    bool CreateSampleCountImage();
    bool CreateImageViews(VkImage noisy_diffuse, VkImage noisy_specular);
    bool CreateDescriptorResources();
    bool CreateAccumulatePipeline(std::string_view shader_dir);
    bool CreateFinalizePipeline(std::string_view shader_dir);
    bool CreateVarianceImages();
    bool CreateConvergenceMaskImage();
    bool CreateConvergedCounterBuffer();
    bool CreateVarianceUpdatePipeline(std::string_view shader_dir);
    bool CreateConvergenceCheckPipeline(std::string_view shader_dir);
    bool CreateVarianceDescriptorResources();
    void DestroyAccumulationImages();
    void DispatchFinalize(VkCommandBuffer cmd);

    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    AccumulatorProcs procs_{};

    // FP32 accumulation images
    VkImage accum_diffuse_ = VK_NULL_HANDLE;
    VkImage accum_specular_ = VK_NULL_HANDLE;
    VmaAllocation accum_diffuse_alloc_ = VK_NULL_HANDLE;
    VmaAllocation accum_specular_alloc_ = VK_NULL_HANDLE;

    // Per-pixel sample count (R32UI)
    VkImage sample_count_ = VK_NULL_HANDLE;
    VmaAllocation sample_count_alloc_ = VK_NULL_HANDLE;
    VkImageView sample_count_view_ = VK_NULL_HANDLE;

    // Image views for descriptor binding
    VkImageView noisy_diffuse_view_ = VK_NULL_HANDLE;
    VkImageView noisy_specular_view_ = VK_NULL_HANDLE;
    VkImageView accum_diffuse_view_ = VK_NULL_HANDLE;
    VkImageView accum_specular_view_ = VK_NULL_HANDLE;

    // Accumulate compute pipeline
    VkPipeline accumulate_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout accumulate_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout accumulate_desc_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool accumulate_desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet accumulate_desc_set_ = VK_NULL_HANDLE;

    // Finalize compute pipeline
    VkPipeline finalize_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout finalize_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout finalize_desc_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool finalize_desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet finalize_desc_set_ = VK_NULL_HANDLE;

    // ── Adaptive sampling resources (only created when adaptive_sampling=true) ──

    // Welford variance images
    VkImage variance_mean_ = VK_NULL_HANDLE;
    VmaAllocation variance_mean_alloc_ = VK_NULL_HANDLE;
    VkImageView variance_mean_view_ = VK_NULL_HANDLE;

    VkImage variance_m2_ = VK_NULL_HANDLE;
    VmaAllocation variance_m2_alloc_ = VK_NULL_HANDLE;
    VkImageView variance_m2_view_ = VK_NULL_HANDLE;

    // Convergence mask (R8UI)
    VkImage convergence_mask_ = VK_NULL_HANDLE;
    VmaAllocation convergence_mask_alloc_ = VK_NULL_HANDLE;
    VkImageView convergence_mask_view_ = VK_NULL_HANDLE;

    // Atomic counter buffer for converged pixel count
    VkBuffer converged_count_buffer_ = VK_NULL_HANDLE;
    VmaAllocation converged_count_alloc_ = VK_NULL_HANDLE;

    // Variance update compute pipeline
    VkPipeline variance_update_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout variance_update_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout variance_update_desc_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool variance_update_desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet variance_update_desc_set_ = VK_NULL_HANDLE;

    // Convergence check compute pipeline
    VkPipeline convergence_check_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout convergence_check_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout convergence_check_desc_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool convergence_check_desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet convergence_check_desc_set_ = VK_NULL_HANDLE;

    bool adaptive_enabled_ = false;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
};

}  // namespace monti::capture
