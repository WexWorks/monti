#pragma once

#include "WeightLoader.h"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace deni::vulkan {

struct DenoiserInput;

// Vulkan device dispatch table for ML inference operations
struct MlDeviceDispatch {
    PFN_vkCreateBuffer        vkCreateBuffer        = nullptr;
    PFN_vkDestroyBuffer       vkDestroyBuffer       = nullptr;
    PFN_vkCreateImageView     vkCreateImageView     = nullptr;
    PFN_vkDestroyImageView    vkDestroyImageView    = nullptr;
    PFN_vkCmdCopyBuffer       vkCmdCopyBuffer       = nullptr;
    PFN_vkCmdPipelineBarrier2 vkCmdPipelineBarrier2 = nullptr;

    // Pipeline / descriptor management (added for F11-2)
    PFN_vkCreateShaderModule          vkCreateShaderModule         = nullptr;
    PFN_vkDestroyShaderModule         vkDestroyShaderModule        = nullptr;
    PFN_vkCreatePipelineLayout        vkCreatePipelineLayout       = nullptr;
    PFN_vkDestroyPipelineLayout       vkDestroyPipelineLayout      = nullptr;
    PFN_vkCreateComputePipelines      vkCreateComputePipelines     = nullptr;
    PFN_vkDestroyPipeline             vkDestroyPipeline            = nullptr;
    PFN_vkCreateDescriptorSetLayout   vkCreateDescriptorSetLayout  = nullptr;
    PFN_vkDestroyDescriptorSetLayout  vkDestroyDescriptorSetLayout = nullptr;
    PFN_vkCreateDescriptorPool        vkCreateDescriptorPool       = nullptr;
    PFN_vkDestroyDescriptorPool       vkDestroyDescriptorPool      = nullptr;
    PFN_vkAllocateDescriptorSets      vkAllocateDescriptorSets     = nullptr;
    PFN_vkUpdateDescriptorSets        vkUpdateDescriptorSets       = nullptr;
    PFN_vkCmdBindPipeline             vkCmdBindPipeline            = nullptr;
    PFN_vkCmdBindDescriptorSets       vkCmdBindDescriptorSets      = nullptr;
    PFN_vkCmdDispatch                 vkCmdDispatch                = nullptr;
    PFN_vkCmdPushConstants            vkCmdPushConstants           = nullptr;

    // GPU timestamp queries
    PFN_vkCreateQueryPool             vkCreateQueryPool            = nullptr;
    PFN_vkDestroyQueryPool            vkDestroyQueryPool           = nullptr;
    PFN_vkCmdWriteTimestamp2          vkCmdWriteTimestamp2         = nullptr;
    PFN_vkCmdResetQueryPool           vkCmdResetQueryPool          = nullptr;
    PFN_vkGetQueryPoolResults         vkGetQueryPoolResults        = nullptr;

    // Descriptor pool reset
    PFN_vkResetDescriptorPool         vkResetDescriptorPool        = nullptr;

    bool Load(VkDevice device, PFN_vkGetDeviceProcAddr get_proc);
};

// GPU buffer holding weights for a single layer (kernel + bias concatenated)
struct WeightBuffer {
    std::string name;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkDeviceSize size_bytes = 0;
};

// Flat storage buffer for intermediate feature maps at one resolution level.
// Layout: channel-major [C][H][W], FP16 storage buffers to halve bandwidth.
struct FeatureBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkDeviceSize size_bytes = 0;
    uint32_t channels = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

// Push constants shared by most ML compute shaders
struct MlPushConstants {
    uint32_t width;
    uint32_t height;
};

// Push constants for downsample (input dimensions)
struct DownsamplePushConstants {
    uint32_t in_width;
    uint32_t in_height;
};

// Push constants for upsample_concat (output dimensions)
struct UpsampleConcatPushConstants {
    uint32_t out_width;
    uint32_t out_height;
};

// Dispatch step in the U-Net inference sequence
struct DispatchStep {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    uint32_t group_count_x = 0;
    uint32_t group_count_y = 0;
    const void* push_constants = nullptr;
    uint32_t push_constants_size = 0;
};

// Manages GPU buffers, pipelines, and dispatch for ML U-Net inference.
class MlInference {
public:
    MlInference(VkDevice device, VmaAllocator allocator,
                PFN_vkGetDeviceProcAddr get_device_proc_addr,
                std::string_view shader_dir, VkPipelineCache pipeline_cache,
                uint32_t width, uint32_t height,
                float timestamp_period = 0.0f);
    ~MlInference();

    MlInference(const MlInference&) = delete;
    MlInference& operator=(const MlInference&) = delete;

    // Upload weight data to GPU storage buffers via staging buffer.
    bool LoadWeights(const WeightData& weights, VkCommandBuffer cmd);

    // (Re)allocate intermediate feature map buffers and descriptor sets.
    bool Resize(uint32_t width, uint32_t height);

    // Record ML inference dispatches into the command buffer.
    // Input images must be in GENERAL layout. Output image is written.
    void Infer(VkCommandBuffer cmd, const DenoiserInput& input, VkImageView output_view);

    // Free staging buffer after the transfer command buffer has completed.
    void FreeStagingBuffer();

    bool IsReady() const { return weights_loaded_ && pipelines_created_ && features_allocated_; }
    uint32_t Width() const { return width_; }
    uint32_t Height() const { return height_; }
    uint32_t WeightBufferCount() const { return static_cast<uint32_t>(weight_buffers_.size()); }

    // GPU timing (call after command buffer submission + fence wait)
    float GpuTimeMs() const { return gpu_time_ms_; }
    void ReadbackTimestamps();

    // U-Net architecture constants (inferred from loaded weights)
    uint32_t Level0Channels() const { return level0_channels_; }
    uint32_t Level1Channels() const { return level1_channels_; }
    uint32_t Level2Channels() const { return level2_channels_; }

    static constexpr uint32_t kInputChannels = 13;
    static constexpr uint32_t kOutputChannels = 3;
    static constexpr uint32_t kNumGroups = 8;
    static constexpr uint32_t kWorkgroupSize = 16;
    static constexpr uint32_t kReduceWorkgroupSize = 256;

private:
    // Weight management
    void DestroyWeightBuffers();
    bool InferArchitectureFromWeights(const WeightData& weights);
    VkBuffer FindWeightBuffer(std::string_view name) const;

    // Feature buffer management
    void DestroyFeatureBuffers();
    bool AllocateFeatureBuffer(FeatureBuffer& buf, uint32_t channels,
                               uint32_t width, uint32_t height);
    void DestroyFeatureBuffer(FeatureBuffer& buf);

    // Pipeline creation
    bool CreatePipelines();
    void DestroyPipelines();
    VkShaderModule LoadShaderModule(std::string_view filename);
    bool CreateConvPipeline(uint32_t in_ch, uint32_t out_ch, VkPipeline& pipeline,
                            VkPipelineLayout& layout, VkDescriptorSetLayout& ds_layout);
    bool CreateGroupNormReducePipeline(uint32_t channels,
                                       VkPipeline& pipeline, VkPipelineLayout& layout,
                                       VkDescriptorSetLayout& ds_layout);
    bool CreateGroupNormApplyPipeline(uint32_t channels, uint32_t activation,
                                      VkPipeline& pipeline, VkPipelineLayout& layout,
                                      VkDescriptorSetLayout& ds_layout);
    bool CreateEncoderInputConvPipeline();
    bool CreateDownsamplePipeline(uint32_t channels);
    bool CreateUpsampleConcatPipeline(uint32_t in_ch, uint32_t skip_ch);
    bool CreateOutputConvPipeline();

    // Descriptor management
    bool CreateDescriptorPool();
    void DestroyDescriptorPool();
    bool AllocateAndWriteDescriptors();

    // GroupNorm reduction buffer
    bool AllocateReductionBuffer(uint32_t max_spatial_elements);
    void DestroyReductionBuffer();

    // Dispatch helpers
    void DispatchConv(VkCommandBuffer cmd, VkBuffer input, VkBuffer output,
                      std::string_view weight_name, uint32_t in_ch, uint32_t out_ch,
                      uint32_t width, uint32_t height);
    void DispatchGroupNorm(VkCommandBuffer cmd, VkBuffer data, std::string_view norm_name,
                           uint32_t channels, uint32_t width, uint32_t height);
    void DispatchDownsample(VkCommandBuffer cmd, VkBuffer input, VkBuffer output,
                            uint32_t channels, uint32_t in_w, uint32_t in_h);
    void DispatchUpsampleConcat(VkCommandBuffer cmd, VkBuffer input, VkBuffer skip,
                                VkBuffer output, uint32_t in_ch, uint32_t skip_ch,
                                uint32_t out_w, uint32_t out_h);
    void InsertBufferBarrier(VkCommandBuffer cmd);

    // GPU timestamp queries
    bool CreateQueryPool();
    void DestroyQueryPool();

    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    MlDeviceDispatch dispatch_;
    std::string shader_dir_;
    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
    float timestamp_period_ = 0.0f;  // nanoseconds per tick

    // Weight storage
    std::vector<WeightBuffer> weight_buffers_;
    std::unordered_map<std::string, uint32_t> weight_index_;
    VkBuffer staging_buffer_ = VK_NULL_HANDLE;
    VmaAllocation staging_allocation_ = VK_NULL_HANDLE;

    // Architecture (inferred from weights)
    uint32_t level0_channels_ = 0;  // base_channels
    uint32_t level1_channels_ = 0;  // base_channels * 2
    uint32_t level2_channels_ = 0;  // base_channels * 4

    // Feature map buffers (flat [C][H][W] FP32 storage)
    // Level 0: full resolution
    FeatureBuffer buf0_a_, buf0_b_, skip0_;
    // Level 1: half resolution
    FeatureBuffer buf1_a_, buf1_b_, skip1_;
    // Level 2: quarter resolution
    FeatureBuffer buf2_a_, buf2_b_;
    // Upsample-concat scratch buffers
    FeatureBuffer concat1_;  // Level 1 resolution, in_ch + skip_ch channels
    FeatureBuffer concat0_;  // Level 0 resolution, in_ch + skip_ch channels

    // GroupNorm reduction buffer (shared across all dispatches)
    VkBuffer reduction_buffer_ = VK_NULL_HANDLE;
    VmaAllocation reduction_allocation_ = VK_NULL_HANDLE;
    VkDeviceSize reduction_buffer_size_ = 0;

    // Pipelines — keyed by (shader_type, in_ch, out_ch)
    // Encoder input conv (special: reads G-buffer images)
    VkPipeline encoder_input_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout encoder_input_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout encoder_input_ds_layout_ = VK_NULL_HANDLE;
    VkDescriptorSet encoder_input_ds_ = VK_NULL_HANDLE;

    // Output conv (special: writes to output image)
    VkPipeline output_conv_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout output_conv_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout output_conv_ds_layout_ = VK_NULL_HANDLE;
    VkDescriptorSet output_conv_ds_ = VK_NULL_HANDLE;

    // Generic conv pipelines: (in_ch, out_ch) -> pipeline
    struct ConvPipelineKey {
        uint32_t in_ch;
        uint32_t out_ch;
        bool operator==(const ConvPipelineKey&) const = default;
    };
    struct ConvPipelineKeyHash {
        size_t operator()(const ConvPipelineKey& k) const {
            return std::hash<uint64_t>{}((uint64_t(k.in_ch) << 32) | k.out_ch);
        }
    };
    struct PipelineSet {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout ds_layout = VK_NULL_HANDLE;
    };
    std::unordered_map<ConvPipelineKey, PipelineSet, ConvPipelineKeyHash> conv_pipelines_;

    // GroupNorm pipelines: channels -> pipeline (reduce + apply)
    struct GroupNormPipelineSet {
        VkPipeline reduce_pipeline = VK_NULL_HANDLE;
        VkPipelineLayout reduce_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout reduce_ds_layout = VK_NULL_HANDLE;
        VkPipeline apply_pipeline = VK_NULL_HANDLE;
        VkPipelineLayout apply_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout apply_ds_layout = VK_NULL_HANDLE;
    };
    std::unordered_map<uint32_t, GroupNormPipelineSet> norm_pipelines_;

    // Downsample pipeline: channels -> pipeline
    struct DownsamplePipelineSet {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout ds_layout = VK_NULL_HANDLE;
    };
    std::unordered_map<uint32_t, DownsamplePipelineSet> downsample_pipelines_;

    // Upsample-concat pipeline: (in_ch, skip_ch) -> pipeline
    struct UpsampleConcatPipelineSet {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout ds_layout = VK_NULL_HANDLE;
    };
    std::unordered_map<ConvPipelineKey, UpsampleConcatPipelineSet, ConvPipelineKeyHash>
        upsample_concat_pipelines_;

    // Descriptor pool for all ML dispatch descriptor sets
    VkDescriptorPool ml_descriptor_pool_ = VK_NULL_HANDLE;

    // Push constant data (updated per-dispatch, pointed to by DispatchStep)
    MlPushConstants pc_level0_{};
    MlPushConstants pc_level1_{};
    MlPushConstants pc_level2_{};
    DownsamplePushConstants pc_down0_{};
    DownsamplePushConstants pc_down1_{};
    UpsampleConcatPushConstants pc_up1_{};
    UpsampleConcatPushConstants pc_up0_{};

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    bool weights_loaded_ = false;
    bool pipelines_created_ = false;
    bool features_allocated_ = false;

    // GPU timestamp profiling
    static constexpr uint32_t kTimestampCount = 2;  // begin + end
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    float gpu_time_ms_ = 0.0f;
    bool timestamps_valid_ = false;  // True after first Infer() writes timestamps
};

}  // namespace deni::vulkan
