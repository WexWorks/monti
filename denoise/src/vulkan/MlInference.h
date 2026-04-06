#pragma once

#include "WeightLoader.h"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace deni::vulkan {

struct DenoiserInput;

// Model version — auto-detected from weight layer names
enum class ModelVersion {
    kV1_SingleFrame,  // 3-level U-Net, standard convolutions, 19ch input
    kV3_Temporal,     // 2-level U-Net, depthwise separable, 26ch temporal input
};

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

    // Image copy / clear (for temporal history)
    PFN_vkCmdCopyImage                vkCmdCopyImage               = nullptr;
    PFN_vkCmdClearColorImage          vkCmdClearColorImage         = nullptr;

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

// 2D VkImage for frame history storage (temporal reprojection).
// Unlike FeatureBuffer (flat [C][H][W] storage buffer), this is a proper
// VkImage suitable for vkCmdCopyImage and imageLoad/imageStore in shaders.
struct HistoryImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0;
};

struct FrameHistory {
    HistoryImage denoised_diffuse;     // Previous frame's denoised output (RGBA16F, 2D)
    HistoryImage denoised_specular;    // Reserved for future separate-lobe output (RGBA16F, 2D)
    HistoryImage reprojected_diffuse;  // Warped previous diffuse (RGBA16F, 2D)
    HistoryImage reprojected_specular; // Warped previous specular (RGBA16F, 2D)
    HistoryImage disocclusion_mask;    // Binary mask (R16F, 2D): 1.0 = valid, 0.0 = disoccluded
    HistoryImage prev_depth;           // Previous frame's linear depth (RG16F, 2D)
    bool valid = false;                // False on first frame or after reset
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
    // output_image is the VkImage for the output (needed for temporal history copy).
    void Infer(VkCommandBuffer cmd, const DenoiserInput& input,
               VkImageView output_view, VkImage output_image);

    // Free staging buffer after the transfer command buffer has completed.
    void FreeStagingBuffer();

    bool IsReady() const { return weights_loaded_ && pipelines_created_ && features_allocated_; }
    uint32_t Width() const { return width_; }
    uint32_t Height() const { return height_; }
    uint32_t WeightBufferCount() const { return static_cast<uint32_t>(weight_buffers_.size()); }

    void SetDebugOutput(uint32_t mode) { debug_output_ = mode; }
    uint32_t DebugOutput() const { return debug_output_; }

    // GPU timing (call after command buffer submission + fence wait)
    float GpuTimeMs() const { return gpu_time_ms_; }
    void ReadbackTimestamps();

    // U-Net architecture constants (inferred from loaded weights)
    uint32_t Level0Channels() const { return level0_channels_; }
    uint32_t Level1Channels() const { return level1_channels_; }
    uint32_t Level2Channels() const { return level2_channels_; }

    static constexpr uint32_t kV1InputChannels = 19;
    static constexpr uint32_t kV1OutputChannels = 6;
    static constexpr uint32_t kV3InputChannels = 26;
    static constexpr uint32_t kV3OutputChannels = 7;  // 3ch delta_d + 3ch delta_s + 1ch blend weight

    ModelVersion GetModelVersion() const { return model_version_; }

    // Validate that a weight file's channel counts match what the shaders expect.
    // Can be called before LoadWeights to reject incompatible models early.
    static bool ValidateWeights(const WeightData& weights);

    // Largest group count ≤ kMaxGroups that evenly divides channels.
    // Must match the Python _num_groups() in training/deni_train/models/blocks.py.
    static constexpr uint32_t kMaxGroups = 8;
    static constexpr uint32_t NumGroups(uint32_t channels) {
        for (uint32_t g = std::min(kMaxGroups, channels); g > 0; --g)
            if (channels % g == 0) return g;
        return 1;
    }
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
    bool AllocateReductionBuffer(VkDeviceSize size);
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
    void DispatchReproject(VkCommandBuffer cmd, const DenoiserInput& input);
    void InsertBufferBarrier(VkCommandBuffer cmd);
    void InsertImageBarrier(VkCommandBuffer cmd, const HistoryImage& image);
    void CopyImageToHistory(VkCommandBuffer cmd, VkImage src, const HistoryImage& dst,
                            uint32_t width, uint32_t height);

    // V3 temporal dispatch helpers
    void DispatchDepthwiseConv(VkCommandBuffer cmd, VkBuffer input, VkBuffer output,
                               std::string_view weight_name, uint32_t channels,
                               uint32_t width, uint32_t height);
    void DispatchPointwiseConv(VkCommandBuffer cmd, VkBuffer input, VkBuffer output,
                               std::string_view weight_name, uint32_t in_ch, uint32_t out_ch,
                               uint32_t width, uint32_t height);
    void DispatchTemporalInputGather(VkCommandBuffer cmd, const DenoiserInput& input);
    void DispatchTemporalOutputConv(VkCommandBuffer cmd, VkBuffer feature_buf,
                                     VkImageView output_view,
                                     const DenoiserInput& input);

    // V3 pipeline creation
    bool CreateDepthwiseConvPipeline(uint32_t channels);
    bool CreatePointwiseConvPipeline(uint32_t in_ch, uint32_t out_ch);
    bool CreateTemporalInputGatherPipeline();
    bool CreateTemporalOutputConvPipeline();

    // V1 / V3 dispatch paths
    void InferV1(VkCommandBuffer cmd, const DenoiserInput& input,
                 VkImageView output_view, VkImage output_image);
    void InferV3Temporal(VkCommandBuffer cmd, const DenoiserInput& input,
                         VkImageView output_view, VkImage output_image);

    // History image management
    bool AllocateHistoryImage(HistoryImage& img, VkFormat format,
                              uint32_t width, uint32_t height);
    void DestroyHistoryImage(HistoryImage& img);
    void DestroyHistoryImages();

    // Reproject pipeline
    bool CreateReprojectPipeline();

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
    ModelVersion model_version_ = ModelVersion::kV1_SingleFrame;
    uint32_t level0_channels_ = 0;  // base_channels
    uint32_t level1_channels_ = 0;  // base_channels * 2
    uint32_t level2_channels_ = 0;  // base_channels * 4 (v1 only, 0 for v3)

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

    // V3 temporal: 26-channel input gather buffer (level 0 resolution)
    FeatureBuffer buf_input_;

    // GroupNorm reduction buffer (shared across all dispatches)
    VkBuffer reduction_buffer_ = VK_NULL_HANDLE;
    VmaAllocation reduction_allocation_ = VK_NULL_HANDLE;
    VkDeviceSize reduction_buffer_size_ = 0;

    // Temporal reprojection frame history
    FrameHistory frame_history_;

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

    // Reprojection pipeline (temporal)
    VkPipeline reproject_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout reproject_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout reproject_ds_layout_ = VK_NULL_HANDLE;

    // Depthwise conv pipelines: channels -> pipeline (v3 temporal)
    struct DepthwisePipelineSet {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout ds_layout = VK_NULL_HANDLE;
    };
    std::unordered_map<uint32_t, DepthwisePipelineSet> depthwise_pipelines_;

    // Pointwise conv pipelines: (in_ch, out_ch) -> pipeline (v3 temporal)
    std::unordered_map<ConvPipelineKey, PipelineSet, ConvPipelineKeyHash> pointwise_pipelines_;

    // Temporal input gather pipeline (v3 temporal)
    VkPipeline temporal_input_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout temporal_input_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout temporal_input_ds_layout_ = VK_NULL_HANDLE;

    // Temporal output conv pipeline (v3 temporal)
    VkPipeline temporal_output_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout temporal_output_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout temporal_output_ds_layout_ = VK_NULL_HANDLE;

    // Descriptor pools — one per frame-in-flight to avoid resetting a pool
    // while a previous frame's command buffer is still executing on the GPU.
    static constexpr uint32_t kPoolCount = 3;
    std::array<VkDescriptorPool, kPoolCount> ml_descriptor_pools_{};
    VkDescriptorPool active_pool_ = VK_NULL_HANDLE;  // Current frame's pool (set in Infer)
    uint32_t pool_index_ = 0;

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
    uint32_t debug_output_ = 0;

    // Velocity prior: motion magnitude (screen-space [0,1]) at which blend weight
    // saturates to 1.0, matching Python DeniTemporalResidualNet.max_mv_for_weight_.
    static constexpr float max_mv_for_weight_ = 0.05f;

    // GPU timestamp profiling
    static constexpr uint32_t kTimestampCount = 2;  // begin + end
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    float gpu_time_ms_ = 0.0f;
    bool timestamps_valid_ = false;  // True after first Infer() writes timestamps
};

}  // namespace deni::vulkan
