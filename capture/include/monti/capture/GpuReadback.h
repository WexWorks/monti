#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/gtc/packing.hpp>

namespace monti::capture {

// ═══════════════════════════════════════════════════════════════════════════
// Half-float conversion (delegates to GLM)
// ═══════════════════════════════════════════════════════════════════════════

inline float HalfToFloat(uint16_t h) { return glm::unpackHalf1x16(h); }
inline uint16_t FloatToHalf(float f) { return glm::packHalf1x16(f); }

// ═══════════════════════════════════════════════════════════════════════════
// Format conversion utilities
// ═══════════════════════════════════════════════════════════════════════════

// Unpack a VK_FORMAT_B10G11R11_UFLOAT_PACK32 value into 3 floats (R, G, B).
// Layout: bits [0:10] = B (5e5), [11:21] = G (5e6), [22:31] = R (5e6).
void UnpackB10G11R11(uint32_t packed, float& r, float& g, float& b);

// Unpack a full image of B10G11R11 packed values into a planar float buffer (3 floats/pixel).
void UnpackB10G11R11Image(const uint32_t* packed, float* out_rgb,
                          uint32_t pixel_count);

// Extract the R channel (view-space linear depth) from an RG16F image buffer.
// Input is raw uint16_t pairs (R, G per pixel); output is 1 float/pixel.
void ExtractDepthFromRG16F(const uint16_t* rg16f_raw, float* out_depth,
                           uint32_t pixel_count);

// ═══════════════════════════════════════════════════════════════════════════
// RAII staging buffer for GPU → CPU readback
// ═══════════════════════════════════════════════════════════════════════════

class StagingBuffer {
public:
    StagingBuffer() = default;
    ~StagingBuffer();

    StagingBuffer(const StagingBuffer&) = delete;
    StagingBuffer& operator=(const StagingBuffer&) = delete;
    StagingBuffer(StagingBuffer&& other) noexcept;
    StagingBuffer& operator=(StagingBuffer&& other) noexcept;

    bool Create(VmaAllocator allocator, VkDeviceSize size);
    void Destroy();

    VkBuffer Handle() const { return buffer_; }
    VkDeviceSize Size() const { return size_; }

    void* Map();
    void Unmap();

private:
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
};

// ═══════════════════════════════════════════════════════════════════════════
// GPU image readback
// ═══════════════════════════════════════════════════════════════════════════

// Readback context: provides the Vulkan resources needed for GPU→CPU copies.
struct ReadbackContext {
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family_index;
    VmaAllocator allocator;
    VkCommandPool command_pool;

    // Vulkan function pointers (required for VK_NO_PROTOTYPES builds)
    PFN_vkAllocateCommandBuffers  pfn_vkAllocateCommandBuffers;
    PFN_vkBeginCommandBuffer      pfn_vkBeginCommandBuffer;
    PFN_vkEndCommandBuffer        pfn_vkEndCommandBuffer;
    PFN_vkCmdPipelineBarrier2     pfn_vkCmdPipelineBarrier2;
    PFN_vkCmdCopyImageToBuffer    pfn_vkCmdCopyImageToBuffer;
    PFN_vkQueueSubmit             pfn_vkQueueSubmit;
    PFN_vkCreateFence             pfn_vkCreateFence;
    PFN_vkWaitForFences           pfn_vkWaitForFences;
    PFN_vkDestroyFence            pfn_vkDestroyFence;
    PFN_vkFreeCommandBuffers      pfn_vkFreeCommandBuffers;
};

// Read back a GPU image to a CPU staging buffer.
// The image must be in VK_IMAGE_LAYOUT_GENERAL. It is transitioned to
// TRANSFER_SRC_OPTIMAL, copied, then transitioned back to GENERAL.
// `pixel_size` is the byte width per pixel (e.g., 8 for RGBA16F, 4 for RG16F or B10G11R11).
// `src_stage`/`dst_stage` specify which pipeline stages produce/consume the image.
constexpr VkDeviceSize kRGBA16FPixelSize = 8;
StagingBuffer ReadbackImage(const ReadbackContext& ctx,
                            VkImage image,
                            uint32_t width, uint32_t height,
                            VkDeviceSize pixel_size = kRGBA16FPixelSize,
                            VkPipelineStageFlags2 src_stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                            VkPipelineStageFlags2 dst_stage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);

// ═══════════════════════════════════════════════════════════════════════════
// Multi-frame accumulation
// ═══════════════════════════════════════════════════════════════════════════

// Result of multi-frame rendering: averaged FP16 diffuse + specular.
struct MultiFrameResult {
    std::vector<float> diffuse_f32;   // RGBA, 4 floats per pixel (accumulated average)
    std::vector<float> specular_f32;  // RGBA, 4 floats per pixel (accumulated average)
};

// Callback type for rendering a single frame. Called once per accumulation frame.
// The implementation should record RenderFrame into cmd and submit.
// `frame_index` is unique per accumulation frame for jitter/blue-noise decorrelation.
using RenderFrameFn = void(*)(VkCommandBuffer cmd, uint32_t frame_index, void* user_data);

// Accumulate multiple frames of noisy_diffuse + noisy_specular into FP32 averages.
// Typical usage: render N frames at K SPP each → total = N*K effective SPP.
// `render_fn` is called N times; after each call the noisy_diffuse and noisy_specular
// images are read back and accumulated. The initial frame_index_offset allows
// continuation from a prior render pass.
MultiFrameResult AccumulateFrames(
    const ReadbackContext& ctx,
    VkImage noisy_diffuse_image,
    VkImage noisy_specular_image,
    uint32_t width, uint32_t height,
    uint32_t num_frames,
    uint32_t frame_index_offset,
    RenderFrameFn render_fn,
    void* user_data);

}  // namespace monti::capture
