#include <monti/capture/GpuReadback.h>

#include <cmath>
#include <cstring>

namespace monti::capture {

// ═══════════════════════════════════════════════════════════════════════════
// Format conversion utilities
// ═══════════════════════════════════════════════════════════════════════════

namespace {

// Decode an unsigned float with `mantissa_bits` mantissa and `exp_bits` exponent.
float DecodeUFloat(uint32_t bits, int mantissa_bits, int exp_bits) {
    uint32_t mantissa_mask = (1u << mantissa_bits) - 1u;
    uint32_t exp_mask = (1u << exp_bits) - 1u;
    uint32_t mantissa = bits & mantissa_mask;
    uint32_t exponent = (bits >> mantissa_bits) & exp_mask;

    if (exponent == 0) {
        // Denormalized
        return std::ldexp(static_cast<float>(mantissa),
                          1 - (1 << (exp_bits - 1)) - mantissa_bits + 1);
    }
    if (exponent == exp_mask) {
        // Inf/NaN — return infinity for simplicity
        return (mantissa == 0) ? std::numeric_limits<float>::infinity()
                               : std::numeric_limits<float>::quiet_NaN();
    }
    // Normalized
    float m = 1.0f + static_cast<float>(mantissa) / static_cast<float>(1u << mantissa_bits);
    int e = static_cast<int>(exponent) - ((1 << (exp_bits - 1)) - 1);
    return std::ldexp(m, e);
}

}  // namespace

void UnpackB10G11R11(uint32_t packed, float& r, float& g, float& b) {
    // VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    // bits [0:10]  = R (6e5 — 6-bit mantissa, 5-bit exponent)
    // bits [11:21] = G (6e5)
    // bits [22:31] = B (5e5 — 5-bit mantissa, 5-bit exponent)
    uint32_t r_bits = packed & 0x7FFu;        // 11 bits
    uint32_t g_bits = (packed >> 11) & 0x7FFu; // 11 bits
    uint32_t b_bits = (packed >> 22) & 0x3FFu; // 10 bits

    r = DecodeUFloat(r_bits, 6, 5);
    g = DecodeUFloat(g_bits, 6, 5);
    b = DecodeUFloat(b_bits, 5, 5);
}

void UnpackB10G11R11Image(const uint32_t* packed, float* out_rgb,
                          uint32_t pixel_count) {
    for (uint32_t i = 0; i < pixel_count; ++i) {
        UnpackB10G11R11(packed[i],
                        out_rgb[i * 3 + 0],
                        out_rgb[i * 3 + 1],
                        out_rgb[i * 3 + 2]);
    }
}

void ExtractDepthFromRG16F(const uint16_t* rg16f_raw, float* out_depth,
                           uint32_t pixel_count) {
    for (uint32_t i = 0; i < pixel_count; ++i)
        out_depth[i] = HalfToFloat(rg16f_raw[i * 2]);  // R channel only
}

// ═══════════════════════════════════════════════════════════════════════════
// StagingBuffer
// ═══════════════════════════════════════════════════════════════════════════

StagingBuffer::~StagingBuffer() { Destroy(); }

StagingBuffer::StagingBuffer(StagingBuffer&& other) noexcept
    : allocator_(other.allocator_)
    , buffer_(other.buffer_)
    , allocation_(other.allocation_)
    , size_(other.size_) {
    other.allocator_ = VK_NULL_HANDLE;
    other.buffer_ = VK_NULL_HANDLE;
    other.allocation_ = VK_NULL_HANDLE;
    other.size_ = 0;
}

StagingBuffer& StagingBuffer::operator=(StagingBuffer&& other) noexcept {
    if (this != &other) {
        Destroy();
        allocator_ = other.allocator_;
        buffer_ = other.buffer_;
        allocation_ = other.allocation_;
        size_ = other.size_;
        other.allocator_ = VK_NULL_HANDLE;
        other.buffer_ = VK_NULL_HANDLE;
        other.allocation_ = VK_NULL_HANDLE;
        other.size_ = 0;
    }
    return *this;
}

bool StagingBuffer::Create(VmaAllocator allocator, VkDeviceSize size) {
    VkBufferCreateInfo buf_info{};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = size;
    buf_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    VkResult result = vmaCreateBuffer(allocator, &buf_info, &alloc_info,
                                       &buffer_, &allocation_, nullptr);
    if (result != VK_SUCCESS) return false;

    allocator_ = allocator;
    size_ = size;
    return true;
}

void StagingBuffer::Destroy() {
    if (buffer_ != VK_NULL_HANDLE && allocator_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
        buffer_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
        allocator_ = VK_NULL_HANDLE;
        size_ = 0;
    }
}

void* StagingBuffer::Map() {
    void* data = nullptr;
    vmaMapMemory(allocator_, allocation_, &data);
    return data;
}

void StagingBuffer::Unmap() {
    vmaUnmapMemory(allocator_, allocation_);
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU image readback
// ═══════════════════════════════════════════════════════════════════════════

namespace {

VkCommandBuffer BeginOneShot(const ReadbackContext& ctx) {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = ctx.command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkResult result = ctx.pfn_vkAllocateCommandBuffers(ctx.device, &alloc_info, &cmd);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuReadback: vkAllocateCommandBuffers failed (%d)\n", result);
        return VK_NULL_HANDLE;
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = ctx.pfn_vkBeginCommandBuffer(cmd, &begin_info);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuReadback: vkBeginCommandBuffer failed (%d)\n", result);
        ctx.pfn_vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd);
        return VK_NULL_HANDLE;
    }
    return cmd;
}

bool SubmitAndWait(const ReadbackContext& ctx, VkCommandBuffer cmd) {
    ctx.pfn_vkEndCommandBuffer(cmd);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    VkResult result = ctx.pfn_vkCreateFence(ctx.device, &fence_info, nullptr, &fence);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuReadback: vkCreateFence failed (%d)\n", result);
        ctx.pfn_vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd);
        return false;
    }

    result = ctx.pfn_vkQueueSubmit(ctx.queue, 1, &submit_info, fence);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuReadback: vkQueueSubmit failed (%d)\n", result);
        ctx.pfn_vkDestroyFence(ctx.device, fence, nullptr);
        ctx.pfn_vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd);
        return false;
    }

    result = ctx.pfn_vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS)
        std::fprintf(stderr, "GpuReadback: vkWaitForFences failed (%d)\n", result);

    ctx.pfn_vkDestroyFence(ctx.device, fence, nullptr);
    ctx.pfn_vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd);
    return result == VK_SUCCESS;
}

}  // namespace

StagingBuffer ReadbackImage(const ReadbackContext& ctx,
                            VkImage image,
                            uint32_t width, uint32_t height,
                            VkDeviceSize pixel_size,
                            VkPipelineStageFlags2 src_stage,
                            VkPipelineStageFlags2 dst_stage) {
    VkDeviceSize readback_size = static_cast<VkDeviceSize>(width) * height * pixel_size;

    StagingBuffer staging;
    if (!staging.Create(ctx.allocator, readback_size)) return {};

    VkCommandBuffer cmd = BeginOneShot(ctx);
    if (cmd == VK_NULL_HANDLE) return {};

    // Transition image: GENERAL → TRANSFER_SRC_OPTIMAL
    VkImageMemoryBarrier2 to_src{};
    to_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_src.srcStageMask = src_stage;
    to_src.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    to_src.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    to_src.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    to_src.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_src.image = image;
    to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &to_src;
    ctx.pfn_vkCmdPipelineBarrier2(cmd, &dep);

    // Copy image to staging buffer
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    ctx.pfn_vkCmdCopyImageToBuffer(cmd, image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           staging.Handle(), 1, &region);

    // Transition back: TRANSFER_SRC_OPTIMAL → GENERAL
    VkImageMemoryBarrier2 to_general{};
    to_general.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_general.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    to_general.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    to_general.dstStageMask = dst_stage;
    to_general.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    to_general.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    to_general.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_general.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_general.image = image;
    to_general.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep2{};
    dep2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep2.imageMemoryBarrierCount = 1;
    dep2.pImageMemoryBarriers = &to_general;
    ctx.pfn_vkCmdPipelineBarrier2(cmd, &dep2);

    SubmitAndWait(ctx, cmd);
    return staging;
}

std::vector<StagingBuffer> ReadbackMultipleImages(
    const ReadbackContext& ctx,
    std::span<const ReadbackRequest> requests) {

    if (requests.empty()) return {};

    // Allocate all staging buffers up front
    std::vector<StagingBuffer> staging_buffers;
    staging_buffers.reserve(requests.size());
    for (const auto& req : requests) {
        VkDeviceSize size = static_cast<VkDeviceSize>(req.width) * req.height * req.pixel_size;
        StagingBuffer buf;
        if (!buf.Create(ctx.allocator, size)) return {};
        staging_buffers.push_back(std::move(buf));
    }

    // Begin single command buffer
    VkCommandBuffer cmd = BeginOneShot(ctx);
    if (cmd == VK_NULL_HANDLE) return {};

    // Record barriers + copies for all images
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& req = requests[i];

        // Transition: GENERAL → TRANSFER_SRC_OPTIMAL
        VkImageMemoryBarrier2 to_src{};
        to_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        to_src.srcStageMask = req.src_stage;
        to_src.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        to_src.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        to_src.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        to_src.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_src.image = req.image;
        to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &to_src;
        ctx.pfn_vkCmdPipelineBarrier2(cmd, &dep);

        // Copy image to staging buffer
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {req.width, req.height, 1};
        ctx.pfn_vkCmdCopyImageToBuffer(cmd, req.image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               staging_buffers[i].Handle(), 1, &region);

        // Transition back: TRANSFER_SRC_OPTIMAL → GENERAL
        VkImageMemoryBarrier2 to_general{};
        to_general.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        to_general.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        to_general.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        to_general.dstStageMask = req.dst_stage;
        to_general.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        to_general.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_general.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_general.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_general.image = req.image;
        to_general.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep2{};
        dep2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep2.imageMemoryBarrierCount = 1;
        dep2.pImageMemoryBarriers = &to_general;
        ctx.pfn_vkCmdPipelineBarrier2(cmd, &dep2);
    }

    // Single submit + wait for all copies
    if (!SubmitAndWait(ctx, cmd)) return {};
    return staging_buffers;
}

// ═══════════════════════════════════════════════════════════════════════════
// Multi-frame accumulation
// ═══════════════════════════════════════════════════════════════════════════

MultiFrameResult AccumulateFrames(
    const ReadbackContext& ctx,
    VkImage noisy_diffuse_image,
    VkImage noisy_specular_image,
    uint32_t width, uint32_t height,
    uint32_t num_frames,
    uint32_t frame_index_offset,
    RenderFrameFn render_fn,
    void* user_data) {

    uint32_t pixel_count = width * height;
    constexpr uint32_t kChannels = 4;

    std::vector<float> accum_diffuse(static_cast<size_t>(pixel_count) * kChannels, 0.0f);
    std::vector<float> accum_specular(static_cast<size_t>(pixel_count) * kChannels, 0.0f);

    for (uint32_t frame = 0; frame < num_frames; ++frame) {
        uint32_t frame_index = frame_index_offset + frame;

        // Let the caller render a frame (records commands + submits)
        VkCommandBuffer cmd = BeginOneShot(ctx);
        if (cmd == VK_NULL_HANDLE) return {};
        render_fn(cmd, frame_index, user_data);
        if (!SubmitAndWait(ctx, cmd)) return {};

        // Read back diffuse and specular (RGBA16F = 8 bytes/pixel)
        auto diffuse_rb = ReadbackImage(ctx, noisy_diffuse_image, width, height, kRGBA16FPixelSize);
        auto specular_rb = ReadbackImage(ctx, noisy_specular_image, width, height, kRGBA16FPixelSize);

        auto* d_raw = static_cast<uint16_t*>(diffuse_rb.Map());
        auto* s_raw = static_cast<uint16_t*>(specular_rb.Map());

        for (size_t i = 0; i < static_cast<size_t>(pixel_count) * kChannels; ++i) {
            accum_diffuse[i] += HalfToFloat(d_raw[i]);
            accum_specular[i] += HalfToFloat(s_raw[i]);
        }

        diffuse_rb.Unmap();
        specular_rb.Unmap();
    }

    // Average
    float inv_frames = 1.0f / static_cast<float>(num_frames);
    for (size_t i = 0; i < static_cast<size_t>(pixel_count) * kChannels; ++i) {
        accum_diffuse[i] *= inv_frames;
        accum_specular[i] *= inv_frames;
    }

    return MultiFrameResult{
        std::move(accum_diffuse),
        std::move(accum_specular)};
}

}  // namespace monti::capture
