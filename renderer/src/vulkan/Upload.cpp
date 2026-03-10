#include <volk.h>

#include "Upload.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace monti::vulkan::upload {

namespace {

void GenerateMipmaps(VkCommandBuffer cmd, const Image& image) {
    int32_t mip_width = static_cast<int32_t>(image.Width());
    int32_t mip_height = static_cast<int32_t>(image.Height());

    for (uint32_t i = 1; i < image.MipLevels(); ++i) {
        // Transition previous level from TRANSFER_DST to TRANSFER_SRC
        VkImageMemoryBarrier2 to_src{};
        to_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        to_src.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        to_src.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        to_src.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        to_src.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        to_src.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_src.image = image.Handle();
        to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &to_src;
        vkCmdPipelineBarrier2(cmd, &dep);

        // Blit from level i-1 to level i
        int32_t next_width = std::max(mip_width / 2, 1);
        int32_t next_height = std::max(mip_height / 2, 1);

        VkImageBlit2 blit{};
        blit.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mip_width, mip_height, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {next_width, next_height, 1};

        VkBlitImageInfo2 blit_info{};
        blit_info.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
        blit_info.srcImage = image.Handle();
        blit_info.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        blit_info.dstImage = image.Handle();
        blit_info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        blit_info.regionCount = 1;
        blit_info.pRegions = &blit;
        blit_info.filter = VK_FILTER_LINEAR;
        vkCmdBlitImage2(cmd, &blit_info);

        // Transition source level to SHADER_READ_ONLY
        VkImageMemoryBarrier2 to_shader{};
        to_shader.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        to_shader.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        to_shader.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        to_shader.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        to_shader.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        to_shader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_shader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_shader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_shader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_shader.image = image.Handle();
        to_shader.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 1};

        dep.pImageMemoryBarriers = &to_shader;
        vkCmdPipelineBarrier2(cmd, &dep);

        mip_width = next_width;
        mip_height = next_height;
    }

    // Transition last mip level to SHADER_READ_ONLY
    VkImageMemoryBarrier2 last_to_shader{};
    last_to_shader.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    last_to_shader.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    last_to_shader.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    last_to_shader.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    last_to_shader.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    last_to_shader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    last_to_shader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    last_to_shader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    last_to_shader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    last_to_shader.image = image.Handle();
    last_to_shader.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                       image.MipLevels() - 1, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &last_to_shader;
    vkCmdPipelineBarrier2(cmd, &dep);
}

}  // anonymous namespace

Buffer ToBuffer(VmaAllocator allocator, VkCommandBuffer cmd,
                const Buffer& dst, const void* data, VkDeviceSize size) {
    Buffer staging;
    if (!staging.Create(allocator, size,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VMA_MEMORY_USAGE_CPU_ONLY))
        return {};

    void* mapped = staging.Map();
    std::memcpy(mapped, data, size);
    staging.Unmap();

    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(cmd, staging.Handle(), dst.Handle(), 1, &region);

    return staging;
}

Buffer ToImage(VmaAllocator allocator, VkCommandBuffer cmd,
               const Image& dst, const void* data, VkDeviceSize size) {
    Buffer staging;
    if (!staging.Create(allocator, size,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VMA_MEMORY_USAGE_CPU_ONLY))
        return {};

    void* mapped = staging.Map();
    std::memcpy(mapped, data, size);
    staging.Unmap();

    // Transition base level to TRANSFER_DST
    VkImageMemoryBarrier2 to_dst{};
    to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_dst.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    to_dst.srcAccessMask = 0;
    to_dst.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    to_dst.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_dst.image = dst.Handle();
    to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, dst.MipLevels(), 0, 1};

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &to_dst;
    vkCmdPipelineBarrier2(cmd, &dep);

    // Copy staging buffer to base mip level
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {dst.Width(), dst.Height(), 1};
    vkCmdCopyBufferToImage(cmd, staging.Handle(), dst.Handle(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    if (dst.MipLevels() > 1) {
        GenerateMipmaps(cmd, dst);
    } else {
        // Single mip level — transition directly to SHADER_READ_ONLY
        VkImageMemoryBarrier2 to_shader{};
        to_shader.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        to_shader.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        to_shader.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        to_shader.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        to_shader.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        to_shader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_shader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_shader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_shader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_shader.image = dst.Handle();
        to_shader.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        dep.pImageMemoryBarriers = &to_shader;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    return staging;
}

}  // namespace monti::vulkan::upload
