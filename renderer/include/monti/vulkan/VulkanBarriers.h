#pragma once

#include <vulkan/vulkan.h>

#include <span>

namespace monti::vulkan {

inline VkImageMemoryBarrier2 MakeImageBarrier(
    VkImage image,
    VkImageLayout old_layout,
    VkImageLayout new_layout,
    VkPipelineStageFlags2 src_stage,
    VkAccessFlags2 src_access,
    VkPipelineStageFlags2 dst_stage,
    VkAccessFlags2 dst_access,
    VkImageSubresourceRange subresource_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}) {
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = src_stage;
    barrier.srcAccessMask = src_access;
    barrier.dstStageMask = dst_stage;
    barrier.dstAccessMask = dst_access;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = subresource_range;
    return barrier;
}

// Dispatch-based CmdPipelineBarrier: works with any type that has a
// vkCmdPipelineBarrier2(VkCommandBuffer, const VkDependencyInfo*) member,
// including DeviceDispatch and volk's global function table.
template <typename Dispatch>
inline void CmdPipelineBarrier(VkCommandBuffer cmd,
                               std::span<const VkImageMemoryBarrier2> image_barriers,
                               const Dispatch& dispatch) {
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = static_cast<uint32_t>(image_barriers.size());
    dep.pImageMemoryBarriers = image_barriers.data();
    dispatch.vkCmdPipelineBarrier2(cmd, &dep);
}

// Overload accepting a raw function pointer (e.g. volk global PFN_vkCmdPipelineBarrier2).
inline void CmdPipelineBarrier(VkCommandBuffer cmd,
                               std::span<const VkImageMemoryBarrier2> image_barriers,
                               PFN_vkCmdPipelineBarrier2 pfn) {
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = static_cast<uint32_t>(image_barriers.size());
    dep.pImageMemoryBarriers = image_barriers.data();
    pfn(cmd, &dep);
}

}  // namespace monti::vulkan
