#pragma once

#include "Buffer.h"
#include "Image.h"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstdint>

namespace monti::vulkan::upload {

// Upload raw bytes into a device-local buffer via staging.
// Records vkCmdCopyBuffer into cmd. Staging buffer is returned
// and must be kept alive until cmd completes.
Buffer ToBuffer(VmaAllocator allocator, VkCommandBuffer cmd,
                const Buffer& dst, const void* data, VkDeviceSize size);

// Upload pixel data into a device-local image via staging.
// Transitions image from UNDEFINED → TRANSFER_DST → copy → SHADER_READ_ONLY.
// Generates mip chain via vkCmdBlitImage when mip_levels > 1.
// Returns the staging buffer that must be kept alive until cmd completes.
Buffer ToImage(VmaAllocator allocator, VkCommandBuffer cmd,
               const Image& dst, const void* data, VkDeviceSize size);

}  // namespace monti::vulkan::upload
