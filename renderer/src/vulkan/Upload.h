#pragma once

#include "Buffer.h"
#include "Image.h"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <span>

namespace monti::vulkan {

struct DeviceDispatch;

}

namespace monti::vulkan::upload {

// Upload raw bytes into a device-local buffer via staging.
// Records vkCmdCopyBuffer into cmd. Staging buffer is returned
// and must be kept alive until cmd completes.
Buffer ToBuffer(VmaAllocator allocator, VkCommandBuffer cmd,
                const Buffer& dst, const void* data, VkDeviceSize size,
                const DeviceDispatch& dispatch);

// Upload pixel data into a device-local image via staging.
// Transitions image from UNDEFINED → TRANSFER_DST → copy → SHADER_READ_ONLY.
// Generates mip chain via vkCmdBlitImage when mip_levels > 1.
// Returns the staging buffer that must be kept alive until cmd completes.
Buffer ToImage(VmaAllocator allocator, VkCommandBuffer cmd,
               const Image& dst, const void* data, VkDeviceSize size,
               const DeviceDispatch& dispatch);

// Per-mip level region for pre-generated mipmap upload.
struct MipRegion {
    uint32_t offset;  // byte offset in staging data
    uint32_t width;
    uint32_t height;
};

// Upload pre-generated mip chain data into a device-local image via staging.
// Transitions image UNDEFINED → TRANSFER_DST → copy all mips → SHADER_READ_ONLY.
// No vkCmdBlitImage step — mipmaps are stored in the source data.
// Returns the staging buffer that must be kept alive until cmd completes.
Buffer ToImageWithMips(VmaAllocator allocator, VkCommandBuffer cmd,
                       const Image& dst, const void* data, VkDeviceSize size,
                       std::span<const MipRegion> mips,
                       const DeviceDispatch& dispatch);

}  // namespace monti::vulkan::upload
