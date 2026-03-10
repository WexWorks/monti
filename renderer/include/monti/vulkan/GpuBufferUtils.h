#pragma once
#include <monti/scene/Material.h>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <span>
#include <utility>
#include <vector>

namespace monti::vulkan {

struct GpuBuffer {
    VkBuffer        buffer         = VK_NULL_HANDLE;
    VmaAllocation   allocation     = VK_NULL_HANDLE;
    VkDeviceAddress device_address = 0;
    VkDeviceSize    size           = 0;
};

std::pair<GpuBuffer, GpuBuffer> UploadMeshToGpu(
    VmaAllocator allocator, VkDevice device, VkCommandBuffer cmd,
    const MeshData& mesh_data);

GpuBuffer CreateVertexBuffer(
    VmaAllocator allocator, VkDevice device, VkCommandBuffer cmd,
    std::span<const monti::Vertex> vertices);

GpuBuffer CreateIndexBuffer(
    VmaAllocator allocator, VkDevice device, VkCommandBuffer cmd,
    std::span<const uint32_t> indices);

void DestroyGpuBuffer(VmaAllocator allocator, GpuBuffer& buffer);

class Renderer;

/// Convenience: upload all meshes from a loader result, register their
/// buffer bindings with the Renderer, and return the GpuBuffers the host
/// must keep alive for the renderer's lifetime. Equivalent to calling
/// UploadMeshToGpu + Renderer::RegisterMeshBuffers per mesh, but less
/// boilerplate. Records vkCmdCopyBuffer commands into cmd.
std::vector<GpuBuffer> UploadAndRegisterMeshes(
    Renderer& renderer, VmaAllocator allocator, VkDevice device,
    VkCommandBuffer cmd, std::span<const MeshData> mesh_data);

} // namespace monti::vulkan
