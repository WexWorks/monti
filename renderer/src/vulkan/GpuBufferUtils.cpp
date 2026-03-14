#include <monti/vulkan/GpuBufferUtils.h>
#include <monti/vulkan/Renderer.h>

#include <cstdio>
#include <cstring>

namespace monti::vulkan {

namespace {

GpuBuffer CreateMappableBuffer(VmaAllocator allocator, VkDevice device,
                               VkCommandBuffer cmd, const void* data,
                               VkDeviceSize size, VkBufferUsageFlags extra_usage,
                               const GpuBufferProcs& procs) {
    // Create host-visible buffer (VMA selects ReBAR device-local on modern GPUs)
    VkBufferCreateInfo dst_ci{};
    dst_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    dst_ci.size = size;
    dst_ci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                   VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   extra_usage;
    dst_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo dst_alloc{};
    dst_alloc.usage = VMA_MEMORY_USAGE_AUTO;
    dst_alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    GpuBuffer result{};
    VkResult vr = vmaCreateBuffer(allocator, &dst_ci, &dst_alloc,
                                  &result.buffer, &result.allocation, nullptr);
    if (vr != VK_SUCCESS) {
        std::fprintf(stderr, "CreateMappableBuffer: buffer failed (VkResult: %d)\n", vr);
        return {};
    }
    result.size = size;

    // Get device address
    VkBufferDeviceAddressInfo addr_info{};
    addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addr_info.buffer = result.buffer;
    result.device_address = procs.vkGetBufferDeviceAddress(device, &addr_info);

    // Write data directly (host-visible buffer — no staging needed)
    void* mapped = nullptr;
    vr = vmaMapMemory(allocator, result.allocation, &mapped);
    if (vr != VK_SUCCESS) {
        std::fprintf(stderr, "CreateMappableBuffer: map failed (VkResult: %d)\n", vr);
        vmaDestroyBuffer(allocator, result.buffer, result.allocation);
        result = {};
        return {};
    }
    std::memcpy(mapped, data, size);
    vmaUnmapMemory(allocator, result.allocation);
    vmaFlushAllocation(allocator, result.allocation, 0, VK_WHOLE_SIZE);

    // Memory barrier to make the host write visible to GPU commands
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    procs.vkCmdPipelineBarrier2(cmd, &dep);

    return result;
}

}  // anonymous namespace

std::pair<GpuBuffer, GpuBuffer> UploadMeshToGpu(
    VmaAllocator allocator, VkDevice device, VkCommandBuffer cmd,
    const MeshData& mesh_data, const GpuBufferProcs& procs) {
    if (mesh_data.vertices.empty() || mesh_data.indices.empty())
        return {};

    VkDeviceSize vb_size = mesh_data.vertices.size() * sizeof(Vertex);
    GpuBuffer vertex_buf = CreateMappableBuffer(
        allocator, device, cmd, mesh_data.vertices.data(), vb_size,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, procs);
    if (vertex_buf.buffer == VK_NULL_HANDLE) return {};

    VkDeviceSize ib_size = mesh_data.indices.size() * sizeof(uint32_t);
    GpuBuffer index_buf = CreateMappableBuffer(
        allocator, device, cmd, mesh_data.indices.data(), ib_size,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT, procs);
    if (index_buf.buffer == VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, vertex_buf.buffer, vertex_buf.allocation);
        return {};
    }

    return {vertex_buf, index_buf};
}

GpuBuffer CreateVertexBuffer(
    VmaAllocator allocator, VkDevice device, VkCommandBuffer cmd,
    std::span<const monti::Vertex> vertices, const GpuBufferProcs& procs) {
    if (vertices.empty()) return {};
    return CreateMappableBuffer(
        allocator, device, cmd, vertices.data(),
        vertices.size_bytes(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, procs);
}

GpuBuffer CreateIndexBuffer(
    VmaAllocator allocator, VkDevice device, VkCommandBuffer cmd,
    std::span<const uint32_t> indices, const GpuBufferProcs& procs) {
    if (indices.empty()) return {};
    return CreateMappableBuffer(
        allocator, device, cmd, indices.data(),
        indices.size_bytes(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, procs);
}

void DestroyGpuBuffer(VmaAllocator allocator, GpuBuffer& buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
        buffer = {};
    }
}

std::vector<GpuBuffer> UploadAndRegisterMeshes(
    Renderer& renderer, VmaAllocator allocator, VkDevice device,
    VkCommandBuffer cmd, std::span<const MeshData> mesh_data,
    const GpuBufferProcs& procs) {
    std::vector<GpuBuffer> all_buffers;
    all_buffers.reserve(mesh_data.size() * 2);

    for (const auto& md : mesh_data) {
        auto [vb, ib] = UploadMeshToGpu(allocator, device, cmd, md, procs);
        if (vb.buffer == VK_NULL_HANDLE) {
            for (auto& buf : all_buffers)
                DestroyGpuBuffer(allocator, buf);
            return {};
        }

        MeshBufferBinding binding{};
        binding.vertex_buffer = vb.buffer;
        binding.vertex_address = vb.device_address;
        binding.index_buffer = ib.buffer;
        binding.index_address = ib.device_address;
        binding.vertex_count = static_cast<uint32_t>(md.vertices.size());
        binding.index_count = static_cast<uint32_t>(md.indices.size());
        binding.vertex_stride = sizeof(Vertex);

        renderer.RegisterMeshBuffers(md.mesh_id, binding);

        all_buffers.push_back(vb);
        all_buffers.push_back(ib);
    }

    return all_buffers;
}

}  // namespace monti::vulkan
