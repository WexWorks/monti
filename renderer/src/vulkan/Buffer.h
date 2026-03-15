#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <functional>

namespace monti::vulkan {

struct DeviceDispatch;

// RAII wrapper for a VMA-allocated Vulkan buffer.
// Supports host-visible (mapped) and device-local modes.
class Buffer {
public:
    Buffer() = default;
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    bool Create(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage,
                VmaMemoryUsage memory_usage, VmaAllocationCreateFlags flags = 0);
    void Destroy();

    // Ensure the buffer has at least required_size bytes. Destroys and recreates
    // if needed. Calls pre_destroy() before Destroy() when recreation is required.
    // Returns true if the buffer is valid and large enough afterward.
    bool EnsureCapacity(VkDeviceSize required_size, VmaAllocator allocator,
                        VkBufferUsageFlags usage, VmaMemoryUsage memory_usage,
                        VmaAllocationCreateFlags flags = 0,
                        std::function<void()> pre_destroy = nullptr);

    VkBuffer Handle() const { return buffer_; }
    VkDeviceSize Size() const { return size_; }
    VkDeviceAddress DeviceAddress(VkDevice device, const DeviceDispatch& dispatch) const;

    void* Map();
    void Unmap();

private:
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
};

}  // namespace monti::vulkan
