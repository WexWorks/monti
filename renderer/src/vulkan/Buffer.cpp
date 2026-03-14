#include "Buffer.h"
#include "DeviceDispatch.h"

#include <cstdio>

namespace monti::vulkan {

Buffer::~Buffer() {
    Destroy();
}

Buffer::Buffer(Buffer&& other) noexcept
    : allocator_(other.allocator_),
      buffer_(other.buffer_),
      allocation_(other.allocation_),
      size_(other.size_) {
    other.allocator_ = VK_NULL_HANDLE;
    other.buffer_ = VK_NULL_HANDLE;
    other.allocation_ = VK_NULL_HANDLE;
    other.size_ = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
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

bool Buffer::Create(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage,
                    VmaMemoryUsage memory_usage, VmaAllocationCreateFlags flags) {
    Destroy();

    allocator_ = allocator;
    size_ = size;

    VkBufferCreateInfo buffer_ci{};
    buffer_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_ci.size = size;
    buffer_ci.usage = usage;
    buffer_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = memory_usage;
    alloc_ci.flags = flags;

    VkResult result = vmaCreateBuffer(allocator_, &buffer_ci, &alloc_ci, &buffer_, &allocation_, nullptr);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "Buffer::Create failed (VkResult: %d)\n", result);
        buffer_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
        return false;
    }

    return true;
}

void Buffer::Destroy() {
    if (buffer_ != VK_NULL_HANDLE && allocator_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
        buffer_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
    }
}

VkDeviceAddress Buffer::DeviceAddress(VkDevice device, const DeviceDispatch& dispatch) const {
    VkBufferDeviceAddressInfo addr_info{};
    addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addr_info.buffer = buffer_;
    return dispatch.vkGetBufferDeviceAddress(device, &addr_info);
}

void* Buffer::Map() {
    void* data = nullptr;
    VkResult result = vmaMapMemory(allocator_, allocation_, &data);
    if (result != VK_SUCCESS) return nullptr;
    return data;
}

void Buffer::Unmap() {
    vmaUnmapMemory(allocator_, allocation_);
}

}  // namespace monti::vulkan
