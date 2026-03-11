#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

#include <optional>
#include <span>
#include <vector>

namespace monti::app {

// Vulkan context supporting both windowed (monti_view) and headless (monti_datagen) modes.
//
// Two-step initialization:
//   1. CreateInstance(extra_instance_extensions) — creates instance, loads volk, sets up debug messenger
//   2. CreateDevice(optional_surface) — selects physical device, creates logical device + VMA allocator
//
// When no surface is provided (headless), swapchain extension is omitted and queue family selection
// requires only graphics capability.
class VulkanContext {
public:
    VulkanContext() = default;
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    VulkanContext(VulkanContext&&) = delete;
    VulkanContext& operator=(VulkanContext&&) = delete;

    // Step 1: Create Vulkan instance with caller-provided extra instance extensions
    // (e.g. SDL_Vulkan_GetInstanceExtensions() for monti_view).
    bool CreateInstance(std::span<const char* const> extra_instance_extensions = {});

    // Step 2: Create device. Pass a surface for windowed mode, or std::nullopt for headless.
    bool CreateDevice(std::optional<VkSurfaceKHR> surface);

    void WaitIdle() const;

    VkInstance Instance() const { return instance_; }
    VkPhysicalDevice PhysicalDevice() const { return physical_device_; }
    VkDevice Device() const { return device_; }
    VkQueue GraphicsQueue() const { return graphics_queue_; }
    uint32_t QueueFamilyIndex() const { return queue_family_index_; }
    VmaAllocator Allocator() const { return allocator_; }

    const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& RaytracePipelineProperties() const {
        return rt_pipeline_properties_;
    }
    const VkPhysicalDeviceAccelerationStructurePropertiesKHR& AccelStructProperties() const {
        return accel_struct_properties_;
    }

    // One-shot command buffer convenience.
    VkCommandBuffer BeginOneShot();
    void SubmitAndWait(VkCommandBuffer cmd);

private:
    bool SelectPhysicalDevice();
    bool CreateLogicalDevice();
    void QueryRtProperties();
    bool CreateAllocator();

    struct QueueFamilyResult {
        uint32_t graphics_family;
    };
    static std::optional<QueueFamilyResult> FindQueueFamilies(
        VkPhysicalDevice device, std::optional<VkSurfaceKHR> surface);
    static bool CheckDeviceExtensionSupport(
        VkPhysicalDevice device, std::span<const char* const> required_extensions);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    std::optional<VkSurfaceKHR> surface_;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;

    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_index_ = 0;

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_pipeline_properties_{};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accel_struct_properties_{};

    VmaAllocator allocator_ = VK_NULL_HANDLE;

    // One-shot command pool
    VkCommandPool oneshot_pool_ = VK_NULL_HANDLE;
};

}  // namespace monti::app
