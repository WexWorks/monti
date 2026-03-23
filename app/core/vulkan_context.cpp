#include "vulkan_context.h"

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <vk_mem_alloc.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

namespace monti::app {

namespace {

#if defined(_DEBUG) || defined(MONTI_ENABLE_VALIDATION)
VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    [[maybe_unused]] void* user_data) {
    const char* prefix = "VERBOSE";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        prefix = "ERROR";
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        prefix = "WARNING";
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
        prefix = "INFO";

    std::fprintf(stderr, "[Vulkan %s] %s\n", prefix, callback_data->pMessage);
    return VK_FALSE;
}
#endif  // _DEBUG || MONTI_ENABLE_VALIDATION

}  // namespace

VulkanContext::~VulkanContext() {
    if (oneshot_pool_ != VK_NULL_HANDLE)
        vkDestroyCommandPool(device_, oneshot_pool_, nullptr);

    if (allocator_ != VK_NULL_HANDLE)
        vmaDestroyAllocator(allocator_);

    if (device_ != VK_NULL_HANDLE)
        vkDestroyDevice(device_, nullptr);

#if defined(_DEBUG) || defined(MONTI_ENABLE_VALIDATION)
    if (debug_messenger_ != VK_NULL_HANDLE)
        vkDestroyDebugUtilsMessengerEXT(instance_, debug_messenger_, nullptr);
#endif

    // Note: surface is destroyed by the caller (SDL) since it was created externally
    if (instance_ != VK_NULL_HANDLE)
        vkDestroyInstance(instance_, nullptr);
}

bool VulkanContext::CreateInstance(std::span<const char* const> extra_instance_extensions) {
    if (volkInitialize() != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to initialize volk\n");
        return false;
    }

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Monti";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "Monti";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> extensions(extra_instance_extensions.begin(),
                                        extra_instance_extensions.end());

#if defined(_DEBUG) || defined(MONTI_ENABLE_VALIDATION)
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app_info;
    ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();

#if defined(_DEBUG) || defined(MONTI_ENABLE_VALIDATION)
    const char* validation_layer = "VK_LAYER_KHRONOS_validation";

    uint32_t layer_count = 0;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    bool validation_available = false;
    for (const auto& layer : available_layers) {
        if (std::strcmp(layer.layerName, validation_layer) == 0) {
            validation_available = true;
            break;
        }
    }

    if (validation_available) {
        ci.enabledLayerCount = 1;
        ci.ppEnabledLayerNames = &validation_layer;
        std::printf("Validation layers enabled\n");
    } else {
        std::fprintf(stderr, "Warning: validation layer not available\n");
    }
#endif  // _DEBUG || MONTI_ENABLE_VALIDATION

    VkResult result = vkCreateInstance(&ci, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to create Vulkan instance (VkResult: %d)\n", result);
        return false;
    }

    volkLoadInstance(instance_);

#if defined(_DEBUG) || defined(MONTI_ENABLE_VALIDATION)
    {
        VkDebugUtilsMessengerCreateInfoEXT dbg_ci{};
        dbg_ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        dbg_ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        dbg_ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dbg_ci.pfnUserCallback = DebugCallback;
        if (vkCreateDebugUtilsMessengerEXT(instance_, &dbg_ci, nullptr, &debug_messenger_) != VK_SUCCESS)
            std::fprintf(stderr, "Warning: failed to create debug messenger\n");
    }
#endif

    return true;
}

bool VulkanContext::CreateDevice(std::optional<VkSurfaceKHR> surface) {
    surface_ = surface;

    if (!SelectPhysicalDevice()) return false;
    if (!CreateLogicalDevice()) return false;

    volkLoadDevice(device_);
    QueryRtProperties();

    if (!CreateAllocator()) return false;

    vkGetDeviceQueue(device_, queue_family_index_, 0, &graphics_queue_);

    // Create one-shot command pool
    VkCommandPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_ci.queueFamilyIndex = queue_family_index_;
    if (vkCreateCommandPool(device_, &pool_ci, nullptr, &oneshot_pool_) != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to create one-shot command pool\n");
        return false;
    }

    return true;
}

void VulkanContext::WaitIdle() const {
    if (device_ != VK_NULL_HANDLE)
        vkDeviceWaitIdle(device_);
}

VkCommandBuffer VulkanContext::BeginOneShot() {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = oneshot_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device_, &alloc_info, &cmd) != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to allocate one-shot command buffer\n");
        return VK_NULL_HANDLE;
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to begin one-shot command buffer\n");
        vkFreeCommandBuffers(device_, oneshot_pool_, 1, &cmd);
        return VK_NULL_HANDLE;
    }

    return cmd;
}

void VulkanContext::SubmitAndWait(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    vkQueueSubmit(graphics_queue_, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphics_queue_);

    vkFreeCommandBuffers(device_, oneshot_pool_, 1, &cmd);
}

bool VulkanContext::SelectPhysicalDevice() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (device_count == 0) {
        std::fprintf(stderr, "No Vulkan-capable GPUs found\n");
        return false;
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

    // Build required extensions list (depends on windowed vs headless)
    std::vector<const char*> required_extensions = {
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    };
    if (surface_.has_value())
        required_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    for (auto device : devices) {
        auto families = FindQueueFamilies(device, surface_);
        if (!families.has_value()) continue;
        if (!CheckDeviceExtensionSupport(device, required_extensions)) continue;

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);

        physical_device_ = device;
        queue_family_index_ = families->graphics_family;

        std::printf("Selected GPU: %s\n", props.deviceName);
        std::printf("  API version: %u.%u.%u\n",
                    VK_API_VERSION_MAJOR(props.apiVersion),
                    VK_API_VERSION_MINOR(props.apiVersion),
                    VK_API_VERSION_PATCH(props.apiVersion));

        // Prefer discrete GPU
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            break;
    }

    if (physical_device_ == VK_NULL_HANDLE) {
        std::fprintf(stderr, "No suitable GPU found with required RT extensions\n");
        return false;
    }

    return true;
}

bool VulkanContext::CreateLogicalDevice() {
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_ci{};
    queue_ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_ci.queueFamilyIndex = queue_family_index_;
    queue_ci.queueCount = 1;
    queue_ci.pQueuePriorities = &priority;

    // Vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features vulkan12_features{};
    vulkan12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12_features.bufferDeviceAddress = VK_TRUE;
    vulkan12_features.descriptorIndexing = VK_TRUE;
    vulkan12_features.runtimeDescriptorArray = VK_TRUE;
    vulkan12_features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    vulkan12_features.scalarBlockLayout = VK_TRUE;
    vulkan12_features.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
    vulkan12_features.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    vulkan12_features.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    vulkan12_features.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
    vulkan12_features.descriptorBindingPartiallyBound = VK_TRUE;

    // Vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features vulkan13_features{};
    vulkan13_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13_features.pNext = &vulkan12_features;
    vulkan13_features.dynamicRendering = VK_TRUE;
    vulkan13_features.synchronization2 = VK_TRUE;
    vulkan13_features.maintenance4 = VK_TRUE;

    // Acceleration structure features
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_features{};
    accel_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accel_features.pNext = &vulkan13_features;
    accel_features.accelerationStructure = VK_TRUE;
    accel_features.descriptorBindingAccelerationStructureUpdateAfterBind = VK_TRUE;

    // Ray tracing pipeline features
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_features{};
    rt_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rt_features.pNext = &accel_features;
    rt_features.rayTracingPipeline = VK_TRUE;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &rt_features;
    features2.features.samplerAnisotropy = VK_TRUE;
    features2.features.shaderInt64 = VK_TRUE;
    features2.features.shaderStorageImageReadWithoutFormat = VK_TRUE;
    features2.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    features2.features.textureCompressionBC = VK_TRUE;

    // Build device extensions
    std::vector<const char*> extensions = {
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    };
    if (surface_.has_value())
        extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    VkDeviceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.pNext = &features2;
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &queue_ci;
    ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();

    VkResult result = vkCreateDevice(physical_device_, &ci, nullptr, &device_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to create logical device (VkResult: %d)\n", result);
        return false;
    }

    return true;
}

void VulkanContext::QueryRtProperties() {
    rt_pipeline_properties_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    accel_struct_properties_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    rt_pipeline_properties_.pNext = &accel_struct_properties_;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &rt_pipeline_properties_;
    vkGetPhysicalDeviceProperties2(physical_device_, &props2);

    std::printf("RT Pipeline Properties:\n");
    std::printf("  Max ray recursion depth: %u\n", rt_pipeline_properties_.maxRayRecursionDepth);
    std::printf("  Shader group handle size: %u\n", rt_pipeline_properties_.shaderGroupHandleSize);
    std::printf("  Shader group handle alignment: %u\n", rt_pipeline_properties_.shaderGroupHandleAlignment);
    std::printf("  Shader group base alignment: %u\n", rt_pipeline_properties_.shaderGroupBaseAlignment);
}

std::optional<VulkanContext::QueueFamilyResult> VulkanContext::FindQueueFamilies(
    VkPhysicalDevice device, std::optional<VkSurfaceKHR> surface) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    for (uint32_t i = 0; i < count; ++i) {
        if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;

        // If we have a surface, require present support on the same queue
        if (surface.has_value()) {
            VkBool32 present_support = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, *surface, &present_support);
            if (!present_support) continue;
        }

        return QueueFamilyResult{i};
    }

    return std::nullopt;
}

bool VulkanContext::CheckDeviceExtensionSupport(
    VkPhysicalDevice device, std::span<const char* const> required_extensions) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, available.data());

    for (auto required : required_extensions) {
        bool found = false;
        for (const auto& ext : available) {
            if (std::strcmp(ext.extensionName, required) == 0) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }

    return true;
}

bool VulkanContext::CreateAllocator() {
    VmaVulkanFunctions vma_funcs{};
    vma_funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vma_funcs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo alloc_ci{};
    alloc_ci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    alloc_ci.physicalDevice = physical_device_;
    alloc_ci.device = device_;
    alloc_ci.instance = instance_;
    alloc_ci.vulkanApiVersion = VK_API_VERSION_1_3;
    alloc_ci.pVulkanFunctions = &vma_funcs;

    VkResult result = vmaCreateAllocator(&alloc_ci, &allocator_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to create VMA allocator (VkResult: %d)\n", result);
        return false;
    }

    return true;
}

}  // namespace monti::app
