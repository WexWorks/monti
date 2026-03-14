#pragma once

#include <vulkan/vulkan.h>

#include <cstdio>
#include <type_traits>

namespace monti::vulkan {

// Private dispatch table for all Vulkan functions used by monti_vulkan.
// Resolved at Renderer::Create() time via host-provided proc addr callbacks.
// Not exposed in any public header.
struct DeviceDispatch {
    // ── Instance-level (resolved via get_instance_proc_addr) ─────
    PFN_vkGetPhysicalDeviceProperties  vkGetPhysicalDeviceProperties  = nullptr;
    PFN_vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties2 = nullptr;

    // ── Buffer / memory ──────────────────────────────────────────
    PFN_vkGetBufferDeviceAddress vkGetBufferDeviceAddress = nullptr;

    // ── Image / sampler ──────────────────────────────────────────
    PFN_vkCreateImageView  vkCreateImageView  = nullptr;
    PFN_vkDestroyImageView vkDestroyImageView = nullptr;
    PFN_vkCreateSampler    vkCreateSampler    = nullptr;
    PFN_vkDestroySampler   vkDestroySampler   = nullptr;

    // ── Descriptor management ────────────────────────────────────
    PFN_vkCreateDescriptorSetLayout  vkCreateDescriptorSetLayout  = nullptr;
    PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = nullptr;
    PFN_vkCreateDescriptorPool       vkCreateDescriptorPool       = nullptr;
    PFN_vkDestroyDescriptorPool      vkDestroyDescriptorPool      = nullptr;
    PFN_vkAllocateDescriptorSets     vkAllocateDescriptorSets     = nullptr;
    PFN_vkUpdateDescriptorSets       vkUpdateDescriptorSets       = nullptr;

    // ── Shader / pipeline ────────────────────────────────────────
    PFN_vkCreateShaderModule          vkCreateShaderModule          = nullptr;
    PFN_vkDestroyShaderModule         vkDestroyShaderModule         = nullptr;
    PFN_vkCreatePipelineLayout        vkCreatePipelineLayout        = nullptr;
    PFN_vkDestroyPipelineLayout       vkDestroyPipelineLayout       = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR = nullptr;
    PFN_vkDestroyPipeline             vkDestroyPipeline             = nullptr;

    // ── Ray tracing / acceleration structures ────────────────────
    PFN_vkGetAccelerationStructureBuildSizesKHR    vkGetAccelerationStructureBuildSizesKHR    = nullptr;
    PFN_vkCreateAccelerationStructureKHR           vkCreateAccelerationStructureKHR           = nullptr;
    PFN_vkDestroyAccelerationStructureKHR          vkDestroyAccelerationStructureKHR          = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR        vkCmdBuildAccelerationStructuresKHR        = nullptr;
    PFN_vkCmdCopyAccelerationStructureKHR          vkCmdCopyAccelerationStructureKHR          = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR       vkGetRayTracingShaderGroupHandlesKHR       = nullptr;

    // ── Query pool ───────────────────────────────────────────────
    PFN_vkCreateQueryPool                              vkCreateQueryPool                              = nullptr;
    PFN_vkDestroyQueryPool                             vkDestroyQueryPool                             = nullptr;
    PFN_vkCmdResetQueryPool                            vkCmdResetQueryPool                            = nullptr;
    PFN_vkCmdWriteAccelerationStructuresPropertiesKHR  vkCmdWriteAccelerationStructuresPropertiesKHR  = nullptr;
    PFN_vkGetQueryPoolResults                          vkGetQueryPoolResults                          = nullptr;

    // ── Command recording ────────────────────────────────────────
    PFN_vkCmdPipelineBarrier2    vkCmdPipelineBarrier2    = nullptr;
    PFN_vkCmdBindPipeline        vkCmdBindPipeline        = nullptr;
    PFN_vkCmdBindDescriptorSets  vkCmdBindDescriptorSets  = nullptr;
    PFN_vkCmdPushConstants       vkCmdPushConstants       = nullptr;
    PFN_vkCmdTraceRaysKHR        vkCmdTraceRaysKHR        = nullptr;
    PFN_vkCmdCopyBuffer          vkCmdCopyBuffer          = nullptr;
    PFN_vkCmdBlitImage2          vkCmdBlitImage2          = nullptr;
    PFN_vkCmdCopyBufferToImage   vkCmdCopyBufferToImage   = nullptr;

    bool Load(VkDevice device, VkInstance instance,
              PFN_vkGetDeviceProcAddr get_device_proc,
              PFN_vkGetInstanceProcAddr get_instance_proc) {
        bool ok = true;

        auto resolve_device = [&](auto& fn_ptr, const char* name) {
            fn_ptr = reinterpret_cast<std::remove_reference_t<decltype(fn_ptr)>>(
                get_device_proc(device, name));
            if (!fn_ptr) {
                std::fprintf(stderr, "monti::Renderer: failed to resolve %s\n", name);
                ok = false;
            }
        };

        auto resolve_instance = [&](auto& fn_ptr, const char* name) {
            fn_ptr = reinterpret_cast<std::remove_reference_t<decltype(fn_ptr)>>(
                get_instance_proc(instance, name));
            if (!fn_ptr) {
                std::fprintf(stderr, "monti::Renderer: failed to resolve %s\n", name);
                ok = false;
            }
        };

        // Instance-level
        resolve_instance(vkGetPhysicalDeviceProperties,  "vkGetPhysicalDeviceProperties");
        resolve_instance(vkGetPhysicalDeviceProperties2, "vkGetPhysicalDeviceProperties2");

        // Device-level — buffer / memory
        resolve_device(vkGetBufferDeviceAddress, "vkGetBufferDeviceAddress");

        // Device-level — image / sampler
        resolve_device(vkCreateImageView,  "vkCreateImageView");
        resolve_device(vkDestroyImageView, "vkDestroyImageView");
        resolve_device(vkCreateSampler,    "vkCreateSampler");
        resolve_device(vkDestroySampler,   "vkDestroySampler");

        // Device-level — descriptor management
        resolve_device(vkCreateDescriptorSetLayout,  "vkCreateDescriptorSetLayout");
        resolve_device(vkDestroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout");
        resolve_device(vkCreateDescriptorPool,       "vkCreateDescriptorPool");
        resolve_device(vkDestroyDescriptorPool,      "vkDestroyDescriptorPool");
        resolve_device(vkAllocateDescriptorSets,     "vkAllocateDescriptorSets");
        resolve_device(vkUpdateDescriptorSets,       "vkUpdateDescriptorSets");

        // Device-level — shader / pipeline
        resolve_device(vkCreateShaderModule,           "vkCreateShaderModule");
        resolve_device(vkDestroyShaderModule,          "vkDestroyShaderModule");
        resolve_device(vkCreatePipelineLayout,         "vkCreatePipelineLayout");
        resolve_device(vkDestroyPipelineLayout,        "vkDestroyPipelineLayout");
        resolve_device(vkCreateRayTracingPipelinesKHR, "vkCreateRayTracingPipelinesKHR");
        resolve_device(vkDestroyPipeline,              "vkDestroyPipeline");

        // Device-level — ray tracing / acceleration structures
        resolve_device(vkGetAccelerationStructureBuildSizesKHR,    "vkGetAccelerationStructureBuildSizesKHR");
        resolve_device(vkCreateAccelerationStructureKHR,           "vkCreateAccelerationStructureKHR");
        resolve_device(vkDestroyAccelerationStructureKHR,          "vkDestroyAccelerationStructureKHR");
        resolve_device(vkCmdBuildAccelerationStructuresKHR,        "vkCmdBuildAccelerationStructuresKHR");
        resolve_device(vkCmdCopyAccelerationStructureKHR,          "vkCmdCopyAccelerationStructureKHR");
        resolve_device(vkGetAccelerationStructureDeviceAddressKHR, "vkGetAccelerationStructureDeviceAddressKHR");
        resolve_device(vkGetRayTracingShaderGroupHandlesKHR,       "vkGetRayTracingShaderGroupHandlesKHR");

        // Device-level — query pool
        resolve_device(vkCreateQueryPool,                             "vkCreateQueryPool");
        resolve_device(vkDestroyQueryPool,                            "vkDestroyQueryPool");
        resolve_device(vkCmdResetQueryPool,                           "vkCmdResetQueryPool");
        resolve_device(vkCmdWriteAccelerationStructuresPropertiesKHR, "vkCmdWriteAccelerationStructuresPropertiesKHR");
        resolve_device(vkGetQueryPoolResults,                         "vkGetQueryPoolResults");

        // Device-level — command recording
        resolve_device(vkCmdPipelineBarrier2,   "vkCmdPipelineBarrier2");
        resolve_device(vkCmdBindPipeline,       "vkCmdBindPipeline");
        resolve_device(vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets");
        resolve_device(vkCmdPushConstants,      "vkCmdPushConstants");
        resolve_device(vkCmdTraceRaysKHR,       "vkCmdTraceRaysKHR");
        resolve_device(vkCmdCopyBuffer,         "vkCmdCopyBuffer");
        resolve_device(vkCmdBlitImage2,         "vkCmdBlitImage2");
        resolve_device(vkCmdCopyBufferToImage,  "vkCmdCopyBufferToImage");

        return ok;
    }
};

}  // namespace monti::vulkan
