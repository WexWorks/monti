#pragma once

#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>
#include <deni/vulkan/Denoiser.h>

namespace monti::vulkan {

// Fill renderer proc addr fields from raw function pointers.
// These are loader-agnostic — the caller provides vkGetDeviceProcAddr
// and vkGetInstanceProcAddr from whatever Vulkan loader they use.
inline void FillRendererProcAddrs(RendererDesc& desc,
                                  VkInstance instance,
                                  PFN_vkGetDeviceProcAddr get_device_proc_addr,
                                  PFN_vkGetInstanceProcAddr get_instance_proc_addr) {
    desc.instance = instance;
    desc.get_device_proc_addr = get_device_proc_addr;
    desc.get_instance_proc_addr = get_instance_proc_addr;
}

// Create GpuBufferProcs from raw function pointers.
inline GpuBufferProcs MakeGpuBufferProcs(
    PFN_vkGetBufferDeviceAddress get_buffer_device_address,
    PFN_vkCmdPipelineBarrier2 cmd_pipeline_barrier2) {
    return {get_buffer_device_address, cmd_pipeline_barrier2};
}

// Fill denoiser proc addr fields from raw function pointers.
inline void FillDenoiserProcAddrs(deni::vulkan::DenoiserDesc& desc,
                                  PFN_vkGetDeviceProcAddr get_device_proc_addr) {
    desc.get_device_proc_addr = get_device_proc_addr;
}

}  // namespace monti::vulkan
