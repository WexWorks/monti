#pragma once

#include "Buffer.h"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace monti::vulkan {

struct DeviceDispatch;
class GpuScene;
class EnvironmentMap;

constexpr uint32_t kMaxRayRecursionDepth = 1;
constexpr uint32_t kMaxBindlessTextures = 1024;

// Per-dispatch data pushed each trace call. Only raygen reads these.
struct PushConstants {
    uint32_t frame_index;            // 4 bytes, offset 0
    uint32_t paths_per_pixel;        // 4 bytes, offset 4
    uint32_t max_bounces;            // 4 bytes, offset 8
    uint32_t debug_mode;             // 4 bytes, offset 12
};

static_assert(sizeof(PushConstants) == 16);

// Descriptor set configuration for updating descriptors from Renderer.
struct DescriptorUpdateInfo {
    VkAccelerationStructureKHR tlas;
    std::array<VkImageView, 7> gbuffer_views;  // noisy_diffuse..specular_albedo (bindings 1–7)
    VkBuffer mesh_address_buffer;
    VkDeviceSize mesh_address_buffer_size;
    VkBuffer material_buffer;
    VkDeviceSize material_buffer_size;
    const GpuScene* gpu_scene;       // for bindless textures
    VkBuffer light_buffer;
    VkDeviceSize light_buffer_size;
    VkBuffer blue_noise_buffer;
    VkDeviceSize blue_noise_buffer_size;
    const EnvironmentMap* environment_map;
    VkBuffer frame_uniforms_buffer;
    VkDeviceSize frame_uniforms_buffer_size;
};

// Encapsulates the ray tracing pipeline, descriptor set layout/pool/set,
// pipeline layout, and shader binding table. Owned by Renderer::Impl.
class RaytracePipeline {
public:
    RaytracePipeline() = default;
    ~RaytracePipeline();

    RaytracePipeline(const RaytracePipeline&) = delete;
    RaytracePipeline& operator=(const RaytracePipeline&) = delete;

    // Initialize the full pipeline: descriptor layout, pool, set, pipeline, SBT.
    // shader_dir: directory containing compiled .spv shader files.
    bool Create(VkDevice device, VkPhysicalDevice physical_device,
                VmaAllocator allocator, VkPipelineCache pipeline_cache,
                std::string_view shader_dir,
                const DeviceDispatch& dispatch);

    void Destroy();

    // Update all descriptors with current resources.
    void UpdateDescriptors(const DescriptorUpdateInfo& info);

    VkPipeline Pipeline() const { return pipeline_; }
    VkPipelineLayout PipelineLayout() const { return pipeline_layout_; }
    VkDescriptorSet DescriptorSet() const { return descriptor_set_; }

    const VkStridedDeviceAddressRegionKHR& RaygenRegion() const { return raygen_region_; }
    const VkStridedDeviceAddressRegionKHR& MissRegion() const { return miss_region_; }
    const VkStridedDeviceAddressRegionKHR& HitRegion() const { return hit_region_; }
    const VkStridedDeviceAddressRegionKHR& CallableRegion() const { return callable_region_; }

private:
    bool CreateDescriptorSetLayout();
    bool CreateDescriptorPool();
    bool CreatePipelineAndLayout(VkPipelineCache pipeline_cache,
                                 std::string_view shader_dir);
    bool CreateSbt(VkPhysicalDevice physical_device);

    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    const DeviceDispatch* dispatch_ = nullptr;

    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    Buffer sbt_buffer_;
    VkStridedDeviceAddressRegionKHR raygen_region_{};
    VkStridedDeviceAddressRegionKHR miss_region_{};
    VkStridedDeviceAddressRegionKHR hit_region_{};
    VkStridedDeviceAddressRegionKHR callable_region_{};
};

}  // namespace monti::vulkan
