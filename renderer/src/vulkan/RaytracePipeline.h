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

class GpuScene;
class EnvironmentMap;

constexpr uint32_t kMaxRayRecursionDepth = 1;
constexpr uint32_t kMaxBindlessTextures = 1024;

struct PushConstants {
    // ── Camera (192 bytes) ───────────────────────────────────────
    glm::mat4 inv_view;              // 64 bytes, offset 0
    glm::mat4 inv_proj;              // 64 bytes, offset 64
    glm::mat4 prev_view_proj;        // 64 bytes, offset 128

    // ── Render parameters (16 bytes) ─────────────────────────────
    uint32_t frame_index;            // 4 bytes, offset 192
    uint32_t paths_per_pixel;        // 4 bytes, offset 196
    uint32_t max_bounces;            // 4 bytes, offset 200
    uint32_t area_light_count;       // 4 bytes, offset 204

    // ── Scene globals (16 bytes) ─────────────────────────────────
    uint32_t env_width;              // 4 bytes, offset 208
    uint32_t env_height;             // 4 bytes, offset 212
    float    env_avg_luminance;      // 4 bytes, offset 216
    float    env_max_luminance;      // 4 bytes, offset 220

    // ── Scene globals continued (16 bytes) ───────────────────────
    float    env_rotation;           // 4 bytes, offset 224 (radians)
    float    skybox_mip_level;       // 4 bytes, offset 228
    float    jitter_x;              // 4 bytes, offset 232
    float    jitter_y;              // 4 bytes, offset 236

    // ── Debug (8 bytes + 8 padding → 16 bytes) ───────────────────
    uint32_t debug_mode;             // 4 bytes, offset 240
    uint32_t pad0;                   // 4 bytes, offset 244 (pad to 248)
};

static_assert(sizeof(PushConstants) == 248);

// Descriptor set configuration for updating descriptors from Renderer.
struct DescriptorUpdateInfo {
    VkAccelerationStructureKHR tlas;
    std::array<VkImageView, 7> gbuffer_views;  // noisy_diffuse..specular_albedo (bindings 1–7)
    VkBuffer mesh_address_buffer;
    VkDeviceSize mesh_address_buffer_size;
    VkBuffer material_buffer;
    VkDeviceSize material_buffer_size;
    const GpuScene* gpu_scene;       // for bindless textures
    VkBuffer area_light_buffer;
    VkDeviceSize area_light_buffer_size;
    VkBuffer blue_noise_buffer;
    VkDeviceSize blue_noise_buffer_size;
    const EnvironmentMap* environment_map;
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
                std::string_view shader_dir);

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

    static std::vector<uint8_t> LoadShaderFile(std::string_view path);

    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;

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
