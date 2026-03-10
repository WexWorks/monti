#pragma once
#include <monti/scene/Scene.h>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <functional>
#include <memory>

namespace monti::vulkan {

struct GBuffer {
    VkImageView noisy_diffuse;    // RGBA16F
    VkImageView noisy_specular;   // RGBA16F
    VkImageView motion_vectors;   // RG16F
    VkImageView linear_depth;     // R16F
    VkImageView world_normals;    // RGBA16F
    VkImageView diffuse_albedo;   // R11G11B10F
    VkImageView specular_albedo;  // R11G11B10F
};

struct RendererDesc {
    VkDevice         device;
    VkPhysicalDevice physical_device;
    VkQueue          queue;
    uint32_t         queue_family_index;
    VkPipelineCache  pipeline_cache = VK_NULL_HANDLE;
    VmaAllocator     allocator;
    uint32_t         width             = 1920;
    uint32_t         height            = 1080;
    uint32_t         samples_per_pixel = 4;
    PFN_vkGetDeviceProcAddr get_device_proc_addr = nullptr;
};

class Renderer {
public:
    static std::unique_ptr<Renderer> Create(const RendererDesc& desc);
    ~Renderer();

    void SetScene(monti::Scene* scene);

    void NotifyMeshDeformed(MeshId mesh, bool topology_changed = false);

    using MeshCleanupCallback = std::function<void(MeshId)>;
    void SetMeshCleanupCallback(MeshCleanupCallback callback);

    bool RenderFrame(VkCommandBuffer cmd, const GBuffer& output,
                     uint32_t frame_index);

    void SetSamplesPerPixel(uint32_t spp);
    uint32_t GetSamplesPerPixel() const;

    void Resize(uint32_t width, uint32_t height);

    float LastFrameTimeMs() const;

private:
    Renderer() = default;
};

} // namespace monti::vulkan
