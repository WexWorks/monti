#pragma once
#include <monti/scene/Scene.h>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <functional>
#include <memory>
#include <string>

namespace monti::vulkan {

struct GBuffer {
    VkImageView noisy_diffuse;    // RGBA16F
    VkImageView noisy_specular;   // RGBA16F
    VkImageView motion_vectors;   // RG16F
    VkImageView linear_depth;     // RG16F
    VkImageView world_normals;    // RGBA16F
    VkImageView diffuse_albedo;   // R11G11B10F
    VkImageView specular_albedo;  // R11G11B10F

    // VkImage handles for per-frame layout transitions (UNDEFINED → GENERAL)
    VkImage noisy_diffuse_image   = VK_NULL_HANDLE;
    VkImage noisy_specular_image  = VK_NULL_HANDLE;
    VkImage motion_vectors_image  = VK_NULL_HANDLE;
    VkImage linear_depth_image    = VK_NULL_HANDLE;
    VkImage world_normals_image   = VK_NULL_HANDLE;
    VkImage diffuse_albedo_image  = VK_NULL_HANDLE;
    VkImage specular_albedo_image = VK_NULL_HANDLE;
};

struct RendererDesc {
    VkDevice         device;
    VkPhysicalDevice physical_device;
    VkInstance       instance;
    VkQueue          queue;
    uint32_t         queue_family_index;
    VmaAllocator     allocator;
    PFN_vkGetDeviceProcAddr   get_device_proc_addr;
    PFN_vkGetInstanceProcAddr get_instance_proc_addr;
    std::string      shader_dir;  // directory containing compiled .spv files
    VkPipelineCache  pipeline_cache = VK_NULL_HANDLE;
    uint32_t         width             = 1920;
    uint32_t         height            = 1080;
    uint32_t         samples_per_pixel = 4;
};

// Host-provided GPU buffer handles and device addresses for a mesh.
// Passed to Renderer::RegisterMeshBuffers() after the host uploads
// vertex/index data to GPU buffers.
struct MeshBufferBinding {
    VkBuffer         vertex_buffer;
    VkDeviceAddress  vertex_address;
    VkBuffer         index_buffer;
    VkDeviceAddress  index_address;
    uint32_t         vertex_count;
    uint32_t         index_count;
    uint32_t         vertex_stride = sizeof(monti::Vertex);
};

class Renderer {
public:
    static std::unique_ptr<Renderer> Create(const RendererDesc& desc);
    ~Renderer();

    void SetScene(monti::Scene* scene);

    // Register host-owned GPU buffers for a mesh. Called after the host
    // uploads vertex/index data to GPU buffers. Delegates to the internal
    // GpuScene. Device addresses are used for BLAS building and shader
    // buffer_reference access.
    void RegisterMeshBuffers(MeshId mesh, const MeshBufferBinding& binding);

    void NotifyMeshDeformed(MeshId mesh, bool topology_changed = false);

    using MeshCleanupCallback = std::function<void(MeshId)>;
    void SetMeshCleanupCallback(MeshCleanupCallback callback);

    bool RenderFrame(VkCommandBuffer cmd, const GBuffer& output,
                     uint32_t frame_index);

    void SetSamplesPerPixel(uint32_t spp);
    uint32_t GetSamplesPerPixel() const;

    void SetMaxBounces(uint32_t bounces);

    void SetDebugMode(uint32_t mode);

    void SetBackgroundMode(bool show_environment, float blur_level = 0.0f);

    void Resize(uint32_t width, uint32_t height);

    float LastFrameTimeMs() const;

private:
    Renderer();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace monti::vulkan
