#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <memory>

namespace deni::vulkan {

enum class ScaleMode {
    kNative,       // 1.0x — denoise only, no upscaling
    kQuality,      // 1.5x — render at 720p for 1080p output
    kPerformance,  // 2.0x — render at 540p for 1080p output
};

struct DenoiserInput {
    VkImageView noisy_diffuse;    // RGBA16F — diffuse radiance (1-N spp)
    VkImageView noisy_specular;   // RGBA16F — specular radiance (1-N spp)
    VkImageView motion_vectors;   // RG16F   — screen-space motion (pixels)
    VkImageView linear_depth;     // R16F    — view-space linear Z
    VkImageView world_normals;    // RGBA16F — world normals (.xyz), roughness (.w)
    VkImageView diffuse_albedo;   // R11G11B10F — diffuse reflectance
    VkImageView specular_albedo;  // R11G11B10F — specular F0

    uint32_t  render_width;
    uint32_t  render_height;
    ScaleMode scale_mode = ScaleMode::kNative;

    bool reset_accumulation;      // True on camera cut or scene reset
};

struct DenoiserOutput {
    VkImage     denoised_image;   // RGBA16F — denoised output (GENERAL layout)
    VkImageView denoised_color;   // RGBA16F — denoised radiance
};

struct DenoiserDesc {
    VkDevice         device;
    VkPhysicalDevice physical_device;
    uint32_t         width  = 1920;
    uint32_t         height = 1080;
    VkPipelineCache  pipeline_cache = VK_NULL_HANDLE;
    VmaAllocator     allocator;
    PFN_vkGetDeviceProcAddr get_device_proc_addr = nullptr;
};

class Denoiser {
public:
    static std::unique_ptr<Denoiser> Create(const DenoiserDesc& desc);
    ~Denoiser();

    DenoiserOutput Denoise(VkCommandBuffer cmd, const DenoiserInput& input);
    void Resize(uint32_t width, uint32_t height);
    float LastPassTimeMs() const;

private:
    Denoiser() = default;
};

} // namespace deni::vulkan
