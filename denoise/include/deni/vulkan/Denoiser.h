#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <cstdint>
#include <memory>
#include <string_view>

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
    std::string_view shader_dir;
    PFN_vkGetDeviceProcAddr get_device_proc_addr;
};

class Denoiser {
public:
    static std::unique_ptr<Denoiser> Create(const DenoiserDesc& desc);
    ~Denoiser();

    Denoiser(const Denoiser&) = delete;
    Denoiser& operator=(const Denoiser&) = delete;

    DenoiserOutput Denoise(VkCommandBuffer cmd, const DenoiserInput& input);
    void Resize(uint32_t width, uint32_t height);
    float LastPassTimeMs() const;

private:
    Denoiser() = default;

    bool CreateOutputImage(uint32_t width, uint32_t height);
    bool CreateDescriptorLayout();
    bool AllocateDescriptorSet();
    bool CreatePipeline(std::string_view shader_dir, VkPipelineCache pipeline_cache);
    void UpdateDescriptorSet(const DenoiserInput& input);
    void DestroyOutputImage();

    struct DeviceDispatch;
    std::unique_ptr<DeviceDispatch> dispatch_;

    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;

    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;

    VkImage output_image_ = VK_NULL_HANDLE;
    VkImageView output_view_ = VK_NULL_HANDLE;
    VmaAllocation output_allocation_ = VK_NULL_HANDLE;
    uint32_t output_width_ = 0;
    uint32_t output_height_ = 0;
};

} // namespace deni::vulkan
