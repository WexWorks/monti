#pragma once

#include "Buffer.h"
#include "Image.h"
#include "Upload.h"

#include <monti/scene/Types.h>
#include <monti/scene/Scene.h>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <glm/glm.hpp>

#include <bit>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace monti::vulkan {

struct MeshBufferBinding;

// GPU-side mesh address entry for buffer_reference access in shaders.
// One entry per registered mesh. Layout matches std430.
struct MeshAddressEntry {
    uint64_t vertex_address;
    uint64_t index_address;
    uint32_t vertex_count;
    uint32_t index_count;
    uint32_t pad_[2] = {};  // Pad to 32 bytes for array stride alignment
};

static_assert(sizeof(MeshAddressEntry) == 32);

// GPU-packed material: five vec4s per material for storage buffer upload.
// All texture indices are float-encoded uint32_t via std::bit_cast<float>().
// UINT32_MAX = no texture. Shader checks: floatBitsToUint(idx) == 0xFFFFFFFFu.
struct alignas(16) PackedMaterial {
    glm::vec4 base_color_roughness;   // .rgb = base_color, .a = roughness
    glm::vec4 metallic_clearcoat;     // .r = metallic, .g = clear_coat,
                                      // .b = clear_coat_roughness,
                                      // .a = base_color_map index
    glm::vec4 opacity_ior;            // .r = opacity, .g = ior,
                                      // .b = normal_map index,
                                      // .a = metallic_roughness_map index
    glm::vec4 transmission_volume;    // .r = transmission_factor, .g = thickness,
                                      // .b = attenuation_distance,
                                      // .a = transmission_map index
    glm::vec4 attenuation_color_pad;  // .rgb = attenuation_color,
                                      // .a = emissive_map index
};

static_assert(sizeof(PackedMaterial) == 80);

// Internal GPU representation of a Scene. Manages:
// - Mesh buffer bindings (host-provided VkBuffer handles + device addresses)
// - Packed material storage buffer (host-visible, direct memcpy)
// - Texture images (device-local, per-texture VkSampler)
//
// This class is internal to the renderer — not exposed in public headers.
class GpuScene {
public:
    GpuScene(VmaAllocator allocator, VkDevice device, VkPhysicalDevice physical_device);
    ~GpuScene();

    GpuScene(const GpuScene&) = delete;
    GpuScene& operator=(const GpuScene&) = delete;

    // Register host-owned GPU buffers for a mesh.
    void RegisterMeshBuffers(MeshId mesh, const MeshBufferBinding& binding);

    // Pack CPU-side materials from Scene into host-visible GPU storage buffer.
    // Allocates buffer on first call; reallocates if material count grows.
    bool UpdateMaterials(const monti::Scene& scene);

    // Upload all textures from Scene to device-local VkImages with staging.
    // Creates VkImageView + VkSampler per texture. Generates mip chain when
    // TextureDesc::mip_levels > 1. Records commands into cmd.
    // Returns staging buffers that must be kept alive until cmd completes.
    std::vector<Buffer> UploadTextures(const monti::Scene& scene,
                                       VkCommandBuffer cmd);

    // Accessors
    const MeshBufferBinding* GetMeshBinding(MeshId id) const;
    VkBuffer MaterialBuffer() const;
    VkDeviceSize MaterialBufferSize() const;
    uint32_t GetMaterialIndex(MaterialId id) const;
    uint32_t TextureCount() const;
    const std::vector<Image>& TextureImages() const { return texture_images_; }

    uint32_t MeshBindingCount() const {
        return static_cast<uint32_t>(mesh_bindings_.size());
    }
    const std::unordered_map<MeshId, MeshBufferBinding>& MeshBindings() const {
        return mesh_bindings_;
    }
    uint32_t MaterialCount() const {
        return static_cast<uint32_t>(material_id_to_index_.size());
    }

    // Buffer address table
    uint32_t GetMeshAddressIndex(MeshId id) const;
    VkBuffer MeshAddressBuffer() const;
    VkDeviceSize MeshAddressBufferSize() const;
    void UploadMeshAddressTable();

private:
    static float EncodeTextureIndex(std::optional<TextureId> tex_id,
                                    const std::unordered_map<TextureId, uint32_t>& id_map);
    static VkFilter ToVkFilter(SamplerFilter filter);
    static VkSamplerAddressMode ToVkAddressMode(SamplerWrap wrap);
    static VkFormat ToVkFormat(PixelFormat format);

    VmaAllocator allocator_;
    VkDevice device_;
    float max_anisotropy_ = 16.0f;

    std::unordered_map<MeshId, MeshBufferBinding> mesh_bindings_;

    Buffer material_buffer_;
    uint32_t material_buffer_capacity_ = 0;
    std::unordered_map<MaterialId, uint32_t> material_id_to_index_;

    std::vector<Image> texture_images_;
    std::unordered_map<TextureId, uint32_t> texture_id_to_index_;

    std::vector<MeshAddressEntry> mesh_address_entries_;
    std::unordered_map<MeshId, uint32_t> mesh_id_to_address_index_;
    Buffer mesh_address_buffer_;
};

}  // namespace monti::vulkan
