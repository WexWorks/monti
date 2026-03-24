#pragma once

#include "Buffer.h"
#include "Image.h"
#include "Upload.h"

#include <monti/scene/Types.h>
#include <monti/scene/Scene.h>

namespace monti::vulkan { struct DeviceDispatch; }

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <glm/glm.hpp>

#include <bit>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace monti::vulkan {

constexpr uint32_t kInvalidIndex = UINT32_MAX;

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

// GPU-packed material: eleven vec4s per material for storage buffer upload.
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
    glm::vec4 transmission_volume;    // .r = transmission_factor, .g = reserved,
                                      // .b = attenuation_distance,
                                      // .a = transmission_map index
    glm::vec4 attenuation_color_pad;  // .rgb = attenuation_color,
                                      // .a = emissive_map index
    glm::vec4 alpha_mode_misc;        // .r = alpha_mode (float-encoded uint: 0/1/2),
                                      // .g = alpha_cutoff,
                                      // .b = normal_scale,
                                      // .a = uv_rotation
    glm::vec4 emissive;               // .rgb = emissive_factor, .a = emissive_strength
    glm::vec4 transmission_ext;       // .r = diffuse_transmission_factor,
                                      // .g = thin_surface (0.0/1.0),
                                      // .b = packHalf2x16(dt_color.rg) as float,
                                      // .a = packHalf2x16(vec2(dt_color.b, nested_priority)) as float
    glm::vec4 uv_transform;           // .rg = uv_offset, .ba = uv_scale
    glm::vec4 sheen;                  // .rgb = sheen_color, .a = sheen_roughness
    glm::vec4 sheen_textures;         // .r = sheen_color_map index,
                                      // .g = sheen_roughness_map index,
                                      // .b = reserved,
                                      // .a = reserved
};

static_assert(sizeof(PackedMaterial) == 176);

// Light type discriminator encoded in PackedLight.data0.w
enum class LightType : uint32_t { kQuad = 0, kSphere = 1, kTriangle = 2 };

// GPU-packed light: unified format for all light types (64 bytes, 4 × vec4).
struct alignas(16) PackedLight {
    glm::vec4 data0;  // Quad: corner.xyz, type
                      // Sphere: center.xyz, type
                      // Triangle: v0.xyz, type
    glm::vec4 data1;  // Quad: edge_a.xyz, two_sided
                      // Sphere: radius, 0, 0, 0
                      // Triangle: v1.xyz, two_sided
    glm::vec4 data2;  // Quad: edge_b.xyz, 0
                      // Sphere: 0, 0, 0, 0
                      // Triangle: v2.xyz, 0
    glm::vec4 data3;  // All: radiance.xyz, 0
};

static_assert(sizeof(PackedLight) == 64);

// Internal GPU representation of a Scene. Manages:
// - Mesh buffer bindings (host-provided VkBuffer handles + device addresses)
// - Packed material storage buffer (host-visible, direct memcpy)
// - Texture images (device-local, per-texture VkSampler)
//
// This class is internal to the renderer — not exposed in public headers.
class GpuScene {
public:
    GpuScene(VmaAllocator allocator, VkDevice device, VkPhysicalDevice physical_device,
             const DeviceDispatch& dispatch);
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

    // Light buffer — creates a placeholder on first call,
    // re-uploads when the scene's lights change.
    bool UpdateLights(const monti::Scene& scene);
    VkBuffer LightBuffer() const;
    VkDeviceSize LightBufferSize() const;

private:
    static float EncodeTextureIndex(std::optional<TextureId> tex_id,
                                    const std::unordered_map<TextureId, uint32_t>& id_map);
    static VkFilter ToVkFilter(SamplerFilter filter);
    static VkSamplerAddressMode ToVkAddressMode(SamplerWrap wrap);
    static VkFormat ToVkFormat(PixelFormat format, bool srgb = false);

    VmaAllocator allocator_;
    VkDevice device_;
    const DeviceDispatch* dispatch_ = nullptr;
    float max_anisotropy_ = 16.0f;

    std::unordered_map<MeshId, MeshBufferBinding> mesh_bindings_;

    Buffer material_buffer_;
    std::unordered_map<MaterialId, uint32_t> material_id_to_index_;

    std::vector<Image> texture_images_;
    std::unordered_map<TextureId, uint32_t> texture_id_to_index_;

    std::vector<MeshAddressEntry> mesh_address_entries_;
    std::unordered_map<MeshId, uint32_t> mesh_id_to_address_index_;
    Buffer mesh_address_buffer_;

    Buffer light_buffer_;
};

}  // namespace monti::vulkan
