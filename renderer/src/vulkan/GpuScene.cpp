#include "GpuScene.h"
#include "DeviceDispatch.h"

#include <monti/vulkan/Renderer.h>

#include <bit>
#include <cassert>
#include <cstdio>
#include <cstring>

#include <glm/packing.hpp>

namespace monti::vulkan {

GpuScene::GpuScene(VmaAllocator allocator, VkDevice device, VkPhysicalDevice physical_device,
                   const DeviceDispatch& dispatch)
    : allocator_(allocator), device_(device), dispatch_(&dispatch) {
    VkPhysicalDeviceProperties props{};
    dispatch.vkGetPhysicalDeviceProperties(physical_device, &props);
    max_anisotropy_ = props.limits.maxSamplerAnisotropy;
}

GpuScene::~GpuScene() {
    texture_images_.clear();
}

void GpuScene::RegisterMeshBuffers(MeshId mesh, const MeshBufferBinding& binding) {
    mesh_bindings_[mesh] = binding;

    // Populate buffer address table entry
    auto it = mesh_id_to_address_index_.find(mesh);
    if (it != mesh_id_to_address_index_.end()) {
        // Update existing entry
        auto& entry = mesh_address_entries_[it->second];
        entry.vertex_address = binding.vertex_address;
        entry.index_address = binding.index_address;
        entry.vertex_count = binding.vertex_count;
        entry.index_count = binding.index_count;
    } else {
        auto index = static_cast<uint32_t>(mesh_address_entries_.size());
        mesh_id_to_address_index_[mesh] = index;
        mesh_address_entries_.push_back({
            binding.vertex_address,
            binding.index_address,
            binding.vertex_count,
            binding.index_count,
        });
    }
}

bool GpuScene::UpdateMaterials(const monti::Scene& scene) {
    const auto& materials = scene.Materials();
    if (materials.empty()) return true;

    // Textures must be uploaded before materials so texture indices resolve correctly
    assert(scene.Textures().empty() || !texture_id_to_index_.empty());

    auto count = static_cast<uint32_t>(materials.size());
    VkDeviceSize required_size = count * sizeof(PackedMaterial);

    if (!material_buffer_.EnsureCapacity(
            required_size, allocator_,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT)) {
        std::fprintf(stderr, "GpuScene::UpdateMaterials buffer creation failed\n");
        return false;
    }

    // Pack materials
    std::vector<PackedMaterial> packed(count);
    material_id_to_index_.clear();

    for (uint32_t i = 0; i < count; ++i) {
        const auto& mat = materials[i];
        auto& p = packed[i];

        p.base_color_roughness = glm::vec4(mat.base_color, mat.roughness);

        p.metallic_clearcoat = glm::vec4(
            mat.metallic,
            mat.clear_coat,
            mat.clear_coat_roughness,
            EncodeTextureIndex(mat.base_color_map, texture_id_to_index_));

        p.opacity_ior = glm::vec4(
            mat.opacity,
            mat.ior,
            EncodeTextureIndex(mat.normal_map, texture_id_to_index_),
            EncodeTextureIndex(mat.metallic_roughness_map, texture_id_to_index_));

        p.transmission_volume = glm::vec4(
            mat.transmission_factor,
            mat.thickness_factor,
            mat.attenuation_distance,
            EncodeTextureIndex(mat.transmission_map, texture_id_to_index_));

        p.attenuation_color_pad = glm::vec4(
            mat.attenuation_color,
            EncodeTextureIndex(mat.emissive_map, texture_id_to_index_));

        p.alpha_mode_misc = glm::vec4(
            std::bit_cast<float>(static_cast<uint32_t>(mat.alpha_mode)),
            mat.alpha_cutoff,
            mat.normal_scale,
            mat.uv_rotation);

        p.emissive = glm::vec4(
            mat.emissive_factor,
            mat.emissive_strength);

        p.transmission_ext = glm::vec4(
            mat.diffuse_transmission_factor,
            mat.thin_surface ? 1.0f : 0.0f,
            std::bit_cast<float>(glm::packHalf2x16(
                glm::vec2(mat.diffuse_transmission_color.r,
                           mat.diffuse_transmission_color.g))),
            std::bit_cast<float>(glm::packHalf2x16(
                glm::vec2(mat.diffuse_transmission_color.b,
                           static_cast<float>(mat.nested_priority)))));

        p.uv_transform = glm::vec4(
            mat.uv_offset.x, mat.uv_offset.y,
            mat.uv_scale.x, mat.uv_scale.y);

        p.sheen = glm::vec4(mat.sheen_color, mat.sheen_roughness);

        p.sheen_textures = glm::vec4(
            EncodeTextureIndex(mat.sheen_color_map, texture_id_to_index_),
            EncodeTextureIndex(mat.sheen_roughness_map, texture_id_to_index_),
            0.0f, 0.0f);

        material_id_to_index_[mat.id] = i;
    }

    // Map and copy
    void* mapped = material_buffer_.Map();
    if (!mapped) {
        std::fprintf(stderr, "GpuScene::UpdateMaterials map failed\n");
        return false;
    }
    std::memcpy(mapped, packed.data(), required_size);
    material_buffer_.Unmap();

    return true;
}

std::vector<Buffer> GpuScene::UploadTextures(const monti::Scene& scene,
                                              VkCommandBuffer cmd) {
    const auto& textures = scene.Textures();
    std::vector<Buffer> staging_buffers;

    if (textures.empty()) return staging_buffers;

    texture_images_.clear();
    texture_id_to_index_.clear();
    texture_images_.reserve(textures.size());
    staging_buffers.reserve(textures.size());

    for (uint32_t i = 0; i < static_cast<uint32_t>(textures.size()); ++i) {
        const auto& tex = textures[i];

        VkFormat vk_format = ToVkFormat(tex.format);

        VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (tex.mip_levels > 1)
            usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

        Image gpu_image;
        if (!gpu_image.Create(allocator_, device_, *dispatch_,
                              tex.width, tex.height, vk_format,
                              usage, VK_IMAGE_ASPECT_COLOR_BIT,
                              tex.mip_levels)) {
            std::fprintf(stderr, "GpuScene::UploadTextures image creation failed for '%s'\n",
                         tex.name.c_str());
            return {};
        }

        // Create per-texture sampler
        VkFilter mag = ToVkFilter(tex.mag_filter);
        VkFilter min = ToVkFilter(tex.min_filter);
        VkSamplerAddressMode wrap_u = ToVkAddressMode(tex.wrap_s);
        VkSamplerAddressMode wrap_v = ToVkAddressMode(tex.wrap_t);

        if (!gpu_image.CreateSampler(mag, min, wrap_u, wrap_v, max_anisotropy_)) {
            std::fprintf(stderr, "GpuScene::UploadTextures sampler creation failed for '%s'\n",
                         tex.name.c_str());
            return {};
        }

        // Upload pixel data via staging
        VkDeviceSize pixel_size = static_cast<VkDeviceSize>(tex.data.size());
        Buffer staging = upload::ToImage(allocator_, cmd, gpu_image,
                                         tex.data.data(), pixel_size, *dispatch_);
        if (staging.Handle() == VK_NULL_HANDLE) {
            std::fprintf(stderr, "GpuScene::UploadTextures staging upload failed for '%s'\n",
                         tex.name.c_str());
            return {};
        }

        staging_buffers.push_back(std::move(staging));
        texture_images_.push_back(std::move(gpu_image));
        texture_id_to_index_[tex.id] = i;
    }

    std::printf("GpuScene: uploaded %u textures\n",
                static_cast<uint32_t>(texture_images_.size()));
    return staging_buffers;
}

const MeshBufferBinding* GpuScene::GetMeshBinding(MeshId id) const {
    auto it = mesh_bindings_.find(id);
    if (it == mesh_bindings_.end()) return nullptr;
    return &it->second;
}

VkBuffer GpuScene::MaterialBuffer() const {
    return material_buffer_.Handle();
}

VkDeviceSize GpuScene::MaterialBufferSize() const {
    return material_buffer_.Size();
}

uint32_t GpuScene::GetMaterialIndex(MaterialId id) const {
    auto it = material_id_to_index_.find(id);
    if (it == material_id_to_index_.end()) {
        std::fprintf(stderr, "GpuScene: unknown MaterialId %llu\n",
                     static_cast<unsigned long long>(id.value));
        return kInvalidIndex;
    }
    return it->second;
}

uint32_t GpuScene::TextureCount() const {
    return static_cast<uint32_t>(texture_images_.size());
}

uint32_t GpuScene::GetMeshAddressIndex(MeshId id) const {
    auto it = mesh_id_to_address_index_.find(id);
    if (it == mesh_id_to_address_index_.end()) {
        std::fprintf(stderr, "GpuScene: unknown MeshId %llu in address table\n",
                     static_cast<unsigned long long>(id.value));
        return kInvalidIndex;
    }
    return it->second;
}

VkBuffer GpuScene::MeshAddressBuffer() const {
    return mesh_address_buffer_.Handle();
}

VkDeviceSize GpuScene::MeshAddressBufferSize() const {
    return mesh_address_buffer_.Size();
}

void GpuScene::UploadMeshAddressTable() {
    if (mesh_address_entries_.empty()) return;

    VkDeviceSize required_size = mesh_address_entries_.size() * sizeof(MeshAddressEntry);

    if (!mesh_address_buffer_.EnsureCapacity(
            required_size, allocator_,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT)) {
        std::fprintf(stderr, "GpuScene::UploadMeshAddressTable buffer creation failed\n");
        return;
    }

    void* mapped = mesh_address_buffer_.Map();
    if (!mapped) {
        std::fprintf(stderr, "GpuScene::UploadMeshAddressTable map failed\n");
        return;
    }
    std::memcpy(mapped, mesh_address_entries_.data(), required_size);
    mesh_address_buffer_.Unmap();
}

bool GpuScene::UpdateAreaLights(const monti::Scene& scene) {
    const auto& lights = scene.AreaLights();

    // Always maintain at least a 1-element placeholder so the descriptor is valid
    uint32_t count = std::max(static_cast<uint32_t>(lights.size()), 1u);
    VkDeviceSize required_size = count * sizeof(PackedAreaLight);

    if (!area_light_buffer_.EnsureCapacity(
            required_size, allocator_,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT)) {
        std::fprintf(stderr, "GpuScene::UpdateAreaLights buffer creation failed\n");
        return false;
    }

    std::vector<PackedAreaLight> packed(count, PackedAreaLight{});
    for (uint32_t i = 0; i < static_cast<uint32_t>(lights.size()); ++i) {
        const auto& l = lights[i];
        auto& p = packed[i];
        p.corner_edge_ax = glm::vec4(l.corner, l.edge_a.x);
        p.edge_a_yz_edge_bx = glm::vec4(l.edge_a.y, l.edge_a.z, l.edge_b.x, l.edge_b.y);
        p.edge_bz_radiance = glm::vec4(l.edge_b.z, l.radiance);
        p.flags_pad = glm::vec4(l.two_sided ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
    }

    void* mapped = area_light_buffer_.Map();
    if (!mapped) {
        std::fprintf(stderr, "GpuScene::UpdateAreaLights map failed\n");
        return false;
    }
    std::memcpy(mapped, packed.data(), count * sizeof(PackedAreaLight));
    area_light_buffer_.Unmap();
    return true;
}

VkBuffer GpuScene::AreaLightBuffer() const {
    return area_light_buffer_.Handle();
}

VkDeviceSize GpuScene::AreaLightBufferSize() const {
    return area_light_buffer_.Size();
}

float GpuScene::EncodeTextureIndex(
    std::optional<TextureId> tex_id,
    const std::unordered_map<TextureId, uint32_t>& id_map) {
    if (!tex_id) return std::bit_cast<float>(UINT32_MAX);
    auto it = id_map.find(*tex_id);
    if (it == id_map.end()) return std::bit_cast<float>(UINT32_MAX);
    return std::bit_cast<float>(it->second);
}

VkFilter GpuScene::ToVkFilter(SamplerFilter filter) {
    switch (filter) {
    case SamplerFilter::kLinear:  return VK_FILTER_LINEAR;
    case SamplerFilter::kNearest: return VK_FILTER_NEAREST;
    }
    return VK_FILTER_LINEAR;
}

VkSamplerAddressMode GpuScene::ToVkAddressMode(SamplerWrap wrap) {
    switch (wrap) {
    case SamplerWrap::kRepeat:         return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case SamplerWrap::kClampToEdge:    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case SamplerWrap::kMirroredRepeat: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    }
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

VkFormat GpuScene::ToVkFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::kRGBA8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case PixelFormat::kRGBA16F:     return VK_FORMAT_R16G16B16A16_SFLOAT;
    case PixelFormat::kRGBA32F:     return VK_FORMAT_R32G32B32A32_SFLOAT;
    case PixelFormat::kRG16F:       return VK_FORMAT_R16G16_SFLOAT;
    case PixelFormat::kRG16_SNORM:  return VK_FORMAT_R16G16_SNORM;
    case PixelFormat::kR32F:        return VK_FORMAT_R32_SFLOAT;
    case PixelFormat::kR8_UNORM:    return VK_FORMAT_R8_UNORM;
    }
    return VK_FORMAT_R8G8B8A8_UNORM;
}

}  // namespace monti::vulkan
