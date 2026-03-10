#pragma once
#include "Types.h"
#include <glm/glm.hpp>
#include <optional>
#include <string>
#include <vector>

namespace monti {

// ── Mesh (metadata only — geometry lives on GPU) ─────────────────────────

struct Mesh {
    MeshId id;
    std::string name;
    uint32_t vertex_count  = 0;
    uint32_t index_count   = 0;
    uint32_t vertex_stride = sizeof(Vertex);
    glm::vec3 bbox_min{0};
    glm::vec3 bbox_max{0};
};

// ── Transient Mesh Data (for loaders) ────────────────────────────────────

struct MeshData {
    MeshId mesh_id;
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;
};

// ── Texture Registration ─────────────────────────────────────────────────

struct TextureDesc {
    TextureId   id;
    std::string name;
    uint32_t    width = 0;
    uint32_t    height = 0;
    uint32_t    mip_levels = 1;
    PixelFormat format = PixelFormat::kRGBA8_UNORM;
    std::vector<uint8_t> data;
};

// ── PBR Material ─────────────────────────────────────────────────────────

struct MaterialDesc {
    MaterialId id;
    std::string name;

    // Base PBR
    glm::vec3 base_color       = {1, 1, 1};
    float     roughness        = 0.5f;
    float     metallic         = 0.0f;
    float     opacity          = 1.0f;
    float     ior              = 1.5f;

    std::optional<TextureId> base_color_map;
    std::optional<TextureId> normal_map;
    std::optional<TextureId> metallic_roughness_map;

    float normal_scale         = 1.0f;

    // Clear coat
    float clear_coat           = 0.0f;
    float clear_coat_roughness = 0.1f;

    // Alpha
    enum class AlphaMode { kOpaque, kMask, kBlend };
    AlphaMode alpha_mode       = AlphaMode::kOpaque;
    float     alpha_cutoff     = 0.5f;
    bool      double_sided     = false;

    // Emissive (parsed and stored; rendering deferred — requires ReSTIR)
    glm::vec3 emissive_factor    = {0, 0, 0};
    std::optional<TextureId> emissive_map;
    float     emissive_strength  = 1.0f;

    // Transmission/volume
    float     transmission_factor  = 0.0f;
    std::optional<TextureId> transmission_map;
    glm::vec3 attenuation_color    = {1, 1, 1};
    float     attenuation_distance = 0.0f;
    float     thickness_factor     = 0.0f;
};

} // namespace monti
