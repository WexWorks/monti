#include "gltf/GltfLoader.h"

#define CGLTF_IMPLEMENTATION
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <cgltf.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#define STB_IMAGE_IMPLEMENTATION
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <stb_image.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#define DDSKTX_IMPLEMENT
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <dds-ktx.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <mikktspace.h>

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>
#include <unordered_map>
#include <vector>

namespace monti::gltf {

namespace {

// ── Helper: read accessor data ───────────────────────────────────────────

glm::vec3 ReadVec3(const cgltf_accessor* accessor, cgltf_size index) {
    float v[3]{};
    cgltf_accessor_read_float(accessor, index, v, 3);
    return {v[0], v[1], v[2]};
}

glm::vec4 ReadVec4(const cgltf_accessor* accessor, cgltf_size index) {
    float v[4]{};
    cgltf_accessor_read_float(accessor, index, v, 4);
    return {v[0], v[1], v[2], v[3]};
}

glm::vec2 ReadVec2(const cgltf_accessor* accessor, cgltf_size index) {
    float v[2]{};
    cgltf_accessor_read_float(accessor, index, v, 2);
    return {v[0], v[1]};
}

// ── Face-weighted normal generation ──────────────────────────────────────

void GenerateFaceWeightedNormals(std::vector<Vertex>& vertices,
                                 const std::vector<uint32_t>& indices) {
    for (auto& v : vertices)
        v.normal = glm::vec3(0.0f);

    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        auto& v0 = vertices[indices[i]];
        auto& v1 = vertices[indices[i + 1]];
        auto& v2 = vertices[indices[i + 2]];

        glm::vec3 edge1 = v1.position - v0.position;
        glm::vec3 edge2 = v2.position - v0.position;
        glm::vec3 face_normal = glm::cross(edge1, edge2);
        // face_normal length is proportional to triangle area (face-weighted)

        v0.normal += face_normal;
        v1.normal += face_normal;
        v2.normal += face_normal;
    }

    for (auto& v : vertices) {
        float len = glm::length(v.normal);
        constexpr float kNormalEpsilon = 1e-8f;
        if (len > kNormalEpsilon) v.normal /= len;
        else v.normal = glm::vec3(0.0f, 1.0f, 0.0f);
    }
}

// ── MikkTSpace tangent generation ────────────────────────────────────────

struct MikkContext {
    std::vector<Vertex>* vertices;
    const std::vector<uint32_t>* indices;
};

int MikkGetNumFaces(const SMikkTSpaceContext* ctx) {
    auto* c = static_cast<MikkContext*>(ctx->m_pUserData);
    return static_cast<int>(c->indices->size() / 3);
}

int MikkGetNumVerticesOfFace(const SMikkTSpaceContext*, int) {
    return 3;
}

void MikkGetPosition(const SMikkTSpaceContext* ctx, float out[], int face, int vert) {
    auto* c = static_cast<MikkContext*>(ctx->m_pUserData);
    uint32_t idx = (*c->indices)[static_cast<size_t>(face) * 3 + vert];
    const auto& pos = (*c->vertices)[idx].position;
    out[0] = pos.x; out[1] = pos.y; out[2] = pos.z;
}

void MikkGetNormal(const SMikkTSpaceContext* ctx, float out[], int face, int vert) {
    auto* c = static_cast<MikkContext*>(ctx->m_pUserData);
    uint32_t idx = (*c->indices)[static_cast<size_t>(face) * 3 + vert];
    const auto& n = (*c->vertices)[idx].normal;
    out[0] = n.x; out[1] = n.y; out[2] = n.z;
}

void MikkGetTexCoord(const SMikkTSpaceContext* ctx, float out[], int face, int vert) {
    auto* c = static_cast<MikkContext*>(ctx->m_pUserData);
    uint32_t idx = (*c->indices)[static_cast<size_t>(face) * 3 + vert];
    const auto& uv = (*c->vertices)[idx].tex_coord_0;
    out[0] = uv.x; out[1] = uv.y;
}

void MikkSetTSpaceBasic(const SMikkTSpaceContext* ctx,
                        const float tangent[], float sign, int face, int vert) {
    auto* c = static_cast<MikkContext*>(ctx->m_pUserData);
    uint32_t idx = (*c->indices)[static_cast<size_t>(face) * 3 + vert];
    (*c->vertices)[idx].tangent = glm::vec4(tangent[0], tangent[1], tangent[2], sign);
}

void GenerateMikkTangents(std::vector<Vertex>& vertices,
                          const std::vector<uint32_t>& indices) {
    MikkContext mctx{&vertices, &indices};

    SMikkTSpaceInterface iface{};
    iface.m_getNumFaces = MikkGetNumFaces;
    iface.m_getNumVerticesOfFace = MikkGetNumVerticesOfFace;
    iface.m_getPosition = MikkGetPosition;
    iface.m_getNormal = MikkGetNormal;
    iface.m_getTexCoord = MikkGetTexCoord;
    iface.m_setTSpaceBasic = MikkSetTSpaceBasic;

    SMikkTSpaceContext ctx{};
    ctx.m_pInterface = &iface;
    ctx.m_pUserData = &mctx;

    genTangSpaceDefault(&ctx);
}

// ── Sampler mapping ──────────────────────────────────────────────────────

constexpr int kGlClampToEdge         = 33071;
constexpr int kGlMirroredRepeat      = 33648;
constexpr int kGlNearest             = 9728;
constexpr int kGlNearestMipmapNearest = 9984;
constexpr int kGlNearestMipmapLinear  = 9986;

SamplerWrap MapWrap(int gltf_wrap) {
    switch (gltf_wrap) {
    case kGlClampToEdge:    return SamplerWrap::kClampToEdge;
    case kGlMirroredRepeat: return SamplerWrap::kMirroredRepeat;
    default:                return SamplerWrap::kRepeat;         // 10497 or unspecified
    }
}

SamplerFilter MapFilter(int gltf_filter) {
    switch (gltf_filter) {
    case kGlNearest:              return SamplerFilter::kNearest;   // NEAREST
    case kGlNearestMipmapNearest: return SamplerFilter::kNearest;   // NEAREST_MIPMAP_NEAREST
    case kGlNearestMipmapLinear:  return SamplerFilter::kNearest;   // NEAREST_MIPMAP_LINEAR
    default:                      return SamplerFilter::kLinear;     // LINEAR or any mipmap variant
    }
}

// ── Texture extraction ───────────────────────────────────────────────────

// Maps cgltf_texture index → monti TextureId.
using TextureLookup = std::unordered_map<cgltf_size, TextureId>;

// ── DDS texture decoding ─────────────────────────────────────────────────

// Map dds-ktx format to monti PixelFormat. Returns nullopt for unsupported formats.
std::optional<PixelFormat> MapDdsFormat(ddsktx_format fmt) {
    switch (fmt) {
    case DDSKTX_FORMAT_BC1:  return PixelFormat::kBC1_UNORM;
    case DDSKTX_FORMAT_BC3:  return PixelFormat::kBC3_UNORM;
    case DDSKTX_FORMAT_BC4:  return PixelFormat::kBC4_UNORM;
    case DDSKTX_FORMAT_BC5:  return PixelFormat::kBC5_UNORM;
    case DDSKTX_FORMAT_BC7:  return PixelFormat::kBC7_UNORM;
    default:                 return std::nullopt;
    }
}

// Decode a DDS file into a TextureDesc with BC-compressed data and mip offsets.
std::optional<TextureDesc> DecodeDdsImage(const uint8_t* raw_data, size_t raw_size,
                                          std::string_view name) {
    ddsktx_texture_info info{};
    if (!ddsktx_parse(&info, raw_data, static_cast<int>(raw_size), nullptr)) {
        std::fprintf(stderr, "DecodeDdsImage: failed to parse DDS '%.*s'\n",
                     static_cast<int>(name.size()), name.data());
        return std::nullopt;
    }

    auto pixel_format = MapDdsFormat(info.format);
    if (!pixel_format) {
        std::fprintf(stderr, "DecodeDdsImage: unsupported DDS format %d in '%.*s'\n",
                     static_cast<int>(info.format),
                     static_cast<int>(name.size()), name.data());
        return std::nullopt;
    }

    TextureDesc desc;
    desc.name       = std::string(name);
    desc.width      = static_cast<uint32_t>(info.width);
    desc.height     = static_cast<uint32_t>(info.height);
    desc.mip_levels = static_cast<uint32_t>(info.num_mips);
    desc.format     = *pixel_format;

    // Concatenate all mip levels into desc.data, record offsets
    desc.mip_offsets.resize(desc.mip_levels);
    uint32_t total_size = 0;
    for (int mip = 0; mip < info.num_mips; ++mip) {
        ddsktx_sub_data sub{};
        ddsktx_get_sub(&info, &sub, raw_data, static_cast<int>(raw_size), 0, 0, mip);
        total_size += static_cast<uint32_t>(sub.size_bytes);
    }
    desc.data.resize(total_size);

    uint32_t offset = 0;
    for (int mip = 0; mip < info.num_mips; ++mip) {
        ddsktx_sub_data sub{};
        ddsktx_get_sub(&info, &sub, raw_data, static_cast<int>(raw_size), 0, 0, mip);
        desc.mip_offsets[mip] = offset;
        std::memcpy(desc.data.data() + offset, sub.buff, sub.size_bytes);
        offset += static_cast<uint32_t>(sub.size_bytes);
    }

    return desc;
}

// Check if a URI ends with ".dds" (case-insensitive).
bool HasDdsExtension(std::string_view uri) {
    if (uri.size() < 4) return false;
    auto ext = uri.substr(uri.size() - 4);
    return (ext[0] == '.') &&
           (ext[1] == 'd' || ext[1] == 'D') &&
           (ext[2] == 'd' || ext[2] == 'D') &&
           (ext[3] == 's' || ext[3] == 'S');
}

// DDS magic bytes: "DDS " = 0x20534444
constexpr uint32_t kDdsMagic = 0x20534444;

// ── Image decoding (DDS + stb_image) ────────────────────────────────────

// Decode a glTF image into a TextureDesc (without adding to Scene).
// Returns nullopt if the image cannot be decoded.
// For URI-based images (typical in .gltf files), resolves relative to base_dir.
std::optional<TextureDesc> DecodeImage(const cgltf_image& image,
                                       std::string_view base_dir) {
    const uint8_t* raw_data = nullptr;
    cgltf_size raw_size = 0;

    if (image.buffer_view) {
        raw_data = static_cast<const uint8_t*>(image.buffer_view->buffer->data)
                 + image.buffer_view->offset;
        raw_size = image.buffer_view->size;
    }

    // Check for DDS: URI extension or buffer magic bytes
    bool is_dds = false;
    if (image.uri && HasDdsExtension(image.uri)) {
        is_dds = true;
    } else if (raw_data && raw_size >= 4) {
        uint32_t magic = 0;
        std::memcpy(&magic, raw_data, 4);
        if (magic == kDdsMagic) is_dds = true;
    }

    if (is_dds) {
        std::string_view name = image.name ? image.name
                              : (image.uri ? image.uri : "");
        if (raw_data && raw_size > 0)
            return DecodeDdsImage(raw_data, raw_size, name);

        // Load DDS file from disk
        std::string image_path = std::string(base_dir) + image.uri;
        std::ifstream file(image_path, std::ios::binary | std::ios::ate);
        if (!file) {
            std::fprintf(stderr, "Failed to load DDS image: %s\n", image_path.c_str());
            return std::nullopt;
        }
        auto file_size = file.tellg();
        file.seekg(0);
        std::vector<uint8_t> file_data(static_cast<size_t>(file_size));
        file.read(reinterpret_cast<char*>(file_data.data()), file_size);
        return DecodeDdsImage(file_data.data(), file_data.size(), name);
    }

    // Standard image path (PNG/JPG/TGA/BMP via stb_image)
    int w = 0, h = 0, channels = 0;
    stbi_uc* pixels = nullptr;

    if (raw_data && raw_size > 0) {
        pixels = stbi_load_from_memory(
            raw_data, static_cast<int>(raw_size), &w, &h, &channels, 4);
    } else if (image.uri) {
        // Load external image file relative to the glTF directory
        std::string image_path = std::string(base_dir) + image.uri;
        pixels = stbi_load(image_path.c_str(), &w, &h, &channels, 4);
        if (!pixels)
            std::fprintf(stderr, "Failed to load image: %s\n", image_path.c_str());
    }

    if (!pixels) return std::nullopt;

    TextureDesc desc;
    desc.name   = image.name ? image.name : (image.uri ? image.uri : "");
    desc.width  = static_cast<uint32_t>(w);
    desc.height = static_cast<uint32_t>(h);
    desc.format = PixelFormat::kRGBA8_UNORM;
    desc.data.assign(pixels, pixels + static_cast<size_t>(w) * h * 4);
    stbi_image_free(pixels);

    return desc;
}

// Resolve the image source for a glTF texture, preferring MSFT_texture_dds
// DDS source over the standard PNG/JPG fallback when available.
const cgltf_image* ResolveDdsImage(const cgltf_texture& tex,
                                   const cgltf_data* data) {
    for (cgltf_size ei = 0; ei < tex.extensions_count; ++ei) {
        const auto& ext = tex.extensions[ei];
        if (!ext.name || !ext.data) continue;
        if (std::string_view(ext.name) != "MSFT_texture_dds") continue;

        // Parse minimal JSON: {"source": N}
        std::string_view json(ext.data);
        auto pos = json.find("\"source\"");
        if (pos == std::string_view::npos) continue;
        pos = json.find(':', pos + 8);
        if (pos == std::string_view::npos) continue;
        ++pos;
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;

        int source_index = 0;
        auto [ptr, ec] = std::from_chars(json.data() + pos,
                                         json.data() + json.size(),
                                         source_index);
        if (ec != std::errc{}) continue;
        if (source_index < 0 || static_cast<cgltf_size>(source_index) >= data->images_count)
            continue;

        return &data->images[source_index];
    }
    return tex.image;
}

TextureLookup ExtractTextures(Scene& scene, const cgltf_data* data,
                              std::string_view base_dir) {
    TextureLookup lookup;
    // Cache decoded pixel data so images shared by multiple glTF textures
    // are only decoded once.
    std::unordered_map<const cgltf_image*, TextureDesc> decoded;

    for (cgltf_size i = 0; i < data->textures_count; ++i) {
        const cgltf_texture& tex = data->textures[i];

        // Prefer DDS source from MSFT_texture_dds extension when available
        const cgltf_image* image = ResolveDdsImage(tex, data);
        if (!image) continue;

        // Decode pixel data on first encounter, cache for reuse
        auto [cache_it, inserted] = decoded.try_emplace(image, TextureDesc{});
        if (inserted) {
            auto result = DecodeImage(*image, base_dir);
            if (!result) {
                decoded.erase(cache_it);
                continue;
            }
            cache_it->second = std::move(*result);
        }

        // Copy the base desc, then apply this texture's sampler settings
        TextureDesc desc = cache_it->second;
        if (tex.sampler) {
            desc.wrap_s     = MapWrap(tex.sampler->wrap_s);
            desc.wrap_t     = MapWrap(tex.sampler->wrap_t);
            desc.mag_filter = MapFilter(tex.sampler->mag_filter);
            desc.min_filter = MapFilter(tex.sampler->min_filter);
        }

        lookup[i] = scene.AddTexture(std::move(desc));
    }

    return lookup;
}

// ── Resolve texture reference from a cgltf_texture_view ──────────────────

std::optional<TextureId> ResolveTexture(const cgltf_texture_view& view,
                                        const cgltf_data* data,
                                        const TextureLookup& lookup) {
    if (!view.texture) return std::nullopt;
    auto idx = static_cast<cgltf_size>(view.texture - data->textures);
    auto it = lookup.find(idx);
    if (it != lookup.end()) return it->second;
    return std::nullopt;
}

// Check if a texture view requires an unsupported texCoord set.
// Returns true if the view needs texCoord != 0.
bool RequiresUnsupportedTexCoord(const cgltf_texture_view& view) {
    if (view.texcoord != 0) return true;
    if (view.has_transform && view.transform.has_texcoord && view.transform.texcoord != 0)
        return true;
    return false;
}

// UV transform: offset, rotation, scale (identity = {0,0}, 0, {1,1})
struct UvTransform {
    float offset[2] = {0.0f, 0.0f};
    float rotation   = 0.0f;
    float scale[2]   = {1.0f, 1.0f};

    bool IsIdentity() const {
        return offset[0] == 0.0f && offset[1] == 0.0f
            && rotation == 0.0f
            && scale[0] == 1.0f && scale[1] == 1.0f;
    }

    bool operator==(const UvTransform& o) const {
        return offset[0] == o.offset[0] && offset[1] == o.offset[1]
            && rotation == o.rotation
            && scale[0] == o.scale[0] && scale[1] == o.scale[1];
    }
};

UvTransform ExtractUvTransform(const cgltf_texture_view& view) {
    UvTransform t;
    if (!view.has_transform) return t;
    t.offset[0] = view.transform.offset[0];
    t.offset[1] = view.transform.offset[1];
    t.rotation  = view.transform.rotation;
    t.scale[0]  = view.transform.scale[0];
    t.scale[1]  = view.transform.scale[1];
    return t;
}

// ── Material extraction ──────────────────────────────────────────────────

using MaterialLookup = std::unordered_map<const cgltf_material*, MaterialId>;

MaterialLookup ExtractMaterials(Scene& scene, const cgltf_data* data,
                                const TextureLookup& tex_lookup) {
    MaterialLookup mat_lookup;

    for (cgltf_size i = 0; i < data->materials_count; ++i) {
        const cgltf_material& gmat = data->materials[i];
        MaterialDesc desc;
        desc.name = gmat.name ? gmat.name : "";

        // PBR metallic-roughness
        if (gmat.has_pbr_metallic_roughness) {
            const auto& pbr = gmat.pbr_metallic_roughness;
            desc.base_color = {pbr.base_color_factor[0],
                               pbr.base_color_factor[1],
                               pbr.base_color_factor[2]};
            desc.opacity    = pbr.base_color_factor[3];
            desc.roughness  = pbr.roughness_factor;
            desc.metallic   = pbr.metallic_factor;

            desc.base_color_map =
                ResolveTexture(pbr.base_color_texture, data, tex_lookup);
            desc.metallic_roughness_map =
                ResolveTexture(pbr.metallic_roughness_texture, data, tex_lookup);
        }

        // Normal map
        desc.normal_map = ResolveTexture(gmat.normal_texture, data, tex_lookup);
        if (desc.normal_map)
            desc.normal_scale = gmat.normal_texture.scale;

        // Emissive
        desc.emissive_factor = {gmat.emissive_factor[0],
                                gmat.emissive_factor[1],
                                gmat.emissive_factor[2]};
        desc.emissive_map = ResolveTexture(gmat.emissive_texture, data, tex_lookup);

        // KHR_materials_emissive_strength
        if (gmat.has_emissive_strength)
            desc.emissive_strength = gmat.emissive_strength.emissive_strength;

        // Alpha
        switch (gmat.alpha_mode) {
        case cgltf_alpha_mode_mask:
            desc.alpha_mode = MaterialDesc::AlphaMode::kMask;
            break;
        case cgltf_alpha_mode_blend:
            desc.alpha_mode = MaterialDesc::AlphaMode::kBlend;
            break;
        default:
            desc.alpha_mode = MaterialDesc::AlphaMode::kOpaque;
            break;
        }
        desc.alpha_cutoff  = gmat.alpha_cutoff;
        desc.double_sided  = gmat.double_sided;

        // KHR_materials_clearcoat
        if (gmat.has_clearcoat) {
            desc.clear_coat           = gmat.clearcoat.clearcoat_factor;
            desc.clear_coat_roughness = gmat.clearcoat.clearcoat_roughness_factor;
        }

        // KHR_materials_sheen
        if (gmat.has_sheen) {
            desc.sheen_color = {gmat.sheen.sheen_color_factor[0],
                                gmat.sheen.sheen_color_factor[1],
                                gmat.sheen.sheen_color_factor[2]};
            desc.sheen_roughness = gmat.sheen.sheen_roughness_factor;
            desc.sheen_color_map =
                ResolveTexture(gmat.sheen.sheen_color_texture, data, tex_lookup);
            desc.sheen_roughness_map =
                ResolveTexture(gmat.sheen.sheen_roughness_texture, data, tex_lookup);
        }

        // KHR_materials_transmission
        if (gmat.has_transmission) {
            desc.transmission_factor = gmat.transmission.transmission_factor;
            desc.transmission_map =
                ResolveTexture(gmat.transmission.transmission_texture, data, tex_lookup);
        }

        // KHR_materials_volume
        if (gmat.has_volume) {
            desc.attenuation_color = {gmat.volume.attenuation_color[0],
                                      gmat.volume.attenuation_color[1],
                                      gmat.volume.attenuation_color[2]};
            desc.attenuation_distance = gmat.volume.attenuation_distance;

            // Thickness textures are a rasterizer approximation; this path tracer
            // uses actual ray-traced distances through manifold volumes instead.
            if (gmat.volume.thickness_texture.texture)
                std::fprintf(stderr,
                    "GltfLoader: ignoring thickness texture on material '%s' "
                    "(ray tracer uses ray-traced distances; manifold mesh required)\n",
                    gmat.name ? gmat.name : "<unnamed>");
        }

        // KHR_materials_ior
        if (gmat.has_ior)
            desc.ior = gmat.ior.ior;

        // KHR_materials_diffuse_transmission (not natively supported by cgltf v1.14)
        for (cgltf_size ei = 0; ei < gmat.extensions_count; ++ei) {
            const auto& ext = gmat.extensions[ei];
            if (!ext.name || !ext.data) continue;
            if (std::string_view(ext.name) != "KHR_materials_diffuse_transmission") continue;

            std::string_view json(ext.data);

            // Parse diffuseTransmissionFactor (float)
            auto parse_float = [&](std::string_view key) -> std::optional<float> {
                auto pos = json.find(key);
                if (pos == std::string_view::npos) return std::nullopt;
                pos += key.size();
                // Skip to colon, then whitespace
                pos = json.find(':', pos);
                if (pos == std::string_view::npos) return std::nullopt;
                ++pos;
                while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
                char* end = nullptr;
                float val = std::strtof(json.data() + pos, &end);
                if (end == json.data() + pos) return std::nullopt;
                return val;
            };

            // Parse a float[3] array value
            auto parse_vec3 = [&](std::string_view key) -> std::optional<glm::vec3> {
                auto pos = json.find(key);
                if (pos == std::string_view::npos) return std::nullopt;
                pos = json.find('[', pos);
                if (pos == std::string_view::npos) return std::nullopt;
                ++pos;
                glm::vec3 v{};
                for (int c = 0; c < 3; ++c) {
                    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ',')) ++pos;
                    char* end = nullptr;
                    v[c] = std::strtof(json.data() + pos, &end);
                    if (end == json.data() + pos) return std::nullopt;
                    pos = static_cast<size_t>(end - json.data());
                }
                return v;
            };

            if (auto f = parse_float("diffuseTransmissionFactor"))
                desc.diffuse_transmission_factor = *f;

            if (auto c = parse_vec3("diffuseTransmissionColorFactor"))
                desc.diffuse_transmission_color = *c;

            // diffuseTransmissionTexture deferred (texture not parsed in v1)
            if (json.find("diffuseTransmissionTexture") != std::string_view::npos)
                std::fprintf(stderr, "GltfLoader: diffuseTransmissionTexture not yet supported\n");

            // glTF diffuse transmission implies thin-surface geometry
            desc.thin_surface = true;
            break;
        }

        // KHR_texture_transform: validate texCoord and extract UV transform
        // Collect all texture views that reference a texture
        const cgltf_texture_view* tex_views[] = {
            gmat.has_pbr_metallic_roughness ? &gmat.pbr_metallic_roughness.base_color_texture : nullptr,
            gmat.has_pbr_metallic_roughness ? &gmat.pbr_metallic_roughness.metallic_roughness_texture : nullptr,
            &gmat.normal_texture,
            &gmat.emissive_texture,
            gmat.has_transmission ? &gmat.transmission.transmission_texture : nullptr,
        };
        const char* tex_view_names[] = {
            "baseColorTexture", "metallicRoughnessTexture",
            "normalTexture", "emissiveTexture", "transmissionTexture",
        };

        bool texcoord_error = false;
        UvTransform first_transform;
        bool found_first = false;

        for (int ti = 0; ti < 5; ++ti) {
            if (!tex_views[ti] || !tex_views[ti]->texture) continue;

            if (RequiresUnsupportedTexCoord(*tex_views[ti])) {
                std::fprintf(stderr, "GltfLoader: material '%s' %s requires texCoord != 0 (unsupported)\n",
                             desc.name.c_str(), tex_view_names[ti]);
                texcoord_error = true;
                break;
            }

            auto t = ExtractUvTransform(*tex_views[ti]);
            if (t.IsIdentity()) continue;

            if (!found_first) {
                first_transform = t;
                found_first = true;
            } else if (!(t == first_transform)) {
                std::fprintf(stderr, "GltfLoader: material '%s' %s has different UV transform than first slot (using first)\n",
                             desc.name.c_str(), tex_view_names[ti]);
            }
        }

        if (texcoord_error) {
            // Fall back to default material for this glTF material
            auto mat_id = scene.AddMaterial(MaterialDesc{.name = desc.name + " (fallback)"});
            mat_lookup[&gmat] = mat_id;
            continue;
        }

        if (found_first) {
            desc.uv_offset   = {first_transform.offset[0], first_transform.offset[1]};
            desc.uv_scale    = {first_transform.scale[0], first_transform.scale[1]};
            desc.uv_rotation = first_transform.rotation;
        }

        auto mat_id = scene.AddMaterial(std::move(desc));
        mat_lookup[&gmat] = mat_id;
    }

    return mat_lookup;
}

// ── Default material (for primitives with no material) ───────────────────

MaterialId GetOrCreateDefaultMaterial(Scene& scene, MaterialId& cached) {
    if (cached) return cached;
    MaterialDesc desc;
    desc.name       = "default";
    desc.base_color = {1.0f, 1.0f, 1.0f};
    desc.roughness  = 0.5f;
    desc.metallic   = 0.0f;
    cached = scene.AddMaterial(std::move(desc));
    return cached;
}

// ── Decompose mat4 → Transform ──────────────────────────────────────────

Transform DecomposeMatrix(const glm::mat4& m) {
    Transform t;
    glm::vec3 skew;
    glm::vec4 perspective;
    if (glm::decompose(m, t.scale, t.rotation, t.translation, skew, perspective))
        return t;

    // glm::decompose fails for matrices with very small scale (determinant
    // below float epsilon).  Fall back to manual extraction.
    t.translation = glm::vec3(m[3]);

    glm::vec3 col0(m[0]);
    glm::vec3 col1(m[1]);
    glm::vec3 col2(m[2]);

    t.scale.x = glm::length(col0);
    t.scale.y = glm::length(col1);
    t.scale.z = glm::length(col2);

    // Correct for negative determinant (reflection)
    if (glm::dot(glm::cross(col0, col1), col2) < 0.0f) {
        t.scale.x = -t.scale.x;
        col0 = -col0;
    }

    // Normalize columns to extract rotation
    if (t.scale.x != 0.0f) col0 /= t.scale.x;
    if (t.scale.y != 0.0f) col1 /= t.scale.y;
    if (t.scale.z != 0.0f) col2 /= t.scale.z;

    glm::mat3 rot(col0, col1, col2);
    t.rotation = glm::quat_cast(rot);

    return t;
}

// ── Node hierarchy walk ──────────────────────────────────────────────────

void ProcessNode(const cgltf_node* node, const glm::mat4& parent_world,
                 const cgltf_data* data,
                 Scene& scene,
                 const MaterialLookup& mat_lookup,
                 MaterialId& default_material,
                 const LoadOptions& options,
                 LoadResult& result) {
    // Compute this node's world transform
    float local_matrix[16];
    cgltf_node_transform_local(node, local_matrix);
    glm::mat4 local = glm::make_mat4(local_matrix);
    glm::mat4 world = parent_world * local;

    // Process mesh primitives
    if (node->mesh) {
        const cgltf_mesh& gltf_mesh = *node->mesh;

        for (cgltf_size pi = 0; pi < gltf_mesh.primitives_count; ++pi) {
            const cgltf_primitive& prim = gltf_mesh.primitives[pi];
            if (prim.type != cgltf_primitive_type_triangles) continue;

            // Find accessors
            const cgltf_accessor* pos_accessor = nullptr;
            const cgltf_accessor* norm_accessor = nullptr;
            const cgltf_accessor* tangent_accessor = nullptr;
            const cgltf_accessor* uv0_accessor = nullptr;
            const cgltf_accessor* uv1_accessor = nullptr;

            for (cgltf_size ai = 0; ai < prim.attributes_count; ++ai) {
                const auto& attr = prim.attributes[ai];
                switch (attr.type) {
                case cgltf_attribute_type_position:  pos_accessor = attr.data; break;
                case cgltf_attribute_type_normal:    norm_accessor = attr.data; break;
                case cgltf_attribute_type_tangent:   tangent_accessor = attr.data; break;
                case cgltf_attribute_type_texcoord:
                    if (attr.index == 0) uv0_accessor = attr.data;
                    else if (attr.index == 1) uv1_accessor = attr.data;
                    break;
                default: break;
                }
            }

            if (!pos_accessor) continue;

            auto vertex_count = pos_accessor->count;

            // Read vertices
            std::vector<Vertex> vertices(vertex_count);
            glm::vec3 bbox_min(std::numeric_limits<float>::max());
            glm::vec3 bbox_max(std::numeric_limits<float>::lowest());

            for (cgltf_size vi = 0; vi < vertex_count; ++vi) {
                auto& v = vertices[vi];

                v.position = ReadVec3(pos_accessor, vi);
                bbox_min = glm::min(bbox_min, v.position);
                bbox_max = glm::max(bbox_max, v.position);

                if (norm_accessor)
                    v.normal = ReadVec3(norm_accessor, vi);

                if (tangent_accessor)
                    v.tangent = ReadVec4(tangent_accessor, vi);

                v.tex_coord_0 = uv0_accessor ? ReadVec2(uv0_accessor, vi)
                                             : glm::vec2(0.0f);
                v.tex_coord_1 = uv1_accessor ? ReadVec2(uv1_accessor, vi)
                                             : glm::vec2(0.0f);
            }

            // Read indices
            std::vector<uint32_t> indices;
            if (prim.indices) {
                indices.resize(prim.indices->count);
                for (cgltf_size ii = 0; ii < prim.indices->count; ++ii)
                    indices[ii] = static_cast<uint32_t>(
                        cgltf_accessor_read_index(prim.indices, ii));
            } else {
                indices.resize(vertex_count);
                for (cgltf_size ii = 0; ii < vertex_count; ++ii)
                    indices[ii] = static_cast<uint32_t>(ii);
            }

            // Generate missing normals
            if (!norm_accessor && options.generate_missing_normals)
                GenerateFaceWeightedNormals(vertices, indices);

            // Generate missing tangents
            if (!tangent_accessor && options.generate_missing_tangents)
                GenerateMikkTangents(vertices, indices);

            // Resolve material
            MaterialId mat_id;
            if (prim.material) {
                auto it = mat_lookup.find(prim.material);
                if (it != mat_lookup.end())
                    mat_id = it->second;
                else
                    mat_id = GetOrCreateDefaultMaterial(scene, default_material);
            } else {
                mat_id = GetOrCreateDefaultMaterial(scene, default_material);
            }

            // Build mesh name
            std::string mesh_name;
            if (node->name) mesh_name = node->name;
            if (gltf_mesh.primitives_count > 1)
                mesh_name += "_prim" + std::to_string(pi);

            // Add mesh to scene
            Mesh mesh_meta;
            mesh_meta.name         = mesh_name;
            mesh_meta.vertex_count = static_cast<uint32_t>(vertices.size());
            mesh_meta.index_count  = static_cast<uint32_t>(indices.size());
            mesh_meta.bbox_min     = bbox_min;
            mesh_meta.bbox_max     = bbox_max;

            auto mesh_id = scene.AddMesh(std::move(mesh_meta));

            // Add node with world transform
            auto node_id = scene.AddNode(mesh_id, mat_id, mesh_name);
            auto* scene_node = scene.GetNode(node_id);
            scene_node->transform = DecomposeMatrix(world);

            // Record mesh data and node ID
            MeshData md;
            md.mesh_id  = mesh_id;
            md.vertices = std::move(vertices);
            md.indices  = std::move(indices);
            result.mesh_data.push_back(std::move(md));
            result.nodes.push_back(node_id);
        }
    }

    // Recurse into children
    for (cgltf_size ci = 0; ci < node->children_count; ++ci)
        ProcessNode(node->children[ci], world, data, scene, mat_lookup,
                    default_material, options, result);
}

} // namespace

// ── Public API ───────────────────────────────────────────────────────────

LoadResult LoadGltf(Scene& scene, const std::string& file_path,
                    const LoadOptions& options) {
    LoadResult result;

    cgltf_options cgltf_opts{};
    cgltf_data* data = nullptr;

    cgltf_result parse_result = cgltf_parse_file(&cgltf_opts, file_path.c_str(), &data);
    if (parse_result != cgltf_result_success) {
        result.error_message = "Failed to parse glTF file: " + file_path;
        return result;
    }

    cgltf_result buf_result = cgltf_load_buffers(&cgltf_opts, data, file_path.c_str());
    if (buf_result != cgltf_result_success) {
        result.error_message = "Failed to load glTF buffers: " + file_path;
        cgltf_free(data);
        return result;
    }

    cgltf_result validate_result = cgltf_validate(data);
    if (validate_result != cgltf_result_success) {
        result.error_message = "glTF validation failed: " + file_path;
        cgltf_free(data);
        return result;
    }

    // Extract textures, then materials (materials reference textures)
    // Compute base directory for resolving URI-referenced images
    std::string base_dir;
    {
        auto last_sep = file_path.find_last_of("/\\");
        if (last_sep != std::string::npos)
            base_dir = file_path.substr(0, last_sep + 1);
    }
    auto tex_lookup = ExtractTextures(scene, data, base_dir);
    auto mat_lookup = ExtractMaterials(scene, data, tex_lookup);

    // Mark sRGB textures.  Per glTF 2.0 spec, base color, emissive, and
    // sheen color textures are authored in sRGB color space.
    for (const auto& mat : scene.Materials()) {
        for (auto id : {mat.base_color_map, mat.emissive_map,
                        mat.sheen_color_map}) {
            if (!id) continue;
            if (auto* tex = scene.GetTexture(*id))
                tex->srgb = true;
        }
    }

    // Walk node hierarchy
    MaterialId default_material;  // Created on demand
    cgltf_size scene_idx = 0;     // Use the default scene (or scene 0)
    if (data->scene)
        scene_idx = static_cast<cgltf_size>(data->scene - data->scenes);

    if (data->scenes_count > 0) {
        const cgltf_scene& gltf_scene = data->scenes[scene_idx];
        glm::mat4 identity(1.0f);
        for (cgltf_size ni = 0; ni < gltf_scene.nodes_count; ++ni)
            ProcessNode(gltf_scene.nodes[ni], identity, data, scene, mat_lookup,
                        default_material, options, result);
    }

    cgltf_free(data);

    result.success = true;
    return result;
}

} // namespace monti::gltf
