#pragma once
#include <compare>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace monti {

template <typename Tag>
struct TypedId {
    static constexpr uint64_t kInvalid = UINT64_MAX;
    uint64_t value = kInvalid;
    bool operator==(const TypedId&) const = default;
    auto operator<=>(const TypedId&) const = default;
    explicit operator bool() const { return value != kInvalid; }
};

struct MeshTag {};
struct MaterialTag {};
struct TextureTag {};
struct NodeTag {};

using MeshId     = TypedId<MeshTag>;
using MaterialId = TypedId<MaterialTag>;
using TextureId  = TypedId<TextureTag>;
using NodeId     = TypedId<NodeTag>;

} // namespace monti

template <typename Tag>
struct std::hash<monti::TypedId<Tag>> {
    size_t operator()(const monti::TypedId<Tag>& id) const noexcept {
        return std::hash<uint64_t>{}(id.value);
    }
};

namespace monti {

struct Transform {
    glm::vec3 translation = {0, 0, 0};
    glm::quat rotation    = {1, 0, 0, 0};
    glm::vec3 scale       = {1, 1, 1};
    glm::mat4 ToMatrix() const;
};

// Fixed vertex layout for glTF PBR path tracing.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 tangent;    // xyz = direction, w = bitangent sign
    glm::vec2 tex_coord_0;
    glm::vec2 tex_coord_1;
};

// Pixel formats for texture descriptions.
enum class PixelFormat {
    kRGBA16F,
    kRGBA32F,
    kRG16F,
    kRGBA8_UNORM,
    kRG16_SNORM,
    kR32F,
    kR8_UNORM,
};

// Texture sampler wrap mode (matches glTF 2.0 / Vulkan conventions).
enum class SamplerWrap {
    kRepeat,
    kClampToEdge,
    kMirroredRepeat,
};

// Texture sampler filter mode.
enum class SamplerFilter {
    kLinear,
    kNearest,
};

} // namespace monti
