#include "Primitives.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace monti::test {

namespace {

struct EdgeKey {
    uint32_t a, b;
    bool operator==(const EdgeKey&) const = default;
};

struct EdgeKeyHash {
    size_t operator()(const EdgeKey& k) const noexcept {
        return std::hash<uint64_t>{}(
            (static_cast<uint64_t>(k.a) << 32) | k.b);
    }
};

uint32_t GetMidpoint(std::vector<glm::vec3>& positions,
                     std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash>& cache,
                     uint32_t i0, uint32_t i1) {
    auto lo = std::min(i0, i1);
    auto hi = std::max(i0, i1);
    EdgeKey key{lo, hi};

    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    auto mid = glm::normalize(positions[i0] + positions[i1]);
    auto idx = static_cast<uint32_t>(positions.size());
    positions.push_back(mid);
    cache[key] = idx;
    return idx;
}

} // anonymous namespace

MeshData MakeIcosphere(const glm::vec3& center, float radius,
                       uint32_t subdivisions) {
    // Golden ratio
    constexpr float t = 1.6180339887498948482f;
    float len = std::sqrt(1.0f + t * t);
    float a = 1.0f / len;
    float b = t / len;

    // 12 vertices of a unit icosahedron
    std::vector<glm::vec3> positions = {
        {-a,  b,  0}, { a,  b,  0}, {-a, -b,  0}, { a, -b,  0},
        { 0, -a,  b}, { 0,  a,  b}, { 0, -a, -b}, { 0,  a, -b},
        { b,  0, -a}, { b,  0,  a}, {-b,  0, -a}, {-b,  0,  a},
    };

    // 20 triangles of icosahedron
    std::vector<uint32_t> indices = {
         0, 11,  5,    0,  5,  1,    0,  1,  7,    0,  7, 10,    0, 10, 11,
         1,  5,  9,    5, 11,  4,   11, 10,  2,   10,  7,  6,    7,  1,  8,
         3,  9,  4,    3,  4,  2,    3,  2,  6,    3,  6,  8,    3,  8,  9,
         4,  9,  5,    2,  4, 11,    6,  2, 10,    8,  6,  7,    9,  8,  1,
    };

    // Subdivide
    std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash> midpoint_cache;
    for (uint32_t sub = 0; sub < subdivisions; ++sub) {
        midpoint_cache.clear();
        std::vector<uint32_t> new_indices;
        new_indices.reserve(indices.size() * 4);

        for (size_t i = 0; i < indices.size(); i += 3) {
            uint32_t v0 = indices[i + 0];
            uint32_t v1 = indices[i + 1];
            uint32_t v2 = indices[i + 2];

            uint32_t m01 = GetMidpoint(positions, midpoint_cache, v0, v1);
            uint32_t m12 = GetMidpoint(positions, midpoint_cache, v1, v2);
            uint32_t m20 = GetMidpoint(positions, midpoint_cache, v2, v0);

            new_indices.insert(new_indices.end(), {v0, m01, m20});
            new_indices.insert(new_indices.end(), {v1, m12, m01});
            new_indices.insert(new_indices.end(), {v2, m20, m12});
            new_indices.insert(new_indices.end(), {m01, m12, m20});
        }

        indices = std::move(new_indices);
    }

    // Build final MeshData with position, normal, tangent, UV
    MeshData result;
    result.vertices.reserve(positions.size());
    for (const auto& p : positions) {
        glm::vec3 n = glm::normalize(p);
        glm::vec3 world_pos = center + n * radius;

        // Tangent from spherical coordinates
        glm::vec3 up = std::abs(n.y) < 0.999f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
        glm::vec3 tangent_dir = glm::normalize(glm::cross(up, n));

        // UV from spherical mapping
        float u = 0.5f + std::atan2(n.z, n.x) / (2.0f * glm::pi<float>());
        float v = 0.5f - std::asin(std::clamp(n.y, -1.0f, 1.0f)) / glm::pi<float>();

        Vertex vert{};
        vert.position = world_pos;
        vert.normal = n;
        vert.tangent = glm::vec4(tangent_dir, 1.0f);
        vert.tex_coord_0 = {u, v};
        vert.tex_coord_1 = {u, v};
        result.vertices.push_back(vert);
    }
    result.indices = std::move(indices);

    return result;
}

} // namespace monti::test
