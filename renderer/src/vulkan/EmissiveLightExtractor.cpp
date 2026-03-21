#include "EmissiveLightExtractor.h"

#include <algorithm>
#include <cstdio>
#include <unordered_map>

#include <glm/glm.hpp>

namespace monti::vulkan {

uint32_t ExtractEmissiveLights(Scene& scene, std::span<const MeshData> mesh_data) {
    const auto& nodes = scene.Nodes();

    // Build a map from MeshId → MeshData for fast lookup
    std::unordered_map<MeshId, const MeshData*> mesh_map;
    for (const auto& md : mesh_data)
        mesh_map[md.mesh_id] = &md;

    uint32_t extracted = 0;

    for (const auto& node : nodes) {
        if (!node.visible) continue;

        const auto* mat = scene.GetMaterial(node.material_id);
        if (!mat) continue;

        // Compute emissive radiance for this material
        glm::vec3 radiance = mat->emissive_factor * mat->emissive_strength;
        float max_component = std::max({radiance.r, radiance.g, radiance.b});
        if (max_component < kMinEmissiveLuminance) continue;

        // Find corresponding mesh data
        auto it = mesh_map.find(node.mesh_id);
        if (it == mesh_map.end()) continue;
        const auto& md = *it->second;

        // Get world-space transform matrix
        glm::mat4 world = node.transform.ToMatrix();

        // Resolve triangle count: indexed meshes use md.indices, non-indexed
        // treat every 3 consecutive vertices as a triangle.
        bool indexed = !md.indices.empty();
        uint32_t tri_count = indexed
            ? static_cast<uint32_t>(md.indices.size()) / 3
            : static_cast<uint32_t>(md.vertices.size()) / 3;

        for (uint32_t t = 0; t < tri_count; ++t) {
            uint32_t i0 = indexed ? md.indices[t * 3 + 0] : t * 3 + 0;
            uint32_t i1 = indexed ? md.indices[t * 3 + 1] : t * 3 + 1;
            uint32_t i2 = indexed ? md.indices[t * 3 + 2] : t * 3 + 2;

            auto v0_world = glm::vec3(world * glm::vec4(md.vertices[i0].position, 1.0f));
            auto v1_world = glm::vec3(world * glm::vec4(md.vertices[i1].position, 1.0f));
            auto v2_world = glm::vec3(world * glm::vec4(md.vertices[i2].position, 1.0f));

            TriangleLight light{};
            light.v0 = v0_world;
            light.v1 = v1_world;
            light.v2 = v2_world;
            light.radiance = radiance;
            light.two_sided = mat->double_sided;

            scene.AddTriangleLight(light);
            ++extracted;
        }
    }

    if (extracted > 0) {
        std::printf("EmissiveLightExtractor: extracted %u triangle lights from emissive meshes\n",
                    extracted);
    }

    return extracted;
}

}  // namespace monti::vulkan
