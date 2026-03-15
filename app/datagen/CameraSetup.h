#pragma once

#include <monti/scene/Camera.h>
#include <monti/scene/Scene.h>

#include <array>
#include <limits>

#include <glm/glm.hpp>

namespace monti::app::datagen {

struct SceneAABB {
    glm::vec3 min{std::numeric_limits<float>::max()};
    glm::vec3 max{std::numeric_limits<float>::lowest()};

    glm::vec3 Center() const { return (min + max) * 0.5f; }
    float Diagonal() const { return glm::length(max - min); }
};

inline SceneAABB ComputeSceneAABB(const monti::Scene& scene) {
    SceneAABB aabb;
    for (const auto& node : scene.Nodes()) {
        const auto* mesh = scene.GetMesh(node.mesh_id);
        if (!mesh) continue;
        auto model = node.transform.ToMatrix();
        std::array<glm::vec3, 8> corners = {{
            {mesh->bbox_min.x, mesh->bbox_min.y, mesh->bbox_min.z},
            {mesh->bbox_max.x, mesh->bbox_min.y, mesh->bbox_min.z},
            {mesh->bbox_min.x, mesh->bbox_max.y, mesh->bbox_min.z},
            {mesh->bbox_max.x, mesh->bbox_max.y, mesh->bbox_min.z},
            {mesh->bbox_min.x, mesh->bbox_min.y, mesh->bbox_max.z},
            {mesh->bbox_max.x, mesh->bbox_min.y, mesh->bbox_max.z},
            {mesh->bbox_min.x, mesh->bbox_max.y, mesh->bbox_max.z},
            {mesh->bbox_max.x, mesh->bbox_max.y, mesh->bbox_max.z},
        }};
        for (const auto& c : corners) {
            glm::vec3 world = glm::vec3(model * glm::vec4(c, 1.0f));
            aabb.min = glm::min(aabb.min, world);
            aabb.max = glm::max(aabb.max, world);
        }
    }
    return aabb;
}

// Compute a default camera that fits the scene bounding box in view.
// Camera is positioned on the +Z axis looking at the AABB center.
inline monti::CameraParams ComputeDefaultCamera(const monti::Scene& scene,
                                                 float aspect_ratio) {
    constexpr float kDefaultFovDegrees = 60.0f;

    auto aabb = ComputeSceneAABB(scene);
    glm::vec3 center = aabb.Center();
    float half_diagonal = aabb.Diagonal() * 0.5f;

    float fov_radians = glm::radians(kDefaultFovDegrees);
    float distance = (half_diagonal / std::tan(fov_radians * 0.5f)) * 1.1f;
    distance = std::max(distance, 0.1f);

    monti::CameraParams cam{};
    cam.position = center + glm::vec3(0.0f, 0.0f, distance);
    cam.target = center;
    cam.up = {0.0f, 1.0f, 0.0f};
    cam.vertical_fov_radians = fov_radians;
    cam.aspect_ratio = aspect_ratio;
    cam.near_plane = 0.01f;
    cam.far_plane = 10000.0f;
    return cam;
}

}  // namespace monti::app::datagen
