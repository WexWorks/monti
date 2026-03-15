#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <glm/glm.hpp>

#include <monti/scene/Camera.h>
#include <monti/scene/Scene.h>

namespace monti::app {

constexpr float kDefaultFovDegrees = 60.0f;
constexpr float kCameraFitPadding = 1.1f;
constexpr float kMinCameraDistance = 0.1f;
constexpr float kDefaultNearPlane = 0.01f;
constexpr float kDefaultFarPlane = 10000.0f;
constexpr float kMinSceneDiagonal = 0.01f;
constexpr float kFallbackSceneDiagonal = 10.0f;

struct SceneAABB {
    glm::vec3 min{std::numeric_limits<float>::max()};
    glm::vec3 max{std::numeric_limits<float>::lowest()};

    glm::vec3 Center() const { return (min + max) * 0.5f; }
    float Diagonal() const { return glm::length(max - min); }
};

inline float ClampSceneDiagonal(float diagonal) {
    return (diagonal < kMinSceneDiagonal) ? kFallbackSceneDiagonal : diagonal;
}

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

// Compute a camera that fits the given AABB in view.
// Camera is positioned on the +Z axis looking at the AABB center.
inline monti::CameraParams AutoFitCamera(const SceneAABB& aabb) {
    glm::vec3 center = aabb.Center();
    float half_diagonal = aabb.Diagonal() * 0.5f;

    float fov_radians = glm::radians(kDefaultFovDegrees);
    float distance = (half_diagonal / std::tan(fov_radians * 0.5f)) * kCameraFitPadding;
    distance = std::max(distance, kMinCameraDistance);

    monti::CameraParams cam{};
    cam.position = center + glm::vec3(0.0f, 0.0f, distance);
    cam.target = center;
    cam.up = {0.0f, 1.0f, 0.0f};
    cam.vertical_fov_radians = fov_radians;
    cam.near_plane = kDefaultNearPlane;
    cam.far_plane = kDefaultFarPlane;
    return cam;
}

// Convenience wrapper: compute AABB then auto-fit camera.
inline monti::CameraParams ComputeDefaultCamera(const monti::Scene& scene) {
    auto aabb = ComputeSceneAABB(scene);
    return AutoFitCamera(aabb);
}

}  // namespace monti::app
