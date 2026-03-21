#pragma once

#include <monti/scene/Scene.h>

#include <cstdint>
#include <span>

namespace monti::vulkan {

// Minimum emissive luminance for triangle extraction into the light buffer.
// Triangles below this threshold rely on random path hits only.
constexpr float kMinEmissiveLuminance = 0.01f;

// Scans scene materials for emissive surfaces, reads triangle vertices from
// the provided mesh data, transforms them to world space, and adds them as
// TriangleLights to the scene for NEE (next-event estimation).
//
// Call once at scene load time (or when the scene changes), before rendering.
// The extracted triangle lights are appended to Scene::TriangleLights() and
// included in the light buffer by GpuScene::UpdateLights().
//
// Returns the number of triangle lights extracted.
uint32_t ExtractEmissiveLights(Scene& scene, std::span<const MeshData> mesh_data);

}  // namespace monti::vulkan
