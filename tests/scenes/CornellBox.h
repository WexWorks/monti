#pragma once

#include <monti/scene/Scene.h>
#include <monti/scene/Material.h>

#include <vector>

namespace monti::test {

struct CornellBoxResult {
    Scene scene;
    std::vector<MeshData> mesh_data;
};

// Build the classic Cornell box via the Scene API using unit scale
// (room from 0 to 1 on all axes).
// 7 meshes: floor, ceiling, back wall, left wall, right wall, short box, tall box
// 4 materials: white diffuse, red diffuse, green diffuse, light emissive
// No lights included — callers add lights explicitly.
// 1 camera at canonical viewpoint
CornellBoxResult BuildCornellBox();

// Add the canonical ceiling area light used by most Cornell box tests.
// Area light geometry is synthesized automatically by RenderSceneMultiFrame.
inline void AddCornellBoxLight(Scene& scene) {
    AreaLight ceiling_light;
    ceiling_light.corner = {0.35f, 0.999f, 0.35f};
    ceiling_light.edge_a = {0.3f, 0.0f, 0.0f};
    ceiling_light.edge_b = {0.0f, 0.0f, 0.3f};
    ceiling_light.radiance = {17.0f, 12.0f, 4.0f};
    ceiling_light.two_sided = false;
    scene.AddAreaLight(ceiling_light);
}

} // namespace monti::test
