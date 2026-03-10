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
// 1 area light on the ceiling, 1 camera at canonical viewpoint
CornellBoxResult BuildCornellBox();

} // namespace monti::test
