#pragma once
#include <monti/scene/Scene.h>
#include <monti/scene/Material.h>
#include <string>
#include <vector>

namespace monti::gltf {

struct LoadResult {
    bool success = false;
    std::string error_message;
    std::vector<NodeId> nodes;
    std::vector<MeshData> mesh_data;
};

struct LoadOptions {
    bool generate_missing_normals  = true;
    bool generate_missing_tangents = true;
};

LoadResult LoadGltf(Scene& scene,
                    const std::string& file_path,
                    const LoadOptions& options = {});

} // namespace monti::gltf
