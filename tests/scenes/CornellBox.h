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

// Build a flat emissive scene for zero-variance convergence tests.
// The scene contains a single large emissive quad that fills the camera FOV.
// Every primary ray hits the emissive surface, returning a constant luminance
// with zero variance — ideal for verifying that the convergence mechanism
// correctly detects a converged state within min_convergence_frames.
//
// Camera setup matches the Cornell Box canonical viewpoint so GBufferImages
// and accumulator dimensions can be shared with other tests.
inline CornellBoxResult BuildFlatEmissiveScene() {
    CornellBoxResult result;
    Scene& scene = result.scene;

    // Emissive material: constant luminance regardless of ray direction.
    MaterialDesc emissive_mat;
    emissive_mat.emissive_factor   = {1.0f, 1.0f, 1.0f};
    emissive_mat.emissive_strength = 4.0f;  // log(4) ≈ 1.39 — good numerical range
    emissive_mat.double_sided      = true;
    const MaterialId mat_id = scene.AddMaterial(emissive_mat, "emissive_flat");

    // A large quad at z=0.0 facing the camera (at z=1.35).
    // The quad covers [0,1]×[0,1] in XY so it overfills any reasonable FOV.
    // Normal points toward +z (toward camera).
    MeshData quad;
    quad.vertices = {
        // position           normal         tangent               uv0          uv1
        {{0.0f, 0.0f, 0.0f}, {0,0,1}, {1,0,0,1}, {0,0}, {0,0}},
        {{1.0f, 0.0f, 0.0f}, {0,0,1}, {1,0,0,1}, {1,0}, {1,0}},
        {{1.0f, 1.0f, 0.0f}, {0,0,1}, {1,0,0,1}, {1,1}, {1,1}},
        {{0.0f, 1.0f, 0.0f}, {0,0,1}, {1,0,0,1}, {0,1}, {0,1}},
    };
    quad.indices = {0, 1, 2, 0, 2, 3};

    Mesh mesh_meta;
    mesh_meta.vertex_count  = static_cast<uint32_t>(quad.vertices.size());
    mesh_meta.index_count   = static_cast<uint32_t>(quad.indices.size());
    mesh_meta.vertex_stride = sizeof(Vertex);
    mesh_meta.bbox_min      = {0.0f, 0.0f, 0.0f};
    mesh_meta.bbox_max      = {1.0f, 1.0f, 0.0f};

    const MeshId mesh_id = scene.AddMesh(mesh_meta, "emissive_quad");
    scene.AddNode(mesh_id, mat_id, "emissive_quad_node");

    quad.mesh_id = mesh_id;
    result.mesh_data.push_back(std::move(quad));

    // Reuse canonical Cornell Box camera (z=1.35, looking toward -z).
    CameraParams cam;
    cam.position = {0.5f, 0.5f, 1.35f};
    cam.target   = {0.5f, 0.5f, 0.0f};
    cam.up       = {0.0f, 1.0f, 0.0f};
    cam.vertical_fov_radians = 0.6283f;  // ~36 degrees in radians
    scene.SetActiveCamera(cam);

    return result;
}

} // namespace monti::test
