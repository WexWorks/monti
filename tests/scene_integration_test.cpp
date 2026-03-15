#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "scenes/CornellBox.h"

#include <monti/scene/Scene.h>
#include <monti/scene/Material.h>
#include <monti/scene/Light.h>
#include <monti/scene/Camera.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

using namespace monti;
using namespace monti::test;
using Catch::Matchers::WithinAbs;

// ── Cornell Box Scene Structure ──────────────────────────────────────────

TEST_CASE("Cornell box has expected entity counts", "[scene][cornell]") {
    auto result = BuildCornellBox();
    auto& scene = result.scene;

    REQUIRE(scene.Meshes().size() == 7);       // 5 walls + 2 boxes
    REQUIRE(scene.Materials().size() == 4);     // white, red, green, light
    REQUIRE(scene.Nodes().size() == 7);         // one node per mesh
}

TEST_CASE("Cornell box mesh data matches scene meshes", "[scene][cornell]") {
    auto result = BuildCornellBox();
    auto& scene = result.scene;

    REQUIRE(result.mesh_data.size() == 7);

    for (const auto& md : result.mesh_data) {
        REQUIRE(static_cast<bool>(md.mesh_id));
        REQUIRE_FALSE(md.vertices.empty());
        REQUIRE_FALSE(md.indices.empty());

        // MeshData mesh_id should match a mesh in the scene
        const auto* mesh = scene.GetMesh(md.mesh_id);
        REQUIRE(mesh != nullptr);
        REQUIRE(mesh->vertex_count == static_cast<uint32_t>(md.vertices.size()));
        REQUIRE(mesh->index_count == static_cast<uint32_t>(md.indices.size()));
    }
}

// ── Data Round-Trip Through Accessors ────────────────────────────────────

TEST_CASE("Mesh metadata round-trips through accessors", "[scene][cornell]") {
    auto result = BuildCornellBox();
    auto& scene = result.scene;

    for (const auto& mesh : scene.Meshes()) {
        const auto* looked_up = scene.GetMesh(mesh.id);
        REQUIRE(looked_up != nullptr);
        REQUIRE(looked_up->id == mesh.id);
        REQUIRE(looked_up->name == mesh.name);
        REQUIRE(looked_up->vertex_count == mesh.vertex_count);
        REQUIRE(looked_up->index_count == mesh.index_count);
    }
}

TEST_CASE("Material properties round-trip through accessors", "[scene][cornell]") {
    auto result = BuildCornellBox();
    auto& scene = result.scene;

    // Find the red material
    const MaterialDesc* red = nullptr;
    for (const auto& mat : scene.Materials()) {
        if (mat.name == "red") {
            red = &mat;
            break;
        }
    }
    REQUIRE(red != nullptr);
    REQUIRE_THAT(red->base_color.r, WithinAbs(0.65, 0.001));
    REQUIRE_THAT(red->base_color.g, WithinAbs(0.05, 0.001));
    REQUIRE_THAT(red->base_color.b, WithinAbs(0.05, 0.001));
    REQUIRE_THAT(red->roughness, WithinAbs(1.0, 0.001));
    REQUIRE_THAT(red->metallic, WithinAbs(0.0, 0.001));

    // Verify lookup by ID
    const auto* looked_up = scene.GetMaterial(red->id);
    REQUIRE(looked_up != nullptr);
    REQUIRE(looked_up->id == red->id);
    REQUIRE(looked_up->name == "red");
}

TEST_CASE("Node properties round-trip through accessors", "[scene][cornell]") {
    auto result = BuildCornellBox();
    auto& scene = result.scene;

    for (const auto& node : scene.Nodes()) {
        const auto* looked_up = scene.GetNode(node.id);
        REQUIRE(looked_up != nullptr);
        REQUIRE(looked_up->id == node.id);
        REQUIRE(looked_up->mesh_id == node.mesh_id);
        REQUIRE(looked_up->material_id == node.material_id);
        REQUIRE(looked_up->visible == true);

        // Verify the referenced mesh and material exist
        REQUIRE(scene.GetMesh(node.mesh_id) != nullptr);
        REQUIRE(scene.GetMaterial(node.material_id) != nullptr);
    }
}

// ── Quad Meshes ──────────────────────────────────────────────────────────

TEST_CASE("Wall quads have correct vertex and index counts", "[scene][cornell]") {
    auto result = BuildCornellBox();

    // First 5 mesh_data entries are quads (4 vertices, 6 indices each)
    for (size_t i = 0; i < 5; ++i) {
        REQUIRE(result.mesh_data[i].vertices.size() == 4);
        REQUIRE(result.mesh_data[i].indices.size() == 6);
    }
}

TEST_CASE("Box meshes have correct vertex and index counts", "[scene][cornell]") {
    auto result = BuildCornellBox();

    // Last 2 mesh_data entries are boxes (24 vertices, 36 indices each)
    REQUIRE(result.mesh_data[5].vertices.size() == 24);
    REQUIRE(result.mesh_data[5].indices.size() == 36);
    REQUIRE(result.mesh_data[6].vertices.size() == 24);
    REQUIRE(result.mesh_data[6].indices.size() == 36);
}

// ── Camera ───────────────────────────────────────────────────────────────

TEST_CASE("Camera is set at canonical Cornell box viewpoint", "[scene][cornell]") {
    auto result = BuildCornellBox();
    const auto& camera = result.scene.GetActiveCamera();

    // Camera centered in X and Y, looking into the box along -Z
    REQUIRE_THAT(camera.position.x, WithinAbs(0.5, 0.001));
    REQUIRE_THAT(camera.position.y, WithinAbs(0.5, 0.001));
    REQUIRE(camera.position.z > 1.0f);  // Behind the room opening

    REQUIRE_THAT(camera.target.x, WithinAbs(0.5, 0.001));
    REQUIRE_THAT(camera.target.y, WithinAbs(0.5, 0.001));
    REQUIRE_THAT(camera.target.z, WithinAbs(0.0, 0.001));

    REQUIRE_THAT(camera.up.y, WithinAbs(1.0, 0.001));
    REQUIRE(camera.vertical_fov_radians > 0.0f);
    REQUIRE(camera.vertical_fov_radians < glm::pi<float>());
}

// ── Area Light ───────────────────────────────────────────────────────────

TEST_CASE("Area light is on the ceiling", "[scene][cornell]") {
    auto result = BuildCornellBox();
    const auto& lights = result.scene.AreaLights();

    REQUIRE(lights.size() == 1);
    const auto& light = lights[0];

    // Light is near the ceiling (Y close to 1)
    REQUIRE_THAT(light.corner.y, WithinAbs(0.999, 0.01));

    // Light has positive radiance
    REQUIRE(light.radiance.r > 0.0f);
    REQUIRE(light.radiance.g > 0.0f);
    REQUIRE(light.radiance.b > 0.0f);

    // Edges define a non-degenerate quad
    float area = glm::length(glm::cross(light.edge_a, light.edge_b));
    REQUIRE(area > 0.0f);

    REQUIRE(light.two_sided == false);
}

// ── RemoveNode ───────────────────────────────────────────────────────────

TEST_CASE("RemoveNode removes a node and decrements count", "[scene]") {
    auto result = BuildCornellBox();
    auto& scene = result.scene;

    auto initial_count = scene.Nodes().size();
    REQUIRE(initial_count == 7);

    auto node_id = scene.Nodes().front().id;
    scene.RemoveNode(node_id);

    REQUIRE(scene.Nodes().size() == initial_count - 1);
    REQUIRE(scene.GetNode(node_id) == nullptr);
}

// ── RemoveMesh ───────────────────────────────────────────────────────────

TEST_CASE("RemoveMesh returns false when nodes reference the mesh", "[scene]") {
    auto result = BuildCornellBox();
    auto& scene = result.scene;

    // The first node references a mesh; removing that mesh should fail
    auto mesh_id = scene.Nodes().front().mesh_id;
    REQUIRE_FALSE(scene.RemoveMesh(mesh_id));
    REQUIRE(scene.GetMesh(mesh_id) != nullptr);  // Still present
}

TEST_CASE("RemoveMesh succeeds after removing all referencing nodes", "[scene]") {
    auto result = BuildCornellBox();
    auto& scene = result.scene;

    // Pick the first mesh and find all nodes referencing it
    auto mesh_id = scene.Meshes().front().id;
    auto initial_mesh_count = scene.Meshes().size();

    // Remove all nodes that reference this mesh
    std::vector<NodeId> to_remove;
    for (const auto& node : scene.Nodes()) {
        if (node.mesh_id == mesh_id)
            to_remove.push_back(node.id);
    }
    for (auto nid : to_remove)
        scene.RemoveNode(nid);

    // Now RemoveMesh should succeed
    REQUIRE(scene.RemoveMesh(mesh_id));
    REQUIRE(scene.GetMesh(mesh_id) == nullptr);
    REQUIRE(scene.Meshes().size() == initial_mesh_count - 1);
}

// ── Empty Scene Robustness ───────────────────────────────────────────────

TEST_CASE("Empty scene handles accessors gracefully", "[scene]") {
    Scene scene;

    REQUIRE(scene.Meshes().empty());
    REQUIRE(scene.Materials().empty());
    REQUIRE(scene.Nodes().empty());
    REQUIRE(scene.Textures().empty());
    REQUIRE(scene.AreaLights().empty());
    REQUIRE(scene.GetEnvironmentLight() == nullptr);
    REQUIRE(scene.GetMesh(MeshId{0}) == nullptr);
    REQUIRE(scene.GetMaterial(MaterialId{0}) == nullptr);
    REQUIRE(scene.GetNode(NodeId{0}) == nullptr);
    REQUIRE(scene.GetTexture(TextureId{0}) == nullptr);
}

// ── SetNodeTransform ─────────────────────────────────────────────────────

TEST_CASE("SetNodeTransform saves previous transform", "[scene]") {
    auto result = BuildCornellBox();
    auto& scene = result.scene;

    auto node_id = scene.Nodes().front().id;
    auto* node = scene.GetNode(node_id);
    REQUIRE(node != nullptr);

    Transform original = node->transform;
    Transform new_tf;
    new_tf.translation = {1.0f, 2.0f, 3.0f};

    scene.SetNodeTransform(node_id, new_tf);

    node = scene.GetNode(node_id);
    REQUIRE(node->transform.translation == new_tf.translation);
    REQUIRE(node->prev_transform.translation == original.translation);
}
