#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "gltf/GltfLoader.h"

#include <monti/scene/Scene.h>
#include <monti/scene/Material.h>

#include <glm/glm.hpp>

#include <string>

using namespace monti;
using namespace monti::gltf;
using Catch::Matchers::WithinAbs;

static std::string AssetPath(const char* filename) {
    return std::string(MONTI_TEST_ASSETS_DIR) + "/" + filename;
}

// ── Box.glb ──────────────────────────────────────────────────────────────

TEST_CASE("Box.glb loads successfully", "[gltf][box]") {
    Scene scene;
    auto result = LoadGltf(scene, AssetPath("Box.glb"));

    REQUIRE(result.success);
    REQUIRE(result.error_message.empty());
}

TEST_CASE("Box.glb has expected entity counts", "[gltf][box]") {
    Scene scene;
    auto result = LoadGltf(scene, AssetPath("Box.glb"));

    REQUIRE(result.success);
    REQUIRE(scene.Meshes().size() == 1);
    REQUIRE(scene.Materials().size() == 1);
    REQUIRE(scene.Nodes().size() == 1);
    REQUIRE(result.nodes.size() == 1);
}

TEST_CASE("Box.glb has correct vertex and index counts", "[gltf][box]") {
    Scene scene;
    auto result = LoadGltf(scene, AssetPath("Box.glb"));

    REQUIRE(result.success);
    REQUIRE(result.mesh_data.size() == 1);

    const auto& md = result.mesh_data[0];
    REQUIRE(md.vertices.size() == 24);
    REQUIRE(md.indices.size() == 36);

    // MeshData mesh_id matches the scene mesh
    const auto* mesh = scene.GetMesh(md.mesh_id);
    REQUIRE(mesh != nullptr);
    REQUIRE(mesh->vertex_count == 24);
    REQUIRE(mesh->index_count == 36);
}

TEST_CASE("Box.glb has non-degenerate bounding box", "[gltf][box]") {
    Scene scene;
    auto result = LoadGltf(scene, AssetPath("Box.glb"));

    REQUIRE(result.success);
    const auto& mesh = scene.Meshes()[0];

    // Bounding box should have non-zero extent in all axes
    glm::vec3 extent = mesh.bbox_max - mesh.bbox_min;
    REQUIRE(extent.x > 0.0f);
    REQUIRE(extent.y > 0.0f);
    REQUIRE(extent.z > 0.0f);
}

// ── DamagedHelmet.glb ────────────────────────────────────────────────────

TEST_CASE("DamagedHelmet.glb loads successfully", "[gltf][damaged_helmet]") {
    Scene scene;
    auto result = LoadGltf(scene, AssetPath("DamagedHelmet.glb"));

    REQUIRE(result.success);
}

TEST_CASE("DamagedHelmet.glb has PBR textures", "[gltf][damaged_helmet]") {
    Scene scene;
    auto result = LoadGltf(scene, AssetPath("DamagedHelmet.glb"));

    REQUIRE(result.success);
    REQUIRE(scene.Meshes().size() >= 1);
    REQUIRE(scene.Materials().size() >= 1);
    REQUIRE(scene.Textures().size() >= 4);  // base color, normal, metallic-roughness, emissive

    // Verify the first material has PBR texture maps assigned
    const auto& mat = scene.Materials()[0];
    REQUIRE(mat.base_color_map.has_value());
    REQUIRE(mat.normal_map.has_value());
    REQUIRE(mat.metallic_roughness_map.has_value());
    REQUIRE(mat.emissive_map.has_value());
}

TEST_CASE("DamagedHelmet.glb material has correct PBR defaults", "[gltf][damaged_helmet]") {
    Scene scene;
    auto result = LoadGltf(scene, AssetPath("DamagedHelmet.glb"));

    REQUIRE(result.success);
    const auto& mat = scene.Materials()[0];

    // DamagedHelmet has factor defaults of 1.0 (texture-driven values)
    REQUIRE_THAT(mat.roughness, WithinAbs(1.0, 0.001));
    REQUIRE_THAT(mat.metallic, WithinAbs(1.0, 0.001));
}

// ── MorphPrimitivesTest.glb (multi-primitive) ────────────────────────────

TEST_CASE("MorphPrimitivesTest.glb produces separate meshes per primitive", "[gltf][multi_prim]") {
    Scene scene;
    auto result = LoadGltf(scene, AssetPath("MorphPrimitivesTest.glb"));

    REQUIRE(result.success);

    // This model has 1 glTF mesh with 2 primitives (different materials)
    REQUIRE(scene.Meshes().size() == 2);
    REQUIRE(scene.Nodes().size() == 2);
    REQUIRE(result.mesh_data.size() == 2);
    REQUIRE(result.nodes.size() == 2);
}

TEST_CASE("MorphPrimitivesTest.glb primitives have distinct materials", "[gltf][multi_prim]") {
    Scene scene;
    auto result = LoadGltf(scene, AssetPath("MorphPrimitivesTest.glb"));

    REQUIRE(result.success);
    REQUIRE(scene.Nodes().size() >= 2);

    // Each primitive should have a different material
    const auto* node0 = scene.GetNode(result.nodes[0]);
    const auto* node1 = scene.GetNode(result.nodes[1]);
    REQUIRE(node0 != nullptr);
    REQUIRE(node1 != nullptr);
    REQUIRE(node0->material_id != node1->material_id);
}

// ── Error handling ───────────────────────────────────────────────────────

TEST_CASE("LoadGltf returns failure for non-existent file", "[gltf][error]") {
    Scene scene;
    auto result = LoadGltf(scene, "non_existent_file.glb");

    REQUIRE_FALSE(result.success);
    REQUIRE_FALSE(result.error_message.empty());
    REQUIRE(result.nodes.empty());
    REQUIRE(result.mesh_data.empty());
}

// ── Sampler ──────────────────────────────────────────────────────────────

TEST_CASE("Texture samplers have correct default wrap/filter modes", "[gltf][sampler]") {
    Scene scene;
    auto result = LoadGltf(scene, AssetPath("DamagedHelmet.glb"));

    REQUIRE(result.success);
    REQUIRE_FALSE(scene.Textures().empty());

    // Verify at least one texture has sampler properties set.
    // glTF 2.0 defaults: wrap = Repeat, filter = Linear.
    bool found_with_defaults = false;
    for (const auto& tex : scene.Textures()) {
        if (tex.wrap_s == SamplerWrap::kRepeat &&
            tex.wrap_t == SamplerWrap::kRepeat &&
            tex.mag_filter == SamplerFilter::kLinear &&
            tex.min_filter == SamplerFilter::kLinear) {
            found_with_defaults = true;
            break;
        }
    }
    REQUIRE(found_with_defaults);
}

// ── Tangent generation ───────────────────────────────────────────────────

TEST_CASE("Box.glb gets tangents generated via MikkTSpace", "[gltf][tangent]") {
    Scene scene;
    auto result = LoadGltf(scene, AssetPath("Box.glb"));

    REQUIRE(result.success);
    REQUIRE(result.mesh_data.size() == 1);

    // Verify tangents are non-zero (MikkTSpace generated them)
    const auto& vertices = result.mesh_data[0].vertices;
    bool has_nonzero_tangent = false;
    for (const auto& v : vertices) {
        if (glm::length(glm::vec3(v.tangent)) > 0.01f) {
            has_nonzero_tangent = true;
            break;
        }
    }
    REQUIRE(has_nonzero_tangent);
}

// ── World transform flattening ───────────────────────────────────────────

TEST_CASE("Box.glb node has a valid transform", "[gltf][transform]") {
    Scene scene;
    auto result = LoadGltf(scene, AssetPath("Box.glb"));

    REQUIRE(result.success);
    REQUIRE(result.nodes.size() == 1);

    const auto* node = scene.GetNode(result.nodes[0]);
    REQUIRE(node != nullptr);

    // The transform matrix should be invertible (non-degenerate)
    glm::mat4 m = node->transform.ToMatrix();
    float det = glm::determinant(m);
    REQUIRE(std::abs(det) > 1e-6f);
}
