#include "CornellBox.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace monti::test {

namespace {

// Helper to create a quad mesh (2 triangles) from 4 corners in CCW order.
MeshData MakeQuad(const glm::vec3& v0, const glm::vec3& v1,
                  const glm::vec3& v2, const glm::vec3& v3,
                  const glm::vec3& normal) {
    MeshData data;

    auto make_vertex = [&](const glm::vec3& pos, const glm::vec2& uv) {
        Vertex v{};
        v.position = pos;
        v.normal = normal;
        // Tangent derived from edge direction
        glm::vec3 edge = glm::normalize(v1 - v0);
        v.tangent = glm::vec4(edge, 1.0f);
        v.tex_coord_0 = uv;
        v.tex_coord_1 = uv;
        return v;
    };

    data.vertices = {
        make_vertex(v0, {0.0f, 0.0f}),
        make_vertex(v1, {1.0f, 0.0f}),
        make_vertex(v2, {1.0f, 1.0f}),
        make_vertex(v3, {0.0f, 1.0f}),
    };

    data.indices = {0, 1, 2, 0, 2, 3};
    return data;
}

// Helper to create a box mesh (12 triangles, 24 vertices — 4 per face).
MeshData MakeBox(const glm::vec3& min, const glm::vec3& max) {
    MeshData data;

    auto add_face = [&](const glm::vec3& v0, const glm::vec3& v1,
                        const glm::vec3& v2, const glm::vec3& v3,
                        const glm::vec3& normal) {
        glm::vec3 edge = glm::normalize(v1 - v0);
        auto base = static_cast<uint32_t>(data.vertices.size());

        auto make_vertex = [&](const glm::vec3& pos, const glm::vec2& uv) {
            Vertex v{};
            v.position = pos;
            v.normal = normal;
            v.tangent = glm::vec4(edge, 1.0f);
            v.tex_coord_0 = uv;
            v.tex_coord_1 = uv;
            return v;
        };

        data.vertices.push_back(make_vertex(v0, {0.0f, 0.0f}));
        data.vertices.push_back(make_vertex(v1, {1.0f, 0.0f}));
        data.vertices.push_back(make_vertex(v2, {1.0f, 1.0f}));
        data.vertices.push_back(make_vertex(v3, {0.0f, 1.0f}));

        data.indices.insert(data.indices.end(),
            {base, base + 1, base + 2, base, base + 2, base + 3});
    };

    // Front face (+Z)
    add_face({min.x, min.y, max.z}, {max.x, min.y, max.z},
             {max.x, max.y, max.z}, {min.x, max.y, max.z},
             {0, 0, 1});
    // Back face (-Z)
    add_face({max.x, min.y, min.z}, {min.x, min.y, min.z},
             {min.x, max.y, min.z}, {max.x, max.y, min.z},
             {0, 0, -1});
    // Right face (+X)
    add_face({max.x, min.y, max.z}, {max.x, min.y, min.z},
             {max.x, max.y, min.z}, {max.x, max.y, max.z},
             {1, 0, 0});
    // Left face (-X)
    add_face({min.x, min.y, min.z}, {min.x, min.y, max.z},
             {min.x, max.y, max.z}, {min.x, max.y, min.z},
             {-1, 0, 0});
    // Top face (+Y)
    add_face({min.x, max.y, max.z}, {max.x, max.y, max.z},
             {max.x, max.y, min.z}, {min.x, max.y, min.z},
             {0, 1, 0});
    // Bottom face (-Y)
    add_face({min.x, min.y, min.z}, {max.x, min.y, min.z},
             {max.x, min.y, max.z}, {min.x, min.y, max.z},
             {0, -1, 0});

    return data;
}

Mesh MakeMeshMetadata(std::string_view name, const MeshData& data) {
    Mesh mesh;
    mesh.name = std::string(name);
    mesh.vertex_count = static_cast<uint32_t>(data.vertices.size());
    mesh.index_count = static_cast<uint32_t>(data.indices.size());

    if (!data.vertices.empty()) {
        mesh.bbox_min = data.vertices[0].position;
        mesh.bbox_max = data.vertices[0].position;
        for (const auto& v : data.vertices) {
            mesh.bbox_min = glm::min(mesh.bbox_min, v.position);
            mesh.bbox_max = glm::max(mesh.bbox_max, v.position);
        }
    }

    return mesh;
}

} // anonymous namespace

CornellBoxResult BuildCornellBox() {
    CornellBoxResult result;
    auto& scene = result.scene;
    auto& all_mesh_data = result.mesh_data;

    // ── Materials ────────────────────────────────────────────────────
    MaterialDesc white_mat;
    white_mat.base_color = {0.73f, 0.73f, 0.73f};
    white_mat.roughness = 1.0f;
    white_mat.metallic = 0.0f;
    auto white_id = scene.AddMaterial(std::move(white_mat), "white");

    MaterialDesc red_mat;
    red_mat.base_color = {0.65f, 0.05f, 0.05f};
    red_mat.roughness = 1.0f;
    red_mat.metallic = 0.0f;
    auto red_id = scene.AddMaterial(std::move(red_mat), "red");

    MaterialDesc green_mat;
    green_mat.base_color = {0.12f, 0.45f, 0.15f};
    green_mat.roughness = 1.0f;
    green_mat.metallic = 0.0f;
    auto green_id = scene.AddMaterial(std::move(green_mat), "green");

    MaterialDesc light_mat;
    light_mat.base_color = {1.0f, 1.0f, 1.0f};
    light_mat.emissive_factor = {17.0f, 12.0f, 4.0f};
    light_mat.emissive_strength = 1.0f;
    light_mat.roughness = 1.0f;
    light_mat.metallic = 0.0f;
    scene.AddMaterial(std::move(light_mat), "light");

    // ── Helper to add mesh + node ────────────────────────────────────
    auto add_mesh = [&](std::string_view name, MeshData data,
                        MaterialId mat_id) -> MeshId {
        auto mesh = MakeMeshMetadata(name, data);
        auto mesh_id = scene.AddMesh(std::move(mesh), name);
        data.mesh_id = mesh_id;
        scene.AddNode(mesh_id, mat_id, name);
        all_mesh_data.push_back(std::move(data));
        return mesh_id;
    };

    // ── Walls (unit-scale room: 0 to 1 on all axes) ────────────────

    // Floor (white) — Y=0 plane
    add_mesh("floor",
        MakeQuad({0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1},
                 {0, 1, 0}),
        white_id);

    // Ceiling (white) — Y=1 plane
    add_mesh("ceiling",
        MakeQuad({0, 1, 1}, {1, 1, 1}, {1, 1, 0}, {0, 1, 0},
                 {0, -1, 0}),
        white_id);

    // Back wall (white) — Z=0 plane
    add_mesh("back_wall",
        MakeQuad({0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0},
                 {0, 0, 1}),
        white_id);

    // Left wall (red) — X=0 plane
    add_mesh("left_wall",
        MakeQuad({0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {0, 0, 0},
                 {1, 0, 0}),
        red_id);

    // Right wall (green) — X=1 plane
    add_mesh("right_wall",
        MakeQuad({1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {1, 0, 1},
                 {-1, 0, 0}),
        green_id);

    // ── Boxes ───────────────────────────────────────────────────────

    // Short box (white)
    add_mesh("short_box",
        MakeBox({0.13f, 0.0f, 0.065f}, {0.43f, 0.33f, 0.38f}),
        white_id);

    // Tall box (white)
    add_mesh("tall_box",
        MakeBox({0.53f, 0.0f, 0.37f}, {0.83f, 0.66f, 0.67f}),
        white_id);

    // ── Area light ──────────────────────────────────────────────────
    AreaLight ceiling_light;
    ceiling_light.corner = {0.35f, 0.999f, 0.35f};
    ceiling_light.edge_a = {0.3f, 0.0f, 0.0f};
    ceiling_light.edge_b = {0.0f, 0.0f, 0.3f};
    ceiling_light.radiance = {17.0f, 12.0f, 4.0f};
    ceiling_light.two_sided = false;
    scene.AddAreaLight(ceiling_light);

    // ── Camera ──────────────────────────────────────────────────────
    CameraParams camera;
    camera.position = {0.5f, 0.5f, 1.94f};
    camera.target = {0.5f, 0.5f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.vertical_fov_radians = glm::radians(39.3077f);
    camera.near_plane = 0.01f;
    camera.far_plane = 10.0f;
    scene.SetActiveCamera(camera);

    return result;
}

} // namespace monti::test
