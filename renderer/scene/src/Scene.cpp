#include <monti/scene/Scene.h>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cstdio>
#include <ranges>
#include <type_traits>

namespace monti {

namespace {

template <typename Container, typename Id>
auto FindById(Container& container, Id id) {
    auto it = std::ranges::find_if(container, [id](const auto& elem) {
        return elem.id == id;
    });
    if constexpr (std::is_const_v<std::remove_reference_t<Container>>)
        return it != container.end() ? &*it : static_cast<const typename Container::value_type*>(nullptr);
    else
        return it != container.end() ? &*it : static_cast<typename Container::value_type*>(nullptr);
}

}  // namespace

glm::mat4 Transform::ToMatrix() const {
    glm::mat4 t = glm::translate(glm::mat4(1.0f), translation);
    glm::mat4 r = glm::mat4_cast(rotation);
    glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
    return t * r * s;
}

MeshId Scene::AddMesh(Mesh mesh, std::string_view name) {
    mesh.id = MeshId{next_mesh_id_++};
    if (!name.empty()) mesh.name = std::string(name);
    meshes_.push_back(std::move(mesh));
    return meshes_.back().id;
}

MaterialId Scene::AddMaterial(MaterialDesc material, std::string_view name) {
    material.id = MaterialId{next_material_id_++};
    if (!name.empty()) material.name = std::string(name);
    materials_.push_back(std::move(material));
    return materials_.back().id;
}

TextureId Scene::AddTexture(TextureDesc texture, std::string_view name) {
    texture.id = TextureId{next_texture_id_++};
    if (!name.empty()) texture.name = std::string(name);
    textures_.push_back(std::move(texture));
    return textures_.back().id;
}

NodeId Scene::AddNode(MeshId mesh, MaterialId material, std::string_view name) {
    SceneNode node;
    node.id = NodeId{next_node_id_++};
    node.mesh_id = mesh;
    node.material_id = material;
    if (!name.empty()) node.name = std::string(name);
    nodes_.push_back(std::move(node));
    ++tlas_generation_;
    return nodes_.back().id;
}

void Scene::RemoveNode(NodeId id) {
    auto it = std::ranges::find_if(nodes_, [id](const SceneNode& n) {
        return n.id == id;
    });
    if (it != nodes_.end()) {
        nodes_.erase(it);
        ++tlas_generation_;
    }
}

bool Scene::RemoveMesh(MeshId id) {
    // Only remove if no nodes reference this mesh
    bool referenced = std::ranges::any_of(nodes_, [id](const SceneNode& n) {
        return n.mesh_id == id;
    });
    if (referenced) return false;

    auto it = std::ranges::find_if(meshes_, [id](const Mesh& m) {
        return m.id == id;
    });
    if (it != meshes_.end()) meshes_.erase(it);
    return true;
}

void Scene::SetNodeTransform(NodeId id, const Transform& new_transform) {
    if (auto* node = GetNode(id)) {
        node->prev_transform = node->transform;
        node->transform = new_transform;
        ++tlas_generation_;
    }
}

const Mesh* Scene::GetMesh(MeshId id) const {
    return FindById(meshes_, id);
}

MaterialDesc* Scene::GetMaterial(MaterialId id) {
    return FindById(materials_, id);
}

const MaterialDesc* Scene::GetMaterial(MaterialId id) const {
    return FindById(materials_, id);
}

SceneNode* Scene::GetNode(NodeId id) {
    return FindById(nodes_, id);
}

const SceneNode* Scene::GetNode(NodeId id) const {
    return FindById(nodes_, id);
}

const TextureDesc* Scene::GetTexture(TextureId id) const {
    return FindById(textures_, id);
}

TextureDesc* Scene::GetTexture(TextureId id) {
    return FindById(textures_, id);
}

const std::vector<Mesh>& Scene::Meshes() const { return meshes_; }
const std::vector<MaterialDesc>& Scene::Materials() const { return materials_; }
const std::vector<SceneNode>& Scene::Nodes() const { return nodes_; }
const std::vector<TextureDesc>& Scene::Textures() const { return textures_; }

void Scene::SetEnvironmentLight(const EnvironmentLight& light) {
    environment_light_ = light;
}

const EnvironmentLight* Scene::GetEnvironmentLight() const {
    return environment_light_ ? &*environment_light_ : nullptr;
}

void Scene::AddAreaLight(const AreaLight& light) {
    area_lights_.push_back(light);
}

const std::vector<AreaLight>& Scene::AreaLights() const {
    return area_lights_;
}

void Scene::AddSphereLight(const SphereLight& light) {
    if (light.radius <= 0.0f) {
        std::fprintf(stderr, "Scene::AddSphereLight: rejected degenerate sphere (radius <= 0)\n");
        return;
    }
    sphere_lights_.push_back(light);
}

void Scene::AddTriangleLight(const TriangleLight& light) {
    float area = glm::length(glm::cross(light.v1 - light.v0, light.v2 - light.v0));
    if (area <= 0.0f) {
        std::fprintf(stderr, "Scene::AddTriangleLight: rejected degenerate triangle (zero area)\n");
        return;
    }
    triangle_lights_.push_back(light);
}

const std::vector<SphereLight>& Scene::SphereLights() const {
    return sphere_lights_;
}

const std::vector<TriangleLight>& Scene::TriangleLights() const {
    return triangle_lights_;
}

uint64_t Scene::TlasGeneration() const { return tlas_generation_; }

std::vector<MeshData> SynthesizeAreaLightGeometry(Scene& scene) {
    std::vector<MeshData> result;
    const auto& area_lights = scene.AreaLights();
    if (area_lights.empty()) return result;

    // Guard against double-call: if "area_light_0" material already exists,
    // geometry was already synthesized for these lights.
    for (const auto& mat : scene.Materials())
        if (mat.name == "area_light_0") return result;

    result.reserve(area_lights.size());

    for (size_t i = 0; i < area_lights.size(); ++i) {
        const auto& light = area_lights[i];

        // Compute face normal from edge cross product
        glm::vec3 normal = glm::normalize(glm::cross(light.edge_a, light.edge_b));

        // Decompose radiance into emissive_factor (direction) and
        // emissive_strength (magnitude) so the shader reconstructs the
        // original HDR radiance.
        float max_comp = std::max({light.radiance.r, light.radiance.g, light.radiance.b});
        glm::vec3 emissive_factor = max_comp > 0.0f
            ? light.radiance / max_comp : glm::vec3(0.0f);
        float emissive_strength = max_comp;

        // Create an emissive material for this light
        MaterialDesc mat;
        mat.base_color = {0.0f, 0.0f, 0.0f};
        mat.roughness = 1.0f;
        mat.metallic = 0.0f;
        mat.emissive_factor = emissive_factor;
        mat.emissive_strength = emissive_strength;
        mat.double_sided = light.two_sided;
        auto mat_name = "area_light_" + std::to_string(i);
        auto mat_id = scene.AddMaterial(std::move(mat), mat_name);

        // Build 4 corner vertices for the quad
        glm::vec3 v0 = light.corner;
        glm::vec3 v1 = light.corner + light.edge_a;
        glm::vec3 v2 = light.corner + light.edge_a + light.edge_b;
        glm::vec3 v3 = light.corner + light.edge_b;

        glm::vec3 tangent_dir = glm::normalize(light.edge_a);

        auto make_vertex = [&](const glm::vec3& pos, const glm::vec2& uv) {
            Vertex v{};
            v.position = pos;
            v.normal = normal;
            v.tangent = glm::vec4(tangent_dir, 1.0f);
            v.tex_coord_0 = uv;
            v.tex_coord_1 = uv;
            return v;
        };

        MeshData mesh_data;
        mesh_data.vertices = {
            make_vertex(v0, {0.0f, 0.0f}),
            make_vertex(v1, {1.0f, 0.0f}),
            make_vertex(v2, {1.0f, 1.0f}),
            make_vertex(v3, {0.0f, 1.0f}),
        };
        mesh_data.indices = {0, 1, 2, 0, 2, 3};

        // Register mesh metadata and scene node
        Mesh mesh;
        mesh.name = mat_name;
        mesh.vertex_count = static_cast<uint32_t>(mesh_data.vertices.size());
        mesh.index_count = static_cast<uint32_t>(mesh_data.indices.size());
        mesh.bbox_min = glm::min(glm::min(v0, v1), glm::min(v2, v3));
        mesh.bbox_max = glm::max(glm::max(v0, v1), glm::max(v2, v3));

        auto mesh_id = scene.AddMesh(std::move(mesh), mat_name);
        mesh_data.mesh_id = mesh_id;
        scene.AddNode(mesh_id, mat_id, mat_name);

        result.push_back(std::move(mesh_data));
    }

    return result;
}

void Scene::SetActiveCamera(const CameraParams& params) {
    active_camera_ = params;
}

const CameraParams& Scene::GetActiveCamera() const {
    return active_camera_;
}

} // namespace monti
