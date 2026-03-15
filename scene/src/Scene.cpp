#include <monti/scene/Scene.h>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
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

uint64_t Scene::TlasGeneration() const { return tlas_generation_; }

void Scene::SetActiveCamera(const CameraParams& params) {
    active_camera_ = params;
}

const CameraParams& Scene::GetActiveCamera() const {
    return active_camera_;
}

} // namespace monti
