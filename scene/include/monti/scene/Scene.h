#pragma once
#include "Types.h"
#include "Material.h"
#include "Light.h"
#include "Camera.h"
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace monti {

struct SceneNode {
    NodeId      id;
    MeshId      mesh_id;
    MaterialId  material_id;
    Transform   transform;
    Transform   prev_transform;   // For motion vectors
    bool        visible = true;
    std::string name;
};

class Scene {
public:
    Scene() = default;

    // ── Entity lifecycle ─────────────────────────────────────────────
    MeshId     AddMesh(Mesh mesh, std::string_view name = "");
    MaterialId AddMaterial(MaterialDesc material, std::string_view name = "");
    TextureId  AddTexture(TextureDesc texture, std::string_view name = "");
    NodeId     AddNode(MeshId mesh, MaterialId material,
                       std::string_view name = "");
    void       RemoveNode(NodeId id);
    void       RemoveMesh(MeshId id);  // Only when no nodes reference it

    // ── Transform ────────────────────────────────────────────────────
    void SetNodeTransform(NodeId id, const Transform& new_transform);

    // ── Accessors ────────────────────────────────────────────────────
    const Mesh*         GetMesh(MeshId id) const;
    MaterialDesc*       GetMaterial(MaterialId id);
    const MaterialDesc* GetMaterial(MaterialId id) const;
    SceneNode*          GetNode(NodeId id);
    const SceneNode*    GetNode(NodeId id) const;
    const TextureDesc*  GetTexture(TextureId id) const;

    const std::vector<Mesh>&         Meshes() const;
    const std::vector<MaterialDesc>& Materials() const;
    const std::vector<SceneNode>&    Nodes() const;
    std::vector<SceneNode>&          Nodes();
    const std::vector<TextureDesc>&  Textures() const;

    // ── Light ────────────────────────────────────────────────────────
    void SetEnvironmentLight(const EnvironmentLight& light);
    const EnvironmentLight* GetEnvironmentLight() const;

    // ── Camera ───────────────────────────────────────────────────────
    void SetActiveCamera(const CameraParams& params);
    const CameraParams& GetActiveCamera() const;

private:
    std::vector<Mesh>         meshes_;
    std::vector<MaterialDesc> materials_;
    std::vector<SceneNode>    nodes_;
    std::vector<TextureDesc>  textures_;

    std::optional<EnvironmentLight> environment_light_;
    CameraParams active_camera_;

    uint64_t next_mesh_id_     = 0;
    uint64_t next_material_id_ = 0;
    uint64_t next_texture_id_  = 0;
    uint64_t next_node_id_     = 0;
};

} // namespace monti
