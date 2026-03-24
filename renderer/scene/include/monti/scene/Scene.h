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
    bool       RemoveMesh(MeshId id);  // Returns false if nodes still reference it

    // ── Transform ────────────────────────────────────────────────────
    void SetNodeTransform(NodeId id, const Transform& new_transform);

    // ── Accessors ────────────────────────────────────────────────────
    const Mesh*         GetMesh(MeshId id) const;
    MaterialDesc*       GetMaterial(MaterialId id);
    const MaterialDesc* GetMaterial(MaterialId id) const;
    SceneNode*          GetNode(NodeId id);
    const SceneNode*    GetNode(NodeId id) const;
    const TextureDesc*  GetTexture(TextureId id) const;
    TextureDesc*        GetTexture(TextureId id);

    const std::vector<Mesh>&         Meshes() const;
    const std::vector<MaterialDesc>& Materials() const;
    const std::vector<SceneNode>&    Nodes() const;
    const std::vector<TextureDesc>&  Textures() const;

    // ── Lights ───────────────────────────────────────────────────────
    void SetEnvironmentLight(const EnvironmentLight& light);
    const EnvironmentLight* GetEnvironmentLight() const;

    void AddAreaLight(const AreaLight& light);
    const std::vector<AreaLight>& AreaLights() const;

    void AddSphereLight(const SphereLight& light);
    void AddTriangleLight(const TriangleLight& light);
    const std::vector<SphereLight>& SphereLights() const;
    const std::vector<TriangleLight>& TriangleLights() const;

    // ── Camera ───────────────────────────────────────────────────────
    void SetActiveCamera(const CameraParams& params);
    const CameraParams& GetActiveCamera() const;
    // ── TLAS dirty tracking ──────────────────────────────────────────
    uint64_t TlasGeneration() const;
private:
    std::vector<Mesh>         meshes_;
    std::vector<MaterialDesc> materials_;
    std::vector<SceneNode>    nodes_;
    std::vector<TextureDesc>  textures_;

    std::optional<EnvironmentLight> environment_light_;
    std::vector<AreaLight> area_lights_;
    std::vector<SphereLight> sphere_lights_;
    std::vector<TriangleLight> triangle_lights_;
    CameraParams active_camera_;

    uint64_t next_mesh_id_     = 0;
    uint64_t next_material_id_ = 0;
    uint64_t next_texture_id_  = 0;
    uint64_t next_node_id_     = 0;
    uint64_t tlas_generation_  = 0;
};

} // namespace monti
