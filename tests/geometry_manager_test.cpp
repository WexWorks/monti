#include <catch2/catch_test_macros.hpp>

#include <volk.h>

#include "../app/core/vulkan_context.h"
#include "scenes/CornellBox.h"

#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>

// Access internal headers for direct testing
#include "../renderer/src/vulkan/GpuScene.h"
#include "../renderer/src/vulkan/GeometryManager.h"

#include <cstring>

using namespace monti;
using namespace monti::vulkan;

namespace {

struct TestContext {
    monti::app::VulkanContext ctx;

    bool Init() {
        if (!ctx.CreateInstance()) return false;
        if (!ctx.CreateDevice(std::nullopt)) return false;
        return true;
    }
};

// Upload mesh data to GPU and register bindings with GpuScene.
// Returns GPU buffers that must be kept alive while in use.
std::vector<GpuBuffer> UploadAndRegisterMeshes(
    monti::app::VulkanContext& ctx,
    GpuScene& gpu_scene,
    const std::vector<MeshData>& mesh_data) {

    VkCommandBuffer cmd = ctx.BeginOneShot();
    std::vector<GpuBuffer> gpu_buffers;
    for (const auto& md : mesh_data) {
        auto [vb, ib] = UploadMeshToGpu(ctx.Allocator(), ctx.Device(), cmd, md);
        MeshBufferBinding binding{};
        binding.vertex_buffer = vb.buffer;
        binding.vertex_address = vb.device_address;
        binding.index_buffer = ib.buffer;
        binding.index_address = ib.device_address;
        binding.vertex_count = static_cast<uint32_t>(md.vertices.size());
        binding.index_count = static_cast<uint32_t>(md.indices.size());
        binding.vertex_stride = sizeof(Vertex);
        gpu_scene.RegisterMeshBuffers(md.mesh_id, binding);
        gpu_buffers.push_back(std::move(vb));
        gpu_buffers.push_back(std::move(ib));
    }
    ctx.SubmitAndWait(cmd);
    return gpu_buffers;
}

}  // anonymous namespace

TEST_CASE("GeometryManager: BLAS build and TLAS build", "[geometry_manager][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;
    auto [scene, mesh_data] = test::BuildCornellBox();
    REQUIRE(mesh_data.size() == 7);

    GpuScene gpu_scene(ctx.Allocator(), ctx.Device(), ctx.PhysicalDevice());
    auto gpu_buffers = UploadAndRegisterMeshes(ctx, gpu_scene, mesh_data);
    REQUIRE(gpu_scene.UpdateMaterials(scene));

    // Upload mesh address table
    gpu_scene.UploadMeshAddressTable();
    REQUIRE(gpu_scene.MeshAddressBuffer() != VK_NULL_HANDLE);

    GeometryManager geom_mgr(ctx.Allocator(), ctx.Device());

    // Build BLAS
    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(cmd != VK_NULL_HANDLE);
    REQUIRE(geom_mgr.BuildDirtyBlas(cmd, gpu_scene));
    ctx.SubmitAndWait(cmd);

    // Build TLAS
    cmd = ctx.BeginOneShot();
    REQUIRE(cmd != VK_NULL_HANDLE);
    REQUIRE(geom_mgr.BuildTlas(cmd, scene, gpu_scene));
    ctx.SubmitAndWait(cmd);

    // Verify TLAS
    REQUIRE(geom_mgr.Tlas() != VK_NULL_HANDLE);
    REQUIRE(geom_mgr.TlasInstanceCount() == scene.Nodes().size());

    // Clean up
    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

TEST_CASE("GeometryManager: BLAS compaction reduces memory", "[geometry_manager][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;
    auto [scene, mesh_data] = test::BuildCornellBox();

    GpuScene gpu_scene(ctx.Allocator(), ctx.Device(), ctx.PhysicalDevice());
    auto gpu_buffers = UploadAndRegisterMeshes(ctx, gpu_scene, mesh_data);
    REQUIRE(gpu_scene.UpdateMaterials(scene));
    gpu_scene.UploadMeshAddressTable();

    GeometryManager geom_mgr(ctx.Allocator(), ctx.Device());

    // Frame 1: Build uncompacted BLAS + write compaction queries
    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(cmd != VK_NULL_HANDLE);
    REQUIRE(geom_mgr.BuildDirtyBlas(cmd, gpu_scene));
    REQUIRE(geom_mgr.BuildTlas(cmd, scene, gpu_scene));
    ctx.SubmitAndWait(cmd);

    REQUIRE(geom_mgr.Tlas() != VK_NULL_HANDLE);

    // Frame 2: Compact BLAS (previous frame's fence has signaled)
    cmd = ctx.BeginOneShot();
    REQUIRE(cmd != VK_NULL_HANDLE);
    REQUIRE(geom_mgr.CompactPendingBlas(cmd));
    // Rebuild TLAS with updated compacted addresses
    REQUIRE(geom_mgr.BuildTlas(cmd, scene, gpu_scene));
    ctx.SubmitAndWait(cmd);

    REQUIRE(geom_mgr.Tlas() != VK_NULL_HANDLE);
    REQUIRE(geom_mgr.TlasInstanceCount() == scene.Nodes().size());

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

TEST_CASE("GeometryManager: TLAS skips rebuild for unchanged scene", "[geometry_manager][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;
    auto [scene, mesh_data] = test::BuildCornellBox();

    GpuScene gpu_scene(ctx.Allocator(), ctx.Device(), ctx.PhysicalDevice());
    auto gpu_buffers = UploadAndRegisterMeshes(ctx, gpu_scene, mesh_data);
    REQUIRE(gpu_scene.UpdateMaterials(scene));
    gpu_scene.UploadMeshAddressTable();

    GeometryManager geom_mgr(ctx.Allocator(), ctx.Device());

    // Build everything
    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(cmd != VK_NULL_HANDLE);
    REQUIRE(geom_mgr.BuildDirtyBlas(cmd, gpu_scene));
    REQUIRE(geom_mgr.BuildTlas(cmd, scene, gpu_scene));
    ctx.SubmitAndWait(cmd);

    auto gen_before = scene.TlasGeneration();

    // Second call without changes — should skip TLAS rebuild
    cmd = ctx.BeginOneShot();
    REQUIRE(cmd != VK_NULL_HANDLE);
    REQUIRE(geom_mgr.BuildTlas(cmd, scene, gpu_scene));
    ctx.SubmitAndWait(cmd);

    // Generation unchanged
    REQUIRE(scene.TlasGeneration() == gen_before);

    // Now modify a transform — should trigger rebuild
    auto& nodes = scene.Nodes();
    REQUIRE(!nodes.empty());
    Transform t = nodes[0].transform;
    t.translation.x += 1.0f;
    scene.SetNodeTransform(nodes[0].id, t);
    REQUIRE(scene.TlasGeneration() == gen_before + 1);

    cmd = ctx.BeginOneShot();
    REQUIRE(cmd != VK_NULL_HANDLE);
    REQUIRE(geom_mgr.BuildTlas(cmd, scene, gpu_scene));
    ctx.SubmitAndWait(cmd);

    REQUIRE(geom_mgr.Tlas() != VK_NULL_HANDLE);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

TEST_CASE("GeometryManager: mesh removal cleans up BLAS", "[geometry_manager][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;
    auto [scene, mesh_data] = test::BuildCornellBox();

    GpuScene gpu_scene(ctx.Allocator(), ctx.Device(), ctx.PhysicalDevice());
    auto gpu_buffers = UploadAndRegisterMeshes(ctx, gpu_scene, mesh_data);
    REQUIRE(gpu_scene.UpdateMaterials(scene));
    gpu_scene.UploadMeshAddressTable();

    GeometryManager geom_mgr(ctx.Allocator(), ctx.Device());

    // Build everything
    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(cmd != VK_NULL_HANDLE);
    REQUIRE(geom_mgr.BuildDirtyBlas(cmd, gpu_scene));
    REQUIRE(geom_mgr.BuildTlas(cmd, scene, gpu_scene));
    ctx.SubmitAndWait(cmd);

    auto initial_count = geom_mgr.TlasInstanceCount();
    REQUIRE(initial_count == scene.Nodes().size());

    // Remove the last node and its mesh
    auto& nodes = scene.Nodes();
    auto last_node_id = nodes.back().id;
    auto last_mesh_id = nodes.back().mesh_id;
    scene.RemoveNode(last_node_id);
    REQUIRE(scene.RemoveMesh(last_mesh_id));

    // Cleanup should destroy the BLAS for the removed mesh
    geom_mgr.CleanupRemovedMeshes(scene);

    // Rebuild TLAS with fewer instances
    cmd = ctx.BeginOneShot();
    REQUIRE(cmd != VK_NULL_HANDLE);
    REQUIRE(geom_mgr.BuildTlas(cmd, scene, gpu_scene));
    ctx.SubmitAndWait(cmd);

    REQUIRE(geom_mgr.TlasInstanceCount() == initial_count - 1);

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}

TEST_CASE("Scene: TlasGeneration increments on structural changes", "[scene]") {
    Scene scene;

    auto mesh_id = scene.AddMesh(Mesh{}, "test");
    auto mat_id = scene.AddMaterial(MaterialDesc{}, "test");

    uint64_t gen0 = scene.TlasGeneration();
    REQUIRE(gen0 == 0);

    auto node_id = scene.AddNode(mesh_id, mat_id, "node1");
    REQUIRE(scene.TlasGeneration() == gen0 + 1);

    Transform t;
    t.translation = {1.0f, 0.0f, 0.0f};
    scene.SetNodeTransform(node_id, t);
    REQUIRE(scene.TlasGeneration() == gen0 + 2);

    scene.RemoveNode(node_id);
    REQUIRE(scene.TlasGeneration() == gen0 + 3);
}

TEST_CASE("GeometryManager: buffer address table populated correctly", "[geometry_manager][vulkan][integration]") {
    TestContext tc;
    REQUIRE(tc.Init());

    auto& ctx = tc.ctx;
    auto [scene, mesh_data] = test::BuildCornellBox();

    GpuScene gpu_scene(ctx.Allocator(), ctx.Device(), ctx.PhysicalDevice());
    auto gpu_buffers = UploadAndRegisterMeshes(ctx, gpu_scene, mesh_data);

    // Upload mesh address table
    gpu_scene.UploadMeshAddressTable();
    REQUIRE(gpu_scene.MeshAddressBuffer() != VK_NULL_HANDLE);
    REQUIRE(gpu_scene.MeshAddressBufferSize() ==
            mesh_data.size() * sizeof(MeshAddressEntry));

    // Verify address index mapping for each mesh
    for (uint32_t i = 0; i < mesh_data.size(); ++i) {
        uint32_t addr_idx = gpu_scene.GetMeshAddressIndex(mesh_data[i].mesh_id);
        REQUIRE(addr_idx == i);
    }

    // Verify all registered meshes have non-zero device addresses
    for (const auto& md : mesh_data) {
        const auto* binding = gpu_scene.GetMeshBinding(md.mesh_id);
        REQUIRE(binding != nullptr);
        REQUIRE(binding->vertex_address != 0);
        REQUIRE(binding->index_address != 0);
    }

    for (auto& buf : gpu_buffers)
        DestroyGpuBuffer(ctx.Allocator(), buf);
    ctx.WaitIdle();
}
