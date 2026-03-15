#pragma once

#include "Buffer.h"

#include <monti/scene/Types.h>

namespace monti::vulkan { struct DeviceDispatch; }

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace monti {
class Scene;
}

namespace monti::vulkan {

class GpuScene;

// Manages bottom-level and top-level acceleration structures for ray tracing.
// Internal to the renderer — not exposed in public headers.
//
// BLAS lifecycle:
//   - One BLAS per registered mesh
//   - Deferred compaction: Frame N builds uncompacted + writes query,
//     Frame N+1 reads query results and compacts
//   - Refit for deformed meshes (topology_changed=false), full rebuild otherwise
//
// TLAS lifecycle:
//   - Rebuilt when Scene::TlasGeneration() advances or after BLAS compaction
//   - Skipped entirely for static scenes
class GeometryManager {
public:
    GeometryManager(VmaAllocator allocator, VkDevice device, const DeviceDispatch& dispatch);
    ~GeometryManager();

    GeometryManager(const GeometryManager&) = delete;
    GeometryManager& operator=(const GeometryManager&) = delete;

    // Build BLAS for any newly registered meshes or meshes flagged for rebuild/refit.
    // Records build commands into cmd. Returns false on error.
    bool BuildDirtyBlas(VkCommandBuffer cmd, GpuScene& gpu_scene);

    // Compact BLAS entries that were built in a previous frame and whose
    // query results are now available. Must be called after the previous
    // frame's fence has signaled. Records copy commands into cmd.
    // Returns false on error.
    bool CompactPendingBlas(VkCommandBuffer cmd);

    // Build or rebuild the TLAS from visible scene nodes.
    // Skips rebuild if Scene::TlasGeneration() hasn't changed and no
    // compaction occurred. Records build commands into cmd.
    bool BuildTlas(VkCommandBuffer cmd, const Scene& scene, GpuScene& gpu_scene);

    // Mark a mesh as needing BLAS refit (topology_changed=false) or
    // full rebuild (topology_changed=true).
    void NotifyMeshDeformed(MeshId mesh, bool topology_changed);

    // Destroy BLAS for meshes no longer in the scene.
    void CleanupRemovedMeshes(const Scene& scene);

    // Current TLAS for descriptor binding.
    VkAccelerationStructureKHR Tlas() const { return tlas_; }

    // Number of instances in the current TLAS.
    uint32_t TlasInstanceCount() const { return tlas_instance_count_; }

private:
    enum class BlasState {
        kReady,              // Compacted and usable
        kPendingCompaction,  // Built uncompacted, query written, waiting for results
        kNeedsRebuild,       // Marked for full rebuild
        kNeedsRefit,         // Marked for refit (vertex data changed, topology same)
    };

    struct BlasEntry {
        Buffer buffer;
        VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
        VkDeviceAddress device_address = 0;
        BlasState state = BlasState::kReady;
        MeshId mesh_id;

        // Uncompacted BLAS kept alive until compaction completes
        Buffer uncompacted_buffer;
        VkAccelerationStructureKHR uncompacted_handle = VK_NULL_HANDLE;
        uint32_t query_index = 0;
    };

    struct PendingDestroy {
        Buffer buffer;
        VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    };

    void DestroyBlas(BlasEntry& entry);
    bool EnsureQueryPool(uint32_t required_count);
    bool EnsureScratchBuffer(VkDeviceSize required_size);
    static VkTransformMatrixKHR ToVkTransformMatrix(const glm::mat4& m);
    static uint32_t EncodeCustomIndex(uint32_t mesh_address_index,
                                      uint32_t material_index);

    VmaAllocator allocator_;
    VkDevice device_;
    const DeviceDispatch* dispatch_ = nullptr;

    std::unordered_map<MeshId, BlasEntry> blas_map_;

    // Uncompacted BLAS awaiting command buffer completion.
    // Each batch is deferred for kDestroyDelay frames before actual destruction
    // to ensure all in-flight command buffers have completed.
    static constexpr uint32_t kDestroyDelay = 3;
    struct DeferredDestroyBatch {
        std::vector<PendingDestroy> resources;
        uint32_t frames_remaining = kDestroyDelay;
    };
    std::vector<DeferredDestroyBatch> deferred_destroys_;

    // Query pool for compaction size queries
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    uint32_t query_pool_capacity_ = 0;
    uint32_t next_query_index_ = 0;

    // Shared scratch buffer (reused for BLAS and TLAS builds)
    Buffer scratch_buffer_;

    // TLAS
    Buffer tlas_buffer_;
    VkAccelerationStructureKHR tlas_ = VK_NULL_HANDLE;
    Buffer instance_buffer_;
    uint32_t tlas_instance_count_ = 0;
    uint64_t cached_tlas_generation_ = UINT64_MAX;
    bool tlas_force_rebuild_ = false;
};

}  // namespace monti::vulkan
