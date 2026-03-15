#include "GeometryManager.h"

#include "DeviceDispatch.h"
#include "GpuScene.h"

#include <monti/vulkan/Renderer.h>
#include <monti/scene/Scene.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

namespace monti::vulkan {

GeometryManager::GeometryManager(VmaAllocator allocator, VkDevice device,
                                 const DeviceDispatch& dispatch)
    : allocator_(allocator), device_(device), dispatch_(&dispatch) {}

GeometryManager::~GeometryManager() {
    for (auto& [id, entry] : blas_map_)
        DestroyBlas(entry);

    for (auto& batch : deferred_destroys_) {
        for (auto& res : batch.resources) {
            if (res.handle != VK_NULL_HANDLE)
                dispatch_->vkDestroyAccelerationStructureKHR(device_, res.handle, nullptr);
        }
    }

    if (tlas_ != VK_NULL_HANDLE)
        dispatch_->vkDestroyAccelerationStructureKHR(device_, tlas_, nullptr);

    if (query_pool_ != VK_NULL_HANDLE)
        dispatch_->vkDestroyQueryPool(device_, query_pool_, nullptr);
}

void GeometryManager::DestroyBlas(BlasEntry& entry) {
    if (entry.uncompacted_handle != VK_NULL_HANDLE) {
        dispatch_->vkDestroyAccelerationStructureKHR(device_, entry.uncompacted_handle, nullptr);
        entry.uncompacted_handle = VK_NULL_HANDLE;
        entry.uncompacted_buffer.Destroy();
    }
    if (entry.handle != VK_NULL_HANDLE) {
        dispatch_->vkDestroyAccelerationStructureKHR(device_, entry.handle, nullptr);
        entry.handle = VK_NULL_HANDLE;
        entry.buffer.Destroy();
    }
    entry.device_address = 0;
}

bool GeometryManager::EnsureQueryPool(uint32_t required_count) {
    if (query_pool_ != VK_NULL_HANDLE && query_pool_capacity_ >= required_count)
        return true;

    if (query_pool_ != VK_NULL_HANDLE) {
        dispatch_->vkDestroyQueryPool(device_, query_pool_, nullptr);
        query_pool_ = VK_NULL_HANDLE;
    }

    uint32_t new_capacity = std::max(required_count, 64u);

    VkQueryPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    ci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    ci.queryCount = new_capacity;

    VkResult result = dispatch_->vkCreateQueryPool(device_, &ci, nullptr, &query_pool_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GeometryManager: failed to create query pool (VkResult: %d)\n",
                     result);
        return false;
    }

    query_pool_capacity_ = new_capacity;
    return true;
}

bool GeometryManager::EnsureScratchBuffer(VkDeviceSize required_size) {
    if (scratch_buffer_.Handle() != VK_NULL_HANDLE && scratch_buffer_.Size() >= required_size)
        return true;

    scratch_buffer_.Destroy();
    return scratch_buffer_.Create(
        allocator_, required_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
}

VkTransformMatrixKHR GeometryManager::ToVkTransformMatrix(const glm::mat4& m) {
    // glm is column-major; VkTransformMatrixKHR is row-major 3x4
    VkTransformMatrixKHR vk_mat{};
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 4; ++col)
            vk_mat.matrix[row][col] = m[col][row];
    return vk_mat;
}

uint32_t GeometryManager::EncodeCustomIndex(uint32_t mesh_address_index,
                                            uint32_t material_index) {
    // Lower 12 bits: mesh address index, upper 12 bits: material index
    return (mesh_address_index & 0xFFFu) | ((material_index & 0xFFFu) << 12);
}

// ─────────────────────────────────────────────────────────────────────────────
// BLAS building
// ─────────────────────────────────────────────────────────────────────────────

bool GeometryManager::BuildDirtyBlas(VkCommandBuffer cmd, GpuScene& gpu_scene) {
    // Collect meshes that need building or refit
    struct BuildItem {
        MeshId mesh_id;
        const MeshBufferBinding* binding;
        bool is_refit;
    };
    std::vector<BuildItem> items;

    // Find new meshes (registered in GpuScene but no BLAS yet)
    // Iterate all mesh bindings in gpu_scene
    for (const auto& mesh_binding_pair : gpu_scene.MeshBindings()) {
        auto mesh_id = mesh_binding_pair.first;
        auto it = blas_map_.find(mesh_id);

        if (it == blas_map_.end()) {
            // New mesh — needs initial build
            items.push_back({mesh_id, &mesh_binding_pair.second, false});
        } else if (it->second.state == BlasState::kNeedsRebuild) {
            items.push_back({mesh_id, &mesh_binding_pair.second, false});
        } else if (it->second.state == BlasState::kNeedsRefit) {
            items.push_back({mesh_id, &mesh_binding_pair.second, true});
        }
    }

    if (items.empty()) return true;

    // Ensure query pool is large enough for new builds (not refits)
    uint32_t new_build_count = 0;
    for (const auto& item : items)
        if (!item.is_refit) ++new_build_count;

    if (new_build_count > 0 && !EnsureQueryPool(new_build_count))
        return false;

    // Reset query index for this batch
    next_query_index_ = 0;

    // Prepare build info for each item
    struct BlasBuildInfo {
        VkAccelerationStructureGeometryKHR geometry{};
        VkAccelerationStructureBuildGeometryInfoKHR build_info{};
        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        VkAccelerationStructureBuildRangeInfoKHR range_info{};
        MeshId mesh_id;
        bool is_refit;
    };

    std::vector<BlasBuildInfo> build_infos;
    build_infos.reserve(items.size());
    VkDeviceSize max_scratch_size = 0;

    for (const auto& item : items) {
        const auto* binding = item.binding;
        if (binding->index_count == 0) continue;

        build_infos.emplace_back();
        auto& info = build_infos.back();
        info.mesh_id = item.mesh_id;
        info.is_refit = item.is_refit;

        auto& geom = info.geometry;
        geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.flags = 0;  // No opaque bit — any-hit shader may run per instance

        auto& triangles = geom.geometry.triangles;
        triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = binding->vertex_address;
        triangles.vertexStride = binding->vertex_stride;
        triangles.maxVertex = binding->vertex_count - 1;
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = binding->index_address;

        auto& build = info.build_info;
        build.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        build.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        if (!item.is_refit)
            build.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
        build.mode = item.is_refit
            ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
            : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build.geometryCount = 1;
        build.pGeometries = &info.geometry;

        info.range_info.primitiveCount = binding->index_count / 3;
        info.range_info.primitiveOffset = 0;
        info.range_info.firstVertex = 0;
        info.range_info.transformOffset = 0;

        info.sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        uint32_t prim_count = info.range_info.primitiveCount;
        dispatch_->vkGetAccelerationStructureBuildSizesKHR(
            device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &info.build_info, &prim_count, &info.sizes);

        VkDeviceSize scratch = item.is_refit
            ? info.sizes.updateScratchSize
            : info.sizes.buildScratchSize;
        max_scratch_size = std::max(max_scratch_size, scratch);
    }

    if (build_infos.empty()) return true;

    if (!EnsureScratchBuffer(max_scratch_size))
        return false;

    VkDeviceAddress scratch_address = scratch_buffer_.DeviceAddress(device_, *dispatch_);

    // Reset query pool for new builds
    if (new_build_count > 0)
        dispatch_->vkCmdResetQueryPool(cmd, query_pool_, 0, new_build_count);

    // Tracks uncompacted handles in query_index order for the query write
    std::vector<VkAccelerationStructureKHR> query_handles;

    // Allocate acceleration structure buffers and record builds
    for (auto& info : build_infos) {
        if (info.is_refit) {
            // Refit existing BLAS in-place
            auto it = blas_map_.find(info.mesh_id);
            assert(it != blas_map_.end());
            auto& entry = it->second;

            info.build_info.srcAccelerationStructure = entry.handle;
            info.build_info.dstAccelerationStructure = entry.handle;
            info.build_info.scratchData.deviceAddress = scratch_address;

            const VkAccelerationStructureBuildRangeInfoKHR* range_ptr = &info.range_info;
            dispatch_->vkCmdBuildAccelerationStructuresKHR(cmd, 1, &info.build_info, &range_ptr);

            entry.state = BlasState::kReady;
            tlas_force_rebuild_ = true;
        } else {
            // New build — create uncompacted BLAS
            auto& entry = blas_map_[info.mesh_id];
            entry.mesh_id = info.mesh_id;

            // Clean up any previous uncompacted BLAS
            if (entry.uncompacted_handle != VK_NULL_HANDLE) {
                dispatch_->vkDestroyAccelerationStructureKHR(device_, entry.uncompacted_handle, nullptr);
                entry.uncompacted_handle = VK_NULL_HANDLE;
                entry.uncompacted_buffer.Destroy();
            }

            // For rebuilds, the old compacted BLAS stays alive until compaction
            // of the new one completes, so don't destroy entry.handle yet.

            if (!entry.uncompacted_buffer.Create(
                    allocator_, info.sizes.accelerationStructureSize,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_GPU_ONLY)) {
                std::fprintf(stderr, "GeometryManager: failed to create BLAS buffer for mesh %llu\n",
                             static_cast<unsigned long long>(info.mesh_id.value));
                return false;
            }

            VkAccelerationStructureCreateInfoKHR as_ci{};
            as_ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
            as_ci.buffer = entry.uncompacted_buffer.Handle();
            as_ci.size = info.sizes.accelerationStructureSize;
            as_ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

            VkResult result = dispatch_->vkCreateAccelerationStructureKHR(
                device_, &as_ci, nullptr, &entry.uncompacted_handle);
            if (result != VK_SUCCESS) {
                std::fprintf(stderr, "GeometryManager: failed to create BLAS (VkResult: %d)\n",
                             result);
                return false;
            }

            info.build_info.dstAccelerationStructure = entry.uncompacted_handle;
            info.build_info.scratchData.deviceAddress = scratch_address;

            const VkAccelerationStructureBuildRangeInfoKHR* range_ptr = &info.range_info;
            dispatch_->vkCmdBuildAccelerationStructuresKHR(cmd, 1, &info.build_info, &range_ptr);

            // Use uncompacted BLAS as the active reference so TLAS can use
            // new geometry immediately (compaction replaces it next frame)
            VkAccelerationStructureDeviceAddressInfoKHR addr_info{};
            addr_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
            addr_info.accelerationStructure = entry.uncompacted_handle;
            entry.device_address = dispatch_->vkGetAccelerationStructureDeviceAddressKHR(device_, &addr_info);

            entry.query_index = next_query_index_++;
            entry.state = BlasState::kPendingCompaction;
            query_handles.push_back(entry.uncompacted_handle);
        }

        // Barrier between sequential builds (reuse scratch)
        VkMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &barrier;
        dispatch_->vkCmdPipelineBarrier2(cmd, &dep);
    }

    // Write compaction size queries for new builds.
    // query_handles is in query_index order (collected during the build loop)
    // so query result i corresponds to the entry with query_index == i.
    if (!query_handles.empty()) {
        dispatch_->vkCmdWriteAccelerationStructuresPropertiesKHR(
            cmd,
            static_cast<uint32_t>(query_handles.size()),
            query_handles.data(),
            VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
            query_pool_, 0);
    }

    tlas_force_rebuild_ = true;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// BLAS compaction (called after previous frame's fence has signaled)
// ─────────────────────────────────────────────────────────────────────────────

bool GeometryManager::CompactPendingBlas(VkCommandBuffer cmd) {
    // Tick down deferred destroy batches and release those that have aged out
    std::erase_if(deferred_destroys_, [this](DeferredDestroyBatch& batch) {
        if (batch.frames_remaining > 0) {
            --batch.frames_remaining;
            return false;
        }
        for (auto& res : batch.resources) {
            if (res.handle != VK_NULL_HANDLE)
                dispatch_->vkDestroyAccelerationStructureKHR(device_, res.handle, nullptr);
        }
        return true;
    });

    // Collect entries pending compaction
    std::vector<BlasEntry*> pending;
    for (auto& [id, entry] : blas_map_) {
        if (entry.state == BlasState::kPendingCompaction)
            pending.push_back(&entry);
    }

    if (pending.empty()) return true;

    // Sort by query index for consistent readback
    std::ranges::sort(pending, {}, &BlasEntry::query_index);

    // Read compaction sizes
    auto count = static_cast<uint32_t>(pending.size());
    std::vector<VkDeviceSize> compacted_sizes(count);
    VkResult result = dispatch_->vkGetQueryPoolResults(
        device_, query_pool_, 0, count,
        count * sizeof(VkDeviceSize), compacted_sizes.data(),
        sizeof(VkDeviceSize), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GeometryManager: failed to read compaction sizes (VkResult: %d)\n",
                     result);
        return false;
    }

    VkDeviceSize total_uncompacted = 0;
    VkDeviceSize total_compacted = 0;

    for (uint32_t i = 0; i < count; ++i) {
        auto& entry = *pending[i];
        VkDeviceSize compacted_size = compacted_sizes[i];

        total_uncompacted += entry.uncompacted_buffer.Size();
        total_compacted += compacted_size;

        // Destroy old compacted BLAS if this is a rebuild
        if (entry.handle != VK_NULL_HANDLE) {
            dispatch_->vkDestroyAccelerationStructureKHR(device_, entry.handle, nullptr);
            entry.handle = VK_NULL_HANDLE;
            entry.buffer.Destroy();
        }

        // Create compacted BLAS
        if (!entry.buffer.Create(
                allocator_, compacted_size,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY)) {
            std::fprintf(stderr, "GeometryManager: failed to create compacted BLAS buffer\n");
            return false;
        }

        VkAccelerationStructureCreateInfoKHR as_ci{};
        as_ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        as_ci.buffer = entry.buffer.Handle();
        as_ci.size = compacted_size;
        as_ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

        result = dispatch_->vkCreateAccelerationStructureKHR(device_, &as_ci, nullptr, &entry.handle);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr, "GeometryManager: failed to create compacted BLAS (VkResult: %d)\n",
                         result);
            return false;
        }

        // Record copy from uncompacted to compacted
        VkCopyAccelerationStructureInfoKHR copy_info{};
        copy_info.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
        copy_info.src = entry.uncompacted_handle;
        copy_info.dst = entry.handle;
        copy_info.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
        dispatch_->vkCmdCopyAccelerationStructureKHR(cmd, &copy_info);
    }

    // Barrier: ensure compaction copies complete before reads
    {
        VkMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &barrier;
        dispatch_->vkCmdPipelineBarrier2(cmd, &dep);
    }

    // Update device addresses. Uncompacted resources must survive until the
    // command buffer completes — queue them for deferred destruction.
    DeferredDestroyBatch destroy_batch;
    for (auto* entry_ptr : pending) {
        auto& entry = *entry_ptr;

        VkAccelerationStructureDeviceAddressInfoKHR addr_info{};
        addr_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addr_info.accelerationStructure = entry.handle;
        entry.device_address = dispatch_->vkGetAccelerationStructureDeviceAddressKHR(device_, &addr_info);

        if (entry.uncompacted_handle != VK_NULL_HANDLE) {
            destroy_batch.resources.push_back({
                std::move(entry.uncompacted_buffer),
                entry.uncompacted_handle,
            });
            entry.uncompacted_handle = VK_NULL_HANDLE;
        }

        entry.state = BlasState::kReady;
    }
    if (!destroy_batch.resources.empty())
        deferred_destroys_.push_back(std::move(destroy_batch));

    tlas_force_rebuild_ = true;

    std::printf("BLAS compaction: %u structures, %llu -> %llu bytes (%.1f%% savings)\n",
                count,
                static_cast<unsigned long long>(total_uncompacted),
                static_cast<unsigned long long>(total_compacted),
                total_uncompacted > 0
                    ? 100.0 * (1.0 - static_cast<double>(total_compacted) /
                                      static_cast<double>(total_uncompacted))
                    : 0.0);

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Cleanup: destroy uncompacted BLAS resources after command buffer completion
// ─────────────────────────────────────────────────────────────────────────────

void GeometryManager::CleanupRemovedMeshes(const Scene& scene) {
    const auto& meshes = scene.Meshes();

    std::vector<MeshId> to_remove;
    for (const auto& [mesh_id, entry] : blas_map_) {
        bool found = std::ranges::any_of(meshes, [&](const Mesh& m) {
            return m.id == mesh_id;
        });
        if (!found) to_remove.push_back(mesh_id);
    }

    for (auto mesh_id : to_remove) {
        auto it = blas_map_.find(mesh_id);
        if (it == blas_map_.end()) continue;
        DestroyBlas(it->second);
        blas_map_.erase(it);
        tlas_force_rebuild_ = true;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TLAS building
// ─────────────────────────────────────────────────────────────────────────────

bool GeometryManager::BuildTlas(VkCommandBuffer cmd, const Scene& scene,
                                GpuScene& gpu_scene) {
    uint64_t current_gen = scene.TlasGeneration();
    if (current_gen == cached_tlas_generation_ && !tlas_force_rebuild_)
        return true;

    const auto& nodes = scene.Nodes();

    std::vector<VkAccelerationStructureInstanceKHR> instances;
    instances.reserve(nodes.size());

    for (const auto& node : nodes) {
        if (!node.visible) continue;

        auto blas_it = blas_map_.find(node.mesh_id);
        if (blas_it == blas_map_.end()) continue;
        if (blas_it->second.device_address == 0) continue;

        uint32_t mesh_addr_idx = gpu_scene.GetMeshAddressIndex(node.mesh_id);
        uint32_t material_idx = gpu_scene.GetMaterialIndex(node.material_id);

        VkAccelerationStructureInstanceKHR instance{};
        instance.transform = ToVkTransformMatrix(node.transform.ToMatrix());
        instance.instanceCustomIndex = EncodeCustomIndex(mesh_addr_idx, material_idx);
        instance.mask = 0xFF;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

        // Skip any-hit shader for non-mask materials (only kMask needs it)
        const auto* mat = scene.GetMaterial(node.material_id);
        if (!mat || mat->alpha_mode != MaterialDesc::AlphaMode::kMask)
            instance.flags |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;

        instance.accelerationStructureReference = blas_it->second.device_address;

        instances.push_back(instance);
    }

    tlas_instance_count_ = static_cast<uint32_t>(instances.size());

    // Instance buffer: host-visible for direct memcpy
    VkDeviceSize instance_size = sizeof(VkAccelerationStructureInstanceKHR) *
                                 std::max(tlas_instance_count_, 1u);

    if (instance_buffer_.Handle() == VK_NULL_HANDLE ||
        instance_buffer_.Size() < instance_size) {
        instance_buffer_.Destroy();
        if (!instance_buffer_.Create(
                allocator_, instance_size,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT)) {
            std::fprintf(stderr, "GeometryManager: failed to create TLAS instance buffer\n");
            return false;
        }
    }

    if (!instances.empty()) {
        void* mapped = instance_buffer_.Map();
        if (!mapped) return false;
        std::memcpy(mapped, instances.data(),
                    instances.size() * sizeof(VkAccelerationStructureInstanceKHR));
        instance_buffer_.Unmap();
    }

    // TLAS geometry
    VkAccelerationStructureGeometryKHR tlas_geometry{};
    tlas_geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlas_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlas_geometry.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlas_geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    tlas_geometry.geometry.instances.data.deviceAddress =
        instance_buffer_.DeviceAddress(device_, *dispatch_);

    VkAccelerationStructureBuildGeometryInfoKHR build_info{};
    build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = 1;
    build_info.pGeometries = &tlas_geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    dispatch_->vkGetAccelerationStructureBuildSizesKHR(
        device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info, &tlas_instance_count_, &sizes);

    // Allocate or reallocate TLAS buffer if needed
    if (tlas_buffer_.Handle() == VK_NULL_HANDLE ||
        tlas_buffer_.Size() < sizes.accelerationStructureSize) {
        if (tlas_ != VK_NULL_HANDLE) {
            dispatch_->vkDestroyAccelerationStructureKHR(device_, tlas_, nullptr);
            tlas_ = VK_NULL_HANDLE;
        }
        tlas_buffer_.Destroy();

        if (!tlas_buffer_.Create(
                allocator_, sizes.accelerationStructureSize,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY)) {
            std::fprintf(stderr, "GeometryManager: failed to create TLAS buffer\n");
            return false;
        }

        VkAccelerationStructureCreateInfoKHR as_ci{};
        as_ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        as_ci.buffer = tlas_buffer_.Handle();
        as_ci.size = sizes.accelerationStructureSize;
        as_ci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

        VkResult result = dispatch_->vkCreateAccelerationStructureKHR(device_, &as_ci, nullptr, &tlas_);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr, "GeometryManager: failed to create TLAS (VkResult: %d)\n",
                         result);
            return false;
        }
    }

    // Ensure scratch buffer is large enough for TLAS
    if (!EnsureScratchBuffer(sizes.buildScratchSize))
        return false;

    build_info.dstAccelerationStructure = tlas_;
    build_info.scratchData.deviceAddress = scratch_buffer_.DeviceAddress(device_, *dispatch_);

    VkAccelerationStructureBuildRangeInfoKHR range_info{};
    range_info.primitiveCount = tlas_instance_count_;
    const VkAccelerationStructureBuildRangeInfoKHR* range_ptr = &range_info;

    dispatch_->vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_ptr);

    // Barrier: TLAS build must complete before ray tracing
    {
        VkMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &barrier;
        dispatch_->vkCmdPipelineBarrier2(cmd, &dep);
    }

    cached_tlas_generation_ = current_gen;
    tlas_force_rebuild_ = false;
    return true;
}

void GeometryManager::NotifyMeshDeformed(MeshId mesh, bool topology_changed) {
    auto it = blas_map_.find(mesh);
    if (it == blas_map_.end()) return;

    if (topology_changed)
        it->second.state = BlasState::kNeedsRebuild;
    else if (it->second.state == BlasState::kReady)
        it->second.state = BlasState::kNeedsRefit;
}

}  // namespace monti::vulkan
