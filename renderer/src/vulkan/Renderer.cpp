#include <volk.h>

#include <monti/vulkan/Renderer.h>

#include "vulkan/GpuScene.h"
#include "vulkan/GeometryManager.h"
#include "vulkan/BlueNoise.h"
#include "vulkan/EnvironmentMap.h"
#include "vulkan/Buffer.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace monti::vulkan {

struct Renderer::Impl {
    RendererDesc desc{};
    monti::Scene* scene = nullptr;
    std::unique_ptr<GpuScene> gpu_scene;
    std::unique_ptr<GeometryManager> geometry_manager;
    EnvironmentMap environment_map;
    BlueNoise blue_noise;
    MeshCleanupCallback mesh_cleanup_callback;
    std::vector<Buffer> pending_staging;  // kept alive until next frame
    uint32_t samples_per_pixel = 4;
    bool scene_dirty = false;
    bool resources_initialized = false;

    void Init(const RendererDesc& d) {
        desc = d;
        samples_per_pixel = d.samples_per_pixel;
        gpu_scene = std::make_unique<GpuScene>(d.allocator, d.device, d.physical_device);
        geometry_manager = std::make_unique<GeometryManager>(d.allocator, d.device);
    }
};

Renderer::Renderer() : impl_(std::make_unique<Impl>()) {}

std::unique_ptr<Renderer> Renderer::Create(const RendererDesc& desc) {
    auto renderer = std::unique_ptr<Renderer>(new Renderer());
    renderer->impl_->Init(desc);
    return renderer;
}

Renderer::~Renderer() = default;

void Renderer::SetScene(monti::Scene* scene) {
    impl_->scene = scene;
    impl_->scene_dirty = true;
}

void Renderer::RegisterMeshBuffers(MeshId mesh, const MeshBufferBinding& binding) {
    impl_->gpu_scene->RegisterMeshBuffers(mesh, binding);
}

void Renderer::NotifyMeshDeformed(MeshId mesh, bool topology_changed) {
    impl_->geometry_manager->NotifyMeshDeformed(mesh, topology_changed);
}

void Renderer::SetMeshCleanupCallback(MeshCleanupCallback callback) {
    impl_->mesh_cleanup_callback = std::move(callback);
}

bool Renderer::RenderFrame(VkCommandBuffer cmd, const GBuffer& /*output*/,
                           uint32_t /*frame_index*/) {
    if (!impl_->scene) return false;

    // Release staging buffers from the previous frame (cmd has completed by now)
    impl_->pending_staging.clear();

    // One-time initialization: placeholders + blue noise
    if (!impl_->resources_initialized) {
        if (!impl_->environment_map.CreatePlaceholders(
                impl_->desc.allocator, impl_->desc.device, cmd,
                impl_->pending_staging)) {
            std::fprintf(stderr, "Renderer::RenderFrame env map placeholder failed\n");
            return false;
        }

        Buffer blue_noise_staging;
        if (!impl_->blue_noise.Generate(impl_->desc.allocator, cmd, blue_noise_staging)) {
            std::fprintf(stderr, "Renderer::RenderFrame blue noise generation failed\n");
            return false;
        }
        impl_->pending_staging.push_back(std::move(blue_noise_staging));

        impl_->resources_initialized = true;
    }

    // On first frame after SetScene(), upload materials and textures
    if (impl_->scene_dirty) {
        // Upload textures (records commands into cmd)
        auto staging = impl_->gpu_scene->UploadTextures(*impl_->scene, cmd);
        for (auto& s : staging)
            impl_->pending_staging.push_back(std::move(s));

        // Update materials (host-visible buffer, no cmd needed)
        if (!impl_->gpu_scene->UpdateMaterials(*impl_->scene)) {
            std::fprintf(stderr, "Renderer::RenderFrame material update failed\n");
            return false;
        }

        // Upload mesh address table (host-visible buffer, no cmd needed)
        impl_->gpu_scene->UploadMeshAddressTable();

        // Load environment map if one is set on the scene
        const auto* env_light = impl_->scene->GetEnvironmentLight();
        if (env_light) {
            const auto* tex = impl_->scene->GetTexture(env_light->hdr_lat_long);
            if (tex && !tex->data.empty()) {
                if (!impl_->environment_map.Load(
                        impl_->desc.allocator, impl_->desc.device, cmd,
                        reinterpret_cast<const float*>(tex->data.data()),
                        tex->width, tex->height,
                        impl_->pending_staging)) {
                    std::fprintf(stderr, "Renderer::RenderFrame env map load failed\n");
                    return false;
                }
            }
        }

        impl_->scene_dirty = false;
    }

    // Compact BLAS from previous frame (query results now available)
    if (!impl_->geometry_manager->CompactPendingBlas(cmd)) {
        std::fprintf(stderr, "Renderer::RenderFrame BLAS compaction failed\n");
        return false;
    }

    // Clean up BLAS for removed meshes
    impl_->geometry_manager->CleanupRemovedMeshes(*impl_->scene);

    // Build BLAS for new/dirty meshes
    if (!impl_->geometry_manager->BuildDirtyBlas(cmd, *impl_->gpu_scene)) {
        std::fprintf(stderr, "Renderer::RenderFrame BLAS build failed\n");
        return false;
    }

    // Build/rebuild TLAS
    if (!impl_->geometry_manager->BuildTlas(cmd, *impl_->scene, *impl_->gpu_scene)) {
        std::fprintf(stderr, "Renderer::RenderFrame TLAS build failed\n");
        return false;
    }

    return true;
}

void Renderer::SetSamplesPerPixel(uint32_t spp) {
    impl_->samples_per_pixel = spp;
}

uint32_t Renderer::GetSamplesPerPixel() const {
    return impl_->samples_per_pixel;
}

void Renderer::Resize(uint32_t /*width*/, uint32_t /*height*/) {
    // Will be implemented when render pipeline is built
}

float Renderer::LastFrameTimeMs() const {
    return 0.0f;
}

} // namespace monti::vulkan
