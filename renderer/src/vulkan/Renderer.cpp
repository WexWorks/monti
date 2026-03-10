#include <volk.h>

#include <monti/vulkan/Renderer.h>

#include "vulkan/GpuScene.h"
#include "vulkan/Buffer.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace monti::vulkan {

struct Renderer::Impl {
    RendererDesc desc{};
    monti::Scene* scene = nullptr;
    std::unique_ptr<GpuScene> gpu_scene;
    MeshCleanupCallback mesh_cleanup_callback;
    std::vector<Buffer> pending_staging;  // kept alive until next frame
    uint32_t samples_per_pixel = 4;
    bool scene_dirty = false;

    void Init(const RendererDesc& d) {
        desc = d;
        samples_per_pixel = d.samples_per_pixel;
        gpu_scene = std::make_unique<GpuScene>(d.allocator, d.device, d.physical_device);
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

void Renderer::NotifyMeshDeformed(MeshId /*mesh*/, bool /*topology_changed*/) {
    // Will be implemented in Phase 6 (GeometryManager)
}

void Renderer::SetMeshCleanupCallback(MeshCleanupCallback callback) {
    impl_->mesh_cleanup_callback = std::move(callback);
}

bool Renderer::RenderFrame(VkCommandBuffer cmd, const GBuffer& /*output*/,
                           uint32_t /*frame_index*/) {
    if (!impl_->scene) return false;

    // Release staging buffers from the previous frame (cmd has completed by now)
    impl_->pending_staging.clear();

    // On first frame after SetScene(), upload materials and textures
    if (impl_->scene_dirty) {
        // Upload textures (records commands into cmd)
        impl_->pending_staging = impl_->gpu_scene->UploadTextures(*impl_->scene, cmd);

        // Update materials (host-visible buffer, no cmd needed)
        if (!impl_->gpu_scene->UpdateMaterials(*impl_->scene)) {
            std::fprintf(stderr, "Renderer::RenderFrame material update failed\n");
            return false;
        }

        impl_->scene_dirty = false;
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
