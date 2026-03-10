#include <monti/vulkan/Renderer.h>

namespace monti::vulkan {

std::unique_ptr<Renderer> Renderer::Create(const RendererDesc& /*desc*/) {
    return std::unique_ptr<Renderer>(new Renderer());
}

Renderer::~Renderer() = default;

void Renderer::SetScene(monti::Scene* /*scene*/) {}

void Renderer::NotifyMeshDeformed(MeshId /*mesh*/, bool /*topology_changed*/) {}

void Renderer::SetMeshCleanupCallback(MeshCleanupCallback /*callback*/) {}

bool Renderer::RenderFrame(VkCommandBuffer /*cmd*/, const GBuffer& /*output*/,
                           uint32_t /*frame_index*/) {
    return true;
}

void Renderer::SetSamplesPerPixel(uint32_t /*spp*/) {}
uint32_t Renderer::GetSamplesPerPixel() const { return 4; }

void Renderer::Resize(uint32_t /*width*/, uint32_t /*height*/) {}

float Renderer::LastFrameTimeMs() const { return 0.0f; }

} // namespace monti::vulkan
