#include <deni/vulkan/Denoiser.h>

namespace deni::vulkan {

std::unique_ptr<Denoiser> Denoiser::Create(const DenoiserDesc& /*desc*/) {
    return std::unique_ptr<Denoiser>(new Denoiser());
}

Denoiser::~Denoiser() = default;

DenoiserOutput Denoiser::Denoise(VkCommandBuffer /*cmd*/,
                                 const DenoiserInput& /*input*/) {
    return {};
}

void Denoiser::Resize(uint32_t /*width*/, uint32_t /*height*/) {}

float Denoiser::LastPassTimeMs() const { return 0.0f; }

} // namespace deni::vulkan
