#pragma once

#include "../core/vulkan_context.h"
#include "../core/GBufferImages.h"

#include <monti/capture/GpuReadback.h>
#include <monti/capture/Writer.h>
#include <monti/vulkan/Renderer.h>
#include <monti/scene/Scene.h>

#include <cstdint>
#include <string>

namespace monti::app::datagen {

struct GenerationConfig {
    uint32_t width = 960;
    uint32_t height = 540;
    uint32_t spp = 4;              // Noisy samples per pixel per frame
    uint32_t ref_frames = 64;      // Frames to accumulate for reference
    float exposure = 0.0f;         // EV100
    std::string output_dir = "./capture/";
};

class GenerationSession {
public:
    GenerationSession(VulkanContext& ctx,
                      vulkan::Renderer& renderer,
                      GBufferImages& gbuffer,
                      capture::Writer& writer,
                      const GenerationConfig& config);
    ~GenerationSession();

    // Run the full generation loop. Returns true on success.
    bool Run();

private:
    // Render noisy frame and read back all G-buffer channels.
    bool RenderAndReadbackNoisy(uint32_t frame_index);

    // Render reference frames via multi-frame accumulation.
    bool RenderReference(uint32_t base_frame_index);

    // Pack readback data and write to EXR.
    bool WriteFrame(uint32_t frame_index);

    VulkanContext& ctx_;
    vulkan::Renderer& renderer_;
    GBufferImages& gbuffer_;
    capture::Writer& writer_;
    GenerationConfig config_;

    // Readback context for GPU→CPU copies
    capture::ReadbackContext readback_ctx_{};
    VkCommandPool readback_pool_ = VK_NULL_HANDLE;

    // CPU-side readback storage (populated per frame)
    std::vector<uint16_t> noisy_diffuse_raw_;   // RGBA16F
    std::vector<uint16_t> noisy_specular_raw_;  // RGBA16F
    std::vector<uint16_t> world_normals_raw_;   // RGBA16F
    std::vector<uint16_t> motion_vectors_raw_;  // RG16F
    std::vector<uint16_t> linear_depth_raw_;    // RG16F
    std::vector<uint32_t> diffuse_albedo_raw_;  // B10G11R11
    std::vector<uint32_t> specular_albedo_raw_; // B10G11R11

    // Reference accumulation result
    capture::MultiFrameResult ref_result_;
};

}  // namespace monti::app::datagen
