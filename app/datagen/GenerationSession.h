#pragma once

#include "../core/vulkan_context.h"
#include "../core/CameraSetup.h"
#include "../core/GBufferImages.h"

#include <monti/capture/GpuAccumulator.h>
#include <monti/capture/GpuReadback.h>
#include <monti/capture/Writer.h>
#include <monti/vulkan/Renderer.h>
#include <monti/scene/Scene.h>

#include <cstdint>
#include <future>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace monti::app::datagen {

struct ViewpointEntry {
    glm::vec3 position;
    glm::vec3 target;
    float fov_degrees = kDefaultFovDegrees;
    std::optional<float> exposure;
    std::optional<std::string> environment;
    std::optional<std::string> lights;
    std::optional<float> environment_blur;
    std::optional<float> environment_intensity;
};

struct GenerationConfig {
    uint32_t width = 960;
    uint32_t height = 540;
    uint32_t spp = 4;              // Noisy samples per pixel per frame
    uint32_t ref_frames = 64;      // Frames to accumulate for reference
    float exposure = 0.0f;         // EV100
    std::string output_dir = "./capture/";
    std::string capture_shader_dir;  // SPIR-V dir for capture shaders (accumulate.comp)
    std::vector<ViewpointEntry> viewpoints;
};

class GenerationSession {
public:
    GenerationSession(VulkanContext& ctx,
                      vulkan::Renderer& renderer,
                      GBufferImages& gbuffer,
                      capture::Writer& writer,
                      Scene& scene,
                      const GenerationConfig& config);
    ~GenerationSession();

    // Run the full generation loop. Returns true on success.
    bool Run();

    // Per-viewpoint structured timing data collected during Run().
    const std::vector<nlohmann::json>& ViewpointTimings() const { return viewpoint_timings_; }

private:
    // Render noisy frame and read back all G-buffer channels.
    bool RenderAndReadbackNoisy(uint32_t frame_index);

    // Render reference frames via multi-frame accumulation.
    bool RenderReference(uint32_t base_frame_index);

    // Self-contained write job: owns all data needed for EXR writing.
    struct WriteJob {
        std::vector<uint16_t> noisy_diffuse_raw;
        std::vector<uint16_t> noisy_specular_raw;
        std::vector<uint16_t> world_normals_raw;
        std::vector<uint16_t> motion_vectors_raw;
        std::vector<uint16_t> linear_depth_raw;
        std::vector<uint32_t> diffuse_albedo_raw;
        std::vector<uint32_t> specular_albedo_raw;
        capture::MultiFrameResult ref_result;
        std::string subdirectory;
        float exposure;
        uint32_t width;
        uint32_t height;
    };

    // Pack readback data and write to EXR from a self-contained job.
    bool WriteFrameFromJob(WriteJob& job);

    VulkanContext& ctx_;
    vulkan::Renderer& renderer_;
    GBufferImages& gbuffer_;
    capture::Writer& writer_;
    Scene& scene_;
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

    // GPU accumulator for reference frame rendering
    std::unique_ptr<capture::GpuAccumulator> accumulator_;

    // Reference accumulation result
    capture::MultiFrameResult ref_result_;

    // Async EXR write future (at most one in-flight write at a time)
    std::future<bool> write_future_;

    // Per-viewpoint structured timing data
    std::vector<nlohmann::json> viewpoint_timings_;
};

}  // namespace monti::app::datagen
