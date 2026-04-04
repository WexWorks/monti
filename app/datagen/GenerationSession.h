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

enum class WriteResult { kSuccess, kSkippedBlack, kSkippedNaN, kError };

struct SkipEntry {
    std::string viewpoint_id;  // e.g. "a1b2c3d4_0003" from path_id+frame, or legacy "id"
    std::string reason;        // "near_black" or "excessive_nan"
    float detail;              // L_avg for near_black, nan_fraction for NaN
};

struct ViewpointEntry {
    glm::vec3 position;
    glm::vec3 target;
    float fov_degrees = kDefaultFovDegrees;
    std::string id;  // Viewpoint identifier (from JSON "id" field)
    std::string path_id;  // Camera path group ID (8-hex string); empty for legacy/CLI viewpoints
    int frame = 0;        // 0-indexed frame within the path
    std::optional<std::string> environment;
    std::optional<float> environment_blur;
    std::optional<float> environment_intensity;
    std::optional<float> environment_rotation;  // Radians around Y axis
    std::optional<glm::vec3> camera_up;         // Camera up vector (default: world Y)
};

struct GenerationConfig {
    uint32_t width = 960;
    uint32_t height = 540;
    uint32_t spp = 4;              // Noisy samples per pixel per frame
    uint32_t ref_spp = 0;          // Reference SPP per frame (0 = use spp)
    uint32_t ref_frames = 64;      // Frames to accumulate for reference
    std::string output_dir = "./capture/";
    std::string capture_shader_dir;  // SPIR-V dir for capture shaders (accumulate.comp)
    std::string scene_name;          // Scene filename stem (for skip reports)
    std::string skipped_path;        // Optional: write skipped viewpoints JSON here
    float nan_threshold = 0.001f;    // Max NaN fraction before skip (0.001 = 0.1%)
    float black_threshold = 0.00005f; // Max log-average luminance before skip
    float default_env_blur = 3.5f;   // CLI default blur level for viewpoints without environmentBlur
    bool force_write = false;        // Write EXR even when skip checks fail
    bool adaptive_sampling = false;          // Enable adaptive reference sampling
    uint32_t convergence_check_interval = 4; // Frames between convergence checks
    uint32_t min_convergence_frames = 16;    // Minimum frames before a pixel can converge
    float convergence_threshold = 0.02f;     // Relative standard error threshold
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

    // Viewpoints skipped during Run() (near-black, excessive NaN).
    const std::vector<SkipEntry>& SkippedViewpoints() const { return skipped_viewpoints_; }

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
        std::vector<uint16_t> diffuse_albedo_raw;
        std::vector<uint16_t> specular_albedo_raw;
        capture::MultiFrameResult ref_result;
        std::string subdirectory;
        uint32_t index;
        uint32_t width;
        uint32_t height;
    };

    // Pack readback data and write to EXR from a self-contained job.
    WriteResult WriteFrameFromJob(WriteJob& job);

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
    std::vector<uint16_t> diffuse_albedo_raw_;  // RGBA16F
    std::vector<uint16_t> specular_albedo_raw_; // RGBA16F

    // GPU accumulator for reference frame rendering
    std::unique_ptr<capture::GpuAccumulator> accumulator_;

    // Reference accumulation result
    capture::MultiFrameResult ref_result_;

    // Adaptive sampling stats updated by RenderReference()
    uint32_t last_ref_frames_rendered_ = 0;
    uint32_t last_converged_pixel_count_ = 0;
    uint64_t last_actual_pixel_frames_ = 0;

    // Async EXR write future (at most one in-flight write at a time)
    std::future<WriteResult> write_future_;

    // Per-viewpoint structured timing data
    std::vector<nlohmann::json> viewpoint_timings_;

    // Viewpoints skipped due to validation failures
    std::vector<SkipEntry> skipped_viewpoints_;
};

}  // namespace monti::app::datagen
