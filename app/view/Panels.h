#pragma once

#include "CameraController.h"

#include <deni/vulkan/Denoiser.h>

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace monti::app {

enum class DebugMode : int {
    kOff = 0,
    kNormals,
    kAlbedo,
    kDepth,
    kMotionVectors,
    kNoisy,
    kTransmissionNdotV,
    kPathLength,
    kVolumeAttenuation,
    kAlphaMode,
    kTextureAlpha,
    kOpacity,
    kNEEOnly,
    kBSDFMissOnly,
    kSingleBounce,
    kEnvValue,
    kCount
};

struct PathTrackingState {
    bool tracking_mode_enabled = false;
    bool is_capturing = false;
    std::string current_path_id;
    int current_frame = 0;
    glm::vec3 last_position{0.0f};
    glm::vec3 last_target{0.0f};
    glm::vec3 last_up{0.0f, 1.0f, 0.0f};
    uint64_t last_motion_time = 0;          // SDL_GetPerformanceCounter() ticks at last recorded frame
    uint64_t last_capture_time = 0;         // rate-limit frame captures
    float capture_interval_sec = 0.1f;      // minimum seconds between captured frames (1/capture_fps)
    std::vector<nlohmann::json> buffered_frames;
    std::vector<std::string> flushed_path_ids;  // undo stack — Backspace pops from back
};

struct PanelState {
    // Render settings
    int spp = 4;
    int max_bounces = 8;
    float exposure_ev = 0.0f;
    bool auto_exposure = false;
    float auto_exposure_luminance = 0.0f;
    float env_rotation_degrees = 0.0f;
    float env_intensity = 1.0f;
    float env_blur = 3.5f;
    bool has_env_map = false;
    DebugMode debug_mode = DebugMode::kOff;

    // Settings panel visibility
    bool show_settings = false;

    // Camera mode (displayed in top bar)
    CameraMode camera_mode = CameraMode::kFly;

    // Scene file name
    std::string scene_name;

    // Frame timing (CPU-measured)
    float fps = 0.0f;
    float frame_time_ms = 0.0f;

    // Viewpoint capture
    int saved_viewpoint_count = 0;
    std::string viewpoints_out_path;
    std::string env_path;  // currently loaded env map path (empty = default grey)
    PathTrackingState path_tracking;

    // Denoiser
    deni::vulkan::DenoiserMode denoiser_mode = deni::vulkan::DenoiserMode::kPassthrough;
    deni::vulkan::MlDebugOutput ml_debug_output = deni::vulkan::MlDebugOutput::kNormal;
    bool has_ml_model = false;
    float denoiser_time_ms = 0.0f;
};

// Draws ImGui panels for the monti_view application.
class Panels {
public:
    void Draw(PanelState& state);

private:
    void DrawTopBar(const PanelState& state);
    void DrawSettingsPanel(PanelState& state);
};

}  // namespace monti::app
