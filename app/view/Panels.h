#pragma once

#include "CameraController.h"

#include <deni/vulkan/Denoiser.h>

#include <string>

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

struct PanelState {
    // Render settings
    int spp = 4;
    int max_bounces = 8;
    float exposure_ev = 0.0f;
    bool auto_exposure = false;
    float auto_exposure_luminance = 0.0f;
    float env_rotation_degrees = 0.0f;
    float env_intensity = 1.0f;
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
    bool viewpoint_just_saved = false;
    float viewpoint_saved_timer = 0.0f;
    std::string viewpoints_out_path;

    // Denoiser
    deni::vulkan::DenoiserMode denoiser_mode = deni::vulkan::DenoiserMode::kPassthrough;
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
