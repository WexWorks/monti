#pragma once

#include "CameraController.h"

#include <deni/vulkan/Denoiser.h>

#include <glm/glm.hpp>

#include <cstdint>
#include <string>

namespace monti::app {

enum class DebugMode : int {
    kOff = 0,
    kNormals,
    kAlbedo,
    kDepth,
    kMotionVectors,
    kNoisy,
    kCount
};

struct PanelState {
    // Render settings
    int spp = 4;
    int max_bounces = 4;
    float exposure_ev = 0.0f;
    float env_rotation_degrees = 0.0f;
    float env_intensity = 1.0f;
    DebugMode debug_mode = DebugMode::kOff;

    // Settings panel visibility
    bool show_settings = false;

    // Camera (read-only display)
    CameraMode camera_mode = CameraMode::kFly;
    glm::vec3 camera_position{0.0f};
    float camera_fov_degrees = 60.0f;

    // Scene info (read-only display)
    uint32_t node_count = 0;
    uint32_t mesh_count = 0;
    uint32_t material_count = 0;
    uint32_t triangle_count = 0;

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
