#include "Panels.h"

#include <cstdio>

#include <imgui.h>

namespace monti::app {

namespace {

const char* DebugModeLabel(DebugMode mode) {
    switch (mode) {
    case DebugMode::kOff:           return "Off";
    case DebugMode::kNormals:       return "Normals";
    case DebugMode::kAlbedo:        return "Albedo";
    case DebugMode::kDepth:         return "Depth";
    case DebugMode::kMotionVectors: return "Motion Vectors";
    case DebugMode::kNoisy:         return "Noisy";
    case DebugMode::kTransmissionNdotV: return "Transmission NdotV";
    case DebugMode::kPathLength:        return "Path Length";
    case DebugMode::kVolumeAttenuation: return "Volume Attenuation";
    case DebugMode::kAlphaMode:         return "Alpha Mode";
    case DebugMode::kTextureAlpha:      return "Texture Alpha";
    case DebugMode::kOpacity:           return "Opacity";
    case DebugMode::kNEEOnly:           return "NEE Only";
    case DebugMode::kBSDFMissOnly:      return "BSDF Miss Only";
    case DebugMode::kSingleBounce:      return "Single Bounce";
    case DebugMode::kEnvValue:          return "Env Value";
    default:                        return "Unknown";
    }
}

const char* CameraModeLabel(CameraMode mode) {
    switch (mode) {
    case CameraMode::kFly:   return "Fly";
    case CameraMode::kOrbit: return "Orbit";
    default:                 return "Unknown";
    }
}

const char* DenoiserModeLabel(deni::vulkan::DenoiserMode mode) {
    switch (mode) {
    case deni::vulkan::DenoiserMode::kMl:          return "ML";
    case deni::vulkan::DenoiserMode::kPassthrough:  return "Passthrough";
    default:                                        return "Unknown";
    }
}

}  // namespace

constexpr float kTopBarHeight = 30.0f;
constexpr float kSettingsPanelX = 10.0f;
constexpr float kSettingsPanelY = 40.0f;
constexpr float kSettingsPanelWidth = 320.0f;
constexpr int kMaxSppSlider = 64;
constexpr float kMinExposure = -10.0f;
constexpr float kMaxExposure = 10.0f;

void Panels::Draw(PanelState& state) {
    DrawTopBar(state);
    if (state.show_settings)
        DrawSettingsPanel(state);
}

void Panels::DrawTopBar(const PanelState& state) {
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoScrollWithMouse |
                             ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoBringToFrontOnFocus;

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, kTopBarHeight));
    ImGui::SetNextWindowBgAlpha(0.6f);

    if (ImGui::Begin("##TopBar", nullptr, flags)) {
        // Fixed-width right-aligned boxes for FPS and frame time
        constexpr float kFpsBoxWidth = 72.0f;
        constexpr float kMsBoxWidth = 80.0f;

        char fps_buf[32];
        std::snprintf(fps_buf, sizeof(fps_buf), "%.1f fps", state.fps);
        float fps_text_w = ImGui::CalcTextSize(fps_buf).x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + kFpsBoxWidth - fps_text_w);
        ImGui::TextUnformatted(fps_buf);

        ImGui::SameLine(0.0f, 0.0f);
        char ms_buf[32];
        std::snprintf(ms_buf, sizeof(ms_buf), "  %.2f ms", state.frame_time_ms);
        float ms_text_w = ImGui::CalcTextSize(ms_buf).x;
        float ms_start = ImGui::GetCursorPosX();
        ImGui::SetCursorPosX(ms_start + kMsBoxWidth - ms_text_w);
        ImGui::TextUnformatted(ms_buf);

        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        ImGui::Text("%s", state.scene_name.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        ImGui::Text("%s", CameraModeLabel(state.camera_mode));
        if (state.debug_mode != DebugMode::kOff) {
            ImGui::SameLine();
            ImGui::TextDisabled("|");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                               "Debug: %s", DebugModeLabel(state.debug_mode));
        }

        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        ImGui::Text("Denoise: %s", DenoiserModeLabel(state.denoiser_mode));

        // Path tracking indicator
        if (state.path_tracking.tracking_mode_enabled) {
            ImGui::SameLine();
            ImGui::TextDisabled("|");
            ImGui::SameLine();
            if (state.path_tracking.is_capturing) {
                ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f),
                                   "● REC  path %s  [%d frames]",
                                   state.path_tracking.current_path_id.c_str(),
                                   state.path_tracking.current_frame);
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f),
                                   "REC READY  [%d saved]",
                                   state.saved_viewpoint_count);
            }
        }
    }
    ImGui::End();
}

void Panels::DrawSettingsPanel(PanelState& state) {
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                             ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoSavedSettings;
    ImGui::SetNextWindowPos(ImVec2(kSettingsPanelX, kSettingsPanelY), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(kSettingsPanelWidth, 0.0f), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("##Settings", nullptr, flags)) {
        // Render
        ImGui::SliderInt("SPP", &state.spp, 1, kMaxSppSlider);
        ImGui::SliderInt("Max Bounces", &state.max_bounces, 1, 16);
        ImGui::Checkbox("Auto Exposure", &state.auto_exposure);
        if (state.auto_exposure)
            ImGui::Text("Avg Luminance: %.4f", state.auto_exposure_luminance);
        ImGui::SliderFloat("Exposure EV", &state.exposure_ev, kMinExposure, kMaxExposure, "%.1f");
        ImGui::SliderFloat("Env Intensity", &state.env_intensity, 0.0f, 20.0f, "%.1f");
        ImGui::SliderFloat("Env Rotation", &state.env_rotation_degrees,
                           0.0f, 360.0f, "%.0f deg");

        ImGui::Separator();

        // Denoiser
        {
            int mode = static_cast<int>(state.denoiser_mode);
            ImGui::RadioButton("Passthrough",
                               &mode,
                               static_cast<int>(deni::vulkan::DenoiserMode::kPassthrough));
            ImGui::SameLine();
            ImGui::BeginDisabled(!state.has_ml_model);
            ImGui::RadioButton("ML",
                               &mode,
                               static_cast<int>(deni::vulkan::DenoiserMode::kMl));
            ImGui::EndDisabled();
            state.denoiser_mode = static_cast<deni::vulkan::DenoiserMode>(mode);

            ImGui::Text("Record time: %.2f ms", state.denoiser_time_ms);
            if (state.has_ml_model)
                ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "ML model loaded");
            else
                ImGui::TextDisabled("No model — passthrough only");
        }

        ImGui::Separator();

        // Debug Visualization
        {
            int mode = static_cast<int>(state.debug_mode);
            if (ImGui::BeginCombo("Debug Mode",
                                  DebugModeLabel(state.debug_mode))) {
                for (int i = 0; i < static_cast<int>(DebugMode::kCount); ++i) {
                    bool selected = (mode == i);
                    if (ImGui::Selectable(DebugModeLabel(static_cast<DebugMode>(i)),
                                          selected))
                        mode = i;
                    if (selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            state.debug_mode = static_cast<DebugMode>(mode);
        }
    }
    ImGui::End();
}

}  // namespace monti::app
