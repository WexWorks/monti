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
    }
    ImGui::End();
}

void Panels::DrawSettingsPanel(PanelState& state) {
    ImGui::SetNextWindowPos(ImVec2(kSettingsPanelX, kSettingsPanelY), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(kSettingsPanelWidth, 0.0f), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Settings", &state.show_settings)) {
        if (ImGui::CollapsingHeader("Render", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderInt("SPP", &state.spp, 1, kMaxSppSlider);
            ImGui::SliderFloat("Exposure EV", &state.exposure_ev, kMinExposure, kMaxExposure, "%.1f");
            ImGui::SliderFloat("Env Rotation", &state.env_rotation_degrees,
                               0.0f, 360.0f, "%.0f deg");
        }

        if (ImGui::CollapsingHeader("Debug Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
            int mode = static_cast<int>(state.debug_mode);
            for (int i = 0; i < static_cast<int>(DebugMode::kCount); ++i) {
                ImGui::RadioButton(DebugModeLabel(static_cast<DebugMode>(i)), &mode, i);
                if (i < static_cast<int>(DebugMode::kCount) - 1)
                    ImGui::SameLine();
            }
            state.debug_mode = static_cast<DebugMode>(mode);
        }

        if (ImGui::CollapsingHeader("Camera")) {
            ImGui::Text("Mode: %s", CameraModeLabel(state.camera_mode));
            ImGui::Text("FOV: %.1f deg", state.camera_fov_degrees);
            ImGui::Text("Position: (%.2f, %.2f, %.2f)",
                        state.camera_position.x,
                        state.camera_position.y,
                        state.camera_position.z);
        }

        if (ImGui::CollapsingHeader("Scene Info")) {
            ImGui::Text("Nodes: %u", state.node_count);
            ImGui::Text("Meshes: %u", state.mesh_count);
            ImGui::Text("Materials: %u", state.material_count);
            ImGui::Text("Triangles: %u", state.triangle_count);
        }
    }
    ImGui::End();
}

}  // namespace monti::app
