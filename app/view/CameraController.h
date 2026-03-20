#pragma once

#include <monti/scene/Camera.h>

#include <glm/glm.hpp>

union SDL_Event;

namespace monti::app {

enum class CameraMode { kFly, kOrbit };

struct SavedViewpoint {
    glm::vec3 position;
    glm::vec3 target;
    float fov_degrees;
};

// Interactive camera controller supporting fly (WASD + mouse look) and orbit
// (left-drag orbit, middle-drag pan, wheel zoom) modes.
class CameraController {
public:
    CameraController() = default;

    // Initialize from an auto-fit camera result and scene bounding box diagonal.
    void Initialize(const CameraParams& initial_camera, float scene_diagonal);

    // Returns true if the event was consumed.
    bool ProcessEvent(const SDL_Event& event);

    // Apply movement for this frame. Returns updated camera params.
    CameraParams Update(float dt);

    // Reset camera to the initial auto-fit position.
    void ResetToFit();

    CameraMode Mode() const { return mode_; }
    float Fov() const { return fov_; }

    // Extract the current camera state as a serializable viewpoint.
    SavedViewpoint CurrentViewpoint() const;

private:
    void OnKeyDown(const SDL_Event& event);
    void OnKeyUp(const SDL_Event& event);
    void OnMouseButtonDown(const SDL_Event& event);
    void OnMouseButtonUp(const SDL_Event& event);
    void OnMouseMotion(const SDL_Event& event);
    void OnMouseWheel(const SDL_Event& event);

    CameraMode mode_ = CameraMode::kFly;

    // Shared state
    glm::vec3 position_{0.0f};
    float yaw_ = 0.0f;    // Radians
    float pitch_ = 0.0f;  // Radians
    float fov_ = 1.047197f;
    float move_speed_ = 5.0f;
    float scene_diagonal_ = 10.0f;

    // Fly mode movement keys
    bool move_forward_ = false;
    bool move_back_ = false;
    bool move_left_ = false;
    bool move_right_ = false;
    bool move_up_ = false;
    bool move_down_ = false;
    bool fast_ = false;

    // Mouse state
    bool right_mouse_down_ = false;
    bool left_mouse_down_ = false;
    bool middle_mouse_down_ = false;

    // Orbit mode state
    glm::vec3 orbit_target_{0.0f};
    float orbit_distance_ = 5.0f;

    // Initial state for ResetToFit
    CameraParams initial_camera_{};
};

}  // namespace monti::app
