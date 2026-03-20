#include "CameraController.h"

#include <glm/gtc/constants.hpp>

#include <SDL3/SDL_events.h>

#include <algorithm>
#include <cmath>

namespace monti::app {

namespace {

constexpr float kLookSensitivity = 0.003f;
constexpr float kOrbitSensitivity = 0.005f;
constexpr float kPanSensitivity = 0.002f;
constexpr float kZoomSensitivity = 0.15f;
constexpr float kSpeedScrollFactor = 1.15f;
constexpr float kFastMultiplier = 3.0f;
constexpr float kMinPitch = -glm::half_pi<float>() + 0.01f;
constexpr float kMaxPitch = glm::half_pi<float>() - 0.01f;
constexpr float kMinOrbitDistance = 0.01f;

glm::vec3 ForwardFromYawPitch(float yaw, float pitch) {
    return {
        std::cos(pitch) * std::sin(yaw),
        std::sin(pitch),
        std::cos(pitch) * std::cos(yaw)
    };
}

}  // namespace

void CameraController::Initialize(const CameraParams& initial_camera, float scene_diagonal) {
    initial_camera_ = initial_camera;
    scene_diagonal_ = scene_diagonal;
    position_ = initial_camera.position;
    fov_ = initial_camera.vertical_fov_radians;

    // Derive yaw/pitch from camera direction
    glm::vec3 dir = glm::normalize(initial_camera.target - initial_camera.position);
    yaw_ = std::atan2(dir.x, dir.z);
    pitch_ = std::asin(std::clamp(dir.y, -1.0f, 1.0f));

    // Orbit state
    orbit_target_ = initial_camera.target;
    orbit_distance_ = glm::length(initial_camera.target - initial_camera.position);

    // Scale movement speed with scene size
    move_speed_ = scene_diagonal * 0.5f;
}

bool CameraController::ProcessEvent(const SDL_Event& event) {
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN:
        OnKeyDown(event);
        return true;
    case SDL_EVENT_KEY_UP:
        OnKeyUp(event);
        return true;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
        OnMouseButtonDown(event);
        return right_mouse_down_ || left_mouse_down_ || middle_mouse_down_;
    case SDL_EVENT_MOUSE_BUTTON_UP:
        OnMouseButtonUp(event);
        return false;
    case SDL_EVENT_MOUSE_MOTION:
        OnMouseMotion(event);
        return right_mouse_down_ || left_mouse_down_ || middle_mouse_down_;
    case SDL_EVENT_MOUSE_WHEEL:
        OnMouseWheel(event);
        return true;
    default:
        return false;
    }
}

CameraParams CameraController::Update(float dt) {
    if (mode_ == CameraMode::kFly) {
        glm::vec3 forward = ForwardFromYawPitch(yaw_, pitch_);
        glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
        glm::vec3 up(0.0f, 1.0f, 0.0f);

        float speed = move_speed_ * (fast_ ? kFastMultiplier : 1.0f);
        glm::vec3 velocity(0.0f);

        if (move_forward_) velocity += forward;
        if (move_back_) velocity -= forward;
        if (move_right_) velocity += right;
        if (move_left_) velocity -= right;
        if (move_up_) velocity += up;
        if (move_down_) velocity -= up;

        if (glm::length(velocity) > 0.0f)
            position_ += glm::normalize(velocity) * speed * dt;

        CameraParams cam{};
        cam.position = position_;
        cam.target = position_ + forward;
        cam.up = up;
        cam.vertical_fov_radians = fov_;
        cam.near_plane = initial_camera_.near_plane;
        cam.far_plane = initial_camera_.far_plane;
        cam.exposure_ev100 = initial_camera_.exposure_ev100;
        return cam;
    }

    // Orbit mode: recompute position from orbit target + distance + angles
    glm::vec3 offset = -ForwardFromYawPitch(yaw_, pitch_) * orbit_distance_;
    position_ = orbit_target_ + offset;

    CameraParams cam{};
    cam.position = position_;
    cam.target = orbit_target_;
    cam.up = {0.0f, 1.0f, 0.0f};
    cam.vertical_fov_radians = fov_;
    cam.near_plane = initial_camera_.near_plane;
    cam.far_plane = initial_camera_.far_plane;
    cam.exposure_ev100 = initial_camera_.exposure_ev100;
    return cam;
}

void CameraController::ResetToFit() {
    Initialize(initial_camera_, scene_diagonal_);
}

SavedViewpoint CameraController::CurrentViewpoint() const {
    SavedViewpoint vp{};
    vp.position = position_;
    vp.fov_degrees = glm::degrees(fov_);
    if (mode_ == CameraMode::kOrbit) {
        vp.target = orbit_target_;
    } else {
        glm::vec3 forward = ForwardFromYawPitch(yaw_, pitch_);
        vp.target = position_ + forward * orbit_distance_;
    }
    return vp;
}

void CameraController::OnKeyDown(const SDL_Event& event) {
    if (event.key.repeat) return;

    switch (event.key.key) {
    case SDLK_W: move_forward_ = true; break;
    case SDLK_S: move_back_ = true; break;
    case SDLK_A: move_left_ = true; break;
    case SDLK_D: move_right_ = true; break;
    case SDLK_Q: move_down_ = true; break;
    case SDLK_E: move_up_ = true; break;
    case SDLK_LSHIFT: fast_ = true; break;
    case SDLK_RSHIFT: fast_ = true; break;
    case SDLK_O:
        mode_ = (mode_ == CameraMode::kFly) ? CameraMode::kOrbit : CameraMode::kFly;
        if (mode_ == CameraMode::kOrbit) {
            // Set orbit target along current view direction
            glm::vec3 forward = ForwardFromYawPitch(yaw_, pitch_);
            orbit_target_ = position_ + forward * orbit_distance_;
        }
        break;
    case SDLK_F:
        ResetToFit();
        break;
    default: break;
    }
}

void CameraController::OnKeyUp(const SDL_Event& event) {
    switch (event.key.key) {
    case SDLK_W: move_forward_ = false; break;
    case SDLK_S: move_back_ = false; break;
    case SDLK_A: move_left_ = false; break;
    case SDLK_D: move_right_ = false; break;
    case SDLK_Q: move_down_ = false; break;
    case SDLK_E: move_up_ = false; break;
    case SDLK_LSHIFT: fast_ = false; break;
    case SDLK_RSHIFT: fast_ = false; break;
    default: break;
    }
}

void CameraController::OnMouseButtonDown(const SDL_Event& event) {
    if (event.button.button == SDL_BUTTON_RIGHT)
        right_mouse_down_ = true;
    else if (event.button.button == SDL_BUTTON_LEFT)
        left_mouse_down_ = true;
    else if (event.button.button == SDL_BUTTON_MIDDLE)
        middle_mouse_down_ = true;
}

void CameraController::OnMouseButtonUp(const SDL_Event& event) {
    if (event.button.button == SDL_BUTTON_RIGHT)
        right_mouse_down_ = false;
    else if (event.button.button == SDL_BUTTON_LEFT)
        left_mouse_down_ = false;
    else if (event.button.button == SDL_BUTTON_MIDDLE)
        middle_mouse_down_ = false;
}

void CameraController::OnMouseMotion(const SDL_Event& event) {
    if (mode_ == CameraMode::kFly) {
        if (right_mouse_down_) {
            yaw_ -= event.motion.xrel * kLookSensitivity;
            pitch_ -= event.motion.yrel * kLookSensitivity;
            pitch_ = std::clamp(pitch_, kMinPitch, kMaxPitch);
        }
    } else {
        // Orbit mode
        if (left_mouse_down_) {
            yaw_ -= event.motion.xrel * kOrbitSensitivity;
            pitch_ -= event.motion.yrel * kOrbitSensitivity;
            pitch_ = std::clamp(pitch_, kMinPitch, kMaxPitch);
        } else if (middle_mouse_down_) {
            glm::vec3 forward = ForwardFromYawPitch(yaw_, pitch_);
            glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
            glm::vec3 up = glm::normalize(glm::cross(right, forward));
            float pan_scale = orbit_distance_ * kPanSensitivity;
            orbit_target_ -= right * event.motion.xrel * pan_scale;
            orbit_target_ += up * event.motion.yrel * pan_scale;
        }
    }
}

void CameraController::OnMouseWheel(const SDL_Event& event) {
    if (mode_ == CameraMode::kFly) {
        // Scroll adjusts movement speed
        if (event.wheel.y > 0)
            move_speed_ *= kSpeedScrollFactor;
        else if (event.wheel.y < 0)
            move_speed_ /= kSpeedScrollFactor;
        move_speed_ = std::clamp(move_speed_, 0.01f, scene_diagonal_ * 10.0f);
    } else {
        // Orbit mode: scroll zooms in/out
        orbit_distance_ -= event.wheel.y * orbit_distance_ * kZoomSensitivity;
        orbit_distance_ = std::max(orbit_distance_, kMinOrbitDistance);
    }
}

}  // namespace monti::app
