#pragma once
#include <glm/glm.hpp>

namespace monti {

struct CameraParams {
    glm::vec3 position = {0, 0, 0};
    glm::vec3 target   = {0, 0, -1};
    glm::vec3 up       = {0, 1, 0};

    float vertical_fov_radians = 1.047197f;   // ~60 degrees
    float aspect_ratio         = 16.0f / 9.0f;
    float near_plane           = 0.1f;
    float far_plane            = 1000.0f;
    float aperture_radius      = 0.0f;        // 0 = pinhole
    float focus_distance       = 10.0f;
    float exposure_ev100       = 0.0f;
};

} // namespace monti
