#pragma once
#include "Types.h"
#include <glm/glm.hpp>

namespace monti {

struct EnvironmentLight {
    TextureId hdr_lat_long;       // HDR equirectangular map
    float     intensity  = 1.0f;
    float     rotation   = 0.0f;  // Radians around Y axis
};

} // namespace monti
