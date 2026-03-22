#pragma once
#include "Types.h"
#include <glm/glm.hpp>

namespace monti {

struct EnvironmentLight {
    TextureId hdr_lat_long;       // HDR equirectangular map
    float     intensity  = 1.0f;
    float     rotation   = 0.0f;  // Radians around Y axis
};

// Quad area light — a planar rectangle defined by a corner and two edge vectors.
// Emits light from the front face (determined by cross(edge_a, edge_b) normal).
struct AreaLight {
    glm::vec3 corner   = {0, 0, 0};   // World-space corner position
    glm::vec3 edge_a   = {1, 0, 0};   // First edge from corner
    glm::vec3 edge_b   = {0, 0, 1};   // Second edge from corner
    glm::vec3 radiance = {1, 1, 1};   // Emitted radiance (linear HDR)
    bool      two_sided = false;       // Emit from both faces
};

// Spherical area light — a sphere with uniform emission in all directions.
struct SphereLight {
    glm::vec3 center   = {0, 0, 0};
    float     radius   = 0.5f;
    glm::vec3 radiance = {1, 1, 1};
};

// Triangle light — an emissive triangle defined by three vertices.
struct TriangleLight {
    glm::vec3 v0       = {0, 0, 0};
    glm::vec3 v1       = {1, 0, 0};
    glm::vec3 v2       = {0, 1, 0};
    glm::vec3 radiance = {1, 1, 1};
    bool      two_sided = false;
};

} // namespace monti
