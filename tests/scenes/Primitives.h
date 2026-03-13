#pragma once

#include <monti/scene/Material.h>

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace monti::test {

// Generate an icosphere mesh by recursive subdivision of an icosahedron.
// center: world-space center; radius: sphere radius;
// subdivisions: recursion depth (0 = icosahedron, 1 = 42 verts, 2 = 162 verts, etc.)
MeshData MakeIcosphere(const glm::vec3& center, float radius,
                       uint32_t subdivisions = 2);

} // namespace monti::test
