#pragma once

#include <glm/glm.hpp>

#include <cstdint>

namespace monti::vulkan {

struct FrameUniforms {
    glm::mat4 inv_view;              // 64 bytes, offset 0
    glm::mat4 inv_proj;              // 64 bytes, offset 64
    glm::mat4 prev_view_proj;        // 64 bytes, offset 128

    uint32_t env_width;              // 4 bytes, offset 192
    uint32_t env_height;             // 4 bytes, offset 196
    float    env_avg_luminance;      // 4 bytes, offset 200
    float    env_max_luminance;      // 4 bytes, offset 204

    float    env_rotation;           // 4 bytes, offset 208
    float    skybox_mip_level;       // 4 bytes, offset 212
    float    jitter_x;              // 4 bytes, offset 216
    float    jitter_y;              // 4 bytes, offset 220

    uint32_t light_count;            // 4 bytes, offset 224
    float    env_intensity;           // 4 bytes, offset 228
    float    bg_env_mip_level = 3.5f; // 4 bytes, offset 232 (mip level for env blur)
    uint32_t pad2;                   // 4 bytes, offset 236
};

static_assert(sizeof(FrameUniforms) == 240);
static_assert(sizeof(FrameUniforms) % 16 == 0, "std140 requires struct size multiple of 16");

}  // namespace monti::vulkan
