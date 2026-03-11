#version 460
#extension GL_EXT_ray_tracing : require

// ── Ray payload ──────────────────────────────────────────────────
struct RayPayload {
    vec3 color;
    float hit_t;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;

// ── Push constants ───────────────────────────────────────────────
layout(push_constant) uniform PushConstants {
    mat4 inv_view;
    mat4 inv_proj;
    mat4 prev_view_proj;

    uint  frame_index;
    uint  paths_per_pixel;
    uint  max_bounces;
    uint  area_light_count;

    uint  env_width;
    uint  env_height;
    float env_avg_luminance;
    float env_max_luminance;

    float env_rotation;
    float skybox_mip_level;
    float jitter_x;
    float jitter_y;

    uint  debug_mode;
    uint  pad0;
} pc;

// ── Entry point ──────────────────────────────────────────────────
void main() {
    // Solid background color (dark blue sky)
    payload.color = vec3(0.1, 0.15, 0.3);
    payload.hit_t = -1.0;
}
