#version 460
#extension GL_EXT_ray_tracing : require

// ── Ray payload ──────────────────────────────────────────────────
struct RayPayload {
    vec3 color;
    float hit_t;
    bool missed;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;

// ── Descriptor bindings used by closest hit ──────────────────────
// Binding 8: Mesh address table
layout(set = 0, binding = 8, std430) readonly buffer MeshAddressTable {
    uvec4 entries[];
} mesh_address_table;

// Binding 9: Material buffer
layout(set = 0, binding = 9, std430) readonly buffer MaterialBuffer {
    vec4 data[];
} materials;

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

// ── Hit attributes ───────────────────────────────────────────────
hitAttributeEXT vec2 hit_attribs;

// ── Entry point ──────────────────────────────────────────────────
void main() {
    // Barycentric coordinates as placeholder color
    vec3 bary = vec3(1.0 - hit_attribs.x - hit_attribs.y,
                     hit_attribs.x, hit_attribs.y);
    payload.color = bary;
    payload.hit_t = gl_HitTEXT;
    payload.missed = false;
}
