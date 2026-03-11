#version 460
#extension GL_EXT_ray_tracing : require

// ── Ray payload ──────────────────────────────────────────────────
struct RayPayload {
    vec3 color;
    float hit_t;
    bool missed;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;

// ── Entry point ──────────────────────────────────────────────────
void main() {
    payload.missed = true;
}
