#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "include/payload.glsl"

layout(location = 0) rayPayloadInEXT HitPayload payload;

// ── Entry point ──────────────────────────────────────────────────
void main() {
    payload.missed = true;
}
