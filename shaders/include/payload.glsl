#ifndef PAYLOAD_GLSL
#define PAYLOAD_GLSL

struct HitPayload {
    vec3 hit_pos;
    float hit_t;
    vec3 normal;
    uint material_index;
    vec2 uv;
    bool missed;
    float _pad;
};

#endif // PAYLOAD_GLSL
