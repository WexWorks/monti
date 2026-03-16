#ifndef UV_TRANSFORM_GLSL
#define UV_TRANSFORM_GLSL

// Apply KHR_texture_transform: uv' = R(rotation) * S(scale) * uv + offset
// Identity early-out when scale == (1,1) && offset == (0,0) && rotation == 0.
vec2 applyUvTransform(vec2 uv, vec2 offset, vec2 scale, float rotation) {
    if (scale == vec2(1.0, 1.0) && offset == vec2(0.0, 0.0) && rotation == 0.0)
        return uv;
    float c = cos(rotation);
    float s = sin(rotation);
    vec2 scaled = uv * scale;
    return vec2(c * scaled.x - s * scaled.y, s * scaled.x + c * scaled.y) + offset;
}

#endif // UV_TRANSFORM_GLSL
