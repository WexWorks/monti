#ifndef COMMON_GLSL
#define COMMON_GLSL

const float PI = 3.14159265358979323846;

// Rotate a direction around the Y axis by the given angle in radians.
vec3 rotateDirectionY(vec3 dir, float rotation) {
    float cos_r = cos(rotation);
    float sin_r = sin(rotation);
    return vec3(dir.x * cos_r - dir.z * sin_r, dir.y, dir.x * sin_r + dir.z * cos_r);
}

// Convert a world-space direction to equirectangular UV coordinates.
vec2 directionToUV(vec3 dir) {
    float phi = atan(dir.x, dir.z);
    float theta = asin(clamp(dir.y, -1.0, 1.0));
    return vec2(phi / (2.0 * PI) + 0.5, 0.5 - theta / PI);
}

// Convert a world-space direction to equirectangular UV with azimuthal rotation (radians).
vec2 directionToUVRotated(vec3 dir, float rotation) {
    return directionToUV(rotateDirectionY(dir, rotation));
}

// 9-tap Gaussian blur for environment map background.
// Spreads samples around the base UV at one mip level below the target, then blends
// with Gaussian weights to smooth out pixelation on large screens at high mip levels.
// mip_level: target blur level (e.g. pc.skybox_mip_level).
vec3 sampleEnvironmentBlurred(sampler2D env_map, vec3 direction, float mip_level, float rotation) {
    vec2 env_uv = directionToUVRotated(direction, rotation);

    // Texel size at the requested mip level
    float mip_texel_size_x = pow(2.0, mip_level) / float(textureSize(env_map, 0).x);

    // Sample one mip level lower so the Gaussian filter has enough detail to blur
    float base_mip = max(0.0, mip_level - 1.0);
    float spread = mip_texel_size_x * 0.5;

    // Center: weight 0.25
    vec3 result = textureLod(env_map, env_uv, base_mip).rgb * 0.25;

    // First ring — cardinal directions: weight 0.125 each
    float o1 = spread;
    result += textureLod(env_map, vec2(fract(env_uv.x + o1), clamp(env_uv.y, 0.0, 1.0)), base_mip).rgb * 0.125;
    result += textureLod(env_map, vec2(fract(env_uv.x - o1), clamp(env_uv.y, 0.0, 1.0)), base_mip).rgb * 0.125;
    result += textureLod(env_map, vec2(env_uv.x, clamp(env_uv.y + o1, 0.0, 1.0)), base_mip).rgb * 0.125;
    result += textureLod(env_map, vec2(env_uv.x, clamp(env_uv.y - o1, 0.0, 1.0)), base_mip).rgb * 0.125;

    // Second ring — diagonal directions: weight 0.0625 each
    float o2 = spread * 0.707; // sqrt(2) / 2
    result += textureLod(env_map, vec2(fract(env_uv.x + o2), clamp(env_uv.y + o2, 0.0, 1.0)), base_mip).rgb * 0.0625;
    result += textureLod(env_map, vec2(fract(env_uv.x - o2), clamp(env_uv.y + o2, 0.0, 1.0)), base_mip).rgb * 0.0625;
    result += textureLod(env_map, vec2(fract(env_uv.x + o2), clamp(env_uv.y - o2, 0.0, 1.0)), base_mip).rgb * 0.0625;
    result += textureLod(env_map, vec2(fract(env_uv.x - o2), clamp(env_uv.y - o2, 0.0, 1.0)), base_mip).rgb * 0.0625;

    return result;
}

#endif // COMMON_GLSL
