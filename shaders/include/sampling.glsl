#ifndef SAMPLING_GLSL
#define SAMPLING_GLSL

#include "brdf.glsl"
#include "constants.glsl"

// Convert a world-space direction to equirectangular UV coordinates.
vec2 directionToUV(vec3 dir) {
    float phi = atan(dir.x, dir.z);
    float theta = asin(clamp(dir.y, -1.0, 1.0));
    return vec2(phi / (2.0 * PI) + 0.5, 0.5 - theta / PI);
}

// Rotate a direction around the Y axis by the given angle in radians.
vec3 rotateDirectionY(vec3 dir, float rotation) {
    float cos_r = cos(rotation);
    float sin_r = sin(rotation);
    return vec3(dir.x * cos_r - dir.z * sin_r, dir.y, dir.x * sin_r + dir.z * cos_r);
}

// Convert a world-space direction to equirectangular UV with azimuthal rotation (radians).
vec2 directionToUVRotated(vec3 dir, float rotation) {
    return directionToUV(rotateDirectionY(dir, rotation));
}

// Convert equirectangular UV coordinates to a world-space direction.
vec3 uvToDirection(vec2 uv) {
    float phi = (uv.x - 0.5) * 2.0 * PI;
    float theta = (0.5 - uv.y) * PI;
    float cos_theta = cos(theta);
    return vec3(sin(phi) * cos_theta, sin(theta), cos(phi) * cos_theta);
}

// Build an orthonormal basis (tangent, bitangent) from a normal vector.
void buildONB(vec3 N, out vec3 T, out vec3 B) {
    vec3 up = abs(N.y) < kONBUpThreshold ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

// Cosine-weighted hemisphere sampling.
// xi: two uniform random numbers in [0,1].
// N: surface normal.
// Returns a direction weighted by cos(theta)/PI.
vec3 cosineSampleHemisphere(vec2 xi, vec3 N) {
    float phi = 2.0 * PI * xi.x;
    float cos_theta = sqrt(1.0 - xi.y);
    float sin_theta = sqrt(xi.y);

    vec3 T, B;
    buildONB(N, T, B);

    return normalize(T * cos(phi) * sin_theta + B * sin(phi) * sin_theta + N * cos_theta);
}

// GGX importance sampling — generates a half-vector H in world space.
// xi: two uniform random numbers in [0,1].
// N: surface normal.
// roughness: material roughness.
// Returns: half-vector H in world space, sampled proportional to D_GGX.
vec3 sampleGGX(vec2 xi, vec3 N, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    float phi = 2.0 * PI * xi.x;
    float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha2 - 1.0) * xi.y));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    // H in tangent space
    vec3 H_tangent = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    // Transform to world space
    vec3 T, B;
    buildONB(N, T, B);

    return normalize(T * H_tangent.x + B * H_tangent.y + N * H_tangent.z);
}

// Binary search in a 1D CDF stored as a texture (R32F, width pixels, height=1).
// Returns the index (as float) where the CDF exceeds the target value.
float binarySearchCDF1D(sampler2D cdf_tex, float target, int size) {
    int lo = 0;
    int hi = size - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        float cdf_val = texelFetch(cdf_tex, ivec2(mid, 0), 0).r;
        if (cdf_val < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return float(lo);
}

// Binary search in a row of a 2D CDF texture (R32F, width x height).
// row: which row to search.
// Returns the column index (as float) where the CDF exceeds the target value.
float binarySearchCDF2D(sampler2D cdf_tex, float target, int row, int width) {
    int lo = 0;
    int hi = width - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        float cdf_val = texelFetch(cdf_tex, ivec2(mid, row), 0).r;
        if (cdf_val < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return float(lo);
}

// Environment map importance sampling via CDF binary search.
// xi: two uniform random numbers in [0,1].
// marginal_cdf: 1D CDF texture (height x 1, R32F) — row selection.
// conditional_cdf: 2D CDF texture (width x height, R32F) — column selection per row.
// env_size: ivec2(width, height) of the environment map.
// Returns: sampled UV coordinates in [0,1]^2.
vec2 environmentCDFSample(vec2 xi, sampler2D marginal_cdf, sampler2D conditional_cdf,
                          ivec2 env_size) {
    // Sample row from marginal CDF
    float row_f = binarySearchCDF1D(marginal_cdf, xi.y, env_size.y);
    int row = int(row_f);

    // Sample column from conditional CDF for this row
    float col_f = binarySearchCDF2D(conditional_cdf, xi.x, row, env_size.x);

    // Convert to UV with 0.5 pixel offset for center-of-pixel sampling
    float u = (col_f + 0.5) / float(env_size.x);
    float v = (row_f + 0.5) / float(env_size.y);

    return vec2(u, v);
}

// PDF for environment CDF sampling at a given UV.
// marginal_cdf / conditional_cdf: same CDF textures used for sampling.
// env_size: ivec2(width, height) of the environment map.
// Returns: probability density with respect to solid angle.
float environmentCDFPdf(vec2 uv, sampler2D marginal_cdf, sampler2D conditional_cdf,
                        ivec2 env_size) {
    int col = int(uv.x * float(env_size.x));
    int row = int(uv.y * float(env_size.y));
    col = clamp(col, 0, env_size.x - 1);
    row = clamp(row, 0, env_size.y - 1);

    // Marginal PDF: difference of adjacent CDF values
    float marginal_cdf_val = texelFetch(marginal_cdf, ivec2(row, 0), 0).r;
    float marginal_cdf_prev = row > 0 ? texelFetch(marginal_cdf, ivec2(row - 1, 0), 0).r : 0.0;
    float marginal_pdf = marginal_cdf_val - marginal_cdf_prev;

    // Conditional PDF: difference of adjacent CDF values in this row
    float cond_cdf_val = texelFetch(conditional_cdf, ivec2(col, row), 0).r;
    float cond_cdf_prev = col > 0 ? texelFetch(conditional_cdf, ivec2(col - 1, row), 0).r : 0.0;
    float cond_pdf = cond_cdf_val - cond_cdf_prev;

    // Joint PDF in UV space
    float pdf_uv = marginal_pdf * float(env_size.y) * cond_pdf * float(env_size.x);

    // Convert from UV-space to solid angle: d_omega = cos(theta) d_theta d_phi
    // UV -> (theta, phi): theta = (0.5 - v) * PI, phi = (u - 0.5) * 2*PI
    // Jacobian: d_theta*d_phi / du*dv = 2*PI^2
    // cos(theta) for equirectangular
    float theta = (0.5 - uv.y) * PI;
    float cos_elevation = cos(theta);
    float jacobian = 2.0 * PI * PI * max(cos_elevation, kMinCosTheta);

    return pdf_uv / jacobian;
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
    float o2 = spread * kDiagonalSpread; // sqrt(2) / 2
    result += textureLod(env_map, vec2(fract(env_uv.x + o2), clamp(env_uv.y + o2, 0.0, 1.0)), base_mip).rgb * 0.0625;
    result += textureLod(env_map, vec2(fract(env_uv.x - o2), clamp(env_uv.y + o2, 0.0, 1.0)), base_mip).rgb * 0.0625;
    result += textureLod(env_map, vec2(fract(env_uv.x + o2), clamp(env_uv.y - o2, 0.0, 1.0)), base_mip).rgb * 0.0625;
    result += textureLod(env_map, vec2(fract(env_uv.x - o2), clamp(env_uv.y - o2, 0.0, 1.0)), base_mip).rgb * 0.0625;

    return result;
}

// Luminance-based firefly clamping. Scales the entire vector proportionally
// when its luminance exceeds the threshold, preserving hue.
vec3 FireflyClamp(vec3 radiance, float threshold) {
    float lum = dot(radiance, vec3(0.2126, 0.7152, 0.0722));
    return (lum > threshold) ? radiance * (threshold / lum) : radiance;
}

#endif // SAMPLING_GLSL
