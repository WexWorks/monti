#ifndef LIGHTS_GLSL
#define LIGHTS_GLSL

// ── Light type constants ─────────────────────────────────────────
const uint kLightTypeQuad     = 0u;
const uint kLightTypeSphere   = 1u;
const uint kLightTypeTriangle = 2u;

// ── LightSample: returned by all light sampling functions ────────
struct LightSample {
    vec3  position;  // sampled point on light surface
    vec3  normal;    // outward geometric normal at sampled point
    vec3  radiance;  // emitted radiance
    float pdf;       // solid-angle PDF (sr⁻¹)
};

// ── Quad light sampling ──────────────────────────────────────────
// Uniform area sampling on the parallelogram, converted to solid-angle PDF.
LightSample sampleQuadLight(vec4 data0, vec4 data1, vec4 data2, vec4 data3,
                            vec2 xi, vec3 shading_pos) {
    vec3 corner = data0.xyz;
    vec3 edge_a = data1.xyz;
    float two_sided = data1.w;
    vec3 edge_b = data2.xyz;
    vec3 light_radiance = data3.xyz;

    vec3 light_pos = corner + xi.x * edge_a + xi.y * edge_b;

    vec3 light_normal = cross(edge_a, edge_b);
    float light_area = length(light_normal);
    light_normal = light_normal / light_area;

    vec3 to_light = light_pos - shading_pos;
    float dist2 = dot(to_light, to_light);
    float dist = sqrt(dist2);
    vec3 L = to_light / dist;

    float light_cos = -dot(light_normal, L);
    if (light_cos <= 0.0 && two_sided < 0.5) {
        // Back-face hit on a single-sided light
        return LightSample(light_pos, light_normal, vec3(0.0), 0.0);
    }
    light_cos = abs(light_cos);

    float pdf = dist2 / (light_area * light_cos);

    LightSample ls;
    ls.position = light_pos;
    ls.normal = light_normal;
    ls.radiance = light_radiance;
    ls.pdf = pdf;
    return ls;
}

// ── Sphere light sampling ────────────────────────────────────────
// Uniform cone sampling: sample a direction within the cone subtended by
// the sphere as seen from the shading point. PDF = 1 / (2π(1 - cos θ_max)).
// Note: solid-angle sampling (Ureña et al. 2018) is a future option for
// improved near-field variance.
LightSample sampleSphereLight(vec4 data0, vec4 data1, vec4 data2, vec4 data3,
                              vec2 xi, vec3 shading_pos) {
    vec3 center = data0.xyz;
    float radius = data1.x;
    vec3 light_radiance = data3.xyz;

    vec3 to_center = center - shading_pos;
    float dist_center2 = dot(to_center, to_center);
    float dist_center = sqrt(dist_center2);

    // Clamp to avoid division by zero when shading point is inside sphere
    float sin_theta_max2 = min(radius * radius / dist_center2, 1.0);
    float cos_theta_max = sqrt(max(1.0 - sin_theta_max2, 0.0));

    // Degenerate: shading point inside or on sphere
    if (cos_theta_max <= 0.0) {
        LightSample ls;
        ls.position = center;
        ls.normal = normalize(shading_pos - center);
        ls.radiance = light_radiance;
        ls.pdf = 1.0;  // degenerate PDF
        return ls;
    }

    // Build local frame around direction to sphere center
    vec3 w = to_center / dist_center;
    vec3 u, v;
    if (abs(w.y) < 0.999) {
        u = normalize(cross(vec3(0.0, 1.0, 0.0), w));
    } else {
        u = normalize(cross(vec3(1.0, 0.0, 0.0), w));
    }
    v = cross(w, u);

    // Uniform cone sampling
    float cos_theta = 1.0 - xi.x * (1.0 - cos_theta_max);
    float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    float phi = 2.0 * PI * xi.y;

    vec3 sample_dir = sin_theta * cos(phi) * u
                    + sin_theta * sin(phi) * v
                    + cos_theta * w;

    // Intersect ray with sphere to find the actual surface point
    float t = dist_center * cos_theta
        - sqrt(max(radius * radius - dist_center2 * sin_theta * sin_theta, 0.0));
    vec3 light_pos = shading_pos + t * sample_dir;
    vec3 light_normal = normalize(light_pos - center);

    float pdf = 1.0 / (2.0 * PI * (1.0 - cos_theta_max));

    LightSample ls;
    ls.position = light_pos;
    ls.normal = light_normal;
    ls.radiance = light_radiance;
    ls.pdf = pdf;
    return ls;
}

// ── Triangle light sampling ──────────────────────────────────────
// Uniform barycentric sampling, converted to solid-angle PDF.
LightSample sampleTriangleLight(vec4 data0, vec4 data1, vec4 data2, vec4 data3,
                                vec2 xi, vec3 shading_pos) {
    vec3 v0 = data0.xyz;
    vec3 v1 = data1.xyz;
    float two_sided = data1.w;
    vec3 v2 = data2.xyz;
    vec3 light_radiance = data3.xyz;

    // Uniform barycentric coordinates
    float su = sqrt(xi.x);
    float b0 = 1.0 - su;
    float b1 = xi.y * su;
    // b2 = 1 - b0 - b1

    vec3 light_pos = b0 * v0 + b1 * v1 + (1.0 - b0 - b1) * v2;

    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 cross_e = cross(edge1, edge2);
    float light_area = length(cross_e) * 0.5;
    vec3 light_normal = cross_e / (light_area * 2.0);

    vec3 to_light = light_pos - shading_pos;
    float dist2 = dot(to_light, to_light);
    float dist = sqrt(dist2);
    vec3 L = to_light / dist;

    float light_cos = -dot(light_normal, L);
    if (light_cos <= 0.0 && two_sided < 0.5) {
        return LightSample(light_pos, light_normal, vec3(0.0), 0.0);
    }
    light_cos = abs(light_cos);

    float pdf = dist2 / (light_area * light_cos);

    LightSample ls;
    ls.position = light_pos;
    ls.normal = light_normal;
    ls.radiance = light_radiance;
    ls.pdf = pdf;
    return ls;
}

// ── Unified light sampling dispatch ──────────────────────────────
LightSample sampleLight(vec4 data0, vec4 data1, vec4 data2, vec4 data3,
                         vec2 xi, vec3 shading_pos) {
    uint light_type = floatBitsToUint(data0.w);
    if (light_type == kLightTypeSphere)
        return sampleSphereLight(data0, data1, data2, data3, xi, shading_pos);
    else if (light_type == kLightTypeTriangle)
        return sampleTriangleLight(data0, data1, data2, data3, xi, shading_pos);
    else
        return sampleQuadLight(data0, data1, data2, data3, xi, shading_pos);
}

#endif // LIGHTS_GLSL
