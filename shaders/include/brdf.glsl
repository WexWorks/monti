#ifndef BRDF_GLSL
#define BRDF_GLSL

#include "constants.glsl"

const float PI = 3.14159265358979323846;

// Schlick Fresnel approximation.
// cos_theta: use VdotH for microfacet models.
vec3 F_Schlick(float cos_theta, vec3 F0) {
    return F0 + (vec3(1.0) - F0) * pow(1.0 - clamp(cos_theta, 0.0, 1.0), 5.0);
}

// GGX normal distribution function.
float D_GGX(float NdotH, float alpha2) {
    float alpha2_c = max(alpha2, kMinGGXAlpha2);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (alpha2_c - 1.0) + 1.0;
    return alpha2_c / (PI * denom * denom);
}

// Smith G1 term (single direction) for GGX (Walter 2007, Eq. 34).
// alpha: GGX roughness parameter (roughness squared).
float G_SmithG1GGX(float NdotW, float alpha) {
    float NdotW_c = max(NdotW, kMinCosTheta);
    float cos2 = NdotW_c * NdotW_c;
    float tan2 = (1.0 - cos2) / cos2;
    float alpha2 = alpha * alpha;
    return 2.0 / (1.0 + sqrt(1.0 + alpha2 * tan2));
}

// Smith geometry term (combined view + light) for GGX.
float G_SmithGGX(float NdotV, float NdotL, float roughness) {
    float alpha = roughness * roughness;
    return G_SmithG1GGX(NdotV, alpha) * G_SmithG1GGX(NdotL, alpha);
}

// Combined Cook-Torrance specular + Lambertian diffuse evaluation.
// Returns the sum of diffuse and specular BRDF components.
vec3 evaluatePBR(vec3 albedo, float roughness, float metallic, vec3 F0,
                 float NdotV, float NdotL, float NdotH, float VdotH) {
    vec3 F = F_Schlick(VdotH, F0);

    // Diffuse: energy-conserving Lambertian
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse_brdf = kD * albedo / PI;

    // Specular: Cook-Torrance GGX
    float alpha = roughness * roughness;
    float D = D_GGX(NdotH, alpha * alpha);
    float G = G_SmithGGX(NdotV, NdotL, roughness);
    vec3 specular_brdf = (D * G * F) / max(4.0 * NdotV * NdotL, kBRDFDenomFloor);

    return diffuse_brdf + specular_brdf;
}

// Diffuse transmission BRDF: Lambertian transmission into the back hemisphere.
// Returns the BRDF value f(L) without the cosine geometry term.
vec3 evaluateDiffuseTransmission(vec3 albedo, vec3 dt_color,
                                  float diffuse_transmission_factor) {
    return albedo * dt_color * diffuse_transmission_factor / PI;
}

#endif // BRDF_GLSL
