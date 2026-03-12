#ifndef CLEARCOAT_GLSL
#define CLEARCOAT_GLSL

#include "brdf.glsl"

// Clear coat F0 for dielectric IOR 1.5: ((1.5-1)/(1.5+1))^2 = 0.04
const vec3 CLEAR_COAT_F0 = vec3(0.04);

// Clear coat attenuation of base layer energy.
// Models the double transmission (entering + exiting) through the clearcoat layer.
float calculateClearCoatAttenuation(float VdotH, float clear_coat_strength) {
    float clearcoat_F = F_Schlick(VdotH, CLEAR_COAT_F0).r;
    float transmission = 1.0 - clearcoat_F;
    float transmission2 = transmission * transmission;
    return mix(1.0, transmission2, clear_coat_strength);
}

// Evaluate dual-layer clearcoat + base PBR BRDF.
// Clearcoat is a dielectric GGX specular layer on top of the standard PBR base layer,
// with energy-conserving Fresnel attenuation of the base.
vec3 evaluateMultilayerBRDF(vec3 albedo, float roughness, float metallic, vec3 F0,
                            float clear_coat_strength, float clear_coat_roughness,
                            float NdotV, float NdotL, float NdotH, float VdotH) {
    vec3 base_brdf = evaluatePBR(albedo, roughness, metallic, F0, NdotV, NdotL, NdotH, VdotH);

    if (clear_coat_strength <= 0.0) return base_brdf;

    // Clearcoat specular layer: Cook-Torrance with clearcoat roughness and F0=0.04
    float cc_alpha = clear_coat_roughness * clear_coat_roughness;
    float cc_D = D_GGX(NdotH, cc_alpha * cc_alpha);
    float cc_G = G_SmithGGX(NdotV, NdotL, clear_coat_roughness);
    vec3 cc_F = F_Schlick(VdotH, CLEAR_COAT_F0);
    vec3 cc_brdf = (cc_D * cc_G * cc_F) / max(4.0 * NdotV * NdotL, kBRDFDenomFloor);

    // Energy-conserving combination: clearcoat on top attenuates base layer
    float attenuation = calculateClearCoatAttenuation(VdotH, clear_coat_strength);
    return cc_brdf * clear_coat_strength + base_brdf * attenuation;
}

#endif // CLEARCOAT_GLSL
