#ifndef SHEEN_GLSL
#define SHEEN_GLSL

#include "constants.glsl"
#include "brdf.glsl"

// ── Charlie NDF (Conty & Kulla 2017) ─────────────────────────────
// Exponentiated sinusoidal distribution for soft fabric/velvet.
// alpha_g = sheenRoughness² gives perceptually linear roughness.
float charlieD(float NdotH, float alpha_g) {
    float inv_r = 1.0 / alpha_g;
    float sin2h = 1.0 - NdotH * NdotH;
    return (2.0 + inv_r) * pow(max(sin2h, 0.0), inv_r * 0.5) / (2.0 * PI);
}

// ── Lambda function for Charlie visibility ───────────────────────
// Rational polynomial fit from Conty & Kulla 2017.
float lambdaSheenL(float x, float alpha_g) {
    float one_minus_alpha_sq = (1.0 - alpha_g) * (1.0 - alpha_g);
    float a = mix(21.5473, 25.3245, one_minus_alpha_sq);
    float b = mix(3.82987, 3.32435, one_minus_alpha_sq);
    float c = mix(0.19823, 0.16801, one_minus_alpha_sq);
    float d = mix(-1.97760, -1.27393, one_minus_alpha_sq);
    float e = mix(-4.32054, -4.85967, one_minus_alpha_sq);
    return a / (1.0 + b * pow(x, c)) + d * x + e;
}

float lambdaSheen(float cos_theta, float alpha_g) {
    float ct = clamp(cos_theta, kMinCosTheta, 1.0);
    return abs(ct) < 0.5
        ? exp(lambdaSheenL(ct, alpha_g))
        : exp(2.0 * lambdaSheenL(0.5, alpha_g) - lambdaSheenL(1.0 - ct, alpha_g));
}

// ── Charlie visibility term ──────────────────────────────────────
// V = 1 / ((1 + lambda(V) + lambda(L)) * 4 * NdotV * NdotL)
float charlieV(float NdotV, float NdotL, float alpha_g) {
    float denom = (1.0 + lambdaSheen(NdotV, alpha_g)
                 + lambdaSheen(NdotL, alpha_g))
                 * (4.0 * NdotV * NdotL);
    return 1.0 / max(denom, kBRDFDenomFloor);
}

// ── Sheen directional albedo LUT (16×16) ─────────────────────────
// Precomputed hemispherical integral E(NdotV, sheenRoughness) of the
// Charlie BRDF. Rows = NdotV (0→1), Columns = roughness (0→1).
// Source: Enterprise PBR Shading Model, section 6.2.3 (Charlie variant).
const float kSheenAlbedoLUT[16 * 16] = float[](
    // NdotV=0.0 (grazing)
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    // NdotV=0.0667
    0.9990, 0.8811, 0.6784, 0.5128, 0.3957, 0.3142, 0.2569, 0.2154,
    0.1845, 0.1608, 0.1422, 0.1273, 0.1151, 0.1050, 0.0965, 0.0893,
    // NdotV=0.1333
    0.9910, 0.7854, 0.5597, 0.4100, 0.3125, 0.2474, 0.2023, 0.1700,
    0.1461, 0.1278, 0.1134, 0.1019, 0.0925, 0.0847, 0.0782, 0.0726,
    // NdotV=0.2000
    0.9632, 0.6918, 0.4647, 0.3345, 0.2529, 0.1997, 0.1634, 0.1377,
    0.1187, 0.1042, 0.0928, 0.0836, 0.0762, 0.0700, 0.0648, 0.0604,
    // NdotV=0.2667
    0.9103, 0.6040, 0.3890, 0.2766, 0.2082, 0.1641, 0.1343, 0.1134,
    0.0980, 0.0864, 0.0771, 0.0697, 0.0637, 0.0587, 0.0545, 0.0510,
    // NdotV=0.3333
    0.8358, 0.5240, 0.3274, 0.2308, 0.1732, 0.1366, 0.1119, 0.0947,
    0.0821, 0.0725, 0.0649, 0.0588, 0.0539, 0.0498, 0.0463, 0.0434,
    // NdotV=0.4000
    0.7467, 0.4524, 0.2769, 0.1941, 0.1454, 0.1147, 0.0941, 0.0797,
    0.0692, 0.0613, 0.0550, 0.0500, 0.0459, 0.0425, 0.0396, 0.0372,
    // NdotV=0.4667
    0.6509, 0.3891, 0.2349, 0.1641, 0.1228, 0.0970, 0.0796, 0.0676,
    0.0588, 0.0521, 0.0469, 0.0427, 0.0393, 0.0364, 0.0340, 0.0320,
    // NdotV=0.5333
    0.5552, 0.3334, 0.1997, 0.1393, 0.1043, 0.0825, 0.0678, 0.0576,
    0.0502, 0.0446, 0.0402, 0.0367, 0.0338, 0.0314, 0.0294, 0.0277,
    // NdotV=0.6000
    0.4641, 0.2841, 0.1697, 0.1184, 0.0888, 0.0703, 0.0579, 0.0493,
    0.0430, 0.0383, 0.0346, 0.0316, 0.0292, 0.0272, 0.0255, 0.0241,
    // NdotV=0.6667
    0.3803, 0.2402, 0.1439, 0.1006, 0.0755, 0.0600, 0.0495, 0.0422,
    0.0369, 0.0329, 0.0298, 0.0273, 0.0253, 0.0236, 0.0222, 0.0210,
    // NdotV=0.7333
    0.3046, 0.1988, 0.1210, 0.0853, 0.0644, 0.0513, 0.0425, 0.0363,
    0.0318, 0.0284, 0.0258, 0.0237, 0.0220, 0.0206, 0.0194, 0.0184,
    // NdotV=0.8000
    0.2371, 0.1605, 0.1005, 0.0718, 0.0547, 0.0439, 0.0365, 0.0313,
    0.0276, 0.0247, 0.0225, 0.0207, 0.0193, 0.0181, 0.0171, 0.0163,
    // NdotV=0.8667
    0.1773, 0.1246, 0.0815, 0.0596, 0.0461, 0.0374, 0.0314, 0.0271,
    0.0240, 0.0216, 0.0197, 0.0183, 0.0171, 0.0161, 0.0152, 0.0146,
    // NdotV=0.9333
    0.1244, 0.0901, 0.0624, 0.0476, 0.0380, 0.0314, 0.0269, 0.0236,
    0.0211, 0.0191, 0.0176, 0.0163, 0.0153, 0.0145, 0.0138, 0.0132,
    // NdotV=1.0
    0.0747, 0.0545, 0.0404, 0.0332, 0.0287, 0.0254, 0.0228, 0.0208,
    0.0192, 0.0179, 0.0168, 0.0159, 0.0151, 0.0145, 0.0139, 0.0134
);

// Bilinear lookup into the 16×16 sheen albedo LUT.
float sheenDirectionalAlbedo(float NdotV, float sheen_roughness) {
    float u = clamp(NdotV, 0.0, 1.0) * 15.0;
    float v = clamp(sheen_roughness, 0.0, 1.0) * 15.0;

    int u0 = int(u);
    int v0 = int(v);
    int u1 = min(u0 + 1, 15);
    int v1 = min(v0 + 1, 15);

    float fu = u - float(u0);
    float fv = v - float(v0);

    float s00 = kSheenAlbedoLUT[u0 * 16 + v0];
    float s10 = kSheenAlbedoLUT[u1 * 16 + v0];
    float s01 = kSheenAlbedoLUT[u0 * 16 + v1];
    float s11 = kSheenAlbedoLUT[u1 * 16 + v1];

    return mix(mix(s00, s10, fu), mix(s01, s11, fu), fv);
}

// ── Full sheen evaluation ────────────────────────────────────────
// Returns sheen BRDF contribution: sheen_color * D_charlie * V_charlie
vec3 evaluateSheen(vec3 sheen_color, float sheen_roughness,
                   float NdotH, float NdotL, float NdotV) {
    float alpha_g = max(sheen_roughness * sheen_roughness, kMinRoughness * kMinRoughness);
    float D = charlieD(NdotH, alpha_g);
    float V = charlieV(NdotV, NdotL, alpha_g);
    return sheen_color * D * V;
}

#endif // SHEEN_GLSL
