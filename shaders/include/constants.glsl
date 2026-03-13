#ifndef CONSTANTS_GLSL
#define CONSTANTS_GLSL

// ── Ray tracing constants ────────────────────────────────────────
const float kRayTMin           = 0.001;     // Minimum ray t to avoid self-intersection
const float kRayTMax           = 10000.0;   // Maximum ray distance
const float kSurfaceBias       = 0.001;     // Offset from hit surface along normal/ray
const float kShadowRayBias     = 0.002;     // Shadow ray max distance reduction

// ── BRDF constants ───────────────────────────────────────────────
const float kMinCosTheta       = 0.001;     // Floor for dot products in BRDF evaluation
const float kMinRoughness      = 0.04;      // Minimum roughness to avoid GGX singularity
const float kMinGGXAlpha2      = 0.0002 * 0.0002;  // D_GGX alpha² floor
const float kDielectricF0      = 0.04;      // Fresnel reflectance at normal incidence (IOR 1.5)
const float kBRDFDenomFloor    = 0.001;     // Floor for 4*NdotV*NdotL denominator

// ── Russian roulette constants ───────────────────────────────────
const int   kRussianRouletteStartBounce   = 3;     // First bounce eligible for RR
const float kRussianRouletteMinThroughput = 0.01;   // Hard cutoff: negligible contribution
const float kRussianRouletteMaxSurvival   = 0.95;   // Cap survival probability

// ── Transparency constants ───────────────────────────────────────
const int   kMaxTransparencyBounces = 8;    // Extra loop iterations for transparency
const float kTIRThreshold       = 1e-6;     // Total internal reflection check threshold

// ── MIS constants ────────────────────────────────────────────────
const float kMinStrategyProb    = 0.03;     // Minimum probability per MIS strategy
const float kMaxRoughnessBoost  = 0.6;      // Specular roughness boost cap
const float kEnvRoughnessBoostFactor = 2.0; // Environment roughness boost multiplier
const float kEnvFresnelBoostFactor   = 0.5; // Environment grazing Fresnel boost
const float kEnvDynamicRangeScale    = 10.0; // Dynamic range normalization divisor
const float kMinDynamicRangeBoost    = 0.1; // Dynamic range boost clamp lower bound
const float kMaxDynamicRangeBoost    = 1.0; // Dynamic range boost clamp upper bound

// ── Texture sentinel ─────────────────────────────────────────────
const uint  kNoTexture          = 0xFFFFFFFFu;  // No texture bound

// ── Material / area light buffer strides ─────────────────────────
const uint  kMaterialStride     = 8u;       // vec4s per material in storage buffer
const uint  kAreaLightStride    = 4u;       // vec4s per area light in storage buffer

// ── Environment map constants ────────────────────────────────────
const float kEnvMapBounceLod    = 0.5;      // Mip level for bounced env map lookups
const float kSentinelDepth      = 1e4;      // G-buffer sentinel for miss depth

// ── Firefly filter constants ─────────────────────────────────────
const float kFireflyClampDiffuse  = 20.0;   // Luminance clamp for diffuse paths
const float kFireflyClampSpecular = 80.0;   // Luminance clamp for specular paths

// ── Sampling constants ───────────────────────────────────────────
const float kONBUpThreshold     = 0.999;    // Threshold for ONB up-vector selection
const float kDiagonalSpread     = 0.707;    // sqrt(2)/2 for Gaussian blur diagonals

#endif // CONSTANTS_GLSL
