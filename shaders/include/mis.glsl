#ifndef MIS_GLSL
#define MIS_GLSL

#include "brdf.glsl"
#include "clearcoat.glsl"

// Strategy constants
const int STRATEGY_DIFFUSE = 0;
const int STRATEGY_SPECULAR = 1;
const int STRATEGY_CLEAR_COAT = 2;
const int STRATEGY_ENVIRONMENT = 3;
const int NUM_STRATEGIES = 4;

struct SamplingProbabilities {
    float diffuse;
    float specular;
    float clear_coat;
    float environment;
};

struct AllPDFs {
    float diffuse;
    float specular;
    float clear_coat;
    float environment;
};

// Cosine-weighted hemisphere PDF: p(L) = NdotL / PI
float calculateDiffusePDF(vec3 N, vec3 L) {
    return max(dot(N, L), 0.0) / PI;
}

// GGX importance sampling PDF: p(L) = D(H) * NdotH / (4 * VdotH)
float calculateGGXPDF(vec3 N, vec3 V, vec3 L, float roughness) {
    vec3 H = normalize(V + L);
    float NdotH = max(dot(N, H), kMinCosTheta);
    float VdotH = max(dot(V, H), kMinCosTheta);

    float alpha = roughness * roughness;
    float D = D_GGX(NdotH, alpha * alpha);

    return (D * NdotH) / (4.0 * VdotH);
}

// Approximate average VdotH for GGX distribution (Monte Carlo fit).
// Used for clearcoat attenuation probability estimation.
float computeAverageVdotH(float NdotV, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float NdotV_safe = max(NdotV, kMinCosTheta);
    float result = sqrt(NdotV_safe * NdotV_safe + alpha2 * (1.0 - NdotV_safe * NdotV_safe)) * 0.5 + 0.5;
    return clamp(result, 0.5, 1.0);
}

// Calculate sampling probabilities for current surface properties.
// All 4 strategies: diffuse, specular, clearcoat, environment.
SamplingProbabilities calculateSamplingProbabilities(
    vec3 N, vec3 V, vec3 F0, float metallic, float roughness,
    float clear_coat, float clear_coat_roughness,
    float env_avg_luminance, float env_max_luminance
) {
    float NdotV = max(dot(N, V), kMinCosTheta);

    // Fresnel at current viewing angle
    vec3 F_fresnel = F_Schlick(NdotV, F0);
    float F_avg = (F_fresnel.r + F_fresnel.g + F_fresnel.b) / 3.0;

    // Base energy split
    float base_specular = F_avg + metallic * (1.0 - F_avg);
    float base_diffuse = (1.0 - metallic) * (1.0 - F_avg);

    // Roughness boost for smooth dielectrics — favors specular sampling on shiny surfaces
    float roughness_factor = 1.0 - roughness;
    float specular_roughness_boost = roughness_factor * kMaxRoughnessBoost * (1.0 - metallic);
    float specular_strength = base_specular + specular_roughness_boost;

    // Clearcoat contribution: Fresnel + roughness boost
    float cc_fresnel = F_Schlick(NdotV, CLEAR_COAT_F0).r;
    float cc_roughness_factor = 1.0 - clear_coat_roughness;
    float cc_roughness_boost = cc_roughness_factor * kMaxRoughnessBoost;
    float cc_strength = clear_coat * (cc_fresnel + cc_roughness_boost);

    // Environment contribution: luminance-weighted with roughness and Fresnel boosts
    float roughness_boost_env = roughness * kEnvRoughnessBoostFactor;
    float fresnel_boost_env = (1.0 - NdotV) * kEnvFresnelBoostFactor;
    float dynamic_range = env_max_luminance / max(env_avg_luminance, kMinCosTheta);
    float dynamic_range_boost = clamp(dynamic_range / kEnvDynamicRangeScale,
                                      kMinDynamicRangeBoost, kMaxDynamicRangeBoost);
    float env_strength = env_avg_luminance *
                         (1.0 + roughness_boost_env + fresnel_boost_env) *
                         dynamic_range_boost;

    // Clearcoat attenuation of base layer in probability calculation
    float avg_vdot_h = computeAverageVdotH(NdotV, clear_coat_roughness);
    float cc_attenuation = calculateClearCoatAttenuation(avg_vdot_h, clear_coat);
    float attenuated_specular = specular_strength * cc_attenuation;
    float attenuated_diffuse = base_diffuse * cc_attenuation;

    float total = attenuated_specular + attenuated_diffuse + cc_strength + env_strength;

    // Normalize with minimum floor per strategy
    float prob_specular = clamp(attenuated_specular / total,
                                kMinStrategyProb, 1.0 - 3.0 * kMinStrategyProb);
    float prob_clear_coat = cc_strength > 0.0
        ? clamp(cc_strength / total, kMinStrategyProb, 1.0 - 3.0 * kMinStrategyProb)
        : 0.0;
    float prob_environment = clamp(env_strength / total,
                                   kMinStrategyProb, 1.0 - 3.0 * kMinStrategyProb);
    float prob_diffuse = 1.0 - prob_specular - prob_clear_coat - prob_environment;

    // Ensure diffuse probability stays above minimum
    if (prob_diffuse < kMinStrategyProb) {
        prob_diffuse = kMinStrategyProb;
        float remaining = 1.0 - prob_diffuse;
        float total_other = prob_specular + prob_clear_coat + prob_environment;
        if (total_other > 0.0) {
            prob_specular = (prob_specular / total_other) * remaining;
            prob_clear_coat = (prob_clear_coat / total_other) * remaining;
            prob_environment = (prob_environment / total_other) * remaining;
        } else {
            prob_specular = remaining / 3.0;
            prob_clear_coat = remaining / 3.0;
            prob_environment = remaining / 3.0;
        }
    }

    return SamplingProbabilities(prob_diffuse, prob_specular, prob_clear_coat, prob_environment);
}

// Choose strategy from probability CDF and a uniform random number in [0,1].
int chooseStrategy(SamplingProbabilities probs, float random_val) {
    if (random_val < probs.diffuse)
        return STRATEGY_DIFFUSE;
    if (random_val < probs.diffuse + probs.specular)
        return STRATEGY_SPECULAR;
    if (random_val < probs.diffuse + probs.specular + probs.clear_coat)
        return STRATEGY_CLEAR_COAT;
    return STRATEGY_ENVIRONMENT;
}

// Evaluate all strategy PDFs for a given sample direction.
// Environment PDF must be filled in by the caller (requires CDF texture access).
AllPDFs calculateAllPDFs(vec3 N, vec3 V, vec3 L, float roughness, float clear_coat_roughness) {
    AllPDFs pdfs;
    pdfs.diffuse = calculateDiffusePDF(N, L);
    pdfs.specular = calculateGGXPDF(N, V, L, roughness);
    pdfs.clear_coat = calculateGGXPDF(N, V, L, clear_coat_roughness);
    pdfs.environment = 0.0;  // Filled in by caller
    return pdfs;
}

// MIS weight using the power heuristic (beta=2).
// chosenStrategy: the strategy that generated this sample.
// pdfs: PDFs of all strategies for the sampled direction.
// probs: sampling probabilities for the current surface.
float calculateMISWeight(int chosen_strategy, AllPDFs pdfs, SamplingProbabilities probs) {
    // Weighted PDFs for each strategy
    float w1 = probs.diffuse * pdfs.diffuse;
    float w2 = probs.specular * pdfs.specular;
    float w3 = probs.clear_coat * pdfs.clear_coat;
    float w4 = probs.environment * pdfs.environment;

    float chosen_weight;
    if (chosen_strategy == STRATEGY_DIFFUSE) chosen_weight = w1;
    else if (chosen_strategy == STRATEGY_SPECULAR) chosen_weight = w2;
    else if (chosen_strategy == STRATEGY_CLEAR_COAT) chosen_weight = w3;
    else chosen_weight = w4;

    float sum_squared = w1 * w1 + w2 * w2 + w3 * w3 + w4 * w4;

    if (sum_squared <= 0.0) return 0.0;

    return (chosen_weight * chosen_weight) / sum_squared;
}

#endif // MIS_GLSL
