#ifndef MIS_GLSL
#define MIS_GLSL

#include "brdf.glsl"
#include "clearcoat.glsl"

// Strategy constants
const int STRATEGY_DIFFUSE = 0;
const int STRATEGY_SPECULAR = 1;
const int STRATEGY_CLEAR_COAT = 2;
const int STRATEGY_ENVIRONMENT = 3;
const int STRATEGY_DIFFUSE_TRANSMISSION = 4;
const int NUM_STRATEGIES = 5;

struct SamplingProbabilities {
    float diffuse;
    float specular;
    float clear_coat;
    float environment;
    float diffuse_transmission;
};

struct AllPDFs {
    float diffuse;
    float specular;
    float clear_coat;
    float environment;
    float diffuse_transmission;
};

// Cosine-weighted hemisphere PDF: p(L) = NdotL / PI
float calculateDiffusePDF(vec3 N, vec3 L) {
    return max(dot(N, L), 0.0) / PI;
}

// GGX importance sampling PDF: p(L) = D(H) * NdotH / (4 * VdotH)
float calculateGGXPDF(vec3 N, vec3 V, vec3 L, float roughness) {
    if (dot(N, L) <= 0.0) return 0.0;

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

// Calculate BSDF sampling probabilities for current surface properties.
// 4 strategies: diffuse, specular, clearcoat, diffuse transmission.
// Environment illumination is handled via separate NEE (shadow rays).
SamplingProbabilities calculateSamplingProbabilities(
    vec3 N, vec3 V, vec3 F0, float metallic, float roughness,
    float clear_coat, float clear_coat_roughness,
    float diffuse_transmission_factor
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

    // Clearcoat attenuation of base layer in probability calculation
    float avg_vdot_h = computeAverageVdotH(NdotV, clear_coat_roughness);
    float cc_attenuation = calculateClearCoatAttenuation(avg_vdot_h, clear_coat);
    float attenuated_specular = specular_strength * cc_attenuation;
    float attenuated_diffuse = base_diffuse * cc_attenuation;

    // Split diffuse energy between reflection and transmission
    float diffuse_reflect_strength = attenuated_diffuse * (1.0 - diffuse_transmission_factor);
    float diffuse_transmit_strength = attenuated_diffuse * diffuse_transmission_factor;

    float total = attenuated_specular + diffuse_reflect_strength + diffuse_transmit_strength
                + cc_strength;

    // Count active strategies for minimum probability floor
    float active_count = 2.0;  // diffuse + specular always active
    if (cc_strength > 0.0) active_count += 1.0;
    if (diffuse_transmission_factor > 0.0) active_count += 1.0;
    float max_prob_floor = 1.0 - (active_count - 1.0) * kMinStrategyProb;

    // Normalize with minimum floor per strategy
    float prob_specular = clamp(attenuated_specular / total,
                                kMinStrategyProb, max_prob_floor);
    float prob_clear_coat = cc_strength > 0.0
        ? clamp(cc_strength / total, kMinStrategyProb, max_prob_floor)
        : 0.0;
    float prob_diffuse_transmit = diffuse_transmission_factor > 0.0
        ? clamp(diffuse_transmit_strength / total, kMinStrategyProb, max_prob_floor)
        : 0.0;
    float prob_diffuse = 1.0 - prob_specular - prob_clear_coat - prob_diffuse_transmit;

    // Ensure diffuse probability stays above minimum
    if (prob_diffuse < kMinStrategyProb) {
        prob_diffuse = kMinStrategyProb;
        float remaining = 1.0 - prob_diffuse;
        float total_other = prob_specular + prob_clear_coat + prob_diffuse_transmit;
        if (total_other > 0.0) {
            prob_specular = (prob_specular / total_other) * remaining;
            prob_clear_coat = (prob_clear_coat / total_other) * remaining;
            prob_diffuse_transmit = (prob_diffuse_transmit / total_other) * remaining;
        } else {
            prob_specular = remaining / 3.0;
            prob_clear_coat = remaining / 3.0;
            prob_diffuse_transmit = remaining / 3.0;
        }
    }

    return SamplingProbabilities(prob_diffuse, prob_specular, prob_clear_coat,
                                0.0, prob_diffuse_transmit);
}

// Choose strategy from probability CDF and a uniform random number in [0,1].
int chooseStrategy(SamplingProbabilities probs, float random_val) {
    if (random_val < probs.diffuse)
        return STRATEGY_DIFFUSE;
    if (random_val < probs.diffuse + probs.specular)
        return STRATEGY_SPECULAR;
    if (random_val < probs.diffuse + probs.specular + probs.clear_coat)
        return STRATEGY_CLEAR_COAT;
    // Only select DT when it has non-zero probability; otherwise float
    // rounding pushed random_val past the CDF — fall back to diffuse.
    if (probs.diffuse_transmission > 0.0)
        return STRATEGY_DIFFUSE_TRANSMISSION;
    return STRATEGY_DIFFUSE;
}

// Cosine-weighted hemisphere PDF for back hemisphere (diffuse transmission):
// p(L) = max(-NdotL, 0.0) / PI
float calculateDiffuseTransmissionPDF(vec3 N, vec3 L) {
    return max(-dot(N, L), 0.0) / PI;
}

// Evaluate all strategy PDFs for a given sample direction.
// Environment PDF must be filled in by the caller (requires CDF texture access).
AllPDFs calculateAllPDFs(vec3 N, vec3 V, vec3 L, float roughness, float clear_coat_roughness) {
    AllPDFs pdfs;
    pdfs.diffuse = calculateDiffusePDF(N, L);
    pdfs.specular = calculateGGXPDF(N, V, L, roughness);
    pdfs.clear_coat = calculateGGXPDF(N, V, L, clear_coat_roughness);
    pdfs.environment = 0.0;  // Filled in by caller
    pdfs.diffuse_transmission = calculateDiffuseTransmissionPDF(N, L);
    return pdfs;
}

// MIS weight using the power heuristic (beta=2).
// chosenStrategy: the strategy that generated this sample.
// pdfs: PDFs of all strategies for the sampled direction.
// probs: sampling probabilities for the current surface.
float calculateMISWeight(int chosen_strategy, AllPDFs pdfs, SamplingProbabilities probs) {
    // Weighted PDFs for each strategy (env is 0 — handled via separate NEE)
    float w1 = probs.diffuse * pdfs.diffuse;
    float w2 = probs.specular * pdfs.specular;
    float w3 = probs.clear_coat * pdfs.clear_coat;
    float w5 = probs.diffuse_transmission * pdfs.diffuse_transmission;

    float chosen_weight;
    if (chosen_strategy == STRATEGY_DIFFUSE) chosen_weight = w1;
    else if (chosen_strategy == STRATEGY_SPECULAR) chosen_weight = w2;
    else if (chosen_strategy == STRATEGY_CLEAR_COAT) chosen_weight = w3;
    else chosen_weight = w5;

    float sum_squared = w1 * w1 + w2 * w2 + w3 * w3 + w5 * w5;

    if (sum_squared <= 0.0) return 0.0;

    return (chosen_weight * chosen_weight) / sum_squared;
}

// Combined BSDF PDF (probability-weighted sum of all BSDF strategy PDFs).
// Used for environment NEE MIS weight computation.
float calculateBsdfCombinedPdf(SamplingProbabilities probs,
                               vec3 N, vec3 V, vec3 L,
                               float roughness, float clear_coat_roughness) {
    return probs.diffuse * calculateDiffusePDF(N, L)
         + probs.specular * calculateGGXPDF(N, V, L, roughness)
         + probs.clear_coat * calculateGGXPDF(N, V, L, clear_coat_roughness)
         + probs.diffuse_transmission * calculateDiffuseTransmissionPDF(N, L);
}

// Power heuristic MIS weight for environment NEE vs BSDF.
float envNeeMISWeight(float env_pdf, float bsdf_combined_pdf) {
    float e2 = env_pdf * env_pdf;
    float b2 = bsdf_combined_pdf * bsdf_combined_pdf;
    float denom = e2 + b2;
    return denom > 0.0 ? e2 / denom : 0.0;
}

// Complementary MIS weight for BSDF miss vs environment NEE.
float bsdfMissMISWeight(float bsdf_combined_pdf, float env_pdf) {
    float b2 = bsdf_combined_pdf * bsdf_combined_pdf;
    float e2 = env_pdf * env_pdf;
    float denom = b2 + e2;
    return denom > 0.0 ? b2 / denom : 1.0;
}

#endif // MIS_GLSL
