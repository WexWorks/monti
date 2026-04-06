// CPU unit tests for Welford's online variance algorithm.
// Validates the same update logic used in variance_update.comp (log-luminance domain).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

namespace {

// CPU mirror of the Welford update in variance_update.comp.
struct WelfordState {
    float mean = 0.0f;
    float m2 = 0.0f;
};

void WelfordUpdate(WelfordState& state, float x, uint32_t n) {
    float fn = static_cast<float>(n);
    float delta = x - state.mean;
    state.mean += delta / std::max(fn, 1.0f);
    float delta2 = x - state.mean;
    state.m2 += delta * delta2;
}

float WelfordVariance(const WelfordState& state, uint32_t n) {
    if (n < 2) return 0.0f;
    return state.m2 / static_cast<float>(n - 1);
}

// Two-pass reference variance
float ReferenceVariance(const std::vector<float>& values) {
    if (values.size() < 2) return 0.0f;
    double mean = 0.0;
    for (float v : values) mean += v;
    mean /= static_cast<double>(values.size());
    double sum_sq = 0.0;
    for (float v : values) {
        double d = static_cast<double>(v) - mean;
        sum_sq += d * d;
    }
    return static_cast<float>(sum_sq / static_cast<double>(values.size() - 1));
}

float ReferenceMean(const std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    double sum = 0.0;
    for (float v : values) sum += v;
    return static_cast<float>(sum / static_cast<double>(values.size()));
}

}  // namespace

TEST_CASE("Welford: constant sequence has zero variance", "[welford][cpu]") {
    constexpr float kValue = 3.14f;
    constexpr uint32_t kCount = 100;

    WelfordState state{};
    for (uint32_t i = 1; i <= kCount; ++i)
        WelfordUpdate(state, kValue, i);

    REQUIRE_THAT(state.mean, WithinRel(kValue, 1e-5f));
    REQUIRE_THAT(WelfordVariance(state, kCount), WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("Welford: linear sequence matches two-pass reference", "[welford][cpu]") {
    constexpr uint32_t kCount = 200;
    std::vector<float> values(kCount);
    for (uint32_t i = 0; i < kCount; ++i)
        values[i] = static_cast<float>(i) * 0.5f;

    WelfordState state{};
    for (uint32_t i = 0; i < kCount; ++i)
        WelfordUpdate(state, values[i], i + 1);

    float ref_mean = ReferenceMean(values);
    float ref_var = ReferenceVariance(values);

    REQUIRE_THAT(state.mean, WithinRel(ref_mean, 1e-4f));
    REQUIRE_THAT(WelfordVariance(state, kCount), WithinRel(ref_var, 1e-4f));
}

TEST_CASE("Welford: large-offset sequence (numerical stability)", "[welford][cpu]") {
    // Values near 1e3 with small perturbation — naive variance would suffer
    // catastrophic cancellation but Welford should handle it.
    constexpr uint32_t kCount = 500;
    constexpr float kOffset = 1e3f;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    std::vector<float> values(kCount);
    for (uint32_t i = 0; i < kCount; ++i)
        values[i] = kOffset + dist(rng);

    WelfordState state{};
    for (uint32_t i = 0; i < kCount; ++i)
        WelfordUpdate(state, values[i], i + 1);

    float ref_mean = ReferenceMean(values);
    float ref_var = ReferenceVariance(values);

    // Mean should be very close to 1e6
    REQUIRE_THAT(state.mean, WithinRel(ref_mean, 1e-4f));

    // Variance should match the two-pass reference within reasonable tolerance.
    // The absolute variance is tiny (~1e-4) so use relative tolerance.
    REQUIRE(ref_var > 0.0f);
    REQUIRE_THAT(WelfordVariance(state, kCount), WithinRel(ref_var, 0.05f));
}

TEST_CASE("Welford: single element has zero variance and correct mean", "[welford][cpu]") {
    constexpr float kValue = -2.5f;

    WelfordState state{};
    WelfordUpdate(state, kValue, 1);

    REQUIRE_THAT(state.mean, WithinRel(kValue, 1e-6f));
    REQUIRE_THAT(state.m2, WithinAbs(0.0f, 1e-10f));
    REQUIRE_THAT(WelfordVariance(state, 1), WithinAbs(0.0f, 1e-10f));
}

TEST_CASE("Welford: log-luminance domain matches shader usage", "[welford][cpu]") {
    // Simulate what variance_update.comp does: log(max(luminance, 1e-7))
    std::vector<float> luminances = {0.1f, 0.5f, 1.0f, 2.0f, 0.01f, 5.0f, 0.001f, 10.0f};
    std::vector<float> log_values;
    log_values.reserve(luminances.size());
    for (float lum : luminances)
        log_values.push_back(std::log(std::max(lum, 1e-7f)));

    WelfordState state{};
    for (uint32_t i = 0; i < log_values.size(); ++i)
        WelfordUpdate(state, log_values[i], i + 1);

    auto n = static_cast<uint32_t>(log_values.size());
    float ref_mean = ReferenceMean(log_values);
    float ref_var = ReferenceVariance(log_values);

    REQUIRE_THAT(state.mean, WithinRel(ref_mean, 1e-4f));
    REQUIRE_THAT(WelfordVariance(state, n), WithinRel(ref_var, 1e-4f));
}
