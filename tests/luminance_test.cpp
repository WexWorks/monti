#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>
#include <limits>
#include <vector>

#include <monti/capture/Luminance.h>

using Catch::Matchers::WithinRel;

namespace {

// Fill both diffuse and specular RGBA32F buffers uniformly.
// Combined luminance per pixel = BT.709(diffuse_rgb + specular_rgb).
void FillUniform(std::vector<float>& diffuse, std::vector<float>& specular,
                 uint32_t pixel_count, float diffuse_val, float specular_val) {
    diffuse.resize(static_cast<size_t>(pixel_count) * 4);
    specular.resize(static_cast<size_t>(pixel_count) * 4);
    for (uint32_t i = 0; i < pixel_count; ++i) {
        auto base = static_cast<size_t>(i) * 4;
        diffuse[base + 0] = diffuse_val;
        diffuse[base + 1] = diffuse_val;
        diffuse[base + 2] = diffuse_val;
        diffuse[base + 3] = 1.0f;
        specular[base + 0] = specular_val;
        specular[base + 1] = specular_val;
        specular[base + 2] = specular_val;
        specular[base + 3] = 1.0f;
    }
}

}  // namespace

TEST_CASE("ComputeLogAverageLuminance — uniform mid-gray", "[capture][luminance]") {
    // Diffuse = (0.18, 0.18, 0.18), Specular = (0.18, 0.18, 0.18)
    // Combined per-pixel RGB = (0.36, 0.36, 0.36)
    // BT.709 luminance = 0.2126*0.36 + 0.7152*0.36 + 0.0722*0.36 = 0.36
    constexpr uint32_t kPixels = 64;
    std::vector<float> diffuse, specular;
    FillUniform(diffuse, specular, kPixels, 0.18f, 0.18f);

    auto result = monti::capture::ComputeLogAverageLuminance(
        diffuse.data(), specular.data(), kPixels);

    REQUIRE_THAT(result.log_average, WithinRel(0.36f, 0.01f));
    REQUIRE(result.nan_count == 0);
    REQUIRE(result.total_pixels == kPixels);

    // norm_mul = 0.18 / 0.36 = 0.5
    float norm_mul = 0.18f / result.log_average;
    REQUIRE_THAT(norm_mul, WithinRel(0.5f, 0.01f));
}

TEST_CASE("ComputeLogAverageLuminance — uniform white diffuse only", "[capture][luminance]") {
    // Diffuse = (1, 1, 1), Specular = (0, 0, 0)
    // Combined luminance = 1.0, norm_mul = 0.18
    constexpr uint32_t kPixels = 64;
    std::vector<float> diffuse, specular;
    FillUniform(diffuse, specular, kPixels, 1.0f, 0.0f);

    auto result = monti::capture::ComputeLogAverageLuminance(
        diffuse.data(), specular.data(), kPixels);

    REQUIRE_THAT(result.log_average, WithinRel(1.0f, 0.01f));

    float norm_mul = 0.18f / result.log_average;
    REQUIRE_THAT(norm_mul, WithinRel(0.18f, 0.01f));
}

TEST_CASE("ComputeLogAverageLuminance — mixed bright/dark", "[capture][luminance]") {
    // Half pixels at combined luminance 0.01, half at 1.0
    // Geometric mean = exp(0.5*log(0.01) + 0.5*log(1.0)) = exp(0.5*(-4.605)) ≈ 0.1
    constexpr uint32_t kPixels = 64;
    std::vector<float> diffuse(static_cast<size_t>(kPixels) * 4);
    std::vector<float> specular(static_cast<size_t>(kPixels) * 4, 0.0f);

    for (uint32_t i = 0; i < kPixels; ++i) {
        auto base = static_cast<size_t>(i) * 4;
        float val = (i < kPixels / 2) ? 0.01f : 1.0f;
        diffuse[base + 0] = val;
        diffuse[base + 1] = val;
        diffuse[base + 2] = val;
        diffuse[base + 3] = 1.0f;
    }

    auto result = monti::capture::ComputeLogAverageLuminance(
        diffuse.data(), specular.data(), kPixels);

    // Geometric mean of 0.01 and 1.0 = sqrt(0.01) = 0.1
    REQUIRE_THAT(result.log_average, WithinRel(0.1f, 0.01f));

    float norm_mul = 0.18f / result.log_average;
    REQUIRE_THAT(norm_mul, WithinRel(1.8f, 0.01f));
}

TEST_CASE("ComputeLogAverageLuminance — NaN handling", "[capture][luminance]") {
    constexpr uint32_t kPixels = 16;
    std::vector<float> diffuse(static_cast<size_t>(kPixels) * 4, 0.0f);
    std::vector<float> specular(static_cast<size_t>(kPixels) * 4, 0.0f);

    // Set first 4 pixels to NaN (with alpha=1 so they aren't skipped as background)
    for (uint32_t i = 0; i < 4; ++i) {
        auto base = static_cast<size_t>(i) * 4;
        diffuse[base + 0] = std::numeric_limits<float>::quiet_NaN();
        diffuse[base + 1] = std::numeric_limits<float>::quiet_NaN();
        diffuse[base + 2] = std::numeric_limits<float>::quiet_NaN();
        diffuse[base + 3] = 1.0f;
    }

    // Set remaining 12 pixels to uniform (1, 1, 1) diffuse
    for (uint32_t i = 4; i < kPixels; ++i) {
        auto base = static_cast<size_t>(i) * 4;
        diffuse[base + 0] = 1.0f;
        diffuse[base + 1] = 1.0f;
        diffuse[base + 2] = 1.0f;
        diffuse[base + 3] = 1.0f;
    }

    auto result = monti::capture::ComputeLogAverageLuminance(
        diffuse.data(), specular.data(), kPixels);

    REQUIRE(result.nan_count == 4);
    REQUIRE(result.total_pixels == kPixels);
    // Only 12 valid pixels at luminance=1.0, so log_average ≈ 1.0
    REQUIRE_THAT(result.log_average, WithinRel(1.0f, 0.01f));
}

TEST_CASE("ComputeLogAverageLuminance — Inf handling", "[capture][luminance]") {
    constexpr uint32_t kPixels = 8;
    std::vector<float> diffuse(static_cast<size_t>(kPixels) * 4, 0.0f);
    std::vector<float> specular(static_cast<size_t>(kPixels) * 4, 0.0f);

    // First pixel: Inf
    diffuse[0] = std::numeric_limits<float>::infinity();
    diffuse[1] = 0.0f;
    diffuse[2] = 0.0f;
    diffuse[3] = 1.0f;

    // Remaining 7 pixels: white
    for (uint32_t i = 1; i < kPixels; ++i) {
        auto base = static_cast<size_t>(i) * 4;
        diffuse[base + 0] = 1.0f;
        diffuse[base + 1] = 1.0f;
        diffuse[base + 2] = 1.0f;
        diffuse[base + 3] = 1.0f;
    }

    auto result = monti::capture::ComputeLogAverageLuminance(
        diffuse.data(), specular.data(), kPixels);

    REQUIRE(result.nan_count == 1);
    REQUIRE_THAT(result.log_average, WithinRel(1.0f, 0.01f));
}

TEST_CASE("ComputeLogAverageLuminance — all NaN returns zero", "[capture][luminance]") {
    constexpr uint32_t kPixels = 4;
    std::vector<float> diffuse(static_cast<size_t>(kPixels) * 4,
                               std::numeric_limits<float>::quiet_NaN());
    std::vector<float> specular(static_cast<size_t>(kPixels) * 4, 0.0f);

    auto result = monti::capture::ComputeLogAverageLuminance(
        diffuse.data(), specular.data(), kPixels);

    REQUIRE(result.nan_count == kPixels);
    REQUIRE(result.log_average == 0.0f);
}

TEST_CASE("ComputeLogAverageLuminance — near-black detection", "[capture][luminance]") {
    // All pixels near zero → log_average < 0.001 (triggers skip in datagen)
    constexpr uint32_t kPixels = 16;
    std::vector<float> diffuse, specular;
    FillUniform(diffuse, specular, kPixels, 0.0001f, 0.0f);

    auto result = monti::capture::ComputeLogAverageLuminance(
        diffuse.data(), specular.data(), kPixels);

    // BT.709 luminance of (0.0001, 0.0001, 0.0001) = 0.0001
    // Epsilon is 1e-6, so log(0.0001) is used
    REQUIRE(result.log_average < 0.001f);
}

TEST_CASE("ComputeLogAverageLuminance — zero-luminance background pixels skipped", "[capture][luminance]") {
    // 8 pixels: 4 foreground (white) + 4 background (black, zero luminance).
    // All pixels have alpha=1.0 (env map always writes alpha=1).
    // Only the foreground pixels should contribute to the geometric mean;
    // zero-luminance background pixels are excluded by the L==0 check.
    constexpr uint32_t kPixels = 8;
    std::vector<float> diffuse(static_cast<size_t>(kPixels) * 4, 0.0f);
    std::vector<float> specular(static_cast<size_t>(kPixels) * 4, 0.0f);

    // First 4 pixels: white foreground (alpha = 1)
    for (uint32_t i = 0; i < 4; ++i) {
        auto base = static_cast<size_t>(i) * 4;
        diffuse[base + 0] = 1.0f;
        diffuse[base + 1] = 1.0f;
        diffuse[base + 2] = 1.0f;
        diffuse[base + 3] = 1.0f;
    }
    // Last 4 pixels: black background (alpha = 1, RGB = 0) — skipped by L==0
    for (uint32_t i = 4; i < kPixels; ++i) {
        auto base = static_cast<size_t>(i) * 4;
        diffuse[base + 3] = 1.0f;
    }

    auto result = monti::capture::ComputeLogAverageLuminance(
        diffuse.data(), specular.data(), kPixels);

    // Only 4 foreground pixels at luminance=1.0 → log_average ≈ 1.0
    REQUIRE_THAT(result.log_average, WithinRel(1.0f, 0.01f));
    REQUIRE(result.nan_count == 0);
    REQUIRE(result.total_pixels == kPixels);
}

TEST_CASE("ComputeLogAverageLuminance — all zero-luminance returns zero", "[capture][luminance]") {
    // All pixels are black background (alpha=1, RGB=0) — zero luminance, no
    // valid pixels after the L==0 skip.
    constexpr uint32_t kPixels = 4;
    std::vector<float> diffuse(static_cast<size_t>(kPixels) * 4, 0.0f);
    std::vector<float> specular(static_cast<size_t>(kPixels) * 4, 0.0f);
    for (uint32_t i = 0; i < kPixels; ++i)
        diffuse[static_cast<size_t>(i) * 4 + 3] = 1.0f;

    auto result = monti::capture::ComputeLogAverageLuminance(
        diffuse.data(), specular.data(), kPixels);

    REQUIRE(result.log_average == 0.0f);
    REQUIRE(result.nan_count == 0);
}

TEST_CASE("ComputeLogAverageLuminance — single pixel", "[capture][luminance]") {
    constexpr uint32_t kPixels = 1;
    std::vector<float> diffuse = {0.5f, 0.5f, 0.5f, 1.0f};
    std::vector<float> specular = {0.5f, 0.5f, 0.5f, 1.0f};

    // Combined = (1.0, 1.0, 1.0), luminance = 1.0
    auto result = monti::capture::ComputeLogAverageLuminance(
        diffuse.data(), specular.data(), kPixels);

    REQUIRE_THAT(result.log_average, WithinRel(1.0f, 0.001f));
    REQUIRE(result.nan_count == 0);
    REQUIRE(result.total_pixels == 1);
}
