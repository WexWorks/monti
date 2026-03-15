#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <monti/capture/GpuReadback.h>
#include <monti/capture/Writer.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <vector>

#include <tinyexr.h>

namespace {

const std::string kTestOutputDir = "test_output/phase11b_test";

struct ScopedCleanup {
    ~ScopedCleanup() {
        std::error_code ec;
        std::filesystem::remove_all(kTestOutputDir, ec);
    }
};

// Load an EXR file and return channel count. Fills header, image, stored_types.
int LoadExr(const std::string& path, EXRHeader& header, EXRImage& image,
            std::vector<int>& stored_types) {
    EXRVersion version;
    if (ParseEXRVersionFromFile(&version, path.c_str()) != TINYEXR_SUCCESS)
        return -1;

    InitEXRHeader(&header);
    const char* err = nullptr;
    if (ParseEXRHeaderFromFile(&header, &version, path.c_str(), &err) != TINYEXR_SUCCESS) {
        if (err) FreeEXRErrorMessage(err);
        return -1;
    }

    stored_types.assign(header.pixel_types,
                        header.pixel_types + header.num_channels);

    for (int i = 0; i < header.num_channels; ++i)
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;

    InitEXRImage(&image);
    if (LoadEXRImageFromFile(&image, &header, path.c_str(), &err) != TINYEXR_SUCCESS) {
        if (err) FreeEXRErrorMessage(err);
        FreeEXRHeader(&header);
        return -1;
    }

    return header.num_channels;
}

int FindChannel(const EXRHeader& header, const char* name) {
    for (int i = 0; i < header.num_channels; ++i)
        if (std::strcmp(header.channels[i].name, name) == 0) return i;
    return -1;
}

bool ChannelHasType(const EXRHeader& header, const std::vector<int>& stored_types,
                    const char* name, int expected_type) {
    int idx = FindChannel(header, name);
    if (idx < 0) return false;
    return stored_types[static_cast<size_t>(idx)] == expected_type;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// UnpackB10G11R11 tests
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("UnpackB10G11R11 round-trips known values", "[capture][format]") {
    SECTION("all zeros") {
        float r, g, b;
        monti::capture::UnpackB10G11R11(0x00000000u, r, g, b);
        REQUIRE(r == 0.0f);
        REQUIRE(g == 0.0f);
        REQUIRE(b == 0.0f);
    }

    SECTION("R = 1.0 (11-bit float 0x3C0)") {
        // R11G11B10: R is bits [0:10], 6-bit mantissa + 5-bit exponent
        // 1.0 in 6e5 = exponent=15 (bias=15), mantissa=0 → binary 01111 000000 = 0x3C0
        float r, g, b;
        monti::capture::UnpackB10G11R11(0x3C0u, r, g, b);
        REQUIRE_THAT(r, Catch::Matchers::WithinAbs(1.0, 0.01));
        REQUIRE(g == 0.0f);
        REQUIRE(b == 0.0f);
    }

    SECTION("G = 1.0 (11-bit float shifted)") {
        // G is bits [11:21], same encoding as R but shifted
        float r, g, b;
        monti::capture::UnpackB10G11R11(0x3C0u << 11, r, g, b);
        REQUIRE(r == 0.0f);
        REQUIRE_THAT(g, Catch::Matchers::WithinAbs(1.0, 0.01));
        REQUIRE(b == 0.0f);
    }

    SECTION("B = 1.0 (10-bit float shifted)") {
        // B is bits [22:31], 5-bit mantissa + 5-bit exponent
        // 1.0 in 5e5 = exponent=15, mantissa=0 → binary 01111 00000 = 0x1E0
        float r, g, b;
        monti::capture::UnpackB10G11R11(0x1E0u << 22, r, g, b);
        REQUIRE(r == 0.0f);
        REQUIRE(g == 0.0f);
        REQUIRE_THAT(b, Catch::Matchers::WithinAbs(1.0, 0.01));
    }

    SECTION("image batch unpacking") {
        constexpr uint32_t kPixels = 4;
        uint32_t packed[kPixels] = {0, 0x3C0u, 0x3C0u << 11, 0x1E0u << 22};
        float rgb[kPixels * 3];
        monti::capture::UnpackB10G11R11Image(packed, rgb, kPixels);

        // pixel 0: all zeros
        REQUIRE(rgb[0] == 0.0f);
        REQUIRE(rgb[1] == 0.0f);
        REQUIRE(rgb[2] == 0.0f);

        // pixel 1: R≈1
        REQUIRE_THAT(rgb[3], Catch::Matchers::WithinAbs(1.0, 0.01));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ExtractDepthFromRG16F tests
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("ExtractDepthFromRG16F extracts R channel only", "[capture][format]") {
    constexpr uint32_t kPixels = 3;
    // RG16F = 2 uint16_t per pixel. R = depth, G = hit distance (ignored).
    uint16_t rg16f[kPixels * 2];

    // Set R to known values, G to something different
    float test_depths[] = {1.5f, 0.0f, 42.0f};
    float test_hits[]   = {99.0f, 77.0f, 0.1f};

    for (uint32_t i = 0; i < kPixels; ++i) {
        rg16f[i * 2 + 0] = monti::capture::FloatToHalf(test_depths[i]);
        rg16f[i * 2 + 1] = monti::capture::FloatToHalf(test_hits[i]);
    }

    float depth_out[kPixels];
    monti::capture::ExtractDepthFromRG16F(rg16f, depth_out, kPixels);

    for (uint32_t i = 0; i < kPixels; ++i) {
        REQUIRE_THAT(depth_out[i], Catch::Matchers::WithinAbs(
            static_cast<double>(test_depths[i]), 0.01));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Raw FP16 writer round-trip test
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("WriteFrameRaw FP16 round-trip", "[capture][writer]") {
    ScopedCleanup cleanup;

    constexpr uint32_t kWidth = 16;
    constexpr uint32_t kHeight = 8;
    constexpr uint32_t kPixels = kWidth * kHeight;

    monti::capture::WriterDesc desc{};
    desc.output_dir = kTestOutputDir;
    desc.input_width = kWidth;
    desc.input_height = kHeight;
    desc.scale_mode = monti::capture::ScaleMode::kNative;

    auto writer = monti::capture::Writer::Create(desc);
    REQUIRE(writer);

    // Generate known FP16 test data for noisy_diffuse (RGBA)
    std::vector<uint16_t> noisy_diffuse(static_cast<size_t>(kPixels) * 4);
    for (uint32_t i = 0; i < kPixels; ++i) {
        float val = static_cast<float>(i + 1) * 0.01f;
        noisy_diffuse[i * 4 + 0] = monti::capture::FloatToHalf(val);
        noisy_diffuse[i * 4 + 1] = monti::capture::FloatToHalf(val * 0.5f);
        noisy_diffuse[i * 4 + 2] = monti::capture::FloatToHalf(val * 0.25f);
        noisy_diffuse[i * 4 + 3] = monti::capture::FloatToHalf(1.0f);
    }

    // Generate float depth data
    std::vector<float> depth(kPixels);
    for (uint32_t i = 0; i < kPixels; ++i)
        depth[i] = static_cast<float>(i) * 0.1f;

    // Generate target data
    std::vector<float> ref_diffuse(static_cast<size_t>(kPixels) * 4, 0.5f);

    monti::capture::RawInputFrame input{};
    input.noisy_diffuse = noisy_diffuse.data();
    input.linear_depth = depth.data();

    monti::capture::TargetFrame target{};
    target.ref_diffuse = ref_diffuse.data();

    REQUIRE(writer->WriteFrameRaw(input, target, 0));

    // Verify input EXR
    std::string input_path = kTestOutputDir + "/frame_000000_input.exr";
    REQUIRE(std::filesystem::exists(input_path));

    EXRHeader header;
    EXRImage image;
    std::vector<int> stored_types;
    int num_ch = LoadExr(input_path, header, image, stored_types);
    REQUIRE(num_ch > 0);

    // Verify FP16 channels are stored as HALF
    REQUIRE(ChannelHasType(header, stored_types, "noisy_diffuse.R", TINYEXR_PIXELTYPE_HALF));
    REQUIRE(ChannelHasType(header, stored_types, "noisy_diffuse.G", TINYEXR_PIXELTYPE_HALF));

    // Verify depth is stored as FLOAT
    REQUIRE(ChannelHasType(header, stored_types, "depth.Z", TINYEXR_PIXELTYPE_FLOAT));

    // Verify pixel values round-trip for FP16 channel (loaded back as float)
    int idx_r = FindChannel(header, "noisy_diffuse.R");
    REQUIRE(idx_r >= 0);
    auto* loaded_r = reinterpret_cast<const float*>(image.images[idx_r]);
    for (uint32_t i = 0; i < kPixels; ++i) {
        float expected = monti::capture::HalfToFloat(noisy_diffuse[i * 4 + 0]);
        float actual = loaded_r[i];
        // Raw half → written as half → loaded as float should be exact
        REQUIRE(expected == actual);
    }

    // Verify depth round-trip (exact for float)
    int idx_depth = FindChannel(header, "depth.Z");
    REQUIRE(idx_depth >= 0);
    auto* loaded_depth = reinterpret_cast<const float*>(image.images[idx_depth]);
    for (uint32_t i = 0; i < kPixels; ++i)
        REQUIRE(depth[i] == loaded_depth[i]);

    FreeEXRImage(&image);
    FreeEXRHeader(&header);
}
