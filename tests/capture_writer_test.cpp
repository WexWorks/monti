#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <tinyexr.h>

#include <monti/capture/Writer.h>

namespace {

// Test output directory (cleaned up after each test section)
const std::string kTestOutputDir = "tests/datagen/capture_writer_test";

// Small test resolution
constexpr uint32_t kInputWidth = 64;
constexpr uint32_t kInputHeight = 48;

// Fill a float buffer with a deterministic pattern. Values stay in [0.001, ~1.0]
// to remain in FP16's high-precision range.
void FillPattern(std::vector<float>& buf, uint32_t pixel_count, int components) {
    buf.resize(static_cast<size_t>(pixel_count) * components);
    for (uint32_t i = 0; i < pixel_count; ++i) {
        for (int c = 0; c < components; ++c) {
            // Use modular arithmetic to keep values small
            uint32_t idx = i * static_cast<uint32_t>(components) + static_cast<uint32_t>(c);
            buf[static_cast<size_t>(i) * components + c] =
                static_cast<float>((idx % 997) + 1) * 0.001f;
        }
    }
}

// Load an EXR file via tinyexr low-level API. Returns channel count on success,
// -1 on failure. Fills header and image for inspection; caller must free them.
// `stored_types` receives the original per-channel pixel types from the file
// (before tinyexr overwrites them when loading as FLOAT).
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

    // Save original pixel types before tinyexr overwrites them
    stored_types.assign(header.pixel_types,
                        header.pixel_types + header.num_channels);

    // Request FLOAT for all channels so we can read back as float
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

// Find a channel index by name in a loaded EXR header. Returns -1 if not found.
int FindChannel(const EXRHeader& header, const char* name) {
    for (int i = 0; i < header.num_channels; ++i) {
        if (std::strcmp(header.channels[i].name, name) == 0) return i;
    }
    return -1;
}

// Check that a channel's stored pixel type matches the expected type.
bool ChannelHasType(const EXRHeader& header, const std::vector<int>& stored_types,
                    const char* name, int expected_type) {
    int idx = FindChannel(header, name);
    if (idx < 0) return false;
    return stored_types[static_cast<size_t>(idx)] == expected_type;
}

// Compare a loaded channel's float data against expected interleaved source.
// `component` is the index within the interleaved stride, `components` is the
// total stride. Uses absolute tolerance `tol`.
bool CompareChannel(const EXRImage& image, int channel_idx,
                    const float* interleaved_src, uint32_t pixel_count,
                    int component, int components, float tol) {
    auto* loaded = reinterpret_cast<const float*>(
        image.images[channel_idx]);
    for (uint32_t i = 0; i < pixel_count; ++i) {
        float expected = interleaved_src[static_cast<size_t>(i) * components + component];
        float actual = loaded[i];
        if (std::abs(expected - actual) > tol) return false;
    }
    return true;
}

struct ScopedCleanup {
    ~ScopedCleanup() {
        std::error_code ec;
        std::filesystem::remove_all(kTestOutputDir, ec);
    }
};

}  // namespace

TEST_CASE("Writer::Create validates inputs", "[capture]") {
    SECTION("zero width returns nullptr") {
        monti::capture::WriterDesc desc{kTestOutputDir, 0, kInputHeight};
        REQUIRE(monti::capture::Writer::Create(desc) == nullptr);
    }
    SECTION("zero height returns nullptr") {
        monti::capture::WriterDesc desc{kTestOutputDir, kInputWidth, 0};
        REQUIRE(monti::capture::Writer::Create(desc) == nullptr);
    }
}

TEST_CASE("Writer target resolution computation", "[capture]") {
    ScopedCleanup cleanup;

    SECTION("Performance mode (2x)") {
        monti::capture::WriterDesc desc{
            kTestOutputDir, kInputWidth, kInputHeight,
            monti::capture::ScaleMode::kPerformance};
        auto writer = monti::capture::Writer::Create(desc);
        REQUIRE(writer);
        REQUIRE(writer->TargetWidth() == 128);
        REQUIRE(writer->TargetHeight() == 96);
    }
    SECTION("Quality mode (1.5x)") {
        monti::capture::WriterDesc desc{
            kTestOutputDir, kInputWidth, kInputHeight,
            monti::capture::ScaleMode::kQuality};
        auto writer = monti::capture::Writer::Create(desc);
        REQUIRE(writer);
        REQUIRE(writer->TargetWidth() == 96);
        REQUIRE(writer->TargetHeight() == 72);
    }
    SECTION("Native mode (1x)") {
        monti::capture::WriterDesc desc{
            kTestOutputDir, kInputWidth, kInputHeight,
            monti::capture::ScaleMode::kNative};
        auto writer = monti::capture::Writer::Create(desc);
        REQUIRE(writer);
        REQUIRE(writer->TargetWidth() == kInputWidth);
        REQUIRE(writer->TargetHeight() == kInputHeight);
    }
}

TEST_CASE("Writer writes input and target EXR files", "[capture]") {
    ScopedCleanup cleanup;

    uint32_t input_pixels = kInputWidth * kInputHeight;

    monti::capture::WriterDesc desc{
        kTestOutputDir, kInputWidth, kInputHeight,
        monti::capture::ScaleMode::kPerformance};
    auto writer = monti::capture::Writer::Create(desc);
    REQUIRE(writer);

    uint32_t target_w = writer->TargetWidth();
    uint32_t target_h = writer->TargetHeight();
    uint32_t target_pixels = target_w * target_h;

    // Generate test data
    std::vector<float> noisy_diffuse, noisy_specular;
    std::vector<float> diffuse_albedo, specular_albedo;
    std::vector<float> normals, depth, motion;
    std::vector<float> ref_diffuse, ref_specular;

    FillPattern(noisy_diffuse, input_pixels, 4);
    FillPattern(noisy_specular, input_pixels, 4);
    FillPattern(diffuse_albedo, input_pixels, 3);
    FillPattern(specular_albedo, input_pixels, 3);
    FillPattern(normals, input_pixels, 4);
    FillPattern(depth, input_pixels, 1);
    FillPattern(motion, input_pixels, 2);
    FillPattern(ref_diffuse, target_pixels, 4);
    FillPattern(ref_specular, target_pixels, 4);

    monti::capture::InputFrame input_frame{};
    input_frame.noisy_diffuse = noisy_diffuse.data();
    input_frame.noisy_specular = noisy_specular.data();
    input_frame.diffuse_albedo = diffuse_albedo.data();
    input_frame.specular_albedo = specular_albedo.data();
    input_frame.world_normals = normals.data();
    input_frame.linear_depth = depth.data();
    input_frame.motion_vectors = motion.data();

    monti::capture::TargetFrame target_frame{};
    target_frame.ref_diffuse = ref_diffuse.data();
    target_frame.ref_specular = ref_specular.data();

    REQUIRE(writer->WriteFrame(input_frame, target_frame));

    std::string input_path = kTestOutputDir + "/input.exr";
    std::string target_path = kTestOutputDir + "/target.exr";

    SECTION("Input EXR has correct structure") {
        REQUIRE(std::filesystem::exists(input_path));
        REQUIRE(std::filesystem::file_size(input_path) > 0);

        EXRHeader header;
        EXRImage image;
        std::vector<int> stored_types;
        int num_ch = LoadExr(input_path, header, image, stored_types);
        REQUIRE(num_ch > 0);
        REQUIRE(image.width == static_cast<int>(kInputWidth));
        REQUIRE(image.height == static_cast<int>(kInputHeight));

        // Verify all expected channel names are present
        // diffuse: R,G,B,A (4) + specular: R,G,B,A (4) +
        // albedo_d: R,G,B (3) + albedo_s: R,G,B (3) +
        // normal: X,Y,Z,W (4) + depth: Z (1) + motion: X,Y (2) = 21
        REQUIRE(num_ch == 21);

        // Spot-check channel names
        REQUIRE(FindChannel(header, "diffuse.R") >= 0);
        REQUIRE(FindChannel(header, "diffuse.A") >= 0);
        REQUIRE(FindChannel(header, "specular.B") >= 0);
        REQUIRE(FindChannel(header, "albedo_d.R") >= 0);
        REQUIRE(FindChannel(header, "albedo_s.G") >= 0);
        REQUIRE(FindChannel(header, "normal.X") >= 0);
        REQUIRE(FindChannel(header, "normal.W") >= 0);
        REQUIRE(FindChannel(header, "depth.Z") >= 0);
        REQUIRE(FindChannel(header, "motion.X") >= 0);
        REQUIRE(FindChannel(header, "motion.Y") >= 0);

        // Verify per-channel bit depths (stored_types reflects on-disk type)
        REQUIRE(ChannelHasType(header, stored_types, "diffuse.R", TINYEXR_PIXELTYPE_HALF));
        REQUIRE(ChannelHasType(header, stored_types, "specular.G", TINYEXR_PIXELTYPE_HALF));
        REQUIRE(ChannelHasType(header, stored_types, "albedo_d.B", TINYEXR_PIXELTYPE_HALF));
        REQUIRE(ChannelHasType(header, stored_types, "albedo_s.R", TINYEXR_PIXELTYPE_HALF));
        REQUIRE(ChannelHasType(header, stored_types, "normal.Z", TINYEXR_PIXELTYPE_HALF));
        REQUIRE(ChannelHasType(header, stored_types, "depth.Z", TINYEXR_PIXELTYPE_FLOAT));
        REQUIRE(ChannelHasType(header, stored_types, "motion.Y", TINYEXR_PIXELTYPE_HALF));

        FreeEXRImage(&image);
        FreeEXRHeader(&header);
    }

    SECTION("Input EXR pixel values round-trip") {
        EXRHeader header;
        EXRImage image;
        std::vector<int> stored_types;
        int num_ch = LoadExr(input_path, header, image, stored_types);
        REQUIRE(num_ch > 0);

        // HALF channels: tolerance ~0.001 for values in our range
        constexpr float kHalfTol = 0.01f;
        // FLOAT channels: exact round-trip
        constexpr float kFloatTol = 0.0f;

        // noisy_diffuse RGBA
        int idx_r = FindChannel(header, "diffuse.R");
        int idx_g = FindChannel(header, "diffuse.G");
        int idx_b = FindChannel(header, "diffuse.B");
        int idx_a = FindChannel(header, "diffuse.A");
        REQUIRE(idx_r >= 0);
        REQUIRE(idx_g >= 0);
        REQUIRE(idx_b >= 0);
        REQUIRE(idx_a >= 0);
        REQUIRE(CompareChannel(image, idx_r, noisy_diffuse.data(), input_pixels, 0, 4, kHalfTol));
        REQUIRE(CompareChannel(image, idx_g, noisy_diffuse.data(), input_pixels, 1, 4, kHalfTol));
        REQUIRE(CompareChannel(image, idx_b, noisy_diffuse.data(), input_pixels, 2, 4, kHalfTol));
        REQUIRE(CompareChannel(image, idx_a, noisy_diffuse.data(), input_pixels, 3, 4, kHalfTol));

        // depth.Z (FLOAT — exact round-trip)
        int idx_depth = FindChannel(header, "depth.Z");
        REQUIRE(idx_depth >= 0);
        REQUIRE(CompareChannel(image, idx_depth, depth.data(), input_pixels, 0, 1, kFloatTol));

        // motion.XY
        int idx_mx = FindChannel(header, "motion.X");
        int idx_my = FindChannel(header, "motion.Y");
        REQUIRE(idx_mx >= 0);
        REQUIRE(idx_my >= 0);
        REQUIRE(CompareChannel(image, idx_mx, motion.data(), input_pixels, 0, 2, kHalfTol));
        REQUIRE(CompareChannel(image, idx_my, motion.data(), input_pixels, 1, 2, kHalfTol));

        FreeEXRImage(&image);
        FreeEXRHeader(&header);
    }

    SECTION("Target EXR has correct structure and values") {
        REQUIRE(std::filesystem::exists(target_path));
        REQUIRE(std::filesystem::file_size(target_path) > 0);

        EXRHeader header;
        EXRImage image;
        std::vector<int> stored_types;
        int num_ch = LoadExr(target_path, header, image, stored_types);
        REQUIRE(num_ch > 0);
        REQUIRE(image.width == static_cast<int>(target_w));
        REQUIRE(image.height == static_cast<int>(target_h));

        // ref_diffuse: R,G,B,A (4) + ref_specular: R,G,B,A (4) = 8
        REQUIRE(num_ch == 8);

        // All channels should be FLOAT
        REQUIRE(ChannelHasType(header, stored_types, "diffuse.R", TINYEXR_PIXELTYPE_FLOAT));
        REQUIRE(ChannelHasType(header, stored_types, "diffuse.A", TINYEXR_PIXELTYPE_FLOAT));
        REQUIRE(ChannelHasType(header, stored_types, "specular.R", TINYEXR_PIXELTYPE_FLOAT));
        REQUIRE(ChannelHasType(header, stored_types, "specular.A", TINYEXR_PIXELTYPE_FLOAT));

        // Verify exact round-trip for FP32 channels
        constexpr float kFloatTol = 0.0f;
        int idx_r = FindChannel(header, "diffuse.R");
        int idx_a = FindChannel(header, "diffuse.A");
        REQUIRE(idx_r >= 0);
        REQUIRE(idx_a >= 0);
        REQUIRE(CompareChannel(image, idx_r, ref_diffuse.data(), target_pixels, 0, 4, kFloatTol));
        REQUIRE(CompareChannel(image, idx_a, ref_diffuse.data(), target_pixels, 3, 4, kFloatTol));

        int idx_sr = FindChannel(header, "specular.R");
        REQUIRE(idx_sr >= 0);
        REQUIRE(CompareChannel(image, idx_sr, ref_specular.data(), target_pixels, 0, 4, kFloatTol));

        FreeEXRImage(&image);
        FreeEXRHeader(&header);
    }
}

TEST_CASE("Writer omits null pointer channels", "[capture]") {
    ScopedCleanup cleanup;

    monti::capture::WriterDesc desc{
        kTestOutputDir, kInputWidth, kInputHeight,
        monti::capture::ScaleMode::kPerformance};
    auto writer = monti::capture::Writer::Create(desc);
    REQUIRE(writer);

    uint32_t input_pixels = kInputWidth * kInputHeight;
    uint32_t target_pixels = writer->TargetWidth() * writer->TargetHeight();

    // Provide only noisy_diffuse and depth for input, only ref_diffuse for target
    std::vector<float> noisy_diffuse, depth_buf, ref_diffuse;
    FillPattern(noisy_diffuse, input_pixels, 4);
    FillPattern(depth_buf, input_pixels, 1);
    FillPattern(ref_diffuse, target_pixels, 4);

    monti::capture::InputFrame input_frame{};
    input_frame.noisy_diffuse = noisy_diffuse.data();
    input_frame.linear_depth = depth_buf.data();
    // Leave other pointers null

    monti::capture::TargetFrame target_frame{};
    target_frame.ref_diffuse = ref_diffuse.data();
    // Leave ref_specular null

    REQUIRE(writer->WriteFrame(input_frame, target_frame, "subdir"));

    std::string input_path = kTestOutputDir + "/subdir/input.exr";
    std::string target_path = kTestOutputDir + "/subdir/target.exr";

    SECTION("Input EXR has only enabled channels") {
        EXRHeader header;
        EXRImage image;
        std::vector<int> stored_types;
        int num_ch = LoadExr(input_path, header, image, stored_types);
        REQUIRE(num_ch > 0);

        // noisy_diffuse (4) + depth (1) = 5 channels
        REQUIRE(num_ch == 5);

        REQUIRE(FindChannel(header, "diffuse.R") >= 0);
        REQUIRE(FindChannel(header, "depth.Z") >= 0);

        // Channels from null pointers should be absent
        REQUIRE(FindChannel(header, "specular.R") == -1);
        REQUIRE(FindChannel(header, "albedo_d.R") == -1);
        REQUIRE(FindChannel(header, "albedo_s.R") == -1);
        REQUIRE(FindChannel(header, "normal.X") == -1);
        REQUIRE(FindChannel(header, "motion.X") == -1);

        FreeEXRImage(&image);
        FreeEXRHeader(&header);
    }

    SECTION("Target EXR has only enabled channels") {
        std::vector<int> stored_types;
        EXRHeader header;
        EXRImage image;
        int num_ch = LoadExr(target_path, header, image, stored_types);
        REQUIRE(num_ch > 0);

        // Only ref_diffuse (4 channels)
        REQUIRE(num_ch == 4);

        REQUIRE(FindChannel(header, "diffuse.R") >= 0);
        REQUIRE(FindChannel(header, "specular.R") == -1);

        FreeEXRImage(&image);
        FreeEXRHeader(&header);
    }
}

TEST_CASE("Writer writes to subdirectory", "[capture]") {
    ScopedCleanup cleanup;

    monti::capture::WriterDesc desc{
        kTestOutputDir, kInputWidth, kInputHeight,
        monti::capture::ScaleMode::kNative};
    auto writer = monti::capture::Writer::Create(desc);
    REQUIRE(writer);

    uint32_t pixels = kInputWidth * kInputHeight;
    std::vector<float> ref_diffuse;
    FillPattern(ref_diffuse, pixels, 4);

    monti::capture::InputFrame input_frame{};
    monti::capture::TargetFrame target_frame{};
    target_frame.ref_diffuse = ref_diffuse.data();

    REQUIRE(writer->WriteFrame(input_frame, target_frame, "vp_0"));

    // Input EXR should not be created when all input pointers are null
    REQUIRE_FALSE(std::filesystem::exists(kTestOutputDir + "/vp_0/input.exr"));
    REQUIRE(std::filesystem::exists(kTestOutputDir + "/vp_0/target.exr"));
}

// Helper: parse only the EXR header to get compression type, without loading image data.
int GetExrCompressionType(const std::string& path) {
    EXRVersion version;
    if (ParseEXRVersionFromFile(&version, path.c_str()) != TINYEXR_SUCCESS)
        return -1;
    EXRHeader header;
    InitEXRHeader(&header);
    const char* err = nullptr;
    if (ParseEXRHeaderFromFile(&header, &version, path.c_str(), &err) != TINYEXR_SUCCESS) {
        if (err) FreeEXRErrorMessage(err);
        return -1;
    }
    int comp = header.compression_type;
    FreeEXRHeader(&header);
    return comp;
}

TEST_CASE("Writer respects ExrCompression setting", "[capture]") {
    ScopedCleanup cleanup;

    uint32_t input_pixels = kInputWidth * kInputHeight;

    std::vector<float> noisy_diffuse, ref_diffuse;
    FillPattern(noisy_diffuse, input_pixels, 4);
    FillPattern(ref_diffuse, input_pixels, 4);

    monti::capture::InputFrame input_frame{};
    input_frame.noisy_diffuse = noisy_diffuse.data();
    monti::capture::TargetFrame target_frame{};
    target_frame.ref_diffuse = ref_diffuse.data();

    SECTION("Default (kNone) writes uncompressed EXR") {
        monti::capture::WriterDesc desc{
            kTestOutputDir, kInputWidth, kInputHeight,
            monti::capture::ScaleMode::kNative};
        // compression defaults to kNone
        auto writer = monti::capture::Writer::Create(desc);
        REQUIRE(writer);
        REQUIRE(writer->WriteFrame(input_frame, target_frame));

        REQUIRE(GetExrCompressionType(kTestOutputDir + "/input.exr") ==
                TINYEXR_COMPRESSIONTYPE_NONE);
        REQUIRE(GetExrCompressionType(kTestOutputDir + "/target.exr") ==
                TINYEXR_COMPRESSIONTYPE_NONE);
    }

    SECTION("kZip writes ZIP-compressed EXR") {
        monti::capture::WriterDesc desc{
            kTestOutputDir, kInputWidth, kInputHeight,
            monti::capture::ScaleMode::kNative,
            monti::capture::ExrCompression::kZip};
        auto writer = monti::capture::Writer::Create(desc);
        REQUIRE(writer);
        REQUIRE(writer->WriteFrame(input_frame, target_frame));

        REQUIRE(GetExrCompressionType(kTestOutputDir + "/input.exr") ==
                TINYEXR_COMPRESSIONTYPE_ZIP);
        REQUIRE(GetExrCompressionType(kTestOutputDir + "/target.exr") ==
                TINYEXR_COMPRESSIONTYPE_ZIP);
    }

    SECTION("Round-trip pixel data matches for both compression modes") {
        for (auto compression : {monti::capture::ExrCompression::kNone,
                                  monti::capture::ExrCompression::kZip}) {
            monti::capture::WriterDesc desc{
                kTestOutputDir, kInputWidth, kInputHeight,
                monti::capture::ScaleMode::kNative, compression};
            auto writer = monti::capture::Writer::Create(desc);
            REQUIRE(writer);
            REQUIRE(writer->WriteFrame(input_frame, target_frame));

            EXRHeader header;
            EXRImage image;
            std::vector<int> stored_types;
            int num_ch = LoadExr(kTestOutputDir + "/input.exr", header, image, stored_types);
            REQUIRE(num_ch > 0);

            constexpr float kHalfTol = 0.01f;
            int idx_r = FindChannel(header, "diffuse.R");
            REQUIRE(idx_r >= 0);
            REQUIRE(CompareChannel(image, idx_r, noisy_diffuse.data(),
                                   input_pixels, 0, 4, kHalfTol));

            FreeEXRImage(&image);
            FreeEXRHeader(&header);

            // Clean up for next iteration
            std::error_code ec;
            std::filesystem::remove_all(kTestOutputDir, ec);
        }
    }
}
