#include <monti/capture/Writer.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <format>
#include <vector>

#include <tinyexr.h>

namespace monti::capture {

namespace {

float ScaleFactor(ScaleMode mode) {
    switch (mode) {
    case ScaleMode::kNative:      return 1.0f;
    case ScaleMode::kQuality:     return 1.5f;
    case ScaleMode::kPerformance: return 2.0f;
    }
    return 1.0f;
}

uint32_t ComputeTargetDim(uint32_t input_dim, float factor) {
    return static_cast<uint32_t>(
        static_cast<float>(input_dim) * factor * 0.5f) * 2;
}

constexpr const char* kRGBA[] = {"R", "G", "B", "A"};
constexpr const char* kRGB[]  = {"R", "G", "B"};
constexpr const char* kXYZW[] = {"X", "Y", "Z", "W"};
constexpr const char* kZ[]    = {"Z"};
constexpr const char* kXY[]   = {"X", "Y"};

// A single de-interleaved channel ready for EXR output.
struct ChannelEntry {
    std::string name;
    std::vector<float> data;  // per-pixel float data (even for HALF output)
    int pixel_type;           // TINYEXR_PIXELTYPE_FLOAT or TINYEXR_PIXELTYPE_HALF
};

// A channel with raw half data (no float conversion needed).
struct RawHalfChannelEntry {
    std::string name;
    std::vector<uint16_t> data;  // per-pixel raw FP16 data
};

// Unified channel entry for WriteExr: either float or raw half.
struct ExrChannel {
    std::string name;
    std::vector<float> float_data;      // used when is_raw_half == false
    std::vector<uint16_t> half_data;    // used when is_raw_half == true
    int pixel_type;                      // TINYEXR_PIXELTYPE_FLOAT or TINYEXR_PIXELTYPE_HALF
    bool is_raw_half = false;            // true = half_data is the source
};

// De-interleave an interleaved float buffer into separate per-channel arrays
// and append them to `channels`. `prefix` is the EXR layer name (e.g.
// "noisy_diffuse"), `suffixes` are the per-component suffixes (e.g. {"R","G","B","A"}),
// `components` is the stride, and `pixel_type` is the requested output precision.
void AppendChannelGroup(std::vector<ChannelEntry>& channels,
                        std::string_view prefix,
                        const float* data,
                        uint32_t pixel_count,
                        const char* const* suffixes,
                        int components,
                        int pixel_type) {
    if (!data) return;

    for (int c = 0; c < components; ++c) {
        ChannelEntry entry;
        entry.name = std::format("{}.{}", prefix, suffixes[static_cast<size_t>(c)]);
        entry.pixel_type = pixel_type;
        entry.data.resize(pixel_count);
        for (uint32_t i = 0; i < pixel_count; ++i)
            entry.data[i] = data[static_cast<size_t>(i) * components + c];
        channels.push_back(std::move(entry));
    }
}

// Append raw FP16 channels to a unified channel list.
void AppendRawHalfChannelGroup(std::vector<ExrChannel>& channels,
                               std::string_view prefix,
                               const uint16_t* data,
                               uint32_t pixel_count,
                               const char* const* suffixes,
                               int components) {
    if (!data) return;

    for (int c = 0; c < components; ++c) {
        ExrChannel entry;
        entry.name = std::format("{}.{}", prefix, suffixes[static_cast<size_t>(c)]);
        entry.pixel_type = TINYEXR_PIXELTYPE_HALF;
        entry.is_raw_half = true;
        entry.half_data.resize(pixel_count);
        for (uint32_t i = 0; i < pixel_count; ++i)
            entry.half_data[i] = data[static_cast<size_t>(i) * components + c];
        channels.push_back(std::move(entry));
    }
}

// Append float channels to a unified channel list.
void AppendFloatChannelGroup(std::vector<ExrChannel>& channels,
                             std::string_view prefix,
                             const float* data,
                             uint32_t pixel_count,
                             const char* const* suffixes,
                             int components,
                             int pixel_type) {
    if (!data) return;

    for (int c = 0; c < components; ++c) {
        ExrChannel entry;
        entry.name = std::format("{}.{}", prefix, suffixes[static_cast<size_t>(c)]);
        entry.pixel_type = pixel_type;
        entry.is_raw_half = false;
        entry.float_data.resize(pixel_count);
        for (uint32_t i = 0; i < pixel_count; ++i)
            entry.float_data[i] = data[static_cast<size_t>(i) * components + c];
        channels.push_back(std::move(entry));
    }
}

// Write a set of channels to an EXR file. Returns true on success.
bool WriteExr(const std::string& path, uint32_t width, uint32_t height,
              std::vector<ChannelEntry>& channels) {
    if (channels.empty()) return true;

    // EXR spec requires channels sorted alphabetically by name
    std::ranges::sort(channels, {}, &ChannelEntry::name);

    auto num_channels = static_cast<int>(channels.size());

    EXRHeader header;
    InitEXRHeader(&header);
    header.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;
    header.num_channels = num_channels;
    header.channels = static_cast<EXRChannelInfo*>(
        malloc(sizeof(EXRChannelInfo) * static_cast<size_t>(num_channels)));
    header.pixel_types = static_cast<int*>(
        malloc(sizeof(int) * static_cast<size_t>(num_channels)));
    header.requested_pixel_types = static_cast<int*>(
        malloc(sizeof(int) * static_cast<size_t>(num_channels)));

    std::vector<unsigned char*> image_ptrs(static_cast<size_t>(num_channels));

    for (int i = 0; i < num_channels; ++i) {
        auto& ch = channels[static_cast<size_t>(i)];

        // Channel info
        memset(&header.channels[i], 0, sizeof(EXRChannelInfo));
#ifdef _MSC_VER
        strncpy_s(header.channels[i].name, ch.name.c_str(), 255);
#else
        strncpy(header.channels[i].name, ch.name.c_str(), 255);
        header.channels[i].name[255] = '\0';
#endif

        // Input is always FLOAT; requested type controls output precision
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
        header.requested_pixel_types[i] = ch.pixel_type;

        image_ptrs[static_cast<size_t>(i)] =
            reinterpret_cast<unsigned char*>(ch.data.data());
    }

    EXRImage image;
    InitEXRImage(&image);
    image.num_channels = num_channels;
    image.width = static_cast<int>(width);
    image.height = static_cast<int>(height);
    image.images = image_ptrs.data();

    const char* err = nullptr;
    int ret = SaveEXRImageToFile(&image, &header, path.c_str(), &err);

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);

    if (ret != TINYEXR_SUCCESS) {
        if (err) FreeEXRErrorMessage(err);
        return false;
    }
    return true;
}

// Write unified channels (mix of float and raw half) to EXR.
bool WriteExrUnified(const std::string& path, uint32_t width, uint32_t height,
                     std::vector<ExrChannel>& channels) {
    if (channels.empty()) return true;

    std::ranges::sort(channels, {}, &ExrChannel::name);

    auto num_channels = static_cast<int>(channels.size());

    EXRHeader header;
    InitEXRHeader(&header);
    header.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;
    header.num_channels = num_channels;
    header.channels = static_cast<EXRChannelInfo*>(
        malloc(sizeof(EXRChannelInfo) * static_cast<size_t>(num_channels)));
    header.pixel_types = static_cast<int*>(
        malloc(sizeof(int) * static_cast<size_t>(num_channels)));
    header.requested_pixel_types = static_cast<int*>(
        malloc(sizeof(int) * static_cast<size_t>(num_channels)));

    std::vector<unsigned char*> image_ptrs(static_cast<size_t>(num_channels));

    for (int i = 0; i < num_channels; ++i) {
        auto& ch = channels[static_cast<size_t>(i)];

        memset(&header.channels[i], 0, sizeof(EXRChannelInfo));
#ifdef _MSC_VER
        strncpy_s(header.channels[i].name, ch.name.c_str(), 255);
#else
        strncpy(header.channels[i].name, ch.name.c_str(), 255);
        header.channels[i].name[255] = '\0';
#endif

        if (ch.is_raw_half) {
            // Raw half: tell tinyexr both pixel_type and requested_pixel_type are HALF.
            // tinyexr copies the half data directly without float_to_half conversion.
            header.pixel_types[i] = TINYEXR_PIXELTYPE_HALF;
            header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF;
            image_ptrs[static_cast<size_t>(i)] =
                reinterpret_cast<unsigned char*>(ch.half_data.data());
        } else {
            // Float data — tinyexr converts float→half if requested_pixel_type is HALF
            header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
            header.requested_pixel_types[i] = ch.pixel_type;
            image_ptrs[static_cast<size_t>(i)] =
                reinterpret_cast<unsigned char*>(ch.float_data.data());
        }
    }

    EXRImage image;
    InitEXRImage(&image);
    image.num_channels = num_channels;
    image.width = static_cast<int>(width);
    image.height = static_cast<int>(height);
    image.images = image_ptrs.data();

    const char* err = nullptr;
    int ret = SaveEXRImageToFile(&image, &header, path.c_str(), &err);

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);

    if (ret != TINYEXR_SUCCESS) {
        if (err) FreeEXRErrorMessage(err);
        return false;
    }
    return true;
}

}  // namespace

std::unique_ptr<Writer> Writer::Create(const WriterDesc& desc) {
    if (desc.input_width == 0 || desc.input_height == 0) return nullptr;

    std::error_code ec;
    std::filesystem::create_directories(desc.output_dir, ec);
    if (ec) return nullptr;

    auto writer = std::unique_ptr<Writer>(new Writer());
    writer->output_dir_ = desc.output_dir;
    writer->input_width_ = desc.input_width;
    writer->input_height_ = desc.input_height;
    float factor = ScaleFactor(desc.scale_mode);
    writer->target_width_  = ComputeTargetDim(desc.input_width, factor);
    writer->target_height_ = ComputeTargetDim(desc.input_height, factor);
    return writer;
}

Writer::~Writer() = default;

uint32_t Writer::TargetWidth() const { return target_width_; }
uint32_t Writer::TargetHeight() const { return target_height_; }

bool Writer::WriteFrame(const InputFrame& input, const TargetFrame& target,
                        uint32_t frame_index) {
    uint32_t input_pixels = input_width_ * input_height_;
    uint32_t target_pixels = target_width_ * target_height_;

    // --- Input EXR ---
    {
        std::vector<ChannelEntry> channels;

        AppendChannelGroup(channels, "noisy_diffuse", input.noisy_diffuse,
                           input_pixels, kRGBA, 4, TINYEXR_PIXELTYPE_HALF);
        AppendChannelGroup(channels, "noisy_specular", input.noisy_specular,
                           input_pixels, kRGBA, 4, TINYEXR_PIXELTYPE_HALF);
        AppendChannelGroup(channels, "diffuse_albedo", input.diffuse_albedo,
                           input_pixels, kRGB, 3, TINYEXR_PIXELTYPE_HALF);
        AppendChannelGroup(channels, "specular_albedo", input.specular_albedo,
                           input_pixels, kRGB, 3, TINYEXR_PIXELTYPE_HALF);
        AppendChannelGroup(channels, "normal", input.world_normals,
                           input_pixels, kXYZW, 4, TINYEXR_PIXELTYPE_HALF);
        AppendChannelGroup(channels, "depth", input.linear_depth,
                           input_pixels, kZ, 1, TINYEXR_PIXELTYPE_FLOAT);
        AppendChannelGroup(channels, "motion", input.motion_vectors,
                           input_pixels, kXY, 2, TINYEXR_PIXELTYPE_HALF);

        if (!channels.empty()) {
            auto path = std::format("{}/frame_{:06d}_input.exr", output_dir_, frame_index);
            if (!WriteExr(path, input_width_, input_height_, channels))
                return false;
        }
    }

    // --- Target EXR ---
    {
        std::vector<ChannelEntry> channels;

        AppendChannelGroup(channels, "ref_diffuse", target.ref_diffuse,
                           target_pixels, kRGBA, 4, TINYEXR_PIXELTYPE_FLOAT);
        AppendChannelGroup(channels, "ref_specular", target.ref_specular,
                           target_pixels, kRGBA, 4, TINYEXR_PIXELTYPE_FLOAT);

        if (!channels.empty()) {
            auto path = std::format("{}/frame_{:06d}_target.exr", output_dir_, frame_index);
            if (!WriteExr(path, target_width_, target_height_, channels))
                return false;
        }
    }

    return true;
}

bool Writer::WriteFrameRaw(const RawInputFrame& input, const TargetFrame& target,
                           uint32_t frame_index) {
    uint32_t input_pixels = input_width_ * input_height_;
    uint32_t target_pixels = target_width_ * target_height_;

    // --- Input EXR (unified: raw half + float channels) ---
    {
        std::vector<ExrChannel> channels;

        AppendRawHalfChannelGroup(channels, "noisy_diffuse", input.noisy_diffuse,
                                  input_pixels, kRGBA, 4);
        AppendRawHalfChannelGroup(channels, "noisy_specular", input.noisy_specular,
                                  input_pixels, kRGBA, 4);
        AppendFloatChannelGroup(channels, "diffuse_albedo", input.diffuse_albedo,
                                input_pixels, kRGB, 3, TINYEXR_PIXELTYPE_HALF);
        AppendFloatChannelGroup(channels, "specular_albedo", input.specular_albedo,
                                input_pixels, kRGB, 3, TINYEXR_PIXELTYPE_HALF);
        AppendRawHalfChannelGroup(channels, "normal", input.world_normals,
                                  input_pixels, kXYZW, 4);
        AppendFloatChannelGroup(channels, "depth", input.linear_depth,
                                input_pixels, kZ, 1, TINYEXR_PIXELTYPE_FLOAT);
        AppendRawHalfChannelGroup(channels, "motion", input.motion_vectors,
                                  input_pixels, kXY, 2);

        if (!channels.empty()) {
            auto path = std::format("{}/frame_{:06d}_input.exr", output_dir_, frame_index);
            if (!WriteExrUnified(path, input_width_, input_height_, channels))
                return false;
        }
    }

    // --- Target EXR (float only, same as WriteFrame) ---
    {
        std::vector<ChannelEntry> channels;

        AppendChannelGroup(channels, "ref_diffuse", target.ref_diffuse,
                           target_pixels, kRGBA, 4, TINYEXR_PIXELTYPE_FLOAT);
        AppendChannelGroup(channels, "ref_specular", target.ref_specular,
                           target_pixels, kRGBA, 4, TINYEXR_PIXELTYPE_FLOAT);

        if (!channels.empty()) {
            auto path = std::format("{}/frame_{:06d}_target.exr", output_dir_, frame_index);
            if (!WriteExr(path, target_width_, target_height_, channels))
                return false;
        }
    }

    return true;
}

} // namespace monti::capture
