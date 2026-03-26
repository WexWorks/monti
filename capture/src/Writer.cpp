#include <monti/capture/Writer.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <format>
#include <span>
#include <utility>
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

// Unified channel entry for WriteExr: either float or raw half.
struct ExrChannel {
    std::string name;
    std::vector<float> float_data;      // used when is_raw_half == false
    std::vector<uint16_t> half_data;    // used when is_raw_half == true
    int pixel_type;                      // TINYEXR_PIXELTYPE_FLOAT or TINYEXR_PIXELTYPE_HALF
    bool is_raw_half = false;            // true = half_data is the source
};

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

int ToTinyexrCompression(ExrCompression compression) {
    switch (compression) {
    case ExrCompression::kNone: return TINYEXR_COMPRESSIONTYPE_NONE;
    case ExrCompression::kZip:  return TINYEXR_COMPRESSIONTYPE_ZIP;
    }
    return TINYEXR_COMPRESSIONTYPE_NONE;
}

// Write channels (mix of float and raw half) to EXR. Returns true on success.
bool WriteExr(const std::string& path, uint32_t width, uint32_t height,
              std::vector<ExrChannel>& channels, ExrCompression compression,
              std::span<const std::pair<std::string, float>> metadata = {}) {
    if (channels.empty()) return true;

    std::ranges::sort(channels, {}, &ExrChannel::name);

    auto num_channels = static_cast<int>(channels.size());

    EXRHeader header;
    InitEXRHeader(&header);
    header.compression_type = ToTinyexrCompression(compression);
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

    // Set custom float attributes (metadata)
    std::vector<EXRAttribute> attrs(metadata.size());
    std::vector<float> attr_values(metadata.size());
    for (size_t i = 0; i < metadata.size(); ++i) {
        memset(&attrs[i], 0, sizeof(EXRAttribute));
#ifdef _MSC_VER
        strncpy_s(attrs[i].name, metadata[i].first.c_str(), 255);
        strncpy_s(attrs[i].type, "float", 255);
#else
        strncpy(attrs[i].name, metadata[i].first.c_str(), 255);
        attrs[i].name[255] = '\0';
        strncpy(attrs[i].type, "float", 255);
        attrs[i].type[255] = '\0';
#endif
        attr_values[i] = metadata[i].second;
        attrs[i].value = reinterpret_cast<unsigned char*>(&attr_values[i]);
        attrs[i].size = sizeof(float);
    }
    if (!attrs.empty()) {
        header.num_custom_attributes = static_cast<int>(attrs.size());
        header.custom_attributes = attrs.data();
    }

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
    writer->compression_ = desc.compression;
    return writer;
}

Writer::~Writer() = default;

uint32_t Writer::TargetWidth() const { return target_width_; }
uint32_t Writer::TargetHeight() const { return target_height_; }

bool Writer::WriteFrame(const InputFrame& input, const TargetFrame& target,
                        std::string_view subdirectory,
                        ExrMetadata metadata) {
    uint32_t input_pixels = input_width_ * input_height_;
    uint32_t target_pixels = target_width_ * target_height_;

    std::string dir = output_dir_;
    if (!subdirectory.empty()) {
        dir = std::format("{}/{}", output_dir_, subdirectory);
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) return false;
    }

    // --- Input EXR ---
    {
        std::vector<ExrChannel> channels;

        AppendFloatChannelGroup(channels, "diffuse", input.noisy_diffuse,
                                input_pixels, kRGBA, 4, TINYEXR_PIXELTYPE_HALF);
        AppendFloatChannelGroup(channels, "specular", input.noisy_specular,
                                input_pixels, kRGBA, 4, TINYEXR_PIXELTYPE_HALF);
        AppendFloatChannelGroup(channels, "albedo_d", input.diffuse_albedo,
                                input_pixels, kRGB, 3, TINYEXR_PIXELTYPE_HALF);
        AppendFloatChannelGroup(channels, "albedo_s", input.specular_albedo,
                                input_pixels, kRGB, 3, TINYEXR_PIXELTYPE_HALF);
        AppendFloatChannelGroup(channels, "normal", input.world_normals,
                                input_pixels, kXYZW, 4, TINYEXR_PIXELTYPE_HALF);
        AppendFloatChannelGroup(channels, "depth", input.linear_depth,
                                input_pixels, kZ, 1, TINYEXR_PIXELTYPE_FLOAT);
        AppendFloatChannelGroup(channels, "motion", input.motion_vectors,
                                input_pixels, kXY, 2, TINYEXR_PIXELTYPE_HALF);

        if (!channels.empty()) {
            auto path = std::format("{}/input.exr", dir);
            if (!WriteExr(path, input_width_, input_height_, channels, compression_, metadata))
                return false;
        }
    }

    // --- Target EXR ---
    {
        std::vector<ExrChannel> channels;

        AppendFloatChannelGroup(channels, "diffuse", target.ref_diffuse,
                                target_pixels, kRGBA, 4, TINYEXR_PIXELTYPE_FLOAT);
        AppendFloatChannelGroup(channels, "specular", target.ref_specular,
                                target_pixels, kRGBA, 4, TINYEXR_PIXELTYPE_FLOAT);

        if (!channels.empty()) {
            auto path = std::format("{}/target.exr", dir);
            if (!WriteExr(path, target_width_, target_height_, channels, compression_, metadata))
                return false;
        }
    }

    return true;
}

bool Writer::WriteFrameRaw(const RawInputFrame& input, const TargetFrame& target,
                           std::string_view subdirectory,
                           ExrMetadata metadata) {
    uint32_t input_pixels = input_width_ * input_height_;
    uint32_t target_pixels = target_width_ * target_height_;

    std::string dir = output_dir_;
    if (!subdirectory.empty()) {
        dir = std::format("{}/{}", output_dir_, subdirectory);
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) return false;
    }

    // --- Input EXR (unified: raw half + float channels) ---
    {
        std::vector<ExrChannel> channels;

        AppendRawHalfChannelGroup(channels, "diffuse", input.noisy_diffuse,
                                  input_pixels, kRGBA, 4);
        AppendRawHalfChannelGroup(channels, "specular", input.noisy_specular,
                                  input_pixels, kRGBA, 4);
        AppendFloatChannelGroup(channels, "albedo_d", input.diffuse_albedo,
                                input_pixels, kRGB, 3, TINYEXR_PIXELTYPE_HALF);
        AppendFloatChannelGroup(channels, "albedo_s", input.specular_albedo,
                                input_pixels, kRGB, 3, TINYEXR_PIXELTYPE_HALF);
        AppendRawHalfChannelGroup(channels, "normal", input.world_normals,
                                  input_pixels, kXYZW, 4);
        AppendFloatChannelGroup(channels, "depth", input.linear_depth,
                                input_pixels, kZ, 1, TINYEXR_PIXELTYPE_FLOAT);
        AppendRawHalfChannelGroup(channels, "motion", input.motion_vectors,
                                  input_pixels, kXY, 2);

        if (!channels.empty()) {
            auto path = std::format("{}/input.exr", dir);
            if (!WriteExr(path, input_width_, input_height_, channels, compression_, metadata))
                return false;
        }
    }

    // --- Target EXR (float only, same as WriteFrame) ---
    {
        std::vector<ExrChannel> channels;

        AppendFloatChannelGroup(channels, "diffuse", target.ref_diffuse,
                                target_pixels, kRGBA, 4, TINYEXR_PIXELTYPE_FLOAT);
        AppendFloatChannelGroup(channels, "specular", target.ref_specular,
                                target_pixels, kRGBA, 4, TINYEXR_PIXELTYPE_FLOAT);

        if (!channels.empty()) {
            auto path = std::format("{}/target.exr", dir);
            if (!WriteExr(path, target_width_, target_height_, channels, compression_, metadata))
                return false;
        }
    }

    return true;
}

} // namespace monti::capture
