#include <monti/capture/Writer.h>

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

}  // namespace

std::unique_ptr<Writer> Writer::Create(const WriterDesc& desc) {
    auto writer = std::unique_ptr<Writer>(new Writer());
    float factor = ScaleFactor(desc.scale_mode);
    writer->target_width_  = ComputeTargetDim(desc.input_width, factor);
    writer->target_height_ = ComputeTargetDim(desc.input_height, factor);
    return writer;
}

Writer::~Writer() = default;

uint32_t Writer::TargetWidth() const { return target_width_; }
uint32_t Writer::TargetHeight() const { return target_height_; }

bool Writer::WriteFrame(const InputFrame& /*input*/, const TargetFrame& /*target*/,
                        uint32_t /*frame_index*/) {
    return false;  // Not implemented
}

} // namespace monti::capture
