#include <monti/capture/Writer.h>

namespace monti::capture {

std::unique_ptr<Writer> Writer::Create(const WriterDesc& /*desc*/) {
    return std::unique_ptr<Writer>(new Writer());
}

Writer::~Writer() = default;

bool Writer::WriteFrame(const CaptureFrame& /*frame*/, uint32_t /*frame_index*/) {
    return false;  // Not implemented
}

} // namespace monti::capture
