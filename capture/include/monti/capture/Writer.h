#pragma once
#include <string>
#include <memory>
#include <cstdint>

namespace monti::capture {

enum class CaptureFormat {
    kFloat16,
    kFloat32,
};

struct WriterDesc {
    std::string   output_dir = "./capture/";
    uint32_t      width;
    uint32_t      height;
    CaptureFormat format = CaptureFormat::kFloat32;
};

struct CaptureFrame {
    const float* noisy_diffuse      = nullptr;  // 4 floats/pixel (RGBA)
    const float* noisy_specular     = nullptr;  // 4 floats/pixel (RGBA)
    const float* ref_diffuse        = nullptr;  // 4 floats/pixel (RGBA)
    const float* ref_specular       = nullptr;  // 4 floats/pixel (RGBA)
    const float* diffuse_albedo     = nullptr;  // 3 floats/pixel (RGB)
    const float* specular_albedo    = nullptr;  // 3 floats/pixel (RGB)
    const float* world_normals      = nullptr;  // 4 floats/pixel (XYZW)
    const float* linear_depth       = nullptr;  // 1 float/pixel
    const float* motion_vectors     = nullptr;  // 2 floats/pixel (XY)
};

class Writer {
public:
    static std::unique_ptr<Writer> Create(const WriterDesc& desc);
    ~Writer();

    bool WriteFrame(const CaptureFrame& frame, uint32_t frame_index);

private:
    Writer() = default;
};

} // namespace monti::capture
