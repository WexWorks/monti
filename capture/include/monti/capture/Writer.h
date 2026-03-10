#pragma once
#include <string>
#include <memory>
#include <cstdint>

namespace monti::capture {

// Scale factor for target resolution relative to input resolution.
// Mirrors deni::vulkan::ScaleMode — the capture writer is CPU-only
// and does not depend on Vulkan or Deni headers.
enum class ScaleMode {
    kNative,       // 1.0× — target resolution = input resolution
    kQuality,      // 1.5× — target = input × 1.5 (rounded to even)
    kPerformance,  // 2.0× — target = input × 2
};

struct WriterDesc {
    std::string   output_dir = "./capture/";
    uint32_t      input_width;          // Input (render) resolution
    uint32_t      input_height;
    ScaleMode     scale_mode = ScaleMode::kPerformance;  // Target = input × scale
    // Target resolution is computed internally:
    //   target_dim = floor(input_dim × scale_factor / 2) × 2
};

// Input channels — all at input resolution (WriterDesc::input_width × input_height).
// All pointers are to CPU-side float arrays. Null pointers are omitted from the
// output EXR. The writer uses per-channel bit depths: FP16 for radiance and
// auxiliary channels, FP32 for linear_depth.
struct InputFrame {
    const float* noisy_diffuse      = nullptr;  // 4 floats/pixel (RGBA) → FP16
    const float* noisy_specular     = nullptr;  // 4 floats/pixel (RGBA) → FP16
    const float* diffuse_albedo     = nullptr;  // 3 floats/pixel (RGB)  → FP16
    const float* specular_albedo    = nullptr;  // 3 floats/pixel (RGB)  → FP16
    const float* world_normals      = nullptr;  // 4 floats/pixel (XYZW) → FP16
    const float* linear_depth       = nullptr;  // 1 float/pixel         → FP32
    const float* motion_vectors     = nullptr;  // 2 floats/pixel (XY)   → FP16
};

// Target channels — at target resolution (derived from input × scale_mode).
// Written in FP32 for maximum ground-truth precision.
struct TargetFrame {
    const float* ref_diffuse        = nullptr;  // 4 floats/pixel (RGBA) → FP32
    const float* ref_specular       = nullptr;  // 4 floats/pixel (RGBA) → FP32
};

class Writer {
public:
    static std::unique_ptr<Writer> Create(const WriterDesc& desc);
    ~Writer();

    // Target resolution derived from WriterDesc.
    uint32_t TargetWidth() const;
    uint32_t TargetHeight() const;

    // Writes two EXR files per frame:
    //   {output_dir}/frame_{NNNN}_input.exr  — input channels at input resolution
    //   {output_dir}/frame_{NNNN}_target.exr — target channels at target resolution
    bool WriteFrame(const InputFrame& input, const TargetFrame& target,
                    uint32_t frame_index);

private:
    Writer() = default;
    uint32_t target_width_ = 0;
    uint32_t target_height_ = 0;
};

} // namespace monti::capture
