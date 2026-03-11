#pragma once

#include "Buffer.h"

#include <cstdint>

namespace monti::vulkan {

// Blue noise table generator for temporal and spatial decorrelation.
// Produces a 16384-entry (128×128 spatial tile) table of hash-scrambled Sobol
// sequences packed as uvec4 (one per bounce, 4 random bytes per u32).
// Uses MurmurHash3-based Owen scrambling: XOR each tile entry with a
// tile-specific hash to decorrelate neighboring pixels.
// Total size: 16384 × 4 × 4 = 256 KB.
class BlueNoise {
public:
    static constexpr uint32_t kTableSize = 16384;
    static constexpr uint32_t kComponentsPerEntry = 4;

    BlueNoise() = default;
    ~BlueNoise() = default;

    // Generate the blue noise table on the CPU and upload to a device-local
    // storage buffer. Records copy commands into cmd. The returned staging
    // buffer must be kept alive until cmd completes.
    bool Generate(VmaAllocator allocator, VkCommandBuffer cmd,
                  Buffer& staging_out);

    const Buffer& TableBuffer() const { return buffer_; }
    VkDeviceSize BufferSize() const { return buffer_.Size(); }

private:
    Buffer buffer_;
};

}  // namespace monti::vulkan
