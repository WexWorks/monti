#include <volk.h>

#include "BlueNoise.h"

#include "Upload.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace monti::vulkan {

namespace {

// MurmurHash3 finalizer — good bit avalanche for Owen scrambling
uint32_t MurmurHash3Mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return x;
}

// Simplified Sobol sequence sample — dimension-aware bit reversal with hash mixing
uint32_t SobolSample(uint32_t index, uint32_t dimension) {
    uint32_t result = 0;
    uint32_t i = index;
    uint32_t d = dimension + 1;

    while (i > 0) {
        if (i & 1) result ^= d;
        i >>= 1;
        d <<= 1;
    }

    result = MurmurHash3Mix(result ^ (dimension * 0x45D9F3Bu));
    return result & 0xFF;
}

// Pack 4 random bytes (one per Sobol dimension) into a single uint32
uint32_t PackRandomValues(uint32_t entry_index, uint32_t bounce_index) {
    uint32_t dimension = bounce_index * 4;
    uint32_t r0 = SobolSample(entry_index, dimension + 0);
    uint32_t r1 = SobolSample(entry_index, dimension + 1);
    uint32_t r2 = SobolSample(entry_index, dimension + 2);
    uint32_t r3 = SobolSample(entry_index, dimension + 3);
    return (r0 << 0) | (r1 << 8) | (r2 << 16) | (r3 << 24);
}

// Generate tile scramble from spatial position for Owen scrambling
uint32_t GenerateTileScramble(uint32_t tile_x, uint32_t tile_y) {
    uint32_t seed = (tile_x * 73 + tile_y * 157) ^ 0x9E3779B9u;
    return MurmurHash3Mix(seed);
}

}  // anonymous namespace

bool BlueNoise::Generate(VmaAllocator allocator, VkCommandBuffer cmd,
                         Buffer& staging_out) {
    std::vector<uint32_t> table(kTableSize * kComponentsPerEntry);

    // Generate packed random values per entry per bounce
    for (uint32_t entry = 0; entry < kTableSize; ++entry) {
        uint32_t base = entry * kComponentsPerEntry;
        for (uint32_t bounce = 0; bounce < kComponentsPerEntry; ++bounce)
            table[base + bounce] = PackRandomValues(entry, bounce);
    }

    // Apply Owen scrambling for blue-noise spatial distribution
    constexpr uint32_t kTileCount = 128;
    for (uint32_t tile_y = 0; tile_y < kTileCount; ++tile_y) {
        for (uint32_t tile_x = 0; tile_x < kTileCount; ++tile_x) {
            uint32_t tile_index = tile_y * kTileCount + tile_x;
            uint32_t scramble = GenerateTileScramble(tile_x, tile_y);
            uint32_t base = tile_index * kComponentsPerEntry;
            for (uint32_t i = 0; i < kComponentsPerEntry; ++i)
                table[base + i] ^= scramble;
        }
    }

    VkDeviceSize size = table.size() * sizeof(uint32_t);

    if (!buffer_.Create(allocator, size,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VMA_MEMORY_USAGE_GPU_ONLY))
        return false;

    staging_out = upload::ToBuffer(allocator, cmd, buffer_, table.data(), size);
    if (staging_out.Handle() == VK_NULL_HANDLE) return false;

    std::fprintf(stderr, "Blue noise table generated: %u entries, %zu bytes\n",
                 kTableSize, static_cast<size_t>(size));
    return true;
}

}  // namespace monti::vulkan
