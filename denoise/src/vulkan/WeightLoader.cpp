#include "WeightLoader.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>

namespace deni::vulkan {

std::optional<WeightData> WeightLoader::Load(std::string_view path) {
    std::ifstream file(std::string(path), std::ios::binary);
    if (!file.is_open()) {
        std::fprintf(stderr, "deni::WeightLoader: failed to open file: %.*s\n",
                     static_cast<int>(path.size()), path.data());
        return std::nullopt;
    }

    auto read_u32 = [&](uint32_t& val) -> bool {
        return static_cast<bool>(file.read(reinterpret_cast<char*>(&val), sizeof(uint32_t)));
    };

    // Read header
    uint32_t magic = 0;
    if (!read_u32(magic) || magic != kMagic) {
        std::fprintf(stderr, "deni::WeightLoader: invalid magic number\n");
        return std::nullopt;
    }

    uint32_t version = 0;
    if (!read_u32(version) || version != kVersion) {
        std::fprintf(stderr, "deni::WeightLoader: unsupported version %u (expected %u)\n",
                     version, kVersion);
        return std::nullopt;
    }

    uint32_t num_layers = 0;
    if (!read_u32(num_layers)) {
        std::fprintf(stderr, "deni::WeightLoader: failed to read num_layers\n");
        return std::nullopt;
    }

    uint32_t total_weight_bytes = 0;
    if (!read_u32(total_weight_bytes)) {
        std::fprintf(stderr, "deni::WeightLoader: failed to read total_weight_bytes\n");
        return std::nullopt;
    }

    WeightData result;
    result.layers.reserve(num_layers);
    uint32_t actual_bytes = 0;

    for (uint32_t i = 0; i < num_layers; ++i) {
        LayerWeights layer;

        // Read name
        uint32_t name_length = 0;
        if (!read_u32(name_length) || name_length > 4096) {
            std::fprintf(stderr, "deni::WeightLoader: invalid name length at layer %u\n", i);
            return std::nullopt;
        }
        layer.name.resize(name_length);
        if (!file.read(layer.name.data(), name_length)) {
            std::fprintf(stderr, "deni::WeightLoader: failed to read name at layer %u\n", i);
            return std::nullopt;
        }

        // Read shape
        uint32_t num_dims = 0;
        if (!read_u32(num_dims) || num_dims > 8) {
            std::fprintf(stderr, "deni::WeightLoader: invalid num_dims at layer %u\n", i);
            return std::nullopt;
        }
        layer.shape.resize(num_dims);
        for (uint32_t d = 0; d < num_dims; ++d) {
            if (!read_u32(layer.shape[d])) {
                std::fprintf(stderr, "deni::WeightLoader: failed to read shape at layer %u\n", i);
                return std::nullopt;
            }
        }

        // Read weight data
        uint32_t num_elements = layer.NumElements();
        uint32_t data_bytes = num_elements * sizeof(float);
        layer.data.resize(num_elements);
        if (!file.read(reinterpret_cast<char*>(layer.data.data()), data_bytes)) {
            std::fprintf(stderr, "deni::WeightLoader: failed to read weights at layer %u\n", i);
            return std::nullopt;
        }

        actual_bytes += data_bytes;
        result.total_parameters += num_elements;
        result.layers.push_back(std::move(layer));
    }

    if (actual_bytes != total_weight_bytes) {
        std::fprintf(stderr,
                     "deni::WeightLoader: weight byte count mismatch: expected %u, got %u\n",
                     total_weight_bytes, actual_bytes);
        return std::nullopt;
    }

    return result;
}

}  // namespace deni::vulkan
