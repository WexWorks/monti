#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace deni::vulkan {

struct LayerWeights {
    std::string name;
    std::vector<uint32_t> shape;
    std::vector<float> data;

    uint32_t NumElements() const {
        uint32_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

struct WeightData {
    std::vector<LayerWeights> layers;
    uint32_t total_parameters = 0;
};

class WeightLoader {
public:
    static std::optional<WeightData> Load(std::string_view path);

private:
    static constexpr uint32_t kMagic = 0x494E4544;  // "DENI" little-endian
    static constexpr uint32_t kVersion = 1;
};

}  // namespace deni::vulkan
