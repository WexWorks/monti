#pragma once

#include <cstdio>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace monti::vulkan {

inline std::vector<uint8_t> LoadShaderFile(std::string_view path) {
    std::ifstream file(std::string(path), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::fprintf(stderr, "LoadShaderFile: failed to open shader: %.*s\n",
                     static_cast<int>(path.size()), path.data());
        return {};
    }
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

}  // namespace monti::vulkan
