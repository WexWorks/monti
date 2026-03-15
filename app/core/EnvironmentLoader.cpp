#include "EnvironmentLoader.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <tinyexr.h>

namespace monti::app {

std::optional<TextureDesc> LoadExrEnvironment(std::string_view path) {
    std::string path_str(path);

    float* raw_data = nullptr;
    int width = 0;
    int height = 0;
    const char* err = nullptr;

    int ret = LoadEXR(&raw_data, &width, &height, path_str.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        std::fprintf(stderr, "Failed to load EXR '%s': %s\n", path_str.c_str(),
                     err ? err : "unknown error");
        FreeEXRErrorMessage(err);
        return std::nullopt;
    }

    auto rgba_data = std::unique_ptr<float[], decltype(&free)>(raw_data, &free);

    TextureDesc tex;
    tex.width = static_cast<uint32_t>(width);
    tex.height = static_cast<uint32_t>(height);
    tex.format = PixelFormat::kRGBA32F;
    tex.data.resize(static_cast<size_t>(width) * height * 4 * sizeof(float));
    std::memcpy(tex.data.data(), rgba_data.get(), tex.data.size());

    std::printf("Loaded environment map: %dx%d from %s\n", width, height, path_str.c_str());
    return tex;
}

TextureDesc MakeDefaultEnvironment(float r, float g, float b) {
    constexpr uint32_t kW = 4;
    constexpr uint32_t kH = 2;
    std::vector<float> pixels(kW * kH * 4);
    for (uint32_t i = 0; i < kW * kH; ++i) {
        pixels[i * 4 + 0] = r;
        pixels[i * 4 + 1] = g;
        pixels[i * 4 + 2] = b;
        pixels[i * 4 + 3] = 1.0f;
    }
    TextureDesc tex;
    tex.width = kW;
    tex.height = kH;
    tex.format = PixelFormat::kRGBA32F;
    tex.data.resize(pixels.size() * sizeof(float));
    std::memcpy(tex.data.data(), pixels.data(), tex.data.size());
    return tex;
}

}  // namespace monti::app
