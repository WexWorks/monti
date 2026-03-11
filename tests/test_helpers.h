#pragma once

#include "../app/core/vulkan_context.h"
#include "../app/core/gbuffer_images.h"

#include <monti/vulkan/Renderer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <stb_image_write.h>

namespace monti::test {

// Populate a GBuffer struct from GBufferImages (views + images).
inline vulkan::GBuffer MakeGBuffer(const monti::app::GBufferImages& images) {
    vulkan::GBuffer gb{};
    gb.noisy_diffuse   = images.NoisyDiffuseView();
    gb.noisy_specular  = images.NoisySpecularView();
    gb.motion_vectors  = images.MotionVectorsView();
    gb.linear_depth    = images.LinearDepthView();
    gb.world_normals   = images.WorldNormalsView();
    gb.diffuse_albedo  = images.DiffuseAlbedoView();
    gb.specular_albedo = images.SpecularAlbedoView();
    gb.noisy_diffuse_image   = images.NoisyDiffuseImage();
    gb.noisy_specular_image  = images.NoisySpecularImage();
    gb.motion_vectors_image  = images.MotionVectorsImage();
    gb.linear_depth_image    = images.LinearDepthImage();
    gb.world_normals_image   = images.WorldNormalsImage();
    gb.diffuse_albedo_image  = images.DiffuseAlbedoImage();
    gb.specular_albedo_image = images.SpecularAlbedoImage();
    return gb;
}

// Convert IEEE 754 half-precision float to single-precision.
inline float HalfToFloat(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;

    if (exp == 0x1F) {
        if (frac != 0) return std::numeric_limits<float>::quiet_NaN();
        return sign ? -std::numeric_limits<float>::infinity()
                    : std::numeric_limits<float>::infinity();
    }
    if (exp == 0) {
        if (frac == 0) return sign ? -0.0f : 0.0f;
        float f = static_cast<float>(frac) / 1024.0f;
        f *= (1.0f / 16384.0f);  // 2^-14
        return sign ? -f : f;
    }
    float f = 1.0f + static_cast<float>(frac) / 1024.0f;
    f *= std::pow(2.0f, static_cast<float>(exp) - 15);
    return sign ? -f : f;
}

// Write an RGBA16F readback buffer to a PNG file for visual inspection.
// raw: pointer to kWidth*kHeight*4 uint16_t values (RGBA half-float).
// Applies Reinhard tone-mapping + gamma correction for viewable output.
inline bool WritePNG(std::string_view path, const uint16_t* raw,
                     uint32_t width, uint32_t height) {
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());
    std::vector<uint8_t> pixels(width * height * 3);
    for (uint32_t i = 0; i < width * height; ++i) {
        float r = HalfToFloat(raw[i * 4 + 0]);
        float g = HalfToFloat(raw[i * 4 + 1]);
        float b = HalfToFloat(raw[i * 4 + 2]);
        // Reinhard tone-map
        r = r / (1.0f + r);
        g = g / (1.0f + g);
        b = b / (1.0f + b);
        // Gamma correction (sRGB approximate)
        r = std::pow(std::max(r, 0.0f), 1.0f / 2.2f);
        g = std::pow(std::max(g, 0.0f), 1.0f / 2.2f);
        b = std::pow(std::max(b, 0.0f), 1.0f / 2.2f);
        pixels[i * 3 + 0] = static_cast<uint8_t>(std::clamp(r * 255.0f + 0.5f, 0.0f, 255.0f));
        pixels[i * 3 + 1] = static_cast<uint8_t>(std::clamp(g * 255.0f + 0.5f, 0.0f, 255.0f));
        pixels[i * 3 + 2] = static_cast<uint8_t>(std::clamp(b * 255.0f + 0.5f, 0.0f, 255.0f));
    }
    std::string path_str(path);
    return stbi_write_png(path_str.c_str(), static_cast<int>(width),
                          static_cast<int>(height), 3, pixels.data(),
                          static_cast<int>(width * 3)) != 0;
}

}  // namespace monti::test
