#pragma once

#include "../app/core/vulkan_context.h"
#include "../app/core/gbuffer_images.h"

#include <monti/vulkan/Renderer.h>

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

}  // namespace monti::test
