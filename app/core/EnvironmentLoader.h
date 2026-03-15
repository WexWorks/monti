#pragma once

#include <monti/scene/Material.h>

#include <optional>
#include <string_view>

namespace monti::app {

// Load an EXR environment map from disk into a TextureDesc (RGBA32F).
// Returns std::nullopt on failure (file not found, decode error, etc.).
std::optional<TextureDesc> LoadExrEnvironment(std::string_view path);

// Create a small constant-color RGBA32F environment map (4x2).
// Used as a default when no --env is specified.
TextureDesc MakeDefaultEnvironment(float r, float g, float b);

}  // namespace monti::app
