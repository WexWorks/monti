#include <catch2/catch_test_macros.hpp>

// Verify FLIP library links by including its header
#include <FLIP.h>

// stb_image_write implementation (single TU)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

// Verify all library headers compile
#include <deni/vulkan/Denoiser.h>
#include <monti/scene/Scene.h>
#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>
#include <monti/capture/Writer.h>

TEST_CASE("Test harness works", "[harness]") {
    REQUIRE(true);
}

TEST_CASE("FLIP library links", "[harness]") {
    // Verify FLIP linkage by calling a basic function
    FLIP::image<float> img(2, 2);
    REQUIRE(img.getWidth() == 2);
    REQUIRE(img.getHeight() == 2);
}

TEST_CASE("TypedId basic operations", "[types]") {
    monti::MeshId id1{0};
    monti::MeshId id2{1};
    monti::MeshId invalid{};

    REQUIRE(id1 != id2);
    REQUIRE(id1 < id2);
    REQUIRE(static_cast<bool>(id1));
    REQUIRE_FALSE(static_cast<bool>(invalid));
}

TEST_CASE("Scene can be default constructed", "[scene]") {
    monti::Scene scene;
    REQUIRE(scene.Meshes().empty());
    REQUIRE(scene.Materials().empty());
    REQUIRE(scene.Nodes().empty());
    REQUIRE(scene.Textures().empty());
    REQUIRE(scene.GetEnvironmentLight() == nullptr);
}
