#include <catch2/catch_test_macros.hpp>

#include "../denoise/src/vulkan/WeightLoader.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {

namespace fs = std::filesystem;

// Write a valid .denimodel file with the given layers
void WriteDeniModel(const std::string& path,
                    const std::vector<deni::vulkan::LayerWeights>& layers) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream file(path, std::ios::binary);
    REQUIRE(file.is_open());

    auto write_u32 = [&](uint32_t val) {
        file.write(reinterpret_cast<const char*>(&val), sizeof(uint32_t));
    };

    // Calculate total weight bytes
    uint32_t total_bytes = 0;
    for (const auto& layer : layers)
        total_bytes += static_cast<uint32_t>(layer.data.size() * sizeof(float));

    // Header
    file.write("DENI", 4);                                     // magic
    write_u32(1);                                               // version
    write_u32(static_cast<uint32_t>(layers.size()));            // num_layers
    write_u32(total_bytes);                                     // total_weight_bytes

    // Per-layer data
    for (const auto& layer : layers) {
        auto name_bytes = layer.name;
        write_u32(static_cast<uint32_t>(name_bytes.size()));    // name_length
        file.write(name_bytes.data(), name_bytes.size());       // name
        write_u32(static_cast<uint32_t>(layer.shape.size()));   // num_dims
        for (auto d : layer.shape)
            write_u32(d);                                       // shape dims
        file.write(reinterpret_cast<const char*>(layer.data.data()),
                   layer.data.size() * sizeof(float));          // weight data
    }
}

// Create test layers mimicking a small conv network
std::vector<deni::vulkan::LayerWeights> MakeTestLayers() {
    std::vector<deni::vulkan::LayerWeights> layers;

    // Conv weight: [out_ch=4][in_ch=3][3][3] = 108 floats
    deni::vulkan::LayerWeights conv_weight;
    conv_weight.name = "encoder.conv1.weight";
    conv_weight.shape = {4, 3, 3, 3};
    conv_weight.data.resize(108);
    for (uint32_t i = 0; i < 108; ++i)
        conv_weight.data[i] = static_cast<float>(i) * 0.01f;
    layers.push_back(std::move(conv_weight));

    // Conv bias: [out_ch=4] = 4 floats
    deni::vulkan::LayerWeights conv_bias;
    conv_bias.name = "encoder.conv1.bias";
    conv_bias.shape = {4};
    conv_bias.data = {0.1f, 0.2f, 0.3f, 0.4f};
    layers.push_back(std::move(conv_bias));

    // GroupNorm gamma: [4] = 4 floats
    deni::vulkan::LayerWeights gn_gamma;
    gn_gamma.name = "encoder.norm1.weight";
    gn_gamma.shape = {4};
    gn_gamma.data = {1.0f, 1.0f, 1.0f, 1.0f};
    layers.push_back(std::move(gn_gamma));

    // GroupNorm beta: [4] = 4 floats
    deni::vulkan::LayerWeights gn_beta;
    gn_beta.name = "encoder.norm1.bias";
    gn_beta.shape = {4};
    gn_beta.data = {0.0f, 0.0f, 0.0f, 0.0f};
    layers.push_back(std::move(gn_beta));

    return layers;
}

const std::string kTestDir = "test_output/ml_weights";

}  // namespace

TEST_CASE("WeightLoader: valid .denimodel round-trip", "[deni][unit]") {
    auto layers = MakeTestLayers();
    std::string path = kTestDir + "/valid_model.denimodel";
    WriteDeniModel(path, layers);

    auto result = deni::vulkan::WeightLoader::Load(path);
    REQUIRE(result.has_value());

    auto& weights = *result;
    REQUIRE(weights.layers.size() == layers.size());

    // Verify total parameter count
    uint32_t expected_params = 108 + 4 + 4 + 4;  // 120
    REQUIRE(weights.total_parameters == expected_params);

    // Verify each layer
    for (size_t i = 0; i < layers.size(); ++i) {
        REQUIRE(weights.layers[i].name == layers[i].name);
        REQUIRE(weights.layers[i].shape == layers[i].shape);
        REQUIRE(weights.layers[i].data.size() == layers[i].data.size());

        for (size_t j = 0; j < layers[i].data.size(); ++j)
            REQUIRE(weights.layers[i].data[j] == layers[i].data[j]);
    }

    // Cleanup
    fs::remove(path);
}

TEST_CASE("WeightLoader: invalid magic number rejected", "[deni][unit]") {
    std::string path = kTestDir + "/bad_magic.denimodel";
    fs::create_directories(kTestDir);

    // Write file with bad magic
    std::ofstream file(path, std::ios::binary);
    REQUIRE(file.is_open());
    file.write("XENI", 4);  // Wrong magic
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), 4);
    uint32_t num_layers = 0;
    file.write(reinterpret_cast<const char*>(&num_layers), 4);
    uint32_t total_bytes = 0;
    file.write(reinterpret_cast<const char*>(&total_bytes), 4);
    file.close();

    auto result = deni::vulkan::WeightLoader::Load(path);
    REQUIRE_FALSE(result.has_value());

    fs::remove(path);
}

TEST_CASE("WeightLoader: unsupported version rejected", "[deni][unit]") {
    std::string path = kTestDir + "/bad_version.denimodel";
    fs::create_directories(kTestDir);

    std::ofstream file(path, std::ios::binary);
    REQUIRE(file.is_open());
    file.write("DENI", 4);
    uint32_t version = 99;
    file.write(reinterpret_cast<const char*>(&version), 4);
    uint32_t num_layers = 0;
    file.write(reinterpret_cast<const char*>(&num_layers), 4);
    uint32_t total_bytes = 0;
    file.write(reinterpret_cast<const char*>(&total_bytes), 4);
    file.close();

    auto result = deni::vulkan::WeightLoader::Load(path);
    REQUIRE_FALSE(result.has_value());

    fs::remove(path);
}

TEST_CASE("WeightLoader: nonexistent file rejected", "[deni][unit]") {
    auto result = deni::vulkan::WeightLoader::Load("nonexistent_file.denimodel");
    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("WeightLoader: byte count mismatch rejected", "[deni][unit]") {
    std::string path = kTestDir + "/bad_bytes.denimodel";
    fs::create_directories(kTestDir);

    // Write valid header but wrong total_weight_bytes
    auto layers = MakeTestLayers();

    std::ofstream file(path, std::ios::binary);
    REQUIRE(file.is_open());

    auto write_u32 = [&](uint32_t val) {
        file.write(reinterpret_cast<const char*>(&val), sizeof(uint32_t));
    };

    file.write("DENI", 4);
    write_u32(1);
    write_u32(static_cast<uint32_t>(layers.size()));
    write_u32(99999);  // Wrong total_weight_bytes

    for (const auto& layer : layers) {
        write_u32(static_cast<uint32_t>(layer.name.size()));
        file.write(layer.name.data(), layer.name.size());
        write_u32(static_cast<uint32_t>(layer.shape.size()));
        for (auto d : layer.shape) write_u32(d);
        file.write(reinterpret_cast<const char*>(layer.data.data()),
                   layer.data.size() * sizeof(float));
    }
    file.close();

    auto result = deni::vulkan::WeightLoader::Load(path);
    REQUIRE_FALSE(result.has_value());

    fs::remove(path);
}

TEST_CASE("WeightLoader: truncated file rejected", "[deni][unit]") {
    std::string path = kTestDir + "/truncated.denimodel";
    fs::create_directories(kTestDir);

    // Write only the header, no layer data
    std::ofstream file(path, std::ios::binary);
    REQUIRE(file.is_open());
    file.write("DENI", 4);
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), 4);
    uint32_t num_layers = 1;  // Claims 1 layer but no data follows
    file.write(reinterpret_cast<const char*>(&num_layers), 4);
    uint32_t total_bytes = 100;
    file.write(reinterpret_cast<const char*>(&total_bytes), 4);
    file.close();

    auto result = deni::vulkan::WeightLoader::Load(path);
    REQUIRE_FALSE(result.has_value());

    fs::remove(path);
}

TEST_CASE("WeightLoader: empty model (zero layers)", "[deni][unit]") {
    std::string path = kTestDir + "/empty.denimodel";
    fs::create_directories(kTestDir);

    std::vector<deni::vulkan::LayerWeights> no_layers;
    WriteDeniModel(path, no_layers);

    auto result = deni::vulkan::WeightLoader::Load(path);
    REQUIRE(result.has_value());
    REQUIRE(result->layers.empty());
    REQUIRE(result->total_parameters == 0);

    fs::remove(path);
}
