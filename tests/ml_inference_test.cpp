#include <catch2/catch_test_macros.hpp>

#include <volk.h>

#include "../app/core/vulkan_context.h"
#include "../denoise/src/vulkan/MlInference.h"
#include "../denoise/src/vulkan/WeightLoader.h"

#include <deni/vulkan/Denoiser.h>

#include <filesystem>
#include <fstream>
#include <vector>

#ifndef DENI_SHADER_SPV_DIR
#define DENI_SHADER_SPV_DIR "build/deni_shaders"
#endif

namespace {

namespace fs = std::filesystem;

constexpr uint32_t kTestWidth = 256;
constexpr uint32_t kTestHeight = 256;
const std::string kTestDir = "test_output/ml_inference";

void WriteDeniModel(const std::string& path,
                    const std::vector<deni::vulkan::LayerWeights>& layers) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream file(path, std::ios::binary);
    REQUIRE(file.is_open());

    auto write_u32 = [&](uint32_t val) {
        file.write(reinterpret_cast<const char*>(&val), sizeof(uint32_t));
    };

    uint32_t total_bytes = 0;
    for (const auto& layer : layers)
        total_bytes += static_cast<uint32_t>(layer.data.size() * sizeof(float));

    file.write("DENI", 4);
    write_u32(1);
    write_u32(static_cast<uint32_t>(layers.size()));
    write_u32(total_bytes);

    for (const auto& layer : layers) {
        write_u32(static_cast<uint32_t>(layer.name.size()));
        file.write(layer.name.data(), layer.name.size());
        write_u32(static_cast<uint32_t>(layer.shape.size()));
        for (auto d : layer.shape) write_u32(d);
        file.write(reinterpret_cast<const char*>(layer.data.data()),
                   layer.data.size() * sizeof(float));
    }
}

std::vector<deni::vulkan::LayerWeights> MakeTestLayers() {
    std::vector<deni::vulkan::LayerWeights> layers;

    deni::vulkan::LayerWeights conv_weight;
    conv_weight.name = "encoder.conv1.weight";
    conv_weight.shape = {4, 3, 3, 3};
    conv_weight.data.resize(108);
    for (uint32_t i = 0; i < 108; ++i)
        conv_weight.data[i] = static_cast<float>(i) * 0.01f;
    layers.push_back(std::move(conv_weight));

    deni::vulkan::LayerWeights conv_bias;
    conv_bias.name = "encoder.conv1.bias";
    conv_bias.shape = {4};
    conv_bias.data = {0.1f, 0.2f, 0.3f, 0.4f};
    layers.push_back(std::move(conv_bias));

    return layers;
}

}  // namespace

TEST_CASE("MlInference: feature map allocation at 256x256", "[deni][integration]") {
    monti::app::VulkanContext ctx;
    REQUIRE(ctx.CreateInstance());
    REQUIRE(ctx.CreateDevice(std::nullopt));

    deni::vulkan::MlInference ml(ctx.Device(), ctx.Allocator(),
                                  vkGetDeviceProcAddr,
                                  kTestWidth, kTestHeight);

    CHECK(ml.Width() == kTestWidth);
    CHECK(ml.Height() == kTestHeight);
    // Features allocated in constructor via Resize
    // Weights not loaded yet, so not fully ready
    CHECK_FALSE(ml.IsReady());

    ctx.WaitIdle();
}

TEST_CASE("MlInference: weight upload via command buffer", "[deni][integration]") {
    monti::app::VulkanContext ctx;
    REQUIRE(ctx.CreateInstance());
    REQUIRE(ctx.CreateDevice(std::nullopt));

    deni::vulkan::MlInference ml(ctx.Device(), ctx.Allocator(),
                                  vkGetDeviceProcAddr,
                                  kTestWidth, kTestHeight);

    auto layers = MakeTestLayers();
    deni::vulkan::WeightData weights;
    for (const auto& layer : layers) {
        weights.total_parameters += layer.NumElements();
        weights.layers.push_back(layer);
    }

    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(ml.LoadWeights(weights, cmd));
    ctx.SubmitAndWait(cmd);

    CHECK(ml.IsReady());
    CHECK(ml.WeightBufferCount() == 2);

    // Free staging buffer after transfer completes
    ml.FreeStagingBuffer();

    ctx.WaitIdle();
}

TEST_CASE("MlInference: resize updates dimensions", "[deni][integration]") {
    monti::app::VulkanContext ctx;
    REQUIRE(ctx.CreateInstance());
    REQUIRE(ctx.CreateDevice(std::nullopt));

    deni::vulkan::MlInference ml(ctx.Device(), ctx.Allocator(),
                                  vkGetDeviceProcAddr,
                                  kTestWidth, kTestHeight);

    REQUIRE(ml.Resize(512, 512));
    CHECK(ml.Width() == 512);
    CHECK(ml.Height() == 512);

    // Resize to same dims is no-op
    REQUIRE(ml.Resize(512, 512));

    // Zero dimensions rejected
    CHECK_FALSE(ml.Resize(0, 0));

    ctx.WaitIdle();
}

TEST_CASE("Denoiser: Create with model_path loads ML model", "[deni][integration]") {
    monti::app::VulkanContext ctx;
    REQUIRE(ctx.CreateInstance());
    REQUIRE(ctx.CreateDevice(std::nullopt));

    auto layers = MakeTestLayers();
    std::string model_path = kTestDir + "/test_model.denimodel";
    WriteDeniModel(model_path, layers);

    deni::vulkan::DenoiserDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.width = kTestWidth;
    desc.height = kTestHeight;
    desc.allocator = ctx.Allocator();
    desc.shader_dir = DENI_SHADER_SPV_DIR;
    desc.get_device_proc_addr = vkGetDeviceProcAddr;
    desc.model_path = model_path;

    auto denoiser = deni::vulkan::Denoiser::Create(desc);
    REQUIRE(denoiser != nullptr);
    CHECK(denoiser->HasMlModel());

    denoiser.reset();
    ctx.WaitIdle();

    fs::remove(model_path);
}

TEST_CASE("Denoiser: Create without model_path falls back to passthrough",
           "[deni][integration]") {
    monti::app::VulkanContext ctx;
    REQUIRE(ctx.CreateInstance());
    REQUIRE(ctx.CreateDevice(std::nullopt));

    deni::vulkan::DenoiserDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.width = kTestWidth;
    desc.height = kTestHeight;
    desc.allocator = ctx.Allocator();
    desc.shader_dir = DENI_SHADER_SPV_DIR;
    desc.get_device_proc_addr = vkGetDeviceProcAddr;
    // model_path left empty

    auto denoiser = deni::vulkan::Denoiser::Create(desc);
    REQUIRE(denoiser != nullptr);
    CHECK_FALSE(denoiser->HasMlModel());

    denoiser.reset();
    ctx.WaitIdle();
}

TEST_CASE("Denoiser: Create with invalid model_path falls back to passthrough",
           "[deni][integration]") {
    monti::app::VulkanContext ctx;
    REQUIRE(ctx.CreateInstance());
    REQUIRE(ctx.CreateDevice(std::nullopt));

    deni::vulkan::DenoiserDesc desc{};
    desc.device = ctx.Device();
    desc.physical_device = ctx.PhysicalDevice();
    desc.width = kTestWidth;
    desc.height = kTestHeight;
    desc.allocator = ctx.Allocator();
    desc.shader_dir = DENI_SHADER_SPV_DIR;
    desc.get_device_proc_addr = vkGetDeviceProcAddr;
    desc.model_path = "nonexistent_model.denimodel";

    auto denoiser = deni::vulkan::Denoiser::Create(desc);
    REQUIRE(denoiser != nullptr);
    CHECK_FALSE(denoiser->HasMlModel());

    denoiser.reset();
    ctx.WaitIdle();
}
