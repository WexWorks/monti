#include <catch2/catch_test_macros.hpp>

#include <volk.h>

#include "../app/core/vulkan_context.h"
#include "shared_context.h"
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

std::vector<deni::vulkan::LayerWeights> MakeTestLayersV3() {
    std::vector<deni::vulkan::LayerWeights> layers;

    // V3 temporal model: depthwise separable using "down0.conv1.pointwise" naming.
    // base_channels=8 (minimum divisible by kNumGroups=8).
    deni::vulkan::LayerWeights dw_weight;
    dw_weight.name = "down0.conv1.depthwise.weight";
    dw_weight.shape = {26, 1, 3, 3};  // 26-channel depthwise, no bias
    dw_weight.data.resize(26 * 9);
    for (uint32_t i = 0; i < dw_weight.data.size(); ++i)
        dw_weight.data[i] = static_cast<float>(i) * 0.001f;
    layers.push_back(std::move(dw_weight));

    deni::vulkan::LayerWeights pw_weight;
    pw_weight.name = "down0.conv1.pointwise.weight";
    pw_weight.shape = {8, 26, 1, 1};  // 26→8 pointwise
    pw_weight.data.resize(8 * 26);
    for (uint32_t i = 0; i < pw_weight.data.size(); ++i)
        pw_weight.data[i] = static_cast<float>(i) * 0.001f;
    layers.push_back(std::move(pw_weight));

    deni::vulkan::LayerWeights pw_bias;
    pw_bias.name = "down0.conv1.pointwise.bias";
    pw_bias.shape = {8};
    pw_bias.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    layers.push_back(std::move(pw_bias));

    return layers;
}

std::vector<deni::vulkan::LayerWeights> MakeTestLayers() {
    return MakeTestLayersV3();
}

}  // namespace

TEST_CASE("MlInference: feature map allocation at 256x256", "[deni][integration]") {
    auto& ctx = monti::test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    deni::vulkan::MlInference ml(ctx.Device(),
                                  ctx.Allocator(), vkGetDeviceProcAddr,
                                  DENI_SHADER_SPV_DIR, VK_NULL_HANDLE,
                                  kTestWidth, kTestHeight);

    CHECK(ml.Width() == kTestWidth);
    CHECK(ml.Height() == kTestHeight);
    // Features allocated in constructor via Resize
    // Weights not loaded yet, so not fully ready
    CHECK_FALSE(ml.IsReady());

    ctx.WaitIdle();
}

TEST_CASE("MlInference: weight upload via command buffer", "[deni][integration]") {
    auto& ctx = monti::test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    deni::vulkan::MlInference ml(ctx.Device(),
                                  ctx.Allocator(), vkGetDeviceProcAddr,
                                  DENI_SHADER_SPV_DIR, VK_NULL_HANDLE,
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
    CHECK(ml.WeightBufferCount() == 2);  // depthwise (no bias) + pointwise (weight+bias)

    // Free staging buffer after transfer completes
    ml.FreeStagingBuffer();

    ctx.WaitIdle();
}

TEST_CASE("MlInference: resize updates dimensions", "[deni][integration]") {
    auto& ctx = monti::test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    deni::vulkan::MlInference ml(ctx.Device(),
                                  ctx.Allocator(), vkGetDeviceProcAddr,
                                  DENI_SHADER_SPV_DIR, VK_NULL_HANDLE,
                                  kTestWidth, kTestHeight);

    // Load weights so channel counts are inferred (required for feature allocation)
    auto layers = MakeTestLayers();
    deni::vulkan::WeightData weights;
    for (const auto& layer : layers) {
        weights.total_parameters += layer.NumElements();
        weights.layers.push_back(layer);
    }
    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(ml.LoadWeights(weights, cmd));
    ctx.SubmitAndWait(cmd);
    ml.FreeStagingBuffer();

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
    auto& ctx = monti::test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

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
    auto& ctx = monti::test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

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
    // Auto-discovery may or may not find a model depending on build config
    if (denoiser->HasMlModel())
        CHECK(denoiser->Mode() == deni::vulkan::DenoiserMode::kMl);
    else
        CHECK(denoiser->Mode() == deni::vulkan::DenoiserMode::kPassthrough);

    denoiser.reset();
    ctx.WaitIdle();
}

TEST_CASE("Denoiser: Create with invalid model_path fails",
           "[deni][integration]") {
    auto& ctx = monti::test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

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
    CHECK(denoiser == nullptr);

    ctx.WaitIdle();
}

TEST_CASE("MlInference: v3 temporal weight upload detects model version",
          "[deni][integration][v3]") {
    auto& ctx = monti::test::SharedVulkanContext();
    REQUIRE(ctx.Device() != VK_NULL_HANDLE);

    deni::vulkan::MlInference ml(ctx.Device(),
                                  ctx.Allocator(), vkGetDeviceProcAddr,
                                  DENI_SHADER_SPV_DIR, VK_NULL_HANDLE,
                                  kTestWidth, kTestHeight);

    auto layers = MakeTestLayersV3();
    deni::vulkan::WeightData weights;
    for (const auto& layer : layers) {
        weights.total_parameters += layer.NumElements();
        weights.layers.push_back(layer);
    }

    VkCommandBuffer cmd = ctx.BeginOneShot();
    REQUIRE(ml.LoadWeights(weights, cmd));
    ctx.SubmitAndWait(cmd);

    CHECK(ml.IsReady());
    CHECK(ml.GetModelVersion() == deni::vulkan::ModelVersion::kV3_Temporal);

    ml.FreeStagingBuffer();
    ctx.WaitIdle();
}
