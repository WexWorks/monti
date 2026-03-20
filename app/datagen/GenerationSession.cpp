#include "GenerationSession.h"

#include <monti/capture/GpuReadback.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>

#include <glm/glm.hpp>
#include <volk.h>

namespace monti::app::datagen {

GenerationSession::~GenerationSession() {
    if (readback_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(ctx_.Device(), readback_pool_, nullptr);
        readback_pool_ = VK_NULL_HANDLE;
    }
}

namespace {

// Struct passed to AccumulateFrames callback.
struct RenderCallbackData {
    vulkan::Renderer* renderer;
    vulkan::GBuffer gbuffer;
};

void RenderFrameCallback(VkCommandBuffer cmd, uint32_t frame_index, void* user_data) {
    auto* data = static_cast<RenderCallbackData*>(user_data);
    data->renderer->RenderFrame(cmd, data->gbuffer, frame_index);
}

}  // namespace

GenerationSession::GenerationSession(VulkanContext& ctx,
                                     vulkan::Renderer& renderer,
                                     GBufferImages& gbuffer,
                                     capture::Writer& writer,
                                     Scene& scene,
                                     const GenerationConfig& config)
    : ctx_(ctx)
    , renderer_(renderer)
    , gbuffer_(gbuffer)
    , writer_(writer)
    , scene_(scene)
    , config_(config) {

    // Create a dedicated command pool for readback operations
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = ctx_.QueueFamilyIndex();
    vkCreateCommandPool(ctx_.Device(), &pool_info, nullptr, &readback_pool_);

    readback_ctx_.device = ctx_.Device();
    readback_ctx_.queue = ctx_.GraphicsQueue();
    readback_ctx_.queue_family_index = ctx_.QueueFamilyIndex();
    readback_ctx_.allocator = ctx_.Allocator();
    readback_ctx_.command_pool = readback_pool_;

    // Fill Vulkan function pointers (volk provides these globally after volkLoadDevice)
    readback_ctx_.pfn_vkAllocateCommandBuffers = vkAllocateCommandBuffers;
    readback_ctx_.pfn_vkBeginCommandBuffer     = vkBeginCommandBuffer;
    readback_ctx_.pfn_vkEndCommandBuffer       = vkEndCommandBuffer;
    readback_ctx_.pfn_vkCmdPipelineBarrier2    = vkCmdPipelineBarrier2;
    readback_ctx_.pfn_vkCmdCopyImageToBuffer   = vkCmdCopyImageToBuffer;
    readback_ctx_.pfn_vkQueueSubmit            = vkQueueSubmit;
    readback_ctx_.pfn_vkCreateFence            = vkCreateFence;
    readback_ctx_.pfn_vkWaitForFences          = vkWaitForFences;
    readback_ctx_.pfn_vkDestroyFence           = vkDestroyFence;
    readback_ctx_.pfn_vkFreeCommandBuffers     = vkFreeCommandBuffers;

    // Pre-allocate CPU-side buffers
    uint32_t pixels = config_.width * config_.height;
    noisy_diffuse_raw_.resize(static_cast<size_t>(pixels) * 4);
    noisy_specular_raw_.resize(static_cast<size_t>(pixels) * 4);
    world_normals_raw_.resize(static_cast<size_t>(pixels) * 4);
    motion_vectors_raw_.resize(static_cast<size_t>(pixels) * 2);
    linear_depth_raw_.resize(static_cast<size_t>(pixels) * 2);
    diffuse_albedo_raw_.resize(pixels);
    specular_albedo_raw_.resize(pixels);
}

bool GenerationSession::Run() {
    auto total_start = std::chrono::steady_clock::now();

    auto num_viewpoints = static_cast<uint32_t>(config_.viewpoints.size());
    if (num_viewpoints == 0) {
        std::fprintf(stderr, "No viewpoints configured\n");
        return false;
    }

    for (uint32_t i = 0; i < num_viewpoints; ++i) {
        auto frame_start = std::chrono::steady_clock::now();

        // Set camera for this viewpoint
        const auto& vp = config_.viewpoints[i];
        monti::CameraParams camera{};
        camera.position = vp.position;
        camera.target = vp.target;
        camera.up = {0.0f, 1.0f, 0.0f};
        camera.vertical_fov_radians = glm::radians(vp.fov_degrees);
        camera.near_plane = app::kDefaultNearPlane;
        camera.far_plane = app::kDefaultFarPlane;
        camera.exposure_ev100 = vp.exposure.value_or(config_.exposure);
        scene_.SetActiveCamera(camera);

        std::printf("[viewpoint %u/%u] pos=(%.2f, %.2f, %.2f) target=(%.2f, %.2f, %.2f) fov=%.1f\n",
                    i + 1, num_viewpoints,
                    vp.position.x, vp.position.y, vp.position.z,
                    vp.target.x, vp.target.y, vp.target.z,
                    vp.fov_degrees);

        // Each viewpoint starts with frame_index 0 for fresh jitter
        // 1. Render noisy frame
        if (!RenderAndReadbackNoisy(0)) return false;

        // 2. Full pipeline barrier (implicit — we waited for queue idle in readback)

        // 3. Render reference via multi-frame accumulation
        // Use frame indices starting from 1 for different jitter than the noisy frame
        if (!RenderReference(1)) return false;

        // 4. Write EXR files into per-viewpoint subdirectory
        auto subdir = std::format("vp_{}", i);
        if (!WriteFrame(subdir)) return false;

        auto frame_end = std::chrono::steady_clock::now();
        double frame_secs = std::chrono::duration<double>(frame_end - frame_start).count();
        std::printf("[viewpoint %u/%u] written to %s/ (%.2fs)\n",
                    i + 1, num_viewpoints, subdir.c_str(), frame_secs);
    }

    auto total_end = std::chrono::steady_clock::now();
    double total_secs = std::chrono::duration<double>(total_end - total_start).count();

    std::printf("\nGeneration complete:\n");
    std::printf("  Viewpoints: %u\n", num_viewpoints);
    std::printf("  Resolution: %ux%u\n", config_.width, config_.height);
    std::printf("  Output:     %s\n", config_.output_dir.c_str());
    std::printf("  Total time: %.2fs\n", total_secs);

    return true;
}

bool GenerationSession::RenderAndReadbackNoisy(uint32_t frame_index) {
    auto gbuffer = gbuffer_.ToGBuffer();

    // Render noisy frame
    renderer_.SetSamplesPerPixel(config_.spp);
    VkCommandBuffer cmd = ctx_.BeginOneShot();
    if (!renderer_.RenderFrame(cmd, gbuffer, frame_index)) {
        ctx_.SubmitAndWait(cmd);
        return false;
    }
    ctx_.SubmitAndWait(cmd);

    uint32_t w = config_.width;
    uint32_t h = config_.height;

    // Read back all 7 G-buffer channels
    // RGBA16F channels (8 bytes/pixel)
    auto diffuse_rb = capture::ReadbackImage(readback_ctx_,
        gbuffer_.NoisyDiffuseImage(), w, h, 8);
    auto specular_rb = capture::ReadbackImage(readback_ctx_,
        gbuffer_.NoisySpecularImage(), w, h, 8);
    auto normals_rb = capture::ReadbackImage(readback_ctx_,
        gbuffer_.WorldNormalsImage(), w, h, 8);

    // RG16F channels (4 bytes/pixel)
    auto motion_rb = capture::ReadbackImage(readback_ctx_,
        gbuffer_.MotionVectorsImage(), w, h, 4);
    auto depth_rb = capture::ReadbackImage(readback_ctx_,
        gbuffer_.LinearDepthImage(), w, h, 4);

    // B10G11R11 channels (4 bytes/pixel)
    auto diff_albedo_rb = capture::ReadbackImage(readback_ctx_,
        gbuffer_.DiffuseAlbedoImage(), w, h, 4);
    auto spec_albedo_rb = capture::ReadbackImage(readback_ctx_,
        gbuffer_.SpecularAlbedoImage(), w, h, 4);

    // Copy to CPU storage
    uint32_t pixels = w * h;

    auto* d = static_cast<uint16_t*>(diffuse_rb.Map());
    std::memcpy(noisy_diffuse_raw_.data(), d, static_cast<size_t>(pixels) * 4 * sizeof(uint16_t));
    diffuse_rb.Unmap();

    auto* s = static_cast<uint16_t*>(specular_rb.Map());
    std::memcpy(noisy_specular_raw_.data(), s, static_cast<size_t>(pixels) * 4 * sizeof(uint16_t));
    specular_rb.Unmap();

    auto* n = static_cast<uint16_t*>(normals_rb.Map());
    std::memcpy(world_normals_raw_.data(), n, static_cast<size_t>(pixels) * 4 * sizeof(uint16_t));
    normals_rb.Unmap();

    auto* mv = static_cast<uint16_t*>(motion_rb.Map());
    std::memcpy(motion_vectors_raw_.data(), mv, static_cast<size_t>(pixels) * 2 * sizeof(uint16_t));
    motion_rb.Unmap();

    auto* dp = static_cast<uint16_t*>(depth_rb.Map());
    std::memcpy(linear_depth_raw_.data(), dp, static_cast<size_t>(pixels) * 2 * sizeof(uint16_t));
    depth_rb.Unmap();

    auto* da = static_cast<uint32_t*>(diff_albedo_rb.Map());
    std::memcpy(diffuse_albedo_raw_.data(), da, static_cast<size_t>(pixels) * sizeof(uint32_t));
    diff_albedo_rb.Unmap();

    auto* sa = static_cast<uint32_t*>(spec_albedo_rb.Map());
    std::memcpy(specular_albedo_raw_.data(), sa, static_cast<size_t>(pixels) * sizeof(uint32_t));
    spec_albedo_rb.Unmap();

    return true;
}

bool GenerationSession::RenderReference(uint32_t base_frame_index) {
    auto gbuffer = gbuffer_.ToGBuffer();

    RenderCallbackData cb_data{};
    cb_data.renderer = &renderer_;
    cb_data.gbuffer = gbuffer;

    renderer_.SetSamplesPerPixel(config_.spp);

    ref_result_ = capture::AccumulateFrames(
        readback_ctx_,
        gbuffer_.NoisyDiffuseImage(),
        gbuffer_.NoisySpecularImage(),
        config_.width, config_.height,
        config_.ref_frames,
        base_frame_index,
        RenderFrameCallback,
        &cb_data);

    return true;
}

bool GenerationSession::WriteFrame(std::string_view subdirectory) {
    uint32_t pixels = config_.width * config_.height;

    // Apply exposure scaling to color channels (diffuse + specular)
    float exposure_mul = std::pow(2.0f, config_.exposure);
    if (exposure_mul != 1.0f) {
        // Scale raw FP16 noisy diffuse/specular (RGBA — skip alpha)
        for (uint32_t i = 0; i < pixels; ++i) {
            auto base = static_cast<size_t>(i) * 4;
            for (int c = 0; c < 3; ++c) {
                float v = capture::HalfToFloat(noisy_diffuse_raw_[base + c]);
                noisy_diffuse_raw_[base + c] = capture::FloatToHalf(v * exposure_mul);
            }
            for (int c = 0; c < 3; ++c) {
                float v = capture::HalfToFloat(noisy_specular_raw_[base + c]);
                noisy_specular_raw_[base + c] = capture::FloatToHalf(v * exposure_mul);
            }
        }
        // Scale float reference diffuse/specular (RGBA — skip alpha)
        for (uint32_t i = 0; i < static_cast<uint32_t>(ref_result_.diffuse_f32.size() / 4); ++i) {
            auto base = static_cast<size_t>(i) * 4;
            ref_result_.diffuse_f32[base + 0] *= exposure_mul;
            ref_result_.diffuse_f32[base + 1] *= exposure_mul;
            ref_result_.diffuse_f32[base + 2] *= exposure_mul;
        }
        for (uint32_t i = 0; i < static_cast<uint32_t>(ref_result_.specular_f32.size() / 4); ++i) {
            auto base = static_cast<size_t>(i) * 4;
            ref_result_.specular_f32[base + 0] *= exposure_mul;
            ref_result_.specular_f32[base + 1] *= exposure_mul;
            ref_result_.specular_f32[base + 2] *= exposure_mul;
        }
    }

    // Unpack B10G11R11 albedo to 3 floats/pixel
    std::vector<float> diffuse_albedo_f32(static_cast<size_t>(pixels) * 3);
    std::vector<float> specular_albedo_f32(static_cast<size_t>(pixels) * 3);
    capture::UnpackB10G11R11Image(diffuse_albedo_raw_.data(), diffuse_albedo_f32.data(), pixels);
    capture::UnpackB10G11R11Image(specular_albedo_raw_.data(), specular_albedo_f32.data(), pixels);

    // Extract depth (R channel from RG16F)
    std::vector<float> depth_f32(pixels);
    capture::ExtractDepthFromRG16F(linear_depth_raw_.data(), depth_f32.data(), pixels);

    // Build raw input frame
    capture::RawInputFrame input{};
    input.noisy_diffuse = noisy_diffuse_raw_.data();
    input.noisy_specular = noisy_specular_raw_.data();
    input.diffuse_albedo = diffuse_albedo_f32.data();
    input.specular_albedo = specular_albedo_f32.data();
    input.world_normals = world_normals_raw_.data();
    input.linear_depth = depth_f32.data();
    input.motion_vectors = motion_vectors_raw_.data();

    // Build target frame from accumulated reference
    capture::TargetFrame target{};
    target.ref_diffuse = ref_result_.diffuse_f32.data();
    target.ref_specular = ref_result_.specular_f32.data();

    return writer_.WriteFrameRaw(input, target, subdirectory);
}

}  // namespace monti::app::datagen
