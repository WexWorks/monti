#include "GenerationSession.h"

#include <monti/capture/GpuAccumulator.h>
#include <monti/capture/GpuReadback.h>
#include <monti/capture/Luminance.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <numeric>
#include <utility>

#include <glm/glm.hpp>
#include <volk.h>

namespace monti::app::datagen {

GenerationSession::~GenerationSession() {
    if (write_future_.valid()) write_future_.wait();
    accumulator_.reset();
    if (readback_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(ctx_.Device(), readback_pool_, nullptr);
        readback_pool_ = VK_NULL_HANDLE;
    }
}

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
    diffuse_albedo_raw_.resize(static_cast<size_t>(pixels) * 4);
    specular_albedo_raw_.resize(static_cast<size_t>(pixels) * 4);

    // Create GPU accumulator for reference frame rendering
    if (!config_.capture_shader_dir.empty()) {
        capture::GpuAccumulatorDesc acc_desc{};
        acc_desc.device = ctx_.Device();
        acc_desc.allocator = ctx_.Allocator();
        acc_desc.width = config_.width;
        acc_desc.height = config_.height;
        acc_desc.shader_dir = config_.capture_shader_dir;
        acc_desc.noisy_diffuse = gbuffer_.NoisyDiffuseImage();
        acc_desc.noisy_specular = gbuffer_.NoisySpecularImage();

        acc_desc.procs.pfn_vkCreateDescriptorSetLayout  = vkCreateDescriptorSetLayout;
        acc_desc.procs.pfn_vkDestroyDescriptorSetLayout = vkDestroyDescriptorSetLayout;
        acc_desc.procs.pfn_vkCreateDescriptorPool       = vkCreateDescriptorPool;
        acc_desc.procs.pfn_vkDestroyDescriptorPool      = vkDestroyDescriptorPool;
        acc_desc.procs.pfn_vkAllocateDescriptorSets     = vkAllocateDescriptorSets;
        acc_desc.procs.pfn_vkUpdateDescriptorSets       = vkUpdateDescriptorSets;
        acc_desc.procs.pfn_vkCreateShaderModule         = vkCreateShaderModule;
        acc_desc.procs.pfn_vkDestroyShaderModule        = vkDestroyShaderModule;
        acc_desc.procs.pfn_vkCreatePipelineLayout       = vkCreatePipelineLayout;
        acc_desc.procs.pfn_vkDestroyPipelineLayout      = vkDestroyPipelineLayout;
        acc_desc.procs.pfn_vkCreateComputePipelines     = vkCreateComputePipelines;
        acc_desc.procs.pfn_vkDestroyPipeline            = vkDestroyPipeline;
        acc_desc.procs.pfn_vkCreateImageView            = vkCreateImageView;
        acc_desc.procs.pfn_vkDestroyImageView           = vkDestroyImageView;
        acc_desc.procs.pfn_vkCmdPipelineBarrier2        = vkCmdPipelineBarrier2;
        acc_desc.procs.pfn_vkCmdBindPipeline            = vkCmdBindPipeline;
        acc_desc.procs.pfn_vkCmdBindDescriptorSets      = vkCmdBindDescriptorSets;
        acc_desc.procs.pfn_vkCmdPushConstants           = vkCmdPushConstants;
        acc_desc.procs.pfn_vkCmdDispatch                = vkCmdDispatch;
        acc_desc.procs.pfn_vkCmdClearColorImage         = vkCmdClearColorImage;

        accumulator_ = capture::GpuAccumulator::Create(acc_desc);
        if (!accumulator_)
            std::fprintf(stderr, "Warning: GPU accumulator creation failed, "
                         "falling back to CPU accumulation\n");
    }
}

bool GenerationSession::Run() {
    auto total_start = std::chrono::steady_clock::now();

    auto num_viewpoints = static_cast<uint32_t>(config_.viewpoints.size());
    if (num_viewpoints == 0) {
        std::fprintf(stderr, "No viewpoints configured\n");
        return false;
    }

    viewpoint_timings_.clear();
    viewpoint_timings_.reserve(num_viewpoints);
    skipped_viewpoints_.clear();

    // Build sorted index for path-grouped rendering.
    // Viewpoints with the same path_id are rendered consecutively, sorted by frame.
    // Viewpoints with empty path_id sort to the end (legacy/CLI viewpoints).
    std::vector<uint32_t> render_order(num_viewpoints);
    std::iota(render_order.begin(), render_order.end(), 0u);
    std::sort(render_order.begin(), render_order.end(),
              [this](uint32_t a, uint32_t b) {
                  const auto& va = config_.viewpoints[a];
                  const auto& vb = config_.viewpoints[b];
                  if (va.path_id != vb.path_id) {
                      if (va.path_id.empty()) return false;  // empty sorts last
                      if (vb.path_id.empty()) return true;
                      return va.path_id < vb.path_id;
                  }
                  return va.frame < vb.frame;
              });

    std::string current_path_id;
    for (uint32_t order_idx = 0; order_idx < num_viewpoints; ++order_idx) {
        uint32_t i = render_order[order_idx];
        auto frame_start = std::chrono::steady_clock::now();

        // Set camera for this viewpoint
        const auto& vp = config_.viewpoints[i];
        monti::CameraParams camera{};
        camera.position = vp.position;
        camera.target = vp.target;
        camera.up = vp.camera_up.value_or(glm::vec3{0.0f, 1.0f, 0.0f});
        camera.vertical_fov_radians = glm::radians(vp.fov_degrees);
        camera.near_plane = app::kDefaultNearPlane;
        camera.far_plane = app::kDefaultFarPlane;
        camera.exposure_ev100 = 0.0f;
        scene_.SetActiveCamera(camera);

        // Reset temporal state on path boundary
        bool new_path = vp.path_id.empty() || vp.path_id != current_path_id;
        if (new_path) {
            renderer_.ResetTemporalState();
            current_path_id = vp.path_id;
        }

        // Update environment rotation/intensity for this viewpoint
        if (vp.environment_rotation.has_value() || vp.environment_intensity.has_value()) {
            if (auto* env_ptr = scene_.GetEnvironmentLight()) {
                auto env_copy = *env_ptr;
                if (vp.environment_rotation.has_value())
                    env_copy.rotation = vp.environment_rotation.value();
                if (vp.environment_intensity.has_value())
                    env_copy.intensity = vp.environment_intensity.value();
                scene_.SetEnvironmentLight(env_copy);
            }
        }

        // Update environment blur for this viewpoint
        if (vp.environment_blur.has_value())
            renderer_.SetEnvironmentBlur(vp.environment_blur.value());

        std::printf("[viewpoint %u/%u] pos=(%.2f, %.2f, %.2f) target=(%.2f, %.2f, %.2f) fov=%.1f\n",
                    i + 1, num_viewpoints,
                    vp.position.x, vp.position.y, vp.position.z,
                    vp.target.x, vp.target.y, vp.target.z,
                    vp.fov_degrees);

        // Each viewpoint starts with frame_index 0 for fresh jitter
        // 1. Render noisy frame
        auto t0 = std::chrono::steady_clock::now();
        if (!RenderAndReadbackNoisy(0)) {
            std::fprintf(stderr, "[viewpoint %u/%u] render/readback failed, skipping\n",
                         i + 1, num_viewpoints);
            skipped_viewpoints_.push_back({config_.viewpoints[i].id,
                                           "gpu_error", 0.0f});
            continue;
        }
        auto t1 = std::chrono::steady_clock::now();

        // 2. Full pipeline barrier (implicit — we waited for queue idle in readback)

        // 3. Render reference via multi-frame accumulation
        // Use frame indices starting from 1 for different jitter than the noisy frame
        if (!RenderReference(1)) {
            std::fprintf(stderr, "[viewpoint %u/%u] reference render failed, skipping\n",
                         i + 1, num_viewpoints);
            skipped_viewpoints_.push_back({config_.viewpoints[i].id,
                                           "gpu_error", 0.0f});
            continue;
        }
        auto t2 = std::chrono::steady_clock::now();

        // 4. Wait for previous viewpoint's async write, then dispatch this one
        if (write_future_.valid()) {
            if (write_future_.get() == WriteResult::kError) return false;
        }

        auto subdir = std::format("vp_{}", i);
        WriteJob job{
            .noisy_diffuse_raw  = std::move(noisy_diffuse_raw_),
            .noisy_specular_raw = std::move(noisy_specular_raw_),
            .world_normals_raw  = std::move(world_normals_raw_),
            .motion_vectors_raw = std::move(motion_vectors_raw_),
            .linear_depth_raw   = std::move(linear_depth_raw_),
            .diffuse_albedo_raw = std::move(diffuse_albedo_raw_),
            .specular_albedo_raw = std::move(specular_albedo_raw_),
            .ref_result         = std::move(ref_result_),
            .subdirectory       = subdir,
            .index              = i,
            .width              = config_.width,
            .height             = config_.height,
        };

        // Re-allocate buffers for next viewpoint (moved-from vectors are empty)
        uint32_t pixels = config_.width * config_.height;
        noisy_diffuse_raw_.resize(static_cast<size_t>(pixels) * 4);
        noisy_specular_raw_.resize(static_cast<size_t>(pixels) * 4);
        world_normals_raw_.resize(static_cast<size_t>(pixels) * 4);
        motion_vectors_raw_.resize(static_cast<size_t>(pixels) * 2);
        linear_depth_raw_.resize(static_cast<size_t>(pixels) * 2);
        diffuse_albedo_raw_.resize(static_cast<size_t>(pixels) * 4);
        specular_albedo_raw_.resize(static_cast<size_t>(pixels) * 4);

        write_future_ = std::async(std::launch::async,
            [this, j = std::move(job)]() mutable { return WriteFrameFromJob(j); });

        auto t3 = std::chrono::steady_clock::now();

        double render_noisy_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double render_ref_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double write_exr_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double total_ms = std::chrono::duration<double, std::milli>(t3 - frame_start).count();

        double avg_ref_frame_ms = (config_.ref_frames > 0)
            ? render_ref_ms / config_.ref_frames : 0.0;

        std::printf("  render noisy:      %.1fms\n", render_noisy_ms);
        std::printf("  render reference:  %.1fms (%u frames, avg %.1fms/frame)\n",
                    render_ref_ms, config_.ref_frames, avg_ref_frame_ms);
        std::printf("  write EXR:         %.1fms\n", write_exr_ms);
        std::printf("[viewpoint %u/%u] written to %s/ (%.2fs)\n",
                    i + 1, num_viewpoints, subdir.c_str(), total_ms / 1000.0);

        viewpoint_timings_.push_back({
            {"index", i},
            {"render_noisy_ms", render_noisy_ms},
            {"render_reference_ms", render_ref_ms},
            {"write_exr_ms", write_exr_ms},
            {"total_ms", total_ms},
        });
    }

    // Wait for the final viewpoint's async write to complete
    if (write_future_.valid()) {
        if (write_future_.get() == WriteResult::kError) return false;
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

    // Batch-read all 7 G-buffer channels in a single command buffer submission
    constexpr auto kRTStage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    std::array<capture::ReadbackRequest, 7> requests{{
        {gbuffer_.NoisyDiffuseImage(),   w, h, 8, kRTStage, kRTStage},  // RGBA16F
        {gbuffer_.NoisySpecularImage(),  w, h, 8, kRTStage, kRTStage},  // RGBA16F
        {gbuffer_.WorldNormalsImage(),   w, h, 8, kRTStage, kRTStage},  // RGBA16F
        {gbuffer_.MotionVectorsImage(),  w, h, 4, kRTStage, kRTStage},  // RG16F
        {gbuffer_.LinearDepthImage(),    w, h, 4, kRTStage, kRTStage},  // RG16F
        {gbuffer_.DiffuseAlbedoImage(),  w, h, 8, kRTStage, kRTStage},  // RGBA16F
        {gbuffer_.SpecularAlbedoImage(), w, h, 8, kRTStage, kRTStage},  // RGBA16F
    }};

    auto staging = capture::ReadbackMultipleImages(readback_ctx_, requests);
    if (staging.size() != 7) return false;

    // Copy to CPU storage
    uint32_t pixels = w * h;

    auto* d = static_cast<uint16_t*>(staging[0].Map());
    std::memcpy(noisy_diffuse_raw_.data(), d, static_cast<size_t>(pixels) * 4 * sizeof(uint16_t));
    staging[0].Unmap();

    auto* s = static_cast<uint16_t*>(staging[1].Map());
    std::memcpy(noisy_specular_raw_.data(), s, static_cast<size_t>(pixels) * 4 * sizeof(uint16_t));
    staging[1].Unmap();

    auto* n = static_cast<uint16_t*>(staging[2].Map());
    std::memcpy(world_normals_raw_.data(), n, static_cast<size_t>(pixels) * 4 * sizeof(uint16_t));
    staging[2].Unmap();

    auto* mv = static_cast<uint16_t*>(staging[3].Map());
    std::memcpy(motion_vectors_raw_.data(), mv, static_cast<size_t>(pixels) * 2 * sizeof(uint16_t));
    staging[3].Unmap();

    auto* dp = static_cast<uint16_t*>(staging[4].Map());
    std::memcpy(linear_depth_raw_.data(), dp, static_cast<size_t>(pixels) * 2 * sizeof(uint16_t));
    staging[4].Unmap();

    auto* da = static_cast<uint16_t*>(staging[5].Map());
    std::memcpy(diffuse_albedo_raw_.data(), da, static_cast<size_t>(pixels) * 4 * sizeof(uint16_t));
    staging[5].Unmap();

    auto* sa = static_cast<uint16_t*>(staging[6].Map());
    std::memcpy(specular_albedo_raw_.data(), sa, static_cast<size_t>(pixels) * 4 * sizeof(uint16_t));
    staging[6].Unmap();

    return true;
}

bool GenerationSession::RenderReference(uint32_t base_frame_index) {
    auto gbuffer = gbuffer_.ToGBuffer();
    uint32_t ref_spp = (config_.ref_spp > 0) ? config_.ref_spp : config_.spp;
    renderer_.SetSamplesPerPixel(ref_spp);

    if (accumulator_) {
        // GPU-side accumulation: render + accumulate in same command buffer per frame,
        // single readback at the end. Reduces sync points from 3N to N+1.
        float weight = 1.0f / static_cast<float>(config_.ref_frames);

        for (uint32_t frame = 0; frame < config_.ref_frames; ++frame) {
            uint32_t frame_index = base_frame_index + frame;

            VkCommandBuffer cmd = ctx_.BeginOneShot();

            // Clear accumulators in the first frame's command buffer
            if (frame == 0) accumulator_->Reset(cmd);

            if (!renderer_.RenderFrame(cmd, gbuffer, frame_index)) {
                ctx_.SubmitAndWait(cmd);
                return false;
            }

            // Barrier: RT output → compute read for both noisy images
            std::array<VkImageMemoryBarrier2, 2> rt_to_compute{};
            for (uint32_t i = 0; i < 2; ++i) {
                rt_to_compute[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                rt_to_compute[i].srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                rt_to_compute[i].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                rt_to_compute[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                rt_to_compute[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                rt_to_compute[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                rt_to_compute[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                rt_to_compute[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                rt_to_compute[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                rt_to_compute[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            }
            rt_to_compute[0].image = gbuffer_.NoisyDiffuseImage();
            rt_to_compute[1].image = gbuffer_.NoisySpecularImage();

            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 2;
            dep.pImageMemoryBarriers = rt_to_compute.data();
            vkCmdPipelineBarrier2(cmd, &dep);

            accumulator_->Accumulate(cmd, weight);

            ctx_.SubmitAndWait(cmd);
        }

        ref_result_ = accumulator_->Finalize(readback_ctx_);
        if (ref_result_.diffuse_f32.empty()) return false;
    } else {
        // CPU fallback: original AccumulateFrames path
        struct RenderCallbackData {
            vulkan::Renderer* renderer;
            vulkan::GBuffer gbuffer;
        };
        auto render_fn = [](VkCommandBuffer cmd, uint32_t frame_index, void* user_data) {
            auto* data = static_cast<RenderCallbackData*>(user_data);
            data->renderer->RenderFrame(cmd, data->gbuffer, frame_index);
        };

        RenderCallbackData cb_data{};
        cb_data.renderer = &renderer_;
        cb_data.gbuffer = gbuffer;

        ref_result_ = capture::AccumulateFrames(
            readback_ctx_,
            gbuffer_.NoisyDiffuseImage(),
            gbuffer_.NoisySpecularImage(),
            config_.width, config_.height,
            config_.ref_frames,
            base_frame_index,
            render_fn,
            &cb_data);
    }

    return true;
}

WriteResult GenerationSession::WriteFrameFromJob(WriteJob& job) {
    uint32_t pixels = job.width * job.height;

    // Compute log-average luminance from reference (high-SPP) data
    auto lum_result = capture::ComputeLogAverageLuminance(
        job.ref_result.diffuse_f32.data(),
        job.ref_result.specular_f32.data(),
        pixels);

    // Validation: reject near-black images
    float nan_fraction = static_cast<float>(lum_result.nan_count) / lum_result.total_pixels;
    bool is_near_black = lum_result.log_average < config_.black_threshold;
    bool is_excessive_nan = nan_fraction > config_.nan_threshold;

    if (is_near_black || is_excessive_nan) {
        if (is_near_black)
            std::fprintf(stderr, "  [%s] %s: near-black (L_avg=%.6f)\n",
                         job.subdirectory.c_str(),
                         config_.force_write ? "WARNING" : "SKIPPED",
                         lum_result.log_average);
        if (is_excessive_nan)
            std::fprintf(stderr, "  [%s] %s: excessive NaN (%.2f%%, %u/%u pixels)\n",
                         job.subdirectory.c_str(),
                         config_.force_write ? "WARNING" : "SKIPPED",
                         nan_fraction * 100.0f,
                         lum_result.nan_count, lum_result.total_pixels);

        if (!config_.force_write) {
            if (is_near_black) {
                skipped_viewpoints_.push_back({config_.viewpoints[job.index].id,
                                               "near_black", lum_result.log_average});
                return WriteResult::kSkippedBlack;
            }
            skipped_viewpoints_.push_back({config_.viewpoints[job.index].id,
                                           "excessive_nan", nan_fraction});
            return WriteResult::kSkippedNaN;
        }
    }

    // Normalize to mid-gray (0.18) — clamp denominator to avoid explosion on near-black
    float safe_avg = std::max(lum_result.log_average, 1e-6f);
    float norm_mul = 0.18f / safe_avg;

    // Apply normalization to raw FP16 noisy diffuse/specular (RGB only — skip alpha)
    for (uint32_t i = 0; i < pixels; ++i) {
        auto base = static_cast<size_t>(i) * 4;
        for (int c = 0; c < 3; ++c) {
            float v = capture::HalfToFloat(job.noisy_diffuse_raw[base + c]);
            job.noisy_diffuse_raw[base + c] = capture::FloatToHalf(v * norm_mul);
        }
        for (int c = 0; c < 3; ++c) {
            float v = capture::HalfToFloat(job.noisy_specular_raw[base + c]);
            job.noisy_specular_raw[base + c] = capture::FloatToHalf(v * norm_mul);
        }
    }
    // Apply normalization to float reference diffuse/specular (RGB only — skip alpha)
    for (uint32_t i = 0; i < static_cast<uint32_t>(job.ref_result.diffuse_f32.size() / 4); ++i) {
        auto base = static_cast<size_t>(i) * 4;
        job.ref_result.diffuse_f32[base + 0] *= norm_mul;
        job.ref_result.diffuse_f32[base + 1] *= norm_mul;
        job.ref_result.diffuse_f32[base + 2] *= norm_mul;
    }
    for (uint32_t i = 0; i < static_cast<uint32_t>(job.ref_result.specular_f32.size() / 4); ++i) {
        auto base = static_cast<size_t>(i) * 4;
        job.ref_result.specular_f32[base + 0] *= norm_mul;
        job.ref_result.specular_f32[base + 1] *= norm_mul;
        job.ref_result.specular_f32[base + 2] *= norm_mul;
    }

    // Extract depth (R channel from RG16F)
    std::vector<float> depth_f32(pixels);
    capture::ExtractDepthFromRG16F(job.linear_depth_raw.data(), depth_f32.data(), pixels);

    // Build raw input frame
    capture::RawInputFrame input{};
    input.noisy_diffuse = job.noisy_diffuse_raw.data();
    input.noisy_specular = job.noisy_specular_raw.data();
    input.diffuse_albedo = job.diffuse_albedo_raw.data();
    input.specular_albedo = job.specular_albedo_raw.data();
    input.world_normals = job.world_normals_raw.data();
    input.linear_depth = depth_f32.data();
    input.motion_vectors = job.motion_vectors_raw.data();

    // Build target frame from accumulated reference
    capture::TargetFrame target{};
    target.ref_diffuse = job.ref_result.diffuse_f32.data();
    target.ref_specular = job.ref_result.specular_f32.data();

    // EXR metadata: normalization multiplier and log-average luminance
    std::pair<std::string, float> metadata_entries[] = {
        {"normalization_multiplier", norm_mul},
        {"log_average_luminance", lum_result.log_average},
    };

    if (!writer_.WriteFrameRaw(input, target, job.subdirectory, metadata_entries))
        return WriteResult::kError;

    return WriteResult::kSuccess;
}

}  // namespace monti::app::datagen
