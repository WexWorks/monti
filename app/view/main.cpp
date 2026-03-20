#include "../core/vulkan_context.h"
#include "../core/CameraSetup.h"
#include "../core/EnvironmentLoader.h"
#include "../core/frame_resources.h"
#include "../core/GBufferImages.h"
#include "../core/ToneMapper.h"
#include "CameraController.h"
#include "Panels.h"
#include "swapchain.h"
#include "UiRenderer.h"

#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>
#include <monti/vulkan/ProcAddrHelpers.h>
#include <monti/vulkan/VulkanBarriers.h>
#include <monti/scene/Light.h>
#include <monti/scene/Scene.h>
#include <deni/vulkan/Denoiser.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

// GltfLoader is in scene/src (not a public header)
#include "../../scene/src/gltf/GltfLoader.h"

#ifndef APP_SHADER_SPV_DIR
#define APP_SHADER_SPV_DIR "build/app_shaders"
#endif

#ifndef MONTI_SHADER_SPV_DIR
#define MONTI_SHADER_SPV_DIR "build/shaders"
#endif

#ifndef DENI_SHADER_SPV_DIR
#define DENI_SHADER_SPV_DIR "build/deni_shaders"
#endif

namespace {

constexpr uint32_t kDefaultWidth = 1280;
constexpr uint32_t kDefaultHeight = 720;
constexpr uint32_t kDefaultSpp = 4;
constexpr float kDefaultExposure = 0.0f;
constexpr float kSavedFlashDuration = 1.5f;

std::string SceneNameFromPath(std::string_view path) {
    // Extract basename without extension (preserves original casing)
    auto last_sep = path.find_last_of("/\\");
    auto basename = (last_sep != std::string_view::npos) ? path.substr(last_sep + 1) : path;
    auto dot = basename.find_last_of('.');
    if (dot != std::string_view::npos)
        basename = basename.substr(0, dot);
    return std::string(basename);
}

int LoadExistingViewpointCount(const std::string& path) {
    std::ifstream file(path);
    if (!file) return 0;
    try {
        nlohmann::json j;
        file >> j;
        if (j.is_array()) return static_cast<int>(j.size());
    } catch (...) {}
    return 0;
}

std::string GenerateViewpointId() {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint32_t> dist;
    char buf[9];
    std::snprintf(buf, sizeof(buf), "%08x", dist(rng));
    return std::string(buf);
}

void SaveViewpoint(const monti::app::CameraController& controller,
                   float exposure_ev,
                   const std::string& viewpoints_path,
                   monti::app::PanelState& panel_state) {
    auto vp = controller.CurrentViewpoint();

    // Read existing array or start fresh
    nlohmann::json arr = nlohmann::json::array();
    {
        std::ifstream in(viewpoints_path);
        if (in) {
            try {
                nlohmann::json existing;
                in >> existing;
                if (existing.is_array()) arr = std::move(existing);
            } catch (...) {}
        }
    }

    // Append new viewpoint
    nlohmann::json entry;
    auto id = GenerateViewpointId();
    entry["id"] = id;
    entry["position"] = {vp.position.x, vp.position.y, vp.position.z};
    entry["target"] = {vp.target.x, vp.target.y, vp.target.z};
    entry["fov"] = vp.fov_degrees;
    entry["exposure"] = exposure_ev;
    arr.push_back(std::move(entry));

    // Write back
    std::ofstream out(viewpoints_path);
    if (!out) {
        std::fprintf(stderr, "Failed to write viewpoints to %s\n", viewpoints_path.c_str());
        return;
    }
    out << arr.dump(2) << "\n";

    panel_state.saved_viewpoint_count = static_cast<int>(arr.size());
    panel_state.viewpoint_just_saved = true;
    panel_state.viewpoint_saved_timer = kSavedFlashDuration;

    std::printf("Saved viewpoint %d [%s] to %s (pos=[%.3f, %.3f, %.3f] fov=%.1f exp=%.1f)\n",
                panel_state.saved_viewpoint_count, id.c_str(), viewpoints_path.c_str(),
                vp.position.x, vp.position.y, vp.position.z,
                vp.fov_degrees, exposure_ev);
}

struct AppState {
    SDL_Window* window = nullptr;
    monti::app::VulkanContext* ctx = nullptr;
    monti::app::Swapchain* swapchain = nullptr;
    monti::app::FrameResources* frame_resources = nullptr;
    monti::app::GBufferImages* gbuffer_images = nullptr;
    monti::app::ToneMapper* tone_mapper = nullptr;
    monti::vulkan::Renderer* renderer = nullptr;
    deni::vulkan::Denoiser* denoiser = nullptr;
    monti::app::CameraController* camera_controller = nullptr;
    monti::app::UiRenderer* ui_renderer = nullptr;
    monti::app::Panels* panels = nullptr;
    monti::app::PanelState* panel_state = nullptr;
    monti::Scene* scene = nullptr;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    uint32_t current_frame = 0;
    uint32_t frame_index = 0;
    bool running = true;
    bool rendering = false;
};

bool RecreateSwapchain(AppState& state) {
    int w, h;
    SDL_GetWindowSizeInPixels(state.window, &w, &h);
    if (w <= 0 || h <= 0) return false;

    state.ctx->WaitIdle();
    if (!state.swapchain->Create(*state.ctx, state.surface,
                                  static_cast<uint32_t>(w), static_cast<uint32_t>(h)))
        return false;

    if (!state.frame_resources->RecreateRenderFinishedSemaphores(state.swapchain->ImageCount())) {
        state.running = false;
        return false;
    }

    if (state.ui_renderer) {
        if (!state.ui_renderer->Resize(*state.swapchain)) {
            state.running = false;
            return false;
        }
    }

    state.current_frame = 0;
    return true;
}

bool RenderFrame(AppState& state) {
    if (state.rendering) return true;
    state.rendering = true;

    auto& ctx = *state.ctx;
    auto& swapchain = *state.swapchain;
    auto& fr = *state.frame_resources;

    fr.WaitForFence(state.current_frame);

    uint32_t image_index;
    VkResult result = vkAcquireNextImageKHR(
        ctx.Device(), swapchain.Handle(), UINT64_MAX,
        fr.ImageAvailableSemaphore(state.current_frame), VK_NULL_HANDLE, &image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        if (!RecreateSwapchain(state)) {
            state.rendering = false;
            return false;
        }
        state.rendering = false;
        return true;
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        std::fprintf(stderr, "Failed to acquire swapchain image (VkResult: %d)\n", result);
        state.rendering = false;
        return false;
    }

    fr.ResetFence(state.current_frame);
    fr.ResetCommandBuffer(state.current_frame);

    VkCommandBuffer cmd = fr.CommandBuffer(state.current_frame);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin_info);

    // ── Trace rays ──
    auto debug_mode = state.panel_state
        ? state.panel_state->debug_mode
        : monti::app::DebugMode::kOff;

    auto gbuffer = state.gbuffer_images->ToGBuffer();
    state.renderer->SetDebugMode(static_cast<uint32_t>(debug_mode));
    state.renderer->RenderFrame(cmd, gbuffer, state.frame_index);

    // ── Memory barrier: RT writes → compute/transfer reads ──
    VkMemoryBarrier2 rt_barrier{};
    rt_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    rt_barrier.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    rt_barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    rt_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    rt_barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                               VK_ACCESS_2_TRANSFER_READ_BIT;

    VkDependencyInfo barrier_dep{};
    barrier_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    barrier_dep.memoryBarrierCount = 1;
    barrier_dep.pMemoryBarriers = &rt_barrier;
    vkCmdPipelineBarrier2(cmd, &barrier_dep);

    // ── Transition swapchain image to TRANSFER_DST ──
    {
        auto barrier = monti::vulkan::MakeImageBarrier(
            swapchain.Image(image_index),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT);
        monti::vulkan::CmdPipelineBarrier(cmd, {&barrier, 1}, vkCmdPipelineBarrier2);
    }

    VkImageBlit blit_region{};
    blit_region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit_region.srcOffsets[0] = {0, 0, 0};
    blit_region.srcOffsets[1] = {static_cast<int32_t>(state.gbuffer_images->Width()),
                                  static_cast<int32_t>(state.gbuffer_images->Height()), 1};
    blit_region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit_region.dstOffsets[0] = {0, 0, 0};
    blit_region.dstOffsets[1] = {static_cast<int32_t>(swapchain.Extent().width),
                                  static_cast<int32_t>(swapchain.Extent().height), 1};

    if (debug_mode == monti::app::DebugMode::kOff) {
        // ── Normal pipeline: denoise → tonemap → blit ──
        deni::vulkan::DenoiserInput denoise_input{};
        denoise_input.noisy_diffuse = gbuffer.noisy_diffuse;
        denoise_input.noisy_specular = gbuffer.noisy_specular;
        denoise_input.motion_vectors = gbuffer.motion_vectors;
        denoise_input.linear_depth = gbuffer.linear_depth;
        denoise_input.world_normals = gbuffer.world_normals;
        denoise_input.diffuse_albedo = gbuffer.diffuse_albedo;
        denoise_input.specular_albedo = gbuffer.specular_albedo;
        denoise_input.render_width = state.gbuffer_images->Width();
        denoise_input.render_height = state.gbuffer_images->Height();
        denoise_input.reset_accumulation = (state.frame_index == 0);

        auto denoise_output = state.denoiser->Denoise(cmd, denoise_input);
        state.tone_mapper->Apply(cmd, denoise_output.denoised_image);

        vkCmdBlitImage(cmd,
            state.tone_mapper->OutputImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            swapchain.Image(image_index), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit_region, VK_FILTER_NEAREST);
    } else {
        // ── Debug visualization: blit selected G-buffer directly to swapchain ──
        VkImage debug_image = VK_NULL_HANDLE;
        switch (debug_mode) {
        case monti::app::DebugMode::kNormals:
            debug_image = state.gbuffer_images->WorldNormalsImage();
            break;
        case monti::app::DebugMode::kAlbedo:
            debug_image = state.gbuffer_images->DiffuseAlbedoImage();
            break;
        case monti::app::DebugMode::kDepth:
            // Shader writes Reinhard-tonemapped grayscale depth to noisy_diffuse when debug_mode == 3
            debug_image = state.gbuffer_images->NoisyDiffuseImage();
            break;
        case monti::app::DebugMode::kMotionVectors:
            // Shader writes amplified abs(motion) to noisy_diffuse when debug_mode == 4
            debug_image = state.gbuffer_images->NoisyDiffuseImage();
            break;
        case monti::app::DebugMode::kNoisy:
            debug_image = state.gbuffer_images->NoisyDiffuseImage();
            break;
        default: break;
        }

        if (debug_image != VK_NULL_HANDLE) {
            // Transition G-buffer image from GENERAL to TRANSFER_SRC_OPTIMAL
            auto src_barrier = monti::vulkan::MakeImageBarrier(
                debug_image,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_READ_BIT);
            monti::vulkan::CmdPipelineBarrier(cmd, {&src_barrier, 1}, vkCmdPipelineBarrier2);

            vkCmdBlitImage(cmd,
                debug_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                swapchain.Image(image_index), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &blit_region, VK_FILTER_NEAREST);

            // Transition G-buffer image back to GENERAL
            auto restore_barrier = monti::vulkan::MakeImageBarrier(
                debug_image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            monti::vulkan::CmdPipelineBarrier(cmd, {&restore_barrier, 1}, vkCmdPipelineBarrier2);
        }
    }

    // ── ImGui render pass (TRANSFER_DST → PRESENT_SRC via render pass) ──
    if (state.ui_renderer)
        state.ui_renderer->EndFrame(cmd, image_index);

    vkEndCommandBuffer(cmd);

    // ── Submit ──
    VkSemaphore wait_semaphores[] = {fr.ImageAvailableSemaphore(state.current_frame)};
    VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphore signal_semaphores[] = {fr.RenderFinishedSemaphore(image_index)};

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = wait_semaphores;
    submit_info.pWaitDstStageMask = wait_stages;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores;

    result = vkQueueSubmit(ctx.GraphicsQueue(), 1, &submit_info,
                           fr.InFlightFence(state.current_frame));
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to submit command buffer (VkResult: %d)\n", result);
        state.rendering = false;
        return false;
    }

    // ── Present ──
    VkSwapchainKHR swapchains[] = {swapchain.Handle()};
    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = signal_semaphores;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = swapchains;
    present_info.pImageIndices = &image_index;

    result = vkQueuePresentKHR(ctx.GraphicsQueue(), &present_info);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        if (!RecreateSwapchain(state)) {
            state.rendering = false;
            return false;
        }
    } else if (result != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to present (VkResult: %d)\n", result);
        state.rendering = false;
        return false;
    }

    state.current_frame = (state.current_frame + 1) % monti::app::FrameResources::kFramesInFlight;
    ++state.frame_index;
    state.rendering = false;
    return true;
}

// Called during the OS modal resize loop so we can render while the window is being dragged
bool EventWatcher(void* userdata, SDL_Event* event) {
    if (event->type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
        auto* state = static_cast<AppState*>(userdata);
        if (RecreateSwapchain(*state))
            RenderFrame(*state);
    }
    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    // ── CLI parsing ──
    CLI::App app{"Monti View — interactive glTF path tracer"};

    std::string scene_path;
    app.add_option("scene", scene_path, "Path to .glb/.gltf scene file")->required();

    uint32_t window_width = kDefaultWidth;
    app.add_option("--width", window_width, "Window width in pixels");

    uint32_t window_height = kDefaultHeight;
    app.add_option("--height", window_height, "Window height in pixels");

    uint32_t spp = kDefaultSpp;
    app.add_option("--spp", spp, "Samples per pixel");

    float exposure = kDefaultExposure;
    app.add_option("--exposure", exposure, "Exposure EV100");

    std::string env_path;
    app.add_option("--env", env_path, "Environment map EXR file")->check(CLI::ExistingFile);

    std::string viewpoint_dir = ".";
    app.add_option("--viewpoint-dir", viewpoint_dir,
                   "Directory for saved viewpoints JSON (default: current directory)");

    CLI11_PARSE(app, argc, argv);

    // ── SDL + window ──
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "Failed to initialize SDL: %s\n", SDL_GetError());
        return EXIT_FAILURE;
    }

    auto sdl_cleanup = std::unique_ptr<void, decltype([](void*) { SDL_Quit(); })>(
        reinterpret_cast<void*>(1));

    SDL_Window* window = SDL_CreateWindow(
        "Monti View",
        static_cast<int>(window_width), static_cast<int>(window_height),
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window) {
        std::fprintf(stderr, "Failed to create window: %s\n", SDL_GetError());
        return EXIT_FAILURE;
    }

    auto window_cleanup = std::unique_ptr<SDL_Window, decltype([](SDL_Window* w) {
        SDL_DestroyWindow(w);
    })>(window);

    // ── Vulkan instance ──
    monti::app::VulkanContext ctx;

    uint32_t sdl_ext_count = 0;
    auto sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&sdl_ext_count);
    if (!sdl_extensions) {
        std::fprintf(stderr, "Failed to get SDL Vulkan extensions: %s\n", SDL_GetError());
        return EXIT_FAILURE;
    }

    std::span<const char* const> ext_span(sdl_extensions, sdl_ext_count);
    if (!ctx.CreateInstance(ext_span)) return EXIT_FAILURE;

    VkSurfaceKHR surface;
    if (!SDL_Vulkan_CreateSurface(window, ctx.Instance(), nullptr, &surface)) {
        std::fprintf(stderr, "Failed to create Vulkan surface: %s\n", SDL_GetError());
        return EXIT_FAILURE;
    }

    if (!ctx.CreateDevice(surface)) return EXIT_FAILURE;

    // ── Swapchain ──
    monti::app::Swapchain swapchain;
    if (!swapchain.Create(ctx, surface, window_width, window_height))
        return EXIT_FAILURE;

    // ── Frame resources ──
    monti::app::FrameResources frame_resources;
    if (!frame_resources.Create(ctx)) return EXIT_FAILURE;
    if (!frame_resources.RecreateRenderFinishedSemaphores(swapchain.ImageCount()))
        return EXIT_FAILURE;

    // ── Load scene ──
    monti::Scene scene;
    auto load_result = monti::gltf::LoadGltf(scene, scene_path);
    if (!load_result.success) {
        std::fprintf(stderr, "Failed to load scene: %s\n", load_result.error_message.c_str());
        return EXIT_FAILURE;
    }
    std::printf("Loaded scene: %zu nodes, %zu meshes, %zu materials\n",
                scene.Nodes().size(), scene.Meshes().size(), scene.Materials().size());

    // ── Set up environment map ──
    if (!env_path.empty()) {
        auto env_tex = monti::app::LoadExrEnvironment(env_path);
        if (env_tex) {
            auto tex_id = scene.AddTexture(std::move(*env_tex), "env_map");
            monti::EnvironmentLight env{};
            env.hdr_lat_long = tex_id;
            scene.SetEnvironmentLight(env);
        } else {
            std::fprintf(stderr, "Warning: falling back to default environment\n");
        }
    }
    if (!scene.GetEnvironmentLight()) {
        auto tex_id = scene.AddTexture(
            monti::app::MakeDefaultEnvironment(0.5f, 0.5f, 0.5f), "default_env");
        monti::EnvironmentLight env{};
        env.hdr_lat_long = tex_id;
        scene.SetEnvironmentLight(env);
        std::printf("Using default mid-gray environment (use --env to specify an HDR map)\n");
    }

    // ── Auto-fit camera ──
    auto scene_aabb = monti::app::ComputeSceneAABB(scene);
    auto camera = monti::app::AutoFitCamera(scene_aabb);
    camera.exposure_ev100 = exposure;
    scene.SetActiveCamera(camera);

    // ── Create renderer ──
    monti::vulkan::RendererDesc renderer_desc{};
    renderer_desc.device = ctx.Device();
    renderer_desc.physical_device = ctx.PhysicalDevice();
    renderer_desc.queue = ctx.GraphicsQueue();
    renderer_desc.queue_family_index = ctx.QueueFamilyIndex();
    renderer_desc.allocator = ctx.Allocator();
    renderer_desc.width = window_width;
    renderer_desc.height = window_height;
    renderer_desc.samples_per_pixel = spp;
    renderer_desc.shader_dir = MONTI_SHADER_SPV_DIR;
    monti::vulkan::FillRendererProcAddrs(renderer_desc, ctx.Instance(),
                                         ctx.GetDeviceProcAddr(), ctx.GetInstanceProcAddr());

    auto renderer = monti::vulkan::Renderer::Create(renderer_desc);
    if (!renderer) {
        std::fprintf(stderr, "Failed to create renderer\n");
        return EXIT_FAILURE;
    }
    renderer->SetScene(&scene);

    // ── Upload meshes ──
    auto procs = monti::vulkan::MakeGpuBufferProcs(vkGetBufferDeviceAddress, vkCmdPipelineBarrier2);
    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = monti::vulkan::UploadAndRegisterMeshes(
        *renderer, ctx.Allocator(), ctx.Device(), upload_cmd, load_result.mesh_data, procs);
    ctx.SubmitAndWait(upload_cmd);

    if (gpu_buffers.empty()) {
        std::fprintf(stderr, "Failed to upload mesh data\n");
        return EXIT_FAILURE;
    }

    // ── Create G-buffer images ──
    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    if (!gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                               window_width, window_height, gbuf_cmd,
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        std::fprintf(stderr, "Failed to create G-buffer images\n");
        return EXIT_FAILURE;
    }
    ctx.SubmitAndWait(gbuf_cmd);

    // ── Create denoiser ──
    deni::vulkan::DenoiserDesc denoiser_desc{};
    denoiser_desc.device = ctx.Device();
    denoiser_desc.physical_device = ctx.PhysicalDevice();
    denoiser_desc.width = window_width;
    denoiser_desc.height = window_height;
    denoiser_desc.allocator = ctx.Allocator();
    denoiser_desc.shader_dir = DENI_SHADER_SPV_DIR;
    monti::vulkan::FillDenoiserProcAddrs(denoiser_desc, ctx.GetDeviceProcAddr());

    auto denoiser = deni::vulkan::Denoiser::Create(denoiser_desc);
    if (!denoiser) {
        std::fprintf(stderr, "Failed to create denoiser\n");
        return EXIT_FAILURE;
    }

    // ── Create tone mapper ──
    // One-shot render+denoise to obtain the denoiser output view,
    // which the tone mapper needs for its descriptor set.
    monti::app::ToneMapper tone_mapper;
    {
        VkCommandBuffer init_cmd = ctx.BeginOneShot();
        auto gbuffer = gbuffer_images.ToGBuffer();
        renderer->RenderFrame(init_cmd, gbuffer, 0);

        VkMemoryBarrier2 rt_to_compute{};
        rt_to_compute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        rt_to_compute.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        rt_to_compute.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        rt_to_compute.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        rt_to_compute.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        VkDependencyInfo barrier_dep{};
        barrier_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        barrier_dep.memoryBarrierCount = 1;
        barrier_dep.pMemoryBarriers = &rt_to_compute;
        vkCmdPipelineBarrier2(init_cmd, &barrier_dep);

        deni::vulkan::DenoiserInput init_input{};
        init_input.noisy_diffuse = gbuffer.noisy_diffuse;
        init_input.noisy_specular = gbuffer.noisy_specular;
        init_input.motion_vectors = gbuffer.motion_vectors;
        init_input.linear_depth = gbuffer.linear_depth;
        init_input.world_normals = gbuffer.world_normals;
        init_input.diffuse_albedo = gbuffer.diffuse_albedo;
        init_input.specular_albedo = gbuffer.specular_albedo;
        init_input.render_width = window_width;
        init_input.render_height = window_height;
        init_input.reset_accumulation = true;

        auto init_output = denoiser->Denoise(init_cmd, init_input);
        ctx.SubmitAndWait(init_cmd);

        if (!tone_mapper.Create(ctx.Device(), ctx.Allocator(), APP_SHADER_SPV_DIR,
                                window_width, window_height, init_output.denoised_color)) {
            std::fprintf(stderr, "Failed to create tone mapper\n");
            return EXIT_FAILURE;
        }
        tone_mapper.SetExposure(exposure);
    }

    // ── Compute scene diagonal for camera controller ──
    float scene_diagonal = monti::app::ClampSceneDiagonal(scene_aabb.Diagonal());

    // ── Count triangles for scene info ──
    uint32_t total_triangles = 0;
    for (const auto& mesh : scene.Meshes())
        total_triangles += mesh.index_count / 3;

    // ── Initialize camera controller ──
    monti::app::CameraController camera_controller;
    camera_controller.Initialize(camera, scene_diagonal);

    // ── Initialize ImGui / UiRenderer ──
    monti::app::UiRenderer ui_renderer;
    if (!ui_renderer.Initialize(ctx, window, swapchain)) {
        std::fprintf(stderr, "Failed to initialize ImGui\n");
        return EXIT_FAILURE;
    }

    // ── Initialize panels ──
    monti::app::Panels panels;
    monti::app::PanelState panel_state{};
    panel_state.spp = static_cast<int>(spp);
    panel_state.exposure_ev = exposure;
    panel_state.env_rotation_degrees = 0.0f;
    panel_state.node_count = static_cast<uint32_t>(scene.Nodes().size());
    panel_state.mesh_count = static_cast<uint32_t>(scene.Meshes().size());
    panel_state.material_count = static_cast<uint32_t>(scene.Materials().size());
    panel_state.triangle_count = total_triangles;
    {
        // Extract filename from path for display
        auto last_sep = scene_path.find_last_of("/\\");
        panel_state.scene_name = (last_sep != std::string::npos)
            ? scene_path.substr(last_sep + 1) : scene_path;
    }

    // ── Viewpoint capture setup ──
    auto viewpoints_out = (std::filesystem::path(viewpoint_dir) /
                           (SceneNameFromPath(scene_path) + ".json")).string();
    panel_state.viewpoints_out_path = viewpoints_out;
    panel_state.saved_viewpoint_count = LoadExistingViewpointCount(viewpoints_out);
    if (panel_state.saved_viewpoint_count > 0)
        std::printf("Existing viewpoints file %s has %d entries\n",
                    viewpoints_out.c_str(), panel_state.saved_viewpoint_count);

    // ── Set up app state ──
    AppState state{};
    state.window = window;
    state.ctx = &ctx;
    state.swapchain = &swapchain;
    state.frame_resources = &frame_resources;
    state.gbuffer_images = &gbuffer_images;
    state.tone_mapper = &tone_mapper;
    state.renderer = renderer.get();
    state.denoiser = denoiser.get();
    state.camera_controller = &camera_controller;
    state.ui_renderer = &ui_renderer;
    state.panels = &panels;
    state.panel_state = &panel_state;
    state.scene = &scene;
    state.surface = surface;
    state.frame_index = 1;  // Frame 0 was used by init pass

    SDL_AddEventWatch(EventWatcher, &state);

    std::printf("Render pipeline ready: %ux%u, %u spp, exposure %.1f EV\n",
                window_width, window_height, spp, exposure);

    // ── Main loop ──
    uint64_t last_perf = SDL_GetPerformanceCounter();
    uint64_t perf_freq = SDL_GetPerformanceFrequency();

    while (state.running) {
        // ── Frame timing ──
        uint64_t now_perf = SDL_GetPerformanceCounter();
        float dt = static_cast<float>(now_perf - last_perf) /
                   static_cast<float>(perf_freq);
        last_perf = now_perf;

        panel_state.frame_time_ms = dt * 1000.0f;
        panel_state.fps = (dt > 0.0f) ? (1.0f / dt) : 0.0f;

        // ── Event handling ──
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                state.running = false;
                continue;
            }
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE) {
                state.running = false;
                continue;
            }

            // Forward to ImGui first
            ui_renderer.ProcessEvent(event);

            // Tab toggles settings panel
            if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat &&
                event.key.key == SDLK_TAB) {
                panel_state.show_settings = !panel_state.show_settings;
                continue;
            }

            // P saves current viewpoint
            if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat &&
                event.key.key == SDLK_P &&
                !ui_renderer.WantCaptureKeyboard()) {
                SaveViewpoint(camera_controller, panel_state.exposure_ev,
                              panel_state.viewpoints_out_path, panel_state);
                continue;
            }

            // Forward to camera controller only if ImGui doesn't want input
            if (!ui_renderer.WantCaptureMouse() && !ui_renderer.WantCaptureKeyboard())
                camera_controller.ProcessEvent(event);
        }

        if (!state.running) break;

        // ── Update saved flash timer ──
        if (panel_state.viewpoint_just_saved) {
            panel_state.viewpoint_saved_timer -= dt;
            if (panel_state.viewpoint_saved_timer <= 0.0f)
                panel_state.viewpoint_just_saved = false;
        }

        // ── Update camera ──
        auto cam = camera_controller.Update(dt);
        cam.exposure_ev100 = panel_state.exposure_ev;
        scene.SetActiveCamera(cam);

        // ── Apply panel state changes ──
        renderer->SetSamplesPerPixel(static_cast<uint32_t>(panel_state.spp));
        renderer->SetMaxBounces(static_cast<uint32_t>(panel_state.max_bounces));
        tone_mapper.SetExposure(panel_state.exposure_ev);

        // Apply environment rotation and intensity
        if (auto* env_ptr = scene.GetEnvironmentLight()) {
            monti::EnvironmentLight env_copy = *env_ptr;
            env_copy.rotation = glm::radians(panel_state.env_rotation_degrees);
            env_copy.intensity = panel_state.env_intensity;
            scene.SetEnvironmentLight(env_copy);
        }

        // Update read-only panel fields
        panel_state.camera_mode = camera_controller.Mode();
        panel_state.camera_position = cam.position;
        panel_state.camera_fov_degrees = glm::degrees(camera_controller.Fov());

        // ── ImGui frame ──
        ui_renderer.BeginFrame();
        panels.Draw(panel_state);

        // ── Render ──
        if (!RenderFrame(state)) {
            state.running = false;
            break;
        }
    }

    // ── Clean shutdown ──
    SDL_RemoveEventWatch(EventWatcher, &state);
    ctx.WaitIdle();

    // Destroy in reverse order
    ui_renderer.Destroy();
    tone_mapper.Destroy();
    denoiser.reset();
    for (auto& buf : gpu_buffers)
        monti::vulkan::DestroyGpuBuffer(ctx.Allocator(), buf);
    renderer.reset();
    gbuffer_images.Destroy();
    frame_resources.Destroy();
    swapchain.Destroy();
    vkDestroySurfaceKHR(ctx.Instance(), surface, nullptr);

    return EXIT_SUCCESS;
}
