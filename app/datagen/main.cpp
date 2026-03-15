#include "../core/vulkan_context.h"
#include "../core/EnvironmentLoader.h"
#include "../core/GBufferImages.h"
#include "CameraSetup.h"
#include "GenerationSession.h"

#include <monti/capture/Writer.h>
#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>
#include <monti/vulkan/ProcAddrHelpers.h>
#include <monti/scene/Scene.h>

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

// GltfLoader is in scene/src (not a public header)
#include "../../scene/src/gltf/GltfLoader.h"

#ifndef MONTI_SHADER_SPV_DIR
#define MONTI_SHADER_SPV_DIR "build/shaders"
#endif

int main(int argc, char* argv[]) {
    // ── CLI parsing ──
    CLI::App app{"Monti Datagen — headless training data generator"};

    std::string scene_path;
    app.add_option("scene", scene_path, "Path to .glb/.gltf scene file")->required();

    std::string output_dir = "./capture/";
    app.add_option("--output", output_dir, "Output directory for EXR files");

    uint32_t width = 960;
    app.add_option("--width", width, "Render width in pixels");

    uint32_t height = 540;
    app.add_option("--height", height, "Render height in pixels");

    uint32_t spp = 4;
    app.add_option("--spp", spp, "Noisy samples per pixel");

    uint32_t ref_frames = 64;
    app.add_option("--ref-frames", ref_frames, "Frames to accumulate for reference");

    float exposure = 0.0f;
    app.add_option("--exposure", exposure, "Exposure EV100");

    std::string env_path;
    app.add_option("--env", env_path, "Environment map EXR file")->check(CLI::ExistingFile);

    CLI11_PARSE(app, argc, argv);

    // ── Print configuration ──
    std::printf("monti_datagen configuration:\n");
    std::printf("  Scene:          %s\n", scene_path.c_str());
    std::printf("  Resolution:     %ux%u\n", width, height);
    std::printf("  Noisy SPP:      %u\n", spp);
    std::printf("  Reference SPP:  %u (%u frames x %u)\n", ref_frames * spp, ref_frames, spp);
    std::printf("  Exposure:       %.1f EV100\n", exposure);
    std::printf("  Output:         %s\n", output_dir.c_str());
    if (!env_path.empty())
        std::printf("  Environment:    %s\n", env_path.c_str());
    std::printf("\n");

    // ── Headless Vulkan context (no window, no surface, no swapchain) ──
    monti::app::VulkanContext ctx;
    if (!ctx.CreateInstance()) return EXIT_FAILURE;
    if (!ctx.CreateDevice(std::nullopt)) return EXIT_FAILURE;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx.PhysicalDevice(), &props);
    std::printf("Device: %s\n\n", props.deviceName);

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
    auto camera = monti::app::datagen::ComputeDefaultCamera(scene);
    camera.exposure_ev100 = exposure;
    scene.SetActiveCamera(camera);

    // ── Create renderer ──
    monti::vulkan::RendererDesc renderer_desc{};
    renderer_desc.device = ctx.Device();
    renderer_desc.physical_device = ctx.PhysicalDevice();
    renderer_desc.queue = ctx.GraphicsQueue();
    renderer_desc.queue_family_index = ctx.QueueFamilyIndex();
    renderer_desc.allocator = ctx.Allocator();
    renderer_desc.width = width;
    renderer_desc.height = height;
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

    // ── Create G-buffer images (with TRANSFER_SRC for readback) ──
    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    if (!gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                               width, height, gbuf_cmd,
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        std::fprintf(stderr, "Failed to create G-buffer images\n");
        return EXIT_FAILURE;
    }
    ctx.SubmitAndWait(gbuf_cmd);

    // ── Create capture writer (native scale — single resolution) ──
    monti::capture::WriterDesc writer_desc{};
    writer_desc.output_dir = output_dir;
    writer_desc.input_width = width;
    writer_desc.input_height = height;
    writer_desc.scale_mode = monti::capture::ScaleMode::kNative;

    auto writer = monti::capture::Writer::Create(writer_desc);
    if (!writer) {
        std::fprintf(stderr, "Failed to create capture writer\n");
        return EXIT_FAILURE;
    }

    // ── Run generation ──
    monti::app::datagen::GenerationConfig gen_config{};
    gen_config.width = width;
    gen_config.height = height;
    gen_config.spp = spp;
    gen_config.ref_frames = ref_frames;
    gen_config.exposure = exposure;
    gen_config.output_dir = output_dir;

    monti::app::datagen::GenerationSession session(ctx, *renderer, gbuffer_images,
                                                   *writer, gen_config);
    if (!session.Run()) {
        std::fprintf(stderr, "Generation failed\n");
        return EXIT_FAILURE;
    }

    // ── Cleanup ──
    ctx.WaitIdle();
    for (auto& buf : gpu_buffers)
        monti::vulkan::DestroyGpuBuffer(ctx.Allocator(), buf);

    return EXIT_SUCCESS;
}
