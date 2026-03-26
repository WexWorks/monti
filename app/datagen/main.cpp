#include "../core/vulkan_context.h"
#include "../core/EnvironmentLoader.h"
#include "../core/GBufferImages.h"
#include "../core/CameraSetup.h"
#include "GenerationSession.h"

#include <monti/capture/Writer.h>
#include <monti/vulkan/Renderer.h>
#include <monti/vulkan/GpuBufferUtils.h>
#include <monti/vulkan/ProcAddrHelpers.h>
#include <monti/scene/Scene.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

// GltfLoader is in scene/src (not a public header)
#include "../../scene/src/gltf/GltfLoader.h"

#ifndef MONTI_SHADER_SPV_DIR
#define MONTI_SHADER_SPV_DIR "build/shaders"
#endif

#ifndef CAPTURE_SHADER_SPV_DIR
#define CAPTURE_SHADER_SPV_DIR "build/capture_shaders"
#endif

namespace {
using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
}  // namespace

int main(int argc, char* argv[]) {
    // ── CLI parsing ──
    CLI::App app{"Monti Datagen — headless training data generator"};

    std::string scene_path;
    app.add_option("scene", scene_path, "Path to .glb/.gltf scene file")->required();

    std::string output_dir = "./capture/";
    app.add_option("--output", output_dir, "Output directory for EXR files");

    constexpr uint32_t kDefaultWidth = 960;
    constexpr uint32_t kDefaultHeight = 540;
    uint32_t width = kDefaultWidth;
    app.add_option("--width", width, "Render width in pixels");

    uint32_t height = kDefaultHeight;
    app.add_option("--height", height, "Render height in pixels");

    uint32_t spp = 4;
    app.add_option("--spp", spp, "Noisy samples per pixel");

    constexpr uint32_t kDefaultRefFrames = 64;
    uint32_t ref_frames = kDefaultRefFrames;
    app.add_option("--ref-frames", ref_frames, "Frames to accumulate for reference");

    std::vector<float> position_vec;
    auto* pos_opt = app.add_option("--position", position_vec,
                                   "Camera world-space position (X Y Z)")
                       ->expected(3);

    std::vector<float> target_vec;
    auto* tgt_opt = app.add_option("--target", target_vec,
                                   "Camera look-at target point (X Y Z)")
                       ->expected(3);

    float fov = monti::app::kDefaultFovDegrees;
    app.add_option("--fov", fov, "Vertical FOV in degrees");

    std::string viewpoints_path;
    app.add_option("--viewpoints", viewpoints_path,
                   "JSON file with array of viewpoint entries")
        ->check(CLI::ExistingFile)
        ->excludes(pos_opt)
        ->excludes(tgt_opt);

    std::string compression_str = "none";
    app.add_option("--exr-compression", compression_str,
                   "EXR compression: none (default), zip")
        ->check(CLI::IsMember({"none", "zip"}));

    std::string skipped_path;
    app.add_option("--skipped-path", skipped_path,
                   "Write skipped-viewpoints JSON to this path (optional)");

    CLI11_PARSE(app, argc, argv);

    // Validate: --position and --target must both be present or both absent
    if (pos_opt->count() != tgt_opt->count()) {
        std::fprintf(stderr, "Error: --position and --target must both be specified\n");
        return EXIT_FAILURE;
    }

    // ── Print configuration ──
    std::printf("monti_datagen configuration:\n");
    std::printf("  Scene:          %s\n", scene_path.c_str());
    std::printf("  Resolution:     %ux%u\n", width, height);
    std::printf("  Noisy SPP:      %u\n", spp);
    std::printf("  Reference SPP:  %u (%u frames x %u)\n", ref_frames * spp, ref_frames, spp);
    std::printf("  Output:         %s\n", output_dir.c_str());
    if (pos_opt->count())
        std::printf("  Position:       (%.2f, %.2f, %.2f)\n",
                    position_vec[0], position_vec[1], position_vec[2]);
    if (tgt_opt->count())
        std::printf("  Target:         (%.2f, %.2f, %.2f)\n",
                    target_vec[0], target_vec[1], target_vec[2]);
    std::printf("  FOV:            %.1f degrees\n", fov);
    if (!viewpoints_path.empty())
        std::printf("  Viewpoints:     %s\n", viewpoints_path.c_str());
    std::printf("\n");

    // ── Headless Vulkan context (no window, no surface, no swapchain) ──
    auto program_start = Clock::now();

    auto t_vulkan_start = Clock::now();
    monti::app::VulkanContext ctx;
    if (!ctx.CreateInstance()) return EXIT_FAILURE;
    if (!ctx.CreateDevice(std::nullopt)) return EXIT_FAILURE;
    auto t_vulkan_end = Clock::now();

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx.PhysicalDevice(), &props);
    std::printf("Device: %s\n\n", props.deviceName);

    // ── Load scene ──
    auto t_scene_start = Clock::now();
    monti::Scene scene;
    auto load_result = monti::gltf::LoadGltf(scene, scene_path);
    if (!load_result.success) {
        std::fprintf(stderr, "Failed to load scene: %s\n", load_result.error_message.c_str());
        return EXIT_FAILURE;
    }
    auto t_scene_end = Clock::now();
    std::printf("Loaded scene: %zu nodes, %zu meshes, %zu materials\n",
                scene.Nodes().size(), scene.Meshes().size(), scene.Materials().size());

    // ── Build viewpoints list (parse before env/lights to read per-VP fields) ──
    std::vector<monti::app::datagen::ViewpointEntry> viewpoints;

    if (!viewpoints_path.empty()) {
        // Parse JSON viewpoints file
        std::ifstream vp_file(viewpoints_path);
        if (!vp_file) {
            std::fprintf(stderr, "Failed to open viewpoints file: %s\n",
                        viewpoints_path.c_str());
            return EXIT_FAILURE;
        }
        nlohmann::json vp_json;
        try {
            vp_file >> vp_json;
        } catch (const nlohmann::json::parse_error& e) {
            std::fprintf(stderr, "Failed to parse viewpoints JSON: %s\n", e.what());
            return EXIT_FAILURE;
        }
        if (!vp_json.is_array() || vp_json.empty()) {
            std::fprintf(stderr, "Viewpoints JSON must be a non-empty array\n");
            return EXIT_FAILURE;
        }
        for (size_t idx = 0; idx < vp_json.size(); ++idx) {
            const auto& entry = vp_json[idx];
            if (!entry.contains("position") || !entry.contains("target")) {
                std::fprintf(stderr, "Viewpoint entry %zu missing position/target\n", idx);
                return EXIT_FAILURE;
            }
            auto pos = entry["position"].get<std::vector<float>>();
            auto tgt = entry["target"].get<std::vector<float>>();
            if (pos.size() != 3 || tgt.size() != 3) {
                std::fprintf(stderr, "Viewpoint entry %zu: position/target must have 3 elements\n", idx);
                return EXIT_FAILURE;
            }
            monti::app::datagen::ViewpointEntry vp{};
            vp.position = glm::vec3(pos[0], pos[1], pos[2]);
            vp.target = glm::vec3(tgt[0], tgt[1], tgt[2]);
            vp.fov_degrees = entry.value("fov", fov);
            vp.id = entry.value("id", std::format("vp_{}", idx));
            if (entry.contains("environment"))
                vp.environment = entry["environment"].get<std::string>();
            if (entry.contains("lights"))
                vp.lights = entry["lights"].get<std::string>();
            if (entry.contains("environmentBlur"))
                vp.environment_blur = entry["environmentBlur"].get<float>();
            if (entry.contains("environmentIntensity"))
                vp.environment_intensity = entry["environmentIntensity"].get<float>();
            viewpoints.push_back(vp);
        }
        std::printf("Loaded %zu viewpoints from %s\n\n", viewpoints.size(),
                    viewpoints_path.c_str());
    } else if (pos_opt->count()) {
        // Single CLI viewpoint
        monti::app::datagen::ViewpointEntry vp{};
        vp.position = glm::vec3(position_vec[0], position_vec[1], position_vec[2]);
        vp.target = glm::vec3(target_vec[0], target_vec[1], target_vec[2]);
        vp.fov_degrees = fov;
        vp.id = "cli_0";
        viewpoints.push_back(vp);
    } else {
        // Auto-fit camera from scene
        auto camera = monti::app::ComputeDefaultCamera(scene);
        monti::app::datagen::ViewpointEntry vp{};
        vp.position = camera.position;
        vp.target = camera.target;
        vp.fov_degrees = glm::degrees(camera.vertical_fov_radians);
        vp.id = "autofit_0";
        viewpoints.push_back(vp);
    }

    // ── Set up environment map (from first viewpoint or default) ──
    auto t_env_start = Clock::now();
    std::string env_path;
    bool show_env_background = false;
    constexpr float kDefaultEnvBlurLevel = 3.5f;
    float env_blur_level = kDefaultEnvBlurLevel;

    if (!viewpoints.empty() && viewpoints[0].environment.has_value())
        env_path = viewpoints[0].environment.value();
    if (!viewpoints.empty() && viewpoints[0].environment_blur.has_value()) {
        show_env_background = true;
        env_blur_level = viewpoints[0].environment_blur.value();
    }

    if (!env_path.empty()) {
        auto env_tex = monti::app::LoadExrEnvironment(env_path);
        if (env_tex) {
            auto tex_id = scene.AddTexture(std::move(*env_tex), "env_map");
            monti::EnvironmentLight env{};
            env.hdr_lat_long = tex_id;
            if (!viewpoints.empty() && viewpoints[0].environment_intensity.has_value())
                env.intensity = viewpoints[0].environment_intensity.value();
            scene.SetEnvironmentLight(env);
            std::printf("  Environment:    %s (intensity=%.1f)\n",
                        env_path.c_str(), env.intensity);
        } else {
            std::fprintf(stderr, "Warning: failed to load environment %s, using default\n",
                        env_path.c_str());
        }
    }
    if (!scene.GetEnvironmentLight()) {
        auto tex_id = scene.AddTexture(
            monti::app::MakeDefaultEnvironment(0.5f, 0.5f, 0.5f), "default_env");
        monti::EnvironmentLight env{};
        env.hdr_lat_long = tex_id;
        scene.SetEnvironmentLight(env);
        std::printf("Using default mid-gray environment\n");
    }

    // ── Load area lights from first viewpoint ──
    std::string lights_path;
    if (!viewpoints.empty() && viewpoints[0].lights.has_value())
        lights_path = viewpoints[0].lights.value();

    if (!lights_path.empty()) {
        std::ifstream lights_file(lights_path);
        if (!lights_file) {
            std::fprintf(stderr, "Failed to open lights file: %s\n", lights_path.c_str());
            return EXIT_FAILURE;
        }
        nlohmann::json lights_json;
        try {
            lights_file >> lights_json;
        } catch (const nlohmann::json::parse_error& e) {
            std::fprintf(stderr, "Failed to parse lights JSON: %s\n", e.what());
            return EXIT_FAILURE;
        }
        if (!lights_json.is_array() || lights_json.empty()) {
            std::fprintf(stderr, "Lights JSON must be a non-empty array\n");
            return EXIT_FAILURE;
        }
        for (size_t idx = 0; idx < lights_json.size(); ++idx) {
            const auto& entry = lights_json[idx];
            auto corner = entry.value("corner", std::vector<float>{});
            auto edge_a = entry.value("edge_a", std::vector<float>{});
            auto edge_b = entry.value("edge_b", std::vector<float>{});
            auto radiance = entry.value("radiance", std::vector<float>{});
            if (corner.size() != 3 || edge_a.size() != 3 ||
                edge_b.size() != 3 || radiance.size() != 3) {
                std::fprintf(stderr, "Light entry %zu: corner, edge_a, edge_b, "
                            "radiance must each have 3 components\n", idx);
                return EXIT_FAILURE;
            }
            if (radiance[0] < 0.0f || radiance[1] < 0.0f || radiance[2] < 0.0f) {
                std::fprintf(stderr, "Light entry %zu: radiance must be non-negative\n", idx);
                return EXIT_FAILURE;
            }
            monti::AreaLight light{};
            light.corner = glm::vec3(corner[0], corner[1], corner[2]);
            light.edge_a = glm::vec3(edge_a[0], edge_a[1], edge_a[2]);
            light.edge_b = glm::vec3(edge_b[0], edge_b[1], edge_b[2]);
            light.radiance = glm::vec3(radiance[0], radiance[1], radiance[2]);
            light.two_sided = entry.value("two_sided", false);
            scene.AddAreaLight(light);
        }
        std::printf("Loaded %zu area light(s) from %s\n\n", lights_json.size(),
                    lights_path.c_str());
    }
    auto t_env_end = Clock::now();

    // ── Create renderer ──
    auto t_renderer_start = Clock::now();
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

    // Default: transparent black background for training data
    // When environmentBlur is set in viewpoint, use blurred environment as background
    if (show_env_background)
        renderer->SetBackgroundMode(true, env_blur_level);
    else
        renderer->SetBackgroundMode(false);
    auto t_renderer_end = Clock::now();

    // ── Upload meshes ──
    auto t_mesh_start = Clock::now();
    auto procs = monti::vulkan::MakeGpuBufferProcs(vkGetBufferDeviceAddress, vkCmdPipelineBarrier2);
    VkCommandBuffer upload_cmd = ctx.BeginOneShot();
    auto gpu_buffers = monti::vulkan::UploadAndRegisterMeshes(
        *renderer, ctx.Allocator(), ctx.Device(), upload_cmd, load_result.mesh_data, procs);
    ctx.SubmitAndWait(upload_cmd);

    if (gpu_buffers.empty()) {
        std::fprintf(stderr, "Failed to upload mesh data\n");
        return EXIT_FAILURE;
    }
    auto t_mesh_end = Clock::now();

    // ── Create G-buffer images (with TRANSFER_SRC for readback) ──
    auto t_gbuffer_start = Clock::now();
    monti::app::GBufferImages gbuffer_images;
    VkCommandBuffer gbuf_cmd = ctx.BeginOneShot();
    if (!gbuffer_images.Create(ctx.Allocator(), ctx.Device(),
                               width, height, gbuf_cmd,
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        std::fprintf(stderr, "Failed to create G-buffer images\n");
        return EXIT_FAILURE;
    }
    ctx.SubmitAndWait(gbuf_cmd);
    auto t_gbuffer_end = Clock::now();

    // ── Create capture writer (native scale — single resolution) ──
    auto exr_compression = (compression_str == "zip")
        ? monti::capture::ExrCompression::kZip
        : monti::capture::ExrCompression::kNone;

    monti::capture::WriterDesc writer_desc{};
    writer_desc.output_dir = output_dir;
    writer_desc.input_width = width;
    writer_desc.input_height = height;
    writer_desc.scale_mode = monti::capture::ScaleMode::kNative;
    writer_desc.compression = exr_compression;

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
    gen_config.output_dir = output_dir;
    gen_config.capture_shader_dir = CAPTURE_SHADER_SPV_DIR;
    // Extract scene filename stem for skip reports
    auto slash_pos = scene_path.find_last_of("/\\");
    auto stem_start = (slash_pos == std::string::npos) ? 0 : slash_pos + 1;
    auto dot_pos = scene_path.rfind('.');
    gen_config.scene_name = scene_path.substr(
        stem_start, (dot_pos > stem_start) ? dot_pos - stem_start : std::string::npos);
    gen_config.skipped_path = skipped_path;
    gen_config.viewpoints = std::move(viewpoints);

    monti::app::datagen::GenerationSession session(ctx, *renderer, gbuffer_images,
                                                   *writer, scene, gen_config);
    if (!session.Run()) {
        std::fprintf(stderr, "Generation failed\n");
        return EXIT_FAILURE;
    }

    auto program_end = Clock::now();

    // ── Print setup timing summary ──
    double vulkan_init_ms = ElapsedMs(t_vulkan_start, t_vulkan_end);
    double scene_load_ms = ElapsedMs(t_scene_start, t_scene_end);
    double env_load_ms = ElapsedMs(t_env_start, t_env_end);
    double renderer_create_ms = ElapsedMs(t_renderer_start, t_renderer_end);
    double mesh_upload_ms = ElapsedMs(t_mesh_start, t_mesh_end);
    double gbuffer_create_ms = ElapsedMs(t_gbuffer_start, t_gbuffer_end);
    double setup_total_ms = vulkan_init_ms + scene_load_ms + env_load_ms
                          + renderer_create_ms + mesh_upload_ms + gbuffer_create_ms;
    double total_ms = ElapsedMs(program_start, program_end);

    std::printf("\nSetup timing:\n");
    std::printf("  Vulkan init:      %.1fms\n", vulkan_init_ms);
    std::printf("  Scene load:       %.1fms\n", scene_load_ms);
    std::printf("  Environment:      %.1fms\n", env_load_ms);
    std::printf("  Renderer create:  %.1fms\n", renderer_create_ms);
    std::printf("  Mesh upload:      %.1fms\n", mesh_upload_ms);
    std::printf("  G-buffer create:  %.1fms\n", gbuffer_create_ms);
    std::printf("  Setup total:      %.1fms\n", setup_total_ms);

    // ── Build and write timing.json ──
    const auto& vp_timings = session.ViewpointTimings();
    auto num_vp = static_cast<uint32_t>(vp_timings.size());

    double avg_viewpoint_ms = 0.0;
    double avg_render_ref_ms = 0.0;
    double avg_write_exr_ms = 0.0;
    for (const auto& vpt : vp_timings) {
        avg_viewpoint_ms += vpt["total_ms"].get<double>();
        avg_render_ref_ms += vpt["render_reference_ms"].get<double>();
        avg_write_exr_ms += vpt["write_exr_ms"].get<double>();
    }
    if (num_vp > 0) {
        avg_viewpoint_ms /= num_vp;
        avg_render_ref_ms /= num_vp;
        avg_write_exr_ms /= num_vp;
    }

    nlohmann::json timing_json = {
        {"version", 1},
        {"device", props.deviceName},
        {"resolution", {width, height}},
        {"spp", spp},
        {"ref_frames", ref_frames},
        {"exr_compression", compression_str},
        {"setup", {
            {"vulkan_init_ms", vulkan_init_ms},
            {"scene_load_ms", scene_load_ms},
            {"env_load_ms", env_load_ms},
            {"renderer_create_ms", renderer_create_ms},
            {"mesh_upload_ms", mesh_upload_ms},
            {"gbuffer_create_ms", gbuffer_create_ms},
        }},
        {"viewpoints", vp_timings},
        {"summary", {
            {"num_viewpoints", num_vp},
            {"total_ms", total_ms},
            {"setup_ms", setup_total_ms},
            {"avg_viewpoint_ms", avg_viewpoint_ms},
            {"avg_render_reference_ms", avg_render_ref_ms},
            {"avg_write_exr_ms", avg_write_exr_ms},
        }},
    };

    auto timing_path = output_dir + "/timing.json";
    std::ofstream timing_file(timing_path);
    if (timing_file) {
        timing_file << timing_json.dump(2) << "\n";
        std::printf("\nTiming written to %s\n", timing_path.c_str());
    } else {
        std::fprintf(stderr, "Warning: failed to write %s\n", timing_path.c_str());
    }

    // ── Write skipped viewpoints JSON (only when --skipped-path is set) ──
    const auto& skipped = session.SkippedViewpoints();
    if (!skipped_path.empty() && !skipped.empty()) {
        nlohmann::json skipped_arr = nlohmann::json::array();
        for (const auto& entry : skipped) {
            skipped_arr.push_back({
                {"viewpoint_id", entry.viewpoint_id},
                {"reason", entry.reason},
                {"detail", entry.detail},
            });
        }
        nlohmann::json skipped_json = {
            {"scene", gen_config.scene_name},
            {"skipped", skipped_arr},
        };
        std::ofstream skipped_file(skipped_path);
        if (skipped_file) {
            skipped_file << skipped_json.dump(2) << "\n";
            std::printf("Skipped viewpoints: %zu (written to %s)\n",
                        skipped.size(), skipped_path.c_str());
        } else {
            std::fprintf(stderr, "Warning: failed to write %s\n",
                         skipped_path.c_str());
        }
    } else if (!skipped.empty()) {
        std::printf("Skipped viewpoints: %zu (no --skipped-path set)\n",
                    skipped.size());
    }

    // ── Cleanup ──
    ctx.WaitIdle();
    for (auto& buf : gpu_buffers)
        monti::vulkan::DestroyGpuBuffer(ctx.Allocator(), buf);

    return EXIT_SUCCESS;
}
