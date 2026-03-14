#include <monti/vulkan/Renderer.h>

#include "vulkan/DeviceDispatch.h"
#include "vulkan/GpuScene.h"
#include "vulkan/GeometryManager.h"
#include "vulkan/BlueNoise.h"
#include "vulkan/EnvironmentMap.h"
#include "vulkan/RaytracePipeline.h"
#include "vulkan/Buffer.h"

#include <array>
#include <cstdio>
#include <memory>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace monti::vulkan {

namespace {

constexpr uint32_t kHaltonPeriod = 16;
constexpr uint32_t kDefaultMaxBounces = 4;

glm::vec2 HaltonJitter(uint32_t frame_index) {
    auto van_der_corput = [](uint32_t index, uint32_t base) -> float {
        float inv_base = 1.0f / static_cast<float>(base);
        float result = 0.0f;
        float fraction = inv_base;
        uint32_t n = index;
        while (n > 0) {
            result += static_cast<float>(n % base) * fraction;
            n /= base;
            fraction *= inv_base;
        }
        return result;
    };
    auto h = glm::vec2(
        van_der_corput((frame_index % kHaltonPeriod) + 1, 2),
        van_der_corput((frame_index % kHaltonPeriod) + 1, 3));
    return h - glm::vec2(0.5f);
}

}  // anonymous namespace

struct Renderer::Impl {
    RendererDesc desc{};
    std::unique_ptr<DeviceDispatch> dispatch;
    monti::Scene* scene = nullptr;
    std::unique_ptr<GpuScene> gpu_scene;
    std::unique_ptr<GeometryManager> geometry_manager;
    EnvironmentMap environment_map;
    BlueNoise blue_noise;
    RaytracePipeline raytrace_pipeline;
    MeshCleanupCallback mesh_cleanup_callback;
    std::vector<Buffer> pending_staging;  // kept alive until next frame
    uint32_t samples_per_pixel = 4;
    bool has_prev_view_proj_ = false;  // first-frame sentinel
    glm::mat4 prev_view_proj_ = glm::mat4(1.0f);  // cached for motion vectors
    bool scene_dirty = false;
    bool resources_initialized = false;
    bool pipeline_initialized = false;

    bool Init(const RendererDesc& d) {
        desc = d;
        samples_per_pixel = d.samples_per_pixel;

        if (!d.get_device_proc_addr || !d.get_instance_proc_addr) {
            std::fprintf(stderr, "Renderer::Create: get_device_proc_addr and "
                         "get_instance_proc_addr are required\n");
            return false;
        }

        dispatch = std::make_unique<DeviceDispatch>();
        if (!dispatch->Load(d.device, d.instance,
                            d.get_device_proc_addr, d.get_instance_proc_addr)) {
            std::fprintf(stderr, "Renderer::Create: failed to load Vulkan dispatch table\n");
            return false;
        }

        gpu_scene = std::make_unique<GpuScene>(d.allocator, d.device, d.physical_device,
                                                *dispatch);
        geometry_manager = std::make_unique<GeometryManager>(d.allocator, d.device,
                                                              *dispatch);
        return true;
    }
};

Renderer::Renderer() : impl_(std::make_unique<Impl>()) {}

std::unique_ptr<Renderer> Renderer::Create(const RendererDesc& desc) {
    auto renderer = std::unique_ptr<Renderer>(new Renderer());
    if (!renderer->impl_->Init(desc)) return nullptr;
    return renderer;
}

Renderer::~Renderer() = default;

void Renderer::SetScene(monti::Scene* scene) {
    impl_->scene = scene;
    impl_->scene_dirty = true;
}

void Renderer::RegisterMeshBuffers(MeshId mesh, const MeshBufferBinding& binding) {
    impl_->gpu_scene->RegisterMeshBuffers(mesh, binding);
}

void Renderer::NotifyMeshDeformed(MeshId mesh, bool topology_changed) {
    impl_->geometry_manager->NotifyMeshDeformed(mesh, topology_changed);
}

void Renderer::SetMeshCleanupCallback(MeshCleanupCallback callback) {
    impl_->mesh_cleanup_callback = std::move(callback);
}

bool Renderer::RenderFrame(VkCommandBuffer cmd, const GBuffer& output,
                           uint32_t frame_index) {
    if (!impl_->scene) return false;

    // Release staging buffers from the previous frame (cmd has completed by now)
    impl_->pending_staging.clear();

    // One-time initialization: placeholders + blue noise
    if (!impl_->resources_initialized) {
        if (!impl_->environment_map.CreatePlaceholders(
                impl_->desc.allocator, impl_->desc.device, cmd,
                impl_->pending_staging, *impl_->dispatch)) {
            std::fprintf(stderr, "Renderer::RenderFrame env map placeholder failed\n");
            return false;
        }

        Buffer blue_noise_staging;
        if (!impl_->blue_noise.Generate(impl_->desc.allocator, cmd, blue_noise_staging,
                                         *impl_->dispatch)) {
            std::fprintf(stderr, "Renderer::RenderFrame blue noise generation failed\n");
            return false;
        }
        impl_->pending_staging.push_back(std::move(blue_noise_staging));

        impl_->resources_initialized = true;
    }

    // On first frame after SetScene(), upload materials and textures
    if (impl_->scene_dirty) {
        // Upload textures (records commands into cmd)
        auto staging = impl_->gpu_scene->UploadTextures(*impl_->scene, cmd);
        for (auto& s : staging)
            impl_->pending_staging.push_back(std::move(s));

        // Update materials (host-visible buffer, no cmd needed)
        if (!impl_->gpu_scene->UpdateMaterials(*impl_->scene)) {
            std::fprintf(stderr, "Renderer::RenderFrame material update failed\n");
            return false;
        }

        // Upload mesh address table (host-visible buffer, no cmd needed)
        impl_->gpu_scene->UploadMeshAddressTable();

        // Update area lights
        if (!impl_->gpu_scene->UpdateAreaLights(*impl_->scene)) {
            std::fprintf(stderr, "Renderer::RenderFrame area light update failed\n");
            return false;
        }

        // Load environment map if one is set on the scene
        const auto* env_light = impl_->scene->GetEnvironmentLight();
        if (env_light) {
            const auto* tex = impl_->scene->GetTexture(env_light->hdr_lat_long);
            if (tex && !tex->data.empty()) {
                if (!impl_->environment_map.Load(
                        impl_->desc.allocator, impl_->desc.device, cmd,
                        reinterpret_cast<const float*>(tex->data.data()),
                        tex->width, tex->height,
                        impl_->pending_staging, *impl_->dispatch)) {
                    std::fprintf(stderr, "Renderer::RenderFrame env map load failed\n");
                    return false;
                }
            }
        }

        impl_->scene_dirty = false;
    }

    // Compact BLAS from previous frame (query results now available)
    if (!impl_->geometry_manager->CompactPendingBlas(cmd)) {
        std::fprintf(stderr, "Renderer::RenderFrame BLAS compaction failed\n");
        return false;
    }

    // Clean up BLAS for removed meshes
    impl_->geometry_manager->CleanupRemovedMeshes(*impl_->scene);

    // Build BLAS for new/dirty meshes
    if (!impl_->geometry_manager->BuildDirtyBlas(cmd, *impl_->gpu_scene)) {
        std::fprintf(stderr, "Renderer::RenderFrame BLAS build failed\n");
        return false;
    }

    // Build/rebuild TLAS
    if (!impl_->geometry_manager->BuildTlas(cmd, *impl_->scene, *impl_->gpu_scene)) {
        std::fprintf(stderr, "Renderer::RenderFrame TLAS build failed\n");
        return false;
    }

    // One-time RT pipeline creation (after all resources are available)
    if (!impl_->pipeline_initialized) {
        if (!impl_->raytrace_pipeline.Create(
                impl_->desc.device, impl_->desc.physical_device,
                impl_->desc.allocator, impl_->desc.pipeline_cache,
                impl_->desc.shader_dir, *impl_->dispatch)) {
            std::fprintf(stderr, "Renderer::RenderFrame RT pipeline creation failed\n");
            return false;
        }
        impl_->pipeline_initialized = true;
    }

    // Update descriptors with current resources
    if (impl_->pipeline_initialized &&
        impl_->geometry_manager->Tlas() != VK_NULL_HANDLE) {
        DescriptorUpdateInfo update{};
        update.tlas = impl_->geometry_manager->Tlas();
        update.gbuffer_views[0] = output.noisy_diffuse;
        update.gbuffer_views[1] = output.noisy_specular;
        update.gbuffer_views[2] = output.motion_vectors;
        update.gbuffer_views[3] = output.linear_depth;
        update.gbuffer_views[4] = output.world_normals;
        update.gbuffer_views[5] = output.diffuse_albedo;
        update.gbuffer_views[6] = output.specular_albedo;
        update.mesh_address_buffer = impl_->gpu_scene->MeshAddressBuffer();
        update.mesh_address_buffer_size = impl_->gpu_scene->MeshAddressBufferSize();
        update.material_buffer = impl_->gpu_scene->MaterialBuffer();
        update.material_buffer_size = impl_->gpu_scene->MaterialBufferSize();
        update.gpu_scene = impl_->gpu_scene.get();
        update.area_light_buffer = impl_->gpu_scene->AreaLightBuffer();
        update.area_light_buffer_size = impl_->gpu_scene->AreaLightBufferSize();
        update.blue_noise_buffer = impl_->blue_noise.TableBuffer().Handle();
        update.blue_noise_buffer_size = impl_->blue_noise.BufferSize();
        update.environment_map = &impl_->environment_map;
        impl_->raytrace_pipeline.UpdateDescriptors(update);

        // ── Populate push constants from scene state ─────────────────
        const auto& camera = impl_->scene->GetActiveCamera();
        float aspect = static_cast<float>(impl_->desc.width) /
                        static_cast<float>(impl_->desc.height);

        auto view = camera.ViewMatrix();
        auto proj = camera.ProjectionMatrix(aspect);

        // Sub-pixel jitter via projection-matrix perturbation
        glm::vec2 jitter = HaltonJitter(frame_index);
        glm::mat4 jittered_proj = proj;
        jittered_proj[2][0] += jitter.x * 2.0f / static_cast<float>(impl_->desc.width);
        jittered_proj[2][1] += jitter.y * 2.0f / static_cast<float>(impl_->desc.height);

        PushConstants pc{};
        pc.inv_view = glm::inverse(view);
        pc.inv_proj = glm::inverse(jittered_proj);
        pc.prev_view_proj = impl_->prev_view_proj_;
        pc.frame_index = frame_index;
        pc.paths_per_pixel = impl_->samples_per_pixel;
        pc.max_bounces = kDefaultMaxBounces;
        pc.area_light_count = static_cast<uint32_t>(impl_->scene->AreaLights().size());
        pc.env_width = impl_->environment_map.Width();
        pc.env_height = impl_->environment_map.Height();
        pc.env_avg_luminance = impl_->environment_map.Statistics().average_luminance;
        pc.env_max_luminance = impl_->environment_map.Statistics().max_luminance;

        const auto* env_light = impl_->scene->GetEnvironmentLight();
        pc.env_rotation = env_light ? env_light->rotation : 0.0f;
        pc.skybox_mip_level = 0.0f;
        pc.jitter_x = jitter.x;
        pc.jitter_y = jitter.y;
        pc.debug_mode = 0;
        pc.pad0 = 0;

        // Cache non-jittered view-projection for next frame's motion vectors
        auto non_jittered_vp = proj * view;
        if (!impl_->has_prev_view_proj_) {
            pc.prev_view_proj = non_jittered_vp;  // No prior frame → zero motion
            impl_->has_prev_view_proj_ = true;
        }
        impl_->prev_view_proj_ = non_jittered_vp;

        // ── Transition G-buffer images: UNDEFINED → GENERAL ──────────
        std::array<VkImageMemoryBarrier2, 7> img_barriers{};
        std::array<VkImage, 7> gbuffer_images = {
            output.noisy_diffuse_image,
            output.noisy_specular_image,
            output.motion_vectors_image,
            output.linear_depth_image,
            output.world_normals_image,
            output.diffuse_albedo_image,
            output.specular_albedo_image,
        };

        for (uint32_t i = 0; i < 7; ++i) {
            auto& b = img_barriers[i];
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            b.srcAccessMask = 0;
            b.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = gbuffer_images[i];
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = static_cast<uint32_t>(img_barriers.size());
        dep.pImageMemoryBarriers = img_barriers.data();
        impl_->dispatch->vkCmdPipelineBarrier2(cmd, &dep);

        // ── Bind pipeline ────────────────────────────────────────────
        impl_->dispatch->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                          impl_->raytrace_pipeline.Pipeline());

        // ── Bind descriptor set ──────────────────────────────────────
        VkDescriptorSet ds = impl_->raytrace_pipeline.DescriptorSet();
        impl_->dispatch->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                impl_->raytrace_pipeline.PipelineLayout(),
                                0, 1, &ds, 0, nullptr);

        // ── Push constants ───────────────────────────────────────────
        impl_->dispatch->vkCmdPushConstants(cmd, impl_->raytrace_pipeline.PipelineLayout(),
                           VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                           VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                           VK_SHADER_STAGE_MISS_BIT_KHR |
                           VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                           0, sizeof(PushConstants), &pc);

        // ── Dispatch trace ───────────────────────────────────────────
        impl_->dispatch->vkCmdTraceRaysKHR(
            cmd,
            &impl_->raytrace_pipeline.RaygenRegion(),
            &impl_->raytrace_pipeline.MissRegion(),
            &impl_->raytrace_pipeline.HitRegion(),
            &impl_->raytrace_pipeline.CallableRegion(),
            impl_->desc.width, impl_->desc.height, 1);
    }

    return true;
}

void Renderer::SetSamplesPerPixel(uint32_t spp) {
    impl_->samples_per_pixel = spp;
}

uint32_t Renderer::GetSamplesPerPixel() const {
    return impl_->samples_per_pixel;
}

void Renderer::Resize(uint32_t /*width*/, uint32_t /*height*/) {
    // Will be implemented when render pipeline is built
}

float Renderer::LastFrameTimeMs() const {
    return 0.0f;
}

} // namespace monti::vulkan
