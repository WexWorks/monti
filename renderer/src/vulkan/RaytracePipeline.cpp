#include "RaytracePipeline.h"

#include "DeviceDispatch.h"
#include "GpuScene.h"
#include "EnvironmentMap.h"
#include "Image.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <string>

#include <monti/vulkan/ShaderFile.h>

namespace monti::vulkan {

namespace {

constexpr VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

}  // anonymous namespace

RaytracePipeline::~RaytracePipeline() {
    Destroy();
}

void RaytracePipeline::Destroy() {
    if (device_ == VK_NULL_HANDLE) return;

    if (pipeline_ != VK_NULL_HANDLE) {
        dispatch_->vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        dispatch_->vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (descriptor_pool_ != VK_NULL_HANDLE) {
        dispatch_->vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
        descriptor_set_ = VK_NULL_HANDLE;
    }
    if (descriptor_set_layout_ != VK_NULL_HANDLE) {
        dispatch_->vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
        descriptor_set_layout_ = VK_NULL_HANDLE;
    }

    sbt_buffer_.Destroy();
    raygen_region_ = {};
    miss_region_ = {};
    hit_region_ = {};
    callable_region_ = {};
}

bool RaytracePipeline::Create(VkDevice device, VkPhysicalDevice physical_device,
                         VmaAllocator allocator, VkPipelineCache pipeline_cache,
                         std::string_view shader_dir,
                         const DeviceDispatch& dispatch) {
    device_ = device;
    allocator_ = allocator;
    dispatch_ = &dispatch;

    if (!CreateDescriptorSetLayout()) return false;
    if (!CreateDescriptorPool()) return false;
    if (!CreatePipelineAndLayout(pipeline_cache, shader_dir)) return false;
    if (!CreateSbt(physical_device)) return false;

    return true;
}

// ── Descriptor Set Layout ────────────────────────────────────────

bool RaytracePipeline::CreateDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 18> bindings{};

    // Binding 0: TLAS
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    // Bindings 1–7: G-buffer storage images (raygen only)
    for (uint32_t i = 1; i <= 7; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    }

    // Binding 8: Mesh address table (storage buffer) — closest hit + any hit
    bindings[8].binding = 8;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[8].descriptorCount = 1;
    bindings[8].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                             VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

    // Binding 9: Material buffer — raygen + closest hit + any hit
    bindings[9].binding = 9;
    bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[9].descriptorCount = 1;
    bindings[9].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                             VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                             VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

    // Binding 10: Bindless texture array — raygen + closest hit + any hit
    // (update-after-bind, partially bound, variable count)
    bindings[10].binding = 10;
    bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[10].descriptorCount = kMaxBindlessTextures;
    bindings[10].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                              VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                              VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

    // Binding 11: Light buffer — raygen
    bindings[11].binding = 11;
    bindings[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[11].descriptorCount = 1;
    bindings[11].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    // Binding 12: Blue noise table — raygen + closest hit
    bindings[12].binding = 12;
    bindings[12].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[12].descriptorCount = 1;
    bindings[12].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                              VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    // Binding 13: Environment map — raygen + closest hit + miss
    bindings[13].binding = 13;
    bindings[13].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[13].descriptorCount = 1;
    bindings[13].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                              VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                              VK_SHADER_STAGE_MISS_BIT_KHR;

    // Binding 14: Marginal CDF — raygen + closest hit
    bindings[14].binding = 14;
    bindings[14].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[14].descriptorCount = 1;
    bindings[14].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                              VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    // Binding 15: Conditional CDF — raygen + closest hit
    bindings[15].binding = 15;
    bindings[15].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[15].descriptorCount = 1;
    bindings[15].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                              VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    // Binding 16: Frame uniforms UBO — raygen only
    bindings[16].binding = 16;
    bindings[16].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[16].descriptorCount = 1;
    bindings[16].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    // Binding 17: Convergence mask (R8UI storage image, readonly) — raygen only
    bindings[17].binding = 17;
    bindings[17].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[17].descriptorCount = 1;
    bindings[17].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    // Binding flags: update-after-bind on all bindings so the single descriptor
    // set can be updated while a previously-submitted command buffer is pending.
    // Binding 10 also gets PARTIALLY_BOUND for unused bindless texture slots.
    std::array<VkDescriptorBindingFlags, 18> binding_flags{};
    constexpr VkDescriptorBindingFlags kUpdateAfterBind =
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    for (auto& flags : binding_flags)
        flags = kUpdateAfterBind;
    binding_flags[10] |= VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

    VkDescriptorSetLayoutBindingFlagsCreateInfo flags_ci{};
    flags_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flags_ci.bindingCount = static_cast<uint32_t>(binding_flags.size());
    flags_ci.pBindingFlags = binding_flags.data();

    VkDescriptorSetLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_ci.pNext = &flags_ci;
    layout_ci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layout_ci.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_ci.pBindings = bindings.data();

    VkResult result = dispatch_->vkCreateDescriptorSetLayout(device_, &layout_ci, nullptr,
                                                  &descriptor_set_layout_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "RaytracePipeline: descriptor set layout creation failed (VkResult: %d)\n",
                     result);
        return false;
    }
    return true;
}

// ── Descriptor Pool + Set Allocation ─────────────────────────────

bool RaytracePipeline::CreateDescriptorPool() {
    std::array<VkDescriptorPoolSize, 5> pool_sizes{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    pool_sizes[0].descriptorCount = 1;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[1].descriptorCount = 8;  // bindings 1–7, 17
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[2].descriptorCount = 4;  // bindings 8, 9, 11, 12
    pool_sizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[3].descriptorCount = kMaxBindlessTextures + 3;  // bindings 10, 13, 14, 15
    pool_sizes[4].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[4].descriptorCount = 1;  // binding 16

    VkDescriptorPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    pool_ci.maxSets = 1;
    pool_ci.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_ci.pPoolSizes = pool_sizes.data();

    VkResult result = dispatch_->vkCreateDescriptorPool(device_, &pool_ci, nullptr, &descriptor_pool_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "RaytracePipeline: descriptor pool creation failed (VkResult: %d)\n",
                     result);
        return false;
    }

    // Allocate descriptor set (kMaxBindlessTextures for binding 10)
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &descriptor_set_layout_;

    result = dispatch_->vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "RaytracePipeline: descriptor set allocation failed (VkResult: %d)\n",
                     result);
        return false;
    }

    return true;
}

// ── Pipeline + Layout ────────────────────────────────────────────

bool RaytracePipeline::CreatePipelineAndLayout(VkPipelineCache pipeline_cache,
                                          std::string_view shader_dir) {
    std::string dir(shader_dir);
    auto raygen_code = LoadShaderFile(dir + "/raygen.rgen.spv");
    auto miss_code = LoadShaderFile(dir + "/miss.rmiss.spv");
    auto chit_code = LoadShaderFile(dir + "/closesthit.rchit.spv");
    auto ahit_code = LoadShaderFile(dir + "/anyhit.rahit.spv");
    if (raygen_code.empty() || miss_code.empty() || chit_code.empty() ||
        ahit_code.empty())
        return false;

    auto create_module = [&](const std::vector<uint8_t>& code) -> VkShaderModule {
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = code.size();
        ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule module = VK_NULL_HANDLE;
        dispatch_->vkCreateShaderModule(device_, &ci, nullptr, &module);
        return module;
    };

    VkShaderModule raygen_module = create_module(raygen_code);
    VkShaderModule miss_module = create_module(miss_code);
    VkShaderModule chit_module = create_module(chit_code);
    VkShaderModule ahit_module = create_module(ahit_code);

    if (!raygen_module || !miss_module || !chit_module || !ahit_module) {
        std::fprintf(stderr, "RaytracePipeline: failed to create shader modules\n");
        if (raygen_module) dispatch_->vkDestroyShaderModule(device_, raygen_module, nullptr);
        if (miss_module) dispatch_->vkDestroyShaderModule(device_, miss_module, nullptr);
        if (chit_module) dispatch_->vkDestroyShaderModule(device_, chit_module, nullptr);
        if (ahit_module) dispatch_->vkDestroyShaderModule(device_, ahit_module, nullptr);
        return false;
    }

    // Shader stages: raygen (0), miss (1), closest hit (2), any hit (3)
    std::array<VkPipelineShaderStageCreateInfo, 4> stages{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = raygen_module;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[1].module = miss_module;
    stages[1].pName = "main";

    stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[2].module = chit_module;
    stages[2].pName = "main";

    stages[3].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[3].stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stages[3].module = ahit_module;
    stages[3].pName = "main";

    // Shader groups
    std::array<VkRayTracingShaderGroupCreateInfoKHR, 3> groups{};

    // Group 0: raygen
    groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Group 1: miss
    groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;
    groups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Group 2: hit group (closest hit + any hit)
    groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[2].generalShader = VK_SHADER_UNUSED_KHR;
    groups[2].closestHitShader = 2;
    groups[2].anyHitShader = 3;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Push constant range — only raygen reads push constants;
    // other stages use descriptors.
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    push_range.offset = 0;
    push_range.size = sizeof(PushConstants);

    // Pipeline layout
    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &descriptor_set_layout_;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &push_range;

    VkResult result = dispatch_->vkCreatePipelineLayout(device_, &layout_ci, nullptr, &pipeline_layout_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "RaytracePipeline: pipeline layout creation failed (VkResult: %d)\n",
                     result);
        dispatch_->vkDestroyShaderModule(device_, raygen_module, nullptr);
        dispatch_->vkDestroyShaderModule(device_, miss_module, nullptr);
        dispatch_->vkDestroyShaderModule(device_, chit_module, nullptr);
        dispatch_->vkDestroyShaderModule(device_, ahit_module, nullptr);
        return false;
    }

    // Ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipeline_ci.stageCount = static_cast<uint32_t>(stages.size());
    pipeline_ci.pStages = stages.data();
    pipeline_ci.groupCount = static_cast<uint32_t>(groups.size());
    pipeline_ci.pGroups = groups.data();
    pipeline_ci.maxPipelineRayRecursionDepth = kMaxRayRecursionDepth;
    pipeline_ci.layout = pipeline_layout_;

    result = dispatch_->vkCreateRayTracingPipelinesKHR(device_, VK_NULL_HANDLE, pipeline_cache,
                                            1, &pipeline_ci, nullptr, &pipeline_);

    // Destroy shader modules (no longer needed after pipeline creation)
    dispatch_->vkDestroyShaderModule(device_, raygen_module, nullptr);
    dispatch_->vkDestroyShaderModule(device_, miss_module, nullptr);
    dispatch_->vkDestroyShaderModule(device_, chit_module, nullptr);
    dispatch_->vkDestroyShaderModule(device_, ahit_module, nullptr);

    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "RaytracePipeline: pipeline creation failed (VkResult: %d)\n", result);
        return false;
    }

    return true;
}

// ── Shader Binding Table ─────────────────────────────────────────

bool RaytracePipeline::CreateSbt(VkPhysicalDevice physical_device) {
    // Query RT pipeline properties
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_props{};
    rt_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &rt_props;
    dispatch_->vkGetPhysicalDeviceProperties2(physical_device, &props2);

    uint32_t handle_size = rt_props.shaderGroupHandleSize;
    uint32_t handle_alignment = rt_props.shaderGroupHandleAlignment;
    uint32_t base_alignment = rt_props.shaderGroupBaseAlignment;

    VkDeviceSize aligned_handle_size = AlignUp(handle_size, handle_alignment);

    // Region sizes (aligned to base alignment)
    VkDeviceSize raygen_size = AlignUp(aligned_handle_size, base_alignment);
    VkDeviceSize miss_size = AlignUp(aligned_handle_size, base_alignment);
    VkDeviceSize hit_size = AlignUp(aligned_handle_size, base_alignment);
    VkDeviceSize sbt_total = raygen_size + miss_size + hit_size;

    // Retrieve shader group handles
    constexpr uint32_t kGroupCount = 3;
    std::vector<uint8_t> handles(static_cast<size_t>(handle_size) * kGroupCount);
    VkResult result = dispatch_->vkGetRayTracingShaderGroupHandlesKHR(
        device_, pipeline_, 0, kGroupCount,
        handles.size(), handles.data());
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "RaytracePipeline: failed to get shader group handles (VkResult: %d)\n",
                     result);
        return false;
    }

    // Allocate SBT buffer
    if (!sbt_buffer_.Create(allocator_, sbt_total,
                            VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                            VMA_MEMORY_USAGE_CPU_TO_GPU)) {
        std::fprintf(stderr, "RaytracePipeline: SBT buffer allocation failed\n");
        return false;
    }

    // Map and copy handles at aligned offsets
    auto* mapped = static_cast<uint8_t*>(sbt_buffer_.Map());
    if (!mapped) {
        std::fprintf(stderr, "RaytracePipeline: SBT buffer map failed\n");
        return false;
    }
    std::memset(mapped, 0, static_cast<size_t>(sbt_total));
    std::memcpy(mapped, handles.data(), handle_size);                          // raygen
    std::memcpy(mapped + raygen_size, handles.data() + handle_size, handle_size);  // miss
    std::memcpy(mapped + raygen_size + miss_size,
                handles.data() + 2 * handle_size, handle_size);               // hit
    sbt_buffer_.Unmap();

    // Set up strided address regions
    VkDeviceAddress sbt_address = sbt_buffer_.DeviceAddress(device_, *dispatch_);

    raygen_region_.deviceAddress = sbt_address;
    raygen_region_.stride = aligned_handle_size;
    raygen_region_.size = aligned_handle_size;  // Vulkan spec: raygen size must equal stride

    miss_region_.deviceAddress = sbt_address + raygen_size;
    miss_region_.stride = aligned_handle_size;
    miss_region_.size = miss_size;

    hit_region_.deviceAddress = sbt_address + raygen_size + miss_size;
    hit_region_.stride = aligned_handle_size;
    hit_region_.size = hit_size;

    callable_region_ = {};  // empty — no callable shaders

    return true;
}

// ── Descriptor Update ────────────────────────────────────────────

void RaytracePipeline::UpdateDescriptors(const DescriptorUpdateInfo& info) {
    // TLAS (binding 0)
    VkWriteDescriptorSetAccelerationStructureKHR tlas_write{};
    tlas_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    tlas_write.accelerationStructureCount = 1;
    tlas_write.pAccelerationStructures = &info.tlas;

    // G-buffer storage images (bindings 1–7)
    std::array<VkDescriptorImageInfo, 7> gbuffer_infos{};
    for (uint32_t i = 0; i < 7; ++i) {
        gbuffer_infos[i].imageView = info.gbuffer_views[i];
        gbuffer_infos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    // Mesh address table (binding 8)
    VkDescriptorBufferInfo mesh_addr_info{};
    mesh_addr_info.buffer = info.mesh_address_buffer;
    mesh_addr_info.offset = 0;
    mesh_addr_info.range = info.mesh_address_buffer_size;

    // Material buffer (binding 9)
    VkDescriptorBufferInfo material_info{};
    material_info.buffer = info.material_buffer;
    material_info.offset = 0;
    material_info.range = info.material_buffer_size;

    // Bindless textures (binding 10)
    uint32_t texture_count = info.gpu_scene ? info.gpu_scene->TextureCount() : 0;
    std::vector<VkDescriptorImageInfo> tex_infos;
    if (texture_count > 0) {
        const auto& tex_images = info.gpu_scene->TextureImages();
        tex_infos.resize(texture_count);
        for (uint32_t i = 0; i < texture_count; ++i) {
            tex_infos[i].sampler = tex_images[i].Sampler();
            tex_infos[i].imageView = tex_images[i].View();
            tex_infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
    }

    // Light buffer (binding 11)
    VkDescriptorBufferInfo light_info{};
    light_info.buffer = info.light_buffer;
    light_info.offset = 0;
    light_info.range = info.light_buffer_size;

    // Blue noise table (binding 12)
    VkDescriptorBufferInfo blue_noise_info{};
    blue_noise_info.buffer = info.blue_noise_buffer;
    blue_noise_info.offset = 0;
    blue_noise_info.range = info.blue_noise_buffer_size;

    // Environment map (binding 13)
    VkDescriptorImageInfo env_info{};
    env_info.sampler = info.environment_map->EnvTexture().Sampler();
    env_info.imageView = info.environment_map->EnvTexture().View();
    env_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Marginal CDF (binding 14)
    VkDescriptorImageInfo marginal_cdf_info{};
    marginal_cdf_info.sampler = info.environment_map->MarginalCdfTexture().Sampler();
    marginal_cdf_info.imageView = info.environment_map->MarginalCdfTexture().View();
    marginal_cdf_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Conditional CDF (binding 15)
    VkDescriptorImageInfo conditional_cdf_info{};
    conditional_cdf_info.sampler = info.environment_map->ConditionalCdfTexture().Sampler();
    conditional_cdf_info.imageView = info.environment_map->ConditionalCdfTexture().View();
    conditional_cdf_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Build write descriptors — up to 16 base writes + optional textures
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(17);

    auto add_write = [&](uint32_t binding, VkDescriptorType type, uint32_t count,
                         const VkDescriptorBufferInfo* buffer, const VkDescriptorImageInfo* image,
                         const void* pnext = nullptr) {
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.pNext = pnext;
        w.dstSet = descriptor_set_;
        w.dstBinding = binding;
        w.descriptorCount = count;
        w.descriptorType = type;
        w.pBufferInfo = buffer;
        w.pImageInfo = image;
        writes.push_back(w);
    };

    // Binding 0: TLAS
    add_write(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
              nullptr, nullptr, &tlas_write);

    // Bindings 1–7: G-buffer images
    for (uint32_t i = 0; i < 7; ++i) {
        add_write(1 + i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                  nullptr, &gbuffer_infos[i]);
    }

    // Binding 8: Mesh address table
    add_write(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &mesh_addr_info, nullptr);

    // Binding 9: Material buffer
    add_write(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &material_info, nullptr);

    // Binding 10: Bindless textures (only if textures exist)
    if (texture_count > 0) {
        add_write(10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texture_count,
                  nullptr, tex_infos.data());
    }

    // Binding 11: Light buffer
    add_write(11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &light_info, nullptr);

    // Binding 12: Blue noise table
    add_write(12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &blue_noise_info, nullptr);

    // Binding 13: Environment map
    add_write(13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, nullptr, &env_info);

    // Binding 14: Marginal CDF
    add_write(14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, nullptr, &marginal_cdf_info);

    // Binding 15: Conditional CDF
    add_write(15, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, nullptr, &conditional_cdf_info);

    // Binding 16: Frame uniforms UBO
    VkDescriptorBufferInfo frame_ubo_info{};
    frame_ubo_info.buffer = info.frame_uniforms_buffer;
    frame_ubo_info.offset = 0;
    frame_ubo_info.range = info.frame_uniforms_buffer_size;
    add_write(16, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, &frame_ubo_info, nullptr);

    // Binding 17: Convergence mask (R8UI storage image)
    VkDescriptorImageInfo convergence_mask_info{};
    convergence_mask_info.imageView = info.convergence_mask_view;
    convergence_mask_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    add_write(17, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, nullptr, &convergence_mask_info);

    dispatch_->vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
}

}  // namespace monti::vulkan
