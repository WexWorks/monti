#include <monti/capture/GpuAccumulator.h>

#include <monti/vulkan/ShaderFile.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <string>

namespace monti::capture {

namespace {
constexpr uint32_t kWorkgroupSize = 16;
}  // namespace

GpuAccumulator::~GpuAccumulator() {
    if (device_ == VK_NULL_HANDLE) return;

    DestroyAccumulationImages();

    if (accumulate_pipeline_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyPipeline(device_, accumulate_pipeline_, nullptr);
    if (accumulate_layout_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyPipelineLayout(device_, accumulate_layout_, nullptr);
    if (accumulate_desc_pool_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyDescriptorPool(device_, accumulate_desc_pool_, nullptr);
    if (accumulate_desc_layout_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyDescriptorSetLayout(device_, accumulate_desc_layout_, nullptr);

    if (finalize_pipeline_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyPipeline(device_, finalize_pipeline_, nullptr);
    if (finalize_layout_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyPipelineLayout(device_, finalize_layout_, nullptr);
    if (finalize_desc_pool_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyDescriptorPool(device_, finalize_desc_pool_, nullptr);
    if (finalize_desc_layout_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyDescriptorSetLayout(device_, finalize_desc_layout_, nullptr);

    if (noisy_diffuse_view_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyImageView(device_, noisy_diffuse_view_, nullptr);
    if (noisy_specular_view_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyImageView(device_, noisy_specular_view_, nullptr);

    // Adaptive sampling resources
    if (variance_update_pipeline_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyPipeline(device_, variance_update_pipeline_, nullptr);
    if (variance_update_layout_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyPipelineLayout(device_, variance_update_layout_, nullptr);
    if (variance_update_desc_pool_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyDescriptorPool(device_, variance_update_desc_pool_, nullptr);
    if (variance_update_desc_layout_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyDescriptorSetLayout(device_, variance_update_desc_layout_, nullptr);

    if (convergence_check_pipeline_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyPipeline(device_, convergence_check_pipeline_, nullptr);
    if (convergence_check_layout_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyPipelineLayout(device_, convergence_check_layout_, nullptr);
    if (convergence_check_desc_pool_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyDescriptorPool(device_, convergence_check_desc_pool_, nullptr);
    if (convergence_check_desc_layout_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyDescriptorSetLayout(device_, convergence_check_desc_layout_, nullptr);

    if (converged_count_buffer_ != VK_NULL_HANDLE)
        vmaDestroyBuffer(allocator_, converged_count_buffer_, converged_count_alloc_);
}

std::unique_ptr<GpuAccumulator> GpuAccumulator::Create(const GpuAccumulatorDesc& desc) {
    auto acc = std::unique_ptr<GpuAccumulator>(new GpuAccumulator());
    if (!acc->Init(desc)) return nullptr;
    return acc;
}

bool GpuAccumulator::Init(const GpuAccumulatorDesc& desc) {
    device_ = desc.device;
    allocator_ = desc.allocator;
    width_ = desc.width;
    height_ = desc.height;
    procs_ = desc.procs;
    adaptive_enabled_ = desc.adaptive_sampling;

    if (!CreateAccumulationImages()) return false;
    if (!CreateSampleCountImage()) return false;

    if (adaptive_enabled_) {
        if (!CreateVarianceImages()) return false;
        if (!CreateConvergenceMaskImage()) return false;
        if (!CreateConvergedCounterBuffer()) return false;
    }

    if (!CreateImageViews(desc.noisy_diffuse, desc.noisy_specular)) return false;
    if (!CreateDescriptorResources()) return false;
    if (!CreateAccumulatePipeline(desc.shader_dir)) return false;
    if (!CreateFinalizePipeline(desc.shader_dir)) return false;

    if (adaptive_enabled_) {
        if (!CreateVarianceDescriptorResources()) return false;
        if (!CreateVarianceUpdatePipeline(desc.shader_dir)) return false;
        if (!CreateConvergenceCheckPipeline(desc.shader_dir)) return false;
    }

    return true;
}

bool GpuAccumulator::CreateAccumulationImages() {
    VkImageCreateInfo image_ci{};
    image_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_ci.imageType = VK_IMAGE_TYPE_2D;
    image_ci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    image_ci.extent = {width_, height_, 1};
    image_ci.mipLevels = 1;
    image_ci.arrayLayers = 1;
    image_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                   | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult result = vmaCreateImage(allocator_, &image_ci, &alloc_ci,
                                     &accum_diffuse_, &accum_diffuse_alloc_, nullptr);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: diffuse accumulation image creation failed (%d)\n",
                     result);
        return false;
    }

    result = vmaCreateImage(allocator_, &image_ci, &alloc_ci,
                            &accum_specular_, &accum_specular_alloc_, nullptr);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: specular accumulation image creation failed (%d)\n",
                     result);
        return false;
    }

    return true;
}

bool GpuAccumulator::CreateSampleCountImage() {
    VkImageCreateInfo image_ci{};
    image_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_ci.imageType = VK_IMAGE_TYPE_2D;
    image_ci.format = VK_FORMAT_R32_UINT;
    image_ci.extent = {width_, height_, 1};
    image_ci.mipLevels = 1;
    image_ci.arrayLayers = 1;
    image_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult result = vmaCreateImage(allocator_, &image_ci, &alloc_ci,
                                     &sample_count_, &sample_count_alloc_, nullptr);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: sample count image creation failed (%d)\n", result);
        return false;
    }

    return true;
}

bool GpuAccumulator::CreateImageViews(VkImage noisy_diffuse, VkImage noisy_specular) {
    auto make_view = [&](VkImage image, VkFormat format) -> VkImageView {
        VkImageViewCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image = image;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = format;
        ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView view = VK_NULL_HANDLE;
        VkResult r = procs_.pfn_vkCreateImageView(device_, &ci, nullptr, &view);
        if (r != VK_SUCCESS)
            std::fprintf(stderr, "GpuAccumulator: image view creation failed (%d)\n", r);
        return view;
    };

    noisy_diffuse_view_ = make_view(noisy_diffuse, VK_FORMAT_R16G16B16A16_SFLOAT);
    if (noisy_diffuse_view_ == VK_NULL_HANDLE) return false;

    noisy_specular_view_ = make_view(noisy_specular, VK_FORMAT_R16G16B16A16_SFLOAT);
    if (noisy_specular_view_ == VK_NULL_HANDLE) return false;

    accum_diffuse_view_ = make_view(accum_diffuse_, VK_FORMAT_R32G32B32A32_SFLOAT);
    if (accum_diffuse_view_ == VK_NULL_HANDLE) return false;

    accum_specular_view_ = make_view(accum_specular_, VK_FORMAT_R32G32B32A32_SFLOAT);
    if (accum_specular_view_ == VK_NULL_HANDLE) return false;

    sample_count_view_ = make_view(sample_count_, VK_FORMAT_R32_UINT);
    if (sample_count_view_ == VK_NULL_HANDLE) return false;

    if (adaptive_enabled_) {
        variance_mean_view_ = make_view(variance_mean_, VK_FORMAT_R32_SFLOAT);
        if (variance_mean_view_ == VK_NULL_HANDLE) return false;

        variance_m2_view_ = make_view(variance_m2_, VK_FORMAT_R32_SFLOAT);
        if (variance_m2_view_ == VK_NULL_HANDLE) return false;

        convergence_mask_view_ = make_view(convergence_mask_, VK_FORMAT_R8_UINT);
        if (convergence_mask_view_ == VK_NULL_HANDLE) return false;
    } else {
        // Create a tiny 1×1 R8UI dummy convergence mask for binding 5 of accumulate.comp
        if (!CreateConvergenceMaskImage()) return false;
        convergence_mask_view_ = make_view(convergence_mask_, VK_FORMAT_R8_UINT);
        if (convergence_mask_view_ == VK_NULL_HANDLE) return false;
    }

    return true;
}

bool GpuAccumulator::CreateDescriptorResources() {
    // ── Accumulate descriptor set: 6 storage image bindings ──
    // 0-1: source (readonly), 2-3: accumulator (read-write), 4: sample_count (read-write),
    // 5: convergence_mask (readonly)
    std::array<VkDescriptorSetLayoutBinding, 6> acc_bindings{};
    for (uint32_t i = 0; i < 6; ++i) {
        acc_bindings[i].binding = i;
        acc_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        acc_bindings[i].descriptorCount = 1;
        acc_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_ci.bindingCount = static_cast<uint32_t>(acc_bindings.size());
    layout_ci.pBindings = acc_bindings.data();

    VkResult result = procs_.pfn_vkCreateDescriptorSetLayout(device_, &layout_ci, nullptr,
                                                             &accumulate_desc_layout_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: accumulate descriptor layout creation failed (%d)\n",
                     result);
        return false;
    }

    VkDescriptorPoolSize acc_pool_size{};
    acc_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    acc_pool_size.descriptorCount = 6;

    VkDescriptorPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets = 1;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes = &acc_pool_size;

    result = procs_.pfn_vkCreateDescriptorPool(device_, &pool_ci, nullptr,
                                               &accumulate_desc_pool_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: accumulate descriptor pool creation failed (%d)\n",
                     result);
        return false;
    }

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = accumulate_desc_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &accumulate_desc_layout_;

    result = procs_.pfn_vkAllocateDescriptorSets(device_, &alloc_info, &accumulate_desc_set_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: accumulate descriptor set allocation failed (%d)\n",
                     result);
        return false;
    }

    // Write accumulate descriptor set
    std::array<VkDescriptorImageInfo, 6> acc_image_infos{};
    acc_image_infos[0].imageView = noisy_diffuse_view_;
    acc_image_infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    acc_image_infos[1].imageView = noisy_specular_view_;
    acc_image_infos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    acc_image_infos[2].imageView = accum_diffuse_view_;
    acc_image_infos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    acc_image_infos[3].imageView = accum_specular_view_;
    acc_image_infos[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    acc_image_infos[4].imageView = sample_count_view_;
    acc_image_infos[4].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    acc_image_infos[5].imageView = convergence_mask_view_;
    acc_image_infos[5].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 6> acc_writes{};
    for (uint32_t i = 0; i < 6; ++i) {
        acc_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        acc_writes[i].dstSet = accumulate_desc_set_;
        acc_writes[i].dstBinding = i;
        acc_writes[i].descriptorCount = 1;
        acc_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        acc_writes[i].pImageInfo = &acc_image_infos[i];
    }

    procs_.pfn_vkUpdateDescriptorSets(device_, static_cast<uint32_t>(acc_writes.size()),
                                      acc_writes.data(), 0, nullptr);

    // ── Finalize descriptor set: 3 storage image bindings ──
    // 0: accum_diffuse (read-write), 1: accum_specular (read-write), 2: sample_count (readonly)
    std::array<VkDescriptorSetLayoutBinding, 3> fin_bindings{};
    for (uint32_t i = 0; i < 3; ++i) {
        fin_bindings[i].binding = i;
        fin_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        fin_bindings[i].descriptorCount = 1;
        fin_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo fin_layout_ci{};
    fin_layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    fin_layout_ci.bindingCount = static_cast<uint32_t>(fin_bindings.size());
    fin_layout_ci.pBindings = fin_bindings.data();

    result = procs_.pfn_vkCreateDescriptorSetLayout(device_, &fin_layout_ci, nullptr,
                                                    &finalize_desc_layout_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: finalize descriptor layout creation failed (%d)\n",
                     result);
        return false;
    }

    VkDescriptorPoolSize fin_pool_size{};
    fin_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    fin_pool_size.descriptorCount = 3;

    VkDescriptorPoolCreateInfo fin_pool_ci{};
    fin_pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    fin_pool_ci.maxSets = 1;
    fin_pool_ci.poolSizeCount = 1;
    fin_pool_ci.pPoolSizes = &fin_pool_size;

    result = procs_.pfn_vkCreateDescriptorPool(device_, &fin_pool_ci, nullptr,
                                               &finalize_desc_pool_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: finalize descriptor pool creation failed (%d)\n",
                     result);
        return false;
    }

    VkDescriptorSetAllocateInfo fin_alloc_info{};
    fin_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    fin_alloc_info.descriptorPool = finalize_desc_pool_;
    fin_alloc_info.descriptorSetCount = 1;
    fin_alloc_info.pSetLayouts = &finalize_desc_layout_;

    result = procs_.pfn_vkAllocateDescriptorSets(device_, &fin_alloc_info, &finalize_desc_set_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: finalize descriptor set allocation failed (%d)\n",
                     result);
        return false;
    }

    // Write finalize descriptor set
    std::array<VkDescriptorImageInfo, 3> fin_image_infos{};
    fin_image_infos[0].imageView = accum_diffuse_view_;
    fin_image_infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    fin_image_infos[1].imageView = accum_specular_view_;
    fin_image_infos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    fin_image_infos[2].imageView = sample_count_view_;
    fin_image_infos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 3> fin_writes{};
    for (uint32_t i = 0; i < 3; ++i) {
        fin_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        fin_writes[i].dstSet = finalize_desc_set_;
        fin_writes[i].dstBinding = i;
        fin_writes[i].descriptorCount = 1;
        fin_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        fin_writes[i].pImageInfo = &fin_image_infos[i];
    }

    procs_.pfn_vkUpdateDescriptorSets(device_, static_cast<uint32_t>(fin_writes.size()),
                                      fin_writes.data(), 0, nullptr);
    return true;
}

bool GpuAccumulator::CreateAccumulatePipeline(std::string_view shader_dir) {
    std::string shader_path = std::string(shader_dir) + "/accumulate.comp.spv";
    auto shader_code = monti::vulkan::LoadShaderFile(shader_path);
    if (shader_code.empty()) return false;

    VkShaderModuleCreateInfo module_ci{};
    module_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_ci.codeSize = shader_code.size();
    module_ci.pCode = reinterpret_cast<const uint32_t*>(shader_code.data());

    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkResult result = procs_.pfn_vkCreateShaderModule(device_, &module_ci, nullptr, &shader_module);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: accumulate shader module creation failed (%d)\n",
                     result);
        return false;
    }

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &accumulate_desc_layout_;

    result = procs_.pfn_vkCreatePipelineLayout(device_, &layout_ci, nullptr, &accumulate_layout_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: accumulate pipeline layout creation failed (%d)\n",
                     result);
        procs_.pfn_vkDestroyShaderModule(device_, shader_module, nullptr);
        return false;
    }

    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_ci.stage.module = shader_module;
    pipeline_ci.stage.pName = "main";
    pipeline_ci.layout = accumulate_layout_;

    result = procs_.pfn_vkCreateComputePipelines(device_, VK_NULL_HANDLE,
                                                 1, &pipeline_ci, nullptr, &accumulate_pipeline_);
    procs_.pfn_vkDestroyShaderModule(device_, shader_module, nullptr);

    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: accumulate compute pipeline creation failed (%d)\n",
                     result);
        return false;
    }

    return true;
}

bool GpuAccumulator::CreateFinalizePipeline(std::string_view shader_dir) {
    std::string shader_path = std::string(shader_dir) + "/finalize.comp.spv";
    auto shader_code = monti::vulkan::LoadShaderFile(shader_path);
    if (shader_code.empty()) return false;

    VkShaderModuleCreateInfo module_ci{};
    module_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_ci.codeSize = shader_code.size();
    module_ci.pCode = reinterpret_cast<const uint32_t*>(shader_code.data());

    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkResult result = procs_.pfn_vkCreateShaderModule(device_, &module_ci, nullptr, &shader_module);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: finalize shader module creation failed (%d)\n",
                     result);
        return false;
    }

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &finalize_desc_layout_;

    result = procs_.pfn_vkCreatePipelineLayout(device_, &layout_ci, nullptr, &finalize_layout_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: finalize pipeline layout creation failed (%d)\n",
                     result);
        procs_.pfn_vkDestroyShaderModule(device_, shader_module, nullptr);
        return false;
    }

    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_ci.stage.module = shader_module;
    pipeline_ci.stage.pName = "main";
    pipeline_ci.layout = finalize_layout_;

    result = procs_.pfn_vkCreateComputePipelines(device_, VK_NULL_HANDLE,
                                                 1, &pipeline_ci, nullptr, &finalize_pipeline_);
    procs_.pfn_vkDestroyShaderModule(device_, shader_module, nullptr);

    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: finalize compute pipeline creation failed (%d)\n",
                     result);
        return false;
    }

    return true;
}

void GpuAccumulator::Reset(VkCommandBuffer cmd) {
    // Always clear 4 base images (accum diffuse/specular, sample_count,
    // convergence_mask).  accumulate.comp unconditionally reads the mask,
    // so it must be zero-initialized even when adaptive sampling is off.
    // When adaptive is on, also clear variance_mean and variance_m2.
    uint32_t image_count = 4;
    if (adaptive_enabled_) image_count = 6;

    // Transition all images to TRANSFER_DST for clear
    std::array<VkImageMemoryBarrier2, 6> to_dst{};
    for (uint32_t i = 0; i < image_count; ++i) {
        to_dst[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        to_dst[i].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        to_dst[i].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
                                | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        to_dst[i].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        to_dst[i].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        to_dst[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_dst[i].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_dst[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    to_dst[0].image = accum_diffuse_;
    to_dst[1].image = accum_specular_;
    to_dst[2].image = sample_count_;
    to_dst[3].image = convergence_mask_;
    if (adaptive_enabled_) {
        to_dst[4].image = variance_mean_;
        to_dst[5].image = variance_m2_;
    }

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = image_count;
    dep.pImageMemoryBarriers = to_dst.data();
    procs_.pfn_vkCmdPipelineBarrier2(cmd, &dep);

    VkClearColorValue clear_float{};
    VkClearColorValue clear_uint{};
    clear_uint.uint32[0] = 0;
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    procs_.pfn_vkCmdClearColorImage(cmd, accum_diffuse_,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    &clear_float, 1, &range);
    procs_.pfn_vkCmdClearColorImage(cmd, accum_specular_,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    &clear_float, 1, &range);
    procs_.pfn_vkCmdClearColorImage(cmd, sample_count_,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    &clear_uint, 1, &range);
    procs_.pfn_vkCmdClearColorImage(cmd, convergence_mask_,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    &clear_uint, 1, &range);

    if (adaptive_enabled_) {
        procs_.pfn_vkCmdClearColorImage(cmd, variance_mean_,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        &clear_float, 1, &range);
        procs_.pfn_vkCmdClearColorImage(cmd, variance_m2_,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        &clear_float, 1, &range);

        // Zero the atomic counter buffer
        procs_.pfn_vkCmdFillBuffer(cmd, converged_count_buffer_, 0, sizeof(uint32_t), 0);
    }

    // Transition all images to GENERAL for compute
    std::array<VkImageMemoryBarrier2, 6> to_general{};
    for (uint32_t i = 0; i < image_count; ++i) {
        to_general[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        to_general[i].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        to_general[i].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        to_general[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        to_general[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
                                    | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        to_general[i].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_general[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_general[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_general[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_general[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    to_general[0].image = accum_diffuse_;
    to_general[1].image = accum_specular_;
    to_general[2].image = sample_count_;
    to_general[3].image = convergence_mask_;
    if (adaptive_enabled_) {
        to_general[4].image = variance_mean_;
        to_general[5].image = variance_m2_;
    }

    VkDependencyInfo dep2{};
    dep2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep2.imageMemoryBarrierCount = image_count;
    dep2.pImageMemoryBarriers = to_general.data();
    procs_.pfn_vkCmdPipelineBarrier2(cmd, &dep2);
}

void GpuAccumulator::Accumulate(VkCommandBuffer cmd) {
    procs_.pfn_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, accumulate_pipeline_);
    procs_.pfn_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       accumulate_layout_, 0, 1, &accumulate_desc_set_,
                                       0, nullptr);

    uint32_t groups_x = (width_ + kWorkgroupSize - 1) / kWorkgroupSize;
    uint32_t groups_y = (height_ + kWorkgroupSize - 1) / kWorkgroupSize;
    procs_.pfn_vkCmdDispatch(cmd, groups_x, groups_y, 1);
}

void GpuAccumulator::DispatchFinalize(VkCommandBuffer cmd) {
    // Barrier: accumulate compute → finalize compute
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                          | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    procs_.pfn_vkCmdPipelineBarrier2(cmd, &dep);

    procs_.pfn_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, finalize_pipeline_);
    procs_.pfn_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       finalize_layout_, 0, 1, &finalize_desc_set_,
                                       0, nullptr);

    uint32_t groups_x = (width_ + kWorkgroupSize - 1) / kWorkgroupSize;
    uint32_t groups_y = (height_ + kWorkgroupSize - 1) / kWorkgroupSize;
    procs_.pfn_vkCmdDispatch(cmd, groups_x, groups_y, 1);
}

MultiFrameResult GpuAccumulator::FinalizeNormalized(const ReadbackContext& ctx) {
    // Submit finalize.comp in a one-shot command buffer, then readback
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = ctx.command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    ctx.pfn_vkAllocateCommandBuffers(ctx.device, &alloc_info, &cmd);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ctx.pfn_vkBeginCommandBuffer(cmd, &begin_info);

    DispatchFinalize(cmd);

    ctx.pfn_vkEndCommandBuffer(cmd);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    VkFenceCreateInfo fence_ci{};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    ctx.pfn_vkCreateFence(ctx.device, &fence_ci, nullptr, &fence);
    ctx.pfn_vkQueueSubmit(ctx.queue, 1, &submit_info, fence);
    ctx.pfn_vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
    ctx.pfn_vkDestroyFence(ctx.device, fence, nullptr);
    ctx.pfn_vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd);

    return Finalize(ctx);
}

MultiFrameResult GpuAccumulator::Finalize(const ReadbackContext& ctx) {
    // Read back accumulation images (RGBA32F = 16 bytes/pixel)
    constexpr VkDeviceSize kRGBA32FPixelSize = 16;
    auto diffuse_rb = ReadbackImage(ctx, accum_diffuse_, width_, height_, kRGBA32FPixelSize,
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    auto specular_rb = ReadbackImage(ctx, accum_specular_, width_, height_, kRGBA32FPixelSize,
                                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    if (!diffuse_rb.Handle() || !specular_rb.Handle()) {
        std::fprintf(stderr, "GpuAccumulator::Finalize: readback failed\n");
        return {};
    }

    uint32_t pixel_count = width_ * height_;
    constexpr uint32_t kChannels = 4;
    size_t byte_count = static_cast<size_t>(pixel_count) * kChannels * sizeof(float);

    std::vector<float> diffuse_f32(static_cast<size_t>(pixel_count) * kChannels);
    std::vector<float> specular_f32(static_cast<size_t>(pixel_count) * kChannels);

    auto* d_raw = static_cast<float*>(diffuse_rb.Map());
    std::memcpy(diffuse_f32.data(), d_raw, byte_count);
    diffuse_rb.Unmap();

    auto* s_raw = static_cast<float*>(specular_rb.Map());
    std::memcpy(specular_f32.data(), s_raw, byte_count);
    specular_rb.Unmap();

    return MultiFrameResult{std::move(diffuse_f32), std::move(specular_f32)};
}

// ── Adaptive sampling: image creation ────────────────────────────

bool GpuAccumulator::CreateVarianceImages() {
    VkImageCreateInfo image_ci{};
    image_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_ci.imageType = VK_IMAGE_TYPE_2D;
    image_ci.format = VK_FORMAT_R32_SFLOAT;
    image_ci.extent = {width_, height_, 1};
    image_ci.mipLevels = 1;
    image_ci.arrayLayers = 1;
    image_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                   | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    image_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult result = vmaCreateImage(allocator_, &image_ci, &alloc_ci,
                                     &variance_mean_, &variance_mean_alloc_, nullptr);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: variance_mean image creation failed (%d)\n", result);
        return false;
    }

    result = vmaCreateImage(allocator_, &image_ci, &alloc_ci,
                            &variance_m2_, &variance_m2_alloc_, nullptr);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: variance_m2 image creation failed (%d)\n", result);
        return false;
    }

    return true;
}

bool GpuAccumulator::CreateConvergenceMaskImage() {
    VkImageCreateInfo image_ci{};
    image_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_ci.imageType = VK_IMAGE_TYPE_2D;
    image_ci.format = VK_FORMAT_R8_UINT;
    image_ci.extent = {width_, height_, 1};
    image_ci.mipLevels = 1;
    image_ci.arrayLayers = 1;
    image_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult result = vmaCreateImage(allocator_, &image_ci, &alloc_ci,
                                     &convergence_mask_, &convergence_mask_alloc_, nullptr);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: convergence_mask image creation failed (%d)\n",
                     result);
        return false;
    }

    return true;
}

bool GpuAccumulator::CreateConvergedCounterBuffer() {
    VkBufferCreateInfo buf_ci{};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = sizeof(uint32_t);
    buf_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                 | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult result = vmaCreateBuffer(allocator_, &buf_ci, &alloc_ci,
                                      &converged_count_buffer_, &converged_count_alloc_, nullptr);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: converged_count buffer creation failed (%d)\n",
                     result);
        return false;
    }

    return true;
}

// ── Adaptive sampling: descriptor sets ───────────────────────────

bool GpuAccumulator::CreateVarianceDescriptorResources() {
    // ── Variance update descriptor set: 6 bindings ──
    // 0: noisy_diffuse (readonly image), 1: noisy_specular (readonly image),
    // 2: variance_mean (read-write image), 3: variance_m2 (read-write image),
    // 4: sample_count (readonly image), 5: convergence_mask (readonly image)
    std::array<VkDescriptorSetLayoutBinding, 6> vu_bindings{};
    for (uint32_t i = 0; i < 6; ++i) {
        vu_bindings[i].binding = i;
        vu_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        vu_bindings[i].descriptorCount = 1;
        vu_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo vu_layout_ci{};
    vu_layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    vu_layout_ci.bindingCount = static_cast<uint32_t>(vu_bindings.size());
    vu_layout_ci.pBindings = vu_bindings.data();

    VkResult result = procs_.pfn_vkCreateDescriptorSetLayout(device_, &vu_layout_ci, nullptr,
                                                             &variance_update_desc_layout_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: variance update descriptor layout failed (%d)\n",
                     result);
        return false;
    }

    VkDescriptorPoolSize vu_pool_size{};
    vu_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    vu_pool_size.descriptorCount = 6;

    VkDescriptorPoolCreateInfo vu_pool_ci{};
    vu_pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    vu_pool_ci.maxSets = 1;
    vu_pool_ci.poolSizeCount = 1;
    vu_pool_ci.pPoolSizes = &vu_pool_size;

    result = procs_.pfn_vkCreateDescriptorPool(device_, &vu_pool_ci, nullptr,
                                               &variance_update_desc_pool_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: variance update descriptor pool failed (%d)\n",
                     result);
        return false;
    }

    VkDescriptorSetAllocateInfo vu_alloc{};
    vu_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    vu_alloc.descriptorPool = variance_update_desc_pool_;
    vu_alloc.descriptorSetCount = 1;
    vu_alloc.pSetLayouts = &variance_update_desc_layout_;

    result = procs_.pfn_vkAllocateDescriptorSets(device_, &vu_alloc, &variance_update_desc_set_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: variance update descriptor set alloc failed (%d)\n",
                     result);
        return false;
    }

    // Write variance update descriptor set
    std::array<VkDescriptorImageInfo, 6> vu_image_infos{};
    vu_image_infos[0].imageView = noisy_diffuse_view_;
    vu_image_infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    vu_image_infos[1].imageView = noisy_specular_view_;
    vu_image_infos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    vu_image_infos[2].imageView = variance_mean_view_;
    vu_image_infos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    vu_image_infos[3].imageView = variance_m2_view_;
    vu_image_infos[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    vu_image_infos[4].imageView = sample_count_view_;
    vu_image_infos[4].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    vu_image_infos[5].imageView = convergence_mask_view_;
    vu_image_infos[5].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 6> vu_writes{};
    for (uint32_t i = 0; i < 6; ++i) {
        vu_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        vu_writes[i].dstSet = variance_update_desc_set_;
        vu_writes[i].dstBinding = i;
        vu_writes[i].descriptorCount = 1;
        vu_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        vu_writes[i].pImageInfo = &vu_image_infos[i];
    }

    procs_.pfn_vkUpdateDescriptorSets(device_, static_cast<uint32_t>(vu_writes.size()),
                                      vu_writes.data(), 0, nullptr);

    // ── Convergence check descriptor set: 4 images + 1 buffer ──
    // 0: variance_mean (readonly image), 1: variance_m2 (readonly image),
    // 2: sample_count (readonly image), 3: convergence_mask (read-write image),
    // 4: converged_count (storage buffer)
    std::array<VkDescriptorSetLayoutBinding, 5> cc_bindings{};
    for (uint32_t i = 0; i < 4; ++i) {
        cc_bindings[i].binding = i;
        cc_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        cc_bindings[i].descriptorCount = 1;
        cc_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    cc_bindings[4].binding = 4;
    cc_bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    cc_bindings[4].descriptorCount = 1;
    cc_bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo cc_layout_ci{};
    cc_layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    cc_layout_ci.bindingCount = static_cast<uint32_t>(cc_bindings.size());
    cc_layout_ci.pBindings = cc_bindings.data();

    result = procs_.pfn_vkCreateDescriptorSetLayout(device_, &cc_layout_ci, nullptr,
                                                    &convergence_check_desc_layout_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: convergence check desc layout failed (%d)\n",
                     result);
        return false;
    }

    std::array<VkDescriptorPoolSize, 2> cc_pool_sizes{};
    cc_pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    cc_pool_sizes[0].descriptorCount = 4;
    cc_pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    cc_pool_sizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo cc_pool_ci{};
    cc_pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    cc_pool_ci.maxSets = 1;
    cc_pool_ci.poolSizeCount = static_cast<uint32_t>(cc_pool_sizes.size());
    cc_pool_ci.pPoolSizes = cc_pool_sizes.data();

    result = procs_.pfn_vkCreateDescriptorPool(device_, &cc_pool_ci, nullptr,
                                               &convergence_check_desc_pool_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: convergence check descriptor pool failed (%d)\n",
                     result);
        return false;
    }

    VkDescriptorSetAllocateInfo cc_alloc{};
    cc_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    cc_alloc.descriptorPool = convergence_check_desc_pool_;
    cc_alloc.descriptorSetCount = 1;
    cc_alloc.pSetLayouts = &convergence_check_desc_layout_;

    result = procs_.pfn_vkAllocateDescriptorSets(device_, &cc_alloc, &convergence_check_desc_set_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: convergence check desc set alloc failed (%d)\n",
                     result);
        return false;
    }

    // Write convergence check descriptor set
    std::array<VkDescriptorImageInfo, 4> cc_image_infos{};
    cc_image_infos[0].imageView = variance_mean_view_;
    cc_image_infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    cc_image_infos[1].imageView = variance_m2_view_;
    cc_image_infos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    cc_image_infos[2].imageView = sample_count_view_;
    cc_image_infos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    cc_image_infos[3].imageView = convergence_mask_view_;
    cc_image_infos[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo counter_info{};
    counter_info.buffer = converged_count_buffer_;
    counter_info.offset = 0;
    counter_info.range = sizeof(uint32_t);

    std::array<VkWriteDescriptorSet, 5> cc_writes{};
    for (uint32_t i = 0; i < 4; ++i) {
        cc_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        cc_writes[i].dstSet = convergence_check_desc_set_;
        cc_writes[i].dstBinding = i;
        cc_writes[i].descriptorCount = 1;
        cc_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        cc_writes[i].pImageInfo = &cc_image_infos[i];
    }
    cc_writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    cc_writes[4].dstSet = convergence_check_desc_set_;
    cc_writes[4].dstBinding = 4;
    cc_writes[4].descriptorCount = 1;
    cc_writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    cc_writes[4].pBufferInfo = &counter_info;

    procs_.pfn_vkUpdateDescriptorSets(device_, static_cast<uint32_t>(cc_writes.size()),
                                      cc_writes.data(), 0, nullptr);
    return true;
}

// ── Adaptive sampling: pipelines ─────────────────────────────────

bool GpuAccumulator::CreateVarianceUpdatePipeline(std::string_view shader_dir) {
    std::string shader_path = std::string(shader_dir) + "/variance_update.comp.spv";
    auto shader_code = monti::vulkan::LoadShaderFile(shader_path);
    if (shader_code.empty()) return false;

    VkShaderModuleCreateInfo module_ci{};
    module_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_ci.codeSize = shader_code.size();
    module_ci.pCode = reinterpret_cast<const uint32_t*>(shader_code.data());

    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkResult result = procs_.pfn_vkCreateShaderModule(device_, &module_ci, nullptr, &shader_module);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: variance_update shader module failed (%d)\n", result);
        return false;
    }

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &variance_update_desc_layout_;

    result = procs_.pfn_vkCreatePipelineLayout(device_, &layout_ci, nullptr,
                                               &variance_update_layout_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: variance_update pipeline layout failed (%d)\n",
                     result);
        procs_.pfn_vkDestroyShaderModule(device_, shader_module, nullptr);
        return false;
    }

    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_ci.stage.module = shader_module;
    pipeline_ci.stage.pName = "main";
    pipeline_ci.layout = variance_update_layout_;

    result = procs_.pfn_vkCreateComputePipelines(device_, VK_NULL_HANDLE,
                                                 1, &pipeline_ci, nullptr,
                                                 &variance_update_pipeline_);
    procs_.pfn_vkDestroyShaderModule(device_, shader_module, nullptr);

    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: variance_update compute pipeline failed (%d)\n",
                     result);
        return false;
    }

    return true;
}

bool GpuAccumulator::CreateConvergenceCheckPipeline(std::string_view shader_dir) {
    std::string shader_path = std::string(shader_dir) + "/convergence_check.comp.spv";
    auto shader_code = monti::vulkan::LoadShaderFile(shader_path);
    if (shader_code.empty()) return false;

    VkShaderModuleCreateInfo module_ci{};
    module_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_ci.codeSize = shader_code.size();
    module_ci.pCode = reinterpret_cast<const uint32_t*>(shader_code.data());

    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkResult result = procs_.pfn_vkCreateShaderModule(device_, &module_ci, nullptr, &shader_module);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: convergence_check shader module failed (%d)\n",
                     result);
        return false;
    }

    // Push constant range for convergence parameters
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(uint32_t) + sizeof(float);  // min_frames + threshold

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &convergence_check_desc_layout_;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &push_range;

    result = procs_.pfn_vkCreatePipelineLayout(device_, &layout_ci, nullptr,
                                               &convergence_check_layout_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: convergence_check pipeline layout failed (%d)\n",
                     result);
        procs_.pfn_vkDestroyShaderModule(device_, shader_module, nullptr);
        return false;
    }

    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_ci.stage.module = shader_module;
    pipeline_ci.stage.pName = "main";
    pipeline_ci.layout = convergence_check_layout_;

    result = procs_.pfn_vkCreateComputePipelines(device_, VK_NULL_HANDLE,
                                                 1, &pipeline_ci, nullptr,
                                                 &convergence_check_pipeline_);
    procs_.pfn_vkDestroyShaderModule(device_, shader_module, nullptr);

    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: convergence_check compute pipeline failed (%d)\n",
                     result);
        return false;
    }

    return true;
}

// ── Adaptive sampling: dispatch methods ──────────────────────────

void GpuAccumulator::UpdateVariance(VkCommandBuffer cmd) {
    // Barrier: accumulate compute → variance update compute
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                          | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    procs_.pfn_vkCmdPipelineBarrier2(cmd, &dep);

    procs_.pfn_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 variance_update_pipeline_);
    procs_.pfn_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       variance_update_layout_, 0, 1,
                                       &variance_update_desc_set_, 0, nullptr);

    uint32_t groups_x = (width_ + kWorkgroupSize - 1) / kWorkgroupSize;
    uint32_t groups_y = (height_ + kWorkgroupSize - 1) / kWorkgroupSize;
    procs_.pfn_vkCmdDispatch(cmd, groups_x, groups_y, 1);
}

uint32_t GpuAccumulator::CheckConvergence(VkCommandBuffer cmd, uint32_t min_frames,
                                           float threshold, const ReadbackContext& ctx) {
    // Zero the atomic counter before convergence check
    {
        VkMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &barrier;
        procs_.pfn_vkCmdPipelineBarrier2(cmd, &dep);

        procs_.pfn_vkCmdFillBuffer(cmd, converged_count_buffer_, 0, sizeof(uint32_t), 0);

        VkMemoryBarrier2 barrier2{};
        barrier2.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier2.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier2.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier2.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier2.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                               | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

        VkDependencyInfo dep2{};
        dep2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep2.memoryBarrierCount = 1;
        dep2.pMemoryBarriers = &barrier2;
        procs_.pfn_vkCmdPipelineBarrier2(cmd, &dep2);
    }

    // Push convergence parameters
    struct ConvergenceParams {
        uint32_t min_frames;
        float threshold;
    } params{min_frames, threshold};

    procs_.pfn_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 convergence_check_pipeline_);
    procs_.pfn_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       convergence_check_layout_, 0, 1,
                                       &convergence_check_desc_set_, 0, nullptr);
    procs_.pfn_vkCmdPushConstants(cmd, convergence_check_layout_,
                                  VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                  sizeof(params), &params);

    uint32_t groups_x = (width_ + kWorkgroupSize - 1) / kWorkgroupSize;
    uint32_t groups_y = (height_ + kWorkgroupSize - 1) / kWorkgroupSize;
    procs_.pfn_vkCmdDispatch(cmd, groups_x, groups_y, 1);

    // Copy counter to a staging buffer for readback
    StagingBuffer counter_staging;
    counter_staging.Create(ctx.allocator, sizeof(uint32_t));

    VkMemoryBarrier2 copy_barrier{};
    copy_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    copy_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    copy_barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    copy_barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    copy_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;

    VkDependencyInfo copy_dep{};
    copy_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    copy_dep.memoryBarrierCount = 1;
    copy_dep.pMemoryBarriers = &copy_barrier;
    procs_.pfn_vkCmdPipelineBarrier2(cmd, &copy_dep);

    VkBufferCopy region{};
    region.size = sizeof(uint32_t);
    procs_.pfn_vkCmdCopyBuffer(cmd, converged_count_buffer_, counter_staging.Handle(),
                               1, &region);

    // Submit and wait
    ctx.pfn_vkEndCommandBuffer(cmd);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    VkFenceCreateInfo fence_ci{};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    ctx.pfn_vkCreateFence(ctx.device, &fence_ci, nullptr, &fence);
    ctx.pfn_vkQueueSubmit(ctx.queue, 1, &submit_info, fence);
    ctx.pfn_vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
    ctx.pfn_vkDestroyFence(ctx.device, fence, nullptr);
    ctx.pfn_vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd);

    auto* mapped = static_cast<uint32_t*>(counter_staging.Map());
    uint32_t count = *mapped;
    counter_staging.Unmap();

    return count;
}

void GpuAccumulator::DestroyAccumulationImages() {
    if (accum_diffuse_view_ != VK_NULL_HANDLE) {
        procs_.pfn_vkDestroyImageView(device_, accum_diffuse_view_, nullptr);
        accum_diffuse_view_ = VK_NULL_HANDLE;
    }
    if (accum_specular_view_ != VK_NULL_HANDLE) {
        procs_.pfn_vkDestroyImageView(device_, accum_specular_view_, nullptr);
        accum_specular_view_ = VK_NULL_HANDLE;
    }
    if (sample_count_view_ != VK_NULL_HANDLE) {
        procs_.pfn_vkDestroyImageView(device_, sample_count_view_, nullptr);
        sample_count_view_ = VK_NULL_HANDLE;
    }
    if (accum_diffuse_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, accum_diffuse_, accum_diffuse_alloc_);
        accum_diffuse_ = VK_NULL_HANDLE;
        accum_diffuse_alloc_ = VK_NULL_HANDLE;
    }
    if (accum_specular_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, accum_specular_, accum_specular_alloc_);
        accum_specular_ = VK_NULL_HANDLE;
        accum_specular_alloc_ = VK_NULL_HANDLE;
    }
    if (sample_count_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, sample_count_, sample_count_alloc_);
        sample_count_ = VK_NULL_HANDLE;
        sample_count_alloc_ = VK_NULL_HANDLE;
    }

    // Adaptive sampling images
    if (variance_mean_view_ != VK_NULL_HANDLE) {
        procs_.pfn_vkDestroyImageView(device_, variance_mean_view_, nullptr);
        variance_mean_view_ = VK_NULL_HANDLE;
    }
    if (variance_m2_view_ != VK_NULL_HANDLE) {
        procs_.pfn_vkDestroyImageView(device_, variance_m2_view_, nullptr);
        variance_m2_view_ = VK_NULL_HANDLE;
    }
    if (convergence_mask_view_ != VK_NULL_HANDLE) {
        procs_.pfn_vkDestroyImageView(device_, convergence_mask_view_, nullptr);
        convergence_mask_view_ = VK_NULL_HANDLE;
    }
    if (variance_mean_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, variance_mean_, variance_mean_alloc_);
        variance_mean_ = VK_NULL_HANDLE;
        variance_mean_alloc_ = VK_NULL_HANDLE;
    }
    if (variance_m2_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, variance_m2_, variance_m2_alloc_);
        variance_m2_ = VK_NULL_HANDLE;
        variance_m2_alloc_ = VK_NULL_HANDLE;
    }
    if (convergence_mask_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, convergence_mask_, convergence_mask_alloc_);
        convergence_mask_ = VK_NULL_HANDLE;
        convergence_mask_alloc_ = VK_NULL_HANDLE;
    }
}

}  // namespace monti::capture
