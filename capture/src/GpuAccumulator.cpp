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

    if (!CreateAccumulationImages()) return false;
    if (!CreateSampleCountImage()) return false;
    if (!CreateImageViews(desc.noisy_diffuse, desc.noisy_specular)) return false;
    if (!CreateDescriptorResources()) return false;
    if (!CreateAccumulatePipeline(desc.shader_dir)) return false;
    if (!CreateFinalizePipeline(desc.shader_dir)) return false;
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

    return true;
}

bool GpuAccumulator::CreateDescriptorResources() {
    // ── Accumulate descriptor set: 5 storage image bindings ──
    // 0-1: source (readonly), 2-3: accumulator (read-write), 4: sample_count (read-write)
    std::array<VkDescriptorSetLayoutBinding, 5> acc_bindings{};
    for (uint32_t i = 0; i < 5; ++i) {
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
    acc_pool_size.descriptorCount = 5;

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
    std::array<VkDescriptorImageInfo, 5> acc_image_infos{};
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

    std::array<VkWriteDescriptorSet, 5> acc_writes{};
    for (uint32_t i = 0; i < 5; ++i) {
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
    // Transition accum images + sample_count to TRANSFER_DST for clear
    std::array<VkImageMemoryBarrier2, 3> to_dst{};
    for (uint32_t i = 0; i < 3; ++i) {
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

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 3;
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

    // Transition all images to GENERAL for compute
    std::array<VkImageMemoryBarrier2, 3> to_general{};
    for (uint32_t i = 0; i < 3; ++i) {
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

    VkDependencyInfo dep2{};
    dep2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep2.imageMemoryBarrierCount = 3;
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
}

}  // namespace monti::capture
