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

    if (pipeline_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyPipeline(device_, pipeline_, nullptr);
    if (pipeline_layout_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    if (desc_pool_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
    if (desc_set_layout_ != VK_NULL_HANDLE)
        procs_.pfn_vkDestroyDescriptorSetLayout(device_, desc_set_layout_, nullptr);

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
    if (!CreateImageViews(desc.noisy_diffuse, desc.noisy_specular)) return false;
    if (!CreateDescriptorResources()) return false;
    if (!CreatePipeline(desc.shader_dir)) return false;
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

    return true;
}

bool GpuAccumulator::CreateDescriptorResources() {
    // 4 storage image bindings: 2 source (readonly) + 2 accumulator (read-write)
    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
    for (uint32_t i = 0; i < 4; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_ci.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_ci.pBindings = bindings.data();

    VkResult result = procs_.pfn_vkCreateDescriptorSetLayout(device_, &layout_ci, nullptr,
                                                             &desc_set_layout_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: descriptor layout creation failed (%d)\n", result);
        return false;
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_size.descriptorCount = 4;

    VkDescriptorPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets = 1;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes = &pool_size;

    result = procs_.pfn_vkCreateDescriptorPool(device_, &pool_ci, nullptr, &desc_pool_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: descriptor pool creation failed (%d)\n", result);
        return false;
    }

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = desc_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &desc_set_layout_;

    result = procs_.pfn_vkAllocateDescriptorSets(device_, &alloc_info, &desc_set_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: descriptor set allocation failed (%d)\n", result);
        return false;
    }

    // Write descriptor set
    std::array<VkDescriptorImageInfo, 4> image_infos{};
    image_infos[0].imageView = noisy_diffuse_view_;
    image_infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_infos[1].imageView = noisy_specular_view_;
    image_infos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_infos[2].imageView = accum_diffuse_view_;
    image_infos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_infos[3].imageView = accum_specular_view_;
    image_infos[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 4> writes{};
    for (uint32_t i = 0; i < 4; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = desc_set_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo = &image_infos[i];
    }

    procs_.pfn_vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()),
                                      writes.data(), 0, nullptr);
    return true;
}

bool GpuAccumulator::CreatePipeline(std::string_view shader_dir) {
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
        std::fprintf(stderr, "GpuAccumulator: shader module creation failed (%d)\n", result);
        return false;
    }

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(float);

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &desc_set_layout_;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &push_range;

    result = procs_.pfn_vkCreatePipelineLayout(device_, &layout_ci, nullptr, &pipeline_layout_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: pipeline layout creation failed (%d)\n", result);
        procs_.pfn_vkDestroyShaderModule(device_, shader_module, nullptr);
        return false;
    }

    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_ci.stage.module = shader_module;
    pipeline_ci.stage.pName = "main";
    pipeline_ci.layout = pipeline_layout_;

    result = procs_.pfn_vkCreateComputePipelines(device_, VK_NULL_HANDLE,
                                                 1, &pipeline_ci, nullptr, &pipeline_);
    procs_.pfn_vkDestroyShaderModule(device_, shader_module, nullptr);

    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "GpuAccumulator: compute pipeline creation failed (%d)\n", result);
        return false;
    }

    return true;
}

void GpuAccumulator::Reset(VkCommandBuffer cmd) {
    // Transition accum images to TRANSFER_DST for clear
    std::array<VkImageMemoryBarrier2, 2> to_dst{};
    for (uint32_t i = 0; i < 2; ++i) {
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

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 2;
    dep.pImageMemoryBarriers = to_dst.data();
    procs_.pfn_vkCmdPipelineBarrier2(cmd, &dep);

    VkClearColorValue clear_value{};
    clear_value.float32[0] = 0.0f;
    clear_value.float32[1] = 0.0f;
    clear_value.float32[2] = 0.0f;
    clear_value.float32[3] = 0.0f;
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    procs_.pfn_vkCmdClearColorImage(cmd, accum_diffuse_,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    &clear_value, 1, &range);
    procs_.pfn_vkCmdClearColorImage(cmd, accum_specular_,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    &clear_value, 1, &range);

    // Transition accum images to GENERAL for compute
    std::array<VkImageMemoryBarrier2, 2> to_general{};
    for (uint32_t i = 0; i < 2; ++i) {
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

    VkDependencyInfo dep2{};
    dep2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep2.imageMemoryBarrierCount = 2;
    dep2.pImageMemoryBarriers = to_general.data();
    procs_.pfn_vkCmdPipelineBarrier2(cmd, &dep2);
}

void GpuAccumulator::Accumulate(VkCommandBuffer cmd, float weight) {
    procs_.pfn_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    procs_.pfn_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline_layout_, 0, 1, &desc_set_, 0, nullptr);
    procs_.pfn_vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(float), &weight);

    uint32_t groups_x = (width_ + kWorkgroupSize - 1) / kWorkgroupSize;
    uint32_t groups_y = (height_ + kWorkgroupSize - 1) / kWorkgroupSize;
    procs_.pfn_vkCmdDispatch(cmd, groups_x, groups_y, 1);
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
}

}  // namespace monti::capture
