#include <volk.h>

#include <deni/vulkan/Denoiser.h>

#include <array>
#include <cstdio>
#include <fstream>
#include <vector>

namespace deni::vulkan {

namespace {

constexpr uint32_t kWorkgroupSize = 16;

std::vector<uint8_t> LoadShaderFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::fprintf(stderr, "deni::Denoiser: failed to open shader file: %s\n", path.c_str());
        return {};
    }
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

}  // namespace

std::unique_ptr<Denoiser> Denoiser::Create(const DenoiserDesc& desc) {
    if (!desc.allocator) {
        std::fprintf(stderr, "deni::Denoiser::Create: allocator must not be null\n");
        return nullptr;
    }

    auto denoiser = std::unique_ptr<Denoiser>(new Denoiser());
    denoiser->device_ = desc.device;
    denoiser->allocator_ = desc.allocator;

    if (!denoiser->CreateDescriptorLayout()) return nullptr;
    if (!denoiser->AllocateDescriptorSet()) return nullptr;
    if (!denoiser->CreatePipeline(desc.shader_dir, desc.pipeline_cache)) return nullptr;
    if (!denoiser->CreateOutputImage(desc.width, desc.height)) return nullptr;

    return denoiser;
}

Denoiser::~Denoiser() {
    if (device_ == VK_NULL_HANDLE) return;

    DestroyOutputImage();

    if (pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, pipeline_, nullptr);
    if (pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    if (descriptor_pool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
    if (descriptor_set_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
}

DenoiserOutput Denoiser::Denoise(VkCommandBuffer cmd, const DenoiserInput& input) {
    if (output_image_ == VK_NULL_HANDLE || pipeline_ == VK_NULL_HANDLE)
        return {};

    UpdateDescriptorSet(input);

    // Transition output image to GENERAL (from UNDEFINED) for compute write
    VkImageMemoryBarrier2 to_general{};
    to_general.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_general.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    to_general.srcAccessMask = 0;
    to_general.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    to_general.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    to_general.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    to_general.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_general.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_general.image = output_image_;
    to_general.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &to_general;
    vkCmdPipelineBarrier2(cmd, &dep);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);

    uint32_t groups_x = (output_width_ + kWorkgroupSize - 1) / kWorkgroupSize;
    uint32_t groups_y = (output_height_ + kWorkgroupSize - 1) / kWorkgroupSize;
    vkCmdDispatch(cmd, groups_x, groups_y, 1);

    return {output_image_, output_view_};
}

void Denoiser::Resize(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) return;
    if (width == output_width_ && height == output_height_) return;

    DestroyOutputImage();
    CreateOutputImage(width, height);
}

float Denoiser::LastPassTimeMs() const { return 0.0f; }

bool Denoiser::CreateOutputImage(uint32_t width, uint32_t height) {
    VkImageCreateInfo image_ci{};
    image_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_ci.imageType = VK_IMAGE_TYPE_2D;
    image_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    image_ci.extent = {width, height, 1};
    image_ci.mipLevels = 1;
    image_ci.arrayLayers = 1;
    image_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    image_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult result = vmaCreateImage(allocator_, &image_ci, &alloc_ci,
                                     &output_image_, &output_allocation_, nullptr);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "deni::Denoiser: failed to create output image (VkResult: %d)\n",
                     result);
        return false;
    }

    VkImageViewCreateInfo view_ci{};
    view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image = output_image_;
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    result = vkCreateImageView(device_, &view_ci, nullptr, &output_view_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "deni::Denoiser: failed to create output image view (VkResult: %d)\n",
                     result);
        vmaDestroyImage(allocator_, output_image_, output_allocation_);
        output_image_ = VK_NULL_HANDLE;
        output_allocation_ = VK_NULL_HANDLE;
        return false;
    }

    output_width_ = width;
    output_height_ = height;
    return true;
}

void Denoiser::DestroyOutputImage() {
    if (output_view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_, output_view_, nullptr);
        output_view_ = VK_NULL_HANDLE;
    }
    if (output_image_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, output_image_, output_allocation_);
        output_image_ = VK_NULL_HANDLE;
        output_allocation_ = VK_NULL_HANDLE;
    }
    output_width_ = 0;
    output_height_ = 0;
}

bool Denoiser::CreateDescriptorLayout() {
    // Binding 0: noisy diffuse (storage image, readonly)
    // Binding 1: noisy specular (storage image, readonly)
    // Binding 2: output (storage image, writeonly)
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_ci.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_ci.pBindings = bindings.data();

    VkResult result = vkCreateDescriptorSetLayout(device_, &layout_ci, nullptr,
                                                  &descriptor_set_layout_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr,
                     "deni::Denoiser: failed to create descriptor set layout (VkResult: %d)\n",
                     result);
        return false;
    }
    return true;
}

bool Denoiser::AllocateDescriptorSet() {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_size.descriptorCount = 3;

    VkDescriptorPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets = 1;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes = &pool_size;

    VkResult result = vkCreateDescriptorPool(device_, &pool_ci, nullptr, &descriptor_pool_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr,
                     "deni::Denoiser: failed to create descriptor pool (VkResult: %d)\n",
                     result);
        return false;
    }

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &descriptor_set_layout_;

    result = vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr,
                     "deni::Denoiser: failed to allocate descriptor set (VkResult: %d)\n",
                     result);
        return false;
    }
    return true;
}

void Denoiser::UpdateDescriptorSet(const DenoiserInput& input) {
    VkDescriptorImageInfo diffuse_info{};
    diffuse_info.imageView = input.noisy_diffuse;
    diffuse_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo specular_info{};
    specular_info.imageView = input.noisy_specular;
    specular_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo output_info{};
    output_info.imageView = output_view_;
    output_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 3> writes{};

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descriptor_set_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].pImageInfo = &diffuse_info;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descriptor_set_;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &specular_info;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = descriptor_set_;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[2].pImageInfo = &output_info;

    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
}

bool Denoiser::CreatePipeline(std::string_view shader_dir, VkPipelineCache pipeline_cache) {
    std::string shader_path = std::string(shader_dir) + "/passthrough_denoise.comp.spv";
    auto shader_code = LoadShaderFile(shader_path);
    if (shader_code.empty()) return false;

    VkShaderModuleCreateInfo module_ci{};
    module_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_ci.codeSize = shader_code.size();
    module_ci.pCode = reinterpret_cast<const uint32_t*>(shader_code.data());

    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkResult result = vkCreateShaderModule(device_, &module_ci, nullptr, &shader_module);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr,
                     "deni::Denoiser: failed to create shader module (VkResult: %d)\n",
                     result);
        return false;
    }

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &descriptor_set_layout_;

    result = vkCreatePipelineLayout(device_, &layout_ci, nullptr, &pipeline_layout_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr,
                     "deni::Denoiser: failed to create pipeline layout (VkResult: %d)\n",
                     result);
        vkDestroyShaderModule(device_, shader_module, nullptr);
        return false;
    }

    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_ci.stage.module = shader_module;
    pipeline_ci.stage.pName = "main";
    pipeline_ci.layout = pipeline_layout_;

    result = vkCreateComputePipelines(device_, pipeline_cache,
                                      1, &pipeline_ci, nullptr, &pipeline_);

    vkDestroyShaderModule(device_, shader_module, nullptr);

    if (result != VK_SUCCESS) {
        std::fprintf(stderr,
                     "deni::Denoiser: failed to create compute pipeline (VkResult: %d)\n",
                     result);
        return false;
    }

    return true;
}

} // namespace deni::vulkan
