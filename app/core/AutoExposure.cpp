#include <volk.h>

#include "AutoExposure.h"

#include <array>
#include <cstdio>
#include <string>

#include <monti/vulkan/ShaderFile.h>
#include <monti/vulkan/VulkanBarriers.h>

namespace monti::app {

namespace {

constexpr uint32_t kWorkgroupSize = 16;

// Push constants for luminance_resolve.comp
struct ResolvePushConstants {
    float adaptation_speed;
    float delta_time;
};

}  // anonymous namespace

AutoExposure::~AutoExposure() {
    Destroy();
}

bool AutoExposure::Create(VkDevice device, VmaAllocator allocator,
                          std::string_view shader_dir,
                          uint32_t width, uint32_t height,
                          VkImageView hdr_input_view) {
    device_ = device;
    allocator_ = allocator;
    width_ = width;
    height_ = height;

    if (!CreateAccumBuffer()) return false;
    if (!CreateResultBuffer()) return false;
    if (!CreateDescriptorLayout()) return false;
    if (!AllocateDescriptorSets()) return false;
    if (!CreatePipelines(shader_dir)) return false;

    UpdateDescriptorSets(hdr_input_view);
    return true;
}

void AutoExposure::Destroy() {
    if (device_ == VK_NULL_HANDLE) return;

    if (accum_pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, accum_pipeline_, nullptr);
        accum_pipeline_ = VK_NULL_HANDLE;
    }
    if (accum_pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, accum_pipeline_layout_, nullptr);
        accum_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (resolve_pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, resolve_pipeline_, nullptr);
        resolve_pipeline_ = VK_NULL_HANDLE;
    }
    if (resolve_pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, resolve_pipeline_layout_, nullptr);
        resolve_pipeline_layout_ = VK_NULL_HANDLE;
    }

    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
        accum_descriptor_set_ = VK_NULL_HANDLE;
        resolve_descriptor_set_ = VK_NULL_HANDLE;
    }
    if (accum_descriptor_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, accum_descriptor_layout_, nullptr);
        accum_descriptor_layout_ = VK_NULL_HANDLE;
    }
    if (resolve_descriptor_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, resolve_descriptor_layout_, nullptr);
        resolve_descriptor_layout_ = VK_NULL_HANDLE;
    }

    if (accum_buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, accum_buffer_, accum_allocation_);
        accum_buffer_ = VK_NULL_HANDLE;
        accum_allocation_ = VK_NULL_HANDLE;
    }
    if (result_buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, result_buffer_, result_allocation_);
        result_buffer_ = VK_NULL_HANDLE;
        result_allocation_ = VK_NULL_HANDLE;
        result_mapped_ = nullptr;
    }

    width_ = 0;
    height_ = 0;
    device_ = VK_NULL_HANDLE;
}

bool AutoExposure::Resize(uint32_t width, uint32_t height, VkImageView hdr_input_view) {
    if (width == 0 || height == 0) return false;
    width_ = width;
    height_ = height;
    // Buffers are resolution-independent — just update the descriptor with the new image view
    UpdateDescriptorSets(hdr_input_view);
    return true;
}

void AutoExposure::Compute(VkCommandBuffer cmd, VkImage hdr_input, float delta_time) {
    if (accum_pipeline_ == VK_NULL_HANDLE || resolve_pipeline_ == VK_NULL_HANDLE) return;

    // 1. First frame: zero the accum buffer (device-local, starts with garbage)
    if (first_frame_) {
        vkCmdFillBuffer(cmd, accum_buffer_, 0, VK_WHOLE_SIZE, 0);

        VkBufferMemoryBarrier2 zero_barrier{};
        zero_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        zero_barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        zero_barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        zero_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        zero_barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        zero_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        zero_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        zero_barrier.buffer = accum_buffer_;
        zero_barrier.offset = 0;
        zero_barrier.size = VK_WHOLE_SIZE;

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.bufferMemoryBarrierCount = 1;
        dep.pBufferMemoryBarriers = &zero_barrier;
        vkCmdPipelineBarrier2(cmd, &dep);

        first_frame_ = false;
    }

    // 2. Barrier: ensure HDR input is readable by compute
    {
        auto barrier = vulkan::MakeImageBarrier(
            hdr_input,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        vulkan::CmdPipelineBarrier(cmd, {&barrier, 1}, vkCmdPipelineBarrier2);
    }

    // 3. Dispatch luminance accumulation
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, accum_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            accum_pipeline_layout_, 0, 1, &accum_descriptor_set_, 0, nullptr);

    uint32_t groups_x = (width_ + kWorkgroupSize - 1) / kWorkgroupSize;
    uint32_t groups_y = (height_ + kWorkgroupSize - 1) / kWorkgroupSize;
    vkCmdDispatch(cmd, groups_x, groups_y, 1);

    // 4. Barrier: SSBO accum write → SSBO resolve read
    {
        VkBufferMemoryBarrier2 buf_barrier{};
        buf_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        buf_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buf_barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        buf_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buf_barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        buf_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        buf_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        buf_barrier.buffer = accum_buffer_;
        buf_barrier.offset = 0;
        buf_barrier.size = VK_WHOLE_SIZE;

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.bufferMemoryBarrierCount = 1;
        dep.pBufferMemoryBarriers = &buf_barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // 5. Dispatch luminance resolve (temporal blend + clear accumulators)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resolve_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            resolve_pipeline_layout_, 0, 1, &resolve_descriptor_set_, 0, nullptr);

    ResolvePushConstants pc{adaptation_speed_, delta_time};
    vkCmdPushConstants(cmd, resolve_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(ResolvePushConstants), &pc);

    vkCmdDispatch(cmd, 1, 1, 1);

    // No additional barrier needed for CPU readback — the host reads the previous
    // frame's value from the persistently mapped host-visible buffer (N-1 latency).
}

float AutoExposure::ExposureMultiplier() const {
    if (!result_mapped_) return 1.0f;
    float adapted = *result_mapped_;
    if (adapted <= 0.0f) return 1.0f;
    float multiplier = 0.18f / adapted;
    // Clamp to prevent extreme over/under-exposure from noisy low-SPP input.
    // Without this, near-black geometric means cause 100x+ multipliers that
    // push all colors into the ACES tonemap shoulder, causing desaturation.
    if (multiplier < 0.01f) return 0.01f;
    if (multiplier > 100.0f) return 100.0f;
    return multiplier;
}

float AutoExposure::AdaptedLuminance() const {
    if (!result_mapped_) return 0.0f;
    return *result_mapped_;
}

bool AutoExposure::CreateAccumBuffer() {
    VkBufferCreateInfo buffer_ci{};
    buffer_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_ci.size = 2 * sizeof(uint32_t);  // log_sum_fixed + pixel_count
    buffer_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;

    VkResult result = vmaCreateBuffer(allocator_, &buffer_ci, &alloc_ci,
                                      &accum_buffer_, &accum_allocation_, nullptr);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "AutoExposure: accum buffer creation failed (VkResult: %d)\n", result);
        return false;
    }

    // Zero-initialize the accumulator via vkCmdFillBuffer on first Compute() call.
    // Device-local memory starts with undefined contents.

    return true;
}

bool AutoExposure::CreateResultBuffer() {
    VkBufferCreateInfo buffer_ci{};
    buffer_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_ci.size = sizeof(float);  // adapted_luminance
    buffer_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_ci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo alloc_info{};
    VkResult result = vmaCreateBuffer(allocator_, &buffer_ci, &alloc_ci,
                                      &result_buffer_, &result_allocation_, &alloc_info);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "AutoExposure: result buffer creation failed (VkResult: %d)\n", result);
        return false;
    }

    result_mapped_ = static_cast<float*>(alloc_info.pMappedData);
    if (!result_mapped_) {
        std::fprintf(stderr, "AutoExposure: result buffer mapping failed\n");
        vmaDestroyBuffer(allocator_, result_buffer_, result_allocation_);
        result_buffer_ = VK_NULL_HANDLE;
        result_allocation_ = VK_NULL_HANDLE;
        return false;
    }

    // Initialize to 0 so the resolve shader's first-frame branch triggers
    *result_mapped_ = 0.0f;
    vmaFlushAllocation(allocator_, result_allocation_, 0, sizeof(float));

    return true;
}

bool AutoExposure::CreateDescriptorLayout() {
    // Accumulation pipeline layout: binding 0 = image (storage image), binding 1 = SSBO (accum)
    {
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};

        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layout_ci{};
        layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_ci.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_ci.pBindings = bindings.data();

        VkResult result = vkCreateDescriptorSetLayout(device_, &layout_ci, nullptr,
                                                      &accum_descriptor_layout_);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr,
                         "AutoExposure: accum descriptor layout creation failed (VkResult: %d)\n",
                         result);
            return false;
        }
    }

    // Resolve pipeline layout: binding 0 = SSBO (accum), binding 1 = SSBO (result)
    {
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};

        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layout_ci{};
        layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_ci.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_ci.pBindings = bindings.data();

        VkResult result = vkCreateDescriptorSetLayout(device_, &layout_ci, nullptr,
                                                      &resolve_descriptor_layout_);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr,
                         "AutoExposure: resolve descriptor layout creation failed (VkResult: %d)\n",
                         result);
            return false;
        }
    }

    return true;
}

bool AutoExposure::AllocateDescriptorSets() {
    // Pool: 1 storage image + 3 storage buffers across 2 descriptor sets
    std::array<VkDescriptorPoolSize, 2> pool_sizes{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[0].descriptorCount = 1;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[1].descriptorCount = 3;

    VkDescriptorPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets = 2;
    pool_ci.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_ci.pPoolSizes = pool_sizes.data();

    VkResult result = vkCreateDescriptorPool(device_, &pool_ci, nullptr, &descriptor_pool_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr,
                     "AutoExposure: descriptor pool creation failed (VkResult: %d)\n", result);
        return false;
    }

    // Allocate both sets
    std::array<VkDescriptorSetLayout, 2> layouts = {
        accum_descriptor_layout_, resolve_descriptor_layout_
    };
    std::array<VkDescriptorSet, 2> sets{};

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    alloc_info.pSetLayouts = layouts.data();

    result = vkAllocateDescriptorSets(device_, &alloc_info, sets.data());
    if (result != VK_SUCCESS) {
        std::fprintf(stderr,
                     "AutoExposure: descriptor set allocation failed (VkResult: %d)\n", result);
        return false;
    }

    accum_descriptor_set_ = sets[0];
    resolve_descriptor_set_ = sets[1];
    return true;
}

bool AutoExposure::CreatePipelines(std::string_view shader_dir) {
    // === Accumulation pipeline ===
    {
        std::string shader_path = std::string(shader_dir) + "/luminance.comp.spv";
        auto shader_code = vulkan::LoadShaderFile(shader_path);
        if (shader_code.empty()) return false;

        VkShaderModuleCreateInfo module_ci{};
        module_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        module_ci.codeSize = shader_code.size();
        module_ci.pCode = reinterpret_cast<const uint32_t*>(shader_code.data());

        VkShaderModule shader_module = VK_NULL_HANDLE;
        VkResult result = vkCreateShaderModule(device_, &module_ci, nullptr, &shader_module);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr,
                         "AutoExposure: accum shader module creation failed (VkResult: %d)\n",
                         result);
            return false;
        }

        VkPipelineLayoutCreateInfo layout_ci{};
        layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_ci.setLayoutCount = 1;
        layout_ci.pSetLayouts = &accum_descriptor_layout_;

        result = vkCreatePipelineLayout(device_, &layout_ci, nullptr, &accum_pipeline_layout_);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr,
                         "AutoExposure: accum pipeline layout creation failed (VkResult: %d)\n",
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
        pipeline_ci.layout = accum_pipeline_layout_;

        result = vkCreateComputePipelines(device_, VK_NULL_HANDLE,
                                          1, &pipeline_ci, nullptr, &accum_pipeline_);
        vkDestroyShaderModule(device_, shader_module, nullptr);

        if (result != VK_SUCCESS) {
            std::fprintf(stderr,
                         "AutoExposure: accum compute pipeline creation failed (VkResult: %d)\n",
                         result);
            return false;
        }
    }

    // === Resolve pipeline ===
    {
        std::string shader_path = std::string(shader_dir) + "/luminance_resolve.comp.spv";
        auto shader_code = vulkan::LoadShaderFile(shader_path);
        if (shader_code.empty()) return false;

        VkShaderModuleCreateInfo module_ci{};
        module_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        module_ci.codeSize = shader_code.size();
        module_ci.pCode = reinterpret_cast<const uint32_t*>(shader_code.data());

        VkShaderModule shader_module = VK_NULL_HANDLE;
        VkResult result = vkCreateShaderModule(device_, &module_ci, nullptr, &shader_module);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr,
                         "AutoExposure: resolve shader module creation failed (VkResult: %d)\n",
                         result);
            return false;
        }

        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = sizeof(ResolvePushConstants);

        VkPipelineLayoutCreateInfo layout_ci{};
        layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_ci.setLayoutCount = 1;
        layout_ci.pSetLayouts = &resolve_descriptor_layout_;
        layout_ci.pushConstantRangeCount = 1;
        layout_ci.pPushConstantRanges = &push_range;

        result = vkCreatePipelineLayout(device_, &layout_ci, nullptr, &resolve_pipeline_layout_);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr,
                         "AutoExposure: resolve pipeline layout creation failed (VkResult: %d)\n",
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
        pipeline_ci.layout = resolve_pipeline_layout_;

        result = vkCreateComputePipelines(device_, VK_NULL_HANDLE,
                                          1, &pipeline_ci, nullptr, &resolve_pipeline_);
        vkDestroyShaderModule(device_, shader_module, nullptr);

        if (result != VK_SUCCESS) {
            std::fprintf(stderr,
                         "AutoExposure: resolve compute pipeline creation failed (VkResult: %d)\n",
                         result);
            return false;
        }
    }

    return true;
}

void AutoExposure::UpdateDescriptorSets(VkImageView hdr_input_view) {
    // Accum set: binding 0 = HDR input image, binding 1 = accum SSBO
    VkDescriptorImageInfo image_info{};
    image_info.imageView = hdr_input_view;
    image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo accum_buffer_info{};
    accum_buffer_info.buffer = accum_buffer_;
    accum_buffer_info.offset = 0;
    accum_buffer_info.range = 2 * sizeof(uint32_t);

    VkDescriptorBufferInfo result_buffer_info{};
    result_buffer_info.buffer = result_buffer_;
    result_buffer_info.offset = 0;
    result_buffer_info.range = sizeof(float);

    std::array<VkWriteDescriptorSet, 4> writes{};

    // Accum set, binding 0: input image
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = accum_descriptor_set_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].pImageInfo = &image_info;

    // Accum set, binding 1: accum buffer
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = accum_descriptor_set_;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &accum_buffer_info;

    // Resolve set, binding 0: accum buffer
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = resolve_descriptor_set_;
    writes[2].dstBinding = 0;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &accum_buffer_info;

    // Resolve set, binding 1: result buffer
    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = resolve_descriptor_set_;
    writes[3].dstBinding = 1;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].pBufferInfo = &result_buffer_info;

    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
}

}  // namespace monti::app
