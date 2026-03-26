#include "MlInference.h"

#include <deni/vulkan/Denoiser.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace deni::vulkan {

// ---------------------------------------------------------------------------
// MlDeviceDispatch
// ---------------------------------------------------------------------------

bool MlDeviceDispatch::Load(VkDevice device, PFN_vkGetDeviceProcAddr get_proc) {
    bool ok = true;
    auto resolve = [&](auto& fn_ptr, const char* name) {
        fn_ptr = reinterpret_cast<std::remove_reference_t<decltype(fn_ptr)>>(
            get_proc(device, name));
        if (!fn_ptr) {
            std::fprintf(stderr, "deni::MlInference: failed to resolve %s\n", name);
            ok = false;
        }
    };

    resolve(vkCreateBuffer,              "vkCreateBuffer");
    resolve(vkDestroyBuffer,             "vkDestroyBuffer");
    resolve(vkCreateImageView,           "vkCreateImageView");
    resolve(vkDestroyImageView,          "vkDestroyImageView");
    resolve(vkCmdCopyBuffer,             "vkCmdCopyBuffer");
    resolve(vkCmdPipelineBarrier2,       "vkCmdPipelineBarrier2");
    resolve(vkCreateShaderModule,        "vkCreateShaderModule");
    resolve(vkDestroyShaderModule,       "vkDestroyShaderModule");
    resolve(vkCreatePipelineLayout,      "vkCreatePipelineLayout");
    resolve(vkDestroyPipelineLayout,     "vkDestroyPipelineLayout");
    resolve(vkCreateComputePipelines,    "vkCreateComputePipelines");
    resolve(vkDestroyPipeline,           "vkDestroyPipeline");
    resolve(vkCreateDescriptorSetLayout, "vkCreateDescriptorSetLayout");
    resolve(vkDestroyDescriptorSetLayout,"vkDestroyDescriptorSetLayout");
    resolve(vkCreateDescriptorPool,      "vkCreateDescriptorPool");
    resolve(vkDestroyDescriptorPool,     "vkDestroyDescriptorPool");
    resolve(vkAllocateDescriptorSets,    "vkAllocateDescriptorSets");
    resolve(vkUpdateDescriptorSets,      "vkUpdateDescriptorSets");
    resolve(vkCmdBindPipeline,           "vkCmdBindPipeline");
    resolve(vkCmdBindDescriptorSets,     "vkCmdBindDescriptorSets");
    resolve(vkCmdDispatch,               "vkCmdDispatch");
    resolve(vkCmdPushConstants,          "vkCmdPushConstants");

    resolve(vkCreateQueryPool,           "vkCreateQueryPool");
    resolve(vkDestroyQueryPool,          "vkDestroyQueryPool");
    resolve(vkCmdWriteTimestamp2,        "vkCmdWriteTimestamp2");
    resolve(vkCmdResetQueryPool,         "vkCmdResetQueryPool");
    resolve(vkGetQueryPoolResults,       "vkGetQueryPoolResults");

    resolve(vkResetDescriptorPool,       "vkResetDescriptorPool");

    return ok;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

std::vector<uint8_t> LoadShaderFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

uint32_t DivCeil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

}  // namespace

// ---------------------------------------------------------------------------
// MlInference lifecycle
// ---------------------------------------------------------------------------

MlInference::MlInference(VkDevice device, VmaAllocator allocator,
                          PFN_vkGetDeviceProcAddr get_device_proc_addr,
                          std::string_view shader_dir, VkPipelineCache pipeline_cache,
                          uint32_t width, uint32_t height,
                          float timestamp_period)
    : device_(device), allocator_(allocator),
      shader_dir_(shader_dir), pipeline_cache_(pipeline_cache),
      width_(width), height_(height), timestamp_period_(timestamp_period) {
    dispatch_.Load(device, get_device_proc_addr);
    CreateQueryPool();
    CreateDescriptorPool();
}

MlInference::~MlInference() {
    DestroyFeatureBuffers();
    DestroyReductionBuffer();
    DestroyPipelines();
    DestroyDescriptorPool();
    DestroyWeightBuffers();
    DestroyQueryPool();

    if (staging_buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, staging_buffer_, staging_allocation_);
        staging_buffer_ = VK_NULL_HANDLE;
        staging_allocation_ = VK_NULL_HANDLE;
    }
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

bool MlInference::LoadWeights(const WeightData& weights, VkCommandBuffer cmd) {
    DestroyWeightBuffers();

    if (!InferArchitectureFromWeights(weights)) return false;

    // Calculate total staging size. For conv layers, we concatenate weight + bias
    // into a single buffer. For norm layers, we concatenate gamma + beta.
    // First pass: identify pairs and calculate sizes.
    struct CombinedLayer {
        std::string name;           // Combined name (e.g., "down0.conv1.conv")
        const LayerWeights* data1;  // First part (weight/gamma)
        const LayerWeights* data2;  // Second part (bias/beta), may be nullptr
    };
    std::vector<CombinedLayer> combined_layers;

    // Build a lookup map for finding bias/beta partners
    std::unordered_map<std::string, size_t> layer_name_to_idx;
    for (size_t i = 0; i < weights.layers.size(); ++i)
        layer_name_to_idx[weights.layers[i].name] = i;

    // Track which layers have been consumed as a second part
    std::vector<bool> consumed(weights.layers.size(), false);

    for (size_t i = 0; i < weights.layers.size(); ++i) {
        if (consumed[i]) continue;
        const auto& layer = weights.layers[i];

        // Check if this is a ".weight" layer with a matching ".bias"
        std::string base_name;
        bool is_weight = false;
        if (layer.name.size() > 7 && layer.name.substr(layer.name.size() - 7) == ".weight") {
            base_name = layer.name.substr(0, layer.name.size() - 7);
            is_weight = true;
        }

        if (is_weight) {
            auto bias_it = layer_name_to_idx.find(base_name + ".bias");
            if (bias_it != layer_name_to_idx.end()) {
                consumed[bias_it->second] = true;
                CombinedLayer cl;
                cl.name = base_name;
                cl.data1 = &layer;
                cl.data2 = &weights.layers[bias_it->second];
                combined_layers.push_back(std::move(cl));
            } else {
                CombinedLayer cl;
                cl.name = layer.name;
                cl.data1 = &layer;
                cl.data2 = nullptr;
                combined_layers.push_back(std::move(cl));
            }
        } else {
            CombinedLayer cl;
            cl.name = layer.name;
            cl.data1 = &layer;
            cl.data2 = nullptr;
            combined_layers.push_back(std::move(cl));
        }
    }

    VkDeviceSize total_size = 0;
    for (const auto& cl : combined_layers) {
        total_size += cl.data1->data.size() * sizeof(float);
        if (cl.data2) total_size += cl.data2->data.size() * sizeof(float);
    }

    if (total_size == 0) {
        std::fprintf(stderr, "deni::MlInference: weight data is empty\n");
        return false;
    }

    // Create staging buffer
    VkBufferCreateInfo staging_ci{};
    staging_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    staging_ci.size = total_size;
    staging_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo staging_alloc_ci{};
    staging_alloc_ci.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    staging_alloc_ci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo staging_info{};
    VkResult result = vmaCreateBuffer(allocator_, &staging_ci, &staging_alloc_ci,
                                      &staging_buffer_, &staging_allocation_, &staging_info);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr,
                     "deni::MlInference: failed to create staging buffer (VkResult: %d)\n",
                     result);
        return false;
    }

    auto* staging_ptr = static_cast<char*>(staging_info.pMappedData);
    VkDeviceSize staging_offset = 0;

    weight_buffers_.reserve(combined_layers.size());

    for (uint32_t i = 0; i < combined_layers.size(); ++i) {
        const auto& cl = combined_layers[i];
        VkDeviceSize part1_size = cl.data1->data.size() * sizeof(float);
        VkDeviceSize part2_size = cl.data2 ? cl.data2->data.size() * sizeof(float) : 0;
        VkDeviceSize layer_size = part1_size + part2_size;

        // Copy data1 (weight/gamma) then data2 (bias/beta) into staging contiguously
        std::memcpy(staging_ptr + staging_offset, cl.data1->data.data(), part1_size);
        if (cl.data2)
            std::memcpy(staging_ptr + staging_offset + part1_size,
                        cl.data2->data.data(), part2_size);

        VkBufferCreateInfo buffer_ci{};
        buffer_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_ci.size = layer_size;
        buffer_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VmaAllocationCreateInfo alloc_ci{};
        alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        WeightBuffer wb;
        wb.name = cl.name;
        wb.size_bytes = layer_size;

        result = vmaCreateBuffer(allocator_, &buffer_ci, &alloc_ci,
                                 &wb.buffer, &wb.allocation, nullptr);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr,
                         "deni::MlInference: failed to create weight buffer for '%s' "
                         "(VkResult: %d)\n",
                         cl.name.c_str(), result);
            DestroyWeightBuffers();
            return false;
        }

        VkBufferCopy copy_region{};
        copy_region.srcOffset = staging_offset;
        copy_region.dstOffset = 0;
        copy_region.size = layer_size;
        dispatch_.vkCmdCopyBuffer(cmd, staging_buffer_, wb.buffer, 1, &copy_region);

        weight_index_[cl.name] = i;
        weight_buffers_.push_back(std::move(wb));
        staging_offset += layer_size;
    }

    // Barrier: transfer → compute
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    dispatch_.vkCmdPipelineBarrier2(cmd, &dep);

    weights_loaded_ = true;

    // Create pipelines now that we know the architecture
    if (!CreatePipelines()) return false;

    // Allocate feature buffers at current resolution
    if (width_ > 0 && height_ > 0) {
        features_allocated_ = false;  // Force reallocation
        if (!Resize(width_, height_)) return false;
    }

    return true;
}

bool MlInference::ValidateWeights(const WeightData& weights) {
    uint32_t model_in_channels = 0;
    uint32_t model_out_channels = 0;

    for (const auto& layer : weights.layers) {
        if (layer.name == "down0.conv1.conv.weight" && layer.shape.size() == 4)
            model_in_channels = layer.shape[1];
        if (layer.name == "out_conv.weight" && layer.shape.size() == 4)
            model_out_channels = layer.shape[0];
    }

    if (model_in_channels == 0) {
        std::fprintf(stderr, "deni::MlInference: could not find down0.conv1.conv.weight "
                     "in model weights\n");
        return false;
    }
    if (model_in_channels != kInputChannels) {
        std::fprintf(stderr,
                     "deni::MlInference: model has %u input channels but shaders expect %u "
                     "(model needs retraining)\n",
                     model_in_channels, kInputChannels);
        return false;
    }
    if (model_out_channels != 0 && model_out_channels != kOutputChannels) {
        std::fprintf(stderr,
                     "deni::MlInference: model has %u output channels but shaders expect %u "
                     "(model needs retraining)\n",
                     model_out_channels, kOutputChannels);
        return false;
    }
    return true;
}

bool MlInference::InferArchitectureFromWeights(const WeightData& weights) {
    if (!ValidateWeights(weights)) return false;

    for (const auto& layer : weights.layers) {
        if (layer.name == "down0.conv1.conv.weight" && layer.shape.size() == 4) {
            level0_channels_ = layer.shape[0];
            level1_channels_ = level0_channels_ * 2;
            level2_channels_ = level0_channels_ * 4;
            break;
        }
    }

    std::fprintf(stderr,
                 "deni::MlInference: inferred architecture base_channels=%u "
                 "(levels: %u, %u, %u)\n",
                 level0_channels_, level0_channels_, level1_channels_, level2_channels_);
    return true;
}

VkBuffer MlInference::FindWeightBuffer(std::string_view name) const {
    auto it = weight_index_.find(std::string(name));
    if (it == weight_index_.end()) {
        std::fprintf(stderr, "deni::MlInference: weight buffer '%.*s' not found\n",
                     static_cast<int>(name.size()), name.data());
        return VK_NULL_HANDLE;
    }
    return weight_buffers_[it->second].buffer;
}

void MlInference::FreeStagingBuffer() {
    if (staging_buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, staging_buffer_, staging_allocation_);
        staging_buffer_ = VK_NULL_HANDLE;
        staging_allocation_ = VK_NULL_HANDLE;
    }
}

// ---------------------------------------------------------------------------
// Feature buffer management
// ---------------------------------------------------------------------------

bool MlInference::Resize(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) return false;
    if (width == width_ && height == height_ && features_allocated_) return true;

    DestroyFeatureBuffers();

    width_ = width;
    height_ = height;

    uint32_t w0 = width, h0 = height;
    uint32_t w1 = DivCeil(w0, 2), h1 = DivCeil(h0, 2);
    uint32_t w2 = DivCeil(w1, 2), h2 = DivCeil(h1, 2);

    // Level 0 buffers (full resolution)
    if (!AllocateFeatureBuffer(buf0_a_, level0_channels_, w0, h0)) return false;
    if (!AllocateFeatureBuffer(buf0_b_, level0_channels_, w0, h0)) return false;
    if (!AllocateFeatureBuffer(skip0_, level0_channels_, w0, h0)) return false;

    // Level 1 buffers (half resolution)
    if (!AllocateFeatureBuffer(buf1_a_, level1_channels_, w1, h1)) return false;
    if (!AllocateFeatureBuffer(buf1_b_, level1_channels_, w1, h1)) return false;
    if (!AllocateFeatureBuffer(skip1_, level1_channels_, w1, h1)) return false;

    // Level 2 buffers (quarter resolution)
    if (!AllocateFeatureBuffer(buf2_a_, level2_channels_, w2, h2)) return false;
    if (!AllocateFeatureBuffer(buf2_b_, level2_channels_, w2, h2)) return false;

    // Concat scratch buffers for decoder upsample+skip
    uint32_t concat1_ch = level2_channels_ + level1_channels_;
    if (!AllocateFeatureBuffer(concat1_, concat1_ch, w1, h1)) return false;
    uint32_t concat0_ch = level1_channels_ + level0_channels_;
    if (!AllocateFeatureBuffer(concat0_, concat0_ch, w0, h0)) return false;

    // Reduction buffer for GroupNorm (sized for largest spatial extent)
    uint32_t max_elements_per_group = (level0_channels_ / kNumGroups) * w0 * h0;
    uint32_t reduce_wg_count = DivCeil(max_elements_per_group, kReduceWorkgroupSize);
    if (!AllocateReductionBuffer(reduce_wg_count)) return false;

    // Update push constants
    pc_level0_ = {w0, h0};
    pc_level1_ = {w1, h1};
    pc_level2_ = {w2, h2};
    pc_down0_ = {w0, h0};
    pc_down1_ = {w1, h1};
    pc_up1_ = {w1, h1};
    pc_up0_ = {w0, h0};

    features_allocated_ = true;
    return true;
}

bool MlInference::AllocateFeatureBuffer(FeatureBuffer& buf, uint32_t channels,
                                         uint32_t width, uint32_t height) {
    // Use FP16 storage to halve memory bandwidth
    VkDeviceSize size = static_cast<VkDeviceSize>(channels) * width * height * sizeof(uint16_t);

    VkBufferCreateInfo buffer_ci{};
    buffer_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_ci.size = size;
    buffer_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult result = vmaCreateBuffer(allocator_, &buffer_ci, &alloc_ci,
                                      &buf.buffer, &buf.allocation, nullptr);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr,
                     "deni::MlInference: failed to create feature buffer %ux%ux%u "
                     "(VkResult: %d)\n",
                     channels, width, height, result);
        return false;
    }

    buf.size_bytes = size;
    buf.channels = channels;
    buf.width = width;
    buf.height = height;
    return true;
}

void MlInference::DestroyFeatureBuffer(FeatureBuffer& buf) {
    if (buf.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, buf.buffer, buf.allocation);
        buf = {};
    }
}

void MlInference::DestroyFeatureBuffers() {
    DestroyFeatureBuffer(buf0_a_);
    DestroyFeatureBuffer(buf0_b_);
    DestroyFeatureBuffer(skip0_);
    DestroyFeatureBuffer(buf1_a_);
    DestroyFeatureBuffer(buf1_b_);
    DestroyFeatureBuffer(skip1_);
    DestroyFeatureBuffer(buf2_a_);
    DestroyFeatureBuffer(buf2_b_);
    DestroyFeatureBuffer(concat1_);
    DestroyFeatureBuffer(concat0_);
    features_allocated_ = false;
}

void MlInference::DestroyWeightBuffers() {
    for (auto& wb : weight_buffers_) {
        if (wb.buffer != VK_NULL_HANDLE)
            vmaDestroyBuffer(allocator_, wb.buffer, wb.allocation);
    }
    weight_buffers_.clear();
    weight_index_.clear();
    weights_loaded_ = false;
}

bool MlInference::AllocateReductionBuffer(uint32_t max_reduce_workgroups) {
    DestroyReductionBuffer();
    // [NUM_GROUPS][max_reduce_workgroups][2]  — partial_sum and partial_sum_sq
    VkDeviceSize size = static_cast<VkDeviceSize>(kNumGroups) *
                        max_reduce_workgroups * 2 * sizeof(float);

    VkBufferCreateInfo buffer_ci{};
    buffer_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_ci.size = size;
    buffer_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult result = vmaCreateBuffer(allocator_, &buffer_ci, &alloc_ci,
                                      &reduction_buffer_, &reduction_allocation_, nullptr);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr,
                     "deni::MlInference: failed to create reduction buffer (VkResult: %d)\n",
                     result);
        return false;
    }
    reduction_buffer_size_ = size;
    return true;
}

void MlInference::DestroyReductionBuffer() {
    if (reduction_buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, reduction_buffer_, reduction_allocation_);
        reduction_buffer_ = VK_NULL_HANDLE;
        reduction_allocation_ = VK_NULL_HANDLE;
        reduction_buffer_size_ = 0;
    }
}

// ---------------------------------------------------------------------------
// Shader / Pipeline creation
// ---------------------------------------------------------------------------

VkShaderModule MlInference::LoadShaderModule(std::string_view filename) {
    std::string path = shader_dir_ + "/" + std::string(filename) + ".spv";
    auto spirv = LoadShaderFile(path);
    if (spirv.empty()) {
        std::fprintf(stderr, "deni::MlInference: failed to load shader '%s'\n", path.c_str());
        return VK_NULL_HANDLE;
    }

    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spirv.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(spirv.data());

    VkShaderModule module = VK_NULL_HANDLE;
    VkResult result = dispatch_.vkCreateShaderModule(device_, &ci, nullptr, &module);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr,
                     "deni::MlInference: failed to create shader module '%s' (VkResult: %d)\n",
                     path.c_str(), result);
        return VK_NULL_HANDLE;
    }
    return module;
}

bool MlInference::CreateConvPipeline(uint32_t in_ch, uint32_t out_ch,
                                      VkPipeline& pipeline, VkPipelineLayout& layout,
                                      VkDescriptorSetLayout& ds_layout) {
    // Descriptor set layout: binding 0=input, 1=output, 2=weights (all storage buffers)
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo ds_ci{};
    ds_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ds_ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ds_ci.pBindings = bindings.data();
    VkResult result = dispatch_.vkCreateDescriptorSetLayout(device_, &ds_ci, nullptr, &ds_layout);
    if (result != VK_SUCCESS) return false;

    // Push constants: width, height
    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.size = sizeof(MlPushConstants);

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &ds_layout;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &pc_range;
    result = dispatch_.vkCreatePipelineLayout(device_, &layout_ci, nullptr, &layout);
    if (result != VK_SUCCESS) return false;

    VkShaderModule module = LoadShaderModule("conv.comp");
    if (module == VK_NULL_HANDLE) return false;

    // Specialization constants: IN_CHANNELS, OUT_CHANNELS
    std::array<VkSpecializationMapEntry, 2> spec_entries{};
    spec_entries[0] = {0, 0, sizeof(uint32_t)};
    spec_entries[1] = {1, sizeof(uint32_t), sizeof(uint32_t)};
    std::array<uint32_t, 2> spec_data = {in_ch, out_ch};

    VkSpecializationInfo spec_info{};
    spec_info.mapEntryCount = static_cast<uint32_t>(spec_entries.size());
    spec_info.pMapEntries = spec_entries.data();
    spec_info.dataSize = sizeof(spec_data);
    spec_info.pData = spec_data.data();

    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_ci.stage.module = module;
    pipeline_ci.stage.pName = "main";
    pipeline_ci.stage.pSpecializationInfo = &spec_info;
    pipeline_ci.layout = layout;

    result = dispatch_.vkCreateComputePipelines(device_, pipeline_cache_, 1, &pipeline_ci,
                                                nullptr, &pipeline);
    dispatch_.vkDestroyShaderModule(device_, module, nullptr);
    return result == VK_SUCCESS;
}

bool MlInference::CreateGroupNormReducePipeline(uint32_t channels,
                                                 VkPipeline& pipeline, VkPipelineLayout& layout,
                                                 VkDescriptorSetLayout& ds_layout) {
    // Bindings: 0=data buffer (readonly), 1=reduction buffer (writeonly)
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    for (uint32_t i = 0; i < 2; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo ds_ci{};
    ds_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ds_ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ds_ci.pBindings = bindings.data();
    VkResult result = dispatch_.vkCreateDescriptorSetLayout(device_, &ds_ci, nullptr, &ds_layout);
    if (result != VK_SUCCESS) return false;

    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.size = sizeof(MlPushConstants);

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &ds_layout;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &pc_range;
    result = dispatch_.vkCreatePipelineLayout(device_, &layout_ci, nullptr, &layout);
    if (result != VK_SUCCESS) return false;

    VkShaderModule module = LoadShaderModule("group_norm_reduce.comp");
    if (module == VK_NULL_HANDLE) return false;

    // Spec constants: CHANNELS, NUM_GROUPS
    std::array<VkSpecializationMapEntry, 2> spec_entries{};
    spec_entries[0] = {0, 0, sizeof(uint32_t)};
    spec_entries[1] = {1, sizeof(uint32_t), sizeof(uint32_t)};
    std::array<uint32_t, 2> spec_data = {channels, kNumGroups};

    VkSpecializationInfo spec_info{};
    spec_info.mapEntryCount = static_cast<uint32_t>(spec_entries.size());
    spec_info.pMapEntries = spec_entries.data();
    spec_info.dataSize = sizeof(spec_data);
    spec_info.pData = spec_data.data();

    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_ci.stage.module = module;
    pipeline_ci.stage.pName = "main";
    pipeline_ci.stage.pSpecializationInfo = &spec_info;
    pipeline_ci.layout = layout;

    result = dispatch_.vkCreateComputePipelines(device_, pipeline_cache_, 1, &pipeline_ci,
                                                nullptr, &pipeline);
    dispatch_.vkDestroyShaderModule(device_, module, nullptr);
    return result == VK_SUCCESS;
}

bool MlInference::CreateGroupNormApplyPipeline(uint32_t channels, uint32_t activation,
                                                VkPipeline& pipeline, VkPipelineLayout& layout,
                                                VkDescriptorSetLayout& ds_layout) {
    // Bindings: 0=data (read/write), 1=norm params (readonly), 2=reduction (readonly)
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo ds_ci{};
    ds_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ds_ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ds_ci.pBindings = bindings.data();
    VkResult result = dispatch_.vkCreateDescriptorSetLayout(device_, &ds_ci, nullptr, &ds_layout);
    if (result != VK_SUCCESS) return false;

    // Push constants: width, height, num_reduce_workgroups (3 uints)
    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.size = 3 * sizeof(uint32_t);

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &ds_layout;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &pc_range;
    result = dispatch_.vkCreatePipelineLayout(device_, &layout_ci, nullptr, &layout);
    if (result != VK_SUCCESS) return false;

    VkShaderModule module = LoadShaderModule("group_norm_apply.comp");
    if (module == VK_NULL_HANDLE) return false;

    // Spec constants: CHANNELS, NUM_GROUPS, ACTIVATION
    std::array<VkSpecializationMapEntry, 3> spec_entries{};
    spec_entries[0] = {0, 0, sizeof(uint32_t)};
    spec_entries[1] = {1, sizeof(uint32_t), sizeof(uint32_t)};
    spec_entries[2] = {2, 2 * sizeof(uint32_t), sizeof(uint32_t)};
    std::array<uint32_t, 3> spec_data = {channels, kNumGroups, activation};

    VkSpecializationInfo spec_info{};
    spec_info.mapEntryCount = static_cast<uint32_t>(spec_entries.size());
    spec_info.pMapEntries = spec_entries.data();
    spec_info.dataSize = sizeof(spec_data);
    spec_info.pData = spec_data.data();

    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_ci.stage.module = module;
    pipeline_ci.stage.pName = "main";
    pipeline_ci.stage.pSpecializationInfo = &spec_info;
    pipeline_ci.layout = layout;

    result = dispatch_.vkCreateComputePipelines(device_, pipeline_cache_, 1, &pipeline_ci,
                                                nullptr, &pipeline);
    dispatch_.vkDestroyShaderModule(device_, module, nullptr);
    return result == VK_SUCCESS;
}

bool MlInference::CreateEncoderInputConvPipeline() {
    // Bindings: 0-6 = G-buffer images (storage image), 7=output buffer, 8=weights buffer
    std::array<VkDescriptorSetLayoutBinding, 9> bindings{};
    // 7 storage images (noisy_d, noisy_s, normals, depth, motion, diffuse_albedo, specular_albedo)
    for (uint32_t i = 0; i < 7; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    // 2 storage buffers
    for (uint32_t i = 7; i < 9; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo ds_ci{};
    ds_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ds_ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ds_ci.pBindings = bindings.data();
    VkResult result = dispatch_.vkCreateDescriptorSetLayout(device_, &ds_ci, nullptr,
                                                            &encoder_input_ds_layout_);
    if (result != VK_SUCCESS) return false;

    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.size = sizeof(MlPushConstants);

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &encoder_input_ds_layout_;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &pc_range;
    result = dispatch_.vkCreatePipelineLayout(device_, &layout_ci, nullptr,
                                              &encoder_input_layout_);
    if (result != VK_SUCCESS) return false;

    VkShaderModule module = LoadShaderModule("encoder_input_conv.comp");
    if (module == VK_NULL_HANDLE) return false;

    // Spec constant: OUT_CHANNELS
    VkSpecializationMapEntry spec_entry = {0, 0, sizeof(uint32_t)};
    uint32_t out_ch = level0_channels_;

    VkSpecializationInfo spec_info{};
    spec_info.mapEntryCount = 1;
    spec_info.pMapEntries = &spec_entry;
    spec_info.dataSize = sizeof(uint32_t);
    spec_info.pData = &out_ch;

    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_ci.stage.module = module;
    pipeline_ci.stage.pName = "main";
    pipeline_ci.stage.pSpecializationInfo = &spec_info;
    pipeline_ci.layout = encoder_input_layout_;

    result = dispatch_.vkCreateComputePipelines(device_, pipeline_cache_, 1, &pipeline_ci,
                                                nullptr, &encoder_input_pipeline_);
    dispatch_.vkDestroyShaderModule(device_, module, nullptr);
    return result == VK_SUCCESS;
}

bool MlInference::CreateDownsamplePipeline(uint32_t channels) {
    // Bindings: 0=input buffer, 1=output buffer
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    for (uint32_t i = 0; i < 2; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    DownsamplePipelineSet ps;

    VkDescriptorSetLayoutCreateInfo ds_ci{};
    ds_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ds_ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ds_ci.pBindings = bindings.data();
    VkResult result = dispatch_.vkCreateDescriptorSetLayout(device_, &ds_ci, nullptr,
                                                            &ps.ds_layout);
    if (result != VK_SUCCESS) return false;

    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.size = sizeof(DownsamplePushConstants);

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &ps.ds_layout;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &pc_range;
    result = dispatch_.vkCreatePipelineLayout(device_, &layout_ci, nullptr, &ps.layout);
    if (result != VK_SUCCESS) return false;

    VkShaderModule module = LoadShaderModule("downsample.comp");
    if (module == VK_NULL_HANDLE) return false;

    VkSpecializationMapEntry spec_entry = {0, 0, sizeof(uint32_t)};
    VkSpecializationInfo spec_info{};
    spec_info.mapEntryCount = 1;
    spec_info.pMapEntries = &spec_entry;
    spec_info.dataSize = sizeof(uint32_t);
    spec_info.pData = &channels;

    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_ci.stage.module = module;
    pipeline_ci.stage.pName = "main";
    pipeline_ci.stage.pSpecializationInfo = &spec_info;
    pipeline_ci.layout = ps.layout;

    result = dispatch_.vkCreateComputePipelines(device_, pipeline_cache_, 1, &pipeline_ci,
                                                nullptr, &ps.pipeline);
    dispatch_.vkDestroyShaderModule(device_, module, nullptr);
    if (result != VK_SUCCESS) return false;

    downsample_pipelines_[channels] = ps;
    return true;
}

bool MlInference::CreateUpsampleConcatPipeline(uint32_t in_ch, uint32_t skip_ch) {
    // Bindings: 0=input, 1=skip, 2=output (all storage buffers)
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    UpsampleConcatPipelineSet ps;

    VkDescriptorSetLayoutCreateInfo ds_ci{};
    ds_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ds_ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ds_ci.pBindings = bindings.data();
    VkResult result = dispatch_.vkCreateDescriptorSetLayout(device_, &ds_ci, nullptr,
                                                            &ps.ds_layout);
    if (result != VK_SUCCESS) return false;

    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.size = sizeof(UpsampleConcatPushConstants);

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &ps.ds_layout;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &pc_range;
    result = dispatch_.vkCreatePipelineLayout(device_, &layout_ci, nullptr, &ps.layout);
    if (result != VK_SUCCESS) return false;

    VkShaderModule module = LoadShaderModule("upsample_concat.comp");
    if (module == VK_NULL_HANDLE) return false;

    std::array<VkSpecializationMapEntry, 2> spec_entries{};
    spec_entries[0] = {0, 0, sizeof(uint32_t)};
    spec_entries[1] = {1, sizeof(uint32_t), sizeof(uint32_t)};
    std::array<uint32_t, 2> spec_data = {in_ch, skip_ch};

    VkSpecializationInfo spec_info{};
    spec_info.mapEntryCount = static_cast<uint32_t>(spec_entries.size());
    spec_info.pMapEntries = spec_entries.data();
    spec_info.dataSize = sizeof(spec_data);
    spec_info.pData = spec_data.data();

    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_ci.stage.module = module;
    pipeline_ci.stage.pName = "main";
    pipeline_ci.stage.pSpecializationInfo = &spec_info;
    pipeline_ci.layout = ps.layout;

    result = dispatch_.vkCreateComputePipelines(device_, pipeline_cache_, 1, &pipeline_ci,
                                                nullptr, &ps.pipeline);
    dispatch_.vkDestroyShaderModule(device_, module, nullptr);
    if (result != VK_SUCCESS) return false;

    upsample_concat_pipelines_[{in_ch, skip_ch}] = ps;
    return true;
}

bool MlInference::CreateOutputConvPipeline() {
    // Bindings: 0=input buffer, 1=output image, 2=weights buffer,
    //           3=noisy_diffuse (hit mask), 4=diffuse_albedo, 5=specular_albedo
    std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // 3 additional storage images for remodulation
    for (uint32_t i = 3; i < 6; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo ds_ci{};
    ds_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ds_ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ds_ci.pBindings = bindings.data();
    VkResult result = dispatch_.vkCreateDescriptorSetLayout(device_, &ds_ci, nullptr,
                                                            &output_conv_ds_layout_);
    if (result != VK_SUCCESS) return false;

    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.size = sizeof(MlPushConstants);

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &output_conv_ds_layout_;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &pc_range;
    result = dispatch_.vkCreatePipelineLayout(device_, &layout_ci, nullptr,
                                              &output_conv_layout_);
    if (result != VK_SUCCESS) return false;

    VkShaderModule module = LoadShaderModule("output_conv.comp");
    if (module == VK_NULL_HANDLE) return false;

    VkSpecializationMapEntry spec_entry = {0, 0, sizeof(uint32_t)};
    uint32_t in_ch = level0_channels_;

    VkSpecializationInfo spec_info{};
    spec_info.mapEntryCount = 1;
    spec_info.pMapEntries = &spec_entry;
    spec_info.dataSize = sizeof(uint32_t);
    spec_info.pData = &in_ch;

    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_ci.stage.module = module;
    pipeline_ci.stage.pName = "main";
    pipeline_ci.stage.pSpecializationInfo = &spec_info;
    pipeline_ci.layout = output_conv_layout_;

    result = dispatch_.vkCreateComputePipelines(device_, pipeline_cache_, 1, &pipeline_ci,
                                                nullptr, &output_conv_pipeline_);
    dispatch_.vkDestroyShaderModule(device_, module, nullptr);
    return result == VK_SUCCESS;
}

bool MlInference::CreatePipelines() {
    DestroyPipelines();

    uint32_t c0 = level0_channels_;
    uint32_t c1 = level1_channels_;
    uint32_t c2 = level2_channels_;

    // Encoder input conv (G-buffer images → level0)
    if (!CreateEncoderInputConvPipeline()) return false;

    // Generic conv pipelines for all (in_ch → out_ch) combinations used
    // Encoder: down0.conv2 (c0→c0), down1.conv1 (c1→c1), down1.conv2 (c1→c1)
    // Bottleneck: bottleneck1 (c2→c2), bottleneck2 (c2→c2)
    // Decoder: up1.conv1 (c2+c1→c1), up1.conv2 (c1→c1),
    //          up0.conv1 (c1+c0→c0), up0.conv2 (c0→c0)
    struct ConvConfig { uint32_t in_ch, out_ch; };
    std::vector<ConvConfig> conv_configs = {
        {c0, c0}, {c0, c1},         // down0.conv1 is encoder_input, down0.conv2 is c0→c0, and first in down1 uses c0→c1? No...
    };

    // Actually: re-derive from architecture.
    // down0: conv1(13→c0) [encoder_input], conv2(c0→c0)
    // down1: conv1(c0→c1), conv2(c1→c1) — but input to down1 is after downsample, channels=c0
    // Actually down1.conv1 takes c0 channels (from downsample of level0) and outputs c1
    // bottleneck1(c1→c2) — after downsample of level1 (c1 channels) but the conv produces c2
    // Actually let me re-read the model architecture...
    // The training code: DownBlock has conv1 and conv2 where
    // down0 = DownBlock(13, c0) → conv1: 13→c0, conv2: c0→c0
    // down1 = DownBlock(c0, c1) → conv1: c0→c1, conv2: c1→c1
    // bottleneck1 = ConvBlock(c1, c2) → conv: c1→c2, norm: c2
    // bottleneck2 = ConvBlock(c2, c2) → conv: c2→c2, norm: c2
    // up1 = UpBlock(c2, c1, c1) → conv1: c2+c1→c1, conv2: c1→c1
    // up0 = UpBlock(c1, c0, c0) → conv1: c1+c0→c0, conv2: c0→c0
    // out_conv: c0→3

    // But down0.conv1 is handled by encoder_input_conv, and out_conv by output_conv.
    // Remaining generic convs needed:
    conv_configs.clear();
    conv_configs.push_back({c0, c0});           // down0.conv2
    conv_configs.push_back({c0, c1});           // down1.conv1
    conv_configs.push_back({c1, c1});           // down1.conv2
    conv_configs.push_back({c1, c2});           // bottleneck1
    conv_configs.push_back({c2, c2});           // bottleneck2
    conv_configs.push_back({c2 + c1, c1});      // up1.conv1
    conv_configs.push_back({c1 + c0, c0});      // up0.conv1

    // Deduplicate: {c1, c1} and {c2, c2} might be dupes of each other with different values.
    // The unordered_map handles deduplication automatically.
    for (const auto& cfg : conv_configs) {
        ConvPipelineKey key{cfg.in_ch, cfg.out_ch};
        if (conv_pipelines_.contains(key)) continue;
        PipelineSet ps;
        if (!CreateConvPipeline(cfg.in_ch, cfg.out_ch, ps.pipeline, ps.layout, ps.ds_layout))
            return false;
        conv_pipelines_[key] = ps;
    }

    // GroupNorm pipelines: need one reduce+apply pair per channel count with activation
    // All norms in encoder/decoder use LeakyReLU activation (activation=1)
    for (uint32_t ch : {c0, c1, c2}) {
        if (norm_pipelines_.contains(ch)) continue;
        GroupNormPipelineSet gps;
        if (!CreateGroupNormReducePipeline(ch, gps.reduce_pipeline, gps.reduce_layout,
                                           gps.reduce_ds_layout))
            return false;
        if (!CreateGroupNormApplyPipeline(ch, 1, gps.apply_pipeline, gps.apply_layout,
                                          gps.apply_ds_layout))
            return false;
        norm_pipelines_[ch] = gps;
    }

    // Downsample pipelines
    if (!CreateDownsamplePipeline(c0)) return false;  // level0 → level1
    if (!CreateDownsamplePipeline(c1)) return false;  // level1 → level2

    // Upsample-concat pipelines
    if (!CreateUpsampleConcatPipeline(c2, c1)) return false;  // level2 → level1
    if (!CreateUpsampleConcatPipeline(c1, c0)) return false;  // level1 → level0

    // Output conv
    if (!CreateOutputConvPipeline()) return false;

    pipelines_created_ = true;
    return true;
}

void MlInference::DestroyPipelines() {
    for (auto& [key, ps] : conv_pipelines_) {
        if (ps.pipeline != VK_NULL_HANDLE) dispatch_.vkDestroyPipeline(device_, ps.pipeline, nullptr);
        if (ps.layout != VK_NULL_HANDLE) dispatch_.vkDestroyPipelineLayout(device_, ps.layout, nullptr);
        if (ps.ds_layout != VK_NULL_HANDLE) dispatch_.vkDestroyDescriptorSetLayout(device_, ps.ds_layout, nullptr);
    }
    conv_pipelines_.clear();

    for (auto& [ch, gps] : norm_pipelines_) {
        if (gps.reduce_pipeline != VK_NULL_HANDLE) dispatch_.vkDestroyPipeline(device_, gps.reduce_pipeline, nullptr);
        if (gps.reduce_layout != VK_NULL_HANDLE) dispatch_.vkDestroyPipelineLayout(device_, gps.reduce_layout, nullptr);
        if (gps.reduce_ds_layout != VK_NULL_HANDLE) dispatch_.vkDestroyDescriptorSetLayout(device_, gps.reduce_ds_layout, nullptr);
        if (gps.apply_pipeline != VK_NULL_HANDLE) dispatch_.vkDestroyPipeline(device_, gps.apply_pipeline, nullptr);
        if (gps.apply_layout != VK_NULL_HANDLE) dispatch_.vkDestroyPipelineLayout(device_, gps.apply_layout, nullptr);
        if (gps.apply_ds_layout != VK_NULL_HANDLE) dispatch_.vkDestroyDescriptorSetLayout(device_, gps.apply_ds_layout, nullptr);
    }
    norm_pipelines_.clear();

    for (auto& [ch, ps] : downsample_pipelines_) {
        if (ps.pipeline != VK_NULL_HANDLE) dispatch_.vkDestroyPipeline(device_, ps.pipeline, nullptr);
        if (ps.layout != VK_NULL_HANDLE) dispatch_.vkDestroyPipelineLayout(device_, ps.layout, nullptr);
        if (ps.ds_layout != VK_NULL_HANDLE) dispatch_.vkDestroyDescriptorSetLayout(device_, ps.ds_layout, nullptr);
    }
    downsample_pipelines_.clear();

    for (auto& [key, ps] : upsample_concat_pipelines_) {
        if (ps.pipeline != VK_NULL_HANDLE) dispatch_.vkDestroyPipeline(device_, ps.pipeline, nullptr);
        if (ps.layout != VK_NULL_HANDLE) dispatch_.vkDestroyPipelineLayout(device_, ps.layout, nullptr);
        if (ps.ds_layout != VK_NULL_HANDLE) dispatch_.vkDestroyDescriptorSetLayout(device_, ps.ds_layout, nullptr);
    }
    upsample_concat_pipelines_.clear();

    if (encoder_input_pipeline_ != VK_NULL_HANDLE) {
        dispatch_.vkDestroyPipeline(device_, encoder_input_pipeline_, nullptr);
        encoder_input_pipeline_ = VK_NULL_HANDLE;
    }
    if (encoder_input_layout_ != VK_NULL_HANDLE) {
        dispatch_.vkDestroyPipelineLayout(device_, encoder_input_layout_, nullptr);
        encoder_input_layout_ = VK_NULL_HANDLE;
    }
    if (encoder_input_ds_layout_ != VK_NULL_HANDLE) {
        dispatch_.vkDestroyDescriptorSetLayout(device_, encoder_input_ds_layout_, nullptr);
        encoder_input_ds_layout_ = VK_NULL_HANDLE;
    }

    if (output_conv_pipeline_ != VK_NULL_HANDLE) {
        dispatch_.vkDestroyPipeline(device_, output_conv_pipeline_, nullptr);
        output_conv_pipeline_ = VK_NULL_HANDLE;
    }
    if (output_conv_layout_ != VK_NULL_HANDLE) {
        dispatch_.vkDestroyPipelineLayout(device_, output_conv_layout_, nullptr);
        output_conv_layout_ = VK_NULL_HANDLE;
    }
    if (output_conv_ds_layout_ != VK_NULL_HANDLE) {
        dispatch_.vkDestroyDescriptorSetLayout(device_, output_conv_ds_layout_, nullptr);
        output_conv_ds_layout_ = VK_NULL_HANDLE;
    }

    pipelines_created_ = false;
}

// ---------------------------------------------------------------------------
// Descriptor pool management
// ---------------------------------------------------------------------------

bool MlInference::CreateDescriptorPool() {
    DestroyDescriptorPool();

    // Count total descriptors needed across all dispatch steps.
    // Conservative upper bound: each Infer() call uses at most ~40 descriptor sets
    // with up to 3 storage buffers + 5 storage images each.
    constexpr uint32_t kMaxSets = 64;
    std::array<VkDescriptorPoolSize, 2> pool_sizes{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[0].descriptorCount = kMaxSets * 4;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[1].descriptorCount = kMaxSets * 6;

    VkDescriptorPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets = kMaxSets;
    pool_ci.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_ci.pPoolSizes = pool_sizes.data();

    VkResult result = dispatch_.vkCreateDescriptorPool(device_, &pool_ci, nullptr,
                                                       &ml_descriptor_pool_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr,
                     "deni::MlInference: failed to create descriptor pool (VkResult: %d)\n",
                     result);
        return false;
    }
    return true;
}

void MlInference::DestroyDescriptorPool() {
    if (ml_descriptor_pool_ != VK_NULL_HANDLE) {
        dispatch_.vkDestroyDescriptorPool(device_, ml_descriptor_pool_, nullptr);
        ml_descriptor_pool_ = VK_NULL_HANDLE;
    }
    encoder_input_ds_ = VK_NULL_HANDLE;
    output_conv_ds_ = VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------------
// GPU timestamp query pool
// ---------------------------------------------------------------------------

bool MlInference::CreateQueryPool() {
    VkQueryPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    pool_ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    pool_ci.queryCount = kTimestampCount;

    VkResult result = dispatch_.vkCreateQueryPool(device_, &pool_ci, nullptr, &query_pool_);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "deni::MlInference: failed to create query pool (VkResult: %d)\n",
                     result);
        return false;
    }
    return true;
}

void MlInference::DestroyQueryPool() {
    if (query_pool_ != VK_NULL_HANDLE) {
        dispatch_.vkDestroyQueryPool(device_, query_pool_, nullptr);
        query_pool_ = VK_NULL_HANDLE;
    }
}

void MlInference::ReadbackTimestamps() {
    if (query_pool_ == VK_NULL_HANDLE || timestamp_period_ == 0.0f) return;
    if (!timestamps_valid_) return;

    std::array<uint64_t, kTimestampCount> timestamps{};
    VkResult result = dispatch_.vkGetQueryPoolResults(
        device_, query_pool_, 0, kTimestampCount,
        sizeof(timestamps), timestamps.data(), sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (result == VK_SUCCESS) {
        uint64_t delta = timestamps[1] - timestamps[0];
        gpu_time_ms_ = static_cast<float>(delta) * timestamp_period_ / 1e6f;
    }
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

void MlInference::InsertBufferBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    dispatch_.vkCmdPipelineBarrier2(cmd, &dep);
}

namespace {

VkDescriptorSet AllocateOneDescriptorSet(const MlDeviceDispatch& dispatch, VkDevice device,
                                          VkDescriptorPool pool,
                                          VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &layout;

    VkDescriptorSet ds = VK_NULL_HANDLE;
    dispatch.vkAllocateDescriptorSets(device, &alloc_info, &ds);
    return ds;
}

void WriteBufferDescriptor(const MlDeviceDispatch& dispatch, VkDevice device,
                            VkDescriptorSet ds, uint32_t binding, VkBuffer buffer,
                            VkDeviceSize size) {
    VkDescriptorBufferInfo buf_info{};
    buf_info.buffer = buffer;
    buf_info.range = size;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = ds;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buf_info;
    dispatch.vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void WriteImageDescriptor(const MlDeviceDispatch& dispatch, VkDevice device,
                           VkDescriptorSet ds, uint32_t binding, VkImageView view) {
    VkDescriptorImageInfo img_info{};
    img_info.imageView = view;
    img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = ds;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &img_info;
    dispatch.vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

}  // namespace

void MlInference::DispatchConv(VkCommandBuffer cmd, VkBuffer input, VkBuffer output,
                                std::string_view weight_name, uint32_t in_ch, uint32_t out_ch,
                                uint32_t width, uint32_t height) {
    ConvPipelineKey key{in_ch, out_ch};
    auto it = conv_pipelines_.find(key);
    if (it == conv_pipelines_.end()) return;
    const auto& ps = it->second;

    VkBuffer weight_buf = FindWeightBuffer(weight_name);
    if (weight_buf == VK_NULL_HANDLE) return;

    VkDescriptorSet ds = AllocateOneDescriptorSet(dispatch_, device_, ml_descriptor_pool_,
                                                   ps.ds_layout);
    VkDeviceSize in_size = static_cast<VkDeviceSize>(in_ch) * width * height * sizeof(uint16_t);
    VkDeviceSize out_size = static_cast<VkDeviceSize>(out_ch) * width * height * sizeof(uint16_t);
    // weights: [out_ch][in_ch][3][3] + bias[out_ch]
    VkDeviceSize weight_size = static_cast<VkDeviceSize>(out_ch) * in_ch * 9 * sizeof(float)
                               + out_ch * sizeof(float);
    WriteBufferDescriptor(dispatch_, device_, ds, 0, input, in_size);
    WriteBufferDescriptor(dispatch_, device_, ds, 1, output, out_size);
    WriteBufferDescriptor(dispatch_, device_, ds, 2, weight_buf, weight_size);

    MlPushConstants pc{width, height};
    dispatch_.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps.pipeline);
    dispatch_.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps.layout,
                                      0, 1, &ds, 0, nullptr);
    dispatch_.vkCmdPushConstants(cmd, ps.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 0, sizeof(pc), &pc);
    dispatch_.vkCmdDispatch(cmd, DivCeil(width, kWorkgroupSize),
                            DivCeil(height, kWorkgroupSize), 1);
    InsertBufferBarrier(cmd);
}

void MlInference::DispatchGroupNorm(VkCommandBuffer cmd, VkBuffer data,
                                     std::string_view norm_name,
                                     uint32_t channels, uint32_t width, uint32_t height) {
    auto it = norm_pipelines_.find(channels);
    if (it == norm_pipelines_.end()) return;
    const auto& gps = it->second;

    VkBuffer norm_buf = FindWeightBuffer(norm_name);
    if (norm_buf == VK_NULL_HANDLE) return;

    VkDeviceSize data_size = static_cast<VkDeviceSize>(channels) * width * height * sizeof(uint16_t);
    // norm params: gamma[channels] + beta[channels]
    VkDeviceSize norm_size = channels * 2 * sizeof(float);

    uint32_t channels_per_group = channels / kNumGroups;
    uint32_t elements_per_group = channels_per_group * width * height;
    uint32_t num_reduce_wg = DivCeil(elements_per_group, kReduceWorkgroupSize);

    // Pass 1: reduce
    {
        VkDescriptorSet ds = AllocateOneDescriptorSet(dispatch_, device_, ml_descriptor_pool_,
                                                       gps.reduce_ds_layout);
        WriteBufferDescriptor(dispatch_, device_, ds, 0, data, data_size);
        WriteBufferDescriptor(dispatch_, device_, ds, 1, reduction_buffer_,
                              reduction_buffer_size_);

        MlPushConstants pc{width, height};
        dispatch_.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gps.reduce_pipeline);
        dispatch_.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          gps.reduce_layout, 0, 1, &ds, 0, nullptr);
        dispatch_.vkCmdPushConstants(cmd, gps.reduce_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                     0, sizeof(pc), &pc);
        // Dispatch: X = num_reduce_wg tiles, Y = NUM_GROUPS
        dispatch_.vkCmdDispatch(cmd, num_reduce_wg, kNumGroups, 1);
        InsertBufferBarrier(cmd);
    }

    // Pass 2: apply
    {
        VkDescriptorSet ds = AllocateOneDescriptorSet(dispatch_, device_, ml_descriptor_pool_,
                                                       gps.apply_ds_layout);
        WriteBufferDescriptor(dispatch_, device_, ds, 0, data, data_size);
        WriteBufferDescriptor(dispatch_, device_, ds, 1, norm_buf, norm_size);
        WriteBufferDescriptor(dispatch_, device_, ds, 2, reduction_buffer_,
                              reduction_buffer_size_);

        struct { uint32_t w, h, nrwg; } pc{width, height, num_reduce_wg};
        dispatch_.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gps.apply_pipeline);
        dispatch_.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          gps.apply_layout, 0, 1, &ds, 0, nullptr);
        dispatch_.vkCmdPushConstants(cmd, gps.apply_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                     0, sizeof(pc), &pc);
        dispatch_.vkCmdDispatch(cmd, DivCeil(width, kWorkgroupSize),
                                DivCeil(height, kWorkgroupSize), 1);
        InsertBufferBarrier(cmd);
    }
}

void MlInference::DispatchDownsample(VkCommandBuffer cmd, VkBuffer input, VkBuffer output,
                                      uint32_t channels, uint32_t in_w, uint32_t in_h) {
    auto it = downsample_pipelines_.find(channels);
    if (it == downsample_pipelines_.end()) return;
    const auto& ps = it->second;

    uint32_t out_w = DivCeil(in_w, 2);
    uint32_t out_h = DivCeil(in_h, 2);

    VkDeviceSize in_size = static_cast<VkDeviceSize>(channels) * in_w * in_h * sizeof(uint16_t);
    VkDeviceSize out_size = static_cast<VkDeviceSize>(channels) * out_w * out_h * sizeof(uint16_t);

    VkDescriptorSet ds = AllocateOneDescriptorSet(dispatch_, device_, ml_descriptor_pool_,
                                                   ps.ds_layout);
    WriteBufferDescriptor(dispatch_, device_, ds, 0, input, in_size);
    WriteBufferDescriptor(dispatch_, device_, ds, 1, output, out_size);

    DownsamplePushConstants pc{in_w, in_h};
    dispatch_.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps.pipeline);
    dispatch_.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps.layout,
                                      0, 1, &ds, 0, nullptr);
    dispatch_.vkCmdPushConstants(cmd, ps.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 0, sizeof(pc), &pc);
    dispatch_.vkCmdDispatch(cmd, DivCeil(out_w, kWorkgroupSize),
                            DivCeil(out_h, kWorkgroupSize), 1);
    InsertBufferBarrier(cmd);
}

void MlInference::DispatchUpsampleConcat(VkCommandBuffer cmd, VkBuffer input, VkBuffer skip,
                                          VkBuffer output, uint32_t in_ch, uint32_t skip_ch,
                                          uint32_t out_w, uint32_t out_h) {
    ConvPipelineKey key{in_ch, skip_ch};
    auto it = upsample_concat_pipelines_.find(key);
    if (it == upsample_concat_pipelines_.end()) return;
    const auto& ps = it->second;

    uint32_t in_w = DivCeil(out_w, 2);
    uint32_t in_h = DivCeil(out_h, 2);

    VkDeviceSize in_size = static_cast<VkDeviceSize>(in_ch) * in_w * in_h * sizeof(uint16_t);
    VkDeviceSize skip_size = static_cast<VkDeviceSize>(skip_ch) * out_w * out_h * sizeof(uint16_t);
    VkDeviceSize out_size = static_cast<VkDeviceSize>(in_ch + skip_ch) * out_w * out_h * sizeof(uint16_t);

    VkDescriptorSet ds = AllocateOneDescriptorSet(dispatch_, device_, ml_descriptor_pool_,
                                                   ps.ds_layout);
    WriteBufferDescriptor(dispatch_, device_, ds, 0, input, in_size);
    WriteBufferDescriptor(dispatch_, device_, ds, 1, skip, skip_size);
    WriteBufferDescriptor(dispatch_, device_, ds, 2, output, out_size);

    UpsampleConcatPushConstants pc{out_w, out_h};
    dispatch_.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps.pipeline);
    dispatch_.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps.layout,
                                      0, 1, &ds, 0, nullptr);
    dispatch_.vkCmdPushConstants(cmd, ps.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 0, sizeof(pc), &pc);
    dispatch_.vkCmdDispatch(cmd, DivCeil(out_w, kWorkgroupSize),
                            DivCeil(out_h, kWorkgroupSize), 1);
    InsertBufferBarrier(cmd);
}

// ---------------------------------------------------------------------------
// Infer — Full U-Net dispatch sequence
// ---------------------------------------------------------------------------

void MlInference::Infer(VkCommandBuffer cmd, const DenoiserInput& input,
                         VkImageView output_view) {
    if (!IsReady()) return;

    // Reset the pre-allocated descriptor pool (no destroy/recreate overhead)
    if (ml_descriptor_pool_ != VK_NULL_HANDLE)
        dispatch_.vkResetDescriptorPool(device_, ml_descriptor_pool_, 0);

    // GPU timestamp: begin
    if (query_pool_ != VK_NULL_HANDLE) {
        dispatch_.vkCmdResetQueryPool(cmd, query_pool_, 0, kTimestampCount);
        dispatch_.vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                       query_pool_, 0);
    }

    uint32_t c0 = level0_channels_;
    uint32_t c1 = level1_channels_;
    uint32_t c2 = level2_channels_;

    uint32_t w0 = width_, h0 = height_;
    uint32_t w1 = DivCeil(w0, 2), h1 = DivCeil(h0, 2);
    uint32_t w2 = DivCeil(w1, 2), h2 = DivCeil(h1, 2);

    // ------ Encoder level 0 ------
    // down0.conv1: G-buffer images (19ch) → buf0_a (c0 channels)
    {
        VkDescriptorSet ds = AllocateOneDescriptorSet(dispatch_, device_, ml_descriptor_pool_,
                                                       encoder_input_ds_layout_);
        // Bind G-buffer images (bindings 0-6)
        WriteImageDescriptor(dispatch_, device_, ds, 0, input.noisy_diffuse);
        WriteImageDescriptor(dispatch_, device_, ds, 1, input.noisy_specular);
        WriteImageDescriptor(dispatch_, device_, ds, 2, input.world_normals);
        WriteImageDescriptor(dispatch_, device_, ds, 3, input.linear_depth);
        WriteImageDescriptor(dispatch_, device_, ds, 4, input.motion_vectors);
        WriteImageDescriptor(dispatch_, device_, ds, 5, input.diffuse_albedo);
        WriteImageDescriptor(dispatch_, device_, ds, 6, input.specular_albedo);
        // Output buffer (binding 7)
        WriteBufferDescriptor(dispatch_, device_, ds, 7, buf0_a_.buffer, buf0_a_.size_bytes);
        // Weights (binding 8)
        VkBuffer enc_weights = FindWeightBuffer("down0.conv1.conv");
        if (enc_weights == VK_NULL_HANDLE) return;
        VkDeviceSize enc_w_size = static_cast<VkDeviceSize>(c0) * kInputChannels * 9 * sizeof(float)
                                  + c0 * sizeof(float);
        WriteBufferDescriptor(dispatch_, device_, ds, 8, enc_weights, enc_w_size);

        MlPushConstants pc{w0, h0};
        dispatch_.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, encoder_input_pipeline_);
        dispatch_.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          encoder_input_layout_, 0, 1, &ds, 0, nullptr);
        dispatch_.vkCmdPushConstants(cmd, encoder_input_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                                     0, sizeof(pc), &pc);
        dispatch_.vkCmdDispatch(cmd, DivCeil(w0, kWorkgroupSize),
                                DivCeil(h0, kWorkgroupSize), 1);
        InsertBufferBarrier(cmd);
    }

    // down0.conv1.norm + activation → buf0_a (in-place)
    DispatchGroupNorm(cmd, buf0_a_.buffer, "down0.conv1.norm", c0, w0, h0);

    // down0.conv2: buf0_a → buf0_b
    DispatchConv(cmd, buf0_a_.buffer, buf0_b_.buffer, "down0.conv2.conv", c0, c0, w0, h0);
    DispatchGroupNorm(cmd, buf0_b_.buffer, "down0.conv2.norm", c0, w0, h0);

    // Save skip0 = buf0_b (copy) — barrier deferred to after downsample (no RAW hazard:
    // downsample reads buf0_b, copy also reads buf0_b; skip0 is not read until decoder)
    {
        VkBufferCopy copy{};
        copy.size = buf0_b_.size_bytes;
        dispatch_.vkCmdCopyBuffer(cmd, buf0_b_.buffer, skip0_.buffer, 1, &copy);
    }

    // ------ Downsample level 0 → level 1 ------
    // buf0_b is only read by both copy and downsample — no WAR hazard between them.
    // But downsample writes buf1_a, so we need a barrier after downsample.
    DispatchDownsample(cmd, buf0_b_.buffer, buf1_a_.buffer, c0, w0, h0);

    // ------ Encoder level 1 ------
    // down1.conv1: buf1_a (c0) → buf1_b (c1)
    DispatchConv(cmd, buf1_a_.buffer, buf1_b_.buffer, "down1.conv1.conv", c0, c1, w1, h1);
    DispatchGroupNorm(cmd, buf1_b_.buffer, "down1.conv1.norm", c1, w1, h1);

    // down1.conv2: buf1_b → buf1_a (c1→c1)
    DispatchConv(cmd, buf1_b_.buffer, buf1_a_.buffer, "down1.conv2.conv", c1, c1, w1, h1);
    DispatchGroupNorm(cmd, buf1_a_.buffer, "down1.conv2.norm", c1, w1, h1);

    // Save skip1 = buf1_a (same pattern: no barrier needed before downsample)
    {
        VkBufferCopy copy{};
        copy.size = buf1_a_.size_bytes;
        dispatch_.vkCmdCopyBuffer(cmd, buf1_a_.buffer, skip1_.buffer, 1, &copy);
    }

    // ------ Downsample level 1 → level 2 ------
    DispatchDownsample(cmd, buf1_a_.buffer, buf2_a_.buffer, c1, w1, h1);

    // ------ Bottleneck ------
    // bottleneck1: buf2_a (c1) → buf2_b (c2)
    DispatchConv(cmd, buf2_a_.buffer, buf2_b_.buffer, "bottleneck1.conv", c1, c2, w2, h2);
    DispatchGroupNorm(cmd, buf2_b_.buffer, "bottleneck1.norm", c2, w2, h2);

    // bottleneck2: buf2_b → buf2_a (c2→c2)
    DispatchConv(cmd, buf2_b_.buffer, buf2_a_.buffer, "bottleneck2.conv", c2, c2, w2, h2);
    DispatchGroupNorm(cmd, buf2_a_.buffer, "bottleneck2.norm", c2, w2, h2);

    // ------ Decoder level 1 ------
    // Upsample buf2_a (c2) + concat skip1 (c1) → concat1 (c2+c1 channels, level1 resolution)
    DispatchUpsampleConcat(cmd, buf2_a_.buffer, skip1_.buffer, concat1_.buffer, c2, c1, w1, h1);

    // up1.conv1: concat1 (c2+c1) → buf1_b (c1)
    DispatchConv(cmd, concat1_.buffer, buf1_b_.buffer, "up1.conv1.conv", c2 + c1, c1, w1, h1);
    DispatchGroupNorm(cmd, buf1_b_.buffer, "up1.conv1.norm", c1, w1, h1);

    // up1.conv2: buf1_b → buf1_a (c1→c1)
    DispatchConv(cmd, buf1_b_.buffer, buf1_a_.buffer, "up1.conv2.conv", c1, c1, w1, h1);
    DispatchGroupNorm(cmd, buf1_a_.buffer, "up1.conv2.norm", c1, w1, h1);

    // ------ Decoder level 0 ------
    // Upsample buf1_a (c1) + concat skip0 (c0) → concat0 (c1+c0 channels, level0 resolution)
    DispatchUpsampleConcat(cmd, buf1_a_.buffer, skip0_.buffer, concat0_.buffer, c1, c0, w0, h0);

    // up0.conv1: concat0 (c1+c0) → buf0_b (c0)
    DispatchConv(cmd, concat0_.buffer, buf0_b_.buffer, "up0.conv1.conv", c1 + c0, c0, w0, h0);
    DispatchGroupNorm(cmd, buf0_b_.buffer, "up0.conv1.norm", c0, w0, h0);

    // up0.conv2: buf0_b → buf0_a (c0→c0)
    DispatchConv(cmd, buf0_b_.buffer, buf0_a_.buffer, "up0.conv2.conv", c0, c0, w0, h0);
    DispatchGroupNorm(cmd, buf0_a_.buffer, "up0.conv2.norm", c0, w0, h0);

    // ------ Output convolution ------
    // out_conv: buf0_a (c0) → output image (6 channels irradiance, remodulated to RGB)
    {
        VkDescriptorSet ds = AllocateOneDescriptorSet(dispatch_, device_, ml_descriptor_pool_,
                                                       output_conv_ds_layout_);
        WriteBufferDescriptor(dispatch_, device_, ds, 0, buf0_a_.buffer, buf0_a_.size_bytes);
        WriteImageDescriptor(dispatch_, device_, ds, 1, output_view);
        VkBuffer out_weights = FindWeightBuffer("out_conv");
        if (out_weights == VK_NULL_HANDLE) return;
        VkDeviceSize out_w_size = static_cast<VkDeviceSize>(kOutputChannels) * c0 * sizeof(float)
                                  + kOutputChannels * sizeof(float);
        WriteBufferDescriptor(dispatch_, device_, ds, 2, out_weights, out_w_size);
        // Remodulation inputs (bindings 3-5)
        WriteImageDescriptor(dispatch_, device_, ds, 3, input.noisy_diffuse);
        WriteImageDescriptor(dispatch_, device_, ds, 4, input.diffuse_albedo);
        WriteImageDescriptor(dispatch_, device_, ds, 5, input.specular_albedo);

        MlPushConstants pc{w0, h0};
        dispatch_.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, output_conv_pipeline_);
        dispatch_.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          output_conv_layout_, 0, 1, &ds, 0, nullptr);
        dispatch_.vkCmdPushConstants(cmd, output_conv_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                                     0, sizeof(pc), &pc);
        dispatch_.vkCmdDispatch(cmd, DivCeil(w0, kWorkgroupSize),
                                DivCeil(h0, kWorkgroupSize), 1);
    }

    // GPU timestamp: end
    if (query_pool_ != VK_NULL_HANDLE) {
        dispatch_.vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                                       query_pool_, 1);
        timestamps_valid_ = true;
    }
}

}  // namespace deni::vulkan
