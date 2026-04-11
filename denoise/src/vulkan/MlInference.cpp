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

    resolve(vkCmdCopyImage,              "vkCmdCopyImage");
    resolve(vkCmdClearColorImage,         "vkCmdClearColorImage");

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
    DestroyHistoryImages();
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
                // Weight with no matching bias (e.g., depthwise conv, bias=False).
                // Use base_name (without .weight suffix) for consistent lookup.
                CombinedLayer cl;
                cl.name = base_name;
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
    // Detect v3 temporal model from weight layer names
    bool has_v3_key = false;

    for (const auto& layer : weights.layers) {
        if (layer.name == "down0.conv1.depthwise.weight" && layer.shape.size() == 4)
            has_v3_key = true;
    }

    if (!has_v3_key) {
        std::fprintf(stderr, "deni::MlInference: could not find expected weight key "
                     "(down0.conv1.depthwise.weight) — only v3 temporal models are supported\n");
        return false;
    }

    for (const auto& layer : weights.layers) {
        if (layer.name == "down0.conv1.depthwise.weight" && layer.shape.size() == 4) {
            if (layer.shape[0] != kV3InputChannels) {
                std::fprintf(stderr,
                             "deni::MlInference: v3 model has %u input channels but "
                             "shaders expect %u\n",
                             layer.shape[0], kV3InputChannels);
                return false;
            }
        }
        if (layer.name == "out_conv.weight" && layer.shape.size() == 4) {
            if (layer.shape[0] != kV3OutputChannels) {
                std::fprintf(stderr,
                             "deni::MlInference: v3 model has %u output channels but "
                             "shaders expect %u\n",
                             layer.shape[0], kV3OutputChannels);
                return false;
            }
        }
    }

    return true;
}

bool MlInference::InferArchitectureFromWeights(const WeightData& weights) {
    if (!ValidateWeights(weights)) return false;

    model_version_ = ModelVersion::kV3_Temporal;
    // base_channels = shape[0] of "down0.conv1.pointwise.weight"
    for (const auto& layer : weights.layers) {
        if (layer.name == "down0.conv1.pointwise.weight" && layer.shape.size() == 4) {
            level0_channels_ = layer.shape[0];
            level1_channels_ = level0_channels_ * 2;
            break;
        }
    }
    std::fprintf(stderr,
                 "deni::MlInference: v3 temporal model, base_channels=%u "
                 "(levels: %u, %u)\n",
                 level0_channels_, level0_channels_, level1_channels_);

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
    DestroyHistoryImages();

    width_ = width;
    height_ = height;

    uint32_t w0 = width, h0 = height;
    uint32_t w1 = DivCeil(w0, 2), h1 = DivCeil(h0, 2);

    // Level 0 buffers (full resolution)
    // For v3, buf0_a_ serves as the intermediate between depthwise and pointwise
    // convolutions. The depthwise conv preserves channel count, so buf0_a_ must
    // hold max(input_channels, c1+c0) to accommodate down0.conv1 (26ch) and
    // up0.conv1 (c1+c0 ch) depthwise intermediates.
    uint32_t buf0a_ch = std::max({kV3InputChannels, level1_channels_ + level0_channels_,
                                  level0_channels_});
    if (!AllocateFeatureBuffer(buf0_a_, buf0a_ch, w0, h0)) return false;
    if (!AllocateFeatureBuffer(buf0_b_, level0_channels_, w0, h0)) return false;
    if (!AllocateFeatureBuffer(skip0_, level0_channels_, w0, h0)) return false;

    // Level 1 buffers (half resolution)
    if (!AllocateFeatureBuffer(buf1_a_, level1_channels_, w1, h1)) return false;
    if (!AllocateFeatureBuffer(buf1_b_, level1_channels_, w1, h1)) return false;

    // 26-channel input gather buffer
    if (!AllocateFeatureBuffer(buf_input_, kV3InputChannels, w0, h0)) return false;

    // Concat scratch for decoder: upsample (c1) + skip0 (c0)
    uint32_t concat0_ch = level1_channels_ + level0_channels_;
    if (!AllocateFeatureBuffer(concat0_, concat0_ch, w0, h0)) return false;

    // Reduction buffer for GroupNorm — sized for the largest dispatch across
    // all channel counts × resolutions.  NumGroups(channels) may differ per level.
    VkDeviceSize max_reduction_bytes = 0;
    auto update_max_reduction = [&](uint32_t ch, uint32_t w, uint32_t h) {
        uint32_t ng  = NumGroups(ch);
        uint32_t cpg = ch / ng;
        uint32_t nrwg = DivCeil(cpg * w * h, kReduceWorkgroupSize);
        VkDeviceSize bytes = static_cast<VkDeviceSize>(ng) * nrwg * 2 * sizeof(float);
        max_reduction_bytes = std::max(max_reduction_bytes, bytes);
    };
    update_max_reduction(level0_channels_, w0, h0);
    update_max_reduction(level1_channels_, w1, h1);
    if (!AllocateReductionBuffer(max_reduction_bytes)) return false;

    // Temporal reprojection history images (full resolution)
    if (!AllocateHistoryImage(frame_history_.denoised_diffuse, VK_FORMAT_R16G16B16A16_SFLOAT, w0, h0)) return false;
    if (!AllocateHistoryImage(frame_history_.denoised_specular, VK_FORMAT_R16G16B16A16_SFLOAT, w0, h0)) return false;
    if (!AllocateHistoryImage(frame_history_.reprojected_diffuse, VK_FORMAT_R16G16B16A16_SFLOAT, w0, h0)) return false;
    if (!AllocateHistoryImage(frame_history_.reprojected_specular, VK_FORMAT_R16G16B16A16_SFLOAT, w0, h0)) return false;
    if (!AllocateHistoryImage(frame_history_.disocclusion_mask, VK_FORMAT_R16_SFLOAT, w0, h0)) return false;
    if (!AllocateHistoryImage(frame_history_.prev_depth, VK_FORMAT_R16G16_SFLOAT, w0, h0)) return false;
    frame_history_.valid = false;

    // Update push constants
    pc_level0_ = {w0, h0};
    pc_level1_ = {w1, h1};
    pc_down0_ = {w0, h0};
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
    DestroyFeatureBuffer(concat0_);
    DestroyFeatureBuffer(buf_input_);
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

bool MlInference::AllocateReductionBuffer(VkDeviceSize size) {
    DestroyReductionBuffer();

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
// History image management (temporal reprojection)
// ---------------------------------------------------------------------------

bool MlInference::AllocateHistoryImage(HistoryImage& img, VkFormat format,
                                        uint32_t width, uint32_t height) {
    VkImageCreateInfo image_ci{};
    image_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_ci.imageType = VK_IMAGE_TYPE_2D;
    image_ci.format = format;
    image_ci.extent = {width, height, 1};
    image_ci.mipLevels = 1;
    image_ci.arrayLayers = 1;
    image_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult result = vmaCreateImage(allocator_, &image_ci, &alloc_ci,
                                     &img.image, &img.allocation, nullptr);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr,
                     "deni::MlInference: failed to create history image %ux%u (VkResult: %d)\n",
                     width, height, result);
        return false;
    }

    VkImageViewCreateInfo view_ci{};
    view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image = img.image;
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format = format;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    result = dispatch_.vkCreateImageView(device_, &view_ci, nullptr, &img.view);
    if (result != VK_SUCCESS) {
        vmaDestroyImage(allocator_, img.image, img.allocation);
        img = {};
        return false;
    }

    img.width = width;
    img.height = height;
    return true;
}

void MlInference::DestroyHistoryImage(HistoryImage& img) {
    if (img.view != VK_NULL_HANDLE) {
        dispatch_.vkDestroyImageView(device_, img.view, nullptr);
    }
    if (img.image != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, img.image, img.allocation);
    }
    img = {};
}

void MlInference::DestroyHistoryImages() {
    DestroyHistoryImage(frame_history_.denoised_diffuse);
    DestroyHistoryImage(frame_history_.denoised_specular);
    DestroyHistoryImage(frame_history_.reprojected_diffuse);
    DestroyHistoryImage(frame_history_.reprojected_specular);
    DestroyHistoryImage(frame_history_.disocclusion_mask);
    DestroyHistoryImage(frame_history_.prev_depth);
    frame_history_.valid = false;
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
    std::array<uint32_t, 2> spec_data = {channels, NumGroups(channels)};

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
    std::array<uint32_t, 3> spec_data = {channels, NumGroups(channels), activation};

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

bool MlInference::CreateReprojectPipeline() {
    // Bindings: 0=motion_vectors(rg16f), 1=prev_diffuse(rgba16f), 2=prev_specular(rgba16f),
    //           3=prev_depth(rg16f), 4=curr_depth(rg16f),
    //           5=reprojected_d(rgba16f), 6=reprojected_s(rgba16f), 7=disocclusion(r16f)
    // All are storage images.
    std::array<VkDescriptorSetLayoutBinding, 8> bindings{};
    for (uint32_t i = 0; i < 8; ++i) {
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
                                                            &reproject_ds_layout_);
    if (result != VK_SUCCESS) return false;

    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.size = sizeof(MlPushConstants);

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &reproject_ds_layout_;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &pc_range;
    result = dispatch_.vkCreatePipelineLayout(device_, &layout_ci, nullptr, &reproject_layout_);
    if (result != VK_SUCCESS) return false;

    VkShaderModule module = LoadShaderModule("reproject.comp");
    if (module == VK_NULL_HANDLE) return false;

    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_ci.stage.module = module;
    pipeline_ci.stage.pName = "main";
    pipeline_ci.layout = reproject_layout_;

    result = dispatch_.vkCreateComputePipelines(device_, pipeline_cache_, 1, &pipeline_ci,
                                                nullptr, &reproject_pipeline_);
    dispatch_.vkDestroyShaderModule(device_, module, nullptr);
    return result == VK_SUCCESS;
}

bool MlInference::CreatePipelines() {
    DestroyPipelines();

    uint32_t c0 = level0_channels_;
    uint32_t c1 = level1_channels_;

    // Temporal reprojection
    if (!CreateReprojectPipeline()) return false;

    // Temporal input gather (10 images → 26ch flat buffer)
    if (!CreateTemporalInputGatherPipeline()) return false;

    // Depthwise conv pipelines: one per unique channel count
    // down0.conv1.depthwise: 26ch, down0.conv2.depthwise: c0,
    // bottleneck1.depthwise: c0, bottleneck2.depthwise: c1,
    // up0.conv1.depthwise: c1+c0, up0.conv2.depthwise: c0
    for (uint32_t ch : {kV3InputChannels, c0, c1, c1 + c0}) {
        if (depthwise_pipelines_.contains(ch)) continue;
        if (!CreateDepthwiseConvPipeline(ch)) return false;
    }

    // Pointwise conv pipelines: (in_ch, out_ch) pairs
    struct PwConfig { uint32_t in_ch, out_ch; };
    std::vector<PwConfig> pw_configs = {
        {kV3InputChannels, c0},  // down0.conv1.pointwise
        {c0, c0},               // down0.conv2.pointwise
        {c0, c1},               // bottleneck1.pointwise
        {c1, c1},               // bottleneck2.pointwise
        {c1 + c0, c0},          // up0.conv1.pointwise
    };
    for (const auto& cfg : pw_configs) {
        ConvPipelineKey key{cfg.in_ch, cfg.out_ch};
        if (pointwise_pipelines_.contains(key)) continue;
        if (!CreatePointwiseConvPipeline(cfg.in_ch, cfg.out_ch)) return false;
    }

    // GroupNorm pipelines (c0 and c1)
    for (uint32_t ch : {c0, c1}) {
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

    // Downsample: level 0 → level 1 (c0 channels)
    if (!CreateDownsamplePipeline(c0)) return false;

    // Upsample-concat: level 1 → level 0
    if (!CreateUpsampleConcatPipeline(c1, c0)) return false;

    // Temporal output conv
    if (!CreateTemporalOutputConvPipeline()) return false;

    pipelines_created_ = true;
    return true;
}

void MlInference::DestroyPipelines() {
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

    if (reproject_pipeline_ != VK_NULL_HANDLE) {
        dispatch_.vkDestroyPipeline(device_, reproject_pipeline_, nullptr);
        reproject_pipeline_ = VK_NULL_HANDLE;
    }
    if (reproject_layout_ != VK_NULL_HANDLE) {
        dispatch_.vkDestroyPipelineLayout(device_, reproject_layout_, nullptr);
        reproject_layout_ = VK_NULL_HANDLE;
    }
    if (reproject_ds_layout_ != VK_NULL_HANDLE) {
        dispatch_.vkDestroyDescriptorSetLayout(device_, reproject_ds_layout_, nullptr);
        reproject_ds_layout_ = VK_NULL_HANDLE;
    }

    // V3 temporal pipelines
    for (auto& [ch, ps] : depthwise_pipelines_) {
        if (ps.pipeline != VK_NULL_HANDLE) dispatch_.vkDestroyPipeline(device_, ps.pipeline, nullptr);
        if (ps.layout != VK_NULL_HANDLE) dispatch_.vkDestroyPipelineLayout(device_, ps.layout, nullptr);
        if (ps.ds_layout != VK_NULL_HANDLE) dispatch_.vkDestroyDescriptorSetLayout(device_, ps.ds_layout, nullptr);
    }
    depthwise_pipelines_.clear();

    for (auto& [key, ps] : pointwise_pipelines_) {
        if (ps.pipeline != VK_NULL_HANDLE) dispatch_.vkDestroyPipeline(device_, ps.pipeline, nullptr);
        if (ps.layout != VK_NULL_HANDLE) dispatch_.vkDestroyPipelineLayout(device_, ps.layout, nullptr);
        if (ps.ds_layout != VK_NULL_HANDLE) dispatch_.vkDestroyDescriptorSetLayout(device_, ps.ds_layout, nullptr);
    }
    pointwise_pipelines_.clear();

    if (temporal_input_pipeline_ != VK_NULL_HANDLE) {
        dispatch_.vkDestroyPipeline(device_, temporal_input_pipeline_, nullptr);
        temporal_input_pipeline_ = VK_NULL_HANDLE;
    }
    if (temporal_input_layout_ != VK_NULL_HANDLE) {
        dispatch_.vkDestroyPipelineLayout(device_, temporal_input_layout_, nullptr);
        temporal_input_layout_ = VK_NULL_HANDLE;
    }
    if (temporal_input_ds_layout_ != VK_NULL_HANDLE) {
        dispatch_.vkDestroyDescriptorSetLayout(device_, temporal_input_ds_layout_, nullptr);
        temporal_input_ds_layout_ = VK_NULL_HANDLE;
    }

    if (temporal_output_pipeline_ != VK_NULL_HANDLE) {
        dispatch_.vkDestroyPipeline(device_, temporal_output_pipeline_, nullptr);
        temporal_output_pipeline_ = VK_NULL_HANDLE;
    }
    if (temporal_output_layout_ != VK_NULL_HANDLE) {
        dispatch_.vkDestroyPipelineLayout(device_, temporal_output_layout_, nullptr);
        temporal_output_layout_ = VK_NULL_HANDLE;
    }
    if (temporal_output_ds_layout_ != VK_NULL_HANDLE) {
        dispatch_.vkDestroyDescriptorSetLayout(device_, temporal_output_ds_layout_, nullptr);
        temporal_output_ds_layout_ = VK_NULL_HANDLE;
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
    // with up to 3 storage buffers + 8 storage images each (reproject uses 8 images).
    constexpr uint32_t kMaxSets = 64;
    std::array<VkDescriptorPoolSize, 2> pool_sizes{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[0].descriptorCount = kMaxSets * 4;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[1].descriptorCount = kMaxSets * 8;

    VkDescriptorPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets = kMaxSets;
    pool_ci.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_ci.pPoolSizes = pool_sizes.data();

    for (uint32_t i = 0; i < kPoolCount; ++i) {
        VkResult result = dispatch_.vkCreateDescriptorPool(device_, &pool_ci, nullptr,
                                                           &ml_descriptor_pools_[i]);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr,
                         "deni::MlInference: failed to create descriptor pool %u (VkResult: %d)\n",
                         i, result);
            return false;
        }
    }
    pool_index_ = 0;
    return true;
}

void MlInference::DestroyDescriptorPool() {
    for (auto& pool : ml_descriptor_pools_) {
        if (pool != VK_NULL_HANDLE) {
            dispatch_.vkDestroyDescriptorPool(device_, pool, nullptr);
            pool = VK_NULL_HANDLE;
        }
    }
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
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                            VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                            VK_ACCESS_2_TRANSFER_READ_BIT;

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    dispatch_.vkCmdPipelineBarrier2(cmd, &dep);
}

void MlInference::InsertImageBarrier(VkCommandBuffer cmd, const HistoryImage& image) {
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &barrier;
    dispatch_.vkCmdPipelineBarrier2(cmd, &dep);
}

void MlInference::CopyImageToHistory(VkCommandBuffer cmd, VkImage src, const HistoryImage& dst,
                                      uint32_t width, uint32_t height) {
    // Barrier: compute write → transfer read on source
    VkImageMemoryBarrier2 src_barrier{};
    src_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    src_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    src_barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    src_barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    src_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    src_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    src_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    src_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    src_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    src_barrier.image = src;
    src_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    // Barrier: previous reads/writes → transfer write on destination
    VkImageMemoryBarrier2 dst_barrier{};
    dst_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    dst_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    dst_barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                VK_ACCESS_2_TRANSFER_WRITE_BIT;
    dst_barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    dst_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    dst_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    dst_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    dst_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    dst_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    dst_barrier.image = dst.image;
    dst_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    std::array<VkImageMemoryBarrier2, 2> barriers = {src_barrier, dst_barrier};
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
    dep.pImageMemoryBarriers = barriers.data();
    dispatch_.vkCmdPipelineBarrier2(cmd, &dep);

    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent = {width, height, 1};

    dispatch_.vkCmdCopyImage(cmd,
                             src, VK_IMAGE_LAYOUT_GENERAL,
                             dst.image, VK_IMAGE_LAYOUT_GENERAL,
                             1, &region);

    // Barrier: transfer write → compute read (for subsequent reproject dispatch)
    VkImageMemoryBarrier2 post_barrier{};
    post_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    post_barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    post_barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    post_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    post_barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    post_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    post_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    post_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post_barrier.image = dst.image;
    post_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo post_dep{};
    post_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    post_dep.imageMemoryBarrierCount = 1;
    post_dep.pImageMemoryBarriers = &post_barrier;
    dispatch_.vkCmdPipelineBarrier2(cmd, &post_dep);
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

void MlInference::DispatchReproject(VkCommandBuffer cmd, const DenoiserInput& input) {
    if (reproject_pipeline_ == VK_NULL_HANDLE) return;

    VkDescriptorSet ds = AllocateOneDescriptorSet(dispatch_, device_, active_pool_,
                                                   reproject_ds_layout_);
    // Bind images: 0=motion_vectors, 1=prev_diffuse, 2=prev_specular,
    //              3=prev_depth, 4=curr_depth,
    //              5=reprojected_d, 6=reprojected_s, 7=disocclusion
    WriteImageDescriptor(dispatch_, device_, ds, 0, input.motion_vectors);
    WriteImageDescriptor(dispatch_, device_, ds, 1, frame_history_.denoised_diffuse.view);
    WriteImageDescriptor(dispatch_, device_, ds, 2, frame_history_.denoised_specular.view);
    WriteImageDescriptor(dispatch_, device_, ds, 3, frame_history_.prev_depth.view);
    WriteImageDescriptor(dispatch_, device_, ds, 4, input.linear_depth);
    WriteImageDescriptor(dispatch_, device_, ds, 5, frame_history_.reprojected_diffuse.view);
    WriteImageDescriptor(dispatch_, device_, ds, 6, frame_history_.reprojected_specular.view);
    WriteImageDescriptor(dispatch_, device_, ds, 7, frame_history_.disocclusion_mask.view);

    MlPushConstants pc{width_, height_};
    dispatch_.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, reproject_pipeline_);
    dispatch_.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      reproject_layout_, 0, 1, &ds, 0, nullptr);
    dispatch_.vkCmdPushConstants(cmd, reproject_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                                 0, sizeof(pc), &pc);
    dispatch_.vkCmdDispatch(cmd, DivCeil(width_, kWorkgroupSize),
                            DivCeil(height_, kWorkgroupSize), 1);
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

    uint32_t num_groups = NumGroups(channels);
    uint32_t channels_per_group = channels / num_groups;
    uint32_t elements_per_group = channels_per_group * width * height;
    uint32_t num_reduce_wg = DivCeil(elements_per_group, kReduceWorkgroupSize);

    // Pass 1: reduce
    {
        VkDescriptorSet ds = AllocateOneDescriptorSet(dispatch_, device_, active_pool_,
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
        dispatch_.vkCmdDispatch(cmd, num_reduce_wg, num_groups, 1);
        InsertBufferBarrier(cmd);
    }

    // Pass 2: apply
    {
        VkDescriptorSet ds = AllocateOneDescriptorSet(dispatch_, device_, active_pool_,
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

    VkDescriptorSet ds = AllocateOneDescriptorSet(dispatch_, device_, active_pool_,
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

    VkDescriptorSet ds = AllocateOneDescriptorSet(dispatch_, device_, active_pool_,
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
// V3 Temporal pipeline creation
// ---------------------------------------------------------------------------

bool MlInference::CreateDepthwiseConvPipeline(uint32_t channels) {
    // Same descriptor layout as generic conv: input, output, weights (3 storage buffers)
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    DepthwisePipelineSet ps;

    VkDescriptorSetLayoutCreateInfo ds_ci{};
    ds_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ds_ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ds_ci.pBindings = bindings.data();
    VkResult result = dispatch_.vkCreateDescriptorSetLayout(device_, &ds_ci, nullptr,
                                                            &ps.ds_layout);
    if (result != VK_SUCCESS) return false;

    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.size = sizeof(MlPushConstants);

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &ps.ds_layout;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &pc_range;
    result = dispatch_.vkCreatePipelineLayout(device_, &layout_ci, nullptr, &ps.layout);
    if (result != VK_SUCCESS) return false;

    VkShaderModule module = LoadShaderModule("depthwise_conv.comp");
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

    depthwise_pipelines_[channels] = ps;
    return true;
}

bool MlInference::CreatePointwiseConvPipeline(uint32_t in_ch, uint32_t out_ch) {
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    PipelineSet ps;

    VkDescriptorSetLayoutCreateInfo ds_ci{};
    ds_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ds_ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ds_ci.pBindings = bindings.data();
    VkResult result = dispatch_.vkCreateDescriptorSetLayout(device_, &ds_ci, nullptr,
                                                            &ps.ds_layout);
    if (result != VK_SUCCESS) return false;

    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.size = sizeof(MlPushConstants);

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &ps.ds_layout;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &pc_range;
    result = dispatch_.vkCreatePipelineLayout(device_, &layout_ci, nullptr, &ps.layout);
    if (result != VK_SUCCESS) return false;

    VkShaderModule module = LoadShaderModule("pointwise_conv.comp");
    if (module == VK_NULL_HANDLE) return false;

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
    pipeline_ci.layout = ps.layout;

    result = dispatch_.vkCreateComputePipelines(device_, pipeline_cache_, 1, &pipeline_ci,
                                                nullptr, &ps.pipeline);
    dispatch_.vkDestroyShaderModule(device_, module, nullptr);
    if (result != VK_SUCCESS) return false;

    pointwise_pipelines_[{in_ch, out_ch}] = ps;
    return true;
}

bool MlInference::CreateTemporalInputGatherPipeline() {
    // Bindings: 0-9=storage images (3 temporal + 7 G-buffer), 10=output buffer
    std::array<VkDescriptorSetLayoutBinding, 11> bindings{};
    for (uint32_t i = 0; i < 10; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    bindings[10].binding = 10;
    bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[10].descriptorCount = 1;
    bindings[10].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo ds_ci{};
    ds_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ds_ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ds_ci.pBindings = bindings.data();
    VkResult result = dispatch_.vkCreateDescriptorSetLayout(device_, &ds_ci, nullptr,
                                                            &temporal_input_ds_layout_);
    if (result != VK_SUCCESS) return false;

    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.size = sizeof(MlPushConstants);

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &temporal_input_ds_layout_;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &pc_range;
    result = dispatch_.vkCreatePipelineLayout(device_, &layout_ci, nullptr,
                                              &temporal_input_layout_);
    if (result != VK_SUCCESS) return false;

    VkShaderModule module = LoadShaderModule("temporal_input_gather.comp");
    if (module == VK_NULL_HANDLE) return false;

    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_ci.stage.module = module;
    pipeline_ci.stage.pName = "main";
    pipeline_ci.layout = temporal_input_layout_;

    result = dispatch_.vkCreateComputePipelines(device_, pipeline_cache_, 1, &pipeline_ci,
                                                nullptr, &temporal_input_pipeline_);
    dispatch_.vkDestroyShaderModule(device_, module, nullptr);
    return result == VK_SUCCESS;
}

bool MlInference::CreateTemporalOutputConvPipeline() {
    // Bindings: 0=input buffer, 1=weights buffer,
    //           2=output image, 3-5=temporal images, 6-8=G-buffer images,
    //           9-10=history output images, 11=motion vectors
    std::array<VkDescriptorSetLayoutBinding, 12> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    for (uint32_t i = 2; i < 12; ++i) {
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
                                                            &temporal_output_ds_layout_);
    if (result != VK_SUCCESS) return false;

    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.size = 3 * sizeof(uint32_t) + sizeof(float);  // width, height, debug_mode, max_mv_for_weight

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &temporal_output_ds_layout_;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &pc_range;
    result = dispatch_.vkCreatePipelineLayout(device_, &layout_ci, nullptr,
                                              &temporal_output_layout_);
    if (result != VK_SUCCESS) return false;

    VkShaderModule module = LoadShaderModule("temporal_output_conv.comp");
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
    pipeline_ci.layout = temporal_output_layout_;

    result = dispatch_.vkCreateComputePipelines(device_, pipeline_cache_, 1, &pipeline_ci,
                                                nullptr, &temporal_output_pipeline_);
    dispatch_.vkDestroyShaderModule(device_, module, nullptr);
    return result == VK_SUCCESS;
}

// ---------------------------------------------------------------------------
// V3 Temporal dispatch helpers
// ---------------------------------------------------------------------------

void MlInference::DispatchDepthwiseConv(VkCommandBuffer cmd, VkBuffer input, VkBuffer output,
                                         std::string_view weight_name, uint32_t channels,
                                         uint32_t width, uint32_t height) {
    auto it = depthwise_pipelines_.find(channels);
    if (it == depthwise_pipelines_.end()) return;
    const auto& ps = it->second;

    VkBuffer weight_buf = FindWeightBuffer(weight_name);
    if (weight_buf == VK_NULL_HANDLE) return;

    VkDescriptorSet ds = AllocateOneDescriptorSet(dispatch_, device_, active_pool_,
                                                   ps.ds_layout);
    VkDeviceSize buf_size = static_cast<VkDeviceSize>(channels) * width * height * sizeof(uint16_t);
    // Depthwise weights: [channels][1][3][3], no bias
    VkDeviceSize weight_size = static_cast<VkDeviceSize>(channels) * 9 * sizeof(float);
    WriteBufferDescriptor(dispatch_, device_, ds, 0, input, buf_size);
    WriteBufferDescriptor(dispatch_, device_, ds, 1, output, buf_size);
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

void MlInference::DispatchPointwiseConv(VkCommandBuffer cmd, VkBuffer input, VkBuffer output,
                                         std::string_view weight_name, uint32_t in_ch,
                                         uint32_t out_ch, uint32_t width, uint32_t height) {
    ConvPipelineKey key{in_ch, out_ch};
    auto it = pointwise_pipelines_.find(key);
    if (it == pointwise_pipelines_.end()) return;
    const auto& ps = it->second;

    VkBuffer weight_buf = FindWeightBuffer(weight_name);
    if (weight_buf == VK_NULL_HANDLE) return;

    VkDescriptorSet ds = AllocateOneDescriptorSet(dispatch_, device_, active_pool_,
                                                   ps.ds_layout);
    VkDeviceSize in_size = static_cast<VkDeviceSize>(in_ch) * width * height * sizeof(uint16_t);
    VkDeviceSize out_size = static_cast<VkDeviceSize>(out_ch) * width * height * sizeof(uint16_t);
    // Pointwise weights: [out_ch][in_ch][1][1] + bias[out_ch]
    VkDeviceSize weight_size = static_cast<VkDeviceSize>(out_ch) * in_ch * sizeof(float)
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

void MlInference::DispatchTemporalInputGather(VkCommandBuffer cmd, const DenoiserInput& input) {
    if (temporal_input_pipeline_ == VK_NULL_HANDLE) return;

    VkDescriptorSet ds = AllocateOneDescriptorSet(dispatch_, device_, active_pool_,
                                                   temporal_input_ds_layout_);
    // Temporal history images (bindings 0-2)
    WriteImageDescriptor(dispatch_, device_, ds, 0, frame_history_.reprojected_diffuse.view);
    WriteImageDescriptor(dispatch_, device_, ds, 1, frame_history_.reprojected_specular.view);
    WriteImageDescriptor(dispatch_, device_, ds, 2, frame_history_.disocclusion_mask.view);
    // G-buffer images (bindings 3-9)
    WriteImageDescriptor(dispatch_, device_, ds, 3, input.noisy_diffuse);
    WriteImageDescriptor(dispatch_, device_, ds, 4, input.noisy_specular);
    WriteImageDescriptor(dispatch_, device_, ds, 5, input.world_normals);
    WriteImageDescriptor(dispatch_, device_, ds, 6, input.linear_depth);
    WriteImageDescriptor(dispatch_, device_, ds, 7, input.motion_vectors);
    WriteImageDescriptor(dispatch_, device_, ds, 8, input.diffuse_albedo);
    WriteImageDescriptor(dispatch_, device_, ds, 9, input.specular_albedo);
    // Output buffer (binding 10)
    WriteBufferDescriptor(dispatch_, device_, ds, 10, buf_input_.buffer, buf_input_.size_bytes);

    MlPushConstants pc{width_, height_};
    dispatch_.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_input_pipeline_);
    dispatch_.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      temporal_input_layout_, 0, 1, &ds, 0, nullptr);
    dispatch_.vkCmdPushConstants(cmd, temporal_input_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                                 0, sizeof(pc), &pc);
    dispatch_.vkCmdDispatch(cmd, DivCeil(width_, kWorkgroupSize),
                            DivCeil(height_, kWorkgroupSize), 1);
    InsertBufferBarrier(cmd);
}

void MlInference::DispatchTemporalOutputConv(VkCommandBuffer cmd, VkBuffer feature_buf,
                                              VkImageView output_view,
                                              const DenoiserInput& input) {
    if (temporal_output_pipeline_ == VK_NULL_HANDLE) return;

    VkBuffer out_weights = FindWeightBuffer("out_conv");
    if (out_weights == VK_NULL_HANDLE) return;

    VkDescriptorSet ds = AllocateOneDescriptorSet(dispatch_, device_, active_pool_,
                                                   temporal_output_ds_layout_);
    // Binding 0: input feature buffer
    VkDeviceSize feat_size = static_cast<VkDeviceSize>(level0_channels_) * width_ * height_ * sizeof(uint16_t);
    WriteBufferDescriptor(dispatch_, device_, ds, 0, feature_buf, feat_size);
    // Binding 1: weights [7][c0][1][1] + bias[7]
    VkDeviceSize w_size = static_cast<VkDeviceSize>(kV3OutputChannels) * level0_channels_ * sizeof(float)
                          + kV3OutputChannels * sizeof(float);
    WriteBufferDescriptor(dispatch_, device_, ds, 1, out_weights, w_size);
    // Binding 2: output image
    WriteImageDescriptor(dispatch_, device_, ds, 2, output_view);
    // Bindings 3-5: temporal blending inputs
    WriteImageDescriptor(dispatch_, device_, ds, 3, frame_history_.reprojected_diffuse.view);
    WriteImageDescriptor(dispatch_, device_, ds, 4, frame_history_.reprojected_specular.view);
    WriteImageDescriptor(dispatch_, device_, ds, 5, frame_history_.disocclusion_mask.view);
    // Bindings 6-8: G-buffer for remodulation
    WriteImageDescriptor(dispatch_, device_, ds, 6, input.noisy_diffuse);
    WriteImageDescriptor(dispatch_, device_, ds, 7, input.diffuse_albedo);
    WriteImageDescriptor(dispatch_, device_, ds, 8, input.specular_albedo);
    // Bindings 9-10: history output images
    WriteImageDescriptor(dispatch_, device_, ds, 9, frame_history_.denoised_diffuse.view);
    WriteImageDescriptor(dispatch_, device_, ds, 10, frame_history_.denoised_specular.view);
    // Binding 11: motion vectors (for velocity prior)
    WriteImageDescriptor(dispatch_, device_, ds, 11, input.motion_vectors);

    struct TemporalOutputPC {
        uint32_t width;
        uint32_t height;
        uint32_t debug_mode;
        float max_mv_for_weight;
    } pc{width_, height_, debug_output_, max_mv_for_weight_};
    dispatch_.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_output_pipeline_);
    dispatch_.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      temporal_output_layout_, 0, 1, &ds, 0, nullptr);
    dispatch_.vkCmdPushConstants(cmd, temporal_output_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                                 0, sizeof(pc), &pc);
    dispatch_.vkCmdDispatch(cmd, DivCeil(width_, kWorkgroupSize),
                            DivCeil(height_, kWorkgroupSize), 1);
}

// ---------------------------------------------------------------------------
// Infer — Full U-Net dispatch sequence
// ---------------------------------------------------------------------------

void MlInference::Infer(VkCommandBuffer cmd, const DenoiserInput& input,
                         VkImageView output_view, VkImage output_image) {
    if (!IsReady()) return;

    // Handle temporal reset
    if (input.reset_accumulation)
        frame_history_.valid = false;

    // Select the next descriptor pool in the ring.
    VkDescriptorPool active_pool = ml_descriptor_pools_[pool_index_];
    pool_index_ = (pool_index_ + 1) % kPoolCount;
    if (active_pool != VK_NULL_HANDLE)
        dispatch_.vkResetDescriptorPool(device_, active_pool, 0);
    active_pool_ = active_pool;

    // GPU timestamp: begin
    if (query_pool_ != VK_NULL_HANDLE) {
        dispatch_.vkCmdResetQueryPool(cmd, query_pool_, 0, kTimestampCount);
        dispatch_.vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                       query_pool_, 0);
    }

    // Transition history images from UNDEFINED to GENERAL on first frame or reset,
    // then clear to zero so temporal_input_gather reads zeros (no history available).
    // Without the clear, UNDEFINED→GENERAL leaves stale GPU memory content which
    // causes non-deterministic output depending on prior allocations.
    if (!frame_history_.valid && frame_history_.denoised_diffuse.image != VK_NULL_HANDLE) {
        auto transition_undefined_to_general = [&](VkImage image) {
            VkImageMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            barrier.srcAccessMask = 0;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &barrier;
            dispatch_.vkCmdPipelineBarrier2(cmd, &dep);
        };
        transition_undefined_to_general(frame_history_.denoised_diffuse.image);
        transition_undefined_to_general(frame_history_.denoised_specular.image);
        transition_undefined_to_general(frame_history_.reprojected_diffuse.image);
        transition_undefined_to_general(frame_history_.reprojected_specular.image);
        transition_undefined_to_general(frame_history_.disocclusion_mask.image);
        transition_undefined_to_general(frame_history_.prev_depth.image);

        // Clear temporal images to zero so the first frame has no history influence
        VkClearColorValue zero{};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        dispatch_.vkCmdClearColorImage(cmd, frame_history_.denoised_diffuse.image,
                                        VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
        dispatch_.vkCmdClearColorImage(cmd, frame_history_.denoised_specular.image,
                                        VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
        dispatch_.vkCmdClearColorImage(cmd, frame_history_.reprojected_diffuse.image,
                                        VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
        dispatch_.vkCmdClearColorImage(cmd, frame_history_.reprojected_specular.image,
                                        VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
        dispatch_.vkCmdClearColorImage(cmd, frame_history_.disocclusion_mask.image,
                                        VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
        dispatch_.vkCmdClearColorImage(cmd, frame_history_.prev_depth.image,
                                        VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);

        // Barrier: clear writes (TRANSFER) → shader reads (COMPUTE)
        VkMemoryBarrier2 clear_barrier{};
        clear_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        clear_barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        clear_barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        clear_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        clear_barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo clear_dep{};
        clear_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        clear_dep.memoryBarrierCount = 1;
        clear_dep.pMemoryBarriers = &clear_barrier;
        dispatch_.vkCmdPipelineBarrier2(cmd, &clear_dep);
    }
    if (frame_history_.valid) {
        DispatchReproject(cmd, input);
        InsertImageBarrier(cmd, frame_history_.reprojected_diffuse);
        InsertImageBarrier(cmd, frame_history_.reprojected_specular);
        InsertImageBarrier(cmd, frame_history_.disocclusion_mask);
    }

    // Dispatch temporal inference
    InferV3Temporal(cmd, input, output_view, output_image);

    // GPU timestamp: end
    if (query_pool_ != VK_NULL_HANDLE) {
        dispatch_.vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                                       query_pool_, 1);
        timestamps_valid_ = true;
    }
}

// ---------------------------------------------------------------------------
// V3 Temporal Residual inference path
// ---------------------------------------------------------------------------

void MlInference::InferV3Temporal(VkCommandBuffer cmd, const DenoiserInput& input,
                                   VkImageView output_view, VkImage output_image) {
    uint32_t c0 = level0_channels_;
    uint32_t c1 = level1_channels_;
    uint32_t w0 = width_, h0 = height_;
    uint32_t w1 = DivCeil(w0, 2), h1 = DivCeil(h0, 2);

    // 1. Gather 26-ch temporal input from images → flat buffer
    DispatchTemporalInputGather(cmd, input);

    // 2. Encoder level 0: down0.conv1 (DepthwiseSeparableConvBlock)
    DispatchDepthwiseConv(cmd, buf_input_.buffer, buf0_a_.buffer,
                          "down0.conv1.depthwise", kV3InputChannels, w0, h0);
    DispatchPointwiseConv(cmd, buf0_a_.buffer, buf0_b_.buffer,
                          "down0.conv1.pointwise", kV3InputChannels, c0, w0, h0);
    DispatchGroupNorm(cmd, buf0_b_.buffer, "down0.conv1.norm", c0, w0, h0);

    // 3. down0.conv2 (DepthwiseSeparableConvBlock)
    DispatchDepthwiseConv(cmd, buf0_b_.buffer, buf0_a_.buffer,
                          "down0.conv2.depthwise", c0, w0, h0);
    DispatchPointwiseConv(cmd, buf0_a_.buffer, buf0_b_.buffer,
                          "down0.conv2.pointwise", c0, c0, w0, h0);
    DispatchGroupNorm(cmd, buf0_b_.buffer, "down0.conv2.norm", c0, w0, h0);

    // Save skip0 + downsample to level 1
    {
        VkBufferCopy copy{};
        copy.size = buf0_b_.size_bytes;
        dispatch_.vkCmdCopyBuffer(cmd, buf0_b_.buffer, skip0_.buffer, 1, &copy);
    }
    DispatchDownsample(cmd, buf0_b_.buffer, buf1_a_.buffer, c0, w0, h0);

    // 4. Bottleneck at H/2 × W/2 (2-level U-Net, no encoder level 1)
    DispatchDepthwiseConv(cmd, buf1_a_.buffer, buf1_b_.buffer,
                          "bottleneck1.depthwise", c0, w1, h1);
    DispatchPointwiseConv(cmd, buf1_b_.buffer, buf1_a_.buffer,
                          "bottleneck1.pointwise", c0, c1, w1, h1);
    DispatchGroupNorm(cmd, buf1_a_.buffer, "bottleneck1.norm", c1, w1, h1);

    DispatchDepthwiseConv(cmd, buf1_a_.buffer, buf1_b_.buffer,
                          "bottleneck2.depthwise", c1, w1, h1);
    DispatchPointwiseConv(cmd, buf1_b_.buffer, buf1_a_.buffer,
                          "bottleneck2.pointwise", c1, c1, w1, h1);
    DispatchGroupNorm(cmd, buf1_a_.buffer, "bottleneck2.norm", c1, w1, h1);

    // 5. Decoder level 0: upsample + concat skip0
    DispatchUpsampleConcat(cmd, buf1_a_.buffer, skip0_.buffer,
                           concat0_.buffer, c1, c0, w0, h0);

    DispatchDepthwiseConv(cmd, concat0_.buffer, buf0_a_.buffer,
                          "up0.conv1.depthwise", c1 + c0, w0, h0);
    DispatchPointwiseConv(cmd, buf0_a_.buffer, buf0_b_.buffer,
                          "up0.conv1.pointwise", c1 + c0, c0, w0, h0);
    DispatchGroupNorm(cmd, buf0_b_.buffer, "up0.conv1.norm", c0, w0, h0);

    DispatchDepthwiseConv(cmd, buf0_b_.buffer, buf0_a_.buffer,
                          "up0.conv2.depthwise", c0, w0, h0);
    DispatchPointwiseConv(cmd, buf0_a_.buffer, buf0_b_.buffer,
                          "up0.conv2.pointwise", c0, c0, w0, h0);
    DispatchGroupNorm(cmd, buf0_b_.buffer, "up0.conv2.norm", c0, w0, h0);

    // 6. Temporal output (fused: 1×1 conv, sigmoid, blend, remodulate, history write)
    DispatchTemporalOutputConv(cmd, buf0_b_.buffer, output_view, input);

    // 7. Save current depth for next frame
    // (denoised irradiance is written to history by temporal_output_conv.comp directly)
    if (input.linear_depth_image != VK_NULL_HANDLE)
        CopyImageToHistory(cmd, input.linear_depth_image, frame_history_.prev_depth, w0, h0);
    frame_history_.valid = (output_image != VK_NULL_HANDLE &&
                            input.linear_depth_image != VK_NULL_HANDLE);
}

}  // namespace deni::vulkan
