#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <monti/vulkan/Renderer.h>

#include <array>
#include <cstdint>

namespace monti::app {

// RAII wrapper for G-buffer storage images used by the renderer and denoiser.
// All images are created with VK_IMAGE_USAGE_STORAGE_BIT and transitioned to
// VK_IMAGE_LAYOUT_GENERAL on creation. Both monti_view and monti_datagen pass
// VK_IMAGE_USAGE_TRANSFER_SRC_BIT via datagen_extra_usage: monti_datagen needs
// it for GPU→CPU readback, and monti_view uses it for debug blit operations.
//
// A single set of images is sufficient — renderer and denoiser are sequential
// within the same command buffer. Temporal denoisers maintain their own history.
class GBufferImages {
public:
    enum class Index : uint32_t {
        kNoisyDiffuse = 0,
        kNoisySpecular,
        kMotionVectors,
        kLinearDepth,
        kWorldNormals,
        kDiffuseAlbedo,
        kSpecularAlbedo,
        kCount
    };
    static constexpr uint32_t kImageCount = static_cast<uint32_t>(Index::kCount);

    GBufferImages() = default;
    ~GBufferImages();

    GBufferImages(const GBufferImages&) = delete;
    GBufferImages& operator=(const GBufferImages&) = delete;
    GBufferImages(GBufferImages&&) = delete;
    GBufferImages& operator=(GBufferImages&&) = delete;

    // Create all G-buffer images at the given resolution.
    // datagen_extra_usage: additional usage flags for datagen readback (0 for monti_view).
    bool Create(VmaAllocator allocator, VkDevice device,
                uint32_t width, uint32_t height,
                VkCommandBuffer cmd,
                VkImageUsageFlags datagen_extra_usage = 0);

    // Destroy and recreate at a new resolution. Records layout transitions into cmd.
    bool Resize(uint32_t width, uint32_t height, VkCommandBuffer cmd);

    void Destroy();

    uint32_t Width() const { return width_; }
    uint32_t Height() const { return height_; }

    VkImage ImageHandle(Index idx) const { return entries_[static_cast<uint32_t>(idx)].image; }
    VkImageView ViewHandle(Index idx) const { return entries_[static_cast<uint32_t>(idx)].view; }

    VkImage NoisyDiffuseImage() const { return ImageHandle(Index::kNoisyDiffuse); }
    VkImage NoisySpecularImage() const { return ImageHandle(Index::kNoisySpecular); }
    VkImage MotionVectorsImage() const { return ImageHandle(Index::kMotionVectors); }
    VkImage LinearDepthImage() const { return ImageHandle(Index::kLinearDepth); }
    VkImage WorldNormalsImage() const { return ImageHandle(Index::kWorldNormals); }
    VkImage DiffuseAlbedoImage() const { return ImageHandle(Index::kDiffuseAlbedo); }
    VkImage SpecularAlbedoImage() const { return ImageHandle(Index::kSpecularAlbedo); }

    VkImageView NoisyDiffuseView() const { return ViewHandle(Index::kNoisyDiffuse); }
    VkImageView NoisySpecularView() const { return ViewHandle(Index::kNoisySpecular); }
    VkImageView MotionVectorsView() const { return ViewHandle(Index::kMotionVectors); }
    VkImageView LinearDepthView() const { return ViewHandle(Index::kLinearDepth); }
    VkImageView WorldNormalsView() const { return ViewHandle(Index::kWorldNormals); }
    VkImageView DiffuseAlbedoView() const { return ViewHandle(Index::kDiffuseAlbedo); }
    VkImageView SpecularAlbedoView() const { return ViewHandle(Index::kSpecularAlbedo); }

    static VkFormat FormatFor(Index idx);

    vulkan::GBuffer ToGBuffer() const;

private:
    struct ImageEntry {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
    };

    bool CreateImage(VkFormat format, VkImageUsageFlags usage, ImageEntry& out);
    void TransitionToGeneral(VkCommandBuffer cmd);
    void DestroyEntry(ImageEntry& entry);

    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    VkImageUsageFlags datagen_extra_usage_ = 0;

    std::array<ImageEntry, kImageCount> entries_{};
};

}  // namespace monti::app
