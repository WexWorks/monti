#pragma once

#include <volk.h>

#include <cstdint>
#include <vector>

struct SDL_Window;
union SDL_Event;

namespace monti::app {

class VulkanContext;
class Swapchain;

// ImGui Vulkan/SDL3 renderer. Creates a render pass that overlays ImGui on top
// of the swapchain image (arrives in TRANSFER_DST_OPTIMAL after blit, departs
// in PRESENT_SRC_KHR ready for presentation).
class UiRenderer {
public:
    UiRenderer() = default;
    ~UiRenderer();

    UiRenderer(const UiRenderer&) = delete;
    UiRenderer& operator=(const UiRenderer&) = delete;

    bool Initialize(VulkanContext& ctx, SDL_Window* window, const Swapchain& swapchain);
    void Destroy();

    // Recreate framebuffers when swapchain is recreated.
    bool Resize(const Swapchain& swapchain);

    // Forward SDL events to ImGui. Returns true if ImGui consumed the event.
    bool ProcessEvent(const SDL_Event& event);

    // Call at the start of each frame before issuing ImGui draw calls.
    void BeginFrame();

    // Record ImGui draw commands into cmd. Swapchain image must be in
    // TRANSFER_DST_OPTIMAL. Transitions it to PRESENT_SRC_KHR via the render pass.
    void EndFrame(VkCommandBuffer cmd, uint32_t image_index);

    bool WantCaptureMouse() const;
    bool WantCaptureKeyboard() const;

private:
    bool CreateRenderPass(VkFormat swapchain_format);
    bool CreateFramebuffers(const Swapchain& swapchain);
    void DestroyFramebuffers();

    VulkanContext* ctx_ = nullptr;

    VkRenderPass render_pass_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_;
};

}  // namespace monti::app
