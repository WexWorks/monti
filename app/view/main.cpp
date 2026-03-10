#include "../core/vulkan_context.h"
#include "../core/frame_resources.h"
#include "swapchain.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

namespace {

constexpr uint32_t kInitialWidth = 1280;
constexpr uint32_t kInitialHeight = 720;

struct AppState {
    SDL_Window* window = nullptr;
    monti::app::VulkanContext* ctx = nullptr;
    monti::app::Swapchain* swapchain = nullptr;
    monti::app::FrameResources* frame_resources = nullptr;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    uint32_t current_frame = 0;
    bool running = true;
    bool rendering = false;
};

bool RecreateSwapchain(AppState& state) {
    int w, h;
    SDL_GetWindowSizeInPixels(state.window, &w, &h);
    if (w <= 0 || h <= 0) return false;

    state.ctx->WaitIdle();
    if (!state.swapchain->Create(*state.ctx, state.surface,
                                  static_cast<uint32_t>(w), static_cast<uint32_t>(h)))
        return false;

    // Recreate render-finished semaphores to match new swapchain image count
    if (!state.frame_resources->RecreateRenderFinishedSemaphores(state.swapchain->ImageCount())) {
        state.running = false;
        return false;
    }
    state.current_frame = 0;
    return true;
}

bool RenderFrame(AppState& state) {
    if (state.rendering) return true;
    state.rendering = true;

    auto& ctx = *state.ctx;
    auto& swapchain = *state.swapchain;
    auto& fr = *state.frame_resources;

    fr.WaitForFence(state.current_frame);

    uint32_t image_index;
    VkResult result = vkAcquireNextImageKHR(
        ctx.Device(), swapchain.Handle(), UINT64_MAX,
        fr.ImageAvailableSemaphore(state.current_frame), VK_NULL_HANDLE, &image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        if (!RecreateSwapchain(state)) {
            state.rendering = false;
            return false;
        }
        state.rendering = false;
        return true;
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        std::fprintf(stderr, "Failed to acquire swapchain image (VkResult: %d)\n", result);
        state.rendering = false;
        return false;
    }

    fr.ResetFence(state.current_frame);
    fr.ResetCommandBuffer(state.current_frame);

    VkCommandBuffer cmd = fr.CommandBuffer(state.current_frame);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin_info);

    // Transition swapchain image to transfer dst
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    barrier.srcAccessMask = 0;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.image = swapchain.Image(image_index);
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkDependencyInfo dep_info{};
    dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep_info.imageMemoryBarrierCount = 1;
    dep_info.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep_info);

    // Clear to cornflower blue
    VkClearColorValue clear_color = {{0.392f, 0.584f, 0.929f, 1.0f}};
    VkImageSubresourceRange clear_range{};
    clear_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    clear_range.baseMipLevel = 0;
    clear_range.levelCount = 1;
    clear_range.baseArrayLayer = 0;
    clear_range.layerCount = 1;
    vkCmdClearColorImage(cmd, swapchain.Image(image_index),
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &clear_color, 1, &clear_range);

    // Transition swapchain image to present
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    barrier.dstAccessMask = 0;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier2(cmd, &dep_info);

    vkEndCommandBuffer(cmd);

    // Submit
    VkSemaphore wait_semaphores[] = {fr.ImageAvailableSemaphore(state.current_frame)};
    VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphore signal_semaphores[] = {fr.RenderFinishedSemaphore(image_index)};

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = wait_semaphores;
    submit_info.pWaitDstStageMask = wait_stages;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores;

    result = vkQueueSubmit(ctx.GraphicsQueue(), 1, &submit_info,
                           fr.InFlightFence(state.current_frame));
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to submit command buffer (VkResult: %d)\n", result);
        state.rendering = false;
        return false;
    }

    // Present
    VkSwapchainKHR swapchains[] = {swapchain.Handle()};
    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = signal_semaphores;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = swapchains;
    present_info.pImageIndices = &image_index;

    result = vkQueuePresentKHR(ctx.GraphicsQueue(), &present_info);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        if (!RecreateSwapchain(state)) {
            state.rendering = false;
            return false;
        }
    } else if (result != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to present (VkResult: %d)\n", result);
        state.rendering = false;
        return false;
    }

    state.current_frame = (state.current_frame + 1) % monti::app::FrameResources::kFramesInFlight;
    state.rendering = false;
    return true;
}

// Called during the OS modal resize loop so we can render while the window is being dragged
bool EventWatcher(void* userdata, SDL_Event* event) {
    if (event->type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
        auto* state = static_cast<AppState*>(userdata);
        if (RecreateSwapchain(*state))
            RenderFrame(*state);
    }
    return true;
}

}  // namespace

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "Failed to initialize SDL: %s\n", SDL_GetError());
        return EXIT_FAILURE;
    }

    auto sdl_cleanup = std::unique_ptr<void, decltype([](void*) { SDL_Quit(); })>(
        reinterpret_cast<void*>(1));

    SDL_Window* window = SDL_CreateWindow(
        "Monti View",
        static_cast<int>(kInitialWidth), static_cast<int>(kInitialHeight),
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window) {
        std::fprintf(stderr, "Failed to create window: %s\n", SDL_GetError());
        return EXIT_FAILURE;
    }

    auto window_cleanup = std::unique_ptr<SDL_Window, decltype([](SDL_Window* w) {
        SDL_DestroyWindow(w);
    })>(window);

    // Step 1: Create Vulkan instance with SDL-required extensions
    monti::app::VulkanContext ctx;

    uint32_t sdl_ext_count = 0;
    auto sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&sdl_ext_count);
    if (!sdl_extensions) {
        std::fprintf(stderr, "Failed to get SDL Vulkan extensions: %s\n", SDL_GetError());
        return EXIT_FAILURE;
    }

    std::span<const char* const> ext_span(sdl_extensions, sdl_ext_count);
    if (!ctx.CreateInstance(ext_span)) return EXIT_FAILURE;

    // Create surface via SDL
    VkSurfaceKHR surface;
    if (!SDL_Vulkan_CreateSurface(window, ctx.Instance(), nullptr, &surface)) {
        std::fprintf(stderr, "Failed to create Vulkan surface: %s\n", SDL_GetError());
        return EXIT_FAILURE;
    }

    // Step 2: Create device with surface for windowed mode
    if (!ctx.CreateDevice(surface)) return EXIT_FAILURE;

    // Create swapchain
    monti::app::Swapchain swapchain;
    if (!swapchain.Create(ctx, surface, kInitialWidth, kInitialHeight))
        return EXIT_FAILURE;

    // Create frame resources
    monti::app::FrameResources frame_resources;
    if (!frame_resources.Create(ctx)) return EXIT_FAILURE;
    if (!frame_resources.RecreateRenderFinishedSemaphores(swapchain.ImageCount()))
        return EXIT_FAILURE;

    // Set up app state for the event watcher
    AppState state{};
    state.window = window;
    state.ctx = &ctx;
    state.swapchain = &swapchain;
    state.frame_resources = &frame_resources;
    state.surface = surface;

    SDL_AddEventWatch(EventWatcher, &state);

    // Main loop
    while (state.running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT)
                state.running = false;
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE)
                state.running = false;
        }

        if (!state.running) break;

        if (!RenderFrame(state)) {
            state.running = false;
            break;
        }
    }

    // Clean shutdown
    SDL_RemoveEventWatch(EventWatcher, &state);
    ctx.WaitIdle();

    // Destroy in reverse order
    frame_resources.Destroy();
    swapchain.Destroy();
    vkDestroySurfaceKHR(ctx.Instance(), surface, nullptr);
    // ctx, window cleaned up by RAII guards

    return EXIT_SUCCESS;
}
