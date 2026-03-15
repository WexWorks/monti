#include "UiRenderer.h"
#include "../core/vulkan_context.h"
#include "swapchain.h"

#include <volk.h>

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <misc/freetype/imgui_freetype.h>

#include <SDL3/SDL.h>

#include <cstdio>

namespace monti::app {

UiRenderer::~UiRenderer() {
    Destroy();
}

bool UiRenderer::Initialize(VulkanContext& ctx, SDL_Window* window, const Swapchain& swapchain) {
    ctx_ = &ctx;

    if (!CreateRenderPass(swapchain.ImageFormat())) return false;
    if (!CreateFramebuffers(swapchain)) return false;

    // Descriptor pool for ImGui
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_size.descriptorCount = 100;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 100;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    if (vkCreateDescriptorPool(ctx.Device(), &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to create ImGui descriptor pool\n");
        return false;
    }

    // Initialize Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Dark style
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.PopupRounding = 4.0f;
    style.WindowBorderSize = 0.0f;
    style.FrameBorderSize = 0.0f;

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg]         = ImVec4(0.15f, 0.15f, 0.15f, 0.85f);
    colors[ImGuiCol_PopupBg]          = ImVec4(0.15f, 0.15f, 0.15f, 0.90f);
    colors[ImGuiCol_Text]             = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    colors[ImGuiCol_FrameBg]          = ImVec4(0.25f, 0.25f, 0.25f, 0.70f);
    colors[ImGuiCol_FrameBgHovered]   = ImVec4(0.30f, 0.30f, 0.30f, 0.80f);
    colors[ImGuiCol_FrameBgActive]    = ImVec4(0.35f, 0.35f, 0.35f, 0.90f);
    colors[ImGuiCol_SliderGrab]       = ImVec4(0.70f, 0.70f, 0.70f, 1.0f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.90f, 0.90f, 0.90f, 1.0f);
    colors[ImGuiCol_Header]           = ImVec4(0.30f, 0.30f, 0.30f, 0.50f);
    colors[ImGuiCol_HeaderHovered]    = ImVec4(0.35f, 0.35f, 0.35f, 0.60f);
    colors[ImGuiCol_HeaderActive]     = ImVec4(0.40f, 0.40f, 0.40f, 0.70f);

    // Load Inter font with FreeType
    {
        constexpr float kFontSize = 16.0f;
        ImFontConfig config;
        config.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LightHinting;

#ifdef MONTI_FONT_DIR
        auto* font = io.Fonts->AddFontFromFileTTF(
            MONTI_FONT_DIR "/Inter-Regular.ttf", kFontSize, &config);
        if (!font) {
            std::fprintf(stderr, "Warning: failed to load Inter font, using default\n");
            io.Fonts->AddFontDefault();
        }
#else
        io.Fonts->AddFontDefault();
#endif
    }

    // Initialize ImGui SDL3 backend
    ImGui_ImplSDL3_InitForVulkan(window);

    // Load Vulkan function pointers for ImGui (required with VK_NO_PROTOTYPES / volk)
    ImGui_ImplVulkan_LoadFunctions([](const char* name, void* user_data) {
        return vkGetInstanceProcAddr(static_cast<VkInstance>(user_data), name);
    }, ctx.Instance());

    // Initialize ImGui Vulkan backend
    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = ctx.Instance();
    init_info.PhysicalDevice = ctx.PhysicalDevice();
    init_info.Device = ctx.Device();
    init_info.QueueFamily = ctx.QueueFamilyIndex();
    init_info.Queue = ctx.GraphicsQueue();
    init_info.DescriptorPool = descriptor_pool_;
    init_info.MinImageCount = swapchain.ImageCount();
    init_info.ImageCount = swapchain.ImageCount();
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.RenderPass = render_pass_;
    init_info.Subpass = 0;

    ImGui_ImplVulkan_Init(&init_info);

    std::printf("ImGui initialized (version %s)\n", IMGUI_VERSION);
    return true;
}

void UiRenderer::Destroy() {
    if (!ctx_) return;

    vkDeviceWaitIdle(ctx_->Device());

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    DestroyFramebuffers();

    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx_->Device(), descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
    }

    if (render_pass_ != VK_NULL_HANDLE) {
        vkDestroyRenderPass(ctx_->Device(), render_pass_, nullptr);
        render_pass_ = VK_NULL_HANDLE;
    }

    ctx_ = nullptr;
}

bool UiRenderer::Resize(const Swapchain& swapchain) {
    DestroyFramebuffers();
    if (!CreateFramebuffers(swapchain)) return false;
    ImGui_ImplVulkan_SetMinImageCount(swapchain.ImageCount());
    return true;
}

bool UiRenderer::ProcessEvent(const SDL_Event& event) {
    return ImGui_ImplSDL3_ProcessEvent(&event);
}

void UiRenderer::BeginFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

void UiRenderer::EndFrame(VkCommandBuffer cmd, uint32_t image_index) {
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();

    VkRenderPassBeginInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = render_pass_;
    render_pass_info.framebuffer = framebuffers_[image_index];
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = {
        static_cast<uint32_t>(draw_data->DisplaySize.x),
        static_cast<uint32_t>(draw_data->DisplaySize.y)
    };

    vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(draw_data, cmd);
    vkCmdEndRenderPass(cmd);
}

bool UiRenderer::WantCaptureMouse() const {
    return ImGui::GetIO().WantCaptureMouse;
}

bool UiRenderer::WantCaptureKeyboard() const {
    return ImGui::GetIO().WantCaptureKeyboard;
}

bool UiRenderer::CreateRenderPass(VkFormat swapchain_format) {
    VkAttachmentDescription attachment{};
    attachment.format = swapchain_format;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    // Swapchain image arrives in TRANSFER_DST_OPTIMAL (after blit from tone mapper)
    attachment.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    // Departs in PRESENT_SRC_KHR for presentation
    attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_ref{};
    color_ref.attachment = 0;
    color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;

    // Wait for the blit (transfer) to complete before rendering
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dependency.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    create_info.attachmentCount = 1;
    create_info.pAttachments = &attachment;
    create_info.subpassCount = 1;
    create_info.pSubpasses = &subpass;
    create_info.dependencyCount = 1;
    create_info.pDependencies = &dependency;

    if (vkCreateRenderPass(ctx_->Device(), &create_info, nullptr, &render_pass_) != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to create ImGui render pass\n");
        return false;
    }

    return true;
}

bool UiRenderer::CreateFramebuffers(const Swapchain& swapchain) {
    framebuffers_.resize(swapchain.ImageCount());

    for (uint32_t i = 0; i < swapchain.ImageCount(); ++i) {
        VkImageView attachment = swapchain.ImageView(i);

        VkFramebufferCreateInfo fb_info{};
        fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass = render_pass_;
        fb_info.attachmentCount = 1;
        fb_info.pAttachments = &attachment;
        fb_info.width = swapchain.Extent().width;
        fb_info.height = swapchain.Extent().height;
        fb_info.layers = 1;

        if (vkCreateFramebuffer(ctx_->Device(), &fb_info, nullptr, &framebuffers_[i]) != VK_SUCCESS) {
            std::fprintf(stderr, "Failed to create ImGui framebuffer %u\n", i);
            return false;
        }
    }

    return true;
}

void UiRenderer::DestroyFramebuffers() {
    if (!ctx_) return;
    for (auto fb : framebuffers_) {
        if (fb != VK_NULL_HANDLE)
            vkDestroyFramebuffer(ctx_->Device(), fb, nullptr);
    }
    framebuffers_.clear();
}

}  // namespace monti::app
