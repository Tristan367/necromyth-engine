#pragma once

#include "renderer/buffer.hpp"
#include "renderer/depth_image.hpp"
#include "renderer/graphics_pipeline.hpp"
#include "renderer/image_barrier.hpp"
#include "renderer/vertex.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace engine {

namespace detail {

constexpr auto validation_layer = "VK_LAYER_KHRONOS_validation";
constexpr auto invalid_queue_family = std::numeric_limits<std::uint32_t>::max();
constexpr auto max_frames_in_flight = 2U;
constexpr auto discrete_gpu_score_bonus = 1000U;
constexpr auto resize_debounce_ms = 100U;
constexpr vk::PresentModeKHR desired_present_mode = vk::PresentModeKHR::eMailbox;
constexpr std::array required_device_extensions{vk::KHRSwapchainExtensionName};

[[nodiscard]] inline auto has_name(const char *name, const std::vector<vk::LayerProperties> &properties) -> bool {
  return std::ranges::any_of(properties, [name](const vk::LayerProperties &property) {
    return std::string_view(property.layerName) == name;
  });
}

[[nodiscard]] inline auto has_name(const char *name, const std::vector<vk::ExtensionProperties> &properties) -> bool {
  return std::ranges::any_of(properties, [name](const vk::ExtensionProperties &property) {
    return std::string_view(property.extensionName) == name;
  });
}

[[nodiscard]] inline auto supports_all_names(
    const char *const *names,
    std::size_t count,
    const std::vector<vk::ExtensionProperties> &available) -> bool {
  return std::ranges::all_of(names, names + count, [&](const char *name) {
    return has_name(name, available);
  });
}

[[nodiscard]] inline auto score_physical_device(const vk::PhysicalDeviceProperties &properties) -> std::uint32_t {
  std::uint32_t score = 0;
  // Discrete GPUs have a significant performance advantage over integrated.
  if (properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
    score += discrete_gpu_score_bonus;
  // Max texture dimension is a coarse capability tie-breaker between similar devices.
  score += properties.limits.maxImageDimension2D;
  return score;
}

[[nodiscard]] inline auto debug_callback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT,
    const vk::DebugUtilsMessengerCallbackDataEXT *data,
    void *) -> vk::Bool32 {
  if (severity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
    std::cerr << "Vulkan validation: " << data->pMessage << '\n';

  return vk::False;
}

[[nodiscard]] inline auto debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
  return {
      .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                         vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                         vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
      .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                     vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                     vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
      .pfnUserCallback = debug_callback,
  };
}

} // namespace detail

struct QueueFamilyIndices {
  std::uint32_t graphics{detail::invalid_queue_family};
  std::uint32_t compute{detail::invalid_queue_family};
  std::uint32_t transfer{detail::invalid_queue_family};
  std::uint32_t present{detail::invalid_queue_family};

  [[nodiscard]] auto has_graphics_and_present() const -> bool {
    return graphics != detail::invalid_queue_family &&
           present != detail::invalid_queue_family;
  }
};

class VulkanContext {
public:
  explicit VulkanContext(SDL_Window *window) {
    window_ = window;
    create_instance();
    create_debug_messenger();
    create_surface(window);
    pick_physical_device();
    create_logical_device();
    create_swapchain();
    create_swapchain_image_views();
    create_depth_image();
    create_command_pool();
    create_vertex_buffer();
    create_graphics_pipeline();
    create_command_buffers();
    create_sync_objects();

#ifndef NDEBUG
    if (validation_enabled_)
      std::cout << "Vulkan validation: enabled\n";
#endif
  }

  ~VulkanContext() {
    if (*device_ != nullptr)
      device_.waitIdle();
  }

  VulkanContext(const VulkanContext &) = delete;
  auto operator=(const VulkanContext &) -> VulkanContext & = delete;

  VulkanContext(VulkanContext &&) = delete;
  auto operator=(VulkanContext &&) -> VulkanContext & = delete;

  [[nodiscard]] auto gpu_name() const -> const std::string & {
    return gpu_name_;
  }

  void mark_framebuffer_resized() {
    framebuffer_resized_ = true;
    last_resize_ticks_ = SDL_GetTicks();
  }

  void draw_frame() {
    if (framebuffer_resized_) {
      if (SDL_GetTicks() - last_resize_ticks_ < detail::resize_debounce_ms)
        return;

      recreate_swapchain();
      framebuffer_resized_ = false;
      return;
    }

    if (device_.waitForFences(*in_flight_fences_[frame_index_], vk::True, UINT64_MAX) != vk::Result::eSuccess)
      throw std::runtime_error("Failed to wait for frame fence");

    auto [acquire_result, image_index] = swapchain_.acquireNextImage(
        UINT64_MAX,
        *image_available_semaphores_[frame_index_],
        nullptr);

    if (acquire_result == vk::Result::eErrorOutOfDateKHR) {
      recreate_swapchain();
      return;
    }
    if (acquire_result != vk::Result::eSuccess && acquire_result != vk::Result::eSuboptimalKHR)
      throw std::runtime_error("Failed to acquire swapchain image");

    device_.resetFences(*in_flight_fences_[frame_index_]);

    auto &command_buffer = command_buffers_[frame_index_];
    command_buffer.reset();
    record_draw_commands(command_buffer, image_index);

    const vk::SemaphoreSubmitInfo wait_semaphore_info{
        .semaphore = *image_available_semaphores_[frame_index_],
        .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
    };
    const vk::CommandBufferSubmitInfo command_buffer_info{
        .commandBuffer = *command_buffer,
    };
    const vk::SemaphoreSubmitInfo signal_semaphore_info{
        .semaphore = *render_finished_semaphores_[image_index],
        .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
    };
    const vk::SubmitInfo2 submit_info{
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &wait_semaphore_info,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &command_buffer_info,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signal_semaphore_info,
    };

    graphics_queue_.submit2(submit_info, *in_flight_fences_[frame_index_]);

    const vk::PresentInfoKHR present_info{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*render_finished_semaphores_[image_index],
        .swapchainCount = 1,
        .pSwapchains = &*swapchain_,
        .pImageIndices = &image_index,
    };

    const vk::Result present_result = present_queue_.presentKHR(present_info);
    if (present_result == vk::Result::eErrorOutOfDateKHR || present_result == vk::Result::eSuboptimalKHR)
      recreate_swapchain();
    else if (present_result != vk::Result::eSuccess)
      throw std::runtime_error("Failed to present swapchain image");

    frame_index_ = (frame_index_ + 1) % detail::max_frames_in_flight;
  }

private:
  void create_instance() {
    if (!SDL_Vulkan_LoadLibrary(nullptr))
      throw std::runtime_error(std::string("Failed to load Vulkan through SDL: ") + SDL_GetError());

    if (SDL_Vulkan_GetVkGetInstanceProcAddr() == nullptr)
      throw std::runtime_error(std::string("Failed to get vkGetInstanceProcAddr: ") + SDL_GetError());

    validation_enabled_ = false;
#ifndef NDEBUG
    if (!validation_layers_available())
      throw std::runtime_error("Vulkan validation layer not available");
    validation_enabled_ = true;
#endif

    std::uint32_t sdl_extension_count{};
    const char *const *sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&sdl_extension_count);
    if (sdl_extensions == nullptr || sdl_extension_count == 0)
      throw std::runtime_error(std::string("Failed to get SDL Vulkan extensions: ") + SDL_GetError());

    std::vector<const char *> extensions(
        sdl_extensions,
        sdl_extensions + sdl_extension_count);

    const auto available_instance_extensions = context_.enumerateInstanceExtensionProperties();
    for (const char *extension : extensions) {
      if (!detail::has_name(extension, available_instance_extensions))
        throw std::runtime_error(std::string("Required instance extension not supported: ") + extension);
    }

    if (validation_enabled_) {
      if (detail::has_name(vk::EXTDebugUtilsExtensionName, available_instance_extensions)) {
        extensions.push_back(vk::EXTDebugUtilsExtensionName);
        debug_utils_enabled_ = true;
      } else
        std::cerr << "VK_EXT_debug_utils not available; continuing without debug messenger.\n";
    }

    const vk::ApplicationInfo application_info{
        .pApplicationName = "Vulkan C++ Engine",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "Vulkan C++ Engine",
        .engineVersion = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = vk::ApiVersion13,
    };

    const std::array layers{detail::validation_layer};
    auto debug_info = detail::debug_messenger_create_info();

    const vk::InstanceCreateInfo create_info{
        .pNext = debug_utils_enabled_ ? &debug_info : nullptr,
        .pApplicationInfo = &application_info,
        .enabledLayerCount = validation_enabled_ ? static_cast<std::uint32_t>(layers.size()) : 0,
        .ppEnabledLayerNames = validation_enabled_ ? layers.data() : nullptr,
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    instance_ = vk::raii::Instance(context_, create_info);
  }

  void create_debug_messenger() {
    if (!validation_enabled_ || !debug_utils_enabled_)
      return;

    auto create_info = detail::debug_messenger_create_info();
    debug_messenger_ = vk::raii::DebugUtilsMessengerEXT(instance_, create_info);
  }

  void create_surface(SDL_Window *window) {
    VkSurfaceKHR raw_surface{};
    if (!SDL_Vulkan_CreateSurface(window, *instance_, nullptr, &raw_surface))
      throw std::runtime_error(std::string("Failed to create Vulkan surface: ") + SDL_GetError());

    surface_ = vk::raii::SurfaceKHR(instance_, raw_surface);
  }

  void pick_physical_device() {
    auto devices = instance_.enumeratePhysicalDevices();
    if (devices.empty())
      throw std::runtime_error("No Vulkan-capable GPU found");

    struct Candidate {
      std::uint32_t score{};
      vk::raii::PhysicalDevice device{nullptr};
      std::string name;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(devices.size());

    for (auto &device : devices) {
      if (!is_device_suitable(device))
        continue;

      const vk::PhysicalDeviceProperties properties = device.getProperties();
      candidates.push_back({detail::score_physical_device(properties), device, properties.deviceName.data()});
    }

    if (candidates.empty())
      throw std::runtime_error("No suitable Vulkan 1.3 GPU found");

    const auto best = std::ranges::max_element(candidates, {}, &Candidate::score);
    physical_device_ = best->device;
    queue_families_ = find_queue_families(physical_device_);
    gpu_name_ = best->name;
  }

  void create_logical_device() {
    std::set unique_queue_families{queue_families_.graphics, queue_families_.present};
    std::vector<vk::DeviceQueueCreateInfo> queue_infos;
    queue_infos.reserve(unique_queue_families.size());

    constexpr float queue_priority = 1.0F;
    for (const auto family : unique_queue_families)
      queue_infos.push_back({
          .queueFamilyIndex = family,
          .queueCount = 1,
          .pQueuePriorities = &queue_priority,
      });

    vk::StructureChain<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> feature_chain{
        {},
        {.shaderDrawParameters = vk::True},
        {.synchronization2 = vk::True, .dynamicRendering = vk::True},
        {.extendedDynamicState = vk::True},
    };

    const std::array device_extensions{vk::KHRSwapchainExtensionName};

    const vk::DeviceCreateInfo create_info{
        .pNext = &feature_chain.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount = static_cast<std::uint32_t>(queue_infos.size()),
        .pQueueCreateInfos = queue_infos.data(),
        .enabledExtensionCount = static_cast<std::uint32_t>(device_extensions.size()),
        .ppEnabledExtensionNames = device_extensions.data(),
    };

    device_ = vk::raii::Device(physical_device_, create_info);
    graphics_queue_ = vk::raii::Queue(device_, queue_families_.graphics, 0);
    present_queue_ = vk::raii::Queue(device_, queue_families_.present, 0);
  }

  [[nodiscard]] static auto choose_swap_min_image_count(const vk::SurfaceCapabilitiesKHR &capabilities) -> std::uint32_t {
    auto image_count = std::max(3U, capabilities.minImageCount);
    if (capabilities.maxImageCount > 0)
      image_count = std::min(image_count, capabilities.maxImageCount);
    return image_count;
  }

  [[nodiscard]] auto choose_swapchain_surface_format(
      const std::vector<vk::SurfaceFormatKHR> &formats) const -> vk::SurfaceFormatKHR {
    const auto preferred = std::ranges::find_if(formats, [](const vk::SurfaceFormatKHR &format) {
      return format.format == vk::Format::eB8G8R8A8Srgb &&
             format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
    });

    return preferred != formats.end() ? *preferred : formats.front();
  }

  [[nodiscard]] auto choose_present_mode(const std::vector<vk::PresentModeKHR> &modes) const -> vk::PresentModeKHR {
    if (std::ranges::find(modes, detail::desired_present_mode) != modes.end())
      return detail::desired_present_mode;

    return vk::PresentModeKHR::eFifo;
  }

  [[nodiscard]] auto choose_swapchain_extent(const vk::SurfaceCapabilitiesKHR &capabilities) const -> vk::Extent2D {
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
      return capabilities.currentExtent;

    int width{};
    int height{};
    if (!SDL_GetWindowSizeInPixels(window_, &width, &height))
      throw std::runtime_error(std::string("Failed to get SDL window pixel size: ") + SDL_GetError());

    return {
        .width = std::clamp(
            static_cast<std::uint32_t>(std::max(width, 1)),
            capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width),
        .height = std::clamp(
            static_cast<std::uint32_t>(std::max(height, 1)),
            capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height),
    };
  }

  void create_swapchain(vk::SwapchainKHR old_swapchain = nullptr) {
    const vk::SurfaceCapabilitiesKHR capabilities = physical_device_.getSurfaceCapabilitiesKHR(*surface_);
    const std::vector<vk::SurfaceFormatKHR> formats = physical_device_.getSurfaceFormatsKHR(*surface_);
    const std::vector<vk::PresentModeKHR> present_modes = physical_device_.getSurfacePresentModesKHR(*surface_);

    const vk::SurfaceFormatKHR surface_format = choose_swapchain_surface_format(formats);
    swapchain_image_format_ = surface_format.format;
    swapchain_extent_ = choose_swapchain_extent(capabilities);

    std::uint32_t image_count = choose_swap_min_image_count(capabilities);

    const std::array queue_family_indices{queue_families_.graphics, queue_families_.present};
    const bool separate_present_queue = queue_families_.graphics != queue_families_.present;

    const vk::SwapchainCreateInfoKHR create_info{
        .surface = *surface_,
        .minImageCount = image_count,
        .imageFormat = swapchain_image_format_,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = swapchain_extent_,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = separate_present_queue ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = separate_present_queue ? static_cast<std::uint32_t>(queue_family_indices.size()) : 0,
        .pQueueFamilyIndices = separate_present_queue ? queue_family_indices.data() : nullptr,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = choose_present_mode(present_modes),
        .clipped = vk::True,
        .oldSwapchain = old_swapchain,
    };

    swapchain_ = vk::raii::SwapchainKHR(device_, create_info);
    swapchain_images_ = swapchain_.getImages();
  }

  void create_swapchain_image_views() {
    swapchain_image_views_.clear();
    swapchain_image_views_.reserve(swapchain_images_.size());

    vk::ImageViewCreateInfo create_info{
        .viewType = vk::ImageViewType::e2D,
        .format = swapchain_image_format_,
        .components = {},
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    for (const vk::Image image : swapchain_images_) {
      create_info.image = image;
      swapchain_image_views_.emplace_back(device_, create_info);
    }
  }

  void create_depth_image() {
    depth_image_.create(physical_device_, device_, swapchain_extent_);
  }

  void create_vertex_buffer() {
    const auto vertices = triangle_vertices();
    vertex_count_ = static_cast<std::uint32_t>(vertices.size());
    vertex_buffer_.upload(
        physical_device_,
        device_,
        command_pool_,
        graphics_queue_,
        static_cast<vk::DeviceSize>(sizeof(ColoredVertex) * vertices.size()),
        vk::BufferUsageFlagBits::eVertexBuffer,
        vertices.data());
  }

  void create_graphics_pipeline() {
    const auto attributes = ColoredVertex::attribute_descriptions();
    graphics_pipeline_.create(
        device_,
        swapchain_image_format_,
        depth_image_.format(),
        ENGINE_SHADER_SPIRV,
        ColoredVertex::binding_description(),
        attributes);
  }

  void create_command_pool() {
    const vk::CommandPoolCreateInfo create_info{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queue_families_.graphics,
    };

    command_pool_ = vk::raii::CommandPool(device_, create_info);
  }

  void create_command_buffers() {
    const vk::CommandBufferAllocateInfo allocate_info{
        .commandPool = *command_pool_,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = detail::max_frames_in_flight,
    };

    command_buffers_ = vk::raii::CommandBuffers(device_, allocate_info);
  }

  void create_render_finished_semaphores() {
    render_finished_semaphores_.clear();
    render_finished_semaphores_.reserve(swapchain_images_.size());
    for (std::size_t i = 0; i < swapchain_images_.size(); ++i)
      render_finished_semaphores_.emplace_back(device_, vk::SemaphoreCreateInfo{});
  }

  void create_sync_objects() {
    create_render_finished_semaphores();

    image_available_semaphores_.clear();
    in_flight_fences_.clear();

    image_available_semaphores_.reserve(detail::max_frames_in_flight);
    in_flight_fences_.reserve(detail::max_frames_in_flight);
    for (std::size_t i = 0; i < detail::max_frames_in_flight; ++i) {
      image_available_semaphores_.emplace_back(device_, vk::SemaphoreCreateInfo{});
      in_flight_fences_.emplace_back(device_, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
    }
  }

  void record_draw_commands(vk::raii::CommandBuffer &command_buffer, std::uint32_t image_index) const {
    command_buffer.begin({});

    transition_image_layout(
        command_buffer,
        swapchain_images_[image_index],
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput);

    transition_image_layout(
        command_buffer,
        depth_image_.image(),
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthAttachmentOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        depth_image_.aspect_mask());

    const vk::ClearValue clear_value{vk::ClearColorValue{std::array{0.02F, 0.03F, 0.05F, 1.0F}}};
    const vk::ClearValue clear_depth{vk::ClearDepthStencilValue{1.0F, 0}};
    const vk::RenderingAttachmentInfo color_attachment{
        .imageView = *swapchain_image_views_[image_index],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clear_value,
    };
    const vk::RenderingAttachmentInfo depth_attachment{
        .imageView = *depth_image_.view(),
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = clear_depth,
    };
    const vk::RenderingInfo rendering_info{
        .renderArea = {.offset = {.x = 0, .y = 0}, .extent = swapchain_extent_},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment,
        .pDepthAttachment = &depth_attachment,
    };
    command_buffer.beginRendering(rendering_info);
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphics_pipeline_.pipeline());
    const vk::Buffer vertex_buffers[] = {vertex_buffer_.handle()};
    const vk::DeviceSize offsets[] = {0};
    command_buffer.bindVertexBuffers(0, vertex_buffers, offsets);
    command_buffer.setViewport(
        0,
        vk::Viewport{
            0.0F,
            0.0F,
            static_cast<float>(swapchain_extent_.width),
            static_cast<float>(swapchain_extent_.height),
            0.0F,
            1.0F,
        });
    command_buffer.setScissor(0, vk::Rect2D{{0, 0}, swapchain_extent_});
    command_buffer.draw(vertex_count_, 1, 0, 0);
    command_buffer.endRendering();

    transition_image_layout(
        command_buffer,
        swapchain_images_[image_index],
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        {},
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eBottomOfPipe);

    command_buffer.end();
  }

  void wait_for_nonzero_extent() const {
    int width{};
    int height{};
    while (true) {
      if (!SDL_GetWindowSizeInPixels(window_, &width, &height))
        throw std::runtime_error(std::string("Failed to get SDL window pixel size: ") + SDL_GetError());

      if (width > 0 && height > 0)
        return;

      SDL_WaitEvent(nullptr);
    }
  }

  void recreate_swapchain() {
    wait_for_nonzero_extent();
    device_.waitIdle();
    swapchain_image_views_.clear();
    render_finished_semaphores_.clear();

    const vk::SwapchainKHR old_swapchain = *swapchain_;
    create_swapchain(old_swapchain);
    create_swapchain_image_views();
    depth_image_.recreate(swapchain_extent_);
    create_render_finished_semaphores();
  }

  [[nodiscard]] auto validation_layers_available() const -> bool {
    return detail::has_name(detail::validation_layer, context_.enumerateInstanceLayerProperties());
  }

  [[nodiscard]] auto find_queue_families(const vk::raii::PhysicalDevice &device) const -> QueueFamilyIndices {
    const std::vector<vk::QueueFamilyProperties> queue_families = device.getQueueFamilyProperties();

    QueueFamilyIndices indices;
    for (std::uint32_t i = 0; i < queue_families.size(); ++i) {
      const auto flags = queue_families[i].queueFlags;

      if ((flags & vk::QueueFlagBits::eGraphics) != vk::QueueFlags{} &&
          indices.graphics == detail::invalid_queue_family)
        indices.graphics = i;

      if ((flags & vk::QueueFlagBits::eCompute) != vk::QueueFlags{} &&
          indices.compute == detail::invalid_queue_family)
        indices.compute = i;

      if ((flags & vk::QueueFlagBits::eTransfer) != vk::QueueFlags{} &&
          indices.transfer == detail::invalid_queue_family)
        indices.transfer = i;

      if (device.getSurfaceSupportKHR(i, *surface_) &&
          indices.present == detail::invalid_queue_family)
        indices.present = i;
    }

    for (std::uint32_t i = 0; i < queue_families.size(); ++i) {
      const auto flags = queue_families[i].queueFlags;
      const bool graphics = (flags & vk::QueueFlagBits::eGraphics) != vk::QueueFlags{};
      const bool compute = (flags & vk::QueueFlagBits::eCompute) != vk::QueueFlags{};
      const bool transfer = (flags & vk::QueueFlagBits::eTransfer) != vk::QueueFlags{};

      if (graphics && compute && transfer && device.getSurfaceSupportKHR(i, *surface_)) {
        indices.graphics = indices.compute = indices.transfer = indices.present = i;
        break;
      }
    }

    return indices;
  }

  [[nodiscard]] auto has_adequate_swapchain_support(const vk::raii::PhysicalDevice &device) const -> bool {
    return !device.getSurfaceFormatsKHR(*surface_).empty() &&
           !device.getSurfacePresentModesKHR(*surface_).empty();
  }

  [[nodiscard]] auto device_supports_required_extensions(const vk::raii::PhysicalDevice &device) const -> bool {
    const auto available = device.enumerateDeviceExtensionProperties();
    return detail::supports_all_names(
        detail::required_device_extensions.data(),
        detail::required_device_extensions.size(),
        available);
  }

  [[nodiscard]] auto device_supports_required_features(const vk::raii::PhysicalDevice &device) const -> bool {
    const auto features = device.getFeatures2<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();

    return features.get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters == vk::True &&
           features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering == vk::True &&
           features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 == vk::True &&
           features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState == vk::True;
  }

  [[nodiscard]] auto is_device_suitable(const vk::raii::PhysicalDevice &device) const -> bool {
    if (device.getProperties().apiVersion < vk::ApiVersion13)
      return false;

    const auto indices = find_queue_families(device);
    return indices.has_graphics_and_present() &&
           device_supports_required_extensions(device) &&
           has_adequate_swapchain_support(device) &&
           device_supports_required_features(device);
  }

  SDL_Window *window_{};
  vk::raii::Context context_;
  vk::raii::Instance instance_{nullptr};
  vk::raii::DebugUtilsMessengerEXT debug_messenger_{nullptr};
  vk::raii::SurfaceKHR surface_{nullptr};
  vk::raii::PhysicalDevice physical_device_{nullptr};
  vk::raii::Device device_{nullptr};
  vk::raii::Queue graphics_queue_{nullptr};
  vk::raii::Queue present_queue_{nullptr};
  vk::raii::SwapchainKHR swapchain_{nullptr};
  vk::Format swapchain_image_format_{vk::Format::eUndefined};
  vk::Extent2D swapchain_extent_{};
  std::vector<vk::Image> swapchain_images_;
  std::vector<vk::raii::ImageView> swapchain_image_views_;
  vk::raii::CommandPool command_pool_{nullptr};
  vk::raii::CommandBuffers command_buffers_{nullptr};
  std::vector<vk::raii::Semaphore> image_available_semaphores_;
  std::vector<vk::raii::Semaphore> render_finished_semaphores_;
  std::vector<vk::raii::Fence> in_flight_fences_;
  DepthImage depth_image_;
  DeviceLocalBuffer vertex_buffer_;
  std::uint32_t vertex_count_{};
  GraphicsPipeline graphics_pipeline_;
  std::uint32_t frame_index_{};
  QueueFamilyIndices queue_families_{};
  bool validation_enabled_{};
  bool debug_utils_enabled_{};
  bool framebuffer_resized_{};
  std::uint64_t last_resize_ticks_{};
  std::string gpu_name_;
};

} // namespace engine
