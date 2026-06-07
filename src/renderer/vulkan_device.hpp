#pragma once

#include <vulkan/vulkan_raii.hpp>

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
constexpr auto discrete_gpu_score_bonus = 1000U;
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
  if (properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
    score += discrete_gpu_score_bonus;
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

[[nodiscard]] inline auto max_usable_sample_count(const vk::PhysicalDeviceProperties &properties) -> vk::SampleCountFlagBits {
  const vk::SampleCountFlags counts = properties.limits.framebufferColorSampleCounts &
                                      properties.limits.framebufferDepthSampleCounts;

  constexpr std::array candidates{
      vk::SampleCountFlagBits::e64,
      vk::SampleCountFlagBits::e32,
      vk::SampleCountFlagBits::e16,
      vk::SampleCountFlagBits::e8,
      vk::SampleCountFlagBits::e4,
      vk::SampleCountFlagBits::e2,
  };

  for (const auto sample : candidates)
    if ((counts & sample) != vk::SampleCountFlags{})
      return sample;

  return vk::SampleCountFlagBits::e1;
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

class VulkanDevice {
public:
  explicit VulkanDevice(SDL_Window *window) {
    create_instance();
    create_debug_messenger();
    create_surface(window);
    pick_physical_device();
    create_logical_device();
    msaa_samples_ = detail::max_usable_sample_count(physical_device_.getProperties());
  }

  ~VulkanDevice() {
    if (*device_ != nullptr)
      device_.waitIdle();
  }

  VulkanDevice(const VulkanDevice &) = delete;
  auto operator=(const VulkanDevice &) -> VulkanDevice & = delete;
  VulkanDevice(VulkanDevice &&) = delete;
  auto operator=(VulkanDevice &&) -> VulkanDevice & = delete;

  [[nodiscard]] auto instance() const -> const vk::raii::Instance & {
    return instance_;
  }

  [[nodiscard]] auto surface() const -> const vk::raii::SurfaceKHR & {
    return surface_;
  }

  [[nodiscard]] auto device() -> vk::raii::Device & {
    return device_;
  }

  [[nodiscard]] auto physical_device() -> vk::raii::PhysicalDevice & {
    return physical_device_;
  }

  [[nodiscard]] auto graphics_queue() -> vk::raii::Queue & {
    return graphics_queue_;
  }

  [[nodiscard]] auto present_queue() -> vk::raii::Queue & {
    return present_queue_;
  }

  [[nodiscard]] auto queue_families() const -> const QueueFamilyIndices & {
    return queue_families_;
  }

  [[nodiscard]] auto gpu_name() const -> const std::string & {
    return gpu_name_;
  }

  [[nodiscard]] auto msaa_samples() const -> vk::SampleCountFlagBits {
    return msaa_samples_;
  }

  [[nodiscard]] auto validation_enabled() const -> bool {
    return validation_enabled_;
  }

  void wait_idle() const {
    if (*device_ != nullptr)
      device_.waitIdle();
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

    const vk::PhysicalDeviceFeatures available_features = physical_device_.getFeatures();
    if (available_features.samplerAnisotropy == vk::True)
      feature_chain.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy = vk::True;

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

  vk::raii::Context context_;
  vk::raii::Instance instance_{nullptr};
  vk::raii::DebugUtilsMessengerEXT debug_messenger_{nullptr};
  vk::raii::SurfaceKHR surface_{nullptr};
  vk::raii::PhysicalDevice physical_device_{nullptr};
  vk::raii::Device device_{nullptr};
  vk::raii::Queue graphics_queue_{nullptr};
  vk::raii::Queue present_queue_{nullptr};
  QueueFamilyIndices queue_families_{};
  vk::SampleCountFlagBits msaa_samples_{vk::SampleCountFlagBits::e1};
  bool validation_enabled_{};
  bool debug_utils_enabled_{};
  std::string gpu_name_;
};

} // namespace engine
