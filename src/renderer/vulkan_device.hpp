#pragma once

#include "renderer/render_settings.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <SDL3/SDL_vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
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
constexpr const char *portability_subset_extension = "VK_KHR_portability_subset";

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

[[nodiscard]] inline auto score_physical_device(
    const vk::PhysicalDeviceProperties &properties,
    const vk::PhysicalDeviceMemoryProperties &memory) -> std::uint32_t {
  std::uint32_t score = 0;
  if (properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
    score += discrete_gpu_score_bonus;
  score += properties.limits.maxImageDimension2D;

  for (std::uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
    if ((memory.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) != vk::MemoryHeapFlags{})
      score += static_cast<std::uint32_t>(memory.memoryHeaps[i].size / (1024ULL * 1024ULL * 1024ULL));
  }

  return score;
}

inline void warn_if_extension_missing(
    std::string_view name,
    const std::vector<vk::ExtensionProperties> &available) {
  if (!has_name(name.data(), available))
    std::cerr << "Vulkan: optional extension not available: " << name << '\n';
}

// Counted so a headless run can fail on validation output instead of printing it
// into a log nobody reads. See VulkanDevice::validation_messages().
inline std::atomic<std::uint64_t> g_validation_messages{0};

[[nodiscard]] inline auto debug_callback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT,
    const vk::DebugUtilsMessengerCallbackDataEXT *data,
    void *) -> vk::Bool32 {
  if (severity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
    g_validation_messages.fetch_add(1, std::memory_order_relaxed);
    std::cerr << "Vulkan validation: " << data->pMessage << '\n';
  }

  return vk::False;
}

// Registers only what the callback actually prints. It used to ask for eVerbose
// as well and then drop it on the floor: the layer still formatted every one of
// those messages, which is real work for a string nobody sees.
[[nodiscard]] inline auto debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
  return {
      .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                         vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
      .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                     vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                     vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
      .pfnUserCallback = debug_callback,
  };
}

// Whether the environment asks for validation, and how much of it.
//
// Debug builds validate by default, as before. The reason this is an
// environment variable at all is that the interesting bugs -- races between
// passes, a barrier that does not cover the hazard it was written for -- only
// show up under the load a Release build produces, and a Debug build is too
// slow to reach it. ENGINE_VALIDATION=1 turns the layer on in any build.
struct ValidationRequest {
  bool enabled{false};
  bool synchronization{false};
};

[[nodiscard]] inline auto validation_request_from_environment() -> ValidationRequest {
  ValidationRequest request{};
#ifndef NDEBUG
  request.enabled = true;
#endif
  if (const char *env = std::getenv("ENGINE_VALIDATION"); env != nullptr && env[0] != '\0')
    request.enabled = env[0] != '0';

  // Synchronization validation is the half that catches what reading cannot.
  // Core validation checks that each call is legal in isolation; this one
  // tracks what every access reads and writes and reports the hazards between
  // them -- a shadow map sampled before its pass finished writing it, a staging
  // copy racing the draw that consumes it. Those render correctly on the card
  // that happened to schedule them in order and corrupt on the next one.
  //
  // It is off by default because it is expensive: it shadows every resource
  // access in the frame.
  if (const char *env = std::getenv("ENGINE_SYNC_VALIDATION"); env != nullptr && env[0] != '\0') {
    request.synchronization = env[0] != '0';
    if (request.synchronization)
      request.enabled = true;
  }
  return request;
}

[[nodiscard]] inline auto max_usable_sample_count(const vk::PhysicalDeviceProperties &properties) -> vk::SampleCountFlagBits {
  return highest_sample_count(supported_framebuffer_sample_counts(properties));
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
  // Whether the frame loop may use presentId / vkWaitForPresentKHR.
  [[nodiscard]] auto present_wait_supported() const -> bool { return present_id_supported_; }

  explicit VulkanDevice(
      SDL_Window *window,
      MsaaSettings msaa_settings = {},
      std::optional<std::uint32_t> gpu_device_index = {}) {
    gpu_device_index_ = gpu_device_index;
    create_instance();
    create_debug_messenger();
    create_surface(window);
    pick_physical_device();
    create_logical_device();
    max_msaa_samples_ = detail::max_usable_sample_count(physical_device_.getProperties());
    const vk::SampleCountFlags supported_sample_counts =
        detail::supported_framebuffer_sample_counts(physical_device_.getProperties());
    msaa_samples_ = detail::resolve_msaa_samples(msaa_settings, supported_sample_counts);
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

  [[nodiscard]] auto physical_device_handle() const -> vk::PhysicalDevice {
    return *physical_device_;
  }

  [[nodiscard]] auto device_handle() const -> vk::Device {
    return *device_;
  }

  [[nodiscard]] auto graphics_queue_handle() const -> vk::Queue {
    return *graphics_queue_;
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

  [[nodiscard]] auto max_msaa_samples() const -> vk::SampleCountFlagBits {
    return max_msaa_samples_;
  }

  [[nodiscard]] auto msaa_enabled() const -> bool {
    return msaa_is_active(msaa_samples_);
  }

  [[nodiscard]] auto validation_enabled() const -> bool {
    return validation_enabled_;
  }

  [[nodiscard]] auto sync_validation_enabled() const -> bool {
    return sync_validation_enabled_;
  }

  // Warnings and errors the layer has reported since process start. A test that
  // renders frames can assert this stayed at zero, which turns validation from
  // something you have to read into something that fails the build.
  [[nodiscard]] static auto validation_messages() -> std::uint64_t {
    return detail::g_validation_messages.load(std::memory_order_relaxed);
  }

  void wait_idle() const {
    if (*device_ != nullptr)
      device_.waitIdle();
  }

private:
  bool present_id_supported_{false};
  void create_instance() {
    if (!SDL_Vulkan_LoadLibrary(nullptr))
      throw std::runtime_error(std::string("Failed to load Vulkan through SDL: ") + SDL_GetError());

    if (SDL_Vulkan_GetVkGetInstanceProcAddr() == nullptr)
      throw std::runtime_error(std::string("Failed to get vkGetInstanceProcAddr: ") + SDL_GetError());

    const detail::ValidationRequest validation = detail::validation_request_from_environment();
    validation_enabled_ = false;
    if (validation.enabled) {
      if (validation_layers_available()) {
        validation_enabled_ = true;
        sync_validation_enabled_ = validation.synchronization;
      } else {
#ifndef NDEBUG
        // A debug build without the layer is a broken development environment,
        // not a runtime condition to work around.
        throw std::runtime_error("Vulkan validation layer not available");
#else
        std::cerr << "ENGINE_VALIDATION requested but VK_LAYER_KHRONOS_validation "
                     "is not installed; continuing without it.\n";
#endif
      }
    }

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

    if (sync_validation_enabled_) {
      // VK_EXT_validation_features is implemented BY the validation layer, so it
      // is absent from the unlayered extension list above and has to be looked
      // up against the layer itself. Enabling it without that check is how you
      // get an instance that silently ignores the feature switch.
      const auto layer_extensions =
          context_.enumerateInstanceExtensionProperties(std::string(detail::validation_layer));
      if (detail::has_name(vk::EXTValidationFeaturesExtensionName, layer_extensions) ||
          detail::has_name(vk::EXTValidationFeaturesExtensionName, available_instance_extensions)) {
        extensions.push_back(vk::EXTValidationFeaturesExtensionName);
      } else {
        std::cerr << "VK_EXT_validation_features not available; synchronization "
                     "validation cannot be enabled.\n";
        sync_validation_enabled_ = false;
      }
    }

    vk::InstanceCreateFlags instance_flags{};
#if defined(__APPLE__)
    if (detail::has_name(vk::KHRPortabilityEnumerationExtensionName, available_instance_extensions)) {
      extensions.push_back(vk::KHRPortabilityEnumerationExtensionName);
      instance_flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
      portability_enumeration_enabled_ = true;
    } else {
      detail::warn_if_extension_missing(vk::KHRPortabilityEnumerationExtensionName, available_instance_extensions);
    }
#else
    (void)available_instance_extensions;
#endif

    const vk::ApplicationInfo application_info{
        .pApplicationName = "Necromyth Engine",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "Necromyth Engine",
        .engineVersion = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = vk::ApiVersion13,
    };

    const std::array layers{detail::validation_layer};
    auto debug_info = detail::debug_messenger_create_info();

    // Chained ahead of the debug messenger so the layer is configured before it
    // reports anything. Both hang off pNext; order between them does not matter
    // to the loader, but keeping the feature switch first reads correctly.
    constexpr std::array sync_features{
        vk::ValidationFeatureEnableEXT::eSynchronizationValidation,
    };
    vk::ValidationFeaturesEXT validation_features{
        .enabledValidationFeatureCount = static_cast<std::uint32_t>(sync_features.size()),
        .pEnabledValidationFeatures = sync_features.data(),
    };

    const void *next = debug_utils_enabled_ ? static_cast<const void *>(&debug_info) : nullptr;
    if (sync_validation_enabled_) {
      validation_features.pNext = next;
      next = &validation_features;
    }

    const vk::InstanceCreateInfo create_info{
        .pNext = next,
        .flags = instance_flags,
        .pApplicationInfo = &application_info,
        .enabledLayerCount = validation_enabled_ ? static_cast<std::uint32_t>(layers.size()) : 0,
        .ppEnabledLayerNames = validation_enabled_ ? layers.data() : nullptr,
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    instance_ = vk::raii::Instance(context_, create_info);

    if (validation_enabled_)
      std::cout << "Vulkan validation: on"
                << (sync_validation_enabled_ ? " (+ synchronization)" : "")
                << " (ENGINE_VALIDATION / ENGINE_SYNC_VALIDATION)\n";
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

    if (gpu_device_index_) {
      if (*gpu_device_index_ >= devices.size())
        throw std::runtime_error("Requested GPU index " + std::to_string(*gpu_device_index_) + " is out of range");

      if (!is_device_suitable(devices[*gpu_device_index_]))
        throw std::runtime_error("Requested GPU is not suitable for this engine");

      physical_device_ = devices[*gpu_device_index_];
      queue_families_ = find_queue_families(physical_device_);
      gpu_name_ = physical_device_.getProperties().deviceName.data();
      return;
    }

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
      const vk::PhysicalDeviceMemoryProperties memory = device.getMemoryProperties();
      candidates.push_back({detail::score_physical_device(properties, memory), device, properties.deviceName.data()});
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
        vk::PhysicalDevicePresentIdFeaturesKHR,
        vk::PhysicalDevicePresentWaitFeaturesKHR> feature_chain{
        {},
        // multiview: single-pass 6-face point-shadow cubemap rendering.
        {.multiview = vk::True, .shaderDrawParameters = vk::True},
        {.synchronization2 = vk::True, .dynamicRendering = vk::True},
        {.presentId = vk::False},
        {.presentWait = vk::False},
    };

    const vk::PhysicalDeviceFeatures available_features = physical_device_.getFeatures();
    if (available_features.samplerAnisotropy == vk::True)
      feature_chain.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy = vk::True;
    // Required for dual-paraboloid point shadows: the depth VS writes
    // SV_ClipDistance0 to discard the far paraboloid hemisphere.
    if (available_features.shaderClipDistance == vk::True)
      feature_chain.get<vk::PhysicalDeviceFeatures2>().features.shaderClipDistance = vk::True;
    if (available_features.imageCubeArray == vk::True)
      feature_chain.get<vk::PhysicalDeviceFeatures2>().features.imageCubeArray = vk::True;

    std::vector<const char *> device_extensions{vk::KHRSwapchainExtensionName};
    const auto available_device_extensions = physical_device_.enumerateDeviceExtensionProperties();

    // VK_KHR_present_id / VK_KHR_present_wait: the only way to learn when a
    // frame was actually SHOWN.
    //
    // Everything else the frame loop can measure is on this side of the
    // compositor -- when we submitted, when acquire returned. None of it says
    // where the vblank was, so a loop that is drifting out of phase with the
    // display looks identical to one that is not, right up until it misses a
    // refresh and the frame time doubles. Present-wait closes that: present
    // with an id, then block until that id has been presented, and the return
    // is a real vblank timestamp.
    //
    // Both are optional. Without them the loop behaves exactly as before.
    present_id_supported_ =
        detail::has_name(vk::KHRPresentIdExtensionName, available_device_extensions) &&
        detail::has_name(vk::KHRPresentWaitExtensionName, available_device_extensions);
    if (present_id_supported_) {
      device_extensions.push_back(vk::KHRPresentIdExtensionName);
      device_extensions.push_back(vk::KHRPresentWaitExtensionName);
      feature_chain.get<vk::PhysicalDevicePresentIdFeaturesKHR>().presentId = vk::True;
      feature_chain.get<vk::PhysicalDevicePresentWaitFeaturesKHR>().presentWait = vk::True;
    } else {
      // Asking for a feature whose extension is not enabled is invalid, so the
      // structures come out of the chain entirely rather than being left off.
      feature_chain.unlink<vk::PhysicalDevicePresentWaitFeaturesKHR>();
      feature_chain.unlink<vk::PhysicalDevicePresentIdFeaturesKHR>();
    }

    if (portability_enumeration_enabled_) {
      if (detail::has_name(detail::portability_subset_extension, available_device_extensions))
        device_extensions.push_back(detail::portability_subset_extension);
      else
        detail::warn_if_extension_missing(detail::portability_subset_extension, available_device_extensions);
    }

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
        vk::PhysicalDeviceVulkan13Features>();

    // Multiview drives the single-pass point-shadow cubemap. Vulkan 1.1
    // guarantees maxMultiviewViewCount >= 6; assert it against the device.
    const auto mv_props = device.getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceMultiviewProperties>()
        .get<vk::PhysicalDeviceMultiviewProperties>();

    return features.get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters == vk::True &&
           features.get<vk::PhysicalDeviceVulkan11Features>().multiview == vk::True &&
           mv_props.maxMultiviewViewCount >= 6 &&
           features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering == vk::True &&
           features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 == vk::True;
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
  vk::SampleCountFlagBits max_msaa_samples_{vk::SampleCountFlagBits::e1};
  bool validation_enabled_{};
  bool sync_validation_enabled_{};
  bool debug_utils_enabled_{};
  bool portability_enumeration_enabled_{};
  std::optional<std::uint32_t> gpu_device_index_{};
  std::string gpu_name_;
};

} // namespace engine
