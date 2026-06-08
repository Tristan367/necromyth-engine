#pragma once

#include "engine_config.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace engine {

struct CliParseResult {
  EngineConfig config{};
  bool exit_after_list_gpus{false};
};

namespace detail {

[[nodiscard]] inline auto device_type_label(vk::PhysicalDeviceType type) -> const char * {
  switch (type) {
  case vk::PhysicalDeviceType::eDiscreteGpu:
    return "discrete";
  case vk::PhysicalDeviceType::eIntegratedGpu:
    return "integrated";
  case vk::PhysicalDeviceType::eVirtualGpu:
    return "virtual";
  case vk::PhysicalDeviceType::eCpu:
    return "cpu";
  default:
    return "other";
  }
}

} // namespace detail

inline void list_physical_devices() {
  if (!SDL_Init(SDL_INIT_VIDEO))
    throw std::runtime_error(std::string("Failed to initialize SDL: ") + SDL_GetError());

  if (!SDL_Vulkan_LoadLibrary(nullptr)) {
    SDL_Quit();
    throw std::runtime_error(std::string("Failed to load Vulkan through SDL: ") + SDL_GetError());
  }

  vk::raii::Context context;
  std::uint32_t extension_count{};
  const char *const *sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&extension_count);
  if (sdl_extensions == nullptr || extension_count == 0)
    throw std::runtime_error(std::string("Failed to get SDL Vulkan extensions: ") + SDL_GetError());

  std::vector<const char *> extensions(sdl_extensions, sdl_extensions + extension_count);
  const auto available_instance_extensions = context.enumerateInstanceExtensionProperties();

  vk::InstanceCreateFlags instance_flags{};
  if (std::ranges::any_of(available_instance_extensions, [](const vk::ExtensionProperties &property) {
        return std::string_view(property.extensionName) == vk::KHRPortabilityEnumerationExtensionName;
      })) {
    extensions.push_back(vk::KHRPortabilityEnumerationExtensionName);
    instance_flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
  }

  const vk::ApplicationInfo application_info{
      .pApplicationName = "Vulkan C++ Engine",
      .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
      .pEngineName = "Vulkan C++ Engine",
      .engineVersion = VK_MAKE_VERSION(0, 1, 0),
      .apiVersion = vk::ApiVersion13,
  };

  const vk::InstanceCreateInfo create_info{
      .flags = instance_flags,
      .pApplicationInfo = &application_info,
      .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
      .ppEnabledExtensionNames = extensions.data(),
  };

  const vk::raii::Instance instance(context, create_info);
  const auto devices = instance.enumeratePhysicalDevices();

  std::cout << "Available Vulkan physical devices:\n";
  for (std::uint32_t index = 0; index < devices.size(); ++index) {
    const vk::PhysicalDeviceProperties properties = devices[index].getProperties();
    const auto api_major = VK_VERSION_MAJOR(properties.apiVersion);
    const auto api_minor = VK_VERSION_MINOR(properties.apiVersion);
    std::cout << "  [" << index << "] " << properties.deviceName << " ("
              << detail::device_type_label(properties.deviceType) << ", Vulkan "
              << api_major << '.' << api_minor << ")\n";
  }
  std::cout << "Use -g <index> or --gpu <index> to select a device.\n";
  SDL_Quit();
}

[[nodiscard]] inline auto parse_engine_cli(int argc, char **argv) -> CliParseResult {
  CliParseResult result{};
  result.config = engine_config_from_environment();

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};

    if (arg == "--listgpus" || arg == "-gl") {
      result.exit_after_list_gpus = true;
      continue;
    }

    if ((arg == "-g" || arg == "--gpu") && i + 1 < argc) {
      const int value = std::atoi(argv[++i]);
      if (value < 0)
        throw std::runtime_error("GPU index must be non-negative");
      result.config.gpu_device_index = static_cast<std::uint32_t>(value);
      continue;
    }

    if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: VulkanCppEngine [options]\n"
                << "  -gl, --listgpus       List physical devices and exit\n"
                << "  -g, --gpu <index>     Select physical device by index\n"
                << "  -h, --help            Show this help\n"
                << "Environment: ENGINE_MSAA=0|2|4|8|off\n";
      std::exit(EXIT_SUCCESS);
    }

    throw std::runtime_error(std::string("Unknown argument: ") + std::string(arg));
  }

  return result;
}

} // namespace engine
