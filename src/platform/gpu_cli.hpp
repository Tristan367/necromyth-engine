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
#include <unistd.h>
#include <vector>

namespace engine {

struct CliParseResult {
  EngineConfig config{};
  bool exit_after_list_gpus{false};
  bool interactive_gpu_pick{false};
};

struct GpuDeviceInfo {
  std::uint32_t index{};
  std::string name;
  std::string type_label;
  std::string api_version;
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

struct VulkanListContext {
  vk::raii::Context context;
  vk::raii::Instance instance{nullptr};
  bool sdl_initialized{false};

  VulkanListContext() {
    if (!SDL_Init(SDL_INIT_VIDEO))
      throw std::runtime_error(std::string("Failed to initialize SDL: ") + SDL_GetError());
    sdl_initialized = true;

    if (!SDL_Vulkan_LoadLibrary(nullptr))
      throw std::runtime_error(std::string("Failed to load Vulkan through SDL: ") + SDL_GetError());

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
        .pApplicationName = "Necromyth Engine",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "Necromyth Engine",
        .engineVersion = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = vk::ApiVersion13,
    };

    const vk::InstanceCreateInfo create_info{
        .flags = instance_flags,
        .pApplicationInfo = &application_info,
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    instance = vk::raii::Instance(context, create_info);
  }

  ~VulkanListContext() {
    instance = nullptr;
    if (sdl_initialized)
      SDL_Quit();
  }

  VulkanListContext(const VulkanListContext &) = delete;
  auto operator=(const VulkanListContext &) -> VulkanListContext & = delete;
};

[[nodiscard]] inline auto make_device_info(std::uint32_t index, const vk::PhysicalDeviceProperties &properties)
    -> GpuDeviceInfo {
  return {
      .index = index,
      .name = properties.deviceName.data(),
      .type_label = device_type_label(properties.deviceType),
      .api_version = std::to_string(VK_VERSION_MAJOR(properties.apiVersion)) + '.' +
                     std::to_string(VK_VERSION_MINOR(properties.apiVersion)),
  };
}

} // namespace detail

[[nodiscard]] inline auto enumerate_physical_devices() -> std::vector<GpuDeviceInfo> {
  detail::VulkanListContext bootstrap;
  const auto devices = bootstrap.instance.enumeratePhysicalDevices();

  std::vector<GpuDeviceInfo> result;
  result.reserve(devices.size());
  for (std::uint32_t index = 0; index < devices.size(); ++index)
    result.push_back(detail::make_device_info(index, devices[index].getProperties()));

  return result;
}

inline void print_gpu_device_line(const GpuDeviceInfo &device) {
  std::cout << "  " << device.name << " [" << device.index << "] (" << device.type_label << ", Vulkan "
            << device.api_version << ")\n";
}

inline void list_physical_devices() {
  const auto devices = enumerate_physical_devices();
  std::cout << "Available GPUs:\n";
  for (const GpuDeviceInfo &device : devices)
    print_gpu_device_line(device);
  std::cout << "\nUse -g <index> or --gpu <index> to select a device.\n";
  std::cout << "Use --pick-gpu for an interactive prompt when multiple GPUs are present.\n";
}

[[nodiscard]] inline auto prompt_gpu_selection(const std::vector<GpuDeviceInfo> &devices) -> std::uint32_t {
  if (devices.empty())
    throw std::runtime_error("No Vulkan-capable GPU found");

  std::cout << "Select GPU:\n";
  for (const GpuDeviceInfo &device : devices)
    print_gpu_device_line(device);

  while (true) {
    std::cout << "\nPick GPU [0]: " << std::flush;

    std::string line;
    if (!std::getline(std::cin, line))
      throw std::runtime_error("Failed to read GPU selection");

    if (line.empty())
      return 0;

    char *end = nullptr;
    const unsigned long value = std::strtoul(line.c_str(), &end, 10);
    if (end != line.c_str() && *end == '\0' && value < devices.size())
      return static_cast<std::uint32_t>(value);

    std::cout << "Invalid choice. Enter a number from 0 to " << devices.size() - 1 << ", or press Enter for [0].\n";
  }
}

inline void resolve_gpu_selection(EngineConfig &config, bool force_interactive_pick) {
  if (config.gpu_device_index)
    return;

  const auto devices = enumerate_physical_devices();
  if (devices.size() <= 1 && !force_interactive_pick)
    return;

  const bool stdin_is_tty = isatty(STDIN_FILENO) != 0;
  if (!force_interactive_pick && !stdin_is_tty)
    return;

  if (devices.size() <= 1) {
    config.gpu_device_index = 0;
    return;
  }

  config.gpu_device_index = prompt_gpu_selection(devices);
  std::cout << "Using GPU [" << *config.gpu_device_index << "]: "
            << devices[*config.gpu_device_index].name << '\n';
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

    if (arg == "--pick-gpu" || arg == "-pg") {
      result.interactive_gpu_pick = true;
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
      std::cout << "Usage: NecromythEngine [options]\n"
                << "  -gl, --listgpus       List physical devices and exit\n"
                << "  -g, --gpu <index>     Select physical device by index\n"
                << "  -pg, --pick-gpu       Prompt to pick a GPU (also auto when multiple GPUs + TTY)\n"
                << "  -h, --help            Show this help\n"
                << "Environment: ENGINE_MSAA=0|2|4|8|off\n";
      std::exit(EXIT_SUCCESS);
    }

    throw std::runtime_error(std::string("Unknown argument: ") + std::string(arg));
  }

  return result;
}

} // namespace engine
