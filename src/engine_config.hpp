#pragma once

#include "renderer/render_settings.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace engine {

struct EngineConfig {
  std::string window_title{"Vulkan C++ Engine"};
  int window_width{1280};
  int window_height{720};
  MsaaSettings msaa{};
  std::optional<std::uint32_t> gpu_device_index{};
};

[[nodiscard]] inline auto engine_config_from_environment() -> EngineConfig {
  EngineConfig config{};
  config.msaa = msaa_settings_from_environment();
  return config;
}

} // namespace engine
