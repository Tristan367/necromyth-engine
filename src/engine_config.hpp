#pragma once

#include "renderer/render_settings.hpp"
#include "scene/shadow_utils.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace engine {

struct EngineConfig {
  std::string window_title{"Necromyth Engine"};
  int window_width{1280};
  int window_height{720};
  MsaaSettings msaa{};
  std::uint32_t render_scale{1};
  std::uint32_t shadow_scale{1};
  PresentModePreference present_mode{PresentModePreference::Fifo};
  std::optional<std::uint32_t> gpu_device_index{};
};

[[nodiscard]] inline auto engine_config_from_environment() -> EngineConfig {
  EngineConfig config{};
  config.msaa = msaa_settings_from_environment();
  config.render_scale = render_scale_settings_from_environment();
  config.shadow_scale = shadow_scale_settings_from_environment();
  config.present_mode = present_mode_preference_from_environment();
  return config;
}

} // namespace engine
