#pragma once

#include "renderer/render_settings.hpp"
#include "scene/shadow_utils.hpp"

#include <cstdint>
#include <cstdlib>
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
  std::uint32_t max_point_shadow_lights{64};
  std::uint32_t max_spot_shadow_lights{16};
  std::uint32_t max_total_lights{256};
  std::uint32_t max_particles{65536};
  // Skinned instances that can be posed in one frame. Backs a single shared
  // bone buffer, so the cost is memory only (about 8 KB per slot per frame in
  // flight) -- not descriptor sets, and not anything that reallocates when
  // characters spawn.
  std::uint32_t max_skinned_instances{256};
  // Print a per-pass CPU/GPU timing breakdown every profiling window.
  // ENGINE_PROFILE=1. Timestamps are always collected and readable via
  // VulkanContext::profile_report(); this only controls the periodic dump.
  bool profile_to_stdout{false};
};

[[nodiscard]] inline auto engine_config_from_environment() -> EngineConfig {
  EngineConfig config{};
  config.msaa = msaa_settings_from_environment();
  config.render_scale = render_scale_settings_from_environment();
  config.shadow_scale = shadow_scale_settings_from_environment();
  config.present_mode = present_mode_preference_from_environment();
  if (const char *env = std::getenv("ENGINE_PROFILE"); env != nullptr && env[0] != '\0')
    config.profile_to_stdout = env[0] != '0';
  return config;
}

} // namespace engine
