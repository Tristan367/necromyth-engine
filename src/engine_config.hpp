#pragma once

#include "renderer/pipeline_id.hpp"
#include "renderer/render_settings.hpp"
#include "scene/mesh_instance.hpp"
#include "scene/shadow_utils.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
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
  // Per-draw instance records per frame, across the main and shadow lists. One
  // 80-byte record per draw; 65536 is 5 MB per frame in flight.
  std::uint32_t max_draw_instances{65536};
  // Staging ring size per frame in flight. Geometry uploaded in one frame has to
  // fit; anything that does not is deferred to the next frame rather than
  // stalling. 32 MB comfortably covers a streaming voxel world.
  std::uint64_t staging_bytes_per_frame{32ULL * 1024 * 1024};
  // How long one frame may spend uploading meshes before the rest waits for the
  // next frame. This is the C#'s number and its mechanism: VoxelClient.cs:3225
  // and :6314 run a Stopwatch and stop when it reads over 2ms.
  //
  // Mesh sync uploads every changed mesh it can, so a batch of voxel sections
  // finishing together landed as one frame's work -- measured at 32.7 MB across
  // 88 meshes in a single frame, which is a stagger arriving precisely when you
  // walk into new terrain. Spreading it costs those sections a few frames of
  // lateness and costs the frame rate nothing. 0 means no budget.
  float mesh_upload_budget_ms{2.0F};
  // Alpha modes the application will use later, on top of whatever the scene
  // already contains when the device is created.
  //
  // Pipelines are compiled once, from what the scene shows at startup, so that
  // a build only carries the ones it needs. That works for everything present
  // up front and not at all for geometry that streams in: the first chunk
  // containing a window arrives long after the pipelines are fixed. Declaring
  // the mode here is how a streaming application says which ones to expect.
  AlphaModeSet streaming_alpha_modes{};

  void declare_alpha_mode(MeshAlphaMode mode) {
    streaming_alpha_modes[static_cast<std::size_t>(mode)] = true;
  }

  // Which alpha modes STREAMING TERRAIN (the packed TerrainVertex format) will
  // use. Separate from the set above because terrain draws with its own
  // pipeline family -- declaring a mode here builds the terrain variant, and
  // an application that never streams terrain builds none of them.
  AlphaModeSet terrain_alpha_modes{};

  void declare_terrain_alpha_mode(MeshAlphaMode mode) {
    terrain_alpha_modes[static_cast<std::size_t>(mode)] = true;
  }

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
  // ENGINE_UPLOAD_BUDGET_MS=<milliseconds>, 0 to remove the budget. Exists so
  // the budget can be measured against no budget in the same session rather
  // than argued about.
  if (const char *env = std::getenv("ENGINE_UPLOAD_BUDGET_MS"); env != nullptr && env[0] != '\0')
    config.mesh_upload_budget_ms = std::strtof(env, nullptr);
  // ENGINE_WINDOW=1920x1080. Chiefly so a benchmark can put a real fragment
  // load on the GPU: a profile taken in a 960x540 window is measuring the
  // vertex path and calling it a frame time.
  if (const char *env = std::getenv("ENGINE_WINDOW"); env != nullptr && env[0] != '\0') {
    int w = 0;
    int h = 0;
    if (std::sscanf(env, "%dx%d", &w, &h) == 2 && w > 0 && h > 0) {
      config.window_width = w;
      config.window_height = h;
    }
  }
  return config;
}

} // namespace engine
