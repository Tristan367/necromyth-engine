#pragma once

#include "scene/camera.hpp"
#include "scene/directional_light.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <optional>
#include <tuple>

namespace engine {

inline constexpr float k_shadow_depth_bias_constant = 1.0F;
inline constexpr float k_shadow_depth_bias_slope = 2.5F;
// Practical split blend (GPU Gems / Sascha): 0 = uniform, 1 = logarithmic.
inline constexpr float k_shadow_split_lambda = 0.75F;

// Caster bias (depth pass polygon offset). Receiver bias lives in shaders/lib/shadow.slang.

enum class ShadowFilterMode : std::uint8_t {
  Hard = 0,
  Pcf3x3 = 1,
};

enum class ShadowCascadeMode : std::uint8_t {
  Single = 1,
  Dual = 2,
};

inline constexpr std::uint32_t k_max_shadow_cascades = 2;
inline constexpr std::uint32_t k_default_shadow_map_resolution = 2048U;
inline constexpr std::uint32_t k_min_shadow_map_resolution = 256U;
inline constexpr std::uint32_t k_max_shadow_scale = 64U;

// Integer divisor on map_resolution: 1 = full size, 2 = half edge length (quarter texels), etc.
[[nodiscard]] inline auto scaled_shadow_map_resolution(
    std::uint32_t base_resolution,
    std::uint32_t shadow_scale) -> std::uint32_t {
  if (shadow_scale <= 1U)
    return base_resolution;

  return std::max(k_min_shadow_map_resolution, base_resolution / shadow_scale);
}

[[nodiscard]] inline auto shadow_scale_active(std::uint32_t shadow_scale) -> bool {
  return shadow_scale > 1U;
}

[[nodiscard]] inline auto shadow_scale_settings_from_environment() -> std::uint32_t {
  const char *env = std::getenv("ENGINE_SHADOW_SCALE");
  if (env == nullptr || env[0] == '\0')
    return 1U;

  const int requested = std::atoi(env);
  if (requested <= 1)
    return 1U;

  return std::min(static_cast<std::uint32_t>(requested), k_max_shadow_scale);
}

struct DirectionalLightShadowSettings {
  // Dual mode only: view-depth range used to place the near/far cascade split.
  float max_distance{100.0F};
  std::uint32_t map_resolution{k_default_shadow_map_resolution};
  // Single-cascade footprint half-extent (meters); also scales the dual near cascade.
  float ortho_half_extent{64.0F};
  ShadowFilterMode filter_mode{ShadowFilterMode::Pcf3x3};
  // Shadow compare sampler: false = bilinear (default), true = nearest.
  bool point_shadow_filter{false};
  ShadowCascadeMode cascade_mode{ShadowCascadeMode::Dual};
  // Cross-fade width at the dual-cascade split (view-space meters).
  float cascade_blend_range{3.0F};
  // Far-cascade footprint half-extent (dual mode only).
  float dual_far_ortho_half_extent{127.0F};
  // UV-space edge fade toward fully lit; 0 = hard cutoff.
  float coverage_fade_uv_width{0.08F};
};

struct DirectionalShadowCascadeData {
  std::array<glm::mat4, k_max_shadow_cascades> light_view_proj{};
  // View-space Z threshold between cascades (negative, Sascha shadowmappingcascade convention).
  float split_view_z{0.0F};
};

// Startup-only fields (cascade mode, filter, compare sampler, map resolution, shadow scale) are frozen in
// VulkanContext; runtime Scene settings may change fade and blend width only.
[[nodiscard]] inline auto effective_shadow_settings(
    const DirectionalLightShadowSettings &scene_settings,
    ShadowCascadeMode startup_cascade_mode,
    std::uint32_t shadow_map_resolution) -> DirectionalLightShadowSettings {
  DirectionalLightShadowSettings settings = scene_settings;
  settings.cascade_mode = startup_cascade_mode;
  settings.map_resolution = shadow_map_resolution;
  return settings;
}

[[nodiscard]] inline auto shadow_cascade_layer_count(ShadowCascadeMode mode) -> std::uint32_t {
  return mode == ShadowCascadeMode::Dual ? 2U : 1U;
}

[[nodiscard]] inline auto shadow_cascade_mode_name(ShadowCascadeMode mode) -> const char * {
  switch (mode) {
  case ShadowCascadeMode::Single:
    return "single";
  case ShadowCascadeMode::Dual:
    return "dual";
  }
  return "unknown";
}

[[nodiscard]] inline auto shadow_filter_mode_name(ShadowFilterMode mode) -> const char * {
  switch (mode) {
  case ShadowFilterMode::Hard:
    return "hard";
  case ShadowFilterMode::Pcf3x3:
    return "pcf3x3";
  }
  return "unknown";
}

namespace detail {

[[nodiscard]] inline auto shadow_filter_token_equals(const char *token, const char *literal) -> bool {
  if (token == nullptr || literal == nullptr)
    return false;

  while (*token != '\0' && *literal != '\0') {
    if (std::tolower(static_cast<unsigned char>(*token)) != std::tolower(static_cast<unsigned char>(*literal)))
      return false;
    ++token;
    ++literal;
  }

  return *token == *literal;
}

[[nodiscard]] inline auto parse_shadow_filter_mode(const char *value) -> std::optional<ShadowFilterMode> {
  if (value == nullptr || value[0] == '\0')
    return std::nullopt;

  if (shadow_filter_token_equals(value, "hard") || shadow_filter_token_equals(value, "none") ||
      shadow_filter_token_equals(value, "0"))
    return ShadowFilterMode::Hard;
  if (shadow_filter_token_equals(value, "pcf") || shadow_filter_token_equals(value, "pcf3x3") ||
      shadow_filter_token_equals(value, "1"))
    return ShadowFilterMode::Pcf3x3;

  return std::nullopt;
}

} // namespace detail

[[nodiscard]] inline auto shadow_settings_from_environment(
    DirectionalLightShadowSettings defaults = {}) -> DirectionalLightShadowSettings {
  DirectionalLightShadowSettings settings = defaults;

  if (const char *env = std::getenv("ENGINE_SHADOW_DISTANCE"); env != nullptr && env[0] != '\0')
    settings.max_distance = std::max(1.0F, static_cast<float>(std::atof(env)));

  if (const char *env = std::getenv("ENGINE_SHADOW_FILTER"); env != nullptr && env[0] != '\0') {
    if (const std::optional<ShadowFilterMode> mode = detail::parse_shadow_filter_mode(env))
      settings.filter_mode = *mode;
  }

  if (const char *env = std::getenv("ENGINE_SHADOW_POINT_FILTER"); env != nullptr && env[0] != '\0')
    settings.point_shadow_filter = env[0] != '0';

  if (const char *env = std::getenv("ENGINE_SHADOW_FADE_WIDTH"); env != nullptr && env[0] != '\0')
    settings.coverage_fade_uv_width = std::max(0.0F, static_cast<float>(std::atof(env)));

  if (const char *env = std::getenv("ENGINE_SHADOW_CASCADES"); env != nullptr && env[0] != '\0') {
    if (std::atoi(env) >= 2)
      settings.cascade_mode = ShadowCascadeMode::Dual;
    else
      settings.cascade_mode = ShadowCascadeMode::Single;
  }

  return settings;
}

namespace detail {

[[nodiscard]] inline auto stable_up_for_light(glm::vec3 light_dir) -> glm::vec3 {
  constexpr glm::vec3 world_up{0.0F, 1.0F, 0.0F};
  if (std::abs(glm::dot(light_dir, world_up)) > 0.99F)
    return glm::vec3{0.0F, 0.0F, 1.0F};
  return world_up;
}

[[nodiscard]] inline auto light_axes(glm::vec3 light_dir, glm::vec3 up)
    -> std::tuple<glm::vec3, glm::vec3, glm::vec3> {
  const glm::vec3 z_axis = glm::normalize(light_dir);
  const glm::vec3 x_axis = glm::normalize(glm::cross(up, z_axis));
  const glm::vec3 y_axis = glm::cross(z_axis, x_axis);
  return {x_axis, y_axis, z_axis};
}

[[nodiscard]] inline auto snap_symmetric_bounds(
    float center_x,
    float center_y,
    float radius,
    float unit) -> std::array<float, 4> {
  float min_x = std::floor((center_x - radius) / unit) * unit;
  float max_x = std::floor((center_x + radius) / unit) * unit;
  float min_y = std::floor((center_y - radius) / unit) * unit;
  float max_y = std::floor((center_y + radius) / unit) * unit;

  const float min_extent = std::ceil((2.0F * radius) / unit) * unit;
  if (max_x - min_x < min_extent)
    max_x = min_x + min_extent;
  if (max_y - min_y < min_extent)
    max_y = min_y + min_extent;

  return {min_x, max_x, min_y, max_y};
}

[[nodiscard]] inline auto directional_light_view_projection_from_bounds(
    const DirectionalLight &light,
    const DirectionalLightShadowSettings &settings,
    glm::vec3 focus_center,
    float radius) -> glm::mat4 {
  const glm::vec3 light_dir = glm::normalize(light.direction_toward_light);
  const glm::vec3 up = stable_up_for_light(light_dir);
  const auto [x_axis, y_axis, z_axis] = light_axes(light_dir, up);

  const float center_x = glm::dot(focus_center, x_axis);
  const float center_y = glm::dot(focus_center, y_axis);
  const float center_z = glm::dot(focus_center, z_axis);

  const float map_size = static_cast<float>(std::max(settings.map_resolution, 1U));
  const float texel_world_size = (2.0F * radius) / map_size;

  float half_x = radius;
  float half_y = radius;
  const std::array<float, 4> snapped =
      snap_symmetric_bounds(center_x, center_y, radius, texel_world_size);
  half_x = (snapped[1] - snapped[0]) * 0.5F;
  half_y = (snapped[3] - snapped[2]) * 0.5F;
  const glm::vec3 snapped_center =
      x_axis * ((snapped[0] + snapped[1]) * 0.5F) + y_axis * ((snapped[2] + snapped[3]) * 0.5F) +
      z_axis * center_z;

  const glm::mat4 light_view = glm::lookAt(
      snapped_center + light_dir * radius,
      snapped_center,
      up);

  glm::mat4 light_ortho = glm::ortho(-half_x, half_x, -half_y, half_y, 0.0F, 2.0F * radius);
  light_ortho[1][1] *= -1.0F;

  return light_ortho * light_view;
}

} // namespace detail

// Camera footprint: ortho box centered on camera position.
// The ortho half-extent (64m near, 127m far) covers the surrounding area.
// Texel-snapping keeps the center stable when rotating.
[[nodiscard]] inline auto directional_light_footprint_projection(
    const Camera &camera,
    const DirectionalLight &light,
    const DirectionalLightShadowSettings &settings,
    float ortho_half_extent) -> glm::mat4 {
  const glm::vec3 focus_center = camera.position();
  const float radius = std::max(ortho_half_extent, 1.0F);
  return detail::directional_light_view_projection_from_bounds(light, settings, focus_center,
                                                               radius);
}

[[nodiscard]] inline auto directional_light_view_projection(
    const Camera &camera,
    const DirectionalLight &light,
    const DirectionalLightShadowSettings &settings = {}) -> glm::mat4 {
  return directional_light_footprint_projection(camera, light, settings, settings.ortho_half_extent);
}

// GPU Gems / Sascha practical split over [near, shadow_far].
[[nodiscard]] inline auto compute_cascade_split_normalized(
    const Camera &camera,
    const DirectionalLightShadowSettings &settings,
    float shadow_far,
    std::uint32_t cascade_index,
    std::uint32_t cascade_count) -> float {
  const float near_clip = camera.near_plane();
  const float clip_range = std::max(shadow_far - near_clip, 0.001F);
  const float min_z = std::max(near_clip, 0.001F);
  const float max_z = near_clip + clip_range;
  const float range = max_z - min_z;
  const float ratio = max_z / min_z;
  const float p = static_cast<float>(cascade_index + 1) / static_cast<float>(cascade_count);
  const float log_split = min_z * std::pow(ratio, p);
  const float uniform_split = min_z + range * p;
  const float lambda = std::clamp(k_shadow_split_lambda, 0.0F, 1.0F);
  const float d = lambda * (log_split - uniform_split) + uniform_split;
  return (d - near_clip) / clip_range;
}

[[nodiscard]] inline auto directional_shadow_cascades(
    const Camera &camera,
    const DirectionalLight &light,
    const DirectionalLightShadowSettings &settings = {}) -> DirectionalShadowCascadeData {
  DirectionalShadowCascadeData result{};

  if (settings.cascade_mode == ShadowCascadeMode::Single) {
    result.light_view_proj[0] = directional_light_view_projection(camera, light, settings);
    return result;
  }

  const float near_clip = camera.near_plane();
  const float shadow_far = std::min(settings.max_distance, camera.far_plane());
  const float clip_range = std::max(shadow_far - near_clip, 0.001F);
  const float split_norm =
      compute_cascade_split_normalized(camera, settings, shadow_far, 0, k_max_shadow_cascades);

  // Near cascade: smaller footprint → higher texel density close to the camera.
  const float near_half = std::max(
      settings.ortho_half_extent * std::clamp(split_norm * 2.0F, 0.25F, 0.55F),
      16.0F);
  const float far_half = std::max(settings.dual_far_ortho_half_extent, 1.0F);

  result.light_view_proj[0] =
      directional_light_footprint_projection(camera, light, settings, near_half);
  result.light_view_proj[1] =
      directional_light_footprint_projection(camera, light, settings, far_half);
  result.split_view_z = -(near_clip + split_norm * clip_range);

  return result;
}

// Vulkan cubemap face rotation matrices — maps viewIndex 0-5 to +X, -X, +Y, -Y, +Z, -Z.
[[nodiscard]] inline auto cubemap_face_views() -> const std::array<glm::mat4, 6> & {
  static const std::array<glm::mat4, 6> views{{
      glm::rotate(glm::rotate(glm::mat4(1.0F), glm::radians( 90.0F), glm::vec3(0,1,0)), glm::radians(180.0F), glm::vec3(1,0,0)),
      glm::rotate(glm::rotate(glm::mat4(1.0F), glm::radians(-90.0F), glm::vec3(0,1,0)), glm::radians(180.0F), glm::vec3(1,0,0)),
      glm::rotate(glm::mat4(1.0F), glm::radians(-90.0F), glm::vec3(1,0,0)),
      glm::rotate(glm::mat4(1.0F), glm::radians( 90.0F), glm::vec3(1,0,0)),
      glm::rotate(glm::mat4(1.0F), glm::radians(180.0F), glm::vec3(1,0,0)),
      glm::rotate(glm::mat4(1.0F), glm::radians(180.0F), glm::vec3(0,0,1)),
  }};
  return views;
}

} // namespace engine
