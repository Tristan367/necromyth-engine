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

enum class ShadowFocusMode {
  // Shadow box follows camera XZ; look direction does not move the map (stable when rotating).
  CameraFootprint,
  // Fit to the view frustum wedge (best coverage when looking around; may shimmer on rotation).
  ViewWedge,
};

// Fragment shadow filter: Hard (single tap) or Pcf3x3.
enum class ShadowFilterMode : std::uint8_t {
  Hard = 0,
  Pcf3x3 = 1,
};

struct DirectionalLightShadowSettings {
  float max_distance{50.0F};
  std::uint32_t map_resolution{2048};
  ShadowFocusMode focus_mode{ShadowFocusMode::CameraFootprint};
  float footprint_focus_y{0.0F};
  float ortho_half_extent{56.0F};
  bool texel_snapping{true};
  ShadowFilterMode filter_mode{ShadowFilterMode::Pcf3x3};
  // false = bilinear compare fetch (default); true = nearest.
  bool point_shadow_filter{false};
};

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

  if (const char *env = std::getenv("ENGINE_SHADOW_TEXEL_SNAP"); env != nullptr && env[0] != '\0')
    settings.texel_snapping = env[0] != '0';

  if (const char *env = std::getenv("ENGINE_SHADOW_FILTER"); env != nullptr && env[0] != '\0') {
    if (const std::optional<ShadowFilterMode> mode = detail::parse_shadow_filter_mode(env))
      settings.filter_mode = *mode;
  } else if (const char *env = std::getenv("ENGINE_SHADOW_PCF"); env != nullptr && env[0] != '\0') {
    settings.filter_mode = env[0] != '0' ? ShadowFilterMode::Pcf3x3 : ShadowFilterMode::Hard;
  }

  if (const char *env = std::getenv("ENGINE_SHADOW_POINT_FILTER"); env != nullptr && env[0] != '\0')
    settings.point_shadow_filter = env[0] != '0';

  if (const char *env = std::getenv("ENGINE_SHADOW_FOCUS"); env != nullptr && env[0] != '\0') {
    if (detail::shadow_filter_token_equals(env, "wedge") || detail::shadow_filter_token_equals(env, "view") ||
        detail::shadow_filter_token_equals(env, "frustum"))
      settings.focus_mode = ShadowFocusMode::ViewWedge;
    else if (detail::shadow_filter_token_equals(env, "footprint") || detail::shadow_filter_token_equals(env, "camera"))
      settings.focus_mode = ShadowFocusMode::CameraFootprint;
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

[[nodiscard]] inline auto unproject_view_frustum_wedge(
    const Camera &camera,
    float split_dist) -> std::array<glm::vec3, 8> {
  std::array<glm::vec3, 8> corners = {{
      {-1.0F, 1.0F, 0.0F},
      {1.0F, 1.0F, 0.0F},
      {1.0F, -1.0F, 0.0F},
      {-1.0F, -1.0F, 0.0F},
      {-1.0F, 1.0F, 1.0F},
      {1.0F, 1.0F, 1.0F},
      {1.0F, -1.0F, 1.0F},
      {-1.0F, -1.0F, 1.0F},
  }};

  const glm::mat4 inv_cam = glm::inverse(camera.projection_matrix() * camera.view_matrix());
  for (glm::vec3 &corner : corners) {
    const glm::vec4 world = inv_cam * glm::vec4(corner, 1.0F);
    corner = world / world.w;
  }

  constexpr float last_split_dist = 0.0F;
  for (std::uint32_t j = 0; j < 4; ++j) {
    const glm::vec3 edge = corners[j + 4] - corners[j];
    corners[j + 4] = corners[j] + edge * split_dist;
    corners[j] = corners[j] + edge * last_split_dist;
  }

  return corners;
}

[[nodiscard]] inline auto wedge_frustum_center_and_radius(const std::array<glm::vec3, 8> &corners)
    -> std::pair<glm::vec3, float> {
  glm::vec3 center{0.0F};
  for (const glm::vec3 &corner : corners)
    center += corner;
  center /= static_cast<float>(corners.size());

  float radius = 0.0F;
  for (const glm::vec3 &corner : corners)
    radius = std::max(radius, glm::length(corner - center));
  radius = std::ceil(radius * 16.0F) / 16.0F;

  return {center, radius};
}

} // namespace detail

// Directional shadow matrix: view-frustum wedge (Sascha cascade 0) or rotation-stable camera footprint.
// Texel snapping on light XY (Godot renderer_scene_cull, VulkanDemos 37_CascadedShadow).
[[nodiscard]] inline auto directional_light_view_projection(
    const Camera &camera,
    const DirectionalLight &light,
    const DirectionalLightShadowSettings &settings = {}) -> glm::mat4 {
  const float near_clip = camera.near_plane();
  const float camera_far = camera.far_plane();
  const float clip_range = camera_far - near_clip;
  const float shadow_far = std::min(settings.max_distance, camera_far);
  const float split_dist = std::clamp((shadow_far - near_clip) / clip_range, 0.01F, 1.0F);

  glm::vec3 focus_center{};
  float radius = settings.ortho_half_extent;

  if (settings.focus_mode == ShadowFocusMode::CameraFootprint) {
    const glm::vec3 position = camera.position();
    focus_center = glm::vec3{position.x, settings.footprint_focus_y, position.z};
    radius = std::max(settings.ortho_half_extent, 1.0F);
  } else {
    const std::array<glm::vec3, 8> frustum_corners =
        detail::unproject_view_frustum_wedge(camera, split_dist);
    std::tie(focus_center, radius) = detail::wedge_frustum_center_and_radius(frustum_corners);
  }

  if (!std::isfinite(radius) || radius < 1.0F) {
    focus_center = camera.position() + camera.look_direction() * std::min(settings.max_distance * 0.5F, 25.0F);
    radius = std::max(settings.ortho_half_extent, 12.0F);
  }

  const glm::vec3 light_dir = glm::normalize(light.direction_toward_light);
  const glm::vec3 up = detail::stable_up_for_light(light_dir);
  const auto [x_axis, y_axis, z_axis] = detail::light_axes(light_dir, up);

  const float center_x = glm::dot(focus_center, x_axis);
  const float center_y = glm::dot(focus_center, y_axis);
  const float center_z = glm::dot(focus_center, z_axis);

  const float map_size = static_cast<float>(std::max(settings.map_resolution, 1U));
  const float texel_world_size = (2.0F * radius) / map_size;

  float half_x = radius;
  float half_y = radius;
  glm::vec3 snapped_center = focus_center;

  if (settings.texel_snapping) {
    const std::array<float, 4> snapped =
        detail::snap_symmetric_bounds(center_x, center_y, radius, texel_world_size);
    half_x = (snapped[1] - snapped[0]) * 0.5F;
    half_y = (snapped[3] - snapped[2]) * 0.5F;
    snapped_center =
        x_axis * ((snapped[0] + snapped[1]) * 0.5F) + y_axis * ((snapped[2] + snapped[3]) * 0.5F) +
        z_axis * center_z;
  } else {
    snapped_center =
        x_axis * center_x + y_axis * center_y + z_axis * center_z;
  }

  const glm::mat4 light_view = glm::lookAt(
      snapped_center + light_dir * radius,
      snapped_center,
      up);

  glm::mat4 light_ortho = glm::ortho(-half_x, half_x, -half_y, half_y, 0.0F, 2.0F * radius);
  light_ortho[1][1] *= -1.0F;

  return light_ortho * light_view;
}

} // namespace engine
