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
#include <cmath>
#include <cstdlib>
#include <limits>

namespace engine {

inline constexpr float k_shadow_depth_bias_constant = 1.0F;
inline constexpr float k_shadow_depth_bias_slope = 2.5F;

struct DirectionalLightShadowSettings {
  float max_distance{50.0F};
  float min_ortho_extent{12.0F};
  float bounds_padding{1.25F};
};

[[nodiscard]] inline auto shadow_settings_from_environment(
    DirectionalLightShadowSettings defaults = {}) -> DirectionalLightShadowSettings {
  DirectionalLightShadowSettings settings = defaults;

  if (const char *env = std::getenv("ENGINE_SHADOW_DISTANCE"); env != nullptr && env[0] != '\0')
    settings.max_distance = std::max(1.0F, static_cast<float>(std::atof(env)));

  return settings;
}

namespace detail {

[[nodiscard]] inline auto shadow_ground_focus(const Camera &camera) -> glm::vec3 {
  const glm::vec3 position = camera.position();
  const glm::vec3 forward = camera.look_direction();

  if (forward.y < -1e-4F) {
    const float t = -position.y / forward.y;
    if (t > 0.0F)
      return position + forward * t;
  }

  return glm::vec3(position.x, 0.0F, position.z);
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

  for (std::uint32_t j = 0; j < 4; ++j) {
    const glm::vec3 edge = corners[j + 4] - corners[j];
    corners[j + 4] = corners[j] + edge * split_dist;
  }

  return corners;
}

} // namespace detail

// View-frustum-fitted orthographic directional shadow (single split from Sascha shadowmappingcascade).
[[nodiscard]] inline auto directional_light_view_projection(
    const Camera &camera,
    const DirectionalLight &light,
    const DirectionalLightShadowSettings &settings = {}) -> glm::mat4 {
  const float near_clip = camera.near_plane();
  const float far_clip = camera.far_plane();
  const float clip_range = far_clip - near_clip;
  const float shadow_far = std::min(settings.max_distance, far_clip);
  const float split_dist = std::clamp((shadow_far - near_clip) / clip_range, 0.01F, 1.0F);

  const std::array<glm::vec3, 8> frustum_corners = detail::unproject_view_frustum_wedge(camera, split_dist);

  const std::array<glm::vec3, 11> bound_points = {{
      frustum_corners[0],
      frustum_corners[1],
      frustum_corners[2],
      frustum_corners[3],
      frustum_corners[4],
      frustum_corners[5],
      frustum_corners[6],
      frustum_corners[7],
      camera.position(),
      camera.target(),
      detail::shadow_ground_focus(camera),
  }};

  glm::vec3 bounds_center{0.0F};
  for (const glm::vec3 &point : bound_points)
    bounds_center += point;
  bounds_center /= static_cast<float>(bound_points.size());

  float radius = 0.0F;
  for (const glm::vec3 &point : bound_points)
    radius = std::max(radius, glm::length(point - bounds_center));
  radius = std::max(std::ceil(radius * 16.0F) / 16.0F, settings.min_ortho_extent);

  const glm::vec3 light_dir = glm::normalize(light.direction_toward_light);
  const glm::mat4 light_view = glm::lookAt(
      bounds_center + light_dir * radius,
      bounds_center,
      glm::vec3{0.0F, 1.0F, 0.0F});

  glm::vec3 min_ls{std::numeric_limits<float>::max()};
  glm::vec3 max_ls{std::numeric_limits<float>::lowest()};
  for (const glm::vec3 &point : bound_points) {
    const glm::vec3 ls = glm::vec3(light_view * glm::vec4(point, 1.0F));
    min_ls = glm::min(min_ls, ls);
    max_ls = glm::max(max_ls, ls);
  }

  const glm::vec3 center_ls = (min_ls + max_ls) * 0.5F;
  glm::vec3 half_extent = (max_ls - min_ls) * 0.5F;
  half_extent.x = std::max(half_extent.x * settings.bounds_padding, settings.min_ortho_extent);
  half_extent.y = std::max(half_extent.y * settings.bounds_padding, settings.min_ortho_extent);
  half_extent.z = std::max(half_extent.z * settings.bounds_padding, settings.min_ortho_extent * 0.5F);

  min_ls = center_ls - half_extent;
  max_ls = center_ls + half_extent;
  min_ls.z -= radius * 0.25F;

  glm::mat4 light_ortho = glm::ortho(
      min_ls.x,
      max_ls.x,
      min_ls.y,
      max_ls.y,
      -max_ls.z,
      -min_ls.z);
  light_ortho[1][1] *= -1.0F;

  return light_ortho * light_view;
}

} // namespace engine
