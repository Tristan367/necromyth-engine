#pragma once

#include "scene/directional_light.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace engine {

struct DirectionalLightShadowSettings {
  float ortho_extent{14.0F};
  float light_distance{35.0F};
  float z_near{0.5F};
  float z_far{80.0F};
  float depth_bias_constant{1.25F};
  float depth_bias_slope{1.75F};
};

[[nodiscard]] inline auto directional_light_view_projection(
    const DirectionalLight &light,
    const DirectionalLightShadowSettings &settings,
    glm::vec3 focus_point = glm::vec3{0.0F}) -> glm::mat4 {
  const glm::vec3 toward_light = glm::normalize(light.direction_toward_light);
  const glm::vec3 light_position = focus_point + toward_light * settings.light_distance;
  const glm::mat4 view = glm::lookAt(light_position, focus_point, glm::vec3{0.0F, 1.0F, 0.0F});
  glm::mat4 projection = glm::ortho(
      -settings.ortho_extent,
      settings.ortho_extent,
      -settings.ortho_extent,
      settings.ortho_extent,
      settings.z_near,
      settings.z_far);
  projection[1][1] *= -1.0F;
  return projection * view;
}

} // namespace engine
