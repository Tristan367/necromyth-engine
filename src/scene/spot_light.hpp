#pragma once

#include <glm/vec3.hpp>

namespace engine {

struct SpotLight {
  glm::vec3 position{0.0F};
  glm::vec3 direction{0.0F, -1.0F, 0.0F};
  glm::vec3 color{1.0F, 1.0F, 1.0F};
  float intensity{1.0F};
  float range{8.0F};
  float inner_angle{0.78F}; // radians (~45°)
  float outer_angle{0.90F}; // radians (~52°)
};

} // namespace engine
