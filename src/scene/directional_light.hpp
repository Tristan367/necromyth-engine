#pragma once

#include <glm/vec3.hpp>

namespace engine {

struct DirectionalLight {
  // Unit vector from a surface point toward the sun (sky direction).
  glm::vec3 direction_toward_light{0.4F, 1.0F, 0.3F};
  glm::vec3 color{1.0F, 0.98F, 0.92F};
  float intensity{1.0F};
  float ambient{0.18F};
};

} // namespace engine
