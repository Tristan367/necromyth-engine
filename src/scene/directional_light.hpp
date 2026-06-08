#pragma once

#include <glm/vec3.hpp>

namespace engine {

struct DirectionalLight {
  glm::vec3 direction_toward_light{0.35F, 0.85F, 0.38F};
  glm::vec3 color{1.0F, 0.98F, 0.92F};
  float intensity{1.0F};
  float ambient{0.18F};
};

} // namespace engine
