#pragma once

#include <glm/vec3.hpp>

namespace engine {

struct PointLight {
  glm::vec3 position{0.0F};
  glm::vec3 color{1.0F, 1.0F, 1.0F}; // linear (not sRGB)
  float intensity{1.0F};
  float range{5.0F};
  bool casts_shadow{false};
};

} // namespace engine
