#pragma once

#include <glm/vec3.hpp>

namespace engine {

struct DirectionalLight {
  // Unit vector from a surface point toward the sun (same vector for shading: N·L, sky disc, shadows).
  // Some engines expose the opposite (direction light rays travel, e.g. Godot -Z, Sascha -lightPos); negate to convert.
  glm::vec3 direction_toward_light{0.371F, 0.928F, 0.278F};  // unit vector toward sun
  glm::vec3 color{1.0F, 0.98F, 0.92F};
  float intensity{1.0F};
  float ambient{0.18F};
  // How high the real sun is, -1 to 1, regardless of where the shading light
  // points. At night `direction_toward_light` is swung to a moon vector well
  // above the horizon so that shadows still have a direction -- which left the
  // sky shader believing the sun was up and painting a blue midnight. This is
  // the sky's clock, and nothing else reads it.
  float sun_elevation{1.0F};
};

} // namespace engine
