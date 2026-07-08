#pragma once

#include <glm/vec4.hpp>

namespace engine {

struct alignas(16) GpuParticle {
  glm::vec4 pos;  // xyz = world position, w = unused
};

} // namespace engine
