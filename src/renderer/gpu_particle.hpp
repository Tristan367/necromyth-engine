#pragma once

#include <glm/vec4.hpp>

namespace engine {

// One particle, as the vertex shader sees it.
//
// Colour and size are per particle rather than per draw. They started as push
// constants shared by every particle in the frame, which is enough for a dust
// puff and not enough for anything else: a flame is orange at the base and dark
// at the tip and smaller as it dies, and a system that can only draw one colour
// at one size can express none of that. Sixteen more bytes a particle buys
// fire, smoke, sparks, rain and blood from the same draw call.
struct alignas(16) GpuParticle {
  glm::vec4 pos_size;   // xyz = world position, w = radius in metres
  glm::vec4 color;      // rgb, a = opacity
};

} // namespace engine
