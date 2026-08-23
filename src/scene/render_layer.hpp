#pragma once

#include <cstdint>

namespace engine {

enum class RenderLayer : std::uint8_t {
  Background = 0,
  Opaque = 1,
  AlphaTested = 2,
  // Drawn last. The draw list sorts by layer before anything else, so putting
  // a blended instance here is the whole of "after the solid world" -- there
  // is no separate pass to schedule and no second traversal of the scene.
  Transparent = 3,
};

} // namespace engine
