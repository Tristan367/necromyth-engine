#pragma once

#include <cstdint>

namespace engine {

enum class RenderLayer : std::uint8_t {
  Background = 0,
  Opaque = 1,
  AlphaTested = 2,
  Transparent = AlphaTested, // Backward-compatible alias; no separate blended pass yet.
  Overlay = 3,
};

} // namespace engine
