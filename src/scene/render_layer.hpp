#pragma once

#include <cstdint>

namespace engine {

enum class RenderLayer : std::uint8_t {
  Background = 0,
  Opaque = 1,
  Transparent = 2,
  Overlay = 3,
};

} // namespace engine
