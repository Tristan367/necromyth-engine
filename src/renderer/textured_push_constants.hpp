#pragma once

#include <glm/mat4x4.hpp>

#include <cstdint>

namespace engine {

enum class TextureSource : std::uint8_t {
  Table = 0,
  ArrayLayer = 1,
};

struct TexturedPushConstants {
  alignas(16) glm::mat4 model{};
  std::uint32_t texture_array_layer{0};
  std::uint32_t sample_texture_array{0};
  std::uint32_t shadow_cascade_index{0};
  // Dual-paraboloid point shadow params (Godot omni model). Only meaningful
  // when shadow_cascade_index >= 6 (point light depth pass).
  //   dp_side  = +1.0 for the +Z paraboloid half, -1.0 for the -Z half.
  //   dp_z_far = point light range (used to normalize radial distance).
  float dp_side{1.0F};
  float dp_z_far{1.0F};
};

} // namespace engine
