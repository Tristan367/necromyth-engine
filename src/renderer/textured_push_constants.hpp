#pragma once

#include <cstdint>

namespace engine {

enum class TextureSource : std::uint8_t {
  Table = 0,
  ArrayLayer = 1,
};

// 16 bytes. This used to carry the model matrix and material selection -- 92
// bytes pushed per object -- which is exactly what made one draw call able to
// describe only one object. Those live in the instance buffer now.
struct TexturedPushConstants {
  std::uint32_t instance_base{0};
  std::uint32_t shadow_cascade_index{0};
  std::uint32_t point_light_index{0};
  std::uint32_t _pad{0};
};

} // namespace engine
