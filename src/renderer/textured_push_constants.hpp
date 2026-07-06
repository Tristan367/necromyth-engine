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
  // Padding to maintain 16-byte alignment of the entire struct
  float _pad{0.0F};
};

} // namespace engine
