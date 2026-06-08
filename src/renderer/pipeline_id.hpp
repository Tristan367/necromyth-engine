#pragma once

#include <cstdint>

namespace engine {

enum class PipelineId : std::uint8_t {
  Background = 0,
  TexturedMesh = 1,
  ShadowDepth = 2,
};

[[nodiscard]] constexpr auto pipeline_count() -> std::uint32_t {
  return 3;
}

} // namespace engine
