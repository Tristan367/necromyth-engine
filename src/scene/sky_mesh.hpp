#pragma once

#include "renderer/vertex.hpp"
#include "scene/mesh_source.hpp"

#include <cstddef>

namespace engine {

[[nodiscard]] inline auto make_sky_cube_mesh() -> MeshSource {
  static constexpr MeshVertex vertices[] = {
      {{-1.0F, -1.0F, -1.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F}},
      {{1.0F, -1.0F, -1.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F}},
      {{1.0F, 1.0F, -1.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F}},
      {{-1.0F, 1.0F, -1.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F}},
      {{-1.0F, -1.0F, 1.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F}},
      {{1.0F, -1.0F, 1.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F}},
      {{1.0F, 1.0F, 1.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F}},
      {{-1.0F, 1.0F, 1.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F}},
  };

  static constexpr std::uint32_t indices[] = {
      0, 1, 2, 2, 3, 0,
      1, 5, 6, 6, 2, 1,
      5, 4, 7, 7, 6, 5,
      4, 0, 3, 3, 7, 4,
      3, 2, 6, 6, 7, 3,
      4, 5, 1, 1, 0, 4,
  };

  return {
      .vertices = {vertices, vertices + std::size(vertices)},
      .indices = {indices, indices + std::size(indices)},
  };
}

} // namespace engine
