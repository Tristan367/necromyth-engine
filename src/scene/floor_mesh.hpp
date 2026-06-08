#pragma once

#include "renderer/vertex.hpp"
#include "scene/mesh_source.hpp"

#include <cstddef>

namespace engine {

[[nodiscard]] inline auto make_floor_quad_mesh(float half_extent = 20.0F, float uv_tiles = 8.0F) -> MeshSource {
  static constexpr float white[] = {1.0F, 1.0F, 1.0F};
  static constexpr float up[] = {0.0F, 1.0F, 0.0F};

  const MeshVertex vertices[] = {
      {{-half_extent, 0.0F, -half_extent}, {up[0], up[1], up[2]}, {white[0], white[1], white[2]}, {0.0F, 0.0F}},
      {{half_extent, 0.0F, -half_extent}, {up[0], up[1], up[2]}, {white[0], white[1], white[2]}, {uv_tiles, 0.0F}},
      {{half_extent, 0.0F, half_extent}, {up[0], up[1], up[2]}, {white[0], white[1], white[2]}, {uv_tiles, uv_tiles}},
      {{-half_extent, 0.0F, half_extent}, {up[0], up[1], up[2]}, {white[0], white[1], white[2]}, {0.0F, uv_tiles}},
  };

  static constexpr std::uint32_t indices[] = {0, 2, 1, 0, 3, 2};

  return {
      .vertices = {vertices, vertices + std::size(vertices)},
      .indices = {indices, indices + std::size(indices)},
  };
}

} // namespace engine
