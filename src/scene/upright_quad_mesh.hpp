#pragma once

#include "renderer/vertex.hpp"
#include "scene/mesh_source.hpp"

#include <cstddef>

namespace engine {

// Vertical quad in the XY plane (normal +Z), bottom edge on Y = 0.
[[nodiscard]] inline auto make_upright_quad_mesh(float width = 1.0F, float height = 1.0F) -> MeshSource {
  static constexpr float white[] = {1.0F, 1.0F, 1.0F};
  static constexpr float normal[] = {0.0F, 0.0F, 1.0F};

  const float half_width = width * 0.5F;

  const MeshVertex vertices[] = {
      {{-half_width, 0.0F, 0.0F}, {normal[0], normal[1], normal[2]}, {white[0], white[1], white[2]}, {0.0F, 1.0F}},
      {{half_width, 0.0F, 0.0F}, {normal[0], normal[1], normal[2]}, {white[0], white[1], white[2]}, {1.0F, 1.0F}},
      {{half_width, height, 0.0F}, {normal[0], normal[1], normal[2]}, {white[0], white[1], white[2]}, {1.0F, 0.0F}},
      {{-half_width, height, 0.0F}, {normal[0], normal[1], normal[2]}, {white[0], white[1], white[2]}, {0.0F, 0.0F}},
  };

  static constexpr std::uint32_t indices[] = {0, 1, 2, 0, 2, 3};

  return {
      .vertices = {vertices, vertices + std::size(vertices)},
      .indices = {indices, indices + std::size(indices)},
  };
}

} // namespace engine
