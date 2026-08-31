#pragma once

#include "renderer/vertex.hpp"

#include <cstdint>
#include <vector>

namespace engine {

struct MeshSource {
  std::vector<MeshVertex> vertices;
  // The packed streaming-terrain format. A mesh uses exactly one of the two
  // vertex vectors; filling this one makes it a terrain mesh, drawn by the
  // terrain pipelines (36-byte stride, no colour, no skinning).
  std::vector<TerrainVertex> terrain_vertices;
  std::vector<std::uint32_t> indices;

  [[nodiscard]] auto is_terrain() const -> bool { return !terrain_vertices.empty(); }
  [[nodiscard]] auto vertex_count() const -> std::size_t {
    return is_terrain() ? terrain_vertices.size() : vertices.size();
  }
};

} // namespace engine
