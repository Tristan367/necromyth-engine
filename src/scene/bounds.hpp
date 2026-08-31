#pragma once

#include "engine_glm.hpp"
#include "scene/mesh_source.hpp"

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include <limits>

namespace engine {

// Object-space axis-aligned bounds of a mesh.
//
// Lives on the scene side rather than in MeshGpu so that culling, physics and
// spatial queries can use it without a GPU upload having happened, and so the
// draw list stays free of renderer types.
struct AABB {
  glm::vec3 min{std::numeric_limits<float>::max()};
  glm::vec3 max{-std::numeric_limits<float>::max()};

  void extend(const glm::vec3 &p) {
    min = glm::min(min, p);
    max = glm::max(max, p);
  }

  // An AABB that never had a point added stays inverted. Callers use this to
  // mean "no usable bounds" and skip culling rather than reject the mesh.
  [[nodiscard]] auto empty() const -> bool { return min.x > max.x; }

  [[nodiscard]] auto center() const -> glm::vec3 {
    return empty() ? glm::vec3(0.0F) : (min + max) * 0.5F;
  }

  [[nodiscard]] auto radius() const -> float {
    return empty() ? 0.0F : glm::distance(min, max) * 0.5F;
  }
};

[[nodiscard]] inline auto compute_bounds(const MeshSource &mesh) -> AABB {
  AABB bounds{};
  for (const MeshVertex &v : mesh.vertices)
    bounds.extend(glm::vec3(v.pos[0], v.pos[1], v.pos[2]));
  for (const TerrainVertex &v : mesh.terrain_vertices)
    bounds.extend(glm::vec3(v.pos[0], v.pos[1], v.pos[2]));
  return bounds;
}

} // namespace engine
