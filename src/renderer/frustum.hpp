#pragma once

#include "engine_glm.hpp"
#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <array>

namespace engine {

// World-space bounding sphere of a mesh instance.
//
// Computed once per instance per frame in build_draw_list() and carried on the
// DrawCommand, because every pass that culls (main, each shadow cascade, each
// spot light, each point light) would otherwise redo the same transform. With
// dual cascades and a handful of lights that is the same work eight or more
// times per mesh per frame.
struct BoundingSphere {
  glm::vec3 center{0.0F};
  float radius{0.0F};
};

// Gribb/Hartmann plane extraction from a view-projection matrix.
//
// Planes face inward and are normalized, so plane.xyz is a unit normal and
// dot(plane.xyz, p) + plane.w is the signed distance from p to the plane.
//
// Near-plane row is r2 alone (not r3 + r2): that is the [0, 1] clip-depth form
// used by Vulkan and required by GLM_FORCE_DEPTH_ZERO_TO_ONE. Using the OpenGL
// [-1, 1] form here culls geometry just in front of the camera.
struct Frustum {
  std::array<glm::vec4, 6> planes{};

  [[nodiscard]] static auto from_view_proj(const glm::mat4 &view_proj) -> Frustum {
    // Column-major glm: row i of the matrix is (m[0][i], m[1][i], m[2][i], m[3][i]).
    const glm::vec4 r0(view_proj[0][0], view_proj[1][0], view_proj[2][0], view_proj[3][0]);
    const glm::vec4 r1(view_proj[0][1], view_proj[1][1], view_proj[2][1], view_proj[3][1]);
    const glm::vec4 r2(view_proj[0][2], view_proj[1][2], view_proj[2][2], view_proj[3][2]);
    const glm::vec4 r3(view_proj[0][3], view_proj[1][3], view_proj[2][3], view_proj[3][3]);

    const auto normalize_plane = [](glm::vec4 plane) -> glm::vec4 {
      const float length = glm::length(glm::vec3(plane));
      return length > 0.0F ? plane / length : plane;
    };

    Frustum frustum;
    frustum.planes = {
        normalize_plane(r3 + r0), // left
        normalize_plane(r3 - r0), // right
        normalize_plane(r3 + r1), // bottom
        normalize_plane(r3 - r1), // top
        normalize_plane(r2),      // near ([0, 1] depth)
        normalize_plane(r3 - r2), // far
    };
    return frustum;
  }

  [[nodiscard]] auto intersects(const BoundingSphere &sphere) const -> bool {
    for (const glm::vec4 &plane : planes) {
      if (glm::dot(glm::vec3(plane), sphere.center) + plane.w < -sphere.radius)
        return false;
    }
    return true;
  }

  [[nodiscard]] auto intersects_sphere(const glm::vec3 &center, float radius) const -> bool {
    return intersects(BoundingSphere{.center = center, .radius = radius});
  }
};

} // namespace engine
