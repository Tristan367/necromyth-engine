#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace engine {

struct MeshVertex {
  float pos[3];
  float normal[3];
  float color[3];
  float tex_coord[2];
  float joint_indices[4];
  float joint_weights[4];

  [[nodiscard]] auto operator==(const MeshVertex &other) const -> bool {
    return std::memcmp(pos, other.pos, sizeof(pos)) == 0 &&
           std::memcmp(normal, other.normal, sizeof(normal)) == 0 &&
           std::memcmp(color, other.color, sizeof(color)) == 0 &&
           std::memcmp(tex_coord, other.tex_coord, sizeof(tex_coord)) == 0 &&
           std::memcmp(joint_indices, other.joint_indices, sizeof(joint_indices)) == 0 &&
           std::memcmp(joint_weights, other.joint_weights, sizeof(joint_weights)) == 0;
  }
};

} // namespace engine
