#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace engine {

// The streaming-terrain vertex: what a voxel chunk actually needs and nothing
// else. 36 bytes against MeshVertex's 80 -- the colour is always white and the
// sixteen bytes of joints and sixteen of weights are meaningless for terrain,
// so carrying MeshVertex cost 44 dead bytes per vertex, which at a streaming
// world's vertex counts was most of the VRAM and most of the upload bandwidth.
//
// This is the C# reference's layout (36 bytes: position, normal, UV, and one
// scalar carrying the texture index and both baked light channels -- decoded in
// Voxel.gdshader). `material` keeps the exact encoding MeshVertex::material
// already had, so the shader-side decode is shared.
struct TerrainVertex {
  float pos[3];
  float normal[3];
  float tex_coord[2];
  std::uint32_t material;
};
static_assert(sizeof(TerrainVertex) == 36, "terrain vertex must stay packed");

struct MeshVertex {
  float pos[3];
  float normal[3];
  float color[3];
  float tex_coord[2];
  // Per-vertex texture array layer, added to the instance's base layer. Lets one
  // mesh carry many materials -- a voxel chunk holds grass, dirt, stone, bark and
  // leaves in a single draw. 0 means "just use the instance's layer", so meshes
  // that do not care are unaffected.
  std::uint32_t material;
  float joint_indices[4];
  float joint_weights[4];

  [[nodiscard]] auto operator==(const MeshVertex &other) const -> bool {
    return std::memcmp(pos, other.pos, sizeof(pos)) == 0 &&
           std::memcmp(normal, other.normal, sizeof(normal)) == 0 &&
           std::memcmp(color, other.color, sizeof(color)) == 0 &&
           std::memcmp(tex_coord, other.tex_coord, sizeof(tex_coord)) == 0 &&
           material == other.material &&
           std::memcmp(joint_indices, other.joint_indices, sizeof(joint_indices)) == 0 &&
           std::memcmp(joint_weights, other.joint_weights, sizeof(joint_weights)) == 0;
  }
};

} // namespace engine
