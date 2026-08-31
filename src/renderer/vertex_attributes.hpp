#pragma once

#include "renderer/vertex.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <cstddef>

namespace engine {

[[nodiscard]] inline auto mesh_binding_description() -> vk::VertexInputBindingDescription {
  return {.binding = 0, .stride = sizeof(MeshVertex), .inputRate = vk::VertexInputRate::eVertex};
}

// Every attribute the vertex format carries. Kept because the sky and shadow
// pipelines index [0] out of it for position, and because it documents the
// layout in one place; no pipeline binds the whole thing.
[[nodiscard]] inline auto attribute_descriptions() -> std::array<vk::VertexInputAttributeDescription, 7> {
  return {{
      {.location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(MeshVertex, pos)},
      {.location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(MeshVertex, normal)},
      {.location = 2, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(MeshVertex, color)},
      {.location = 3, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(MeshVertex, tex_coord)},
      {.location = 4, .binding = 0, .format = vk::Format::eR32G32B32A32Sfloat, .offset = offsetof(MeshVertex, joint_indices)},
      {.location = 5, .binding = 0, .format = vk::Format::eR32G32B32A32Sfloat, .offset = offsetof(MeshVertex, joint_weights)},
      {.location = 6, .binding = 0, .format = vk::Format::eR32Uint, .offset = offsetof(MeshVertex, material)},
  }};
}

// What the skinned main-pass shader actually reads: VSInputSkinned in
// lib/mesh_types_skinned.slang, which is position through joint weights.
//
// This used to bind the full seven, so every skinned vertex fetched a material
// index no shader consumed -- wasted bandwidth on every character in both the
// main and shadow passes. Validation says so plainly ("Vertex attribute at
// location 6 not consumed by vertex shader") and nothing was reading it.
//
// Note the asymmetry with static meshes, which DO carry a per-vertex material:
// a chunk packs many textures into one mesh, a character does not. If a skinned
// mesh ever needs per-vertex materials, add inMaterial to VSInputSkinned and
// this list together -- they have to agree.
[[nodiscard]] inline auto skinned_attribute_descriptions() -> std::array<vk::VertexInputAttributeDescription, 6> {
  return {{
      {.location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(MeshVertex, pos)},
      {.location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(MeshVertex, normal)},
      {.location = 2, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(MeshVertex, color)},
      {.location = 3, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(MeshVertex, tex_coord)},
      {.location = 4, .binding = 0, .format = vk::Format::eR32G32B32A32Sfloat, .offset = offsetof(MeshVertex, joint_indices)},
      {.location = 5, .binding = 0, .format = vk::Format::eR32G32B32A32Sfloat, .offset = offsetof(MeshVertex, joint_weights)},
  }};
}

[[nodiscard]] inline auto static_attribute_descriptions() -> std::array<vk::VertexInputAttributeDescription, 5> {
  return {{
      {.location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(MeshVertex, pos)},
      {.location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(MeshVertex, normal)},
      {.location = 2, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(MeshVertex, color)},
      {.location = 3, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(MeshVertex, tex_coord)},
      {.location = 6, .binding = 0, .format = vk::Format::eR32Uint, .offset = offsetof(MeshVertex, material)},
  }};
}

// The packed terrain format. Same locations as the static set minus colour
// (location 2), so the fragment shaders are shared and only the vertex entry
// differs (vertMainTerrain supplies white).
[[nodiscard]] inline auto terrain_binding_description() -> vk::VertexInputBindingDescription {
  return {.binding = 0, .stride = sizeof(TerrainVertex), .inputRate = vk::VertexInputRate::eVertex};
}

[[nodiscard]] inline auto terrain_attribute_descriptions() -> std::array<vk::VertexInputAttributeDescription, 4> {
  return {{
      {.location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(TerrainVertex, pos)},
      {.location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(TerrainVertex, normal)},
      {.location = 3, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(TerrainVertex, tex_coord)},
      {.location = 6, .binding = 0, .format = vk::Format::eR32Uint, .offset = offsetof(TerrainVertex, material)},
  }};
}

// Depth-only passes read position alone, but the STRIDE still has to be the
// terrain one -- the shadow pipelines built from mesh_binding_description walk
// 80-byte steps through a 36-byte buffer.
[[nodiscard]] inline auto terrain_shadow_attribute_descriptions() -> std::array<vk::VertexInputAttributeDescription, 1> {
  return {{
      {.location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(TerrainVertex, pos)},
  }};
}

[[nodiscard]] inline auto shadow_skinned_attribute_descriptions() -> std::array<vk::VertexInputAttributeDescription, 3> {
  return {{
      {.location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(MeshVertex, pos)},
      {.location = 4, .binding = 0, .format = vk::Format::eR32G32B32A32Sfloat, .offset = offsetof(MeshVertex, joint_indices)},
      {.location = 5, .binding = 0, .format = vk::Format::eR32G32B32A32Sfloat, .offset = offsetof(MeshVertex, joint_weights)},
  }};
}

} // namespace engine
