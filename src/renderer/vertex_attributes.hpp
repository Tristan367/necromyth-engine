#pragma once

#include "renderer/vertex.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <cstddef>

namespace engine {

[[nodiscard]] inline auto mesh_binding_description() -> vk::VertexInputBindingDescription {
  return {.binding = 0, .stride = sizeof(MeshVertex), .inputRate = vk::VertexInputRate::eVertex};
}

[[nodiscard]] inline auto attribute_descriptions() -> std::array<vk::VertexInputAttributeDescription, 6> {
  return {{
      {.location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(MeshVertex, pos)},
      {.location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(MeshVertex, normal)},
      {.location = 2, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(MeshVertex, color)},
      {.location = 3, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(MeshVertex, tex_coord)},
      {.location = 4, .binding = 0, .format = vk::Format::eR32G32B32A32Sfloat, .offset = offsetof(MeshVertex, joint_indices)},
      {.location = 5, .binding = 0, .format = vk::Format::eR32G32B32A32Sfloat, .offset = offsetof(MeshVertex, joint_weights)},
  }};
}

[[nodiscard]] inline auto static_attribute_descriptions() -> std::array<vk::VertexInputAttributeDescription, 4> {
  return {{
      {.location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(MeshVertex, pos)},
      {.location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(MeshVertex, normal)},
      {.location = 2, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(MeshVertex, color)},
      {.location = 3, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(MeshVertex, tex_coord)},
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
