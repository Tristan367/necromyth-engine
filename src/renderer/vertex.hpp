#pragma once

#include <vulkan/vulkan_raii.hpp>

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

  [[nodiscard]] auto operator==(const MeshVertex &other) const -> bool {
    return std::memcmp(pos, other.pos, sizeof(pos)) == 0 &&
           std::memcmp(normal, other.normal, sizeof(normal)) == 0 &&
           std::memcmp(color, other.color, sizeof(color)) == 0 &&
           std::memcmp(tex_coord, other.tex_coord, sizeof(tex_coord)) == 0;
  }

  [[nodiscard]] static auto binding_description() -> vk::VertexInputBindingDescription {
    return {
        .binding = 0,
        .stride = sizeof(MeshVertex),
        .inputRate = vk::VertexInputRate::eVertex,
    };
  }

  [[nodiscard]] static auto attribute_descriptions() -> std::array<vk::VertexInputAttributeDescription, 4> {
    return {{
        {
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(MeshVertex, pos),
        },
        {
            .location = 1,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(MeshVertex, normal),
        },
        {
            .location = 2,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(MeshVertex, color),
        },
        {
            .location = 3,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = offsetof(MeshVertex, tex_coord),
        },
    }};
  }
};

} // namespace engine
