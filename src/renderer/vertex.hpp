#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <cstddef>
#include <vector>

namespace engine {

struct ColoredVertex {
  float pos[2];
  float color[3];

  [[nodiscard]] static auto binding_description() -> vk::VertexInputBindingDescription {
    return {
        .binding = 0,
        .stride = sizeof(ColoredVertex),
        .inputRate = vk::VertexInputRate::eVertex,
    };
  }

  [[nodiscard]] static auto attribute_descriptions() -> std::array<vk::VertexInputAttributeDescription, 2> {
    return {{
        {
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = offsetof(ColoredVertex, pos),
        },
        {
            .location = 1,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(ColoredVertex, color),
        },
    }};
  }
};

[[nodiscard]] inline auto triangle_vertices() -> std::vector<ColoredVertex> {
  return {
      {{0.0F, -0.5F}, {1.0F, 0.0F, 0.0F}},
      {{0.5F, 0.5F}, {0.0F, 1.0F, 0.0F}},
      {{-0.5F, 0.5F}, {0.0F, 0.0F, 1.0F}},
  };
}

} // namespace engine
