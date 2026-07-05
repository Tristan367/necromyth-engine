#pragma once

#include "renderer/buffer.hpp"
#include "scene/mesh_source.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <glm/vec3.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace engine {

struct AABB {
  glm::vec3 min{std::numeric_limits<float>::max()};
  glm::vec3 max{-std::numeric_limits<float>::max()};

  void extend(const glm::vec3 &p) {
    min = glm::min(min, p);
    max = glm::max(max, p);
  }

  [[nodiscard]] auto center() const -> glm::vec3 { return (min + max) * 0.5F; }
  [[nodiscard]] auto radius() const -> float { return glm::distance(min, max) * 0.5F; }
};

class MeshGpu {
public:
  void upload(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      vk::raii::CommandPool &command_pool,
      vk::raii::Queue &queue,
      const MeshSource &mesh) {
    index_count_ = static_cast<std::uint32_t>(mesh.indices.size());
    for (const MeshVertex &v : mesh.vertices)
      bounds_.extend(glm::vec3(v.pos[0], v.pos[1], v.pos[2]));
    vertex_buffer_.upload(
        physical_device,
        device,
        command_pool,
        queue,
        static_cast<vk::DeviceSize>(sizeof(MeshVertex) * mesh.vertices.size()),
        vk::BufferUsageFlagBits::eVertexBuffer,
        mesh.vertices.data());
    index_buffer_.upload(
        physical_device,
        device,
        command_pool,
        queue,
        static_cast<vk::DeviceSize>(sizeof(std::uint32_t) * mesh.indices.size()),
        vk::BufferUsageFlagBits::eIndexBuffer,
        mesh.indices.data());
  }

  [[nodiscard]] auto index_count() const -> std::uint32_t {
    return index_count_;
  }

  [[nodiscard]] auto vertex_buffer() const -> vk::Buffer {
    return vertex_buffer_.handle();
  }

  [[nodiscard]] auto index_buffer() const -> vk::Buffer {
    return index_buffer_.handle();
  }

  [[nodiscard]] auto bounds() const -> const AABB & { return bounds_; }

private:
  DeviceLocalBuffer vertex_buffer_;
  DeviceLocalBuffer index_buffer_;
  std::uint32_t index_count_{};
  AABB bounds_{};
};

} // namespace engine
