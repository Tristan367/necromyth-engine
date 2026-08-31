#pragma once

#include "renderer/buffer.hpp"
#include "scene/bounds.hpp"
#include "scene/mesh_source.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace engine {

class MeshGpu {
public:
  // Queues both buffers for upload without blocking. The copies are replayed
  // into the frame's command buffer, so a streaming world pays no GPU round trip
  // per mesh.
  auto upload(
      DeviceAllocator &allocator,
      vk::raii::Device &device,
      UploadQueue &uploads,
      const MeshSource &mesh,
      const AABB &bounds) -> bool {
    index_count_ = static_cast<std::uint32_t>(mesh.indices.size());
    bounds_ = bounds;
    // Whichever vertex vector the mesh filled decides the stride; the terrain
    // pipelines are built for exactly this 36-byte layout.
    const vk::DeviceSize vertex_bytes = mesh.is_terrain()
        ? static_cast<vk::DeviceSize>(sizeof(TerrainVertex) * mesh.terrain_vertices.size())
        : static_cast<vk::DeviceSize>(sizeof(MeshVertex) * mesh.vertices.size());
    const void *vertex_data = mesh.is_terrain()
        ? static_cast<const void *>(mesh.terrain_vertices.data())
        : static_cast<const void *>(mesh.vertices.data());
    const bool vertices_staged = vertex_buffer_.upload_deferred(
        allocator,
        device,
        uploads,
        vertex_bytes,
        vk::BufferUsageFlagBits::eVertexBuffer,
        vertex_data);
    const bool indices_staged = index_buffer_.upload_deferred(
        allocator,
        device,
        uploads,
        static_cast<vk::DeviceSize>(sizeof(std::uint32_t) * mesh.indices.size()),
        vk::BufferUsageFlagBits::eIndexBuffer,
        mesh.indices.data());
    return vertices_staged && indices_staged;
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
