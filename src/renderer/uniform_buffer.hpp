#pragma once

#include "renderer/buffer.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>

#include <cstdint>
#include <cstring>
#include <vector>

namespace engine {

struct FrameUniformBufferObject {
  alignas(16) glm::mat4 view{};
  alignas(16) glm::mat4 proj{};
  alignas(16) glm::vec4 camera_position{};
  alignas(16) glm::mat4 view_sky{};
  alignas(16) glm::vec4 light_direction{};
  alignas(16) glm::vec4 light_color{};
  alignas(16) glm::mat4 light_view_proj{};
};

[[nodiscard]] inline auto view_without_translation(const glm::mat4 &view) -> glm::mat4 {
  glm::mat4 result = view;
  result[3] = glm::vec4(0.0F, 0.0F, 0.0F, 1.0F);
  return result;
}

class UniformBufferSet {
public:
  void create(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      std::uint32_t frame_count) {
    const auto memory_properties = physical_device.getMemoryProperties();
    const vk::DeviceSize buffer_size = sizeof(FrameUniformBufferObject);

    buffers_.clear();
    memory_.clear();
    mapped_.clear();
    buffers_.reserve(frame_count);
    memory_.reserve(frame_count);
    mapped_.reserve(frame_count);

    for (std::uint32_t i = 0; i < frame_count; ++i) {
      const vk::BufferCreateInfo buffer_info{
          .size = buffer_size,
          .usage = vk::BufferUsageFlagBits::eUniformBuffer,
          .sharingMode = vk::SharingMode::eExclusive,
      };

      buffers_.emplace_back(device, buffer_info);
      const vk::MemoryRequirements requirements = buffers_.back().getMemoryRequirements();

      memory_.emplace_back(
          device,
          vk::MemoryAllocateInfo{
              .allocationSize = requirements.size,
              .memoryTypeIndex = detail::find_memory_type(
                  memory_properties,
                  requirements.memoryTypeBits,
                  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
          });

      buffers_.back().bindMemory(*memory_.back(), 0);
      mapped_.push_back(memory_.back().mapMemory(0, buffer_size));
    }
  }

  void write(std::uint32_t frame_index, const FrameUniformBufferObject &ubo) const {
    std::memcpy(mapped_[frame_index], &ubo, sizeof(FrameUniformBufferObject));
  }

  [[nodiscard]] auto buffer(std::uint32_t frame_index) const -> vk::Buffer {
    return *buffers_[frame_index];
  }

private:
  std::vector<vk::raii::Buffer> buffers_;
  std::vector<vk::raii::DeviceMemory> memory_;
  std::vector<void *> mapped_;
};

} // namespace engine
