#pragma once

#include "renderer/buffer.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

namespace engine {

struct UniformBufferObject {
  glm::mat4 model{};
  glm::mat4 view{};
  glm::mat4 proj{};
};

class UniformBufferSet {
public:
  void create(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      std::uint32_t frame_count) {
    const auto memory_properties = physical_device.getMemoryProperties();
    const vk::DeviceSize buffer_size = sizeof(UniformBufferObject);

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

  void write(std::uint32_t frame_index, const UniformBufferObject &ubo) const {
    std::memcpy(mapped_[frame_index], &ubo, sizeof(UniformBufferObject));
  }

  [[nodiscard]] auto buffer(std::uint32_t frame_index) const -> vk::Buffer {
    return *buffers_[frame_index];
  }

  [[nodiscard]] static auto make_rotating_ubo(vk::Extent2D extent) -> UniformBufferObject {
    static const auto start_time = std::chrono::high_resolution_clock::now();

    const auto current_time = std::chrono::high_resolution_clock::now();
    const float time = std::chrono::duration<float>(current_time - start_time).count();

    UniformBufferObject ubo{};
    ubo.model = glm::rotate(glm::mat4(1.0F), time * glm::radians(90.0F), glm::vec3(0.0F, 0.0F, 1.0F));
    ubo.view = glm::lookAt(glm::vec3(2.0F, 2.0F, 2.0F), glm::vec3(0.0F), glm::vec3(0.0F, 0.0F, 1.0F));
    ubo.proj = glm::perspective(
        glm::radians(45.0F),
        static_cast<float>(extent.width) / static_cast<float>(extent.height),
        0.1F,
        10.0F);
    ubo.proj[1][1] *= -1.0F;
    return ubo;
  }

private:
  std::vector<vk::raii::Buffer> buffers_;
  std::vector<vk::raii::DeviceMemory> memory_;
  std::vector<void *> mapped_;
};

} // namespace engine
