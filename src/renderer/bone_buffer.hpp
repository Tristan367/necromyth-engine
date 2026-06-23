#pragma once

#include "renderer/buffer.hpp"
#include "scene/animation_types.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <glm/mat4x4.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace engine {

class BoneStorageBufferSet {
public:
  void create(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      std::uint32_t bone_count,
      std::uint32_t frame_count) {
    if (bone_count == 0 || bone_count > k_max_bones)
      throw std::runtime_error("Invalid bone count for SSBO");

    bone_count_ = bone_count;
    const vk::DeviceSize buffer_size = sizeof(glm::mat4) * bone_count;
    const auto memory_properties = physical_device.getMemoryProperties();

    buffers_.clear();
    memory_.clear();
    mapped_.clear();
    buffers_.reserve(frame_count);
    memory_.reserve(frame_count);
    mapped_.reserve(frame_count);

    for (std::uint32_t i = 0; i < frame_count; ++i) {
      const vk::BufferCreateInfo buffer_info{
          .size = buffer_size,
          .usage = vk::BufferUsageFlagBits::eStorageBuffer,
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
                  vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent),
          });

      buffers_.back().bindMemory(*memory_.back(), 0);
      mapped_.push_back(memory_.back().mapMemory(0, buffer_size));
    }
  }

  void write(std::uint32_t frame_index, std::span<const glm::mat4> joint_matrices) {
    if (frame_index >= mapped_.size())
      throw std::runtime_error("Bone SSBO write: frame_index out of range");
    const std::size_t bytes = std::min(
        joint_matrices.size() * sizeof(glm::mat4),
        sizeof(glm::mat4) * static_cast<std::size_t>(bone_count_));
    std::memcpy(mapped_[frame_index], joint_matrices.data(), bytes);
  }

  [[nodiscard]] auto buffer(std::uint32_t frame_index) const -> vk::Buffer {
    if (frame_index >= buffers_.size())
      throw std::runtime_error("Bone SSBO buffer: frame_index out of range");
    return *buffers_[frame_index];
  }

  [[nodiscard]] auto bone_count() const -> std::uint32_t {
    return bone_count_;
  }

private:
  std::vector<vk::raii::Buffer> buffers_;
  std::vector<vk::raii::DeviceMemory> memory_;
  std::vector<void *> mapped_;
  std::uint32_t bone_count_{};
};

} // namespace engine
