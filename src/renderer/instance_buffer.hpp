#pragma once

#include "renderer/buffer.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <glm/mat4x4.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace engine {

inline constexpr std::uint32_t k_no_bone_base = 0xFFFFFFFFU;

// One record per draw. Everything here used to travel in push constants, which
// meant a draw call could describe exactly one object -- 92 bytes pushed per
// object, and no way to draw two of anything at once.
//
// With the data in a buffer instead, the vertex shader looks its object up by
// instance index, so N objects sharing a mesh and material collapse into one
// drawIndexed. That is the whole point: a horde, a forest, or a field of
// impostor billboards should cost one draw call, not one per object.
struct GpuInstance {
  glm::mat4 model{1.0F};
  std::uint32_t texture_layer{0};
  std::uint32_t sample_texture_array{0};
  // First bone matrix of this instance's slice of the shared palette, in matrix
  // units. k_no_bone_base for unskinned geometry.
  std::uint32_t bone_base{k_no_bone_base};
  std::uint32_t pad{0};
};
static_assert(sizeof(GpuInstance) == 80, "shader GpuInstance layout must match");

// Per-frame instance records, host-visible so they can be written directly while
// building the frame's draw lists.
class InstanceBuffer {
public:
  void create(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      std::uint32_t capacity,
      std::uint32_t frame_count) {
    capacity_ = std::max(capacity, 1U);
    frame_count_ = std::max(frame_count, 1U);

    const vk::DeviceSize per_frame = static_cast<vk::DeviceSize>(capacity_) * sizeof(GpuInstance);
    const auto memory_properties = physical_device.getMemoryProperties();

    buffers_.clear();
    memory_.clear();
    mapped_.clear();
    for (std::uint32_t i = 0; i < frame_count_; ++i) {
      buffers_.emplace_back(device, vk::BufferCreateInfo{
          .size = per_frame,
          .usage = vk::BufferUsageFlagBits::eStorageBuffer,
          .sharingMode = vk::SharingMode::eExclusive,
      });
      const vk::MemoryRequirements requirements = buffers_.back().getMemoryRequirements();
      memory_.emplace_back(device, vk::MemoryAllocateInfo{
          .allocationSize = requirements.size,
          .memoryTypeIndex = detail::find_memory_type(
              memory_properties,
              requirements.memoryTypeBits,
              vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
      });
      buffers_.back().bindMemory(*memory_.back(), 0);
      mapped_.push_back(static_cast<GpuInstance *>(memory_.back().mapMemory(0, per_frame)));
    }
  }

  // Begins a frame's records. Returns false if the buffer is not usable.
  auto begin_frame(std::uint32_t frame_index) -> bool {
    if (frame_index >= mapped_.size())
      return false;
    current_ = mapped_[frame_index];
    used_ = 0;
    overflowed_ = false;
    return true;
  }

  // Appends one record, returning its index. On overflow the record is dropped
  // and the last valid index is returned; the caller reports it once rather than
  // letting objects quietly render on top of each other.
  auto push(const GpuInstance &instance) -> std::uint32_t {
    if (current_ == nullptr || used_ >= capacity_) {
      overflowed_ = true;
      return capacity_ > 0 ? capacity_ - 1 : 0;
    }
    const std::uint32_t index = used_++;
    current_[index] = instance;
    return index;
  }

  [[nodiscard]] auto buffer(std::uint32_t frame_index) const -> vk::Buffer {
    return frame_index < buffers_.size() ? *buffers_[frame_index] : vk::Buffer{};
  }

  [[nodiscard]] auto used() const -> std::uint32_t { return used_; }
  [[nodiscard]] auto capacity() const -> std::uint32_t { return capacity_; }
  [[nodiscard]] auto overflowed() const -> bool { return overflowed_; }
  [[nodiscard]] auto valid() const -> bool { return !mapped_.empty(); }

private:
  std::vector<vk::raii::Buffer> buffers_;
  std::vector<vk::raii::DeviceMemory> memory_;
  std::vector<GpuInstance *> mapped_;
  GpuInstance *current_{nullptr};
  std::uint32_t used_{0};
  std::uint32_t capacity_{0};
  std::uint32_t frame_count_{0};
  bool overflowed_{false};
};

} // namespace engine
