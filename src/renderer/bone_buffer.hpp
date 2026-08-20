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

// Per-slot stride: k_max_bones matrices rounded up to the device's
// minStorageBufferOffsetAlignment. Free functions so the offset arithmetic --
// the part that decides which character reads which matrices -- can be tested
// without a device.
[[nodiscard]] constexpr auto bone_slot_stride(vk::DeviceSize min_alignment) -> std::uint32_t {
  constexpr vk::DeviceSize k_slot_bytes = sizeof(glm::mat4) * k_max_bones;
  const vk::DeviceSize alignment = min_alignment > 0 ? min_alignment : 1;
  return static_cast<std::uint32_t>(((k_slot_bytes + alignment - 1) / alignment) * alignment);
}

[[nodiscard]] constexpr auto bone_slot_offset(
    std::uint32_t frame_index,
    std::uint32_t slot,
    std::uint32_t slot_capacity,
    std::uint32_t slot_stride) -> std::uint32_t {
  return (frame_index * slot_capacity + slot) * slot_stride;
}

// One shared bone-matrix buffer for every skinned instance in the scene, bound
// through a dynamic-offset descriptor.
//
// The previous design gave each instance its own pair of buffers and four
// descriptor sets (main and shadow, times frames in flight), all rebuilt from
// scratch whenever the skinned instance count changed -- behind a device
// wait_idle. Spawning one more character in a horde of a hundred therefore
// reallocated two hundred buffers and four hundred descriptor sets, and stalled
// the GPU to do it.
//
// Here the descriptor is written once against the whole buffer. A draw selects
// its instance's slice with a dynamic offset, so spawning and despawning cost
// nothing but a slot index: no reallocation, no descriptor rewrite, no stall.
// The descriptor set then only has to vary by texture, not by instance.
//
// Layout is [frame][slot], with a fixed per-slot stride of k_max_bones matrices
// rounded up to the device's minStorageBufferOffsetAlignment. A fixed stride
// wastes memory on small skeletons (an 11-bone model uses 704 of 8192 bytes) but
// makes an offset pure arithmetic -- no table to rebuild, and nothing that can
// fall out of step with the slot assignment. At a few megabytes total that is
// the right trade for this engine.
class BonePalette {
public:
  void create(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      std::uint32_t slot_capacity,
      std::uint32_t frame_count) {
    const vk::PhysicalDeviceProperties properties = physical_device.getProperties();
    slot_stride_ = bone_slot_stride(properties.limits.minStorageBufferOffsetAlignment);
    slot_capacity_ = std::max(slot_capacity, 1U);
    frame_count_ = std::max(frame_count, 1U);

    const vk::DeviceSize buffer_size =
        static_cast<vk::DeviceSize>(slot_stride_) * slot_capacity_ * frame_count_;

    const vk::BufferCreateInfo buffer_info{
        .size = buffer_size,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    buffer_ = vk::raii::Buffer(device, buffer_info);

    const vk::MemoryRequirements requirements = buffer_.getMemoryRequirements();
    memory_ = vk::raii::DeviceMemory(
        device,
        vk::MemoryAllocateInfo{
            .allocationSize = requirements.size,
            .memoryTypeIndex = detail::find_memory_type(
                physical_device.getMemoryProperties(),
                requirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent),
        });
    buffer_.bindMemory(*memory_, 0);
    mapped_ = static_cast<std::byte *>(memory_.mapMemory(0, buffer_size));

    // A slot that is never written must not feed the vertex shader garbage --
    // an instance can exist for a frame before it is first posed.
    auto *matrices = reinterpret_cast<glm::mat4 *>(mapped_);
    const std::size_t matrix_count = static_cast<std::size_t>(buffer_size) / sizeof(glm::mat4);
    for (std::size_t i = 0; i < matrix_count; ++i)
      matrices[i] = glm::mat4(1.0F);
  }

  // Byte offset of one instance's slice, for bindDescriptorSets' dynamic offset.
  [[nodiscard]] auto dynamic_offset(std::uint32_t frame_index, std::uint32_t slot) const
      -> std::uint32_t {
    return bone_slot_offset(frame_index, slot, slot_capacity_, slot_stride_);
  }

  void write(std::uint32_t frame_index, std::uint32_t slot, std::span<const glm::mat4> joint_matrices) {
    if (mapped_ == nullptr || frame_index >= frame_count_ || slot >= slot_capacity_)
      return;
    const std::size_t count = std::min<std::size_t>(joint_matrices.size(), k_max_bones);
    if (count == 0)
      return;
    std::memcpy(mapped_ + dynamic_offset(frame_index, slot),
                joint_matrices.data(),
                count * sizeof(glm::mat4));
  }

  [[nodiscard]] auto buffer() const -> vk::Buffer { return *buffer_; }
  [[nodiscard]] auto slot_capacity() const -> std::uint32_t { return slot_capacity_; }
  [[nodiscard]] auto slot_stride() const -> std::uint32_t { return slot_stride_; }
  [[nodiscard]] auto valid() const -> bool { return mapped_ != nullptr; }

  // Size of the window a dynamic-offset descriptor exposes at each slot. The
  // buffer is an exact multiple of this, so offset + range never runs past the
  // end for any valid slot.
  [[nodiscard]] auto descriptor_range() const -> vk::DeviceSize { return slot_stride_; }

private:
  vk::raii::Buffer buffer_{nullptr};
  vk::raii::DeviceMemory memory_{nullptr};
  std::byte *mapped_{nullptr};
  std::uint32_t slot_stride_{0};
  std::uint32_t slot_capacity_{0};
  std::uint32_t frame_count_{0};
};

} // namespace engine
