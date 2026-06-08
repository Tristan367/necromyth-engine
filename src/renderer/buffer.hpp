#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace engine {

namespace detail {

[[nodiscard]] inline auto find_memory_type(
    vk::PhysicalDeviceMemoryProperties memory_properties,
    std::uint32_t type_filter,
    vk::MemoryPropertyFlags properties) -> std::uint32_t {
  for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
    if ((type_filter & (1U << i)) &&
        (memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
      return i;
  }

  throw std::runtime_error("Failed to find suitable memory type for buffer");
}

inline void copy_buffer(
    vk::raii::Device &device,
    vk::raii::CommandPool &command_pool,
    vk::raii::Queue &queue,
    vk::Buffer src,
    vk::Buffer dst,
    vk::DeviceSize size) {
  const vk::CommandBufferAllocateInfo allocate_info{
      .commandPool = *command_pool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1,
  };

  vk::raii::CommandBuffer command_buffer = std::move(
      vk::raii::CommandBuffers(device, allocate_info).front());

  command_buffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  command_buffer.copyBuffer(src, dst, vk::BufferCopy{.size = size});
  command_buffer.end();

  const vk::CommandBufferSubmitInfo command_buffer_info{.commandBuffer = *command_buffer};
  const vk::SubmitInfo2 submit_info{
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &command_buffer_info,
  };

  vk::raii::Fence fence(device, vk::FenceCreateInfo{});
  queue.submit2(submit_info, *fence);
  if (device.waitForFences(*fence, vk::True, UINT64_MAX) != vk::Result::eSuccess)
    throw std::runtime_error("Failed to wait for buffer upload fence");
}

} // namespace detail

class DeviceLocalBuffer {
public:
  void upload(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      vk::raii::CommandPool &command_pool,
      vk::raii::Queue &queue,
      vk::DeviceSize size,
      vk::BufferUsageFlags usage,
      const void *data) {
    const vk::BufferCreateInfo staging_info{
        .size = size,
        .usage = vk::BufferUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    vk::raii::Buffer staging_buffer{device, staging_info};
    const vk::MemoryRequirements staging_requirements = staging_buffer.getMemoryRequirements();
    const auto memory_properties = physical_device.getMemoryProperties();

    vk::raii::DeviceMemory staging_memory{
        device,
        vk::MemoryAllocateInfo{
            .allocationSize = staging_requirements.size,
            .memoryTypeIndex = detail::find_memory_type(
                memory_properties,
                staging_requirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
        }};

    staging_buffer.bindMemory(*staging_memory, 0);

    void *mapped = staging_memory.mapMemory(0, size);
    std::memcpy(mapped, data, static_cast<std::size_t>(size));
    staging_memory.unmapMemory();

    const vk::BufferCreateInfo device_info{
        .size = size,
        .usage = usage | vk::BufferUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    buffer_ = vk::raii::Buffer(device, device_info);
    const vk::MemoryRequirements device_requirements = buffer_.getMemoryRequirements();
    memory_ = vk::raii::DeviceMemory{
        device,
        vk::MemoryAllocateInfo{
            .allocationSize = device_requirements.size,
            .memoryTypeIndex = detail::find_memory_type(
                memory_properties,
                device_requirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal),
        }};

    buffer_.bindMemory(*memory_, 0);
    detail::copy_buffer(device, command_pool, queue, *staging_buffer, *buffer_, size);
  }

  [[nodiscard]] auto handle() const -> vk::Buffer {
    return *buffer_;
  }

private:
  vk::raii::Buffer buffer_{nullptr};
  vk::raii::DeviceMemory memory_{nullptr};
};

} // namespace engine
