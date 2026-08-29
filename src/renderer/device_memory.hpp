#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <cstring>
#include <utility>
#include <stdexcept>

namespace engine::detail {

// Extracted from buffer.hpp so the upload queue can allocate staging memory
// without buffer.hpp having to include it -- they would otherwise include each
// other.
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

// A host-visible staging buffer with `bytes` of `data` already in it.
//
// This was written out three times -- buffer.hpp, texture_image.hpp and
// texture_array.hpp -- identically each time: describe a transfer-source
// buffer, ask it what memory it wants, find a host-visible coherent type, bind,
// map, memcpy, unmap. Twenty-odd lines, three copies, and the kind of thing
// that only stays correct while nobody edits one of them.
struct Staging {
  vk::raii::Buffer buffer{nullptr};
  vk::raii::DeviceMemory memory{nullptr};
};

[[nodiscard]] inline auto make_staging(const vk::raii::Device &device,
                                       const vk::raii::PhysicalDevice &physical_device,
                                       vk::DeviceSize bytes, const void *data) -> Staging {
  vk::raii::Buffer buffer{device, vk::BufferCreateInfo{
                                      .size = bytes,
                                      .usage = vk::BufferUsageFlagBits::eTransferSrc,
                                      .sharingMode = vk::SharingMode::eExclusive,
                                  }};
  const vk::MemoryRequirements requirements = buffer.getMemoryRequirements();
  vk::raii::DeviceMemory memory{
      device, vk::MemoryAllocateInfo{
                  .allocationSize = requirements.size,
                  .memoryTypeIndex = find_memory_type(physical_device.getMemoryProperties(),
                                                      requirements.memoryTypeBits,
                                                      vk::MemoryPropertyFlagBits::eHostVisible |
                                                          vk::MemoryPropertyFlagBits::eHostCoherent),
              }};
  buffer.bindMemory(*memory, 0);
  if (data != nullptr) {
    void *mapped = memory.mapMemory(0, bytes);
    std::memcpy(mapped, data, static_cast<std::size_t>(bytes));
    memory.unmapMemory();
  }
  return {std::move(buffer), std::move(memory)};
}

} // namespace engine::detail
