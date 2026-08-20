#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
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

} // namespace engine::detail
