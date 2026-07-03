#pragma once

#include "renderer/buffer.hpp"
#include "scene/point_light.hpp"
#include "scene/spot_light.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace engine {

class LightStorageBuffer {
public:
  struct GpuPointLight {
    float pos_range[4];
    float color_intensity[4];
  };

  struct GpuSpotLight {
    float pos_range[4];
    float color_intensity[4];
    float dir_pad[4];
    float angles[4];
  };

  void create(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      std::size_t max_lights) {
    max_lights_ = max_lights;
    const auto mem_props = physical_device.getMemoryProperties();
    const vk::DeviceSize buf_size = 16 + max_lights * (sizeof(GpuPointLight) + sizeof(GpuSpotLight));
    buffer_ = vk::raii::Buffer(device, vk::BufferCreateInfo{
        .size = buf_size,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer,
    });
    const auto requirements = buffer_.getMemoryRequirements();
    const auto mem_type = detail::find_memory_type(
        mem_props, requirements.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    memory_ = vk::raii::DeviceMemory(device,
        vk::MemoryAllocateInfo{.allocationSize = requirements.size, .memoryTypeIndex = mem_type});
    buffer_.bindMemory(*memory_, 0);
  }

  void write(const std::vector<PointLight> &point_lights,
             const std::vector<SpotLight> &spot_lights) {
    const std::size_t num_point = std::min(point_lights.size(), max_lights_);
    const std::size_t num_spot = std::min(spot_lights.size(), max_lights_ - num_point);
    const vk::DeviceSize size = 16 + num_point * sizeof(GpuPointLight) + num_spot * sizeof(GpuSpotLight);
    if (size == 0) return;

    auto *data = static_cast<std::uint8_t *>(memory_.mapMemory(0, size));
    auto *header = reinterpret_cast<std::uint32_t *>(data);
    header[0] = static_cast<std::uint32_t>(num_point);
    header[1] = static_cast<std::uint32_t>(num_spot);

    auto *ptrs = reinterpret_cast<GpuPointLight *>(data + 16);
    for (std::size_t i = 0; i < num_point; ++i) {
      ptrs[i].pos_range[0] = point_lights[i].position.x;
      ptrs[i].pos_range[1] = point_lights[i].position.y;
      ptrs[i].pos_range[2] = point_lights[i].position.z;
      ptrs[i].pos_range[3] = point_lights[i].range;
      ptrs[i].color_intensity[0] = point_lights[i].color.r;
      ptrs[i].color_intensity[1] = point_lights[i].color.g;
      ptrs[i].color_intensity[2] = point_lights[i].color.b;
      ptrs[i].color_intensity[3] = point_lights[i].intensity;
    }

    auto *sptr = reinterpret_cast<GpuSpotLight *>(data + 16 + num_point * sizeof(GpuPointLight));
    for (std::size_t i = 0; i < num_spot; ++i) {
      sptr[i].pos_range[0] = spot_lights[i].position.x;
      sptr[i].pos_range[1] = spot_lights[i].position.y;
      sptr[i].pos_range[2] = spot_lights[i].position.z;
      sptr[i].pos_range[3] = spot_lights[i].range;
      sptr[i].color_intensity[0] = spot_lights[i].color.r;
      sptr[i].color_intensity[1] = spot_lights[i].color.g;
      sptr[i].color_intensity[2] = spot_lights[i].color.b;
      sptr[i].color_intensity[3] = spot_lights[i].intensity;
      sptr[i].dir_pad[0] = spot_lights[i].direction.x;
      sptr[i].dir_pad[1] = spot_lights[i].direction.y;
      sptr[i].dir_pad[2] = spot_lights[i].direction.z;
      sptr[i].dir_pad[3] = 0;
      sptr[i].angles[0] = spot_lights[i].inner_angle;
      sptr[i].angles[1] = spot_lights[i].outer_angle;
    }

    memory_.unmapMemory();
  }

  [[nodiscard]] auto buffer() const -> const vk::raii::Buffer & { return buffer_; }
  [[nodiscard]] auto buffer_ptr() const -> vk::Buffer { return *buffer_; }

private:
  vk::raii::Buffer buffer_{nullptr};
  vk::raii::DeviceMemory memory_{nullptr};
  std::size_t max_lights_{0};
};

} // namespace engine
