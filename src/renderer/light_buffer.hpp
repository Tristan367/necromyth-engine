#pragma once

#include "renderer/buffer.hpp"
#include "scene/point_light.hpp"
#include "scene/shadow_assignment.hpp"
#include "scene/shadow_utils.hpp"
#include "scene/spot_light.hpp"

#include "engine_glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>

#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

namespace engine {

class LightStorageBuffer {
public:
  static constexpr std::uint32_t k_frames_in_flight = 2;
  static constexpr vk::DeviceSize k_header_size = 16;  // num_point + num_spot (2× u32), 8B pad

  // 48 bytes. The previous layout carried a 64-byte shadow_matrix that a point
  // light never uses -- only its [0][0] element was read, as a "casts shadow"
  // boolean -- plus an unused atlas_rect. Point lights sample a cubemap, not an
  // atlas, so the matrix and rect were dead weight in a buffer the fragment
  // shader walks once per light per pixel.
  struct GpuPointLight {
    float pos_range[4];
    float color_intensity[4];
    // x = cubemap array layer this light was assigned this frame, or a negative
    // value for "no shadow map"; yzw reserved.
    float shadow_params[4];
  };
  static_assert(sizeof(GpuPointLight) == 48, "shader lighting.slang assumes 48-byte point lights");

  struct GpuSpotLight {
    float pos_range[4];
    float color_intensity[4];
    float dir_pad[4];
    float angles[4];
    float shadow_matrix[16];
    float atlas_rect[4];
  };

  void create(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      std::size_t max_lights) {
    max_lights_ = max_lights;
    const auto mem_props = physical_device.getMemoryProperties();
    const vk::DeviceSize buf_size = k_header_size + max_lights * (sizeof(GpuPointLight) + sizeof(GpuSpotLight));

    for (std::size_t i = 0; i < k_frames_in_flight; ++i) {
      buffers_[i].emplace(device, vk::BufferCreateInfo{
          .size = buf_size,
          .usage = vk::BufferUsageFlagBits::eStorageBuffer,
      });
      auto reqs = buffers_[i]->getMemoryRequirements();
      auto mt = detail::find_memory_type(mem_props, reqs.memoryTypeBits,
          vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
      memories_[i].emplace(device,
          vk::MemoryAllocateInfo{.allocationSize = reqs.size, .memoryTypeIndex = mt});
      buffers_[i]->bindMemory(**memories_[i], 0);
      mapped_[i] = static_cast<std::uint8_t *>(memories_[i]->mapMemory(0, buf_size));
    }
  }

  static auto compute_shadow_view_proj(const SpotLight &l) -> glm::mat4 {
    const glm::vec3 dir = glm::normalize(l.direction);
    const glm::mat4 proj = glm::perspective(l.outer_angle * 2.0F, 1.0F, 0.1F, l.range);
    const glm::vec3 up = detail::stable_up_for_light(dir);
    const glm::mat4 view = glm::lookAt(l.position, l.position + dir, up);
    return proj * view;
  }

  static auto compute_shadow_matrix(const SpotLight &l) -> glm::mat4 {
    const glm::mat4 bias(glm::vec4(0.5, 0.0, 0.0, 0.0),
                         glm::vec4(0.0, 0.5, 0.0, 0.0),
                         glm::vec4(0.0, 0.0, 1.0, 0.0),
                         glm::vec4(0.5, 0.5, 0.0, 1.0));
    return bias * compute_shadow_view_proj(l);
  }


  void write(std::uint32_t frame_index,
             const std::vector<PointLight> &point_lights,
             const std::vector<SpotLight> &spot_lights,
             const ShadowSlotAssignment &shadows,
             std::uint32_t spot_shadow_capacity) {
    if (frame_index >= k_frames_in_flight) return;
    const std::size_t num_point = std::min(point_lights.size(), max_lights_);
    const std::size_t num_spot = std::min(spot_lights.size(), max_lights_ - num_point);
    const vk::DeviceSize size = 16 + num_point * sizeof(GpuPointLight) + num_spot * sizeof(GpuSpotLight);
    if (size == 0 || !mapped_[frame_index]) return;

    auto *data = mapped_[frame_index];
    auto *header = reinterpret_cast<std::uint32_t *>(data);
    header[0] = static_cast<std::uint32_t>(num_point);
    header[1] = static_cast<std::uint32_t>(num_spot);

    auto *ptrs = reinterpret_cast<GpuPointLight *>(data + k_header_size);
    for (std::size_t i = 0; i < num_point; ++i) {
      ptrs[i].pos_range[0] = point_lights[i].position.x;
      ptrs[i].pos_range[1] = point_lights[i].position.y;
      ptrs[i].pos_range[2] = point_lights[i].position.z;
      ptrs[i].pos_range[3] = point_lights[i].range;
      ptrs[i].color_intensity[0] = point_lights[i].color.r;
      ptrs[i].color_intensity[1] = point_lights[i].color.g;
      ptrs[i].color_intensity[2] = point_lights[i].color.b;
      ptrs[i].color_intensity[3] = point_lights[i].intensity;

      // The slot the shadow pass rendered this light into, not its index in the
      // scene. A light that was skipped this frame reports no slot and is shaded
      // unshadowed -- which is correct, because it was only skipped when nothing
      // it lights is on screen.
      const std::int32_t slot = i < shadows.point_slots.size()
          ? shadows.point_slots[i]
          : k_no_shadow_slot;
      ptrs[i].shadow_params[0] = static_cast<float>(slot);
      ptrs[i].shadow_params[1] = 0.0F;
      ptrs[i].shadow_params[2] = 0.0F;
      ptrs[i].shadow_params[3] = 0.0F;
    }

    auto *sptr = reinterpret_cast<GpuSpotLight *>(data + k_header_size + num_point * sizeof(GpuPointLight));
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
      sptr[i].angles[2] = 0.0f;
      sptr[i].angles[3] = 0.0f;

      // The slot the shadow pass actually rendered this light into -- not its
      // index in the scene array. Point lights were fixed to work this way
      // already (see shadow_assignment.hpp, which explains why); spot lights
      // still read their own scene index and divided the atlas by the total
      // light count, so a light that casts no shadow shrank everyone else's map
      // and a culled light pointed at a tile nobody had drawn.
      const std::int32_t slot =
          i < shadows.spot_slots.size() ? shadows.spot_slots[i] : k_no_shadow_slot;

      if (slot != k_no_shadow_slot) {
        const glm::mat4 sm = glm::transpose(compute_shadow_matrix(spot_lights[i]));
        std::memcpy(sptr[i].shadow_matrix, &sm[0][0], sizeof(sm));
        const SpotAtlasTile tile =
            spot_atlas_tile(static_cast<std::uint32_t>(slot), spot_shadow_capacity);
        sptr[i].atlas_rect[0] = tile.u;
        sptr[i].atlas_rect[1] = tile.v;
        sptr[i].atlas_rect[2] = tile.width;
        sptr[i].atlas_rect[3] = tile.height;
      } else {
        std::memset(sptr[i].shadow_matrix, 0, sizeof(sptr[i].shadow_matrix));
        std::memset(sptr[i].atlas_rect, 0, sizeof(sptr[i].atlas_rect));
      }
    }
  }

  [[nodiscard]] auto buffer_ptr(std::uint32_t frame_index) const -> vk::Buffer {
    return frame_index < k_frames_in_flight && buffers_[frame_index] ? **buffers_[frame_index] : vk::Buffer{};
  }

private:
  std::array<std::optional<vk::raii::Buffer>, k_frames_in_flight> buffers_{};
  std::array<std::optional<vk::raii::DeviceMemory>, k_frames_in_flight> memories_{};
  std::array<std::uint8_t *, k_frames_in_flight> mapped_{};
  std::size_t max_lights_{0};
};

} // namespace engine
