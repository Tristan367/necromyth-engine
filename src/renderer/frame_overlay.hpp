#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <functional>

namespace engine {

struct FrameOverlayContext {
  vk::raii::CommandBuffer &command_buffer;
  std::uint32_t frame_index{};
  std::uint32_t image_index{};
  vk::Extent2D extent{};
};

using FrameOverlayCallback = std::function<void(const FrameOverlayContext &)>;

} // namespace engine
