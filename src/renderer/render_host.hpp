#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>

namespace engine {

// Read-only GPU/swapchain handles for app-side tools (e.g. ImGui) without linking them into the engine.
struct RenderHostInfo {
  vk::Instance instance{};
  vk::PhysicalDevice physical_device{};
  vk::Device device{};
  vk::Queue graphics_queue{};
  std::uint32_t graphics_queue_family_index{};
  vk::PipelineCache pipeline_cache{};
  vk::Format swapchain_color_format{};
  vk::Extent2D swapchain_extent{};
  std::uint32_t swapchain_image_count{};
  std::uint32_t frames_in_flight{2};
};

} // namespace engine
