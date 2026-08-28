#pragma once

// How many frames the CPU may work on before it has to wait for the GPU.
//
// One number, in one place, because it was three numbers in three places and
// they disagreed: vulkan_context said 2, light_buffer said 2, ui_renderer said
// 3, and descriptors::update_light_buffers took a std::array<vk::Buffer, 2>
// that silently decided the question for everybody. Raising any one of them on
// its own is a bug rather than a change -- the light buffer would stop writing
// on the third slot and its descriptor would never be bound, so every third
// frame would sample lights that were never uploaded.
//
// Three, to match the swapchain, which asks for max(3, minImageCount) images.
// Two frames in flight against three images means the CPU blocks on the fence
// two frames back, and under FIFO a frame that misses its refresh deadline
// waits out a whole extra refresh -- a 16.6ms frame becomes 33.3ms. A real
// session logged 388 frames sitting at exactly 33ms with 0.4ms of CPU work in
// them, which is that mechanism and nothing else.

#include <cstdint>

namespace engine {

constexpr std::uint32_t k_frames_in_flight = 3;

} // namespace engine
