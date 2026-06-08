#pragma once

#include <vulkan/vulkan_raii.hpp>

namespace engine {

inline void transition_image_layout(
    vk::raii::CommandBuffer &command_buffer,
    vk::Image image,
    vk::ImageLayout old_layout,
    vk::ImageLayout new_layout,
    vk::AccessFlags2 src_access_mask,
    vk::AccessFlags2 dst_access_mask,
    vk::PipelineStageFlags2 src_stage_mask,
    vk::PipelineStageFlags2 dst_stage_mask,
    vk::ImageAspectFlags aspect_mask = vk::ImageAspectFlagBits::eColor,
    std::uint32_t base_mip_level = 0,
    std::uint32_t mip_level_count = 1,
    std::uint32_t base_array_layer = 0,
    std::uint32_t array_layer_count = 1) {
  const vk::ImageMemoryBarrier2 barrier{
      .srcStageMask = src_stage_mask,
      .srcAccessMask = src_access_mask,
      .dstStageMask = dst_stage_mask,
      .dstAccessMask = dst_access_mask,
      .oldLayout = old_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {
          .aspectMask = aspect_mask,
          .baseMipLevel = base_mip_level,
          .levelCount = mip_level_count,
          .baseArrayLayer = base_array_layer,
          .layerCount = array_layer_count,
      },
  };

  command_buffer.pipelineBarrier2({
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &barrier,
  });
}

} // namespace engine
