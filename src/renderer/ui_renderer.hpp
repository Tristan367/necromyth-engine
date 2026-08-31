#pragma once

#include "renderer/frames_in_flight.hpp"

#include "renderer/device_memory.hpp"
#include "renderer/image_barrier.hpp"
#include "renderer/texture_image.hpp"
#include "renderer/ui_draw_list.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>

namespace engine {

// The GPU side of the game's own HUD.
//
// One instanced draw for the whole thing: every rectangle and every glyph is a
// quad in one storage buffer, sampling one atlas, through one pipeline. That is
// not an optimisation reached for early -- it falls out of the atlas carrying a
// solid white texel alongside the glyphs, so an untextured panel and a letter
// are the same kind of thing and neither needs its own pass.
//
// Why not keep using the debug GUI: it is a debug GUI. It draws at whatever size
// the window happens to be, in a font designed for tool panels, and a survival
// game's HUD is not a tool panel. The requirement was "our own UI, pixel font,
// integer scale", and every one of those three words is a thing ImGui will not
// do.
class UiRenderer {
public:
  static constexpr std::size_t k_frames_in_flight = engine::k_frames_in_flight;
  // Room for a HUD, an inventory grid and a crafting list at once. A quad is 48
  // bytes, so this is 384 KB per frame in flight -- host-visible and written
  // straight through, which at a few hundred quads a frame is a memcpy nobody
  // will ever see in a profile.
  static constexpr std::size_t k_max_quads = 8192;

  void create(const vk::raii::PhysicalDevice &physical_device, vk::raii::Device &device,
              vk::raii::CommandPool &command_pool, vk::raii::Queue &queue) {
    create_atlas(physical_device, device, command_pool, queue);

    const auto memory_properties = physical_device.getMemoryProperties();
    const vk::DeviceSize bytes = k_max_quads * sizeof(ui::Quad);
    for (std::size_t i = 0; i < k_frames_in_flight; ++i) {
      buffers_[i].emplace(device, vk::BufferCreateInfo{
                                      .size = bytes,
                                      .usage = vk::BufferUsageFlagBits::eStorageBuffer,
                                      .sharingMode = vk::SharingMode::eExclusive,
                                  });
      const vk::MemoryRequirements requirements = buffers_[i]->getMemoryRequirements();
      memories_[i].emplace(
          device, vk::MemoryAllocateInfo{
                      .allocationSize = requirements.size,
                      .memoryTypeIndex = detail::find_memory_type(
                          memory_properties, requirements.memoryTypeBits,
                          vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent),
                  });
      buffers_[i]->bindMemory(**memories_[i], 0);
      mapped_[i] = static_cast<std::uint8_t *>(memories_[i]->mapMemory(0, bytes));
    }
  }

  // Copies this frame's quads into the frame's buffer. Returns how many there
  // are to draw.
  auto write(std::uint32_t frame_index, const ui::DrawList &list) -> std::uint32_t {
    if (frame_index >= k_frames_in_flight || mapped_[frame_index] == nullptr)
      return 0;
    const std::size_t count = std::min(list.quads().size(), k_max_quads);
    if (count > 0)
      std::memcpy(mapped_[frame_index], list.quads().data(), count * sizeof(ui::Quad));
    // Silently dropping the tail is the right failure here: a HUD that is one
    // quad over budget should lose its last label, not stop drawing.
    return static_cast<std::uint32_t>(count);
  }

  [[nodiscard]] auto buffer(std::uint32_t frame_index) const -> vk::Buffer {
    return frame_index < k_frames_in_flight && buffers_[frame_index].has_value()
               ? **buffers_[frame_index]
               : vk::Buffer{};
  }
  [[nodiscard]] auto buffer_bytes() const -> vk::DeviceSize {
    return k_max_quads * sizeof(ui::Quad);
  }
  [[nodiscard]] auto atlas_view() const -> vk::ImageView { return *view_; }
  [[nodiscard]] auto atlas_sampler() const -> vk::Sampler { return *sampler_; }
  [[nodiscard]] auto ready() const -> bool { return *view_ != VK_NULL_HANDLE; }

private:
  void create_atlas(const vk::raii::PhysicalDevice &physical_device, vk::raii::Device &device,
                    vk::raii::CommandPool &command_pool, vk::raii::Queue &queue) {
    const std::vector<std::uint8_t> pixels = ui::build_atlas();
    const auto width = static_cast<std::uint32_t>(ui::k_atlas_width);
    const auto height = static_cast<std::uint32_t>(ui::k_atlas_height);
    const vk::DeviceSize bytes = pixels.size();
    const auto memory_properties = physical_device.getMemoryProperties();

    vk::raii::Buffer staging(device, vk::BufferCreateInfo{
                                         .size = bytes,
                                         .usage = vk::BufferUsageFlagBits::eTransferSrc,
                                         .sharingMode = vk::SharingMode::eExclusive,
                                     });
    const vk::MemoryRequirements staging_requirements = staging.getMemoryRequirements();
    vk::raii::DeviceMemory staging_memory(
        device, vk::MemoryAllocateInfo{
                    .allocationSize = staging_requirements.size,
                    .memoryTypeIndex = detail::find_memory_type(
                        memory_properties, staging_requirements.memoryTypeBits,
                        vk::MemoryPropertyFlagBits::eHostVisible |
                            vk::MemoryPropertyFlagBits::eHostCoherent),
                });
    staging.bindMemory(*staging_memory, 0);
    void *mapped = staging_memory.mapMemory(0, bytes);
    std::memcpy(mapped, pixels.data(), pixels.size());
    staging_memory.unmapMemory();

    // R8_UNORM: the glyph is coverage, and the colour is on the quad. Not sRGB
    // -- a mask is not a colour, and putting it through a transfer function
    // would thin every stroke.
    image_ = vk::raii::Image(device, vk::ImageCreateInfo{
                                         .imageType = vk::ImageType::e2D,
                                         .format = vk::Format::eR8Unorm,
                                         .extent = {width, height, 1},
                                         .mipLevels = 1,
                                         .arrayLayers = 1,
                                         .samples = vk::SampleCountFlagBits::e1,
                                         .tiling = vk::ImageTiling::eOptimal,
                                         .usage = vk::ImageUsageFlagBits::eTransferDst |
                                                  vk::ImageUsageFlagBits::eSampled,
                                         .sharingMode = vk::SharingMode::eExclusive,
                                     });
    const vk::MemoryRequirements image_requirements = image_.getMemoryRequirements();
    memory_ = vk::raii::DeviceMemory(
        device, vk::MemoryAllocateInfo{
                    .allocationSize = image_requirements.size,
                    .memoryTypeIndex = detail::find_memory_type(
                        memory_properties, image_requirements.memoryTypeBits,
                        vk::MemoryPropertyFlagBits::eDeviceLocal),
                });
    image_.bindMemory(*memory_, 0);

    detail::execute_one_time_commands(
        device, command_pool, queue, [&](vk::raii::CommandBuffer &command_buffer) {
          transition_image_layout(command_buffer, *image_, vk::ImageLayout::eUndefined,
                                  vk::ImageLayout::eTransferDstOptimal, {},
                                  vk::AccessFlagBits2::eTransferWrite,
                                  vk::PipelineStageFlagBits2::eTopOfPipe,
                                  vk::PipelineStageFlagBits2::eTransfer,
                                  vk::ImageAspectFlagBits::eColor, 0, 1);
          const vk::BufferImageCopy region{
              .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                   .mipLevel = 0,
                                   .baseArrayLayer = 0,
                                   .layerCount = 1},
              .imageExtent = {width, height, 1},
          };
          command_buffer.copyBufferToImage(*staging, *image_,
                                           vk::ImageLayout::eTransferDstOptimal, region);
          transition_image_layout(command_buffer, *image_, vk::ImageLayout::eTransferDstOptimal,
                                  vk::ImageLayout::eShaderReadOnlyOptimal,
                                  vk::AccessFlagBits2::eTransferWrite,
                                  vk::AccessFlagBits2::eShaderRead,
                                  vk::PipelineStageFlagBits2::eTransfer,
                                  vk::PipelineStageFlagBits2::eFragmentShader,
                                  vk::ImageAspectFlagBits::eColor, 0, 1);
        });

    view_ = vk::raii::ImageView(
        device, vk::ImageViewCreateInfo{
                    .image = *image_,
                    .viewType = vk::ImageViewType::e2D,
                    .format = vk::Format::eR8Unorm,
                    .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                         .baseMipLevel = 0,
                                         .levelCount = 1,
                                         .baseArrayLayer = 0,
                                         .layerCount = 1},
                });

    // NEAREST, and it has to be.
    //
    // The entire point of a pixel font at integer scale is that one texel
    // becomes an exact square of pixels. Linear filtering would round every edge
    // and give back exactly the soft, half-lit look the pixel font exists to
    // avoid -- and it would do it while looking like a subtle bug rather than a
    // decision.
    sampler_ = vk::raii::Sampler(device, vk::SamplerCreateInfo{
                                             .magFilter = vk::Filter::eNearest,
                                             .minFilter = vk::Filter::eNearest,
                                             .mipmapMode = vk::SamplerMipmapMode::eNearest,
                                             .addressModeU = vk::SamplerAddressMode::eClampToEdge,
                                             .addressModeV = vk::SamplerAddressMode::eClampToEdge,
                                             .addressModeW = vk::SamplerAddressMode::eClampToEdge,
                                         });
  }

  vk::raii::Image image_{nullptr};
  vk::raii::DeviceMemory memory_{nullptr};
  vk::raii::ImageView view_{nullptr};
  vk::raii::Sampler sampler_{nullptr};

  std::array<std::optional<vk::raii::Buffer>, k_frames_in_flight> buffers_{};
  std::array<std::optional<vk::raii::DeviceMemory>, k_frames_in_flight> memories_{};
  std::array<std::uint8_t *, k_frames_in_flight> mapped_{};
};

} // namespace engine
