#pragma once

#include "renderer/texture_image.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine {

class TextureArray {
public:
  void load_from_files(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      vk::raii::CommandPool &command_pool,
      vk::raii::Queue &queue,
      std::span<const std::string> paths) {
    if (paths.empty())
      throw std::runtime_error("Texture array requires at least one layer path");

    physical_device_ = &physical_device;
    device_ = &device;
    format_ = vk::Format::eR8G8B8A8Srgb;
    detail::assert_blit_support(physical_device, format_);

    std::vector<detail::RgbaImageData> layers;
    layers.reserve(paths.size());

    for (const std::string &path : paths) {
      detail::RgbaImageData layer = detail::load_rgba_image(path);
      if (layers.empty()) {
        extent_ = vk::Extent3D{
            .width = static_cast<std::uint32_t>(layer.width),
            .height = static_cast<std::uint32_t>(layer.height),
            .depth = 1,
        };
      } else if (static_cast<std::uint32_t>(layer.width) != extent_.width ||
                 static_cast<std::uint32_t>(layer.height) != extent_.height)
        throw std::runtime_error("Texture array layers must share dimensions: " + path);

      layers.push_back(std::move(layer));
    }

    layer_count_ = static_cast<std::uint32_t>(layers.size());
    mip_levels_ = detail::mip_level_count(static_cast<std::int32_t>(extent_.width), static_cast<std::int32_t>(extent_.height));

    const vk::DeviceSize layer_size = static_cast<vk::DeviceSize>(extent_.width) * extent_.height * 4;
    const vk::DeviceSize staging_size = layer_size * layer_count_;

    const vk::BufferCreateInfo staging_info{
        .size = staging_size,
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

    void *mapped = staging_memory.mapMemory(0, staging_size);
    for (std::uint32_t layer = 0; layer < layer_count_; ++layer)
      std::memcpy(
          static_cast<char *>(mapped) + static_cast<std::size_t>(layer) * static_cast<std::size_t>(layer_size),
          layers[layer].pixels.data(),
          static_cast<std::size_t>(layer_size));
    staging_memory.unmapMemory();

    const vk::ImageCreateInfo image_info{
        .imageType = vk::ImageType::e2D,
        .format = format_,
        .extent = extent_,
        .mipLevels = mip_levels_,
        .arrayLayers = layer_count_,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
                 vk::ImageUsageFlagBits::eSampled,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    image_ = vk::raii::Image(device, image_info);
    const vk::MemoryRequirements image_requirements = image_.getMemoryRequirements();
    memory_ = vk::raii::DeviceMemory{
        device,
        vk::MemoryAllocateInfo{
            .allocationSize = image_requirements.size,
            .memoryTypeIndex = detail::find_memory_type(
                memory_properties,
                image_requirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal),
        }};
    image_.bindMemory(*memory_, 0);

    detail::execute_one_time_commands(device, command_pool, queue, [&](vk::raii::CommandBuffer &command_buffer) {
      transition_image_layout(
          command_buffer,
          *image_,
          vk::ImageLayout::eUndefined,
          vk::ImageLayout::eTransferDstOptimal,
          {},
          vk::AccessFlagBits2::eTransferWrite,
          vk::PipelineStageFlagBits2::eTopOfPipe,
          vk::PipelineStageFlagBits2::eTransfer,
          vk::ImageAspectFlagBits::eColor,
          0,
          mip_levels_,
          0,
          layer_count_);

      for (std::uint32_t layer = 0; layer < layer_count_; ++layer) {
        const vk::BufferImageCopy region{
            .bufferOffset = layer_size * layer,
            .imageSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = layer,
                .layerCount = 1,
            },
            .imageExtent = extent_,
        };
        command_buffer.copyBufferToImage(*staging_buffer, *image_, vk::ImageLayout::eTransferDstOptimal, region);
      }

      for (std::uint32_t layer = 0; layer < layer_count_; ++layer)
        generate_mipmaps(command_buffer, layer);
    });

    view_ = vk::raii::ImageView(
        device,
        vk::ImageViewCreateInfo{
            .image = *image_,
            .viewType = vk::ImageViewType::e2DArray,
            .format = format_,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = mip_levels_,
                .baseArrayLayer = 0,
                .layerCount = layer_count_,
            },
        });

    create_sampler(physical_device);
  }

  [[nodiscard]] auto sampler() const -> vk::Sampler {
    return *sampler_;
  }

  [[nodiscard]] auto view() const -> vk::ImageView {
    return *view_;
  }

  [[nodiscard]] auto layer_count() const -> std::uint32_t {
    return layer_count_;
  }

private:
  void generate_mipmaps(vk::raii::CommandBuffer &command_buffer, std::uint32_t layer) {
    std::int32_t mip_width = static_cast<std::int32_t>(extent_.width);
    std::int32_t mip_height = static_cast<std::int32_t>(extent_.height);

    for (std::uint32_t level = 1; level < mip_levels_; ++level) {
      transition_image_layout(
          command_buffer,
          *image_,
          vk::ImageLayout::eTransferDstOptimal,
          vk::ImageLayout::eTransferSrcOptimal,
          vk::AccessFlagBits2::eTransferWrite,
          vk::AccessFlagBits2::eTransferRead,
          vk::PipelineStageFlagBits2::eTransfer,
          vk::PipelineStageFlagBits2::eTransfer,
          vk::ImageAspectFlagBits::eColor,
          level - 1,
          1,
          layer,
          1);

      const vk::ImageBlit blit{
          .srcSubresource = {
              .aspectMask = vk::ImageAspectFlagBits::eColor,
              .mipLevel = level - 1,
              .baseArrayLayer = layer,
              .layerCount = 1,
          },
          .srcOffsets = std::array{
              vk::Offset3D{0, 0, 0},
              vk::Offset3D{mip_width, mip_height, 1},
          },
          .dstSubresource = {
              .aspectMask = vk::ImageAspectFlagBits::eColor,
              .mipLevel = level,
              .baseArrayLayer = layer,
              .layerCount = 1,
          },
          .dstOffsets = std::array{
              vk::Offset3D{0, 0, 0},
              vk::Offset3D{std::max(mip_width / 2, 1), std::max(mip_height / 2, 1), 1},
          },
      };

      command_buffer.blitImage(
          *image_,
          vk::ImageLayout::eTransferSrcOptimal,
          *image_,
          vk::ImageLayout::eTransferDstOptimal,
          blit,
          vk::Filter::eLinear);

      transition_image_layout(
          command_buffer,
          *image_,
          vk::ImageLayout::eTransferSrcOptimal,
          vk::ImageLayout::eShaderReadOnlyOptimal,
          vk::AccessFlagBits2::eTransferRead,
          vk::AccessFlagBits2::eShaderRead,
          vk::PipelineStageFlagBits2::eTransfer,
          vk::PipelineStageFlagBits2::eFragmentShader,
          vk::ImageAspectFlagBits::eColor,
          level - 1,
          1,
          layer,
          1);

      if (mip_width > 1)
        mip_width /= 2;
      if (mip_height > 1)
        mip_height /= 2;
    }

    transition_image_layout(
        command_buffer,
        *image_,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eTransferWrite,
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor,
        mip_levels_ - 1,
        1,
        layer,
        1);
  }

  void create_sampler(const vk::raii::PhysicalDevice &physical_device) {
    sampler_ = detail::create_mipmapped_sampler(*device_, physical_device, mip_levels_);
  }

  const vk::raii::PhysicalDevice *physical_device_{nullptr};
  vk::raii::Device *device_{nullptr};
  vk::Format format_{vk::Format::eUndefined};
  vk::Extent3D extent_{};
  std::uint32_t layer_count_{};
  std::uint32_t mip_levels_{1};
  vk::raii::Image image_{nullptr};
  vk::raii::DeviceMemory memory_{nullptr};
  vk::raii::ImageView view_{nullptr};
  vk::raii::Sampler sampler_{nullptr};
};

} // namespace engine
