#pragma once

#include "renderer/buffer.hpp"
#include "renderer/image_barrier.hpp"

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace engine {

namespace detail {

inline void execute_one_time_commands(
    vk::raii::Device &device,
    vk::raii::CommandPool &command_pool,
    vk::raii::Queue &queue,
    const std::function<void(vk::raii::CommandBuffer &)> &record) {
  const vk::CommandBufferAllocateInfo allocate_info{
      .commandPool = *command_pool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1,
  };

  vk::raii::CommandBuffer command_buffer = std::move(
      vk::raii::CommandBuffers(device, allocate_info).front());

  command_buffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  record(command_buffer);
  command_buffer.end();

  const vk::CommandBufferSubmitInfo command_buffer_info{.commandBuffer = *command_buffer};
  const vk::SubmitInfo2 submit_info{
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &command_buffer_info,
  };

  queue.submit2(submit_info);
  queue.waitIdle();
}

[[nodiscard]] inline auto mip_level_count(std::int32_t width, std::int32_t height) -> std::uint32_t {
  return static_cast<std::uint32_t>(std::floor(std::log2(static_cast<float>(std::max(width, height))))) + 1U;
}

inline void assert_blit_support(const vk::raii::PhysicalDevice &physical_device, vk::Format format) {
  const vk::FormatProperties format_properties = physical_device.getFormatProperties(format);
  const auto features = format_properties.optimalTilingFeatures;

  if ((features & vk::FormatFeatureFlagBits::eBlitSrc) == vk::FormatFeatureFlags{} ||
      (features & vk::FormatFeatureFlagBits::eBlitDst) == vk::FormatFeatureFlags{})
    throw std::runtime_error("Texture format does not support linear blitting for mip generation");
}

struct RgbaImageData {
  std::vector<stbi_uc> pixels;
  std::int32_t width{};
  std::int32_t height{};
};

[[nodiscard]] inline auto load_rgba_image(std::string_view path) -> RgbaImageData {
  std::int32_t width{};
  std::int32_t height{};
  std::int32_t channels{};
  stbi_uc *pixels = stbi_load(std::string(path).c_str(), &width, &height, &channels, STBI_rgb_alpha);
  if (pixels == nullptr)
    throw std::runtime_error(std::string("Failed to load texture: ") + std::string(path));

  RgbaImageData data{
      .pixels = {pixels, pixels + static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4},
      .width = width,
      .height = height,
  };
  stbi_image_free(pixels);
  return data;
}

inline void write_rgba_png(std::string_view path, const stbi_uc *pixels, std::int32_t width, std::int32_t height) {
  if (pixels == nullptr || width <= 0 || height <= 0)
    throw std::runtime_error("Cannot write empty image to PNG");

  if (stbi_write_png(
          std::string(path).c_str(),
          width,
          height,
          4,
          pixels,
          width * 4) == 0)
    throw std::runtime_error(std::string("Failed to write PNG: ") + std::string(path));
}

} // namespace detail

class TextureImage {
public:
  void load_from_file(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      vk::raii::CommandPool &command_pool,
      vk::raii::Queue &queue,
      std::string_view path) {
    physical_device_ = &physical_device;
    device_ = &device;
    format_ = vk::Format::eR8G8B8A8Srgb;
    detail::assert_blit_support(physical_device, format_);

    const detail::RgbaImageData image = detail::load_rgba_image(path);
    const std::int32_t width = image.width;
    const std::int32_t height = image.height;

    mip_levels_ = detail::mip_level_count(width, height);

    const vk::DeviceSize image_size = static_cast<vk::DeviceSize>(width) * static_cast<vk::DeviceSize>(height) * 4;

    const vk::BufferCreateInfo staging_info{
        .size = image_size,
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

    void *mapped = staging_memory.mapMemory(0, image_size);
    std::memcpy(mapped, image.pixels.data(), static_cast<std::size_t>(image_size));
    staging_memory.unmapMemory();

    extent_ = vk::Extent3D{
        .width = static_cast<std::uint32_t>(width),
        .height = static_cast<std::uint32_t>(height),
        .depth = 1,
    };

    const vk::ImageCreateInfo image_info{
        .imageType = vk::ImageType::e2D,
        .format = format_,
        .extent = extent_,
        .mipLevels = mip_levels_,
        .arrayLayers = 1,
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
          mip_levels_);

      const vk::BufferImageCopy region{
          .bufferOffset = 0,
          .imageSubresource = {
              .aspectMask = vk::ImageAspectFlagBits::eColor,
              .mipLevel = 0,
              .baseArrayLayer = 0,
              .layerCount = 1,
          },
          .imageExtent = extent_,
      };
      command_buffer.copyBufferToImage(*staging_buffer, *image_, vk::ImageLayout::eTransferDstOptimal, region);

      generate_mipmaps(command_buffer, width, height);
    });

    view_ = vk::raii::ImageView(
        device,
        vk::ImageViewCreateInfo{
            .image = *image_,
            .viewType = vk::ImageViewType::e2D,
            .format = format_,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = mip_levels_,
                .baseArrayLayer = 0,
                .layerCount = 1,
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

  [[nodiscard]] auto format() const -> vk::Format {
    return format_;
  }

private:
  void generate_mipmaps(vk::raii::CommandBuffer &command_buffer, std::int32_t width, std::int32_t height) {
    std::int32_t mip_width = width;
    std::int32_t mip_height = height;

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
          1);

      const vk::ImageBlit blit{
          .srcSubresource = {
              .aspectMask = vk::ImageAspectFlagBits::eColor,
              .mipLevel = level - 1,
              .baseArrayLayer = 0,
              .layerCount = 1,
          },
          .srcOffsets = std::array{
              vk::Offset3D{0, 0, 0},
              vk::Offset3D{mip_width, mip_height, 1},
          },
          .dstSubresource = {
              .aspectMask = vk::ImageAspectFlagBits::eColor,
              .mipLevel = level,
              .baseArrayLayer = 0,
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
        1);
  }

  void create_sampler(const vk::raii::PhysicalDevice &physical_device) {
    const vk::PhysicalDeviceProperties properties = physical_device.getProperties();
    const vk::PhysicalDeviceFeatures features = physical_device.getFeatures();
    const bool anisotropy_supported = features.samplerAnisotropy == vk::True;

    sampler_ = vk::raii::Sampler(
        *device_,
        vk::SamplerCreateInfo{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eRepeat,
            .anisotropyEnable = anisotropy_supported ? vk::True : vk::False,
            .maxAnisotropy = anisotropy_supported ? properties.limits.maxSamplerAnisotropy : 1.0F,
            .maxLod = static_cast<float>(mip_levels_),
        });
  }

  const vk::raii::PhysicalDevice *physical_device_{nullptr};
  vk::raii::Device *device_{nullptr};
  vk::Format format_{vk::Format::eUndefined};
  vk::Extent3D extent_{};
  std::uint32_t mip_levels_{1};
  vk::raii::Image image_{nullptr};
  vk::raii::DeviceMemory memory_{nullptr};
  vk::raii::ImageView view_{nullptr};
  vk::raii::Sampler sampler_{nullptr};
};

} // namespace engine
