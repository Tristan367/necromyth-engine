#pragma once

#include "renderer/buffer.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine {

class DepthImage {
public:
  void create(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      vk::Extent2D extent,
      vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1) {
    physical_device_ = &physical_device;
    device_ = &device;
    format_ = find_depth_format(physical_device);
    samples_ = samples;
    create_resources(extent);
  }

  void recreate(vk::Extent2D extent) {
    image_ = nullptr;
    memory_ = nullptr;
    view_ = nullptr;
    create_resources(extent);
  }

  [[nodiscard]] auto format() const -> vk::Format {
    return format_;
  }

  [[nodiscard]] auto image() const -> vk::Image {
    return *image_;
  }

  [[nodiscard]] auto view() const -> const vk::raii::ImageView & {
    return view_;
  }

  [[nodiscard]] auto aspect_mask() const -> vk::ImageAspectFlags {
    return aspect_mask_;
  }

private:
  void create_resources(vk::Extent2D extent) {
    const vk::ImageCreateInfo image_info{
        .imageType = vk::ImageType::e2D,
        .format = format_,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = samples_,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    image_ = vk::raii::Image(*device_, image_info);

    const vk::MemoryRequirements memory_requirements = image_.getMemoryRequirements();
    const vk::MemoryAllocateInfo allocate_info{
        .allocationSize = memory_requirements.size,
        .memoryTypeIndex = detail::find_memory_type(
            physical_device_->getMemoryProperties(),
            memory_requirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eDeviceLocal),
    };

    memory_ = vk::raii::DeviceMemory(*device_, allocate_info);
    image_.bindMemory(*memory_, 0);

    aspect_mask_ = vk::ImageAspectFlagBits::eDepth;
    if (format_ == vk::Format::eD16UnormS8Uint ||
        format_ == vk::Format::eD24UnormS8Uint ||
        format_ == vk::Format::eD32SfloatS8Uint)
      aspect_mask_ |= vk::ImageAspectFlagBits::eStencil;

    const vk::ImageViewCreateInfo view_info{
        .image = *image_,
        .viewType = vk::ImageViewType::e2D,
        .format = format_,
        .subresourceRange = {
            .aspectMask = aspect_mask_,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    view_ = vk::raii::ImageView(*device_, view_info);
  }

  [[nodiscard]] static auto find_depth_format(const vk::raii::PhysicalDevice &physical_device) -> vk::Format {
    return find_supported_format(
        physical_device,
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment);
  }

  [[nodiscard]] static auto find_supported_format(
      const vk::raii::PhysicalDevice &physical_device,
      const std::vector<vk::Format> &candidates,
      vk::ImageTiling tiling,
      vk::FormatFeatureFlags features) -> vk::Format {
    for (const vk::Format format : candidates) {
      const vk::FormatProperties properties = physical_device.getFormatProperties(format);
      const vk::FormatFeatureFlags supported = tiling == vk::ImageTiling::eLinear
                                                   ? properties.linearTilingFeatures
                                                   : properties.optimalTilingFeatures;
      if ((supported & features) == features)
        return format;
    }

    throw std::runtime_error("No supported depth format found");
  }

  const vk::raii::PhysicalDevice *physical_device_{};
  vk::raii::Device *device_{};
  vk::Format format_{vk::Format::eUndefined};
  vk::SampleCountFlagBits samples_{vk::SampleCountFlagBits::e1};
  vk::ImageAspectFlags aspect_mask_{};
  vk::raii::Image image_{nullptr};
  vk::raii::DeviceMemory memory_{nullptr};
  vk::raii::ImageView view_{nullptr};
};

} // namespace engine
