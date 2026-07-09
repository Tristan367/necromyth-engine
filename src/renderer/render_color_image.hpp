#pragma once
#include "renderer/buffer.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <stdexcept>

namespace engine {

// Single-sample color target for scaled rendering (resolve destination + blit source).
class RenderColorImage {
public:
  void create(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      vk::Extent2D extent,
      vk::Format format) {
    physical_device_ = &physical_device;
    device_ = &device;
    format_ = format;
    extent_ = extent;
    create_resources();
  }

  void recreate(vk::Extent2D extent) {
    extent_ = extent;
    image_ = nullptr;
    memory_ = nullptr;
    view_ = nullptr;
    create_resources();
  }

  [[nodiscard]] auto active() const -> bool {
    return static_cast<bool>(*image_);
  }

  [[nodiscard]] auto image() const -> vk::Image {
    return *image_;
  }

  [[nodiscard]] auto view() const -> const vk::raii::ImageView & {
    return view_;
  }

  [[nodiscard]] auto extent() const -> vk::Extent2D {
    return extent_;
  }

private:
  void create_resources() {
    if (extent_.width == 0 || extent_.height == 0)
      throw std::runtime_error("Render color image extent must be non-zero");

    const vk::ImageCreateInfo image_info{
        .imageType = vk::ImageType::e2D,
        .format = format_,
        .extent = {extent_.width, extent_.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    image_ = vk::raii::Image(*device_, image_info);

    const vk::MemoryRequirements memory_requirements = image_.getMemoryRequirements();
    memory_ = vk::raii::DeviceMemory(
        *device_,
        vk::MemoryAllocateInfo{
            .allocationSize = memory_requirements.size,
            .memoryTypeIndex = detail::find_memory_type(
                physical_device_->getMemoryProperties(),
                memory_requirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal),
        });
    image_.bindMemory(*memory_, 0);

    view_ = vk::raii::ImageView(
        *device_,
        vk::ImageViewCreateInfo{
            .image = *image_,
            .viewType = vk::ImageViewType::e2D,
            .format = format_,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        });
  }

  const vk::raii::PhysicalDevice *physical_device_{};
  vk::raii::Device *device_{};
  vk::Format format_{vk::Format::eUndefined};
  vk::Extent2D extent_{};
  vk::raii::Image image_{nullptr};
  vk::raii::DeviceMemory memory_{nullptr};
  vk::raii::ImageView view_{nullptr};
};

} // namespace engine
