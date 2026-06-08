#pragma once

#include "renderer/render_settings.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <stdexcept>
#include <vector>

namespace engine {

class MsaaColorImage {
public:
  void create(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      vk::Extent2D extent,
      vk::Format format,
      vk::SampleCountFlagBits samples) {
    physical_device_ = &physical_device;
    device_ = &device;
    format_ = format;
    samples_ = samples;
    if (!msaa_is_active(samples_)) {
      clear_resources();
      return;
    }
    create_resources(extent);
  }

  void recreate(vk::Extent2D extent, vk::SampleCountFlagBits samples) {
    samples_ = samples;
    if (!msaa_is_active(samples_)) {
      clear_resources();
      return;
    }
    clear_resources();
    create_resources(extent);
  }

  [[nodiscard]] auto active() const -> bool {
    return msaa_is_active(samples_) && static_cast<bool>(*image_);
  }

  [[nodiscard]] auto image() const -> vk::Image {
    return *image_;
  }

  [[nodiscard]] auto view() const -> const vk::raii::ImageView & {
    return view_;
  }

private:
  void clear_resources() {
    image_ = nullptr;
    memory_ = nullptr;
    view_ = nullptr;
  }

  void create_resources(vk::Extent2D extent) {
    const vk::ImageCreateInfo image_info{
        .imageType = vk::ImageType::e2D,
        .format = format_,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = samples_,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    image_ = vk::raii::Image(*device_, image_info);

    const vk::MemoryRequirements memory_requirements = image_.getMemoryRequirements();
    const vk::MemoryAllocateInfo allocate_info{
        .allocationSize = memory_requirements.size,
        .memoryTypeIndex = find_memory_type(
            physical_device_->getMemoryProperties(),
            memory_requirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eDeviceLocal),
    };

    memory_ = vk::raii::DeviceMemory(*device_, allocate_info);
    image_.bindMemory(*memory_, 0);

    const vk::ImageViewCreateInfo view_info{
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
    };

    view_ = vk::raii::ImageView(*device_, view_info);
  }

  [[nodiscard]] static auto find_memory_type(
      vk::PhysicalDeviceMemoryProperties memory_properties,
      std::uint32_t type_filter,
      vk::MemoryPropertyFlags properties) -> std::uint32_t {
    for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
      if ((type_filter & (1U << i)) &&
          (memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
        return i;
    }

    throw std::runtime_error("Failed to find suitable memory type for MSAA color image");
  }

  const vk::raii::PhysicalDevice *physical_device_{};
  vk::raii::Device *device_{};
  vk::Format format_{vk::Format::eUndefined};
  vk::SampleCountFlagBits samples_{vk::SampleCountFlagBits::e1};
  vk::raii::Image image_{nullptr};
  vk::raii::DeviceMemory memory_{nullptr};
  vk::raii::ImageView view_{nullptr};
};

} // namespace engine
