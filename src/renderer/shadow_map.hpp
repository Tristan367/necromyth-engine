#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <stdexcept>
#include <vector>

namespace engine {

class ShadowMap {
public:
  static constexpr std::uint32_t k_default_size = 2048;

  void create(const vk::raii::PhysicalDevice &physical_device, vk::raii::Device &device) {
    physical_device_ = &physical_device;
    device_ = &device;
    format_ = find_shadow_format(physical_device);
    create_resources();
  }

  [[nodiscard]] auto format() const -> vk::Format {
    return format_;
  }

  [[nodiscard]] auto image() const -> vk::Image {
    return *image_;
  }

  [[nodiscard]] auto view() const -> vk::ImageView {
    return *view_;
  }

  [[nodiscard]] auto sampler() const -> vk::Sampler {
    return *sampler_;
  }

  [[nodiscard]] auto extent() const -> vk::Extent2D {
    return {k_default_size, k_default_size};
  }

  [[nodiscard]] auto aspect_mask() const -> vk::ImageAspectFlags {
    return vk::ImageAspectFlagBits::eDepth;
  }

  [[nodiscard]] auto descriptor_image_info() const -> vk::DescriptorImageInfo {
    return {
        .sampler = *sampler_,
        .imageView = *view_,
        .imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal,
    };
  }

private:
  [[nodiscard]] static auto find_shadow_format(const vk::raii::PhysicalDevice &physical_device) -> vk::Format {
    const std::array candidates{
        vk::Format::eD16Unorm,
        vk::Format::eD32Sfloat,
        vk::Format::eD32SfloatS8Uint,
    };

    for (const vk::Format format : candidates) {
      const vk::FormatProperties properties = physical_device.getFormatProperties(format);
      const auto features = properties.optimalTilingFeatures;
      if ((features & vk::FormatFeatureFlagBits::eDepthStencilAttachment) != vk::FormatFeatureFlags{} &&
          (features & vk::FormatFeatureFlagBits::eSampledImage) != vk::FormatFeatureFlags{})
        return format;
    }

    throw std::runtime_error("No sampleable depth format found for shadow map");
  }

  [[nodiscard]] static auto format_is_filterable(
      const vk::raii::PhysicalDevice &physical_device,
      vk::Format format) -> bool {
    const vk::FormatProperties properties = physical_device.getFormatProperties(format);
    return (properties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear) !=
           vk::FormatFeatureFlags{};
  }

  void create_resources() {
    const vk::ImageCreateInfo image_info{
        .imageType = vk::ImageType::e2D,
        .format = format_,
        .extent = {k_default_size, k_default_size, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    image_ = vk::raii::Image(*device_, image_info);

    const vk::MemoryRequirements memory_requirements = image_.getMemoryRequirements();
    memory_ = vk::raii::DeviceMemory(
        *device_,
        vk::MemoryAllocateInfo{
            .allocationSize = memory_requirements.size,
            .memoryTypeIndex = find_memory_type(
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
                .aspectMask = vk::ImageAspectFlagBits::eDepth,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        });

    const vk::Filter filter = format_is_filterable(*physical_device_, format_) ? vk::Filter::eLinear : vk::Filter::eNearest;
    sampler_ = vk::raii::Sampler(
        *device_,
        vk::SamplerCreateInfo{
            .magFilter = filter,
            .minFilter = filter,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .anisotropyEnable = vk::False,
            .maxAnisotropy = 1.0F,
            .compareEnable = vk::False,
            .minLod = 0.0F,
            .maxLod = 1.0F,
        });
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

    throw std::runtime_error("Failed to find suitable memory type for shadow map");
  }

  const vk::raii::PhysicalDevice *physical_device_{};
  vk::raii::Device *device_{};
  vk::Format format_{vk::Format::eUndefined};
  vk::raii::Image image_{nullptr};
  vk::raii::DeviceMemory memory_{nullptr};
  vk::raii::ImageView view_{nullptr};
  vk::raii::Sampler sampler_{nullptr};
};

} // namespace engine
