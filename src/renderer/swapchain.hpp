#pragma once

#include "renderer/vulkan_device.hpp"

#include <SDL3/SDL_video.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace engine {

namespace detail {

constexpr vk::PresentModeKHR desired_present_mode = vk::PresentModeKHR::eFifo;

} // namespace detail

class Swapchain {
public:
  void create(VulkanDevice &device, SDL_Window *window) {
    device_ = &device;
    window_ = window;
    create_swapchain(nullptr);
    create_image_views();
  }

  void recreate() {
    swapchain_image_views_.clear();
    const vk::SwapchainKHR old_swapchain = *swapchain_;
    create_swapchain(old_swapchain);
    create_image_views();
  }

  [[nodiscard]] auto handle() const -> const vk::raii::SwapchainKHR & {
    return swapchain_;
  }

  [[nodiscard]] auto images() const -> const std::vector<vk::Image> & {
    return swapchain_images_;
  }

  [[nodiscard]] auto image_views() const -> const std::vector<vk::raii::ImageView> & {
    return swapchain_image_views_;
  }

  [[nodiscard]] auto image_format() const -> vk::Format {
    return swapchain_image_format_;
  }

  [[nodiscard]] auto extent() const -> vk::Extent2D {
    return swapchain_extent_;
  }

  [[nodiscard]] auto image_count() const -> std::size_t {
    return swapchain_images_.size();
  }

private:
  [[nodiscard]] static auto choose_min_image_count(const vk::SurfaceCapabilitiesKHR &capabilities) -> std::uint32_t {
    auto image_count = std::max(3U, capabilities.minImageCount);
    if (capabilities.maxImageCount > 0)
      image_count = std::min(image_count, capabilities.maxImageCount);
    return image_count;
  }

  [[nodiscard]] static auto choose_surface_format(const std::vector<vk::SurfaceFormatKHR> &formats)
      -> vk::SurfaceFormatKHR {
    const auto preferred = std::ranges::find_if(formats, [](const vk::SurfaceFormatKHR &format) {
      return format.format == vk::Format::eB8G8R8A8Srgb &&
             format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
    });

    return preferred != formats.end() ? *preferred : formats.front();
  }

  [[nodiscard]] static auto choose_present_mode(const std::vector<vk::PresentModeKHR> &modes) -> vk::PresentModeKHR {
    if (std::ranges::find(modes, detail::desired_present_mode) != modes.end())
      return detail::desired_present_mode;

    return vk::PresentModeKHR::eFifo;
  }

  [[nodiscard]] auto choose_extent(const vk::SurfaceCapabilitiesKHR &capabilities) const -> vk::Extent2D {
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
      return capabilities.currentExtent;

    int width{};
    int height{};
    if (!SDL_GetWindowSizeInPixels(window_, &width, &height))
      throw std::runtime_error(std::string("Failed to get SDL window pixel size: ") + SDL_GetError());

    return {
        .width = std::clamp(
            static_cast<std::uint32_t>(std::max(width, 1)),
            capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width),
        .height = std::clamp(
            static_cast<std::uint32_t>(std::max(height, 1)),
            capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height),
    };
  }

  void create_swapchain(vk::SwapchainKHR old_swapchain) {
    const vk::SurfaceCapabilitiesKHR capabilities =
        device_->physical_device().getSurfaceCapabilitiesKHR(*device_->surface());
    const std::vector<vk::SurfaceFormatKHR> formats =
        device_->physical_device().getSurfaceFormatsKHR(*device_->surface());
    const std::vector<vk::PresentModeKHR> present_modes =
        device_->physical_device().getSurfacePresentModesKHR(*device_->surface());

    const vk::SurfaceFormatKHR surface_format = choose_surface_format(formats);
    swapchain_image_format_ = surface_format.format;
    swapchain_extent_ = choose_extent(capabilities);

    const std::uint32_t image_count = choose_min_image_count(capabilities);
    const auto &queue_families = device_->queue_families();
    const std::array queue_family_indices{queue_families.graphics, queue_families.present};
    const bool separate_present_queue = queue_families.graphics != queue_families.present;

    const vk::SwapchainCreateInfoKHR create_info{
        .surface = *device_->surface(),
        .minImageCount = image_count,
        .imageFormat = swapchain_image_format_,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = swapchain_extent_,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = separate_present_queue ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = separate_present_queue ? static_cast<std::uint32_t>(queue_family_indices.size()) : 0,
        .pQueueFamilyIndices = separate_present_queue ? queue_family_indices.data() : nullptr,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = choose_present_mode(present_modes),
        .clipped = vk::True,
        .oldSwapchain = old_swapchain,
    };

    swapchain_ = vk::raii::SwapchainKHR(device_->device(), create_info);
    swapchain_images_ = swapchain_.getImages();
  }

  void create_image_views() {
    swapchain_image_views_.clear();
    swapchain_image_views_.reserve(swapchain_images_.size());

    vk::ImageViewCreateInfo create_info{
        .viewType = vk::ImageViewType::e2D,
        .format = swapchain_image_format_,
        .components = {},
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    for (const vk::Image image : swapchain_images_) {
      create_info.image = image;
      swapchain_image_views_.emplace_back(device_->device(), create_info);
    }
  }

  VulkanDevice *device_{nullptr};
  SDL_Window *window_{nullptr};
  vk::raii::SwapchainKHR swapchain_{nullptr};
  vk::Format swapchain_image_format_{vk::Format::eUndefined};
  vk::Extent2D swapchain_extent_{};
  std::vector<vk::Image> swapchain_images_;
  std::vector<vk::raii::ImageView> swapchain_image_views_;
};

} // namespace engine
