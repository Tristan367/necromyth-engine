#pragma once

#include "renderer/texture_image.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace engine {

class TextureTable {
public:
  void load_from_paths(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      vk::raii::CommandPool &command_pool,
      vk::raii::Queue &queue,
      std::span<const std::string> paths) {
    textures_.clear();
    textures_.reserve(paths.size());

    for (const std::string &path : paths) {
      TextureImage texture{};
      texture.load_from_file(physical_device, device, command_pool, queue, path);
      textures_.push_back(std::move(texture));
    }
  }

  void append_from_file(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      vk::raii::CommandPool &command_pool,
      vk::raii::Queue &queue,
      const std::string &path) {
    TextureImage texture{};
    texture.load_from_file(physical_device, device, command_pool, queue, path);
    textures_.push_back(std::move(texture));
  }

  [[nodiscard]] auto count() const -> std::uint32_t {
    return static_cast<std::uint32_t>(textures_.size());
  }

  [[nodiscard]] auto texture(std::uint32_t index) const -> const TextureImage & {
    return textures_.at(index);
  }

  [[nodiscard]] auto texture_pointers() const -> std::vector<const TextureImage *> {
    std::vector<const TextureImage *> pointers;
    pointers.reserve(textures_.size());
    for (const TextureImage &texture : textures_)
      pointers.push_back(&texture);
    return pointers;
  }

private:
  std::vector<TextureImage> textures_;
};

} // namespace engine
