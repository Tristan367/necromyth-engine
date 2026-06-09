#pragma once

#include "renderer/mesh_gpu.hpp"
#include "renderer/texture_array.hpp"
#include "renderer/texture_table.hpp"
#include "scene/scene.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <stdexcept>
#include <vector>

namespace engine {

inline void upload_scene_meshes(
    const Scene &scene,
    const vk::raii::PhysicalDevice &physical_device,
    vk::raii::Device &device,
    vk::raii::CommandPool &command_pool,
    vk::raii::Queue &graphics_queue,
    std::vector<MeshGpu> &out) {
  out.clear();
  out.reserve(scene.meshes().size());
  for (const MeshSource &mesh : scene.meshes()) {
    MeshGpu gpu{};
    gpu.upload(physical_device, device, command_pool, graphics_queue, mesh);
    out.push_back(std::move(gpu));
  }
}

inline void load_scene_textures(
    const Scene &scene,
    const vk::raii::PhysicalDevice &physical_device,
    vk::raii::Device &device,
    vk::raii::CommandPool &command_pool,
    vk::raii::Queue &graphics_queue,
    TextureTable &texture_table) {
  if (scene.texture_paths().empty())
    throw std::runtime_error("Scene must provide at least one texture path");

  texture_table.load_from_paths(
      physical_device,
      device,
      command_pool,
      graphics_queue,
      scene.texture_paths());
}

inline void load_texture_array_layers(
    const Scene &scene,
    const vk::raii::PhysicalDevice &physical_device,
    vk::raii::Device &device,
    vk::raii::CommandPool &command_pool,
    vk::raii::Queue &graphics_queue,
    TextureArray &texture_array) {
  if (scene.texture_array_layer_paths().empty())
    throw std::runtime_error("Scene must provide at least one texture array layer path");

  texture_array.load_from_files(
      physical_device,
      device,
      command_pool,
      graphics_queue,
      scene.texture_array_layer_paths());
}

} // namespace engine
