#pragma once

#include "renderer/mesh_gpu.hpp"
#include "renderer/texture_array.hpp"
#include "renderer/texture_table.hpp"
#include "scene/scene.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <stdexcept>
#include <vector>

namespace engine {

// GPU side of one Scene mesh slot. `revision` is the Scene revision this was
// uploaded from; when they differ the slot needs re-uploading. Keeping the
// revision alongside the resource (rather than in a parallel array) means the
// two cannot drift out of step.
struct MeshGpuSlot {
  MeshGpu gpu;
  std::uint32_t revision{0}; // 0 = nothing uploaded
  bool alive{false};
};

// Brings the GPU mesh slots in line with the Scene's, uploading created and
// updated slots and retiring removed ones.
//
// Retired resources go to `retire` rather than being destroyed here: an
// in-flight command buffer may still reference them. That is what lets a
// streaming world remesh continuously without a device stall.
//
// Returns the number of slots that changed.
template <typename RetireFn>
auto sync_scene_meshes(
    const Scene &scene,
    const vk::raii::PhysicalDevice &physical_device,
    vk::raii::Device &device,
    vk::raii::CommandPool &command_pool,
    vk::raii::Queue &graphics_queue,
    std::vector<MeshGpuSlot> &slots,
    RetireFn &&retire) -> std::size_t {
  if (slots.size() < scene.meshes().size())
    slots.resize(scene.meshes().size());

  std::size_t changed = 0;
  for (std::size_t i = 0; i < scene.meshes().size(); ++i) {
    const MeshSlot &source = scene.meshes()[i];
    MeshGpuSlot &slot = slots[i];
    if (slot.revision == source.revision)
      continue;

    if (slot.revision != 0)
      retire(std::move(slot.gpu));
    slot.gpu = MeshGpu{};

    if (source.alive) {
      slot.gpu.upload(physical_device, device, command_pool, graphics_queue,
                      source.source, source.bounds);
    }
    slot.revision = source.revision;
    slot.alive = source.alive;
    ++changed;
  }
  return changed;
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
