#pragma once

#include "renderer/device_allocator.hpp"
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
    DeviceAllocator &allocator,
    vk::raii::Device &device,
    UploadQueue &uploads,
    std::vector<MeshGpuSlot> &slots,
    RetireFn &&retire,
    std::size_t *out_pending = nullptr) -> std::size_t {
  if (slots.size() < scene.meshes().size())
    slots.resize(scene.meshes().size());

  std::size_t changed = 0;
  std::size_t pending = 0;
  for (std::size_t i = 0; i < scene.meshes().size(); ++i) {
    const MeshSlot &source = scene.meshes()[i];
    MeshGpuSlot &slot = slots[i];
    if (slot.revision == source.revision)
      continue;
    ++pending; // alive in the Scene, not yet matching on the GPU

    if (source.alive) {
      // Upload into a fresh slot and only take it once it has succeeded.
      //
      // Retiring the live mesh first and then bailing out on a full staging
      // ring left the slot holding an empty MeshGpu while still marked alive,
      // so the rest of the frame drew an unbound vertex buffer -- stretched
      // black triangles wherever a section happened to lose the race. The old
      // mesh is perfectly good until the new one is ready; keep drawing it.
      MeshGpu fresh;
      if (!fresh.upload(
        allocator, device, uploads, source.source, source.bounds))
        continue; // retried next frame, still drawing the previous mesh
      if (slot.revision != 0)
        retire(std::move(slot.gpu));
      slot.gpu = std::move(fresh);
    } else {
      if (slot.revision != 0)
        retire(std::move(slot.gpu));
      slot.gpu = MeshGpu{};
    }

    slot.revision = source.revision;
    slot.alive = source.alive;
    ++changed;
  }
  if (out_pending != nullptr)
    *out_pending = pending;
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
