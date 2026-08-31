#pragma once

#include "renderer/device_allocator.hpp"
#include "renderer/mesh_gpu.hpp"
#include "renderer/texture_array.hpp"
#include "renderer/texture_table.hpp"
#include "scene/scene.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <chrono>
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
// `budget_ms` is how long this is allowed to spend uploading before the rest
// waits for the next frame. Ported from the C# reference implementation, which
// does exactly this at VoxelClient.cs:3225 and :6314 -- a Stopwatch, uploads
// while it reads under 2ms, then a yield of exactly one frame:
//
//     var sw = Stopwatch.StartNew();
//     foreach (var i in heightmapMeshesToUpdate) {
//         UpdateHeightmapMeshInstance(...);
//         if (sw.ElapsedMilliseconds > 2) {
//             await ROOT_NODE.ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
//             sw.Restart();
//         }
//     }
//
// A byte cap was tried here first and is the wrong instrument: it prices the
// work by how much memory moves, when what actually costs is the memory copy
// AND the vkCreateBuffer pair AND the suballocation AND whatever the driver
// decides to do -- on a machine whose speed at all of that is unknown at build
// time. The clock measures the real cost of the real work, and needs no
// tuning per machine. Zero means no budget.
//
// The check is AFTER the upload, not before, so a frame always makes at least
// one mesh of progress. A single mesh larger than the whole budget would
// otherwise be deferred forever and never drawn.
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
    std::size_t *out_pending = nullptr,
    float budget_ms = 0.0F,
    float *out_worst_mesh_ms = nullptr) -> std::size_t {
  if (slots.size() < scene.meshes().size())
    slots.resize(scene.meshes().size());

  const auto started = std::chrono::steady_clock::now();
  auto mesh_started = started;
  bool out_of_time = false;

  std::size_t changed = 0;
  std::size_t pending = 0;
  for (std::size_t i = 0; i < scene.meshes().size(); ++i) {
    const MeshSlot &source = scene.meshes()[i];
    MeshGpuSlot &slot = slots[i];
    if (slot.revision == source.revision)
      continue;
    ++pending; // alive in the Scene, not yet matching on the GPU

    // Out of time: keep scanning so `pending` still reports the whole backlog,
    // but upload nothing more this frame. The scan itself is a revision
    // compare, which is free next to what it is choosing not to do.
    if (out_of_time)
      continue;

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

    // The budget can only stop BETWEEN meshes, so one expensive mesh sets the
    // floor on how small a frame's upload can be made. Reporting the worst
    // single mesh is how you find out whether the budget is the limit or the
    // mesh is.
    const auto now = std::chrono::steady_clock::now();
    const float this_mesh_ms = std::chrono::duration<float, std::milli>(now - mesh_started).count();
    if (out_worst_mesh_ms != nullptr && this_mesh_ms > *out_worst_mesh_ms)
      *out_worst_mesh_ms = this_mesh_ms;
    mesh_started = now;

    if (budget_ms > 0.0F)
      out_of_time = std::chrono::duration<float, std::milli>(now - started).count() >= budget_ms;
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
