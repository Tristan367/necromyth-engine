#pragma once

#include "scene/animation_types.hpp"
#include "scene/texture_array_layer.hpp"
#include "scene/bounds.hpp"
#include "scene/camera.hpp"
#include "scene/directional_light.hpp"
#include "scene/shadow_utils.hpp"
#include "scene/mesh_instance.hpp"
#include "scene/mesh_source.hpp"
#include "scene/point_light.hpp"
#include "scene/spot_light.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <iostream>
#include <vector>

namespace engine {

// One mesh slot. `revision` is bumped on every content change (create, update,
// remove); the renderer stores the revision it last uploaded and re-uploads when
// they differ, which makes add/update/remove a single uniform code path.
struct MeshSlot {
  MeshSource source;
  AABB bounds{};
  std::uint32_t revision{0};
  bool alive{true};
};

class Scene {
public:
  [[nodiscard]] auto camera() -> Camera & {
    return camera_;
  }

  [[nodiscard]] auto camera() const -> const Camera & {
    return camera_;
  }

  [[nodiscard]] auto meshes() const -> const std::vector<MeshSlot> & {
    return meshes_;
  }

  [[nodiscard]] auto mesh_alive(std::uint32_t index) const -> bool {
    return index < meshes_.size() && meshes_[index].alive;
  }

  [[nodiscard]] auto mesh_bounds(std::uint32_t index) const -> const AABB & {
    static const AABB k_empty{};
    return index < meshes_.size() ? meshes_[index].bounds : k_empty;
  }

  [[nodiscard]] auto instances() const -> const std::vector<MeshInstance> & {
    return instances_;
  }

  [[nodiscard]] auto instances() -> std::vector<MeshInstance> & {
    return instances_;
  }

  [[nodiscard]] auto texture_paths() const -> const std::vector<std::string> & {
    return texture_paths_;
  }

  [[nodiscard]] auto texture_array_layer_paths() const -> const std::vector<TextureArrayLayer> & {
    return texture_array_layer_paths_;
  }

  // Throws on a stale handle. Use try_instance() where the entity may legitimately
  // have been removed (a projectile whose target despawned mid-flight).
  [[nodiscard]] auto instance(InstanceHandle handle) -> MeshInstance & {
    MeshInstance *found = try_instance(handle);
    if (found == nullptr)
      throw std::out_of_range("Scene::instance: stale or invalid InstanceHandle");
    return *found;
  }

  [[nodiscard]] auto try_instance(InstanceHandle handle) -> MeshInstance * {
    if (!handle.is_set() || handle.index >= instances_.size())
      return nullptr;
    MeshInstance &instance = instances_[handle.index];
    // Generation only. NOT `alive`.
    //
    // `alive` means "draw this", and a caller is entitled to set it false to
    // hide something and true again later. Refusing to resolve a hidden
    // instance made that impossible: the handle went dead the moment it was
    // hidden, so the slot could never be revived and whatever owned it silently
    // stopped being drawn -- forever.
    //
    // That is what made zombies invisible. The horde parks its surplus part
    // instances between frames by setting alive = false; the next busy night it
    // could not get them back, so only the first nine zombies were ever drawn
    // while the rest walked up and hit you unseen.
    //
    // Whether the slot is FREE is a different question, and remove_instance
    // bumps the generation so a handle to a released slot stops matching.
    if (instance.generation != handle.generation)
      return nullptr;
    return &instance;
  }

  [[nodiscard]] auto try_instance(InstanceHandle handle) const -> const MeshInstance * {
    return const_cast<Scene *>(this)->try_instance(handle);
  }

  [[nodiscard]] auto is_valid(InstanceHandle handle) const -> bool {
    return try_instance(handle) != nullptr;
  }

  [[nodiscard]] auto directional_light() -> DirectionalLight & {
    return directional_light_;
  }

  [[nodiscard]] auto directional_light() const -> const DirectionalLight & {
    return directional_light_;
  }

  [[nodiscard]] auto shadow_settings() -> DirectionalLightShadowSettings & {
    return shadow_settings_;
  }

  [[nodiscard]] auto shadow_settings() const -> const DirectionalLightShadowSettings & {
    return shadow_settings_;
  }

  [[nodiscard]] auto point_lights() const -> const std::vector<PointLight> & { return point_lights_; }
  [[nodiscard]] auto point_lights() -> std::vector<PointLight> & { return point_lights_; }
  [[nodiscard]] auto spot_lights() const -> const std::vector<SpotLight> & { return spot_lights_; }
  [[nodiscard]] auto spot_lights() -> std::vector<SpotLight> & { return spot_lights_; }

  [[nodiscard]] auto skeletons() const -> const std::vector<SkeletonAsset> & {
    return skeletons_;
  }
  [[nodiscard]] auto skeletons() -> std::vector<SkeletonAsset> & {
    return skeletons_;
  }

  [[nodiscard]] auto animations() const -> const std::vector<AnimationClip> & {
    return animations_;
  }
  [[nodiscard]] auto animations() -> std::vector<AnimationClip> & {
    return animations_;
  }

  [[nodiscard]] auto instance_count() const -> std::size_t { return instances_.size(); }
  // Slot capacity, including free slots -- not the number of live meshes.
  [[nodiscard]] auto mesh_count() const -> std::size_t { return meshes_.size(); }
  [[nodiscard]] auto live_mesh_count() const -> std::size_t {
    return meshes_.size() - free_mesh_slots_.size();
  }

  // Mesh slots are stable: MeshInstance::mesh_index refers to a slot, not a
  // position, so removal frees the slot for reuse rather than compacting the
  // vector. A streaming world recycles a bounded pool of slots this way instead
  // of growing one forever.
  [[nodiscard]] auto add_mesh(MeshSource mesh) -> std::uint32_t {
    validate_mesh(static_cast<std::uint32_t>(meshes_.size()), mesh);
    if (!free_mesh_slots_.empty()) {
      const std::uint32_t index = free_mesh_slots_.back();
      free_mesh_slots_.pop_back();
      MeshSlot &slot = meshes_[index];
      slot.bounds = compute_bounds(mesh);
      slot.source = std::move(mesh);
      slot.alive = true;
      ++slot.revision;
      return index;
    }

    const std::uint32_t index = static_cast<std::uint32_t>(meshes_.size());
    MeshSlot slot;
    slot.bounds = compute_bounds(mesh);
    slot.source = std::move(mesh);
    slot.revision = 1;
    meshes_.push_back(std::move(slot));
    return index;
  }

  // Replace a slot's geometry in place, keeping its index valid. This is the
  // remesh path -- an edited or LOD-swapped chunk keeps its instances.
  // Every index must name a vertex that exists.
  //
  // An index past the end of the vertex buffer is not a crash and not a
  // validation error -- Vulkan reads whatever is there, which is usually zero,
  // so the triangle collapses to the origin. On screen that is a fan of
  // geometry all converging on one point, which is what got reported from play.
  // Checking it here names the mesh that did it; checking it on the GPU is not
  // possible at all.
  //
  // Counted rather than thrown: a bad mesh should be visible in a report, not
  // take the game down mid-session.
  void validate_mesh(std::uint32_t index, const MeshSource &mesh) {
    const auto vertex_count = static_cast<std::uint32_t>(mesh.vertices.size());
    for (const std::uint32_t i : mesh.indices) {
      if (i < vertex_count)
        continue;
      ++bad_mesh_uploads_;
      if (bad_mesh_uploads_ <= 8)
        std::cerr << "MESH BUG: slot " << index << " has index " << i << " but only "
                  << vertex_count << " vertices (" << mesh.indices.size()
                  << " indices). Triangles will collapse to the origin.\n";
      return;
    }
    if (mesh.indices.size() % 3 != 0) {
      ++bad_mesh_uploads_;
      if (bad_mesh_uploads_ <= 8)
        std::cerr << "MESH BUG: slot " << index << " has " << mesh.indices.size()
                  << " indices, which is not a whole number of triangles.\n";
    }
  }

  [[nodiscard]] auto bad_mesh_uploads() const -> std::uint64_t { return bad_mesh_uploads_; }

  void update_mesh(std::uint32_t index, MeshSource mesh) {
    if (index >= meshes_.size())
      return;
    validate_mesh(index, mesh);
    MeshSlot &slot = meshes_[index];
    slot.bounds = compute_bounds(mesh);
    slot.source = std::move(mesh);
    slot.alive = true;
    ++slot.revision;
  }

  // Frees the slot and drops the CPU-side geometry. The renderer reclaims the
  // GPU buffers once no in-flight frame can still reference them.
  void remove_mesh(std::uint32_t index) {
    if (index >= meshes_.size() || !meshes_[index].alive)
      return;
    MeshSlot &slot = meshes_[index];
    slot.source = {};
    slot.bounds = {};
    slot.alive = false;
    ++slot.revision;
    free_mesh_slots_.push_back(index);
  }

  [[nodiscard]] auto add_texture(std::string path) -> std::uint32_t {
    auto it = texture_path_index_.find(path);
    if (it != texture_path_index_.end())
      return it->second;
    const std::uint32_t index = static_cast<std::uint32_t>(texture_paths_.size());
    texture_path_index_[path] = index;
    texture_paths_.push_back(std::move(path));
    return index;
  }

  // `overlay`, when given, is alpha-composited over the base image as the layer
  // is loaded. It exists so a game can ship one wear/crack/frost image and get a
  // worn variant of every tile for free, instead of authoring and shipping N of
  // them. Doing it here rather than in a shader keeps the cost at load time:
  // sampling a second texture and blending per fragment would be paid forever.
  [[nodiscard]] auto add_texture_array_layer(std::string path, std::string overlay = {})
      -> std::uint32_t {
    const std::uint32_t index = static_cast<std::uint32_t>(texture_array_layer_paths_.size());
    texture_array_layer_paths_.push_back({std::move(path), std::move(overlay)});
    return index;
  }

  // Instance slots are recycled like mesh slots. Without reuse a game that
  // spawns and despawns grows this vector for the lifetime of the session, and
  // every frame walks the tombstones to build the draw list.
  [[nodiscard]] auto add_instance(MeshInstance instance) -> InstanceHandle {
    if (!free_instance_slots_.empty()) {
      const std::uint32_t index = free_instance_slots_.back();
      free_instance_slots_.pop_back();
      const std::uint32_t generation = instances_[index].generation + 1;
      instances_[index] = std::move(instance);
      instances_[index].alive = true;
      instances_[index].generation = generation;
      return {.index = index, .generation = generation};
    }

    const std::uint32_t index = static_cast<std::uint32_t>(instances_.size());
    instance.alive = true;
    instance.generation = 1;
    instances_.push_back(std::move(instance));
    return {.index = index, .generation = 1};
  }

  void remove_instance(InstanceHandle handle) {
    MeshInstance *instance = try_instance(handle);
    if (instance == nullptr)
      return;
    instance->alive = false;
    // Bumped here so handles to a released slot stop resolving. try_instance
    // no longer tests `alive`, so the generation is the only thing separating
    // "hidden" from "gone".
    ++instance->generation;
    // Drop per-instance storage now rather than holding it until the slot is
    // reused; a horde's worth of cached bone transforms is not free.
    instance->bone_attachments.clear();
    instance->bone_attachments.shrink_to_fit();
    instance->cached_bone_worlds.clear();
    instance->cached_bone_worlds.shrink_to_fit();
    instance->pose_layers.reset();
    instance->joint_overrides.reset();
    free_instance_slots_.push_back(handle.index);
  }

  [[nodiscard]] auto live_instance_count() const -> std::size_t {
    return instances_.size() - free_instance_slots_.size();
  }

  [[nodiscard]] auto add_point_light(PointLight light) -> std::uint32_t {
    const std::uint32_t index = static_cast<std::uint32_t>(point_lights_.size());
    point_lights_.push_back(std::move(light));
    return index;
  }

  [[nodiscard]] auto add_spot_light(SpotLight light) -> std::uint32_t {
    const std::uint32_t index = static_cast<std::uint32_t>(spot_lights_.size());
    spot_lights_.push_back(std::move(light));
    return index;
  }

  // Light indices are stable (the GPU cubemap layer for a shadow-casting point
  // light IS its array index), so removal blanks the slot in place rather than
  // erasing it. A default-constructed light is NOT blank — it is a white,
  // intensity-1, range-5 lamp at the world origin — so zero the emissive fields
  // explicitly.
  void remove_point_light(std::uint32_t index) {
    if (index >= point_lights_.size())
      return;
    PointLight &light = point_lights_[index];
    light = {};
    light.color = glm::vec3(0.0F);
    light.intensity = 0.0F;
    light.range = 0.0F;
    light.casts_shadow = false;
  }

  void remove_spot_light(std::uint32_t index) {
    if (index >= spot_lights_.size())
      return;
    SpotLight &light = spot_lights_[index];
    light = {};
    light.color = glm::vec3(0.0F);
    light.intensity = 0.0F;
    light.range = 0.0F;
    light.casts_shadow = false;
  }

  [[nodiscard]] auto add_skeleton(SkeletonAsset skeleton) -> std::uint32_t {
    const std::uint32_t index = static_cast<std::uint32_t>(skeletons_.size());
    skeletons_.push_back(std::move(skeleton));
    return index;
  }

  [[nodiscard]] auto add_animation(AnimationClip animation) -> std::uint32_t {
    const std::uint32_t index = static_cast<std::uint32_t>(animations_.size());
    animations_.push_back(std::move(animation));
    return index;
  }

private:
  Camera camera_;
  DirectionalLight directional_light_{};
  DirectionalLightShadowSettings shadow_settings_{};
  std::vector<MeshSlot> meshes_;
  std::vector<std::uint32_t> free_mesh_slots_;
  std::uint64_t bad_mesh_uploads_{0};
  std::vector<std::string> texture_paths_;
  std::unordered_map<std::string, std::uint32_t> texture_path_index_;
  std::vector<TextureArrayLayer> texture_array_layer_paths_;
  std::vector<MeshInstance> instances_;
  std::vector<std::uint32_t> free_instance_slots_;
  std::vector<SkeletonAsset> skeletons_;
  std::vector<AnimationClip> animations_;
  std::vector<PointLight> point_lights_;
  std::vector<SpotLight> spot_lights_;
};

// Single source of truth for "does this instance own a per-instance bone buffer
// and skinned descriptor set?".
//
// Bone buffers (`create_bone_buffers`), skinned descriptor sets
// (`count_skinned_instances`), the per-frame joint-matrix upload in
// `draw_frame`, and `DrawCommand::bone_instance_index` all index the SAME
// sequential slot list. Every one of those sites must walk `instances()` in
// order and advance its counter on exactly this predicate. If any two disagree
// the slot lists shift relative to each other and instances silently render
// with another model's pose. Do not inline a variant of this check.
//
// Note it deliberately does NOT test `pose_layers`: an instance without a pose
// stack still owns a slot and renders in bind pose
// (`compute_joint_matrices_for_instance` handles the null case).
[[nodiscard]] inline auto instance_uses_skinning(const MeshInstance &instance, const Scene &scene) -> bool {
  return instance.alive
      && instance.skin_index != k_invalid_skin_index
      && instance.skin_index < scene.skeletons().size()
      && !scene.skeletons()[instance.skin_index].joint_nodes.empty();
}

} // namespace engine
