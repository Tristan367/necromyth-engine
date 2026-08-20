#pragma once

#include "scene/animation_types.hpp"
#include "scene/camera.hpp"
#include "scene/directional_light.hpp"
#include "scene/shadow_utils.hpp"
#include "scene/mesh_instance.hpp"
#include "scene/mesh_source.hpp"
#include "scene/point_light.hpp"
#include "scene/spot_light.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine {

class Scene {
public:
  [[nodiscard]] auto camera() -> Camera & {
    return camera_;
  }

  [[nodiscard]] auto camera() const -> const Camera & {
    return camera_;
  }

  [[nodiscard]] auto meshes() const -> const std::vector<MeshSource> & {
    return meshes_;
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

  [[nodiscard]] auto texture_array_layer_paths() const -> const std::vector<std::string> & {
    return texture_array_layer_paths_;
  }

  [[nodiscard]] auto instance(std::uint32_t index) -> MeshInstance & {
    return instances_.at(index);
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
  [[nodiscard]] auto mesh_count() const -> std::size_t { return meshes_.size(); }

  [[nodiscard]] auto add_mesh(MeshSource mesh) -> std::uint32_t {
    const std::uint32_t index = static_cast<std::uint32_t>(meshes_.size());
    meshes_.push_back(std::move(mesh));
    return index;
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

  [[nodiscard]] auto add_texture_array_layer(std::string path) -> std::uint32_t {
    const std::uint32_t index = static_cast<std::uint32_t>(texture_array_layer_paths_.size());
    texture_array_layer_paths_.push_back(std::move(path));
    return index;
  }

  [[nodiscard]] auto add_instance(MeshInstance instance) -> std::uint32_t {
    const std::uint32_t index = static_cast<std::uint32_t>(instances_.size());
    instances_.push_back(std::move(instance));
    return index;
  }

  void remove_instance(std::uint32_t index) {
    if (index < instances_.size())
      instances_[index].alive = false;
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
  std::vector<MeshSource> meshes_;
  std::vector<std::string> texture_paths_;
  std::unordered_map<std::string, std::uint32_t> texture_path_index_;
  std::vector<std::string> texture_array_layer_paths_;
  std::vector<MeshInstance> instances_;
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
