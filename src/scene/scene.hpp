#pragma once

#include "scene/animation_types.hpp"
#include "scene/camera.hpp"
#include "scene/directional_light.hpp"
#include "scene/shadow_utils.hpp"
#include "scene/mesh_instance.hpp"
#include "scene/mesh_source.hpp"

#include <cstdint>
#include <string>
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

  [[nodiscard]] auto skeletons() const -> const std::vector<SkeletonAsset> & {
    return skeletons_;
  }

  [[nodiscard]] auto animations() const -> const std::vector<AnimationClip> & {
    return animations_;
  }

  [[nodiscard]] auto add_mesh(MeshSource mesh) -> std::uint32_t {
    const std::uint32_t index = static_cast<std::uint32_t>(meshes_.size());
    meshes_.push_back(std::move(mesh));
    return index;
  }

  [[nodiscard]] auto add_texture(std::string path) -> std::uint32_t {
    const std::uint32_t index = static_cast<std::uint32_t>(texture_paths_.size());
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
    instances_.push_back(instance);
    return index;
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
  std::vector<std::string> texture_array_layer_paths_;
  std::vector<MeshInstance> instances_;
  std::vector<SkeletonAsset> skeletons_;
  std::vector<AnimationClip> animations_;
};

[[nodiscard]] inline auto is_skinned_instance(const MeshInstance &instance, const Scene &scene) -> bool {
  if (instance.skin_index >= scene.skeletons().size())
    return false;
  return !scene.skeletons()[instance.skin_index].joint_nodes.empty();
}

} // namespace engine
