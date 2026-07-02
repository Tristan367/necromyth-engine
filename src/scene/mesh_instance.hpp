#pragma once

#include "renderer/textured_push_constants.hpp"
#include "scene/animation_types.hpp"
#include "scene/render_layer.hpp"

#include <glm/mat4x4.hpp>

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace engine {

class Scene;

enum class MeshAlphaMode : std::uint8_t {
  Opaque = 0,
  Cutout = 1,
  AlphaToCoverage = 2,
};

struct MeshInstance {
  std::uint32_t mesh_index{0};
  std::uint32_t texture_index{0};
  TextureSource texture_source{TextureSource::Table};
  glm::mat4 model{1.0F};
  RenderLayer layer{RenderLayer::Opaque};
  MeshAlphaMode alpha_mode{MeshAlphaMode::Opaque};

  std::uint32_t skin_index{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t animation_index{std::numeric_limits<std::uint32_t>::max()};
  float animation_time{0.0F};
  float animation_speed{1.0F};
  bool animation_loop{true};

  std::uint32_t next_animation_index{std::numeric_limits<std::uint32_t>::max()};
  float next_animation_time{0.0F};
  float blend_factor{1.0F};
  float blend_duration{0.3F};

  const std::vector<std::uint32_t> *secondary_joints{nullptr};
  std::uint32_t secondary_animation_index{std::numeric_limits<std::uint32_t>::max()};
  float secondary_animation_time{0.0F};
  const std::unordered_map<std::uint32_t, BoneTRS> *joint_overrides{nullptr};
};

constexpr auto k_invalid_skin_index = std::numeric_limits<std::uint32_t>::max();

[[nodiscard]] inline auto is_skinned_instance(const MeshInstance &instance, const Scene &scene) -> bool;

} // namespace engine
