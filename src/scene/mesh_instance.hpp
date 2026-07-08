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

  const std::unordered_map<std::uint32_t, BoneTRS> *joint_overrides{nullptr};
  const std::vector<PoseLayer> *pose_layers{nullptr};
};

constexpr auto k_invalid_skin_index = std::numeric_limits<std::uint32_t>::max();

} // namespace engine
