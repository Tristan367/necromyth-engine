#pragma once

#include "renderer/textured_push_constants.hpp"
#include "scene/animation_types.hpp"
#include "scene/render_layer.hpp"

#include <glm/mat4x4.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace engine {

class Scene;

// Reference to a scene instance that survives other instances being removed.
//
// Instance slots are recycled, so a bare index is not enough: after a despawn
// the same index can belong to an entirely different entity, and code holding
// the old index would silently move, animate or delete the wrong one. The
// generation is bumped every time a slot is reused, so a stale handle is
// detectable rather than merely wrong.
struct InstanceHandle {
  static constexpr std::uint32_t k_invalid_index = std::numeric_limits<std::uint32_t>::max();

  std::uint32_t index{k_invalid_index};
  std::uint32_t generation{0};

  [[nodiscard]] auto is_set() const -> bool { return index != k_invalid_index; }

  auto operator==(const InstanceHandle &) const -> bool = default;
};

struct BoneAttachment {
  std::uint32_t joint_index{};
  InstanceHandle target_instance{};
  glm::mat4 world_transform{1.0F};
};

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

  // Shared, not raw pointers into caller storage. These used to point straight
  // into an AnimStateMachine the game owned, so anything that relocated it -- a
  // std::vector of them growing, an entity being moved between containers --
  // left every instance referencing freed memory, silently.
  //
  // Sharing the pose stack instead means the state machine can move freely (the
  // pointee does not) and, if it is destroyed outright, the character freezes in
  // its last pose rather than reading dead memory.
  std::shared_ptr<const std::unordered_map<std::uint32_t, BoneTRS>> joint_overrides;
  std::shared_ptr<const std::vector<PoseLayer>> pose_layers;

  std::vector<BoneAttachment> bone_attachments;
  std::vector<glm::mat4> cached_bone_worlds;
  bool alive{true};
  // Bumped each time this slot is reused; compared against InstanceHandle.
  std::uint32_t generation{1};
};

constexpr auto k_invalid_skin_index = std::numeric_limits<std::uint32_t>::max();

} // namespace engine
