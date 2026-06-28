#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine {

enum class HitboxShape : std::uint8_t { Box, Sphere, Capsule };

struct HitboxAttachment {
  std::string name;
  std::uint32_t joint_index;  // index into joint_nodes (0..N-1)
  HitboxShape shape;
  glm::vec3 offset{0.0F};
  glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
  glm::vec3 half_extent{0.3F};  // Box: xyz half-extent. Sphere/Capsule: x = radius
  float half_height{0.5F};      // Capsule cylinder half-height
};

struct BodyColliderDef {
  enum class Shape : std::uint8_t { Capsule, Cylinder, Box, Sphere };
  Shape shape{Shape::Capsule};
  float half_height{0.8F};     // Capsule/Cylinder: half height
  float radius{0.4F};          // Capsule/Cylinder/Sphere: radius
  glm::vec3 half_extent{0.5F}; // Box: half extents
  glm::vec3 offset{0.0F};      // bone-local translation from skeleton root
};

struct SkeletonAsset {
  std::vector<glm::mat4> inverse_bind_matrices;
  std::vector<std::uint32_t> joint_nodes;
  std::vector<std::string> joint_names;  // per-joint name (from glTF node names)
  std::vector<std::uint32_t> node_parents;
  glm::mat4 inverse_skin_node_transform{1.0F};
  std::uint32_t skeleton_root{std::numeric_limits<std::uint32_t>::max()};

  [[nodiscard]] auto find_joint_index(const std::string &name) const -> std::optional<std::uint32_t> {
    for (std::size_t i = 0; i < joint_names.size(); ++i)
      if (joint_names[i] == name)
        return static_cast<std::uint32_t>(i);
    return std::nullopt;
  }

  std::vector<HitboxAttachment> hitboxes;
  std::optional<BodyColliderDef> body_collider;
};

struct AnimationSampler {
  std::string interpolation;
  std::vector<float> inputs;
  std::vector<glm::vec4> outputs;
};

struct AnimationChannel {
  std::uint32_t node_index;
  std::string path;
  std::uint32_t sampler_index;
};

struct AnimationClip {
  std::string name;
  std::vector<AnimationSampler> samplers;
  std::vector<AnimationChannel> channels;
  float duration{0.0F};
};

inline constexpr std::uint32_t k_max_bones = 128;

} // namespace engine
