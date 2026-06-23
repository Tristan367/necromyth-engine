#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace engine {

struct SkeletonAsset {
  std::vector<glm::mat4> inverse_bind_matrices;
  std::vector<std::uint32_t> joint_nodes;
  std::vector<std::uint32_t> node_parents;
  glm::mat4 inverse_skin_node_transform{1.0F};
  std::uint32_t skeleton_root{std::numeric_limits<std::uint32_t>::max()};
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
