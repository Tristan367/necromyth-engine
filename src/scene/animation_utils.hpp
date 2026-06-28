#pragma once

#include "scene/animation_types.hpp"

#define GLM_FORCE_RADIANS
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace engine {

namespace detail {

struct KeyframeIndex {
  std::size_t lower;
  std::size_t upper;
  float alpha;
};

inline auto find_keyframes(const AnimationSampler &sampler, float time, bool loop)
    -> KeyframeIndex {
  const std::size_t n = sampler.inputs.size();
  if (n == 0)
    throw std::runtime_error("Animation sampler has no keyframes");

  const float duration = sampler.inputs.back();
  if (duration <= 0.0F)
    return {0, 0, 0.0F};

  float t = time;
  if (loop && t > duration)
    t = std::fmod(t, duration);

  if (t <= sampler.inputs.front())
    return {0, 0, 0.0F};
  if (t >= duration)
    return {n - 1, n - 1, 0.0F};

  const auto it = std::upper_bound(sampler.inputs.begin(), sampler.inputs.end(), t);
  const std::size_t upper = static_cast<std::size_t>(it - sampler.inputs.begin());
  const std::size_t lower = upper - 1;
  const float span = sampler.inputs[upper] - sampler.inputs[lower];
  const float alpha = span > 0.0F ? (t - sampler.inputs[lower]) / span : 0.0F;
  return {lower, upper, alpha};
}

using ChannelNodeMap = std::unordered_map<std::uint32_t, std::vector<const AnimationChannel *>>;

inline auto build_channel_map(const AnimationClip &clip) -> ChannelNodeMap {
  ChannelNodeMap map;
  for (const AnimationChannel &ch : clip.channels)
    map[ch.node_index].push_back(&ch);
  return map;
}

inline auto sample_animation_trs(const AnimationClip &clip, float time,
                                  std::uint32_t node_index,
                                  const ChannelNodeMap &channel_map) -> BoneTRS {
  BoneTRS result;
  const auto it = channel_map.find(node_index);
  if (it == channel_map.end())
    return result;

  for (const AnimationChannel *channel : it->second) {
    const AnimationSampler &sampler = clip.samplers[channel->sampler_index];
    const KeyframeIndex kf = find_keyframes(sampler, time, true);

    if (sampler.interpolation == "STEP") {
      const glm::vec4 &value = sampler.outputs[kf.lower];
      const std::string &path = channel->path;
      if (path == "translation")
        result.translation = glm::vec3(value);
      else if (path == "rotation")
        result.rotation = glm::quat(value.w, value.x, value.y, value.z);
      else if (path == "scale")
        result.scale = glm::vec3(value);
      continue;
    }

    const glm::vec4 &lower = sampler.outputs[kf.lower];
    const glm::vec4 &upper = sampler.outputs[kf.upper];
    const std::string &path = channel->path;

    if (path == "translation") {
      result.translation = glm::vec3(glm::mix(lower, upper, kf.alpha));
    } else if (path == "rotation") {
      const glm::quat q0(lower.w, lower.x, lower.y, lower.z);
      const glm::quat q1(upper.w, upper.x, upper.y, upper.z);
      result.rotation = glm::normalize(glm::slerp(q0, q1, kf.alpha));
    } else if (path == "scale") {
      result.scale = glm::vec3(glm::mix(lower, upper, kf.alpha));
    }
  }

  return result;
}

} // namespace detail

inline auto trs_to_mat4(const BoneTRS &trs) -> glm::mat4 {
  return glm::translate(glm::mat4(1.0F), trs.translation) *
         glm::mat4_cast(trs.rotation) *
         glm::scale(glm::mat4(1.0F), trs.scale);
}

inline auto blend_bone_trs(const BoneTRS &a, const BoneTRS &b, float factor) -> BoneTRS {
  if (factor <= 0.0F)
    return a;
  if (factor >= 1.0F)
    return b;
  return {
      glm::mix(a.translation, b.translation, factor),
      glm::normalize(glm::slerp(a.rotation, b.rotation, factor)),
      glm::mix(a.scale, b.scale, factor),
  };
}

namespace detail {

inline void build_world_matrices(
    const SkeletonAsset &skeleton,
    const std::unordered_map<std::uint32_t, glm::mat4> &node_anim,
    std::vector<glm::mat4> &out_joint_matrices,
    std::vector<glm::mat4> *out_bone_worlds = nullptr) {
  static constexpr std::uint32_t k_invalid = std::numeric_limits<std::uint32_t>::max();
  const std::size_t joint_count = skeleton.joint_nodes.size();
  out_joint_matrices.resize(joint_count);
  if (out_bone_worlds)
    out_bone_worlds->resize(joint_count);

  thread_local std::vector<std::uint32_t> s_chain;
  s_chain.reserve(64);

  for (std::size_t i = 0; i < joint_count; ++i) {
    const std::uint32_t node_index = skeleton.joint_nodes[i];

    s_chain.clear();
    std::uint32_t current = node_index;
    while (current != k_invalid) {
      s_chain.push_back(current);
      if (current < skeleton.node_parents.size())
        current = skeleton.node_parents[current];
      else
        current = k_invalid;
    }

    glm::mat4 world(1.0F);
    for (auto it = s_chain.rbegin(); it != s_chain.rend(); ++it) {
      auto anim_it = node_anim.find(*it);
      if (anim_it != node_anim.end())
        world = world * anim_it->second;
    }

    out_joint_matrices[i] =
        skeleton.inverse_skin_node_transform * world * skeleton.inverse_bind_matrices[i];

    if (out_bone_worlds)
      (*out_bone_worlds)[i] = skeleton.inverse_skin_node_transform * world;
  }
}

} // namespace detail

inline void compute_joint_matrices_blended(
    const SkeletonAsset &skeleton,
    const AnimationClip &clip_a,
    float time_a,
    const AnimationClip &clip_b,
    float time_b,
    float blend_factor,
    std::vector<glm::mat4> &out_joint_matrices,
    std::vector<glm::mat4> *out_bone_worlds = nullptr) {
  const std::size_t joint_count = skeleton.joint_nodes.size();
  const detail::ChannelNodeMap channel_map_a = detail::build_channel_map(clip_a);
  const detail::ChannelNodeMap channel_map_b = detail::build_channel_map(clip_b);

  std::unordered_map<std::uint32_t, glm::mat4> node_anim;
  node_anim.reserve(joint_count);
  for (std::size_t i = 0; i < joint_count; ++i) {
    const std::uint32_t node_index = skeleton.joint_nodes[i];
    const BoneTRS trs_a =
        detail::sample_animation_trs(clip_a, time_a, node_index, channel_map_a);
    const BoneTRS trs_b =
        detail::sample_animation_trs(clip_b, time_b, node_index, channel_map_b);
    node_anim[node_index] = trs_to_mat4(blend_bone_trs(trs_a, trs_b, blend_factor));
  }

  detail::build_world_matrices(skeleton, node_anim, out_joint_matrices, out_bone_worlds);
}

inline void compute_joint_matrices(
    const SkeletonAsset &skeleton,
    const AnimationClip &clip,
    float time,
    std::vector<glm::mat4> &out_joint_matrices,
    std::vector<glm::mat4> *out_bone_worlds = nullptr) {
  compute_joint_matrices_blended(skeleton, clip, time, clip, time, 1.0F,
                                 out_joint_matrices, out_bone_worlds);
}

inline void compute_joint_matrices_masked(
    const SkeletonAsset &skeleton,
    const AnimationMask &mask,
    const AnimationClip &clip_a,
    float time_a,
    const AnimationClip &clip_b,
    float time_b,
    std::vector<glm::mat4> &out_joint_matrices,
    std::vector<glm::mat4> *out_bone_worlds = nullptr) {
  const std::size_t joint_count = skeleton.joint_nodes.size();
  const detail::ChannelNodeMap channel_map_a = detail::build_channel_map(clip_a);
  const detail::ChannelNodeMap channel_map_b = detail::build_channel_map(clip_b);

  std::unordered_map<std::uint32_t, glm::mat4> node_anim;
  node_anim.reserve(joint_count);
  for (std::size_t i = 0; i < joint_count; ++i) {
    const std::uint32_t node_index = skeleton.joint_nodes[i];

    if (i < mask.entries.size() && mask.entries[i].mode == BoneControlMode::Manual) {
      node_anim[node_index] = trs_to_mat4(mask.entries[i].manual_trs);
    } else if (i < mask.entries.size() && mask.entries[i].mode == BoneControlMode::Secondary) {
      node_anim[node_index] = trs_to_mat4(
          detail::sample_animation_trs(clip_b, time_b, node_index, channel_map_b));
    } else {
      node_anim[node_index] = trs_to_mat4(
          detail::sample_animation_trs(clip_a, time_a, node_index, channel_map_a));
    }
  }

  detail::build_world_matrices(skeleton, node_anim, out_joint_matrices, out_bone_worlds);
}

} // namespace engine
