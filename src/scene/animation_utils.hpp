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
#include <unordered_set>
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

inline void compute_joint_matrices_split(
    const SkeletonAsset &skeleton,
    const AnimationClip &clip_a,
    float time_a,
    const AnimationClip &clip_b,
    float time_b,
    const std::vector<std::uint32_t> &joints_using_b,
    const std::unordered_map<std::uint32_t, BoneTRS> *joint_overrides,
    std::vector<glm::mat4> &out_joint_matrices,
    std::vector<glm::mat4> *out_bone_worlds = nullptr) {
  const std::size_t joint_count = skeleton.joint_nodes.size();
  const detail::ChannelNodeMap channel_map_a = detail::build_channel_map(clip_a);
  const detail::ChannelNodeMap channel_map_b = detail::build_channel_map(clip_b);

  const std::unordered_set<std::uint32_t> b_set(
      joints_using_b.begin(), joints_using_b.end());

  std::unordered_map<std::uint32_t, glm::mat4> node_anim;
  node_anim.reserve(joint_count);
  for (std::size_t i = 0; i < joint_count; ++i) {
    const std::uint32_t node_index = skeleton.joint_nodes[i];

    if (joint_overrides) {
      const auto it = joint_overrides->find(static_cast<std::uint32_t>(i));
      if (it != joint_overrides->end()) {
        // Start from the base animation, overrides only replace non-identity fields
        BoneTRS base = b_set.count(static_cast<std::uint32_t>(i))
            ? detail::sample_animation_trs(clip_b, time_b, node_index, channel_map_b)
            : detail::sample_animation_trs(clip_a, time_a, node_index, channel_map_a);
        if (it->second.rotation != glm::quat{1, 0, 0, 0}) base.rotation = it->second.rotation;
        if (it->second.translation != glm::vec3{0}) base.translation = it->second.translation;
        if (it->second.scale != glm::vec3{1}) base.scale = it->second.scale;
        node_anim[node_index] = trs_to_mat4(base);
        continue;
      }
    }

    const BoneTRS trs = b_set.count(static_cast<std::uint32_t>(i))
        ? detail::sample_animation_trs(clip_b, time_b, node_index, channel_map_b)
        : detail::sample_animation_trs(clip_a, time_a, node_index, channel_map_a);
    node_anim[node_index] = trs_to_mat4(trs);
  }

  detail::build_world_matrices(skeleton, node_anim, out_joint_matrices, out_bone_worlds);
}

// Convenience: dispatch the correct compute_* function based on instance state.
inline void compute_joint_matrices_for_instance(
    const SkeletonAsset &skel,
    const MeshInstance &instance,
    const std::vector<AnimationClip> &clips,
    std::vector<glm::mat4> &out_joint_matrices,
    std::vector<glm::mat4> *out_bone_worlds = nullptr) {
  const AnimationClip &clip_a = clips[instance.animation_index];

  if (instance.secondary_joints && !instance.secondary_joints->empty() &&
      instance.next_animation_index < clips.size()) {
    compute_joint_matrices_split(skel, clip_a, instance.animation_time,
                                  clips[instance.next_animation_index],
                                  instance.next_animation_time,
                                  *instance.secondary_joints,
                                  instance.joint_overrides,
                                  out_joint_matrices, out_bone_worlds);
  } else if (instance.next_animation_index < clips.size()) {
    const AnimationClip &clip_b = clips[instance.next_animation_index];
    if (instance.blend_factor < 1.0F)
      compute_joint_matrices_blended(skel, clip_a, instance.animation_time,
                                      clip_b, instance.next_animation_time,
                                      instance.blend_factor,
                                      out_joint_matrices, out_bone_worlds);
    else
      compute_joint_matrices(skel, clip_a, instance.animation_time,
                              out_joint_matrices, out_bone_worlds);
  } else {
    compute_joint_matrices(skel, clip_a, instance.animation_time,
                            out_joint_matrices, out_bone_worlds);
  }
}

} // namespace engine
