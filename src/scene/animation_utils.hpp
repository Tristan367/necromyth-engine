#pragma once

#include "scene/animation_types.hpp"
#include "scene/mesh_instance.hpp"

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
    if (channel->sampler_index >= clip.samplers.size()) continue;
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
    std::size_t depth_guard = 0;
    const std::size_t max_depth = skeleton.node_parents.size() + 1;
    while (current != k_invalid && depth_guard++ < max_depth) {
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

inline auto trs_to_mat4(const BoneTRS &trs) -> glm::mat4 {
  return glm::translate(glm::mat4(1.0F), trs.translation) *
         glm::mat4_cast(trs.rotation) *
         glm::scale(glm::mat4(1.0F), trs.scale);
}

inline auto blend_bone_trs(const BoneTRS &a, const BoneTRS &b, float factor) -> BoneTRS {
  if (factor <= 0.0F) return a;
  if (factor >= 1.0F) return b;
  return {
      glm::mix(a.translation, b.translation, factor),
      glm::normalize(glm::slerp(a.rotation, b.rotation, factor)),
      glm::mix(a.scale, b.scale, factor),
  };
}

// Evaluate an ordered stack of pose layers into joint matrices.
//
// Each layer samples its clip (with its own internal A->B crossfade) and is
// composited OVER the accumulated pose per masked joint by its weight. Layer 0
// is the full-body base; higher layers are masked overrides blended on top.
// Transitions inside a layer always crossfade, and layers never fight over
// shared fields.
inline void evaluate_pose_layers(
    const SkeletonAsset &skeleton,
    const std::vector<PoseLayer> &layers,
    const std::vector<AnimationClip> &clips,
    const std::unordered_map<std::uint32_t, BoneTRS> *joint_overrides,
    std::vector<glm::mat4> &out_joint_matrices,
    std::vector<glm::mat4> *out_bone_worlds = nullptr) {
  const std::size_t joint_count = skeleton.joint_nodes.size();

  thread_local std::unordered_map<std::uint32_t, detail::ChannelNodeMap> tls_channel_maps;
  auto &channel_maps = tls_channel_maps;
  channel_maps.clear();
  auto clip_map = [&](std::uint32_t idx) -> const detail::ChannelNodeMap & {
    auto it = channel_maps.find(idx);
    if (it == channel_maps.end())
      it = channel_maps.emplace(idx, detail::build_channel_map(clips[idx])).first;
    return it->second;
  };

  auto sample_layer = [&](const PoseLayer &layer, std::uint32_t node_index) -> BoneTRS {
    const BoneTRS a =
        detail::sample_animation_trs(clips[layer.clip_index], layer.time, node_index,
                                     clip_map(layer.clip_index));
    if (layer.xfade_index >= clips.size() || layer.xfade_weight <= 0.0F)
      return a;
    const BoneTRS b =
        detail::sample_animation_trs(clips[layer.xfade_index], layer.xfade_time, node_index,
                                     clip_map(layer.xfade_index));
    return blend_bone_trs(a, b, layer.xfade_weight);
  };

  // Reused across calls (per thread) to avoid per-frame heap churn. resize()
  // keeps capacity; each set is cleared then refilled from the (static) mask.
  thread_local std::vector<std::unordered_set<std::uint32_t>> mask_sets;
  mask_sets.resize(layers.size());
  for (std::size_t li = 0; li < layers.size(); ++li) {
    mask_sets[li].clear();
    if (layers[li].mask && !layers[li].mask->empty())
      mask_sets[li].insert(layers[li].mask->begin(), layers[li].mask->end());
  }

  thread_local std::unordered_map<std::uint32_t, glm::mat4> tls_node_anim;
  auto &node_anim = tls_node_anim;
  node_anim.clear();
  node_anim.reserve(joint_count);

  for (std::size_t i = 0; i < joint_count; ++i) {
    const std::uint32_t node_index = skeleton.joint_nodes[i];

    bool has_pose = false;
    BoneTRS accum;
    for (std::size_t li = 0; li < layers.size(); ++li) {
      const PoseLayer &layer = layers[li];
      if (layer.clip_index >= clips.size() || layer.weight <= 0.0F) continue;
      if (!mask_sets[li].empty() && !mask_sets[li].count(static_cast<std::uint32_t>(i)))
        continue;

      const BoneTRS sampled = sample_layer(layer, node_index);
      if (!has_pose) {
        accum = sampled;
        has_pose = true;
      } else {
        accum = blend_bone_trs(accum, sampled, std::clamp(layer.weight, 0.0F, 1.0F));
      }
    }

    if (!has_pose)
      accum = BoneTRS{};

    if (joint_overrides) {
      const auto it = joint_overrides->find(static_cast<std::uint32_t>(i));
      if (it != joint_overrides->end())
        accum = it->second;
    }

    node_anim[node_index] = trs_to_mat4(accum);
  }

  detail::build_world_matrices(skeleton, node_anim, out_joint_matrices, out_bone_worlds);
}

// Compute joint matrices from an instance's pose-layer stack.
inline void compute_joint_matrices_for_instance(
    const SkeletonAsset &skel,
    const MeshInstance &instance,
    const std::vector<AnimationClip> &clips,
    std::vector<glm::mat4> &out_joint_matrices,
    std::vector<glm::mat4> *out_bone_worlds = nullptr) {
  if (!instance.pose_layers || instance.pose_layers->empty()) {
    out_joint_matrices.resize(skel.joint_nodes.size(), glm::mat4(1.0F));
    if (out_bone_worlds)
      out_bone_worlds->resize(skel.joint_nodes.size(), glm::mat4(1.0F));
    return;
  }
  evaluate_pose_layers(skel, *instance.pose_layers, clips,
                       instance.joint_overrides,
                       out_joint_matrices, out_bone_worlds);
}

inline void update_bone_attachments(
    std::vector<MeshInstance> &instances) {
  for (MeshInstance &instance : instances) {
    if (instance.bone_attachments.empty()) continue;
    if (instance.cached_bone_worlds.empty()) continue;

    for (BoneAttachment &att : instance.bone_attachments) {
      if (att.joint_index < instance.cached_bone_worlds.size()) {
        att.world_transform = instance.model * instance.cached_bone_worlds[att.joint_index];
        if (att.target_instance != k_invalid_skin_index && att.target_instance < instances.size())
          instances[att.target_instance].model = att.world_transform;
      }
    }
  }
}

} // namespace engine
