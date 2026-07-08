#pragma once

#include "scene/animation_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine {

enum class AnimConditionOp : std::uint8_t { OnTrue, Greater, Less, AtEnd };

struct AnimStateDef {
  std::string name;
  std::uint32_t clip_index;
  bool looping{true};
};

struct AnimTransitionDef {
  std::string from_state;     // "*" = from any state
  std::string to_state;
  std::string condition_param;
  AnimConditionOp op{AnimConditionOp::OnTrue};
  float value{0.0F};
  float blend_time{0.15F};
  std::uint8_t priority{0};   // lower wins (like Godot)
};

// Locomotion state machine driving a layered pose stack.
//
// Layer 0 is the full-body base layer this state machine owns and crossfades.
// Additional masked override layers (e.g. an upper-body "hold weapon" pose) can
// be registered with add_override_layer() and controlled independently; they are
// composited OVER the base per their bone mask and weight. Point a MeshInstance
// at layers() (via instance.pose_layers) once, then call tick() each frame.
class AnimStateMachine {
public:
  AnimStateMachine() {
    layers_.resize(1);  // layer 0 = base locomotion (full body)
  }

  void add_state(AnimStateDef s) {
    if (find_state(s.name) != k_invalid) return;
    state_map_[s.name] = states_.size();
    states_.push_back(std::move(s));
  }

  void add_transition(AnimTransitionDef t) { transitions_.push_back(std::move(t)); }

  void set_param(const std::string &name, float v) { params_[name] = v; }
  void set_param(const std::string &name, bool v)  { params_[name] = v ? 1.0F : 0.0F; }

  // The pose-layer stack. Assign to MeshInstance::pose_layers once at setup.
  [[nodiscard]] auto layers() const -> const std::vector<PoseLayer> & { return layers_; }

  // Register a masked override layer (e.g. upper body). Returns its layer index.
  // mask is joint indices the layer affects; weight is its blend over the base.
  auto add_override_layer(const std::vector<std::uint32_t> *mask, float weight = 1.0F)
      -> std::size_t {
    PoseLayer layer;
    layer.mask = mask;
    layer.weight = weight;
    layer.clip_index = std::numeric_limits<std::uint32_t>::max();  // inactive until set
    layers_.push_back(layer);
    return layers_.size() - 1;
  }

  // Directly drive an override layer's clip (with its own internal crossfade).
  void set_override_clip(std::size_t layer_index, std::uint32_t clip_index,
                         float blend_time) {
    if (layer_index == 0 || layer_index >= layers_.size()) return;
    PoseLayer &l = layers_[layer_index];
    if (l.clip_index >= k_invalid_clip) {  // was inactive: snap on
      l.clip_index = clip_index;
      l.time = 0.0F;
      l.xfade_index = k_invalid_clip;
      l.xfade_weight = 1.0F;
      return;
    }
    if (l.clip_index == clip_index) return;
    l.xfade_index = clip_index;
    l.xfade_time = 0.0F;
    l.xfade_weight = 0.0F;
    override_xfade_dur_[layer_index] = std::max(blend_time, 0.001F);
  }

  // Smoothly fade an override layer's compositing weight toward `weight` over
  // `fade_time` seconds. Use fade_time <= 0 for an instant set.
  void set_override_weight(std::size_t layer_index, float weight, float fade_time = 0.2F) {
    if (layer_index == 0 || layer_index >= layers_.size()) return;
    const float target = std::clamp(weight, 0.0F, 1.0F);
    override_target_weight_[layer_index] = target;
    if (fade_time <= 0.0F) {
      layers_[layer_index].weight = target;
      override_weight_rate_[layer_index] = 0.0F;
    } else {
      override_weight_rate_[layer_index] =
          std::abs(target - layers_[layer_index].weight) / fade_time;
    }
  }

  // Force-switch the base layer to a state immediately (no crossfade).
  void start(const std::string &name) {
    const std::size_t idx = find_state(name);
    if (idx == k_invalid) return;
    current_index_ = idx;
    transitioning_ = false;
    PoseLayer &base = layers_[0];
    base.clip_index = states_[idx].clip_index;
    base.time = 0.0F;
    base.xfade_index = k_invalid_clip;
    base.xfade_weight = 1.0F;
  }

  // Switch the base layer to a state via the best-matching transition (crossfade).
  // Sets a manual hold — auto conditions are suppressed until clear_hold().
  void travel(const std::string &name) {
    const std::size_t idx = find_state(name);
    if (idx == k_invalid || idx == current_index_ || states_.empty()) return;

    manual_hold_ = true;

    const AnimTransitionDef *best = nullptr;
    for (const AnimTransitionDef &t : transitions_) {
      if (t.to_state != name) continue;
      if (t.from_state != "*" && t.from_state != states_[current_index_].name) continue;
      if (!best || t.priority < best->priority) best = &t;
    }
    begin_base_transition(idx, best ? best->blend_time : 0.15F);
  }

  // Release a manual hold — auto condition checking resumes next tick.
  void clear_hold() { manual_hold_ = false; }

  // Advance layer times, check base-layer transitions, drive all crossfades.
  void tick(float delta, const std::vector<AnimationClip> &clips) {
    if (states_.empty()) return;

    advance_base_layer(delta, clips);
    advance_override_layers(delta, clips);

    if (transitioning_) return;  // hold auto-conditions during a transition

    if (!manual_hold_)
      check_conditions(clips);
  }

  [[nodiscard]] auto current() const -> const std::string & {
    return states_.empty() ? empty_ : states_[current_index_].name;
  }

  [[nodiscard]] auto is_transitioning() const -> bool { return transitioning_; }

private:
  static constexpr std::size_t k_invalid = ~std::size_t{0};
  static constexpr std::uint32_t k_invalid_clip = std::numeric_limits<std::uint32_t>::max();
  static inline const std::string empty_{};

  std::vector<PoseLayer> layers_;
  std::unordered_map<std::size_t, float> override_xfade_dur_;
  std::unordered_map<std::size_t, float> override_target_weight_;
  std::unordered_map<std::size_t, float> override_weight_rate_;

  std::vector<AnimStateDef> states_;
  std::unordered_map<std::string, std::size_t> state_map_;
  std::vector<AnimTransitionDef> transitions_;
  std::unordered_map<std::string, float> params_;
  std::size_t current_index_{0};
  std::size_t next_index_{0};
  bool transitioning_{false};
  float transition_duration_{0.15F};
  bool manual_hold_{false};

  [[nodiscard]] auto find_state(const std::string &name) const -> std::size_t {
    auto it = state_map_.find(name);
    return it != state_map_.end() ? it->second : k_invalid;
  }

  static void loop_time(float &time, std::uint32_t clip,
                        const std::vector<AnimationClip> &clips) {
    if (clip < clips.size() && clips[clip].duration > 0.0F && time > clips[clip].duration)
      time = std::fmod(time, clips[clip].duration);
  }

  void begin_base_transition(std::size_t to_idx, float blend_time) {
    if (to_idx == current_index_) return;
    next_index_ = to_idx;
    transition_duration_ = std::max(blend_time, 0.001F);
    transitioning_ = true;

    PoseLayer &base = layers_[0];
    base.xfade_index = states_[to_idx].clip_index;
    base.xfade_time = 0.0F;
    base.xfade_weight = 0.0F;
  }

  void advance_base_layer(float delta, const std::vector<AnimationClip> &clips) {
    PoseLayer &base = layers_[0];
    const bool looping = states_[current_index_].looping;

    base.time += delta;
    if (looping) loop_time(base.time, base.clip_index, clips);

    if (!transitioning_) return;

    base.xfade_time += delta;
    loop_time(base.xfade_time, base.xfade_index, clips);

    base.xfade_weight += delta / transition_duration_;
    if (base.xfade_weight >= 1.0F) {
      // Promote target to current.
      base.clip_index = base.xfade_index;
      base.time = base.xfade_time;
      base.xfade_index = k_invalid_clip;
      base.xfade_weight = 1.0F;
      transitioning_ = false;
      current_index_ = next_index_;
    }
  }

  void advance_override_layers(float delta, const std::vector<AnimationClip> &clips) {
    for (std::size_t li = 1; li < layers_.size(); ++li) {
      PoseLayer &l = layers_[li];

      // Fade layer weight toward its target.
      auto rate_it = override_weight_rate_.find(li);
      if (rate_it != override_weight_rate_.end() && rate_it->second > 0.0F) {
        const float target = override_target_weight_.count(li)
                                 ? override_target_weight_[li] : l.weight;
        const float step = rate_it->second * delta;
        if (l.weight < target)
          l.weight = std::min(target, l.weight + step);
        else
          l.weight = std::max(target, l.weight - step);
        if (l.weight == target) rate_it->second = 0.0F;
      }

      if (l.clip_index >= k_invalid_clip) continue;  // inactive

      l.time += delta;
      loop_time(l.time, l.clip_index, clips);

      if (l.xfade_index >= k_invalid_clip) continue;

      l.xfade_time += delta;
      loop_time(l.xfade_time, l.xfade_index, clips);

      const float dur = override_xfade_dur_.count(li) ? override_xfade_dur_[li] : 0.15F;
      l.xfade_weight += delta / dur;
      if (l.xfade_weight >= 1.0F) {
        l.clip_index = l.xfade_index;
        l.time = l.xfade_time;
        l.xfade_index = k_invalid_clip;
        l.xfade_weight = 1.0F;
      }
    }
  }

  void check_conditions(const std::vector<AnimationClip> &clips) {
    const AnimStateDef &state = states_[current_index_];
    const PoseLayer &base = layers_[0];
    const AnimTransitionDef *firing = nullptr;
    for (const AnimTransitionDef &t : transitions_) {
      if (t.from_state != "*" && t.from_state != state.name) continue;

      bool met = false;
      switch (t.op) {
      case AnimConditionOp::OnTrue: {
        auto it = params_.find(t.condition_param);
        met = (it != params_.end() && it->second > 0.5F);
        break;
      }
      case AnimConditionOp::Greater: {
        auto it = params_.find(t.condition_param);
        met = (it != params_.end() && it->second > t.value);
        break;
      }
      case AnimConditionOp::Less: {
        auto it = params_.find(t.condition_param);
        met = (it != params_.end() && it->second < t.value);
        break;
      }
      case AnimConditionOp::AtEnd:
        met = !state.looping && base.clip_index < clips.size() &&
              base.time >= clips[base.clip_index].duration;
        break;
      }

      if (met && (!firing || t.priority < firing->priority))
        firing = &t;
    }

    if (firing) {
      const std::size_t to_idx = find_state(firing->to_state);
      if (to_idx != k_invalid && to_idx != current_index_)
        begin_base_transition(to_idx, firing->blend_time);
    }
  }
};

} // namespace engine
