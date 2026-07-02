#pragma once

#include "scene/animation_types.hpp"
#include "scene/mesh_instance.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
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

// Minimal state machine. Call tick() each frame with the MeshInstance to drive.
class AnimStateMachine {
public:
  void add_state(AnimStateDef s) {
    if (find_state(s.name) != k_invalid) return;
    state_map_[s.name] = states_.size();
    states_.push_back(std::move(s));
  }

  void add_transition(AnimTransitionDef t) { transitions_.push_back(std::move(t)); }

  void set_param(const std::string &name, float v) { params_[name] = v; }
  void set_param(const std::string &name, bool v)  { params_[name] = v ? 1.0F : 0.0F; }

  // Force-switch to a state immediately (no crossfade).
  void start(const std::string &name) {
    const std::size_t idx = find_state(name);
    if (idx == k_invalid) return;
    current_index_ = idx;
    transitioning_ = false;
  }

  // Switch to a state via the best-matching transition (with crossfade).
  // Falls back to instant switch if no transition defined.
  void travel(MeshInstance &instance, const std::string &name, bool split_active) {
    const std::size_t idx = find_state(name);
    if (idx == k_invalid || idx == current_index_) return;

    const AnimTransitionDef *best = nullptr;
    for (const AnimTransitionDef &t : transitions_) {
      if (t.to_state != name) continue;
      if (t.from_state != "*" && t.from_state != states_[current_index_].name) continue;
      if (!best || t.priority < best->priority) best = &t;
    }
    force_transition(instance, idx, best ? best->blend_time : 0.0F, split_active);
  }

  // Advance animation time, check transitions, handle crossfade.
  // clips is needed for AtEnd transitions to check animation completion.
  void tick(float delta, MeshInstance &instance,
            const std::vector<AnimationClip> &clips) {
    if (states_.empty()) return;

    const AnimStateDef &state = states_[current_index_];
    const bool split_active = instance.secondary_joints && !instance.secondary_joints->empty() &&
                              instance.next_animation_index < clips.size();

    // Advance current animation time
    instance.animation_time += delta * instance.animation_speed;
    if (state.looping && state.clip_index < clips.size() &&
        instance.animation_time > clips[state.clip_index].duration)
      instance.animation_time =
          std::fmod(instance.animation_time, clips[state.clip_index].duration);

    // Process crossfade
    if (transitioning_ && !split_active) {
      static int xfade_dbg = 0;
      if (++xfade_dbg % 5 == 0)
        std::cout << "xfade: blend=" << instance.blend_factor
                  << " cur=" << instance.animation_index
                  << " next=" << instance.next_animation_index << "\n";
      instance.next_animation_time += delta * instance.animation_speed;
      instance.blend_factor += delta / transition_duration_;
      if (instance.blend_factor >= 1.0F) {
        instance.animation_index = states_[next_index_].clip_index;
        instance.animation_time = instance.next_animation_time;
        if (!split_active)
          instance.next_animation_index = std::numeric_limits<std::uint32_t>::max();
        instance.blend_factor = 1.0F;
        transitioning_ = false;
        current_index_ = next_index_;
      }
      return;
    }

    if (transitioning_ && split_active) {
      // Crossfade and split can't share next_animation_index — instant switch instead
      instance.animation_index = states_[next_index_].clip_index;
      instance.animation_time = 0.0F;
      instance.blend_factor = 1.0F;
      transitioning_ = false;
      current_index_ = next_index_;
      return;
    }

    // Check condition-based transitions
    const AnimTransitionDef *firing = nullptr;
    for (const AnimTransitionDef &t : transitions_) {
      if (t.from_state != "*" && t.from_state != states_[current_index_].name) continue;

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
        met = !state.looping && !transitioning_ &&
              instance.animation_time >= clips[state.clip_index].duration;
        break;
      }

      if (met && (!firing || t.priority < firing->priority))
        firing = &t;
    }

    if (firing) {
      const std::size_t to_idx = find_state(firing->to_state);
      if (to_idx != k_invalid && to_idx != current_index_)
        force_transition(instance, to_idx, firing->blend_time, split_active);
    }
  }

  [[nodiscard]] auto current() const -> const std::string & {
    return states_.empty() ? empty_ : states_[current_index_].name;
  }

private:
  static constexpr std::size_t k_invalid = ~std::size_t{0};
  static inline const std::string empty_{};

  std::vector<AnimStateDef> states_;
  std::unordered_map<std::string, std::size_t> state_map_;
  std::vector<AnimTransitionDef> transitions_;
  std::unordered_map<std::string, float> params_;
  std::size_t current_index_{0};
  std::size_t next_index_{0};
  bool transitioning_{false};
  float transition_duration_{0.0F};

  [[nodiscard]] auto find_state(const std::string &name) const -> std::size_t {
    auto it = state_map_.find(name);
    return it != state_map_.end() ? it->second : k_invalid;
  }

  void force_transition(MeshInstance &instance, std::size_t to_idx, float blend_time,
                         bool split_active) {
    if (to_idx == current_index_) return;
    next_index_ = to_idx;
    transition_duration_ = std::max(blend_time, 0.001F);
    transitioning_ = true;

    if (!split_active) {
      instance.next_animation_index = states_[to_idx].clip_index;
      instance.next_animation_time = 0.0F;
      instance.blend_factor = 0.0F;
    }
  }
};

} // namespace engine
