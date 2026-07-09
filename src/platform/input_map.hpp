#pragma once

#include <SDL3/SDL.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine {

enum class InputBindingType : std::uint8_t { Key, GamepadButton, GamepadAxis };

struct InputBinding {
  InputBindingType type{InputBindingType::Key};
  SDL_Scancode key{SDL_SCANCODE_UNKNOWN};
  int gamepad_button{-1};
  SDL_GamepadAxis gamepad_axis{};
  int axis_direction{0};

  static auto key_binding(SDL_Scancode sc) -> InputBinding {
    InputBinding b;
    b.type = InputBindingType::Key;
    b.key = sc;
    return b;
  }
  static auto button_binding(int btn) -> InputBinding {
    InputBinding b;
    b.type = InputBindingType::GamepadButton;
    b.gamepad_button = btn;
    return b;
  }
  static auto axis_binding(SDL_GamepadAxis ax, int dir) -> InputBinding {
    InputBinding b;
    b.type = InputBindingType::GamepadAxis;
    b.gamepad_axis = ax;
    b.axis_direction = dir;
    return b;
  }
};

class InputMap {
public:
  void register_action(std::string_view name, float deadzone = 0.2F) {
    if (name.empty() || name_map_.find(std::string(name)) != name_map_.end())
      return;
    name_map_[std::string(name)] = actions_.size();
    actions_.push_back({std::string(name), deadzone, {}, {}});
  }

  [[nodiscard]] auto has_action(std::string_view name) const -> bool {
    return name_map_.find(std::string(name)) != name_map_.end();
  }

  void bind(std::string_view action, InputBinding binding) {
    const std::size_t idx = find_action(action);
    if (idx != k_invalid)
      actions_[idx].bindings.push_back(binding);
  }

  // Call at the START of each frame, before processing events
  void new_frame() { ++frame_; }

  // Feed a raw SDL event — updates persistent key/button/axis state,
  // then recomputes action states from the updated raw state.
  void process_event(const SDL_Event &event) {
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN:
      held_keys_[event.key.scancode] = true;
      recompute_key_actions();
      break;
    case SDL_EVENT_KEY_UP:
      held_keys_[event.key.scancode] = false;
      recompute_key_actions();
      break;
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
      held_buttons_[event.gbutton.button] = true;
      recompute_button_actions();
      break;
    case SDL_EVENT_GAMEPAD_BUTTON_UP:
      held_buttons_[event.gbutton.button] = false;
      recompute_button_actions();
      break;
    case SDL_EVENT_GAMEPAD_AXIS_MOTION:
      axis_values_[static_cast<int>(event.gaxis.axis)] =
          static_cast<float>(event.gaxis.value) / static_cast<float>(SDL_JOYSTICK_AXIS_MAX);
      recompute_axis_actions();
      break;
    default:
      break;
    }
  }

  // --- Queries ---

  [[nodiscard]] auto pressed(std::string_view name) const -> bool {
    const std::size_t idx = find_action(name);
    return idx != k_invalid && actions_[idx].state.pressed;
  }

  [[nodiscard]] auto just_pressed(std::string_view name) const -> bool {
    const std::size_t idx = find_action(name);
    return idx != k_invalid && actions_[idx].state.pressed &&
           actions_[idx].state.pressed_frame == frame_;
  }

  [[nodiscard]] auto just_released(std::string_view name) const -> bool {
    const std::size_t idx = find_action(name);
    return idx != k_invalid && !actions_[idx].state.pressed &&
           actions_[idx].state.released_frame == frame_;
  }

  [[nodiscard]] auto strength(std::string_view name) const -> float {
    const std::size_t idx = find_action(name);
    return idx != k_invalid ? actions_[idx].state.strength : 0.0F;
  }

  [[nodiscard]] auto raw_strength(std::string_view name) const -> float {
    const std::size_t idx = find_action(name);
    return idx != k_invalid ? actions_[idx].state.raw : 0.0F;
  }

  [[nodiscard]] auto axis(std::string_view neg, std::string_view pos) const -> float {
    return strength(pos) - strength(neg);
  }

private:
  static constexpr std::size_t k_invalid = ~std::size_t{0};

  struct ActionState {
    bool pressed{false};
    float strength{0.0F};
    float raw{0.0F};
    std::uint64_t pressed_frame{0};
    std::uint64_t released_frame{0};
  };

  struct Action {
    std::string name;
    float deadzone{0.2F};
    std::vector<InputBinding> bindings;
    ActionState state;
  };

  [[nodiscard]] auto find_action(std::string_view name) const -> std::size_t {
    const auto it = name_map_.find(std::string(name));
    return it != name_map_.end() ? it->second : k_invalid;
  }

  void update_action_state(Action &action, bool any_pressed, float max_strength, float max_raw) {
    const bool was = action.state.pressed;
    if (any_pressed != was) {
      if (any_pressed)
        action.state.pressed_frame = frame_;
      else
        action.state.released_frame = frame_;
    }
    action.state.pressed = any_pressed;
    action.state.strength = max_strength;
    action.state.raw = max_raw;
  }

  void recompute_key_actions() {
    for (Action &action : actions_) {
      bool any_pressed = false;
      float max_strength = 0.0F;
      for (const InputBinding &b : action.bindings) {
        if (b.type != InputBindingType::Key) continue;
        auto it = held_keys_.find(b.key);
        if (it != held_keys_.end() && it->second) {
          any_pressed = true;
          max_strength = 1.0F;
        }
      }
      update_action_state(action, any_pressed, max_strength, max_strength);
    }
  }

  void recompute_button_actions() {
    for (Action &action : actions_) {
      bool any_pressed = false;
      float max_strength = 0.0F;
      for (const InputBinding &b : action.bindings) {
        if (b.type != InputBindingType::GamepadButton) continue;
        auto it = held_buttons_.find(b.gamepad_button);
        if (it != held_buttons_.end() && it->second) {
          any_pressed = true;
          max_strength = 1.0F;
        }
      }
      update_action_state(action, any_pressed, max_strength, max_strength);
    }
  }

  void recompute_axis_actions() {
    for (Action &action : actions_) {
      bool any_pressed = false;
      float max_strength = 0.0F;
      float max_raw = 0.0F;
      for (const InputBinding &b : action.bindings) {
        if (b.type != InputBindingType::GamepadAxis) continue;
        auto it = axis_values_.find(static_cast<int>(b.gamepad_axis));
        if (it == axis_values_.end()) continue;
        const float val = it->second;
        const bool same_dir = (b.axis_direction < 0 && val <= 0.0F) ||
                              (b.axis_direction > 0 && val >= 0.0F);
        if (!same_dir) continue;
        const float raw = std::abs(val);
        const float s = raw >= action.deadzone
            ? std::clamp((raw - action.deadzone) / (1.0F - action.deadzone), 0.0F, 1.0F)
            : 0.0F;
        max_raw = std::max(max_raw, raw);
        max_strength = std::max(max_strength, s);
        any_pressed = any_pressed || (raw >= action.deadzone);
      }
      update_action_state(action, any_pressed, max_strength, max_raw);
    }
  }

  std::vector<Action> actions_;
  std::unordered_map<std::string, std::size_t> name_map_;
  std::uint64_t frame_{0};

  std::unordered_map<SDL_Scancode, bool> held_keys_;
  std::unordered_map<int, bool> held_buttons_;
  std::unordered_map<int, float> axis_values_;
};

} // namespace engine
