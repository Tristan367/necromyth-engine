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
  int axis_direction{0}; // +1 positive half-axis, -1 negative half-axis

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

  void new_frame() { ++frame_; }

  void process_event(const SDL_Event &event) {
    float event_val = 0.0F;
    bool is_down = false;
    InputBindingType event_type;

    switch (event.type) {
    case SDL_EVENT_KEY_DOWN:
      is_down = true;
      event_val = 1.0F;
      event_type = InputBindingType::Key;
      break;
    case SDL_EVENT_KEY_UP:
      is_down = false;
      event_val = 0.0F;
      event_type = InputBindingType::Key;
      break;
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
      is_down = true;
      event_val = 1.0F;
      event_type = InputBindingType::GamepadButton;
      break;
    case SDL_EVENT_GAMEPAD_BUTTON_UP:
      is_down = false;
      event_val = 0.0F;
      event_type = InputBindingType::GamepadButton;
      break;
    case SDL_EVENT_GAMEPAD_AXIS_MOTION:
      event_val = static_cast<float>(event.gaxis.value) / static_cast<float>(SDL_JOYSTICK_AXIS_MAX);
      event_type = InputBindingType::GamepadAxis;
      break;
    default:
      return;
    }

    for (Action &action : actions_) {
      bool any_pressed = false;
      float max_raw = 0.0F;
      float max_strength = 0.0F;

      for (const InputBinding &binding : action.bindings) {
        if (binding.type != event_type)
          continue;

        bool bind_pressed = false;
        float bind_raw = 0.0F;
        float bind_strength = 0.0F;

        switch (binding.type) {
        case InputBindingType::Key:
          if (event.key.scancode == binding.key) {
            bind_pressed = is_down;
            bind_raw = event_val;
            bind_strength = event_val;
          }
          break;
        case InputBindingType::GamepadButton:
          if (event.gbutton.button == binding.gamepad_button) {
            bind_pressed = is_down;
            bind_raw = event_val;
            bind_strength = event_val;
          }
          break;
        case InputBindingType::GamepadAxis:
          if (event.gaxis.axis == binding.gamepad_axis) {
            const int dir = binding.axis_direction;
            const bool same_dir = (dir < 0 && event.gaxis.value <= 0) ||
                                  (dir > 0 && event.gaxis.value >= 0);
            if (same_dir) {
              const float raw = std::abs(event_val);
              bind_raw = raw;
              bind_pressed = raw >= action.deadzone;
              bind_strength =
                  bind_pressed
                      ? std::clamp((raw - action.deadzone) / (1.0F - action.deadzone), 0.0F, 1.0F)
                      : 0.0F;
            }
          }
          break;
        }

        if (bind_raw > 0.0F || bind_pressed) {
          max_raw = std::max(max_raw, bind_raw);
          max_strength = std::max(max_strength, bind_strength);
          any_pressed = any_pressed || bind_pressed;
        }
      }

      const bool was_pressed = action.state.pressed;

      if (any_pressed != was_pressed) {
        if (any_pressed)
          action.state.pressed_frame = frame_;
        else
          action.state.released_frame = frame_;
      }

      action.state.pressed = any_pressed;
      action.state.strength = max_strength;
      action.state.raw = max_raw;
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

  // Composite: pos_action - neg_action (using deadzone-remapped strengths)
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

  std::vector<Action> actions_;
  std::unordered_map<std::string, std::size_t> name_map_;
  std::uint64_t frame_{0};
};

} // namespace engine
