#pragma once

#include <SDL3/SDL.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>

namespace engine {

class FlyCamera {
public:
  void init(SDL_Window *window) {
    window_ = window;
  }

  void sync_from(glm::vec3 forward) {
    pitch_ = std::asin(std::clamp(forward.y, -1.0F, 1.0F));
    yaw_ = std::atan2(forward.z, forward.x);
  }

  void set_capture(bool enabled) {
    if (!window_) return;
    if (enabled && !captured_) {
      SDL_SetWindowRelativeMouseMode(window_, true);
      captured_ = true;
    } else if (!enabled && captured_) {
      SDL_SetWindowRelativeMouseMode(window_, false);
      captured_ = false;
    }
  }

  void toggle_capture() { set_capture(!captured_); }

  void handle_event(const SDL_Event &event) {
    if (event.type == SDL_EVENT_WINDOW_FOCUS_LOST) {
      set_capture(false);
      return;
    }
    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && !captured_) {
      set_capture(true);
      return;
    }
    if (event.type == SDL_EVENT_MOUSE_MOTION && captured_) {
      yaw_ += static_cast<float>(event.motion.xrel) * 0.002F;
      pitch_ -= static_cast<float>(event.motion.yrel) * 0.002F;
      pitch_ = std::clamp(pitch_, -1.5F, 1.5F);
    }
    if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE) {
      toggle_capture();
    }
  }

  void update(glm::vec3 &position, glm::vec3 &forward_out, float dt,
              const bool *keys) {
    if (!captured_) return;

    const float cos_pitch = std::cos(pitch_);
    forward_out = glm::normalize(glm::vec3{
        cos_pitch * std::cos(yaw_),
        std::sin(pitch_),
        cos_pitch * std::sin(yaw_),
    });

    const glm::vec3 right = glm::normalize(glm::cross(forward_out, glm::vec3{0, 1, 0}));

    const float speed = keys[SDL_SCANCODE_LSHIFT] ? 48.0F : 12.0F;
    glm::vec3 vel{0};
    if (keys[SDL_SCANCODE_W]) vel += forward_out;
    if (keys[SDL_SCANCODE_S]) vel -= forward_out;
    if (keys[SDL_SCANCODE_D]) vel += right;
    if (keys[SDL_SCANCODE_A]) vel -= right;
    if (keys[SDL_SCANCODE_SPACE]) vel.y += 1;
    if (keys[SDL_SCANCODE_C]) vel.y -= 1;

    if (glm::dot(vel, vel) > 0.0F)
      position += glm::normalize(vel) * speed * dt;
  }

private:
  SDL_Window *window_{};
  float yaw_{};
  float pitch_{};
  bool captured_{};
};

} // namespace engine
