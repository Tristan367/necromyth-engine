#pragma once

#include "demo_scene.hpp"
#include "engine_config.hpp"
#include "platform/sdl_window.hpp"
#include "renderer/vulkan_context.hpp"
#include "scene/scene.hpp"

#include <SDL3/SDL_events.h>

#include <csignal>
#include <iostream>

namespace engine {

namespace {

volatile std::sig_atomic_t g_quit_requested = 0;

void on_quit_signal(int) {
  g_quit_requested = 1;
}

} // namespace

class App {
public:
  explicit App(EngineConfig config = engine_config_from_environment())
      : config_(std::move(config)),
        window_(config_.window_title, config_.window_width, config_.window_height),
        scene_(create_demo_scene()),
        vulkan_(window_.handle(), config_, scene_) {
    std::signal(SIGINT, on_quit_signal);
    std::signal(SIGTERM, on_quit_signal);
    std::cout << "Selected GPU: " << vulkan_.gpu_name();
    if (config_.gpu_device_index)
      std::cout << " (requested index " << *config_.gpu_device_index << ')';
    std::cout << '\n';
  }

  void run() {
    while (running_) {
      if (g_quit_requested != 0)
        running_ = false;

      SDL_Event event{};
      while (SDL_PollEvent(&event)) {
        if (event.type == SDL_EVENT_QUIT)
          running_ = false;
        else if (event.type == SDL_EVENT_WINDOW_RESIZED)
          vulkan_.mark_framebuffer_resized();
      }

      if (!running_)
        break;

      update_demo_scene(scene_);
      vulkan_.draw_frame(scene_);
    }
  }

private:
  EngineConfig config_;
  SdlContext sdl_;
  SdlWindow window_;
  Scene scene_;
  VulkanContext vulkan_;
  bool running_{true};
};

} // namespace engine
