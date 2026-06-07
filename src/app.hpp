#pragma once

#include "platform/sdl_window.hpp"
#include "renderer/vulkan_context.hpp"

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
    App()
        : window_("Vulkan C++ Engine", 1280, 720),
          vulkan_(window_.handle()) {
        std::signal(SIGINT, on_quit_signal);
        std::signal(SIGTERM, on_quit_signal);
        std::cout << "Selected GPU: " << vulkan_.gpu_name() << '\n';
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

            vulkan_.draw_frame();
        }
    }

private:
    SdlContext sdl_;
    SdlWindow window_;
    VulkanContext vulkan_;
    bool running_{true};
};

} // namespace engine
