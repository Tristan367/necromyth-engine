#pragma once

#include "platform/sdl_window.hpp"
#include "renderer/vulkan_context.hpp"

#include <SDL3/SDL_events.h>

#include <iostream>

namespace engine {

class App {
public:
    App()
        : window_("Vulkan C++ Engine", 1280, 720),
          vulkan_(window_.handle()) {
        std::cout << "Selected GPU: " << vulkan_.gpu_name() << '\n';
    }

    void run() {
        while (running_) {
            SDL_Event event{};
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_EVENT_QUIT)
                    running_ = false;
                else if (event.type == SDL_EVENT_WINDOW_RESIZED)
                    vulkan_.mark_framebuffer_resized();
            }

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
