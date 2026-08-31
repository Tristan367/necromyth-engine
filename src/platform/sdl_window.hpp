#pragma once

#include <SDL3/SDL.h>
#include <SDL3/SDL_video.h>

#include <stdexcept>
#include <cstdlib>
#include <string>
#include <string_view>

namespace engine {

namespace detail {

[[nodiscard]] inline auto sdl_error(std::string_view message) -> std::runtime_error {
    return std::runtime_error(std::string(message) + ": " + SDL_GetError());
}

} // namespace detail

class SdlContext {
public:
    SdlContext() {
        if (!SDL_Init(SDL_INIT_VIDEO))
            throw detail::sdl_error("Failed to initialize SDL");
    }

    ~SdlContext() {
        SDL_Quit();
    }

    SdlContext(const SdlContext&) = delete;
    auto operator=(const SdlContext&) -> SdlContext& = delete;

    SdlContext(SdlContext&&) = delete;
    auto operator=(SdlContext&&) -> SdlContext& = delete;
};

class SdlWindow {
public:
    SdlWindow(std::string_view title, int width, int height) {
        // ENGINE_BACKGROUND_WINDOW=1 asks the compositor not to hand this
        // window the keyboard.
        //
        // For automated runs -- benchmarks, screenshot passes, soak tests --
        // the window has to exist, because Vulkan needs a surface, but it has
        // no business taking focus from whatever the person at the keyboard is
        // actually doing. Without this, every headless run yanks the caret out
        // of their editor for the second and a half it lives.
        SDL_WindowFlags flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE;
        //
        // =1 hides the window entirely. =2 maps it but refuses focus.
        //
        // The difference is not cosmetic and it cost days. A HIDDEN window is
        // never composited, so vkQueuePresentKHR has no compositor on the other
        // end of it and the whole presentation path -- the part where a
        // stuttering game actually stutters -- is not exercised at all. Every
        // "no frame over 28ms" measured headless was measuring a code path that
        // cannot stagger. =2 is the mode that can: a real mapped surface going
        // through the real compositor, which still will not take the keyboard
        // away from whoever is using the desktop.
        if (const char *env = std::getenv("ENGINE_BACKGROUND_WINDOW");
            env != nullptr && (env[0] == '1' || env[0] == '2')) {
            flags |= SDL_WINDOW_NOT_FOCUSABLE;
            if (env[0] == '1')
                flags |= SDL_WINDOW_HIDDEN;
        }

        // A benchmark window says so in its title, so a compositor rule can
        // put it somewhere out of the way without also catching the real game
        // window. On Hyprland:
        //
        //   hyprctl keyword windowrulev2 \
        //     'workspace 7 silent, title:^(Necromyth \(benchmark\))$'
        //
        // `silent` opens it on workspace 7 without switching to it, which is
        // the difference between a benchmark you can ignore and one that takes
        // over the screen every time it runs.
        std::string window_title(title);
        if (const char *env = std::getenv("ENGINE_BACKGROUND_WINDOW");
            env != nullptr && env[0] == '2')
            window_title += " (benchmark)";
        window_ = SDL_CreateWindow(window_title.c_str(), width, height, flags);

        if (window_ == nullptr)
            throw detail::sdl_error("Failed to create SDL window");
    }

    ~SdlWindow() {
        if (window_ != nullptr)
            SDL_DestroyWindow(window_);
    }

    SdlWindow(const SdlWindow&) = delete;
    auto operator=(const SdlWindow&) -> SdlWindow& = delete;

    SdlWindow(SdlWindow&& other) noexcept
        : window_(other.window_) {
        other.window_ = nullptr;
    }

    auto operator=(SdlWindow&& other) noexcept -> SdlWindow& {
        if (this != &other) {
            if (window_ != nullptr)
                SDL_DestroyWindow(window_);

            window_ = other.window_;
            other.window_ = nullptr;
        }

        return *this;
    }

    [[nodiscard]] auto handle() const -> SDL_Window* {
        return window_;
    }

private:
    SDL_Window* window_{};
};

} // namespace engine
