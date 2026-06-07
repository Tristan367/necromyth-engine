#pragma once

#include <SDL3/SDL.h>
#include <SDL3/SDL_video.h>

#include <stdexcept>
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
        window_ = SDL_CreateWindow(
            std::string(title).c_str(),
            width,
            height,
            SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

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
