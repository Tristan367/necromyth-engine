#pragma once

#include "engine_config.hpp"
#include "platform/sdl_window.hpp"
#include "platform/timer_sync.hpp"
#include "renderer/vulkan_context.hpp"
#include "scene/scene.hpp"

#include <csignal>
#include <atomic>
#include <utility>

namespace engine {

// Bundles SDL window, Vulkan context, fixed-timestep accumulator, and signal handlers
// so the demo doesn't need to own any of them individually.
class EngineRuntime {
public:
  explicit EngineRuntime(EngineConfig config, Scene &scene)
      : sdl_(),
        window_(config.window_title, config.window_width, config.window_height),
        vulkan_(window_.handle(), config, scene),
        timer_() {
    std::signal(SIGINT, on_quit_signal);
    std::signal(SIGTERM, on_quit_signal);
  }

  EngineRuntime(const EngineRuntime &) = delete;
  auto operator=(const EngineRuntime &) -> EngineRuntime & = delete;
  EngineRuntime(EngineRuntime &&) = delete;
  auto operator=(EngineRuntime &&) -> EngineRuntime & = delete;

  void shutdown() { vulkan_.shutdown(); }

  [[nodiscard]] auto window_handle() const -> SDL_Window * { return window_.handle(); }
  [[nodiscard]] auto vulkan() -> VulkanContext & { return vulkan_; }
  [[nodiscard]] auto timer() -> TimerSync & { return timer_; }
  [[nodiscard]] static auto quit_requested() -> bool {
    return g_quit_requested.load(std::memory_order_relaxed) != 0;
  }

private:
  static void on_quit_signal(int) {
    g_quit_requested.store(1, std::memory_order_relaxed);
  }

  static inline std::atomic<std::sig_atomic_t> g_quit_requested = 0;

  SdlContext sdl_;
  SdlWindow window_;
  VulkanContext vulkan_;
  TimerSync timer_;
};

} // namespace engine
