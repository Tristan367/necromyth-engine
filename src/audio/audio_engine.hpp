#pragma once

#include <glm/vec3.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace engine {
namespace audio {

using SoundHandle = std::uint32_t;
inline constexpr SoundHandle k_invalid_sound = ~SoundHandle{0};

class AudioEngine {
public:
  AudioEngine();
  ~AudioEngine();

  AudioEngine(const AudioEngine &) = delete;
  AudioEngine &operator=(const AudioEngine &) = delete;
  AudioEngine(AudioEngine &&) noexcept;
  AudioEngine &operator=(AudioEngine &&) noexcept;

  [[nodiscard]] auto init() -> bool;
  void shutdown();

  // Load a sound from file. Returns handle for later operations.
  [[nodiscard]] auto load_sound(const std::string &path, bool loop) -> SoundHandle;

  void play(SoundHandle h);
  void stop(SoundHandle h);
  [[nodiscard]] auto is_playing(SoundHandle h) const -> bool;

  void set_volume(SoundHandle h, float vol);   // 0..1
  void set_position(SoundHandle h, const glm::vec3 &pos);
  void set_looping(SoundHandle h, bool loop);

  // Listener position (typically camera position) — call each frame.
  void set_listener(const glm::vec3 &pos, const glm::vec3 &forward, const glm::vec3 &up);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace audio
} // namespace engine
