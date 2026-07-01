#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "audio/audio_engine.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine {
namespace audio {

struct AudioEngine::Impl {
  ma_engine engine{};
  std::vector<std::unique_ptr<ma_sound>> sounds;
  bool initialized{false};
};

AudioEngine::AudioEngine() : impl_(std::make_unique<Impl>()) {}

AudioEngine::~AudioEngine() {
  if (impl_)
    shutdown();
}

AudioEngine::AudioEngine(AudioEngine &&) noexcept = default;
AudioEngine &AudioEngine::operator=(AudioEngine &&) noexcept = default;

auto AudioEngine::init() -> bool {
  if (impl_->initialized)
    return true;

  ma_engine_config config = ma_engine_config_init();
  config.listenerCount = 1;

  if (ma_engine_init(&config, &impl_->engine) != MA_SUCCESS)
    return false;

  impl_->initialized = true;
  return true;
}

void AudioEngine::shutdown() {
  if (!impl_->initialized)
    return;

  for (auto &s : impl_->sounds) {
    if (s)
      ma_sound_uninit(s.get());
  }
  impl_->sounds.clear();

  ma_engine_uninit(&impl_->engine);
  impl_->initialized = false;
}

auto AudioEngine::load_sound(const std::string &path, bool loop) -> SoundHandle {
  if (!impl_->initialized)
    return k_invalid_sound;

  auto sound = std::make_unique<ma_sound>();
  const ma_uint32 flags = loop ? MA_SOUND_FLAG_NO_PITCH : 0;

  if (ma_sound_init_from_file(&impl_->engine, path.c_str(), flags, nullptr, nullptr,
                              sound.get()) != MA_SUCCESS)
    return k_invalid_sound;

  ma_sound_set_looping(sound.get(), loop ? MA_TRUE : MA_FALSE);

  const SoundHandle h = static_cast<SoundHandle>(impl_->sounds.size());
  impl_->sounds.push_back(std::move(sound));
  return h;
}

void AudioEngine::play(SoundHandle h) {
  if (h < impl_->sounds.size() && impl_->sounds[h])
    ma_sound_start(impl_->sounds[h].get());
}

void AudioEngine::stop(SoundHandle h) {
  if (h < impl_->sounds.size() && impl_->sounds[h])
    ma_sound_stop(impl_->sounds[h].get());
}

auto AudioEngine::is_playing(SoundHandle h) const -> bool {
  return h < impl_->sounds.size() && impl_->sounds[h] &&
         ma_sound_is_playing(impl_->sounds[h].get()) != MA_FALSE;
}

void AudioEngine::set_volume(SoundHandle h, float vol) {
  if (h < impl_->sounds.size() && impl_->sounds[h])
    ma_sound_set_volume(impl_->sounds[h].get(), vol);
}

void AudioEngine::set_position(SoundHandle h, const glm::vec3 &pos) {
  if (h < impl_->sounds.size() && impl_->sounds[h])
    ma_sound_set_position(impl_->sounds[h].get(), pos.x, pos.y, pos.z);
}

void AudioEngine::set_looping(SoundHandle h, bool loop) {
  if (h < impl_->sounds.size() && impl_->sounds[h])
    ma_sound_set_looping(impl_->sounds[h].get(), loop ? MA_TRUE : MA_FALSE);
}

void AudioEngine::set_listener(const glm::vec3 &pos, const glm::vec3 &forward,
                               const glm::vec3 &up) {
  if (!impl_->initialized)
    return;
  ma_engine_listener_set_position(&impl_->engine, 0, pos.x, pos.y, pos.z);
  ma_engine_listener_set_direction(&impl_->engine, 0, forward.x, forward.y, forward.z);
  ma_engine_listener_set_world_up(&impl_->engine, 0, up.x, up.y, up.z);
}

} // namespace audio
} // namespace engine
