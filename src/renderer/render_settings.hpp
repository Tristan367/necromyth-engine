#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace engine {

enum class PresentModePreference {
  Fifo,
  Mailbox,
};

[[nodiscard]] inline auto present_mode_preference_from_environment() -> PresentModePreference {
  const char *env = std::getenv("ENGINE_PRESENT");
  if (env == nullptr || env[0] == '\0')
    return PresentModePreference::Fifo;

  if (std::string_view(env) == "mailbox" || std::string_view(env) == "uncapped")
    return PresentModePreference::Mailbox;

  return PresentModePreference::Fifo;
}

[[nodiscard]] inline auto present_mode_name(PresentModePreference preference) -> const char * {
  switch (preference) {
  case PresentModePreference::Fifo:
    return "fifo";
  case PresentModePreference::Mailbox:
    return "mailbox";
  }
  return "unknown";
}

[[nodiscard]] inline auto present_mode_name(vk::PresentModeKHR mode) -> const char * {
  switch (mode) {
  case vk::PresentModeKHR::eFifo:
    return "FIFO";
  case vk::PresentModeKHR::eMailbox:
    return "Mailbox";
  case vk::PresentModeKHR::eFifoRelaxed:
    return "FIFO relaxed";
  case vk::PresentModeKHR::eImmediate:
    return "Immediate";
  default:
    return "Unknown";
  }
}

struct MsaaSettings {
  bool enabled = true;
  // When enabled and samples is e1, use up to max_msaa_cap (not device max).
  // Otherwise clamp samples to what the GPU supports (e.g. request e8, get e4).
  vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
  vk::SampleCountFlagBits max_cap = vk::SampleCountFlagBits::e4;
};

namespace detail {

[[nodiscard]] inline auto clamp_sample_count(
    vk::SampleCountFlagBits requested,
    vk::SampleCountFlagBits max_supported) -> vk::SampleCountFlagBits {
  const vk::SampleCountFlags supported{max_supported};
  const auto requested_value = static_cast<std::uint32_t>(requested);

  constexpr std::array candidates{
      vk::SampleCountFlagBits::e64,
      vk::SampleCountFlagBits::e32,
      vk::SampleCountFlagBits::e16,
      vk::SampleCountFlagBits::e8,
      vk::SampleCountFlagBits::e4,
      vk::SampleCountFlagBits::e2,
      vk::SampleCountFlagBits::e1,
  };

  for (const auto candidate : candidates) {
    if (static_cast<std::uint32_t>(candidate) <= requested_value && (supported & candidate) != vk::SampleCountFlags{})
      return candidate;
  }

  return vk::SampleCountFlagBits::e1;
}

[[nodiscard]] inline auto resolve_msaa_samples(MsaaSettings settings, vk::SampleCountFlagBits max_supported)
    -> vk::SampleCountFlagBits {
  if (!settings.enabled)
    return vk::SampleCountFlagBits::e1;

  if (settings.samples == vk::SampleCountFlagBits::e1)
    return clamp_sample_count(settings.max_cap, max_supported);

  return clamp_sample_count(settings.samples, max_supported);
}

} // namespace detail

[[nodiscard]] inline auto msaa_is_active(vk::SampleCountFlagBits samples) -> bool {
  return samples != vk::SampleCountFlagBits::e1;
}

[[nodiscard]] inline auto msaa_settings_from_environment() -> MsaaSettings {
  MsaaSettings settings{};

  const char *env = std::getenv("ENGINE_MSAA");
  if (env == nullptr || env[0] == '\0')
    return settings;

  if (std::string_view(env) == "0" || std::string_view(env) == "off") {
    settings.enabled = false;
    return settings;
  }

  settings.enabled = true;
  const int requested = std::atoi(env);
  switch (requested) {
  case 2:
    settings.samples = vk::SampleCountFlagBits::e2;
    break;
  case 4:
    settings.samples = vk::SampleCountFlagBits::e4;
    break;
  case 8:
    settings.samples = vk::SampleCountFlagBits::e8;
    break;
  default:
    settings.samples = vk::SampleCountFlagBits::e1;
    break;
  }

  return settings;
}

} // namespace engine
