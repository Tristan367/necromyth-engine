#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
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
  vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
};

inline constexpr std::uint32_t k_max_render_scale = 64U;

// Integer divisor: 1 = full swapchain resolution, 2 = half, etc.
[[nodiscard]] inline auto scaled_render_extent(vk::Extent2D swapchain_extent, std::uint32_t render_scale)
    -> vk::Extent2D {
  if (render_scale <= 1U)
    return swapchain_extent;

  return {
      .width = std::max(1U, swapchain_extent.width / render_scale),
      .height = std::max(1U, swapchain_extent.height / render_scale),
  };
}

[[nodiscard]] inline auto render_scale_active(std::uint32_t render_scale) -> bool {
  return render_scale > 1U;
}

namespace detail {

[[nodiscard]] inline auto supported_framebuffer_sample_counts(const vk::PhysicalDeviceProperties &properties)
    -> vk::SampleCountFlags {
  return properties.limits.framebufferColorSampleCounts & properties.limits.framebufferDepthSampleCounts;
}

[[nodiscard]] inline auto highest_sample_count(vk::SampleCountFlags supported) -> vk::SampleCountFlagBits {
  constexpr std::array candidates{
      vk::SampleCountFlagBits::e8,
      vk::SampleCountFlagBits::e4,
      vk::SampleCountFlagBits::e2,
  };

  for (const auto sample : candidates)
    if ((supported & sample) != vk::SampleCountFlags{})
      return sample;

  return vk::SampleCountFlagBits::e1;
}

[[nodiscard]] inline auto resolve_msaa_samples(MsaaSettings settings, vk::SampleCountFlags supported)
    -> vk::SampleCountFlagBits {
  if (!settings.enabled)
    return vk::SampleCountFlagBits::e1;

  if (settings.samples != vk::SampleCountFlagBits::e1) {
    if ((supported & settings.samples) != vk::SampleCountFlags{})
      return settings.samples;
    return highest_sample_count(supported);
  }

  if ((supported & vk::SampleCountFlagBits::e4) != vk::SampleCountFlags{})
    return vk::SampleCountFlagBits::e4;
  if ((supported & vk::SampleCountFlagBits::e2) != vk::SampleCountFlags{})
    return vk::SampleCountFlagBits::e2;

  return vk::SampleCountFlagBits::e1;
}

} // namespace detail

[[nodiscard]] inline auto msaa_is_active(vk::SampleCountFlagBits samples) -> bool {
  return samples != vk::SampleCountFlagBits::e1;
}

[[nodiscard]] inline auto resolve_msaa_for_scene(MsaaSettings settings, bool scene_uses_alpha_to_coverage)
    -> MsaaSettings {
  if (scene_uses_alpha_to_coverage && !settings.enabled)
    settings.enabled = true;
  return settings;
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

[[nodiscard]] inline auto render_scale_settings_from_environment() -> std::uint32_t {
  const char *env = std::getenv("ENGINE_RENDER_SCALE");
  if (env == nullptr || env[0] == '\0')
    return 1U;

  const int requested = std::atoi(env);
  if (requested <= 1)
    return 1U;

  return std::min(static_cast<std::uint32_t>(requested), k_max_render_scale);
}

} // namespace engine
