#pragma once

#include "scene/mesh_instance.hpp"
#include "scene/render_layer.hpp"
#include "scene/shadow_utils.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace engine {

enum class PipelineId : std::uint8_t {
  Background = 0,
  TexturedOpaque = 1,
  TexturedCutout = 2,
  TexturedAlphaToCoverage = 3,
  ShadowDepth = 4,
};

[[nodiscard]] constexpr auto textured_pipeline(MeshAlphaMode alpha_mode) -> PipelineId {
  switch (alpha_mode) {
  case MeshAlphaMode::Cutout:
    return PipelineId::TexturedCutout;
  case MeshAlphaMode::AlphaToCoverage:
    return PipelineId::TexturedAlphaToCoverage;
  case MeshAlphaMode::Opaque:
  default:
    return PipelineId::TexturedOpaque;
  }
}

[[nodiscard]] inline auto textured_fragment_entry(
    ShadowFilterMode filter,
    MeshAlphaMode alpha_mode,
    ShadowCascadeMode cascade_mode) -> const char * {
  static constexpr std::array<const char *, 3> hard{
      "fragOpaqueHard",
      "fragCutoutHard",
      "fragA2CHard",
  };
  static constexpr std::array<const char *, 3> pcf{
      "fragOpaquePcf",
      "fragCutoutPcf",
      "fragA2CPcf",
  };
  static constexpr std::array<const char *, 3> hard_csm2{
      "fragOpaqueHardCsm2",
      "fragCutoutHardCsm2",
      "fragA2CHardCsm2",
  };
  static constexpr std::array<const char *, 3> pcf_csm2{
      "fragOpaquePcfCsm2",
      "fragCutoutPcfCsm2",
      "fragA2CPcfCsm2",
  };

  const auto alpha_index = static_cast<std::size_t>(alpha_mode);
  if (cascade_mode == ShadowCascadeMode::Dual) {
    if (filter == ShadowFilterMode::Pcf3x3)
      return pcf_csm2[alpha_index];
    return hard_csm2[alpha_index];
  }
  if (filter == ShadowFilterMode::Pcf3x3)
    return pcf[alpha_index];
  return hard[alpha_index];
}

[[nodiscard]] inline auto collect_used_alpha_modes(const std::vector<MeshInstance> &instances) -> std::array<bool, 3> {
  std::array<bool, 3> used{false, false, false};
  for (const MeshInstance &instance : instances) {
    if (instance.layer == RenderLayer::Background)
      continue;
    used[static_cast<std::size_t>(instance.alpha_mode)] = true;
  }
  if (!used[0] && !used[1] && !used[2])
    used[0] = true;
  return used;
}

[[nodiscard]] inline auto scene_uses_alpha_to_coverage(const std::vector<MeshInstance> &instances) -> bool {
  return collect_used_alpha_modes(instances)[static_cast<std::size_t>(MeshAlphaMode::AlphaToCoverage)];
}

[[nodiscard]] constexpr auto is_textured_surface_pipeline(PipelineId id) -> bool {
  return id == PipelineId::TexturedOpaque || id == PipelineId::TexturedCutout ||
         id == PipelineId::TexturedAlphaToCoverage;
}

[[nodiscard]] constexpr auto casts_shadow(PipelineId id) -> bool {
  return is_textured_surface_pipeline(id);
}

struct PipelineBuildProfile {
  ShadowFilterMode shadow_filter{ShadowFilterMode::Pcf3x3};
  ShadowCascadeMode cascade_mode{ShadowCascadeMode::Single};
  std::array<bool, 3> textured_alpha_modes{{true, false, false}};
};

} // namespace engine
