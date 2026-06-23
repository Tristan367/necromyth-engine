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
  TexturedOpaqueSkinned = 5,
  TexturedCutoutSkinned = 6,
  TexturedAlphaToCoverageSkinned = 7,
  ShadowDepthSkinned = 8,
};

[[nodiscard]] constexpr auto textured_pipeline(MeshAlphaMode alpha_mode, bool skinned = false) -> PipelineId {
  if (skinned) {
    switch (alpha_mode) {
    case MeshAlphaMode::Cutout:
      return PipelineId::TexturedCutoutSkinned;
    case MeshAlphaMode::AlphaToCoverage:
      return PipelineId::TexturedAlphaToCoverageSkinned;
    case MeshAlphaMode::Opaque:
    default:
      return PipelineId::TexturedOpaqueSkinned;
    }
  }
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
  static constexpr const char *k_entries[2][2][3] = {
      {{"fragOpaqueHard", "fragCutoutHard", "fragA2CHard"},
       {"fragOpaquePcf", "fragCutoutPcf", "fragA2CPcf"}},
      {{"fragOpaqueHardCsm2", "fragCutoutHardCsm2", "fragA2CHardCsm2"},
       {"fragOpaquePcfCsm2", "fragCutoutPcfCsm2", "fragA2CPcfCsm2"}},
  };
  return k_entries[cascade_mode == ShadowCascadeMode::Dual ? 1 : 0]
                  [filter == ShadowFilterMode::Pcf3x3 ? 1 : 0]
                  [static_cast<std::uint8_t>(alpha_mode)];
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
  const auto v = static_cast<std::uint8_t>(id);
  return (v >= 1 && v <= 3) || (v >= 5 && v <= 7);
}

[[nodiscard]] constexpr auto is_skinned_pipeline(PipelineId id) -> bool {
  return static_cast<std::uint8_t>(id) >= 5;
}

[[nodiscard]] constexpr auto casts_shadow(PipelineId id) -> bool {
  return is_textured_surface_pipeline(id);
}

struct PipelineBuildProfile {
  ShadowFilterMode shadow_filter{ShadowFilterMode::Pcf3x3};
  ShadowCascadeMode cascade_mode{ShadowCascadeMode::Dual};
  std::array<bool, 3> textured_alpha_modes{{true, false, false}};
  bool build_skinned{false};
};

} // namespace engine
