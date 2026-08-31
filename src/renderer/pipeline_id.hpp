#pragma once

#include "scene/mesh_instance.hpp"
#include "scene/render_layer.hpp"
#include "scene/shadow_utils.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine {

// The four textured surface pipelines are kept contiguous, static first and
// then skinned, because three predicates below are range checks over them and a
// range check is only correct while the range is unbroken.
inline constexpr std::size_t k_alpha_mode_count = 4;

enum class PipelineId : std::uint8_t {
  Background = 0,
  TexturedOpaque = 1,
  TexturedCutout = 2,
  TexturedAlphaToCoverage = 3,
  TexturedBlend = 4,
  ShadowDepth = 5,
  TexturedOpaqueSkinned = 6,
  TexturedCutoutSkinned = 7,
  TexturedAlphaToCoverageSkinned = 8,
  TexturedBlendSkinned = 9,
  ShadowDepthSkinned = 10,
  PointShadowDepth = 11,
  PointShadowDepthSkinned = 12,
  ParticleBillboard = 13,
  // The game's own HUD: screen-space quads, pixel font, one instanced draw.
  Ui = 14,
  // The packed 36-byte terrain format (TerrainVertex): same fragment shaders
  // as the static set, a vertex input without colour or joints, and its own
  // depth-only variants because the stride differs. Appended rather than
  // inserted so every range check above keeps meaning what it says.
  TexturedOpaqueTerrain = 15,
  TexturedCutoutTerrain = 16,
  TexturedAlphaToCoverageTerrain = 17,
  TexturedBlendTerrain = 18,
  ShadowDepthTerrain = 19,
  PointShadowDepthTerrain = 20,
};

inline constexpr auto k_first_textured = static_cast<std::uint8_t>(PipelineId::TexturedOpaque);
inline constexpr auto k_first_textured_skinned =
    static_cast<std::uint8_t>(PipelineId::TexturedOpaqueSkinned);
inline constexpr auto k_first_textured_terrain =
    static_cast<std::uint8_t>(PipelineId::TexturedOpaqueTerrain);

[[nodiscard]] constexpr auto textured_pipeline(MeshAlphaMode alpha_mode, bool skinned = false) -> PipelineId {
  auto index = static_cast<std::uint8_t>(alpha_mode);
  if (index >= k_alpha_mode_count)
    index = 0;
  return static_cast<PipelineId>((skinned ? k_first_textured_skinned : k_first_textured) + index);
}

[[nodiscard]] constexpr auto terrain_textured_pipeline(MeshAlphaMode alpha_mode) -> PipelineId {
  auto index = static_cast<std::uint8_t>(alpha_mode);
  if (index >= k_alpha_mode_count)
    index = 0;
  return static_cast<PipelineId>(k_first_textured_terrain + index);
}

[[nodiscard]] constexpr auto is_terrain_pipeline(PipelineId id) -> bool {
  const auto v = static_cast<std::uint8_t>(id);
  return v >= k_first_textured_terrain &&
         v <= static_cast<std::uint8_t>(PipelineId::PointShadowDepthTerrain);
}

[[nodiscard]] inline auto textured_fragment_entry(
    ShadowFilterMode filter,
    MeshAlphaMode alpha_mode,
    ShadowCascadeMode cascade_mode) -> const char * {
  // Blend shares the opaque entry point deliberately. A blended surface shades
  // exactly like a solid one and then hands its alpha to the blender, so it
  // needs pipeline state -- blending on, depth writes off -- and not a line of
  // shader code of its own. Four more entry points here would be four more
  // SPIR-V functions saying the same thing.
  static constexpr const char *k_entries[2][2][k_alpha_mode_count] = {
      {{"fragOpaqueHard", "fragCutoutHard", "fragA2CHard", "fragOpaqueHard"},
       {"fragOpaquePcf", "fragCutoutPcf", "fragA2CPcf", "fragOpaquePcf"}},
      {{"fragOpaqueHardCsm2", "fragCutoutHardCsm2", "fragA2CHardCsm2", "fragOpaqueHardCsm2"},
       {"fragOpaquePcfCsm2", "fragCutoutPcfCsm2", "fragA2CPcfCsm2", "fragOpaquePcfCsm2"}},
  };
  return k_entries[cascade_mode == ShadowCascadeMode::Dual ? 1 : 0]
                  [filter == ShadowFilterMode::Pcf3x3 ? 1 : 0]
                  [static_cast<std::uint8_t>(alpha_mode)];
}

using AlphaModeSet = std::array<bool, k_alpha_mode_count>;

[[nodiscard]] inline auto collect_used_alpha_modes(const std::vector<MeshInstance> &instances) -> AlphaModeSet {
  AlphaModeSet used{};
  for (const MeshInstance &instance : instances) {
    if (instance.layer == RenderLayer::Background)
      continue;
    used[static_cast<std::size_t>(instance.alpha_mode)] = true;
  }
  if (std::ranges::none_of(used, [](bool b) { return b; }))
    used[0] = true;
  return used;
}

[[nodiscard]] inline auto scene_uses_alpha_to_coverage(const std::vector<MeshInstance> &instances) -> bool {
  return collect_used_alpha_modes(instances)[static_cast<std::size_t>(MeshAlphaMode::AlphaToCoverage)];
}

[[nodiscard]] constexpr auto is_textured_surface_pipeline(PipelineId id) -> bool {
  const auto v = static_cast<std::uint8_t>(id);
  return (v >= k_first_textured && v < k_first_textured + k_alpha_mode_count) ||
         (v >= k_first_textured_skinned && v < k_first_textured_skinned + k_alpha_mode_count) ||
         (v >= k_first_textured_terrain && v < k_first_textured_terrain + k_alpha_mode_count);
}

[[nodiscard]] constexpr auto is_blend_pipeline(PipelineId id) -> bool {
  return id == PipelineId::TexturedBlend || id == PipelineId::TexturedBlendSkinned ||
         id == PipelineId::TexturedBlendTerrain;
}

[[nodiscard]] constexpr auto is_skinned_pipeline(PipelineId id) -> bool {
  const auto v = static_cast<std::uint8_t>(id);
  return (v >= k_first_textured_skinned &&
          v <= static_cast<std::uint8_t>(PipelineId::ShadowDepthSkinned)) ||
         id == PipelineId::PointShadowDepthSkinned;
}

// Translucent surfaces do not cast shadows.
//
// Not a simplification -- it is what a shadow map can actually represent. The
// depth-only pass records "something is here", so a blended caster would throw
// the same hard black shadow a wall does, and a lake would sit in a lake-shaped
// hole of darkness. Skipping it is both more correct and one fewer draw.
[[nodiscard]] constexpr auto casts_shadow(PipelineId id) -> bool {
  return is_textured_surface_pipeline(id) && !is_blend_pipeline(id);
}

struct PipelineBuildProfile {
  ShadowFilterMode shadow_filter{ShadowFilterMode::Pcf3x3};
  ShadowCascadeMode cascade_mode{ShadowCascadeMode::Dual};
  AlphaModeSet textured_alpha_modes{{true, false, false, false}};
  // Which alpha modes the packed terrain format needs. All false = no terrain
  // pipelines at all; an app that never streams terrain pays nothing.
  AlphaModeSet terrain_alpha_modes{};
  bool build_skinned{false};
  bool has_point_shadows{false};

  [[nodiscard]] constexpr auto build_terrain() const -> bool {
    for (const bool used : terrain_alpha_modes)
      if (used)
        return true;
    return false;
  }
};

} // namespace engine
