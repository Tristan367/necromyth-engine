#pragma once

#include "renderer/frustum.hpp"
#include "scene/point_light.hpp"
#include "scene/spot_light.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace engine {

inline constexpr std::int32_t k_no_shadow_slot = -1;

// Which lights get a shadow map this frame, and where.
//
// The cubemap layer used to be implied by the light's position in the scene
// array. That coupling had two consequences: a shadow-casting light past
// max_point_shadow_lights was skipped by the renderer while the shader kept
// sampling its (nonexistent) layer, and no light could ever be skipped for being
// irrelevant without the shader reading a stale layer underneath it.
//
// Making the slot explicit -- assigned here, written into the light buffer, read
// by the shader -- decouples the two, so the renderer is free to render only the
// lights that can actually affect the image.
struct ShadowSlotAssignment {
  // Indexed by scene light index; k_no_shadow_slot when the light gets no map.
  std::vector<std::int32_t> point_slots;
  std::vector<std::int32_t> spot_slots;
  std::uint32_t point_count{0};
  std::uint32_t spot_count{0};
};

// A light can be ignored entirely when nothing it could illuminate is on screen.
// Attenuation reaches zero at `range`, so if the sphere of influence misses the
// camera frustum, no visible pixel receives light from it -- and therefore no
// shadow map is needed either. This is a much stronger filter than it looks:
// a scene lit by lamps throughout a level only ever has a handful of them
// touching the view at once.
[[nodiscard]] inline auto light_affects_view(const glm::vec3 &position, float range,
                                             const Frustum &camera_frustum) -> bool {
  return range > 0.0F && camera_frustum.intersects_sphere(position, range);
}

// Assigns shadow slots in scene order, nearest-first would be better once there
// are more shadow casters than slots; scene order is predictable and stable,
// which matters more while the counts are small.
[[nodiscard]] inline auto assign_shadow_slots(
    const std::vector<PointLight> &point_lights,
    const std::vector<SpotLight> &spot_lights,
    const Frustum &camera_frustum,
    std::uint32_t max_point_shadows,
    std::uint32_t max_spot_shadows) -> ShadowSlotAssignment {
  ShadowSlotAssignment assignment;
  assignment.point_slots.assign(point_lights.size(), k_no_shadow_slot);
  assignment.spot_slots.assign(spot_lights.size(), k_no_shadow_slot);

  for (std::size_t i = 0; i < point_lights.size(); ++i) {
    const PointLight &light = point_lights[i];
    if (!light.casts_shadow || light.intensity <= 0.0F)
      continue;
    if (!light_affects_view(light.position, light.range, camera_frustum))
      continue;
    if (assignment.point_count >= max_point_shadows)
      break;
    assignment.point_slots[i] = static_cast<std::int32_t>(assignment.point_count++);
  }

  for (std::size_t i = 0; i < spot_lights.size(); ++i) {
    const SpotLight &light = spot_lights[i];
    if (!light.casts_shadow || light.intensity <= 0.0F)
      continue;
    if (!light_affects_view(light.position, light.range, camera_frustum))
      continue;
    if (assignment.spot_count >= max_spot_shadows)
      break;
    assignment.spot_slots[i] = static_cast<std::int32_t>(assignment.spot_count++);
  }

  return assignment;
}

} // namespace engine
