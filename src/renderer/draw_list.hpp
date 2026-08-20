#pragma once

#include "renderer/frustum.hpp"
#include "renderer/mesh_gpu.hpp"
#include "renderer/pipeline_id.hpp"
#include "scene/mesh_instance.hpp"
#include "scene/scene.hpp"

#include <glm/mat4x4.hpp>

#include <algorithm>
#include <span>
#include <vector>

namespace engine {

struct DrawCommand {
  std::uint32_t mesh_index{};
  std::uint32_t texture_index{};
  TextureSource texture_source{TextureSource::Table};
  glm::mat4 model{1.0F};
  RenderLayer layer{RenderLayer::Opaque};
  PipelineId pipeline{PipelineId::TexturedOpaque};
  std::uint32_t skin_index{k_invalid_skin_index};
  std::uint32_t bone_instance_index{k_invalid_skin_index};
  // World-space bounds, computed once here and reused by every culling pass.
  // radius == 0 means "bounds unknown" and disables culling for this draw.
  BoundingSphere world_bounds{};
};

// World-space bounding sphere of a mesh AABB under a model transform.
//
// Radius uses the per-axis scaled half-extent rather than scaling the AABB's own
// radius, so a non-uniformly scaled mesh gets a tight sphere instead of one
// inflated by its largest axis.
[[nodiscard]] inline auto world_bounding_sphere(const AABB &bounds, const glm::mat4 &model)
    -> BoundingSphere {
  const glm::vec3 half_extent = (bounds.max - bounds.min) * 0.5F;
  return {
      .center = glm::vec3(model * glm::vec4(bounds.center(), 1.0F)),
      .radius = glm::length(glm::vec3(
          glm::length(glm::vec3(model[0])) * half_extent.x,
          glm::length(glm::vec3(model[1])) * half_extent.y,
          glm::length(glm::vec3(model[2])) * half_extent.z)),
  };
}

inline void build_draw_list(const Scene &scene, std::span<const MeshGpu> mesh_gpus,
                            std::vector<DrawCommand> &out) {
  out.clear();
  out.reserve(scene.instances().size());

  std::uint32_t bone_instance_count = 0;
  for (const MeshInstance &instance : scene.instances()) {
    if (!instance.alive) continue;

    // Must match every other bone-slot walker exactly — see instance_uses_skinning().
    const bool has_valid_skin = instance_uses_skinning(instance, scene);
    const PipelineId pipeline = instance.layer == RenderLayer::Background
        ? PipelineId::Background
        : textured_pipeline(instance.alpha_mode, has_valid_skin);

    const std::uint32_t bone_index = has_valid_skin ? bone_instance_count : k_invalid_skin_index;

    // Skinned meshes animate outside their bind-pose AABB, so their bounds are
    // not a reliable cull volume; leave the radius at 0 (never culled) rather
    // than pop limbs at the screen edge. Background geometry is never culled
    // either -- the sky is drawn with a translation-free view matrix.
    BoundingSphere world_bounds{};
    if (!has_valid_skin && instance.layer != RenderLayer::Background &&
        instance.mesh_index < mesh_gpus.size())
      world_bounds = world_bounding_sphere(mesh_gpus[instance.mesh_index].bounds(), instance.model);

    out.push_back({
        .mesh_index = instance.mesh_index,
        .texture_index = instance.texture_index,
        .texture_source = instance.texture_source,
        .model = instance.model,
        .layer = instance.layer,
        .pipeline = pipeline,
        .skin_index = instance.skin_index,
        .bone_instance_index = bone_index,
        .world_bounds = world_bounds,
    });

    if (has_valid_skin)
      ++bone_instance_count;
  }

  std::ranges::sort(out, [](const DrawCommand &a, const DrawCommand &b) {
    if (a.layer != b.layer)
      return a.layer < b.layer;
    if (a.pipeline != b.pipeline)
      return a.pipeline < b.pipeline;
    if (a.texture_source != b.texture_source)
      return a.texture_source < b.texture_source;
    if (a.texture_index != b.texture_index)
      return a.texture_index < b.texture_index;
    if (a.bone_instance_index != b.bone_instance_index)
      return a.bone_instance_index < b.bone_instance_index;
    return a.mesh_index < b.mesh_index;
  });
}

// Shadow casters: all textured surface pipelines (opaque silhouettes for cutout/A2C until alpha-tested shadow FS).
inline void build_shadow_draw_list(const std::vector<DrawCommand> &draw_list, std::vector<DrawCommand> &out) {
  out.clear();
  out.reserve(draw_list.size());

  for (const DrawCommand &draw : draw_list) {
    if (casts_shadow(draw.pipeline))
      out.push_back(draw);
  }

  std::ranges::sort(out, [](const DrawCommand &a, const DrawCommand &b) {
    if (a.layer != b.layer)
      return a.layer < b.layer;
    return a.mesh_index < b.mesh_index;
  });
}

} // namespace engine
