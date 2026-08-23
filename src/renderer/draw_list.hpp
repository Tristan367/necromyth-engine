#pragma once

#include "renderer/frustum.hpp"
#include "renderer/pipeline_id.hpp"
#include "scene/mesh_instance.hpp"
#include "scene/scene.hpp"

#include <glm/mat4x4.hpp>

#include <algorithm>
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
  // Index of this draw's record in the frame's instance buffer. Assigned when
  // the records are written, which happens per list -- so a draw's index in the
  // main list and in the shadow list differ.
  std::uint32_t instance_index{0};
};

// True when `b` can be appended to a batch ending at `a`: same pipeline, same
// mesh, same material binding, and adjacent instance records so one drawIndexed
// with instanceCount > 1 covers both.
//
// The instance-record check is what makes this safe rather than clever: the
// vertex shader reads record `instanceBase + SV_InstanceID`, so a gap in the
// indices would silently draw the wrong objects.
[[nodiscard]] inline auto draws_can_batch(const DrawCommand &a, const DrawCommand &b) -> bool {
  return a.pipeline == b.pipeline
      && a.mesh_index == b.mesh_index
      && a.texture_source == b.texture_source
      && a.texture_index == b.texture_index
      && b.instance_index == a.instance_index + 1;
}

// The same question for a depth-only pass, which is a weaker one.
//
// draw_shadow_mesh binds a pipeline chosen purely by skinned-ness, the bone set,
// the mesh, and push constants. It binds no material and reads no texture, and
// the shadow pass draws cutout and alpha-to-coverage surfaces as opaque
// silhouettes anyway. So neither the texture nor the alpha mode can change what
// gets recorded, and requiring them to match -- which the shared rule above does
// -- split batches on a difference the pass cannot observe.
//
// Two instances of one mesh wearing different textures are one shadow draw.
[[nodiscard]] inline auto shadow_draws_can_batch(const DrawCommand &a, const DrawCommand &b) -> bool {
  return is_skinned_pipeline(a.pipeline) == is_skinned_pipeline(b.pipeline)
      && a.mesh_index == b.mesh_index
      && b.instance_index == a.instance_index + 1;
}

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

inline void build_draw_list(const Scene &scene, std::vector<DrawCommand> &out) {
  out.clear();
  out.reserve(scene.instances().size());

  std::uint32_t bone_instance_count = 0;
  for (const MeshInstance &instance : scene.instances()) {
    if (!instance.alive) continue;
    // An instance left pointing at a freed mesh slot (a chunk that streamed out
    // before its instances were retired) draws nothing rather than reading a
    // recycled or destroyed buffer.
    if (!scene.mesh_alive(instance.mesh_index)) continue;

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
    if (!has_valid_skin && instance.layer != RenderLayer::Background) {
      const AABB &bounds = scene.mesh_bounds(instance.mesh_index);
      if (!bounds.empty())
        world_bounds = world_bounding_sphere(bounds, instance.model);
    }

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

  // Sort so that batchable draws end up adjacent: layer, then pipeline, then
  // material, then mesh. Mesh now outranks the bone slot, because the bone slice
  // is selected per instance inside the shader and no longer forces one draw per
  // character -- a horde sharing a mesh and texture becomes one draw call.
  std::ranges::sort(out, [](const DrawCommand &a, const DrawCommand &b) {
    if (a.layer != b.layer)
      return a.layer < b.layer;
    if (a.pipeline != b.pipeline)
      return a.pipeline < b.pipeline;
    if (a.texture_source != b.texture_source)
      return a.texture_source < b.texture_source;
    if (a.texture_index != b.texture_index)
      return a.texture_index < b.texture_index;
    if (a.mesh_index != b.mesh_index)
      return a.mesh_index < b.mesh_index;
    return a.bone_instance_index < b.bone_instance_index;
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

  // Sorted by exactly what shadow_draws_can_batch compares, and nothing else.
  //
  // The sort and the batch rule have to agree: a batch is a run of ADJACENT
  // draws, so sorting on a key the rule ignores (texture, alpha mode) scatters
  // draws that could have merged, and sorting on less than the rule needs would
  // merge draws that must not. This used to sort by layer and full pipeline id,
  // which separated opaque from cutout instances of the same mesh even though
  // the depth pass records them identically.
  std::ranges::sort(out, [](const DrawCommand &a, const DrawCommand &b) {
    const bool a_skinned = is_skinned_pipeline(a.pipeline);
    const bool b_skinned = is_skinned_pipeline(b.pipeline);
    if (a_skinned != b_skinned)
      return static_cast<int>(a_skinned) < static_cast<int>(b_skinned);
    if (a.mesh_index != b.mesh_index)
      return a.mesh_index < b.mesh_index;
    return a.bone_instance_index < b.bone_instance_index;
  });
}

} // namespace engine
