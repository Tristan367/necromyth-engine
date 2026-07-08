#pragma once

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
};

inline void build_draw_list(const Scene &scene, std::vector<DrawCommand> &out) {
  out.clear();
  out.reserve(scene.instances().size());

  std::uint32_t bone_instance_count = 0;
  for (const MeshInstance &instance : scene.instances()) {
    if (!instance.alive) continue;

    const bool has_valid_skin = instance.skin_index != k_invalid_skin_index
        && instance.skin_index < scene.skeletons().size()
        && !scene.skeletons()[instance.skin_index].joint_nodes.empty();
    const PipelineId pipeline = instance.layer == RenderLayer::Background
        ? PipelineId::Background
        : textured_pipeline(instance.alpha_mode, has_valid_skin);

    const std::uint32_t bone_index = has_valid_skin ? bone_instance_count : k_invalid_skin_index;

    out.push_back({
        .mesh_index = instance.mesh_index,
        .texture_index = instance.texture_index,
        .texture_source = instance.texture_source,
        .model = instance.model,
        .layer = instance.layer,
        .pipeline = pipeline,
        .skin_index = instance.skin_index,
        .bone_instance_index = bone_index,
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
