#pragma once

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
  PipelineId pipeline{PipelineId::TexturedMesh};
};

inline void build_draw_list(const Scene &scene, std::vector<DrawCommand> &out) {
  out.clear();
  out.reserve(scene.instances().size());

  for (const MeshInstance &instance : scene.instances()) {
    out.push_back({
        .mesh_index = instance.mesh_index,
        .texture_index = instance.texture_index,
        .texture_source = instance.texture_source,
        .model = instance.model,
        .layer = instance.layer,
        .pipeline = instance.pipeline,
    });
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
    return a.mesh_index < b.mesh_index;
  });
}

// Shadow depth only needs mesh + model matrix. Sort by layer then mesh to reuse vertex/index buffers
// (Sascha gltfscenerendering binds geometry once per mesh; main pass keeps texture-first order above).
inline void build_shadow_draw_list(const std::vector<DrawCommand> &draw_list, std::vector<DrawCommand> &out) {
  out.clear();
  out.reserve(draw_list.size());

  for (const DrawCommand &draw : draw_list) {
    if (draw.pipeline == PipelineId::TexturedMesh)
      out.push_back(draw);
  }

  std::ranges::sort(out, [](const DrawCommand &a, const DrawCommand &b) {
    if (a.layer != b.layer)
      return a.layer < b.layer;
    return a.mesh_index < b.mesh_index;
  });
}

} // namespace engine
