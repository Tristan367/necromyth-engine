#pragma once

#include "renderer/pipeline_id.hpp"
#include "scene/render_layer.hpp"

#include <glm/mat4x4.hpp>

#include <cstdint>

namespace engine {

struct MeshInstance {
  std::uint32_t mesh_index{0};
  std::uint32_t texture_index{0};
  glm::mat4 model{1.0F};
  RenderLayer layer{RenderLayer::Opaque};
  PipelineId pipeline{PipelineId::TexturedMesh};
};

} // namespace engine
