#pragma once

#include "renderer/vertex.hpp"

#include <cstdint>
#include <vector>

namespace engine {

struct MeshSource {
  std::vector<MeshVertex> vertices;
  std::vector<std::uint32_t> indices;
};

} // namespace engine
