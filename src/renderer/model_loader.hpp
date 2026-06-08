#pragma once

#include "renderer/vertex.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine {

namespace detail {

struct MeshVertexHash {
  [[nodiscard]] auto operator()(const MeshVertex &vertex) const noexcept -> std::size_t {
    std::size_t hash = 0;
    for (float component : vertex.pos)
      hash ^= std::hash<float>{}(component) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
    for (float component : vertex.normal)
      hash ^= std::hash<float>{}(component) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
    for (float component : vertex.color)
      hash ^= std::hash<float>{}(component) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
    for (float component : vertex.tex_coord)
      hash ^= std::hash<float>{}(component) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
    return hash;
  }
};

} // namespace detail

struct LoadedMesh {
  std::vector<MeshVertex> vertices;
  std::vector<std::uint32_t> indices;
};

[[nodiscard]] inline auto load_obj_model(std::string_view path) -> LoadedMesh {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn;
  std::string err;

  if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, std::string(path).c_str()))
    throw std::runtime_error(warn + err);

  LoadedMesh mesh;
  std::unordered_map<MeshVertex, std::uint32_t, detail::MeshVertexHash> unique_vertices;

  for (const auto &shape : shapes) {
    for (const auto &index : shape.mesh.indices) {
      MeshVertex vertex{};

      vertex.pos[0] = attrib.vertices[3 * index.vertex_index + 0];
      vertex.pos[1] = attrib.vertices[3 * index.vertex_index + 1];
      vertex.pos[2] = attrib.vertices[3 * index.vertex_index + 2];

      if (index.normal_index >= 0) {
        vertex.normal[0] = attrib.normals[3 * index.normal_index + 0];
        vertex.normal[1] = attrib.normals[3 * index.normal_index + 1];
        vertex.normal[2] = attrib.normals[3 * index.normal_index + 2];
      } else {
        vertex.normal[0] = 0.0F;
        vertex.normal[1] = 1.0F;
        vertex.normal[2] = 0.0F;
      }

      vertex.tex_coord[0] = attrib.texcoords[2 * index.texcoord_index + 0];
      vertex.tex_coord[1] = 1.0F - attrib.texcoords[2 * index.texcoord_index + 1];

      vertex.color[0] = 1.0F;
      vertex.color[1] = 1.0F;
      vertex.color[2] = 1.0F;

      const auto [iterator, inserted] = unique_vertices.insert({vertex, static_cast<std::uint32_t>(mesh.vertices.size())});
      if (inserted)
        mesh.vertices.push_back(vertex);

      mesh.indices.push_back(iterator->second);
    }
  }

  return mesh;
}

} // namespace engine
