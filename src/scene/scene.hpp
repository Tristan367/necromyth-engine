#pragma once

#include "scene/camera.hpp"
#include "scene/mesh_instance.hpp"
#include "scene/mesh_source.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace engine {

class Scene {
public:
  [[nodiscard]] auto camera() -> Camera & {
    return camera_;
  }

  [[nodiscard]] auto camera() const -> const Camera & {
    return camera_;
  }

  [[nodiscard]] auto meshes() const -> const std::vector<MeshSource> & {
    return meshes_;
  }

  [[nodiscard]] auto instances() const -> const std::vector<MeshInstance> & {
    return instances_;
  }

  [[nodiscard]] auto texture_paths() const -> const std::vector<std::string> & {
    return texture_paths_;
  }

  [[nodiscard]] auto texture_array_layer_paths() const -> const std::vector<std::string> & {
    return texture_array_layer_paths_;
  }

  [[nodiscard]] auto instance(std::uint32_t index) -> MeshInstance & {
    return instances_.at(index);
  }

  [[nodiscard]] auto add_mesh(MeshSource mesh) -> std::uint32_t {
    const std::uint32_t index = static_cast<std::uint32_t>(meshes_.size());
    meshes_.push_back(std::move(mesh));
    return index;
  }

  [[nodiscard]] auto add_texture(std::string path) -> std::uint32_t {
    const std::uint32_t index = static_cast<std::uint32_t>(texture_paths_.size());
    texture_paths_.push_back(std::move(path));
    return index;
  }

  [[nodiscard]] auto add_texture_array_layer(std::string path) -> std::uint32_t {
    const std::uint32_t index = static_cast<std::uint32_t>(texture_array_layer_paths_.size());
    texture_array_layer_paths_.push_back(std::move(path));
    return index;
  }

  [[nodiscard]] auto add_instance(MeshInstance instance) -> std::uint32_t {
    const std::uint32_t index = static_cast<std::uint32_t>(instances_.size());
    instances_.push_back(instance);
    return index;
  }

private:
  Camera camera_;
  std::vector<MeshSource> meshes_;
  std::vector<std::string> texture_paths_;
  std::vector<std::string> texture_array_layer_paths_;
  std::vector<MeshInstance> instances_;
};

} // namespace engine
