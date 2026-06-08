#pragma once

#include "renderer/model_loader.hpp"

#define GLM_FORCE_RADIANS
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat4x4.hpp>

#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#define TINYGLTF_IMPLEMENTATION
#include <tinygltf/tiny_gltf.h>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace engine {

struct LoadedGltfMaterial {
  std::optional<std::string> base_color_texture_path;
  std::array<float, 4> base_color_factor{1.0F, 1.0F, 1.0F, 1.0F};
};

struct LoadedGltfPrimitive {
  LoadedMesh mesh;
  LoadedGltfMaterial material;
  glm::mat4 node_transform{1.0F};
};

struct LoadedGltfModel {
  std::string base_directory;
  std::vector<LoadedGltfPrimitive> primitives;
};

namespace detail {

inline auto parent_directory(std::string_view path) -> std::string {
  const std::filesystem::path file_path(path);
  if (file_path.has_parent_path())
    return file_path.parent_path().string();
  return ".";
}

inline auto resolve_uri(std::string_view base_directory, std::string_view uri) -> std::string {
  const std::filesystem::path uri_path(uri);
  if (uri_path.is_absolute())
    return uri_path.string();

  std::filesystem::path resolved = std::filesystem::path(base_directory) / uri_path;
  resolved = resolved.lexically_normal();
  return resolved.string();
}

inline auto decode_gltf_image_data(
    tinygltf::Image *image,
    int /*image_index*/,
    std::string *error,
    std::string * /*warning*/,
    int /*req_width*/,
    int /*req_height*/,
    const unsigned char *bytes,
    int size,
    void * /*user_data*/) -> bool {
  if (image == nullptr || bytes == nullptr || size <= 0) {
    if (error != nullptr)
      *error += "Invalid glTF embedded image payload\n";
    return false;
  }

  std::int32_t width{};
  std::int32_t height{};
  std::int32_t channels{};
  stbi_uc *pixels = stbi_load_from_memory(bytes, size, &width, &height, &channels, STBI_rgb_alpha);
  if (pixels == nullptr) {
    if (error != nullptr)
      *error += "stbi_load_from_memory failed for embedded glTF image\n";
    return false;
  }

  image->width = width;
  image->height = height;
  image->component = 4;
  image->image.assign(pixels, pixels + static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4U);
  stbi_image_free(pixels);
  return true;
}

inline void materialize_embedded_images(tinygltf::Model &model, std::string_view source_path) {
  const std::filesystem::path source_file(source_path);
  const std::filesystem::path cache_dir = source_file.parent_path() / ".gltf_cache";
  std::filesystem::create_directories(cache_dir);
  const std::string stem = source_file.stem().string();

  for (std::size_t image_index = 0; image_index < model.images.size(); ++image_index) {
    tinygltf::Image &image = model.images[image_index];
    if (!image.uri.empty() || image.image.empty())
      continue;

    const std::filesystem::path cache_path =
        cache_dir / (stem + "_image" + std::to_string(image_index) + ".png");
    if (!std::filesystem::exists(cache_path)) {
      if (stbi_write_png(
              cache_path.string().c_str(),
              image.width,
              image.height,
              4,
              image.image.data(),
              image.width * 4) == 0)
        throw std::runtime_error("Failed to write embedded glTF image cache: " + cache_path.string());
    }

    image.uri = cache_path.string();
  }
}

inline auto node_local_matrix(const tinygltf::Node &node) -> glm::mat4 {
  if (node.matrix.size() == 16)
    return glm::make_mat4x4(node.matrix.data());

  glm::mat4 matrix(1.0F);
  if (node.translation.size() == 3) {
    const glm::vec3 translation{
        static_cast<float>(node.translation[0]),
        static_cast<float>(node.translation[1]),
        static_cast<float>(node.translation[2]),
    };
    matrix = glm::translate(matrix, translation);
  }
  if (node.rotation.size() == 4) {
    const glm::quat rotation{
        static_cast<float>(node.rotation[3]),
        static_cast<float>(node.rotation[0]),
        static_cast<float>(node.rotation[1]),
        static_cast<float>(node.rotation[2]),
    };
    matrix *= glm::mat4_cast(rotation);
  }
  if (node.scale.size() == 3) {
    const glm::vec3 scale{
        static_cast<float>(node.scale[0]),
        static_cast<float>(node.scale[1]),
        static_cast<float>(node.scale[2]),
    };
    matrix = glm::scale(matrix, scale);
  }
  return matrix;
}

inline void read_float_accessor(
    const tinygltf::Model &model,
    int accessor_index,
    std::uint32_t components,
    std::vector<float> &out) {
  if (accessor_index < 0)
    throw std::runtime_error("Missing glTF accessor");

  const tinygltf::Accessor &accessor = model.accessors[static_cast<std::size_t>(accessor_index)];
  const tinygltf::BufferView &view = model.bufferViews[static_cast<std::size_t>(accessor.bufferView)];
  const tinygltf::Buffer &buffer = model.buffers[static_cast<std::size_t>(view.buffer)];

  if (accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)
    throw std::runtime_error("Expected float accessor data in glTF mesh");

  const std::size_t stride = accessor.ByteStride(view);
  const std::size_t element_bytes = components * sizeof(float);
  out.resize(accessor.count * components);

  for (std::size_t index = 0; index < accessor.count; ++index) {
    const std::size_t offset = static_cast<std::size_t>(accessor.byteOffset + view.byteOffset + index * stride);
    std::memcpy(&out[index * components], buffer.data.data() + offset, element_bytes);
  }
}

inline void read_indices(
    const tinygltf::Model &model,
    int accessor_index,
    std::uint32_t vertex_offset,
    std::vector<std::uint32_t> &out) {
  const tinygltf::Accessor &accessor = model.accessors[static_cast<std::size_t>(accessor_index)];
  const tinygltf::BufferView &view = model.bufferViews[static_cast<std::size_t>(accessor.bufferView)];
  const tinygltf::Buffer &buffer = model.buffers[static_cast<std::size_t>(view.buffer)];
  const std::size_t stride = accessor.ByteStride(view);
  const std::uint8_t *base = buffer.data.data() + accessor.byteOffset + view.byteOffset;

  out.reserve(out.size() + accessor.count);

  for (std::size_t index = 0; index < accessor.count; ++index) {
    const std::uint8_t *element = base + index * (stride == 0 ? accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT
                                                                                    ? sizeof(std::uint32_t)
                                                                                    : accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT
                                                                                          ? sizeof(std::uint16_t)
                                                                                          : sizeof(std::uint8_t)
                                                                : stride);
    std::uint32_t value{};
    switch (accessor.componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
      value = *reinterpret_cast<const std::uint32_t *>(element);
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
      value = *reinterpret_cast<const std::uint16_t *>(element);
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
      value = *reinterpret_cast<const std::uint8_t *>(element);
      break;
    default:
      throw std::runtime_error("Unsupported glTF index component type");
    }
    out.push_back(value + vertex_offset);
  }
}

inline auto material_for_primitive(const tinygltf::Model &model, int material_index, std::string_view base_directory)
    -> LoadedGltfMaterial {
  LoadedGltfMaterial material;
  if (material_index < 0)
    return material;

  const tinygltf::Material &gltf_material = model.materials[static_cast<std::size_t>(material_index)];
  const auto &pbr = gltf_material.pbrMetallicRoughness;
  if (pbr.baseColorFactor.size() == 4) {
    material.base_color_factor = {
        static_cast<float>(pbr.baseColorFactor[0]),
        static_cast<float>(pbr.baseColorFactor[1]),
        static_cast<float>(pbr.baseColorFactor[2]),
        static_cast<float>(pbr.baseColorFactor[3]),
    };
  }

  if (pbr.baseColorTexture.index < 0)
    return material;

  const tinygltf::Texture &texture = model.textures[static_cast<std::size_t>(pbr.baseColorTexture.index)];
  const tinygltf::Image &image = model.images[static_cast<std::size_t>(texture.source)];
  if (!image.uri.empty())
    material.base_color_texture_path = resolve_uri(base_directory, image.uri);

  return material;
}

inline void load_primitive(
    const tinygltf::Model &model,
    const tinygltf::Primitive &primitive,
    const glm::mat4 &node_transform,
    std::string_view base_directory,
    std::vector<LoadedGltfPrimitive> &out) {
  const auto position_it = primitive.attributes.find("POSITION");
  if (position_it == primitive.attributes.end())
    throw std::runtime_error("glTF primitive is missing POSITION attribute");
  if (primitive.indices < 0)
    throw std::runtime_error("glTF primitive is missing indices");

  std::vector<float> positions;
  read_float_accessor(model, position_it->second, 3, positions);
  const std::uint32_t vertex_count = static_cast<std::uint32_t>(positions.size() / 3);

  std::vector<float> texcoords;
  if (const auto uv_it = primitive.attributes.find("TEXCOORD_0"); uv_it != primitive.attributes.end())
    read_float_accessor(model, uv_it->second, 2, texcoords);

  std::vector<float> normals;
  if (const auto normal_it = primitive.attributes.find("NORMAL"); normal_it != primitive.attributes.end())
    read_float_accessor(model, normal_it->second, 3, normals);

  std::vector<float> colors;
  std::uint32_t color_components = 0;
  if (const auto color_it = primitive.attributes.find("COLOR_0"); color_it != primitive.attributes.end()) {
    const tinygltf::Accessor &accessor = model.accessors[static_cast<std::size_t>(color_it->second)];
    color_components = accessor.type == TINYGLTF_TYPE_VEC3 ? 3U : 4U;
    read_float_accessor(model, color_it->second, color_components, colors);
  }

  LoadedGltfPrimitive loaded{
      .mesh = {},
      .material = material_for_primitive(model, primitive.material, base_directory),
      .node_transform = node_transform,
  };
  loaded.mesh.vertices.reserve(vertex_count);
  loaded.mesh.indices.clear();

  for (std::uint32_t vertex_index = 0; vertex_index < vertex_count; ++vertex_index) {
    MeshVertex vertex{};
    vertex.pos[0] = positions[vertex_index * 3 + 0];
    vertex.pos[1] = positions[vertex_index * 3 + 1];
    vertex.pos[2] = positions[vertex_index * 3 + 2];

    if (!normals.empty()) {
      vertex.normal[0] = normals[vertex_index * 3 + 0];
      vertex.normal[1] = normals[vertex_index * 3 + 1];
      vertex.normal[2] = normals[vertex_index * 3 + 2];
    } else {
      vertex.normal[0] = 0.0F;
      vertex.normal[1] = 1.0F;
      vertex.normal[2] = 0.0F;
    }

    if (!texcoords.empty()) {
      vertex.tex_coord[0] = texcoords[vertex_index * 2 + 0];
      vertex.tex_coord[1] = texcoords[vertex_index * 2 + 1];
    }

    if (!colors.empty()) {
      vertex.color[0] = colors[vertex_index * color_components + 0];
      vertex.color[1] = colors[vertex_index * color_components + 1];
      vertex.color[2] = color_components == 3 ? 1.0F : colors[vertex_index * color_components + 2];
    } else if (loaded.material.base_color_factor[3] < 1.0F) {
      vertex.color[0] = loaded.material.base_color_factor[0];
      vertex.color[1] = loaded.material.base_color_factor[1];
      vertex.color[2] = loaded.material.base_color_factor[2];
    } else {
      vertex.color[0] = 1.0F;
      vertex.color[1] = 1.0F;
      vertex.color[2] = 1.0F;
    }

    loaded.mesh.vertices.push_back(vertex);
  }

  read_indices(model, primitive.indices, 0, loaded.mesh.indices);

  if (loaded.mesh.vertices.empty() || loaded.mesh.indices.empty())
    return;

  out.push_back(std::move(loaded));
}

inline void load_node(
    const tinygltf::Model &model,
    int node_index,
    const glm::mat4 &parent_transform,
    std::string_view base_directory,
    std::vector<LoadedGltfPrimitive> &out) {
  if (node_index < 0 || node_index >= static_cast<int>(model.nodes.size()))
    return;

  const tinygltf::Node &node = model.nodes[static_cast<std::size_t>(node_index)];
  const glm::mat4 world_transform = parent_transform * node_local_matrix(node);

  if (node.mesh >= 0) {
    const tinygltf::Mesh &mesh = model.meshes[static_cast<std::size_t>(node.mesh)];
    for (const tinygltf::Primitive &primitive : mesh.primitives)
      load_primitive(model, primitive, world_transform, base_directory, out);
  }

  for (int child_index : node.children)
    load_node(model, child_index, world_transform, base_directory, out);
}

} // namespace detail

[[nodiscard]] inline auto load_gltf_model(std::string_view path) -> LoadedGltfModel {
  tinygltf::TinyGLTF loader;
  loader.SetImageLoader(detail::decode_gltf_image_data, nullptr);
  tinygltf::Model model;
  std::string error;
  std::string warning;

  const std::string file_path(path);
  const bool loaded = file_path.ends_with(".glb") || file_path.ends_with(".GLB")
                          ? loader.LoadBinaryFromFile(&model, &error, &warning, file_path)
                          : loader.LoadASCIIFromFile(&model, &error, &warning, file_path);

  if (!warning.empty())
    (void)warning;
  if (!loaded)
    throw std::runtime_error("Failed to load glTF: " + error);

  detail::materialize_embedded_images(model, path);

  LoadedGltfModel result{
      .base_directory = detail::parent_directory(path),
      .primitives = {},
  };

  const int scene_index = model.defaultScene >= 0 ? model.defaultScene : 0;
  if (model.scenes.empty())
    throw std::runtime_error("glTF file contains no scenes");

  const tinygltf::Scene &scene = model.scenes[static_cast<std::size_t>(scene_index)];
  for (int node_index : scene.nodes)
    detail::load_node(model, node_index, glm::mat4(1.0F), result.base_directory, result.primitives);

  if (result.primitives.empty())
    throw std::runtime_error("glTF file contains no renderable mesh primitives");

  return result;
}

} // namespace engine
