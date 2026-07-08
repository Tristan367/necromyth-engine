#pragma once

#include "renderer/model_loader.hpp"
#include "scene/animation_types.hpp"
#include "scene/scene.hpp"

#define GLM_FORCE_RADIANS
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat4x4.hpp>

#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcpp"
#endif
#include <tinygltf/tiny_gltf.h>
#include <iostream>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

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
  std::int32_t skin_index{-1};
};

struct LoadedGltfModel {
  std::string base_directory;
  std::vector<LoadedGltfPrimitive> primitives;
  std::vector<SkeletonAsset> skeletons;
  std::vector<AnimationClip> animations;
  std::vector<std::uint32_t> node_parents;
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

inline void read_joint_accessor(
    const tinygltf::Model &model,
    int accessor_index,
    std::uint32_t components,
    std::vector<std::uint16_t> &out) {
  if (accessor_index < 0)
    throw std::runtime_error("Missing glTF accessor");

  const tinygltf::Accessor &accessor = model.accessors[static_cast<std::size_t>(accessor_index)];
  const tinygltf::BufferView &view = model.bufferViews[static_cast<std::size_t>(accessor.bufferView)];
  const tinygltf::Buffer &buffer = model.buffers[static_cast<std::size_t>(view.buffer)];

  const std::size_t stride = accessor.ByteStride(view);
  out.resize(accessor.count * components);

  for (std::size_t index = 0; index < accessor.count; ++index) {
    const std::size_t offset = static_cast<std::size_t>(accessor.byteOffset + view.byteOffset + index * stride);
    for (std::uint32_t c = 0; c < components; ++c) {
      if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
        const std::uint8_t value = *reinterpret_cast<const std::uint8_t *>(
            buffer.data.data() + offset + c * sizeof(std::uint8_t));
        out[index * components + c] = static_cast<std::uint16_t>(value);
      } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
        std::memcpy(&out[index * components + c],
                    buffer.data.data() + offset + c * sizeof(std::uint16_t),
                    sizeof(std::uint16_t));
      } else {
        throw std::runtime_error("Expected unsigned byte or short accessor data in glTF joints");
      }
    }
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
    std::int32_t skin_index,
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

    std::vector<std::uint16_t> joints;
    if (const auto joint_it = primitive.attributes.find("JOINTS_0"); joint_it != primitive.attributes.end())
      read_joint_accessor(model, joint_it->second, 4, joints);

  std::vector<float> weights;
  if (const auto weight_it = primitive.attributes.find("WEIGHTS_0"); weight_it != primitive.attributes.end())
    read_float_accessor(model, weight_it->second, 4, weights);

  LoadedGltfPrimitive loaded{
      .mesh = {},
      .material = material_for_primitive(model, primitive.material, base_directory),
      .node_transform = node_transform,
      .skin_index = skin_index,
  };
  loaded.mesh.vertices.reserve(vertex_count);
  loaded.mesh.indices.clear();

  const bool has_joints = joints.size() >= vertex_count * 4;
  const bool has_weights = weights.size() >= vertex_count * 4;

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

    if (has_joints) {
      vertex.joint_indices[0] = static_cast<float>(joints[vertex_index * 4 + 0]);
      vertex.joint_indices[1] = static_cast<float>(joints[vertex_index * 4 + 1]);
      vertex.joint_indices[2] = static_cast<float>(joints[vertex_index * 4 + 2]);
      vertex.joint_indices[3] = static_cast<float>(joints[vertex_index * 4 + 3]);
    }

    if (has_weights) {
      vertex.joint_weights[0] = weights[vertex_index * 4 + 0];
      vertex.joint_weights[1] = weights[vertex_index * 4 + 1];
      vertex.joint_weights[2] = weights[vertex_index * 4 + 2];
      vertex.joint_weights[3] = weights[vertex_index * 4 + 3];
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
    const std::int32_t skin_index = node.skin;
    // Skinned meshes: don't bake the accumulated world transform — bone matrices handle placement.
    const glm::mat4 transform_for_primitive = skin_index >= 0 ? glm::mat4(1.0F) : world_transform;
    const tinygltf::Mesh &mesh = model.meshes[static_cast<std::size_t>(node.mesh)];
    for (const tinygltf::Primitive &primitive : mesh.primitives)
      load_primitive(model, primitive, transform_for_primitive, skin_index, base_directory, out);
  }

  for (int child_index : node.children)
    load_node(model, child_index, world_transform, base_directory, out);
}

inline void load_skeletons(
    const tinygltf::Model &model,
    std::vector<SkeletonAsset> &out_skeletons,
    const std::vector<std::uint32_t> &node_parents) {
  out_skeletons.clear();
  out_skeletons.reserve(model.skins.size());

  for (const tinygltf::Skin &gltf_skin : model.skins) {
    SkeletonAsset skeleton{};
    skeleton.joint_nodes.reserve(gltf_skin.joints.size());
    for (const int joint_index : gltf_skin.joints) {
      skeleton.joint_nodes.push_back(static_cast<std::uint32_t>(joint_index));
      const std::size_t ni = static_cast<std::size_t>(joint_index);
      skeleton.joint_names.push_back(ni < model.nodes.size() ? model.nodes[ni].name : "");
    }
    skeleton.skeleton_root = gltf_skin.skeleton >= 0
        ? static_cast<std::uint32_t>(gltf_skin.skeleton)
        : std::numeric_limits<std::uint32_t>::max();
    skeleton.node_parents = node_parents;

    if (gltf_skin.inverseBindMatrices >= 0) {
      const tinygltf::Accessor &accessor = model.accessors[static_cast<std::size_t>(gltf_skin.inverseBindMatrices)];
      const tinygltf::BufferView &view = model.bufferViews[static_cast<std::size_t>(accessor.bufferView)];
      const tinygltf::Buffer &buffer = model.buffers[static_cast<std::size_t>(view.buffer)];

      skeleton.inverse_bind_matrices.resize(accessor.count);
      const std::size_t stride = accessor.ByteStride(view);
      constexpr std::size_t k_mat4_bytes = sizeof(glm::mat4);

      for (std::size_t i = 0; i < accessor.count; ++i) {
        const std::size_t offset =
            static_cast<std::size_t>(accessor.byteOffset + view.byteOffset + i * (stride == 0 ? k_mat4_bytes : stride));
        std::memcpy(&skeleton.inverse_bind_matrices[i], buffer.data.data() + offset, k_mat4_bytes);
      }
    }

    out_skeletons.push_back(std::move(skeleton));
  }
}

inline void load_animations(const tinygltf::Model &model, std::vector<AnimationClip> &out_animations) {
  out_animations.clear();
  out_animations.reserve(model.animations.size());

  for (const tinygltf::Animation &gltf_anim : model.animations) {
    AnimationClip clip{};
    clip.name = gltf_anim.name;

    clip.samplers.reserve(gltf_anim.samplers.size());
    for (const tinygltf::AnimationSampler &gltf_sampler : gltf_anim.samplers) {
      AnimationSampler sampler{};
      sampler.interpolation = gltf_sampler.interpolation;

      {
        const tinygltf::Accessor &accessor = model.accessors[static_cast<std::size_t>(gltf_sampler.input)];
        sampler.inputs.resize(accessor.count);
        const tinygltf::BufferView &view = model.bufferViews[static_cast<std::size_t>(accessor.bufferView)];
        const tinygltf::Buffer &buffer = model.buffers[static_cast<std::size_t>(view.buffer)];
        const std::size_t stride = accessor.ByteStride(view);
        for (std::size_t i = 0; i < accessor.count; ++i) {
          const std::size_t offset = static_cast<std::size_t>(accessor.byteOffset + view.byteOffset + i * (stride == 0 ? sizeof(float) : stride));
          std::memcpy(&sampler.inputs[i], buffer.data.data() + offset, sizeof(float));
        }
        if (!sampler.inputs.empty())
          clip.duration = std::max(clip.duration, sampler.inputs.back());
      }

      {
        const tinygltf::Accessor &accessor = model.accessors[static_cast<std::size_t>(gltf_sampler.output)];
        sampler.outputs.resize(accessor.count);
        const tinygltf::BufferView &view = model.bufferViews[static_cast<std::size_t>(accessor.bufferView)];
        const tinygltf::Buffer &buffer = model.buffers[static_cast<std::size_t>(view.buffer)];
        const std::size_t stride = accessor.ByteStride(view);
        const std::size_t component_count = accessor.type == TINYGLTF_TYPE_VEC3 ? 3U : 4U;
        const std::size_t element_bytes = component_count * sizeof(float);

        for (std::size_t i = 0; i < accessor.count; ++i) {
          const std::size_t offset = static_cast<std::size_t>(accessor.byteOffset + view.byteOffset + i * (stride == 0 ? element_bytes : stride));
          glm::vec4 value(0.0F, 0.0F, 0.0F, 1.0F);
          std::memcpy(glm::value_ptr(value), buffer.data.data() + offset, element_bytes);
          sampler.outputs[i] = value;
        }
      }

      clip.samplers.push_back(std::move(sampler));
    }

    for (const tinygltf::AnimationChannel &gltf_channel : gltf_anim.channels) {
      AnimationChannel channel{};
      channel.node_index = static_cast<std::uint32_t>(gltf_channel.target_node);
      channel.path = gltf_channel.target_path;
      channel.sampler_index = static_cast<std::uint32_t>(gltf_channel.sampler);
      clip.channels.push_back(channel);
    }

    out_animations.push_back(std::move(clip));
  }
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
    std::cerr << "glTF warning: " << warning << '\n';
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

  result.node_parents.assign(model.nodes.size(), std::numeric_limits<std::uint32_t>::max());
  for (std::size_t i = 0; i < model.nodes.size(); ++i) {
    for (int child : model.nodes[i].children) {
      if (child < 0 || static_cast<std::size_t>(child) >= result.node_parents.size()) continue;
      result.node_parents[static_cast<std::size_t>(child)] = static_cast<std::uint32_t>(i);
    }
  }

  if (!model.skins.empty())
    detail::load_skeletons(model, result.skeletons, result.node_parents);

  if (!model.animations.empty())
    detail::load_animations(model, result.animations);

  if (result.primitives.empty())
    throw std::runtime_error("glTF file contains no renderable mesh primitives");

  return result;
}

struct GltfImportResult {
  std::uint32_t first_instance{};
  std::uint32_t instance_count{};
  std::uint32_t skeleton_index{k_invalid_skin_index};
};

// Single-call glTF import: loads file, adds meshes/textures/instances/skeletons/animations,
// wires skin_index, and returns handles. Caller can then set up pose layers, hitboxes, etc.
[[nodiscard]] inline auto import_gltf(Scene &scene, const std::string &path,
                                       glm::mat4 instance_transform = glm::mat4(1.0F))
    -> GltfImportResult {
  const LoadedGltfModel model = load_gltf_model(path);
  GltfImportResult result;
  if (model.primitives.empty()) return result;

  const std::uint32_t before = static_cast<std::uint32_t>(scene.instances().size());

  for (const LoadedGltfPrimitive &prim : model.primitives) {
    const std::uint32_t mesh_idx = scene.add_mesh({prim.mesh.vertices, prim.mesh.indices});
    const std::uint32_t tex_idx = prim.material.base_color_texture_path
        ? scene.add_texture(*prim.material.base_color_texture_path)
        : 0;
    (void)scene.add_instance({
        .mesh_index = mesh_idx,
        .texture_index = tex_idx,
        .texture_source = TextureSource::Table,
        .model = instance_transform * prim.node_transform,
        .layer = RenderLayer::Opaque,
    });
  }

  result.first_instance = before;
  result.instance_count = static_cast<std::uint32_t>(scene.instances().size()) - before;

  if (!model.skeletons.empty()) {
    SkeletonAsset skeleton = model.skeletons.front();
    result.skeleton_index = scene.add_skeleton(std::move(skeleton));

    for (const AnimationClip &anim : model.animations)
      (void)scene.add_animation(anim);

    for (std::uint32_t i = 0; i < result.instance_count; ++i)
      scene.instance(result.first_instance + i).skin_index = result.skeleton_index;
  }

  return result;
}

} // namespace engine
