#pragma once

#include "renderer/graphics_pipeline.hpp"
#include "renderer/pipeline_id.hpp"
#include "renderer/vertex.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace engine {

class PipelineRegistry {
public:
  void create(
      vk::raii::Device &device,
      vk::Format color_format,
      vk::Format depth_format,
      std::string_view textured_mesh_spirv,
      std::string_view background_spirv,
      vk::DescriptorSetLayout descriptor_set_layout,
      vk::SampleCountFlagBits sample_count,
      const vk::raii::PipelineCache &pipeline_cache) {
    descriptor_set_layout_ = descriptor_set_layout;
    color_format_ = color_format;
    depth_format_ = depth_format;
    sample_count_ = sample_count;
    textured_mesh_spirv_ = textured_mesh_spirv;
    background_spirv_ = background_spirv;
    pipeline_cache_ = &pipeline_cache;

    rebuild(device);
  }

  void recreate(vk::raii::Device &device, vk::Format color_format, vk::Format depth_format) {
    color_format_ = color_format;
    depth_format_ = depth_format;
    rebuild(device);
  }

  [[nodiscard]] auto layout() const -> vk::PipelineLayout {
    return *pipeline_layout_;
  }

  [[nodiscard]] auto pipeline(PipelineId id) const -> vk::Pipeline {
    const auto index = static_cast<std::size_t>(id);
    if (index >= pipelines_.size() || *pipelines_[index] == nullptr)
      throw std::runtime_error("Requested graphics pipeline was not created");
    return *pipelines_[index];
  }

private:
  void rebuild(vk::raii::Device &device) {
    if (descriptor_set_layout_ == nullptr || pipeline_cache_ == nullptr)
      throw std::runtime_error("PipelineRegistry is missing create dependencies");

    pipeline_layout_ = create_pipeline_layout(device, descriptor_set_layout_);

    const auto mesh_binding = MeshVertex::binding_description();
    const auto mesh_attributes = MeshVertex::attribute_descriptions();

    const std::array sky_attributes{MeshVertex::attribute_descriptions()[0]};

    pipelines_[static_cast<std::size_t>(PipelineId::Background)] = create_graphics_pipeline(
        device,
        color_format_,
        depth_format_,
        background_spirv_,
        *pipeline_layout_,
        sample_count_,
        mesh_binding,
        sky_attributes,
        *pipeline_cache_,
        {
            .cull_mode = vk::CullModeFlagBits::eBack,
            .front_face = vk::FrontFace::eCounterClockwise,
            .depth_test = false,
            .depth_write = false,
        });

    pipelines_[static_cast<std::size_t>(PipelineId::TexturedMesh)] = create_graphics_pipeline(
        device,
        color_format_,
        depth_format_,
        textured_mesh_spirv_,
        *pipeline_layout_,
        sample_count_,
        mesh_binding,
        mesh_attributes,
        *pipeline_cache_);
  }

  vk::DescriptorSetLayout descriptor_set_layout_{nullptr};
  vk::Format color_format_{};
  vk::Format depth_format_{};
  vk::SampleCountFlagBits sample_count_{};
  std::string_view textured_mesh_spirv_{};
  std::string_view background_spirv_{};
  const vk::raii::PipelineCache *pipeline_cache_{nullptr};
  vk::raii::PipelineLayout pipeline_layout_{nullptr};
  std::array<vk::raii::Pipeline, pipeline_count()> pipelines_{
      vk::raii::Pipeline{nullptr},
      vk::raii::Pipeline{nullptr},
  };
};

} // namespace engine
