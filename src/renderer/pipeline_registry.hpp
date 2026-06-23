#pragma once

#include "renderer/graphics_pipeline.hpp"
#include "renderer/pipeline_id.hpp"
#include "renderer/vertex_attributes.hpp"
#include "scene/shadow_utils.hpp"
#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace engine {

class PipelineRegistry {
public:
  void create(
      vk::raii::Device &device,
      vk::Format color_format,
      vk::Format depth_format,
      vk::Format shadow_depth_format,
      std::string_view textured_mesh_spirv,
      std::string_view background_spirv,
      std::string_view shadow_depth_spirv,
      std::string_view skinned_textured_spirv,
      std::string_view skinned_shadow_depth_spirv,
      vk::DescriptorSetLayout frame_layout,
      vk::DescriptorSetLayout material_layout,
      vk::DescriptorSetLayout material_skinned_layout,
      vk::SampleCountFlagBits sample_count,
      const vk::raii::PipelineCache &pipeline_cache,
      PipelineBuildProfile profile) {
    frame_layout_ = frame_layout;
    material_layout_ = material_layout;
    material_skinned_layout_ = material_skinned_layout;
    color_format_ = color_format;
    depth_format_ = depth_format;
    shadow_depth_format_ = shadow_depth_format;
    sample_count_ = sample_count;
    textured_mesh_spirv_ = textured_mesh_spirv;
    background_spirv_ = background_spirv;
    shadow_depth_spirv_ = shadow_depth_spirv;
    skinned_textured_spirv_ = skinned_textured_spirv;
    skinned_shadow_depth_spirv_ = skinned_shadow_depth_spirv;
    pipeline_cache_ = &pipeline_cache;
    profile_ = profile;

    rebuild(device);
  }

  void recreate(vk::raii::Device &device, vk::Format color_format, vk::Format depth_format) {
    color_format_ = color_format;
    depth_format_ = depth_format;
    rebuild(device);
  }

  [[nodiscard]] auto layout() const -> vk::PipelineLayout { return *pipeline_layout_; }

  [[nodiscard]] auto layout_for(PipelineId id) const -> vk::PipelineLayout {
    return is_skinned_pipeline(id) ? *skinned_pipeline_layout_ : *pipeline_layout_;
  }

  [[nodiscard]] auto pipeline(PipelineId id) const -> vk::Pipeline {
    const auto index = static_cast<std::size_t>(id);
    if (index >= pipelines_.size() || !pipelines_[index].has_value())
      throw std::runtime_error("Requested graphics pipeline was not created");
    return **pipelines_[index];
  }

private:
  void rebuild(vk::raii::Device &device) {
    if (frame_layout_ == nullptr || material_layout_ == nullptr || pipeline_cache_ == nullptr)
      throw std::runtime_error("PipelineRegistry is missing create dependencies");

    for (std::optional<vk::raii::Pipeline> &pipeline : pipelines_)
      pipeline.reset();

    const std::array set_layouts{frame_layout_, material_layout_};
    pipeline_layout_ = create_pipeline_layout(device, set_layouts);

    if (profile_.build_skinned) {
      const std::array skinned_set_layouts{frame_layout_, material_skinned_layout_};
      skinned_pipeline_layout_ = create_pipeline_layout(device, skinned_set_layouts);
    }

    const auto mesh_binding = mesh_binding_description();
    const auto static_mesh_attributes = static_attribute_descriptions();
    const auto mesh_attributes = attribute_descriptions();
    const std::array sky_attributes{attribute_descriptions()[0]};
    const std::array shadow_attributes{attribute_descriptions()[0]};
    const auto shadow_skinned_attributes = shadow_skinned_attribute_descriptions();

    pipelines_[static_cast<std::size_t>(PipelineId::Background)] = create_graphics_pipeline(
        device, color_format_, depth_format_, background_spirv_,
        *pipeline_layout_, sample_count_, mesh_binding, sky_attributes,
        *pipeline_cache_,
        {.cull_mode = vk::CullModeFlagBits::eBack, .front_face = vk::FrontFace::eCounterClockwise,
         .depth_test = false, .depth_write = false});

    static constexpr std::array k_alpha_modes{
        MeshAlphaMode::Opaque, MeshAlphaMode::Cutout, MeshAlphaMode::AlphaToCoverage};

    for (const MeshAlphaMode alpha_mode : k_alpha_modes) {
      const auto alpha_index = static_cast<std::size_t>(alpha_mode);
      if (!profile_.textured_alpha_modes[alpha_index])
        continue;

      GraphicsPipelineRasterState raster{};
      if (alpha_mode != MeshAlphaMode::Opaque)
        raster.cull_mode = vk::CullModeFlagBits::eNone;
      if (alpha_mode == MeshAlphaMode::AlphaToCoverage)
        raster.alpha_to_coverage = sample_count_ != vk::SampleCountFlagBits::e1;

      const char *frag_entry = textured_fragment_entry(
          profile_.shadow_filter, alpha_mode, profile_.cascade_mode);

      pipelines_[static_cast<std::size_t>(textured_pipeline(alpha_mode))] =
          create_graphics_pipeline(
              device, color_format_, depth_format_, textured_mesh_spirv_,
              *pipeline_layout_, sample_count_, mesh_binding, static_mesh_attributes,
              *pipeline_cache_, raster, frag_entry);

      if (profile_.build_skinned)
        pipelines_[static_cast<std::size_t>(textured_pipeline(alpha_mode, true))] =
            create_graphics_pipeline(
                device, color_format_, depth_format_,
                skinned_textured_spirv_, textured_mesh_spirv_,
                *skinned_pipeline_layout_, sample_count_, mesh_binding, mesh_attributes,
                *pipeline_cache_, raster, "vertMainSkinned", frag_entry);
    }

    pipelines_[static_cast<std::size_t>(PipelineId::ShadowDepth)] =
        create_depth_only_graphics_pipeline(
            device, shadow_depth_format_, shadow_depth_spirv_,
            *pipeline_layout_, *pipeline_cache_, mesh_binding, shadow_attributes,
            {.cull_mode = vk::CullModeFlagBits::eNone,
             .front_face = vk::FrontFace::eCounterClockwise,
             .depth_test = true, .depth_write = true,
             .depth_compare = vk::CompareOp::eLessOrEqual,
             .depth_bias_enable = true,
             .depth_bias_constant = k_shadow_depth_bias_constant,
             .depth_bias_slope = k_shadow_depth_bias_slope});

    if (profile_.build_skinned)
      pipelines_[static_cast<std::size_t>(PipelineId::ShadowDepthSkinned)] =
          create_depth_only_graphics_pipeline(
              device, shadow_depth_format_, skinned_shadow_depth_spirv_,
              *skinned_pipeline_layout_, *pipeline_cache_, mesh_binding,
              shadow_skinned_attributes,
              {.cull_mode = vk::CullModeFlagBits::eNone,
               .front_face = vk::FrontFace::eCounterClockwise,
               .depth_test = true, .depth_write = true,
               .depth_compare = vk::CompareOp::eLessOrEqual,
               .depth_bias_enable = true,
               .depth_bias_constant = k_shadow_depth_bias_constant,
               .depth_bias_slope = k_shadow_depth_bias_slope},
              "vertMainSkinned");
  }

  vk::DescriptorSetLayout frame_layout_{nullptr};
  vk::DescriptorSetLayout material_layout_{nullptr};
  vk::DescriptorSetLayout material_skinned_layout_{nullptr};
  vk::Format color_format_{};
  vk::Format depth_format_{};
  vk::Format shadow_depth_format_{};
  vk::SampleCountFlagBits sample_count_{};
  std::string_view textured_mesh_spirv_{};
  std::string_view background_spirv_{};
  std::string_view shadow_depth_spirv_{};
  std::string_view skinned_textured_spirv_{};
  std::string_view skinned_shadow_depth_spirv_{};
  const vk::raii::PipelineCache *pipeline_cache_{nullptr};
  PipelineBuildProfile profile_{};
  vk::raii::PipelineLayout pipeline_layout_{nullptr};
  vk::raii::PipelineLayout skinned_pipeline_layout_{nullptr};
  std::array<std::optional<vk::raii::Pipeline>, 9> pipelines_{};
};

} // namespace engine
