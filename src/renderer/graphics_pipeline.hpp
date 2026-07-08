#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "renderer/textured_push_constants.hpp"

namespace engine {

struct GraphicsPipelineRasterState {
  vk::CullModeFlags cull_mode{vk::CullModeFlagBits::eBack};
  vk::FrontFace front_face{vk::FrontFace::eCounterClockwise};
  bool depth_test{true};
  bool depth_write{true};
  vk::CompareOp depth_compare{vk::CompareOp::eLessOrEqual};
  bool depth_bias_enable{false};
  float depth_bias_constant{0.0F};
  float depth_bias_slope{0.0F};
  bool alpha_to_coverage{false};
  bool blend_enable{false};
  vk::BlendFactor src_color_blend{vk::BlendFactor::eSrcAlpha};
  vk::BlendFactor dst_color_blend{vk::BlendFactor::eOneMinusSrcAlpha};
  vk::BlendFactor src_alpha_blend{vk::BlendFactor::eOne};
  vk::BlendFactor dst_alpha_blend{vk::BlendFactor::eOneMinusSrcAlpha};
};

[[nodiscard]] inline auto read_spirv_file(std::string_view path) -> std::vector<char> {
  std::ifstream file{std::string(path), std::ios::ate | std::ios::binary};
  if (!file.is_open())
    throw std::runtime_error(std::string("Failed to open shader SPIR-V: ") + std::string(path));

  std::vector<char> buffer(static_cast<std::size_t>(file.tellg()));
  file.seekg(0, std::ios::beg);
  file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  return buffer;
}

[[nodiscard]] inline auto create_shader_module(vk::raii::Device &device, const std::vector<char> &code)
    -> vk::raii::ShaderModule {
  const vk::ShaderModuleCreateInfo create_info{
      .codeSize = code.size(),
      .pCode = reinterpret_cast<const std::uint32_t *>(code.data()),
  };
  return vk::raii::ShaderModule(device, create_info);
}

[[nodiscard]] inline auto create_pipeline_layout(
    vk::raii::Device &device,
    std::span<const vk::DescriptorSetLayout> descriptor_set_layouts,
    vk::PushConstantRange push_constant_range) -> vk::raii::PipelineLayout {
  const bool has_pc = push_constant_range.size > 0;
  return vk::raii::PipelineLayout(
      device,
      vk::PipelineLayoutCreateInfo{
          .setLayoutCount = static_cast<std::uint32_t>(descriptor_set_layouts.size()),
          .pSetLayouts = descriptor_set_layouts.data(),
          .pushConstantRangeCount = has_pc ? 1u : 0u,
          .pPushConstantRanges = has_pc ? &push_constant_range : nullptr,
      });
}

[[nodiscard]] inline auto create_pipeline_layout(
    vk::raii::Device &device,
    std::span<const vk::DescriptorSetLayout> descriptor_set_layouts) -> vk::raii::PipelineLayout {
  const vk::PushConstantRange push_constant_range{
      .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
      .offset = 0,
      .size = sizeof(TexturedPushConstants),
  };
  return create_pipeline_layout(device, descriptor_set_layouts, push_constant_range);
}

[[nodiscard]] inline auto create_graphics_pipeline(
    vk::raii::Device &device,
    vk::Format color_format,
    vk::Format depth_format,
    std::string_view vertex_spirv_path,
    std::string_view fragment_spirv_path,
    vk::PipelineLayout pipeline_layout,
    vk::SampleCountFlagBits sample_count,
    vk::VertexInputBindingDescription binding,
    std::span<const vk::VertexInputAttributeDescription> attributes,
    const vk::raii::PipelineCache &pipeline_cache,
    GraphicsPipelineRasterState raster_state = {},
    const char *vertex_entry_point = "vertMain",
    const char *fragment_entry_point = "fragMain",
    const vk::SpecializationInfo *frag_spec_info = nullptr,
    std::uint32_t view_mask = 0) -> vk::raii::Pipeline {
  const auto vertex_spirv = read_spirv_file(vertex_spirv_path);
  const auto fragment_spirv = read_spirv_file(fragment_spirv_path);
  const vk::raii::ShaderModule vertex_module = create_shader_module(device, vertex_spirv);
  const vk::raii::ShaderModule fragment_module = create_shader_module(device, fragment_spirv);

  const std::array stages{
      vk::PipelineShaderStageCreateInfo{
          .stage = vk::ShaderStageFlagBits::eVertex,
          .module = *vertex_module,
          .pName = vertex_entry_point,
      },
      vk::PipelineShaderStageCreateInfo{
          .stage = vk::ShaderStageFlagBits::eFragment,
          .module = *fragment_module,
          .pName = fragment_entry_point,
          .pSpecializationInfo = frag_spec_info,
      },
  };

  const vk::PipelineVertexInputStateCreateInfo vertex_input_info{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &binding,
      .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size()),
      .pVertexAttributeDescriptions = attributes.data(),
  };
  const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
      .topology = vk::PrimitiveTopology::eTriangleList,
  };
  const vk::PipelineViewportStateCreateInfo viewport_state{
      .viewportCount = 1,
      .scissorCount = 1,
  };
  const vk::PipelineRasterizationStateCreateInfo rasterizer{
      .depthClampEnable = vk::False,
      .rasterizerDiscardEnable = vk::False,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = raster_state.cull_mode,
      .frontFace = raster_state.front_face,
      .depthBiasEnable = raster_state.depth_bias_enable ? vk::True : vk::False,
      .depthBiasConstantFactor = raster_state.depth_bias_constant,
      .depthBiasSlopeFactor = raster_state.depth_bias_slope,
      .lineWidth = 1.0F,
  };
  const vk::PipelineMultisampleStateCreateInfo multisampling{
      .rasterizationSamples = sample_count,
      .alphaToCoverageEnable = raster_state.alpha_to_coverage ? vk::True : vk::False,
  };
  const vk::PipelineColorBlendAttachmentState color_blend_attachment{
      .blendEnable = raster_state.blend_enable ? vk::True : vk::False,
      .srcColorBlendFactor = raster_state.src_color_blend,
      .dstColorBlendFactor = raster_state.dst_color_blend,
      .colorBlendOp = vk::BlendOp::eAdd,
      .srcAlphaBlendFactor = raster_state.src_alpha_blend,
      .dstAlphaBlendFactor = raster_state.dst_alpha_blend,
      .alphaBlendOp = vk::BlendOp::eAdd,
      .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  };
  const bool has_color = color_format != vk::Format::eUndefined;
  const vk::PipelineColorBlendStateCreateInfo color_blending{
      .attachmentCount = has_color ? 1u : 0u,
      .pAttachments = has_color ? &color_blend_attachment : nullptr,
  };

  const vk::PipelineDepthStencilStateCreateInfo depth_stencil{
      .depthTestEnable = raster_state.depth_test ? vk::True : vk::False,
      .depthWriteEnable = raster_state.depth_write ? vk::True : vk::False,
      .depthCompareOp = raster_state.depth_compare,
  };

  const std::array dynamic_states{
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor,
  };
  std::vector<vk::DynamicState> dynamic_state_list{dynamic_states.begin(), dynamic_states.end()};
  if (raster_state.depth_bias_enable)
    dynamic_state_list.push_back(vk::DynamicState::eDepthBias);

  const vk::PipelineDynamicStateCreateInfo dynamic_state{
      .dynamicStateCount = static_cast<std::uint32_t>(dynamic_state_list.size()),
      .pDynamicStates = dynamic_state_list.data(),
  };

  vk::Format color_attachment_format = color_format;
  vk::Format depth_attachment_format = depth_format;

  vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipeline_chain{
      vk::GraphicsPipelineCreateInfo{
          .stageCount = static_cast<std::uint32_t>(stages.size()),
          .pStages = stages.data(),
          .pVertexInputState = &vertex_input_info,
          .pInputAssemblyState = &input_assembly,
          .pViewportState = &viewport_state,
          .pRasterizationState = &rasterizer,
          .pMultisampleState = &multisampling,
          .pDepthStencilState = &depth_stencil,
          .pColorBlendState = &color_blending,
          .pDynamicState = &dynamic_state,
          .layout = pipeline_layout,
          .renderPass = nullptr,
      },
      vk::PipelineRenderingCreateInfo{
          .viewMask = view_mask,
          .colorAttachmentCount = has_color ? 1u : 0u,
          .pColorAttachmentFormats = has_color ? &color_attachment_format : nullptr,
          .depthAttachmentFormat = depth_attachment_format,
      },
  };

  return vk::raii::Pipeline(
      device,
      pipeline_cache,
      pipeline_chain.get<vk::GraphicsPipelineCreateInfo>());
}

[[nodiscard]] inline auto create_graphics_pipeline(
    vk::raii::Device &device,
    vk::Format color_format,
    vk::Format depth_format,
    std::string_view spirv_path,
    vk::PipelineLayout pipeline_layout,
    vk::SampleCountFlagBits sample_count,
    vk::VertexInputBindingDescription binding,
    std::span<const vk::VertexInputAttributeDescription> attributes,
    const vk::raii::PipelineCache &pipeline_cache,
    GraphicsPipelineRasterState raster_state = {},
    const char *fragment_entry_point = "fragMain",
    const vk::SpecializationInfo *frag_spec_info = nullptr) -> vk::raii::Pipeline {
  const auto spirv = read_spirv_file(spirv_path);
  const vk::raii::ShaderModule shader_module = create_shader_module(device, spirv);

  const std::array stages{
      vk::PipelineShaderStageCreateInfo{
          .stage = vk::ShaderStageFlagBits::eVertex,
          .module = *shader_module,
          .pName = "vertMain",
      },
      vk::PipelineShaderStageCreateInfo{
          .stage = vk::ShaderStageFlagBits::eFragment,
          .module = *shader_module,
          .pName = fragment_entry_point,
          .pSpecializationInfo = frag_spec_info,
      },
  };

  const vk::PipelineVertexInputStateCreateInfo vertex_input_info{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &binding,
      .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size()),
      .pVertexAttributeDescriptions = attributes.data(),
  };
  const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
      .topology = vk::PrimitiveTopology::eTriangleList,
  };
  const vk::PipelineViewportStateCreateInfo viewport_state{
      .viewportCount = 1,
      .scissorCount = 1,
  };
  const vk::PipelineRasterizationStateCreateInfo rasterizer{
      .depthClampEnable = vk::False,
      .rasterizerDiscardEnable = vk::False,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = raster_state.cull_mode,
      .frontFace = raster_state.front_face,
      .depthBiasEnable = raster_state.depth_bias_enable ? vk::True : vk::False,
      .depthBiasConstantFactor = raster_state.depth_bias_constant,
      .depthBiasSlopeFactor = raster_state.depth_bias_slope,
      .lineWidth = 1.0F,
  };
  const vk::PipelineMultisampleStateCreateInfo multisampling{
      .rasterizationSamples = sample_count,
      .alphaToCoverageEnable = raster_state.alpha_to_coverage ? vk::True : vk::False,
  };
  const vk::PipelineColorBlendAttachmentState color_blend_attachment{
      .blendEnable = raster_state.blend_enable ? vk::True : vk::False,
      .srcColorBlendFactor = raster_state.src_color_blend,
      .dstColorBlendFactor = raster_state.dst_color_blend,
      .colorBlendOp = vk::BlendOp::eAdd,
      .srcAlphaBlendFactor = raster_state.src_alpha_blend,
      .dstAlphaBlendFactor = raster_state.dst_alpha_blend,
      .alphaBlendOp = vk::BlendOp::eAdd,
      .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  };
  const vk::PipelineColorBlendStateCreateInfo color_blending{
      .attachmentCount = 1,
      .pAttachments = &color_blend_attachment,
  };

  const vk::PipelineDepthStencilStateCreateInfo depth_stencil{
      .depthTestEnable = raster_state.depth_test ? vk::True : vk::False,
      .depthWriteEnable = raster_state.depth_write ? vk::True : vk::False,
      .depthCompareOp = raster_state.depth_compare,
  };

  const std::array dynamic_states{
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor,
  };
  std::vector<vk::DynamicState> dynamic_state_list{dynamic_states.begin(), dynamic_states.end()};
  if (raster_state.depth_bias_enable)
    dynamic_state_list.push_back(vk::DynamicState::eDepthBias);

  const vk::PipelineDynamicStateCreateInfo dynamic_state{
      .dynamicStateCount = static_cast<std::uint32_t>(dynamic_state_list.size()),
      .pDynamicStates = dynamic_state_list.data(),
  };

  vk::Format color_attachment_format = color_format;
  vk::Format depth_attachment_format = depth_format;

  vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipeline_chain{
      vk::GraphicsPipelineCreateInfo{
          .stageCount = static_cast<std::uint32_t>(stages.size()),
          .pStages = stages.data(),
          .pVertexInputState = &vertex_input_info,
          .pInputAssemblyState = &input_assembly,
          .pViewportState = &viewport_state,
          .pRasterizationState = &rasterizer,
          .pMultisampleState = &multisampling,
          .pDepthStencilState = &depth_stencil,
          .pColorBlendState = &color_blending,
          .pDynamicState = &dynamic_state,
          .layout = pipeline_layout,
          .renderPass = nullptr,
      },
      vk::PipelineRenderingCreateInfo{
          .colorAttachmentCount = 1,
          .pColorAttachmentFormats = &color_attachment_format,
          .depthAttachmentFormat = depth_attachment_format,
      },
  };

  return vk::raii::Pipeline(
      device,
      pipeline_cache,
      pipeline_chain.get<vk::GraphicsPipelineCreateInfo>());
}

[[nodiscard]] inline auto create_depth_only_graphics_pipeline(
    vk::raii::Device &device,
    vk::Format depth_format,
    std::string_view vert_spirv_path,
    vk::PipelineLayout pipeline_layout,
    const vk::raii::PipelineCache &pipeline_cache,
    vk::VertexInputBindingDescription binding,
    std::span<const vk::VertexInputAttributeDescription> attributes,
    GraphicsPipelineRasterState raster_state = {},
    const char *entry_point = "vertMain") -> vk::raii::Pipeline {
  const auto spirv = read_spirv_file(vert_spirv_path);
  const vk::raii::ShaderModule shader_module = create_shader_module(device, spirv);

  const vk::PipelineShaderStageCreateInfo vert_stage{
      .stage = vk::ShaderStageFlagBits::eVertex,
      .module = *shader_module,
      .pName = entry_point,
  };

  const vk::PipelineVertexInputStateCreateInfo vertex_input_info{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &binding,
      .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size()),
      .pVertexAttributeDescriptions = attributes.data(),
  };
  const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
      .topology = vk::PrimitiveTopology::eTriangleList,
  };
  const vk::PipelineViewportStateCreateInfo viewport_state{
      .viewportCount = 1,
      .scissorCount = 1,
  };
  const vk::PipelineRasterizationStateCreateInfo rasterizer{
      .depthClampEnable = vk::False,
      .rasterizerDiscardEnable = vk::False,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = raster_state.cull_mode,
      .frontFace = raster_state.front_face,
      .depthBiasEnable = raster_state.depth_bias_enable ? vk::True : vk::False,
      .depthBiasConstantFactor = raster_state.depth_bias_constant,
      .depthBiasSlopeFactor = raster_state.depth_bias_slope,
      .lineWidth = 1.0F,
  };
  const vk::PipelineMultisampleStateCreateInfo multisampling{
      .rasterizationSamples = vk::SampleCountFlagBits::e1,
  };
  const vk::PipelineDepthStencilStateCreateInfo depth_stencil{
      .depthTestEnable = raster_state.depth_test ? vk::True : vk::False,
      .depthWriteEnable = raster_state.depth_write ? vk::True : vk::False,
      .depthCompareOp = raster_state.depth_compare,
  };

  std::vector<vk::DynamicState> dynamic_state_list{
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor,
  };
  if (raster_state.depth_bias_enable)
    dynamic_state_list.push_back(vk::DynamicState::eDepthBias);

  const vk::PipelineDynamicStateCreateInfo dynamic_state{
      .dynamicStateCount = static_cast<std::uint32_t>(dynamic_state_list.size()),
      .pDynamicStates = dynamic_state_list.data(),
  };

  vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipeline_chain{
      vk::GraphicsPipelineCreateInfo{
          .stageCount = 1,
          .pStages = &vert_stage,
          .pVertexInputState = &vertex_input_info,
          .pInputAssemblyState = &input_assembly,
          .pViewportState = &viewport_state,
          .pRasterizationState = &rasterizer,
          .pMultisampleState = &multisampling,
          .pDepthStencilState = &depth_stencil,
          .pDynamicState = &dynamic_state,
          .layout = pipeline_layout,
          .renderPass = nullptr,
      },
      vk::PipelineRenderingCreateInfo{
          .colorAttachmentCount = 0,
          .depthAttachmentFormat = depth_format,
      },
  };

  return vk::raii::Pipeline(
      device,
      pipeline_cache,
      pipeline_chain.get<vk::GraphicsPipelineCreateInfo>());
}

} // namespace engine
