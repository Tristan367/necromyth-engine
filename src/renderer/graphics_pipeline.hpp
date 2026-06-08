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
#include "renderer/vertex.hpp"

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
    vk::DescriptorSetLayout descriptor_set_layout) -> vk::raii::PipelineLayout {
  const vk::PushConstantRange push_constant_range{
      .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
      .offset = 0,
      .size = sizeof(TexturedPushConstants),
  };

  return vk::raii::PipelineLayout(
      device,
      vk::PipelineLayoutCreateInfo{
          .setLayoutCount = 1,
          .pSetLayouts = &descriptor_set_layout,
          .pushConstantRangeCount = 1,
          .pPushConstantRanges = &push_constant_range,
      });
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
    GraphicsPipelineRasterState raster_state = {}) -> vk::raii::Pipeline {
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
          .pName = "fragMain",
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
  };
  const vk::PipelineColorBlendAttachmentState color_blend_attachment{
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
    GraphicsPipelineRasterState raster_state = {}) -> vk::raii::Pipeline {
  const auto spirv = read_spirv_file(vert_spirv_path);
  const vk::raii::ShaderModule shader_module = create_shader_module(device, spirv);

  const vk::PipelineShaderStageCreateInfo vert_stage{
      .stage = vk::ShaderStageFlagBits::eVertex,
      .module = *shader_module,
      .pName = "vertMain",
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

class GraphicsPipeline {
public:
  void create(
      vk::raii::Device &device,
      vk::Format color_format,
      vk::Format depth_format,
      std::string_view spirv_path,
      vk::DescriptorSetLayout descriptor_set_layout,
      vk::SampleCountFlagBits sample_count,
      vk::VertexInputBindingDescription binding,
      std::span<const vk::VertexInputAttributeDescription> attributes,
      const vk::raii::PipelineCache &pipeline_cache,
      GraphicsPipelineRasterState raster_state = {}) {
    pipeline_layout_ = create_pipeline_layout(device, descriptor_set_layout);
    graphics_pipeline_ = create_graphics_pipeline(
        device,
        color_format,
        depth_format,
        spirv_path,
        *pipeline_layout_,
        sample_count,
        binding,
        attributes,
        pipeline_cache,
        raster_state);
  }

  [[nodiscard]] auto layout() const -> const vk::raii::PipelineLayout & {
    return pipeline_layout_;
  }

  [[nodiscard]] auto pipeline() const -> const vk::raii::Pipeline & {
    return graphics_pipeline_;
  }

private:
  vk::raii::PipelineLayout pipeline_layout_{nullptr};
  vk::raii::Pipeline graphics_pipeline_{nullptr};
};

} // namespace engine
