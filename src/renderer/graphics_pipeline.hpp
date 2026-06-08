#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/mat4x4.hpp>

namespace engine {

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
      const vk::raii::PipelineCache &pipeline_cache) {
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
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
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
        .depthTestEnable = vk::True,
        .depthWriteEnable = vk::True,
        .depthCompareOp = vk::CompareOp::eLessOrEqual,
    };

    const std::array dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    const vk::PipelineDynamicStateCreateInfo dynamic_state{
        .dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    vk::Format color_attachment_format = color_format;
    vk::Format depth_attachment_format = depth_format;

    const vk::PushConstantRange push_constant_range{
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
        .offset = 0,
        .size = sizeof(glm::mat4),
    };

    pipeline_layout_ = vk::raii::PipelineLayout(
        device,
        vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant_range,
        });

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
            .layout = *pipeline_layout_,
            .renderPass = nullptr,
        },
        vk::PipelineRenderingCreateInfo{
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &color_attachment_format,
            .depthAttachmentFormat = depth_attachment_format,
        },
    };

    graphics_pipeline_ = vk::raii::Pipeline(
        device,
        pipeline_cache,
        pipeline_chain.get<vk::GraphicsPipelineCreateInfo>());
  }

  [[nodiscard]] auto layout() const -> const vk::raii::PipelineLayout & {
    return pipeline_layout_;
  }

  [[nodiscard]] auto pipeline() const -> const vk::raii::Pipeline & {
    return graphics_pipeline_;
  }

private:
  [[nodiscard]] static auto read_spirv_file(std::string_view path) -> std::vector<char> {
    std::ifstream file{std::string(path), std::ios::ate | std::ios::binary};
    if (!file.is_open())
      throw std::runtime_error(std::string("Failed to open shader SPIR-V: ") + std::string(path));

    std::vector<char> buffer(static_cast<std::size_t>(file.tellg()));
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    return buffer;
  }

  [[nodiscard]] static auto create_shader_module(vk::raii::Device &device, const std::vector<char> &code)
      -> vk::raii::ShaderModule {
    const vk::ShaderModuleCreateInfo create_info{
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const std::uint32_t *>(code.data()),
    };
    return vk::raii::ShaderModule(device, create_info);
  }

  vk::raii::PipelineLayout pipeline_layout_{nullptr};
  vk::raii::Pipeline graphics_pipeline_{nullptr};
};

} // namespace engine
