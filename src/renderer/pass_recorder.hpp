#pragma once

#include "renderer/descriptors.hpp"
#include "renderer/draw_list.hpp"
#include "renderer/image_barrier.hpp"
#include "renderer/mesh_gpu.hpp"
#include "renderer/msaa_color_image.hpp"
#include "renderer/pipeline_id.hpp"
#include "renderer/pipeline_registry.hpp"
#include "renderer/shadow_map.hpp"
#include "renderer/swapchain.hpp"
#include "renderer/texture_array.hpp"
#include "renderer/texture_table.hpp"
#include "renderer/textured_push_constants.hpp"
#include "renderer/depth_image.hpp"
#include "scene/shadow_utils.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace engine {

struct PassLayoutState {
  mutable vk::ImageLayout shadow_image_layout{vk::ImageLayout::eUndefined};
  mutable std::vector<vk::ImageLayout> swapchain_image_layouts;
  mutable vk::ImageLayout depth_image_layout{vk::ImageLayout::eUndefined};
  mutable vk::ImageLayout msaa_color_layout{vk::ImageLayout::eUndefined};
};

struct PassRecorder {
  const PipelineRegistry &pipelines;
  const DescriptorResources &descriptors;
  const Swapchain &swapchain;
  const DepthImage &depth_image;
  const MsaaColorImage &msaa_color_image;
  const ShadowMap &shadow_map;
  const TextureTable &texture_table;
  const TextureArray &texture_array;
  const std::vector<MeshGpu> &mesh_gpus;
  bool msaa_enabled{};

  void bind_pass_descriptor_sets(vk::raii::CommandBuffer &command_buffer, std::uint32_t frame_index) const {
    const vk::DescriptorSet descriptor_sets[] = {
        descriptors.frame_set(frame_index),
        descriptors.material_set(0),
    };
    command_buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        pipelines.layout(),
        0,
        descriptor_sets,
        nullptr);
  }

  void draw_mesh(
      vk::raii::CommandBuffer &command_buffer,
      const DrawCommand &draw,
      std::optional<PipelineId> &bound_pipeline,
      std::optional<TextureSource> &bound_texture_source,
      std::optional<std::uint32_t> &bound_material_index) const {
    if (draw.mesh_index >= mesh_gpus.size())
      return;

    if (!bound_pipeline || *bound_pipeline != draw.pipeline) {
      command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.pipeline(draw.pipeline));
      bound_pipeline = draw.pipeline;
      bound_texture_source.reset();
      bound_material_index.reset();
    }

    if (draw.pipeline == PipelineId::TexturedMesh) {
      const std::uint32_t descriptor_texture_index =
          draw.texture_source == TextureSource::Table ? draw.texture_index : 0;

      if (draw.texture_source == TextureSource::Table && draw.texture_index >= texture_table.count())
        return;
      if (draw.texture_source == TextureSource::ArrayLayer && draw.texture_index >= texture_array.layer_count())
        return;

      if (!bound_material_index || *bound_material_index != descriptor_texture_index) {
        const vk::DescriptorSet material_set = descriptors.material_set(descriptor_texture_index);
        command_buffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            pipelines.layout(),
            1,
            material_set,
            nullptr);
        bound_texture_source = draw.texture_source;
        bound_material_index = descriptor_texture_index;
      }

      const TexturedPushConstants push_constants{
          .model = draw.model,
          .texture_array_layer = draw.texture_index,
          .sample_texture_array = draw.texture_source == TextureSource::ArrayLayer ? 1U : 0U,
      };
      command_buffer.pushConstants(
          pipelines.layout(),
          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
          0,
          sizeof(TexturedPushConstants),
          &push_constants);
    } else if (draw.pipeline == PipelineId::Background) {
      const TexturedPushConstants sky_push_constants{.model = draw.model};
      command_buffer.pushConstants(
          pipelines.layout(),
          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
          0,
          sizeof(TexturedPushConstants),
          &sky_push_constants);
    } else if (draw.pipeline == PipelineId::ShadowDepth) {
      const TexturedPushConstants shadow_push_constants{.model = draw.model};
      command_buffer.pushConstants(
          pipelines.layout(),
          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
          0,
          sizeof(TexturedPushConstants),
          &shadow_push_constants);
    }

    const MeshGpu &mesh = mesh_gpus[draw.mesh_index];
    const vk::Buffer vertex_buffers[] = {mesh.vertex_buffer()};
    const vk::DeviceSize offsets[] = {0};
    command_buffer.bindVertexBuffers(0, vertex_buffers, offsets);
    command_buffer.bindIndexBuffer(mesh.index_buffer(), 0, vk::IndexType::eUint32);
    command_buffer.drawIndexed(mesh.index_count(), 1, 0, 0, 0);
  }

  void transition_swapchain_to_color_attachment(
      vk::raii::CommandBuffer &command_buffer,
      std::uint32_t image_index,
      PassLayoutState &layouts) const {
    vk::ImageLayout &tracked_layout = layouts.swapchain_image_layouts.at(image_index);
    const vk::Image image = swapchain.images()[image_index];

    if (tracked_layout == vk::ImageLayout::eUndefined) {
      transition_image_layout(
          command_buffer,
          image,
          vk::ImageLayout::eUndefined,
          vk::ImageLayout::eColorAttachmentOptimal,
          {},
          vk::AccessFlagBits2::eColorAttachmentWrite,
          vk::PipelineStageFlagBits2::eTopOfPipe,
          vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    } else if (tracked_layout == vk::ImageLayout::ePresentSrcKHR) {
      transition_image_layout(
          command_buffer,
          image,
          vk::ImageLayout::ePresentSrcKHR,
          vk::ImageLayout::eColorAttachmentOptimal,
          {},
          vk::AccessFlagBits2::eColorAttachmentWrite,
          vk::PipelineStageFlagBits2::eBottomOfPipe,
          vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    }

    tracked_layout = vk::ImageLayout::eColorAttachmentOptimal;
  }

  void transition_msaa_to_color_attachment(vk::raii::CommandBuffer &command_buffer, PassLayoutState &layouts) const {
    if (layouts.msaa_color_layout == vk::ImageLayout::eColorAttachmentOptimal)
      return;

    transition_image_layout(
        command_buffer,
        msaa_color_image.image(),
        layouts.msaa_color_layout,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eColorAttachmentWrite,
        layouts.msaa_color_layout == vk::ImageLayout::eUndefined ? vk::PipelineStageFlagBits2::eTopOfPipe
                                                                 : vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    layouts.msaa_color_layout = vk::ImageLayout::eColorAttachmentOptimal;
  }

  void transition_depth_to_attachment(vk::raii::CommandBuffer &command_buffer, PassLayoutState &layouts) const {
    if (layouts.depth_image_layout == vk::ImageLayout::eDepthAttachmentOptimal) {
      transition_image_layout(
          command_buffer,
          depth_image.image(),
          vk::ImageLayout::eDepthAttachmentOptimal,
          vk::ImageLayout::eDepthAttachmentOptimal,
          vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
          vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
          vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
          vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
          depth_image.aspect_mask());
      return;
    }

    transition_image_layout(
        command_buffer,
        depth_image.image(),
        layouts.depth_image_layout,
        vk::ImageLayout::eDepthAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        depth_image.aspect_mask());
    layouts.depth_image_layout = vk::ImageLayout::eDepthAttachmentOptimal;
  }

  void record_shadow_pass(
      vk::raii::CommandBuffer &command_buffer,
      std::uint32_t frame_index,
      PassLayoutState &layouts,
      const std::vector<DrawCommand> &draw_list) const {
    const vk::ImageLayout previous_layout = layouts.shadow_image_layout;
    const vk::AccessFlags2 previous_access =
        previous_layout == vk::ImageLayout::eUndefined ? vk::AccessFlagBits2{} : vk::AccessFlagBits2::eShaderRead;
    const vk::PipelineStageFlags2 previous_stage =
        previous_layout == vk::ImageLayout::eUndefined ? vk::PipelineStageFlagBits2::eTopOfPipe
                                                       : vk::PipelineStageFlagBits2::eFragmentShader;

    transition_image_layout(
        command_buffer,
        shadow_map.image(),
        previous_layout,
        vk::ImageLayout::eDepthAttachmentOptimal,
        previous_access,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        previous_stage,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        shadow_map.aspect_mask());

    const vk::ClearValue clear_depth{vk::ClearDepthStencilValue{1.0F, 0}};
    const vk::RenderingAttachmentInfo depth_attachment{
        .imageView = shadow_map.view(),
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clear_depth,
    };
    const vk::RenderingInfo rendering_info{
        .renderArea = {.offset = {.x = 0, .y = 0}, .extent = shadow_map.extent()},
        .layerCount = 1,
        .colorAttachmentCount = 0,
        .pDepthAttachment = &depth_attachment,
    };

    command_buffer.beginRendering(rendering_info);
    bind_pass_descriptor_sets(command_buffer, frame_index);
    command_buffer.setViewport(
        0,
        vk::Viewport{
            0.0F,
            0.0F,
            static_cast<float>(shadow_map.extent().width),
            static_cast<float>(shadow_map.extent().height),
            0.0F,
            1.0F,
        });
    command_buffer.setScissor(0, vk::Rect2D{{0, 0}, shadow_map.extent()});
    command_buffer.setDepthBias(k_shadow_depth_bias_constant, 0.0F, k_shadow_depth_bias_slope);

    std::optional<PipelineId> bound_pipeline;
    std::optional<TextureSource> bound_texture_source;
    std::optional<std::uint32_t> bound_material_index;

    for (const DrawCommand &draw : draw_list) {
      if (draw.pipeline != PipelineId::TexturedMesh)
        continue;

      DrawCommand shadow_draw = draw;
      shadow_draw.pipeline = PipelineId::ShadowDepth;
      draw_mesh(command_buffer, shadow_draw, bound_pipeline, bound_texture_source, bound_material_index);
    }

    command_buffer.endRendering();

    transition_image_layout(
        command_buffer,
        shadow_map.image(),
        vk::ImageLayout::eDepthAttachmentOptimal,
        vk::ImageLayout::eDepthStencilReadOnlyOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eFragmentShader,
        shadow_map.aspect_mask());

    layouts.shadow_image_layout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
  }

  void record_main_pass(
      vk::raii::CommandBuffer &command_buffer,
      std::uint32_t frame_index,
      std::uint32_t image_index,
      PassLayoutState &layouts,
      const std::vector<DrawCommand> &draw_list) const {
    transition_swapchain_to_color_attachment(command_buffer, image_index, layouts);

    if (msaa_enabled)
      transition_msaa_to_color_attachment(command_buffer, layouts);

    transition_depth_to_attachment(command_buffer, layouts);

    const vk::ClearValue clear_value{vk::ClearColorValue{std::array{0.02F, 0.03F, 0.05F, 1.0F}}};
    const vk::ClearValue clear_depth{vk::ClearDepthStencilValue{1.0F, 0}};

    const vk::RenderingAttachmentInfo color_attachment = msaa_enabled
        ? vk::RenderingAttachmentInfo{
              .imageView = *msaa_color_image.view(),
              .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
              .resolveMode = vk::ResolveModeFlagBits::eAverage,
              .resolveImageView = *swapchain.image_views()[image_index],
              .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
              .loadOp = vk::AttachmentLoadOp::eClear,
              .storeOp = vk::AttachmentStoreOp::eStore,
              .clearValue = clear_value,
          }
        : vk::RenderingAttachmentInfo{
              .imageView = *swapchain.image_views()[image_index],
              .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
              .loadOp = vk::AttachmentLoadOp::eClear,
              .storeOp = vk::AttachmentStoreOp::eStore,
              .clearValue = clear_value,
          };
    const vk::RenderingAttachmentInfo depth_attachment{
        .imageView = *depth_image.view(),
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = clear_depth,
    };
    const vk::RenderingInfo rendering_info{
        .renderArea = {.offset = {.x = 0, .y = 0}, .extent = swapchain.extent()},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment,
        .pDepthAttachment = &depth_attachment,
    };
    command_buffer.beginRendering(rendering_info);
    bind_pass_descriptor_sets(command_buffer, frame_index);
    command_buffer.setViewport(
        0,
        vk::Viewport{
            0.0F,
            0.0F,
            static_cast<float>(swapchain.extent().width),
            static_cast<float>(swapchain.extent().height),
            0.0F,
            1.0F,
        });
    command_buffer.setScissor(0, vk::Rect2D{{0, 0}, swapchain.extent()});

    std::optional<PipelineId> bound_pipeline;
    std::optional<TextureSource> bound_texture_source;
    std::optional<std::uint32_t> bound_material_index;

    for (const DrawCommand &draw : draw_list)
      draw_mesh(command_buffer, draw, bound_pipeline, bound_texture_source, bound_material_index);

    command_buffer.endRendering();

    transition_image_layout(
        command_buffer,
        swapchain.images()[image_index],
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        {},
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eBottomOfPipe);
    layouts.swapchain_image_layouts[image_index] = vk::ImageLayout::ePresentSrcKHR;

    command_buffer.end();
  }
};

} // namespace engine
