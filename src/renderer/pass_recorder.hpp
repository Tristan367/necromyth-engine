#pragma once

#include "renderer/frame_overlay.hpp"
#include "renderer/descriptors.hpp"
#include "renderer/draw_list.hpp"
#include "renderer/image_barrier.hpp"
#include "renderer/light_buffer.hpp"
#include "renderer/mesh_gpu.hpp"
#include "renderer/msaa_color_image.hpp"
#include "renderer/pipeline_id.hpp"
#include "renderer/pipeline_registry.hpp"
#include "renderer/render_color_image.hpp"
#include "renderer/shadow_map.hpp"
#include "renderer/swapchain.hpp"
#include "renderer/texture_array.hpp"
#include "renderer/texture_table.hpp"
#include "renderer/textured_push_constants.hpp"
#include "renderer/depth_image.hpp"
#include "scene/shadow_utils.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace engine {

struct PassLayoutState {
  mutable vk::ImageLayout shadow_image_layout{vk::ImageLayout::eUndefined};
  mutable vk::ImageLayout spot_atlas_layout{vk::ImageLayout::eUndefined};
  mutable vk::ImageLayout point_cube_layout{vk::ImageLayout::eUndefined};
  mutable std::vector<vk::ImageLayout> swapchain_image_layouts;
  mutable vk::ImageLayout depth_image_layout{vk::ImageLayout::eUndefined};
  mutable vk::ImageLayout msaa_color_layout{vk::ImageLayout::eUndefined};
  mutable vk::ImageLayout render_color_layout{vk::ImageLayout::eUndefined};
};

struct DrawBindState {
  static constexpr std::uint32_t k_unbound_mesh = UINT32_MAX;

  struct MaterialKey {
    TextureSource texture_source{};
    std::uint32_t descriptor_index{};

    auto operator<=>(const MaterialKey &) const = default;
  };

  std::optional<PipelineId> pipeline;
  std::optional<MaterialKey> material;
  std::uint32_t mesh_index{k_unbound_mesh};
  std::uint32_t frame_index{};
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
  std::uint32_t shadow_cascade_count{1};
  vk::Extent2D render_extent{};
  const RenderColorImage *render_color_image{nullptr};

  [[nodiscard]] auto uses_render_scale() const -> bool {
    return render_color_image != nullptr;
  }

  [[nodiscard]] auto main_pass_color_target(std::uint32_t image_index) const -> vk::ImageView {
    if (render_color_image != nullptr)
      return *render_color_image->view();
    return *swapchain.image_views()[image_index];
  }

  void bind_pass_descriptors(vk::raii::CommandBuffer &command_buffer, std::uint32_t frame_index,
                               bool bind_material = false) const {
    if (bind_material) {
      const vk::DescriptorSet sets[] = {descriptors.frame_set(frame_index), descriptors.material_set(0)};
      command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelines.layout(), 0, sets, nullptr);
    } else {
      command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelines.layout(), 0,
                                        descriptors.frame_set(frame_index), nullptr);
    }
  }

  [[nodiscard]] auto material_descriptor_index(const DrawCommand &draw) const -> std::optional<std::uint32_t> {
    if (!is_textured_surface_pipeline(draw.pipeline))
      return std::nullopt;

    if (draw.texture_source == TextureSource::Table) {
      if (draw.texture_index >= texture_table.count())
        return std::nullopt;
      return draw.texture_index;
    }

    if (draw.texture_index >= texture_array.layer_count())
      return std::nullopt;
    return 0;
  }

  void bind_mesh_buffers(vk::raii::CommandBuffer &command_buffer, std::uint32_t mesh_index, DrawBindState &state) const {
    if (state.mesh_index == mesh_index)
      return;

    const MeshGpu &mesh = mesh_gpus[mesh_index];
    const vk::Buffer vertex_buffers[] = {mesh.vertex_buffer()};
    const vk::DeviceSize offsets[] = {0};
    command_buffer.bindVertexBuffers(0, vertex_buffers, offsets);
    command_buffer.bindIndexBuffer(mesh.index_buffer(), 0, vk::IndexType::eUint32);
    state.mesh_index = mesh_index;
  }

  void bind_pipeline(
      vk::raii::CommandBuffer &command_buffer,
      PipelineId pipeline_id,
      DrawBindState &state) const {
    if (state.pipeline.has_value() && *state.pipeline == pipeline_id)
      return;

    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.pipeline(pipeline_id));
    state.pipeline = pipeline_id;
    state.material.reset();
    state.mesh_index = DrawBindState::k_unbound_mesh;
  }

  void bind_material(
      vk::raii::CommandBuffer &command_buffer,
      const DrawCommand &draw,
      DrawBindState &state) const {
    const std::optional<std::uint32_t> descriptor_index = material_descriptor_index(draw);
    if (!descriptor_index)
      return;

    const DrawBindState::MaterialKey material_key{
        .texture_source = draw.texture_source,
        .descriptor_index = *descriptor_index,
    };

    if (state.material.has_value() && *state.material == material_key)
      return;

    const bool is_skinned = is_skinned_pipeline(draw.pipeline);

    if (is_skinned && draw.bone_instance_index != k_invalid_skin_index) {
      command_buffer.bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics,
          pipelines.layout_for(draw.pipeline),
          1,
          descriptors.skinned_set(draw.bone_instance_index, state.frame_index),
          nullptr);
    } else {
      command_buffer.bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics,
          pipelines.layout_for(draw.pipeline),
          1,
          descriptors.material_set(*descriptor_index),
          nullptr);
    }
    state.material = material_key;
  }

   void draw_shadow_mesh(
       vk::raii::CommandBuffer &command_buffer,
       const DrawCommand &draw,
       std::uint32_t cascade_index,
       std::uint32_t frame_index,
       const glm::mat4 &light_vp,
       DrawBindState &state,
       std::uint32_t point_light_index = 0) const {
    if (draw.mesh_index >= mesh_gpus.size())
      return;

    const bool is_skinned = is_skinned_pipeline(draw.pipeline);
    const bool is_point = cascade_index >= 10;

    // Multiview point-shadow pass renders all six cube faces in one draw, so a
    // single-face frustum test would wrongly cull geometry seen by other faces.
    // Skip per-face culling here (the light's range already bounds coverage).
    if (!is_point) {
      const AABB &bounds = mesh_gpus[draw.mesh_index].bounds();
      const glm::mat4 &m = draw.model;
      const glm::vec3 center = glm::vec3(m * glm::vec4(bounds.center(), 1.0F));
      const glm::vec3 half_extent = (bounds.max - bounds.min) * 0.5F;
      const float world_radius = glm::length(glm::vec3(
          glm::length(glm::vec3(m[0])) * half_extent.x,
          glm::length(glm::vec3(m[1])) * half_extent.y,
          glm::length(glm::vec3(m[2])) * half_extent.z));

      // Frustum cull using Gribb/Hartmann planes (Godot: normalize normals).
      const glm::vec4 r0(light_vp[0][0], light_vp[1][0], light_vp[2][0], light_vp[3][0]);
      const glm::vec4 r1(light_vp[0][1], light_vp[1][1], light_vp[2][1], light_vp[3][1]);
      const glm::vec4 r2(light_vp[0][2], light_vp[1][2], light_vp[2][2], light_vp[3][2]);
      const glm::vec4 r3(light_vp[0][3], light_vp[1][3], light_vp[2][3], light_vp[3][3]);

      auto norm = [](glm::vec4 p) { return p / glm::length(glm::vec3(p)); };
      const glm::vec4 planes[6] = {
          norm(r3 + r0), norm(r3 - r0), norm(r3 + r1),
          norm(r3 - r1), norm(r3 + r2), norm(r3 - r2)};

      auto test = [&](const glm::vec4 &p) { return glm::dot(glm::vec3(p), center) + p.w > -world_radius; };
      for (const glm::vec4 &p : planes)
        if (!test(p)) return;
    }

    if (is_point) {
      bind_pipeline(command_buffer, is_skinned ? PipelineId::PointShadowDepthSkinned : PipelineId::PointShadowDepth, state);
    } else if (is_skinned) {
      bind_pipeline(command_buffer, PipelineId::ShadowDepthSkinned, state);
    } else {
      bind_pipeline(command_buffer, PipelineId::ShadowDepth, state);
    }

    const PipelineId shadow_pipeline = is_point
        ? (is_skinned ? PipelineId::PointShadowDepthSkinned : PipelineId::PointShadowDepth)
        : (is_skinned ? PipelineId::ShadowDepthSkinned : PipelineId::ShadowDepth);

    if (is_skinned && draw.bone_instance_index != k_invalid_skin_index) {
      command_buffer.bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics,
          pipelines.layout_for(shadow_pipeline),
          1,
          descriptors.shadow_bone_set(draw.bone_instance_index, frame_index),
          nullptr);
    }

    bind_mesh_buffers(command_buffer, draw.mesh_index, state);

    const TexturedPushConstants push_constants{
        .model = draw.model,
        .shadow_cascade_index = cascade_index,
        .point_light_index = point_light_index,
    };
    command_buffer.pushConstants(
        pipelines.layout_for(shadow_pipeline),
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        0,
        sizeof(TexturedPushConstants),
        &push_constants);

    const MeshGpu &mesh = mesh_gpus[draw.mesh_index];
    command_buffer.drawIndexed(mesh.index_count(), 1, 0, 0, 0);
  }

  void draw_mesh(
      vk::raii::CommandBuffer &command_buffer,
      const DrawCommand &draw,
      DrawBindState &state) const {
    if (draw.mesh_index >= mesh_gpus.size())
      return;

    bind_pipeline(command_buffer, draw.pipeline, state);

    if (is_textured_surface_pipeline(draw.pipeline)) {
      if (!material_descriptor_index(draw))
        return;

      bind_material(command_buffer, draw, state);

      const TexturedPushConstants push_constants{
          .model = draw.model,
          .texture_array_layer = draw.texture_index,
          .sample_texture_array = draw.texture_source == TextureSource::ArrayLayer ? 1U : 0U,
      };
      command_buffer.pushConstants(
          pipelines.layout_for(draw.pipeline),
          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
          0,
          sizeof(TexturedPushConstants),
          &push_constants);
    } else if (draw.pipeline == PipelineId::Background) {
      const TexturedPushConstants push_constants{.model = draw.model};
      command_buffer.pushConstants(
          pipelines.layout_for(draw.pipeline),
          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
          0,
          sizeof(TexturedPushConstants),
          &push_constants);
    }

    bind_mesh_buffers(command_buffer, draw.mesh_index, state);

    const MeshGpu &mesh = mesh_gpus[draw.mesh_index];
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

  void transition_render_color_to_color_attachment(vk::raii::CommandBuffer &command_buffer, PassLayoutState &layouts) const {
    if (render_color_image == nullptr)
      return;

    if (layouts.render_color_layout == vk::ImageLayout::eColorAttachmentOptimal)
      return;

    const vk::PipelineStageFlags2 previous_stage =
        layouts.render_color_layout == vk::ImageLayout::eUndefined ? vk::PipelineStageFlagBits2::eTopOfPipe
                                                                   : vk::PipelineStageFlagBits2::eTransfer;

    transition_image_layout(
        command_buffer,
        render_color_image->image(),
        layouts.render_color_layout,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eColorAttachmentWrite,
        previous_stage,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    layouts.render_color_layout = vk::ImageLayout::eColorAttachmentOptimal;
  }

  void transition_swapchain_to_transfer_dst(
      vk::raii::CommandBuffer &command_buffer,
      std::uint32_t image_index,
      PassLayoutState &layouts) const {
    vk::ImageLayout &tracked_layout = layouts.swapchain_image_layouts.at(image_index);
    const vk::Image image = swapchain.images()[image_index];

    if (tracked_layout == vk::ImageLayout::eTransferDstOptimal)
      return;

    if (tracked_layout == vk::ImageLayout::eUndefined) {
      transition_image_layout(
          command_buffer,
          image,
          vk::ImageLayout::eUndefined,
          vk::ImageLayout::eTransferDstOptimal,
          {},
          vk::AccessFlagBits2::eTransferWrite,
          vk::PipelineStageFlagBits2::eTopOfPipe,
          vk::PipelineStageFlagBits2::eTransfer);
    } else if (tracked_layout == vk::ImageLayout::ePresentSrcKHR) {
      transition_image_layout(
          command_buffer,
          image,
          vk::ImageLayout::ePresentSrcKHR,
          vk::ImageLayout::eTransferDstOptimal,
          {},
          vk::AccessFlagBits2::eTransferWrite,
          vk::PipelineStageFlagBits2::eBottomOfPipe,
          vk::PipelineStageFlagBits2::eTransfer);
    } else {
      transition_image_layout(
          command_buffer,
          image,
          tracked_layout,
          vk::ImageLayout::eTransferDstOptimal,
          vk::AccessFlagBits2::eColorAttachmentWrite,
          vk::AccessFlagBits2::eTransferWrite,
          vk::PipelineStageFlagBits2::eColorAttachmentOutput,
          vk::PipelineStageFlagBits2::eTransfer);
    }

    tracked_layout = vk::ImageLayout::eTransferDstOptimal;
  }

  void blit_render_color_to_swapchain(
      vk::raii::CommandBuffer &command_buffer,
      std::uint32_t image_index,
      PassLayoutState &layouts) const {
    transition_image_layout(
        command_buffer,
        render_color_image->image(),
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eTransferSrcOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eTransferRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eTransfer);
    layouts.render_color_layout = vk::ImageLayout::eTransferSrcOptimal;

    transition_swapchain_to_transfer_dst(command_buffer, image_index, layouts);

    const std::array<vk::Offset3D, 2> src_offsets{
        vk::Offset3D{0, 0, 0},
        vk::Offset3D{
            static_cast<int32_t>(render_extent.width),
            static_cast<int32_t>(render_extent.height),
            1,
        },
    };
    const std::array<vk::Offset3D, 2> dst_offsets{
        vk::Offset3D{0, 0, 0},
        vk::Offset3D{
            static_cast<int32_t>(swapchain.extent().width),
            static_cast<int32_t>(swapchain.extent().height),
            1,
        },
    };

    command_buffer.blitImage(
        render_color_image->image(),
        vk::ImageLayout::eTransferSrcOptimal,
        swapchain.images()[image_index],
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageBlit{
            .srcSubresource =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            .srcOffsets = src_offsets,
            .dstSubresource =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            .dstOffsets = dst_offsets,
        },
        vk::Filter::eNearest);
  }

  void transition_swapchain_to_overlay_target(
      vk::raii::CommandBuffer &command_buffer,
      std::uint32_t image_index,
      PassLayoutState &layouts) const {
    transition_image_layout(
        command_buffer,
        swapchain.images()[image_index],
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlagBits2::eTransferWrite,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    layouts.swapchain_image_layouts[image_index] = vk::ImageLayout::eColorAttachmentOptimal;
  }

  [[nodiscard]] auto make_main_color_attachment(
      std::uint32_t image_index,
      const vk::ClearValue &clear_value) const -> vk::RenderingAttachmentInfo {
    const vk::ImageView color_target = main_pass_color_target(image_index);

    if (msaa_enabled) {
      return vk::RenderingAttachmentInfo{
          .imageView = *msaa_color_image.view(),
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .resolveMode = vk::ResolveModeFlagBits::eAverage,
          .resolveImageView = color_target,
          .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eClear,
          .storeOp = vk::AttachmentStoreOp::eStore,
          .clearValue = clear_value,
      };
    }

    return vk::RenderingAttachmentInfo{
        .imageView = color_target,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clear_value,
    };
  }

  void record_scene_rendering(
      vk::raii::CommandBuffer &command_buffer,
      std::uint32_t frame_index,
      const vk::RenderingAttachmentInfo &color_attachment,
      const vk::RenderingAttachmentInfo &depth_attachment) const {
    const vk::RenderingInfo rendering_info{
        .renderArea = {.offset = {.x = 0, .y = 0}, .extent = render_extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment,
        .pDepthAttachment = &depth_attachment,
    };
    command_buffer.beginRendering(rendering_info);
    bind_pass_descriptors(command_buffer, frame_index, true);
    command_buffer.setViewport(
        0,
        vk::Viewport{
            0.0F,
            0.0F,
            static_cast<float>(render_extent.width),
            static_cast<float>(render_extent.height),
            0.0F,
            1.0F,
        });
    command_buffer.setScissor(0, vk::Rect2D{{0, 0}, render_extent});
  }

  void record_overlay_pass(
      vk::raii::CommandBuffer &command_buffer,
      std::uint32_t frame_index,
      std::uint32_t image_index,
      const FrameOverlayCallback &overlay) const {
    const vk::RenderingAttachmentInfo overlay_color{
        .imageView = *swapchain.image_views()[image_index],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    const vk::RenderingInfo overlay_info{
        .renderArea = {.offset = {.x = 0, .y = 0}, .extent = swapchain.extent()},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &overlay_color,
    };
    command_buffer.beginRendering(overlay_info);
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
    overlay(FrameOverlayContext{
        .command_buffer = command_buffer,
        .frame_index = frame_index,
        .image_index = image_index,
        .extent = swapchain.extent(),
    });
    command_buffer.endRendering();
  }

   void record_shadow_pass(
       vk::raii::CommandBuffer &command_buffer,
       std::uint32_t frame_index,
       PassLayoutState &layouts,
       const std::vector<DrawCommand> &shadow_draws,
       const std::array<glm::mat4, k_max_shadow_cascades> &cascade_vps) const {
    if (layouts.shadow_image_layout != vk::ImageLayout::eDepthAttachmentOptimal) {
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
          shadow_map.aspect_mask(),
          0,
          1,
          0,
          shadow_map.layer_count());
      layouts.shadow_image_layout = vk::ImageLayout::eDepthAttachmentOptimal;
    }

    for (std::uint32_t cascade_index = 0; cascade_index < shadow_cascade_count; ++cascade_index) {
      const vk::ClearValue clear_depth{vk::ClearDepthStencilValue{1.0F, 0}};
      const vk::RenderingAttachmentInfo depth_attachment{
          .imageView = shadow_map.layer_view(cascade_index),
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
      bind_pass_descriptors(command_buffer, frame_index);
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

      DrawBindState bind_state{};
      bind_state.frame_index = frame_index;
      for (const DrawCommand &draw : shadow_draws) {
        const glm::mat4 &vp = cascade_vps[cascade_index];
        draw_shadow_mesh(command_buffer, draw, cascade_index, frame_index, vp, bind_state);
      }

      command_buffer.endRendering();
    }

    if (layouts.shadow_image_layout != vk::ImageLayout::eDepthStencilReadOnlyOptimal) {
      transition_image_layout(
          command_buffer,
          shadow_map.image(),
          vk::ImageLayout::eDepthAttachmentOptimal,
          vk::ImageLayout::eDepthStencilReadOnlyOptimal,
          vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
          vk::AccessFlagBits2::eShaderRead,
          vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
          vk::PipelineStageFlagBits2::eFragmentShader,
          shadow_map.aspect_mask(),
          0,
          1,
          0,
          shadow_map.layer_count());

      layouts.shadow_image_layout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
    }
  }

  void record_spot_shadow_pass(
      vk::raii::CommandBuffer &command_buffer,
      std::uint32_t frame_index,
      PassLayoutState &layouts,
      const Scene &scene,
      const std::vector<DrawCommand> &draw_list,
      vk::Image atlas_image,
      vk::ImageView atlas_view,
      std::uint32_t atlas_size = 1024) const {
    DrawBindState bind_state{};
    bind_state.frame_index = frame_index;

    const vk::Extent2D atlas_ext{atlas_size, atlas_size};

    // Barrier: previous layout → depth attachment (skip if already in correct layout)
    if (layouts.spot_atlas_layout != vk::ImageLayout::eDepthAttachmentOptimal) {
      transition_image_layout(command_buffer, atlas_image,
          layouts.spot_atlas_layout, vk::ImageLayout::eDepthAttachmentOptimal,
          layouts.spot_atlas_layout == vk::ImageLayout::eUndefined ? vk::AccessFlagBits2{} : vk::AccessFlagBits2::eShaderRead,
          vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
          layouts.spot_atlas_layout == vk::ImageLayout::eUndefined ? vk::PipelineStageFlagBits2::eTopOfPipe : vk::PipelineStageFlagBits2::eFragmentShader,
          vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
          vk::ImageAspectFlagBits::eDepth, 0, 1);
      layouts.spot_atlas_layout = vk::ImageLayout::eDepthAttachmentOptimal;
    }

    const float num_spot = static_cast<float>(scene.spot_lights().size());
    const float region_h = num_spot > 0.0F ? static_cast<float>(atlas_size) / num_spot : static_cast<float>(atlas_size);

    // Clear the entire atlas once up front (single clear, not per-light).
    {
      vk::RenderingAttachmentInfo clear_attach{};
      clear_attach.imageView = atlas_view;
      clear_attach.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
      clear_attach.loadOp = vk::AttachmentLoadOp::eClear;
      clear_attach.storeOp = vk::AttachmentStoreOp::eStore;
      clear_attach.clearValue = vk::ClearDepthStencilValue{1.0F, 0};

      vk::RenderingInfo ri{};
      ri.renderArea = vk::Rect2D{{0, 0}, atlas_ext};
      ri.layerCount = 1;
      ri.pDepthAttachment = &clear_attach;

      command_buffer.beginRendering(ri);
      command_buffer.endRendering();
    }

    std::uint32_t light_idx = 0;
    for (std::uint32_t si = 0; si < scene.spot_lights().size(); ++si) {
      const SpotLight &sl = scene.spot_lights()[si];
      if (!sl.casts_shadow) continue;
      if (light_idx >= 4) break; // spotLightViewProj UBO holds 4 matrices (see vulkan_context)

      // Sub-rect: each spotlight slot occupies a vertical strip of the atlas.
      // The sub-rect index matches the spotlight's index in the scene array,
      // which is the same slot index used in the SSBO (atlas_rect UVs).
      const int y_off = static_cast<int>(static_cast<float>(si) * region_h);
      const std::uint32_t rh = static_cast<std::uint32_t>(region_h);
      const vk::Rect2D sub_rect{{0, y_off}, {atlas_ext.width, rh}};

      vk::RenderingAttachmentInfo depth_attach{};
      depth_attach.imageView = atlas_view;
      depth_attach.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
      depth_attach.loadOp = vk::AttachmentLoadOp::eLoad;
      depth_attach.storeOp = vk::AttachmentStoreOp::eStore;

      vk::RenderingInfo ri{};
      ri.renderArea = sub_rect;
      ri.layerCount = 1;
      ri.pDepthAttachment = &depth_attach;

      command_buffer.beginRendering(ri);
      bind_pass_descriptors(command_buffer, frame_index);
      command_buffer.setViewport(0, vk::Viewport{0.0F, static_cast<float>(y_off),
          static_cast<float>(atlas_ext.width), static_cast<float>(rh), 0.0F, 1.0F});
      command_buffer.setScissor(0, sub_rect);
      command_buffer.setDepthBias(k_shadow_depth_bias_constant, 0.0F, k_shadow_depth_bias_slope);

      const std::uint32_t cascade_idx = 2 + light_idx;
      const glm::mat4 spot_vp = LightStorageBuffer::compute_shadow_view_proj(sl);
      for (const DrawCommand &draw : draw_list) {
        if (draw.mesh_index >= mesh_gpus.size()) continue;
        const AABB &bounds = mesh_gpus[draw.mesh_index].bounds();
        const float s0 = glm::length(glm::vec3(draw.model[0]));
        const float s1 = glm::length(glm::vec3(draw.model[1]));
        const float s2 = glm::length(glm::vec3(draw.model[2]));
        const float max_scale = std::max({s0, s1, s2});
        const float world_radius = bounds.radius() * max_scale;
        const glm::vec3 world_center = glm::vec3(draw.model * glm::vec4(bounds.center(), 1.0F));
        const glm::vec3 delta = world_center - sl.position;
        const float r_sum = world_radius + sl.range;
        if (glm::dot(delta, delta) > r_sum * r_sum) continue;
        draw_shadow_mesh(command_buffer, draw, cascade_idx, frame_index, spot_vp, bind_state);
      }

      command_buffer.endRendering();
      ++light_idx;
    }

    // Barrier: depth attachment → shader read (skip if already correct)
    if (layouts.spot_atlas_layout != vk::ImageLayout::eDepthStencilReadOnlyOptimal) {
      transition_image_layout(command_buffer, atlas_image,
          vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eDepthStencilReadOnlyOptimal,
          vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::AccessFlagBits2::eShaderRead,
          vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
          vk::PipelineStageFlagBits2::eFragmentShader,
          vk::ImageAspectFlagBits::eDepth, 0, 1);
      layouts.spot_atlas_layout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
    }
  }

  void record_point_shadow_pass(
      vk::raii::CommandBuffer &command_buffer,
      std::uint32_t frame_index,
      PassLayoutState &layouts,
      const Scene &scene,
      const std::vector<DrawCommand> &draw_list,
      vk::Image cube_image,
      std::span<const vk::raii::ImageView> cube_face_views,
      float face_size) const {
    DrawBindState bind_state{};
    bind_state.frame_index = frame_index;

    const vk::Extent2D face_ext{static_cast<std::uint32_t>(face_size), static_cast<std::uint32_t>(face_size)};
    const vk::ClearValue clear_depth{vk::ClearDepthStencilValue{1.0F, 0}};

    // Barrier: D32 depth → depth attachment (hardware z-test, early-Z active)
    if (layouts.point_cube_layout != vk::ImageLayout::eDepthAttachmentOptimal) {
      transition_image_layout(command_buffer, cube_image,
          layouts.point_cube_layout, vk::ImageLayout::eDepthAttachmentOptimal,
          layouts.point_cube_layout == vk::ImageLayout::eUndefined ? vk::AccessFlagBits2{} : vk::AccessFlagBits2::eShaderRead,
          vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
          layouts.point_cube_layout == vk::ImageLayout::eUndefined ? vk::PipelineStageFlagBits2::eTopOfPipe : vk::PipelineStageFlagBits2::eFragmentShader,
          vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
          vk::ImageAspectFlagBits::eDepth, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS);
      layouts.point_cube_layout = vk::ImageLayout::eDepthAttachmentOptimal;
    }

    // Sascha Willems face rotation matrices (positive X = first face).
    // These match Vulkan's cubemap sampler convention.
    static const std::array<glm::mat4, 6> face_views = [] {
      using namespace glm;
      return std::array<glm::mat4, 6>{{
          rotate(rotate(mat4(1.0F), radians( 90.0F), vec3(0,1,0)), radians(180.0F), vec3(1,0,0)),
          rotate(rotate(mat4(1.0F), radians(-90.0F), vec3(0,1,0)), radians(180.0F), vec3(1,0,0)),
          rotate(mat4(1.0F), radians(-90.0F), vec3(1,0,0)),
          rotate(mat4(1.0F), radians( 90.0F), vec3(1,0,0)),
          rotate(mat4(1.0F), radians(180.0F), vec3(1,0,0)),
          rotate(mat4(1.0F), radians(180.0F), vec3(0,0,1)),
      }};
    }();

    for (std::uint32_t si = 0; si < scene.point_lights().size() && si < cube_face_views.size(); ++si) {
      const PointLight &pl = scene.point_lights()[si];
      if (!pl.casts_shadow) continue;

      vk::RenderingAttachmentInfo depth_attach{};
      depth_attach.imageView = *cube_face_views[si];
      depth_attach.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
      depth_attach.loadOp = vk::AttachmentLoadOp::eClear;
      depth_attach.storeOp = vk::AttachmentStoreOp::eStore;
      depth_attach.clearValue = clear_depth;

      vk::RenderingInfo ri{};
      ri.renderArea = vk::Rect2D{vk::Offset2D{0, 0}, face_ext};
      ri.layerCount = 1;
      ri.viewMask = 0b111111;
      ri.colorAttachmentCount = 0;
      ri.pColorAttachments = nullptr;
      ri.pDepthAttachment = &depth_attach;

      command_buffer.beginRendering(ri);
      bind_pass_descriptors(command_buffer, frame_index);
      command_buffer.setViewport(0, vk::Viewport{0.0F, 0.0F,
          static_cast<float>(face_ext.width), static_cast<float>(face_ext.height), 0.0F, 1.0F});
      command_buffer.setScissor(0, vk::Rect2D{vk::Offset2D{0, 0}, face_ext});
      command_buffer.setDepthBias(k_shadow_depth_bias_constant, 0.0F, k_shadow_depth_bias_slope);

      for (const DrawCommand &draw : draw_list) {
        if (draw.mesh_index >= mesh_gpus.size()) continue;
        // Per-light sphere culling: skip meshes whose world-space bounding
        // sphere doesn't intersect this light's range sphere.
        const AABB &bounds = mesh_gpus[draw.mesh_index].bounds();
        const float s0 = glm::length(glm::vec3(draw.model[0]));
        const float s1 = glm::length(glm::vec3(draw.model[1]));
        const float s2 = glm::length(glm::vec3(draw.model[2]));
        const float max_scale = std::max({s0, s1, s2});
        const float world_radius = bounds.radius() * max_scale;
        const glm::vec3 world_center = glm::vec3(draw.model * glm::vec4(bounds.center(), 1.0F));
        const glm::vec3 delta = world_center - pl.position;
        const float r_sum = world_radius + pl.range;
        if (glm::dot(delta, delta) > r_sum * r_sum) continue;

        draw_shadow_mesh(command_buffer, draw, 10, frame_index, glm::mat4(1.0F), bind_state, si);
      }

      command_buffer.endRendering();
    }

    // Barrier: depth attachment → shader read for comparison sampling in main pass
    if (layouts.point_cube_layout != vk::ImageLayout::eDepthStencilReadOnlyOptimal) {
      transition_image_layout(command_buffer, cube_image,
          vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eDepthStencilReadOnlyOptimal,
          vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::AccessFlagBits2::eShaderRead,
          vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
          vk::PipelineStageFlagBits2::eFragmentShader,
          vk::ImageAspectFlagBits::eDepth, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS);
      layouts.point_cube_layout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
    }
  }  void finish_main_pass(
      vk::raii::CommandBuffer &command_buffer,
      std::uint32_t image_index,
      PassLayoutState &layouts) const {
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

  void record_main_pass(
      vk::raii::CommandBuffer &command_buffer,
      std::uint32_t frame_index,
      std::uint32_t image_index,
      PassLayoutState &layouts,
      const std::vector<DrawCommand> &draw_list,
      const FrameOverlayCallback *overlay = nullptr) const {
    if (uses_render_scale())
      transition_render_color_to_color_attachment(command_buffer, layouts);
    else
      transition_swapchain_to_color_attachment(command_buffer, image_index, layouts);

    if (msaa_enabled)
      transition_msaa_to_color_attachment(command_buffer, layouts);

    transition_depth_to_attachment(command_buffer, layouts);

    const vk::ClearValue clear_value{vk::ClearColorValue{std::array{0.02F, 0.03F, 0.05F, 1.0F}}};
    const vk::ClearValue clear_depth{vk::ClearDepthStencilValue{1.0F, 0}};
    const vk::RenderingAttachmentInfo color_attachment = make_main_color_attachment(image_index, clear_value);
    const vk::RenderingAttachmentInfo depth_attachment{
        .imageView = *depth_image.view(),
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = clear_depth,
    };

    record_scene_rendering(command_buffer, frame_index, color_attachment, depth_attachment);

    DrawBindState bind_state{};
    bind_state.frame_index = frame_index;
    for (const DrawCommand &draw : draw_list)
      draw_mesh(command_buffer, draw, bind_state);

    command_buffer.endRendering();

    if (uses_render_scale()) {
      blit_render_color_to_swapchain(command_buffer, image_index, layouts);
      if (overlay != nullptr && *overlay)
        transition_swapchain_to_overlay_target(command_buffer, image_index, layouts);
      else {
        transition_image_layout(
            command_buffer,
            swapchain.images()[image_index],
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eTransferWrite,
            {},
            vk::PipelineStageFlagBits2::eTransfer,
            vk::PipelineStageFlagBits2::eBottomOfPipe);
        layouts.swapchain_image_layouts[image_index] = vk::ImageLayout::ePresentSrcKHR;
        command_buffer.end();
        return;
      }
    }

    if (overlay != nullptr && *overlay)
      record_overlay_pass(command_buffer, frame_index, image_index, *overlay);

    finish_main_pass(command_buffer, image_index, layouts);
  }

  void draw_particles(
      vk::raii::CommandBuffer &command_buffer,
      std::uint32_t frame_index,
      std::uint32_t active_count,
      vk::Pipeline pipeline,
      vk::PipelineLayout particle_layout,
      glm::mat4 view_proj,
      glm::vec3 cam_right,
      glm::vec3 cam_up,
      float size,
      glm::vec4 color) const {
    if (active_count == 0) return;

    // Push constants: viewProj(64) + camRight(16) + camUp(16) + size(4) + color(16) = 116 bytes
    struct ParticlePC {
      glm::mat4 viewProj;
      glm::vec4 camRight;
      glm::vec4 camUp;
      float size;
      float _pad[3];
      glm::vec4 color;
    } pc{};
    pc.viewProj = view_proj;
    pc.camRight = glm::vec4(cam_right, 0.0F);
    pc.camUp = glm::vec4(cam_up, 0.0F);
    pc.size = size;
    pc.color = color;

    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
    command_buffer.pushConstants<ParticlePC>(
        particle_layout,
        vk::ShaderStageFlagBits::eVertex,
        0,
        pc);
    command_buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        particle_layout,
        0,
        descriptors.frame_set(frame_index),
        nullptr);
    command_buffer.draw(3, active_count, 0, 0);
  }
};

} // namespace engine
