#pragma once

#include "renderer/frame_overlay.hpp"
#include "renderer/bone_buffer.hpp"
#include "renderer/descriptors.hpp"
#include "renderer/draw_list.hpp"
#include "renderer/frustum.hpp"
#include "renderer/image_barrier.hpp"
#include "renderer/light_buffer.hpp"
#include "renderer/scene_gpu.hpp"
#include "renderer/msaa_color_image.hpp"
#include "renderer/pipeline_id.hpp"
#include "renderer/profiler.hpp"
#include "renderer/pipeline_registry.hpp"
#include "renderer/render_color_image.hpp"
#include "renderer/shadow_map.hpp"
#include "renderer/swapchain.hpp"
#include "renderer/texture_array.hpp"
#include "renderer/texture_table.hpp"
#include "renderer/textured_push_constants.hpp"
#include "renderer/depth_image.hpp"
#include "scene/shadow_assignment.hpp"
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

// Per-frame counters. Optimising a renderer without them is guesswork, and the
// numbers are almost free: they are incremented on paths that already branch.
struct RenderStats {
  std::uint32_t draws_submitted{};   // main pass instances that survived culling
  std::uint32_t draws_culled{};      // main pass instances rejected by the frustum
  std::uint32_t batches_submitted{}; // main pass drawIndexed calls actually recorded
  std::uint32_t shadow_draws_submitted{};
  std::uint32_t shadow_draws_culled{};
  std::uint32_t shadow_batches_submitted{};

  void reset() { *this = {}; }

  [[nodiscard]] auto main_pass_total() const -> std::uint32_t {
    return draws_submitted + draws_culled;
  }
};

struct DrawBindState {
  static constexpr std::uint32_t k_unbound_mesh = UINT32_MAX;

  // Identifies what is currently bound to set 1.
  //
  // The bone slot is deliberately NOT part of this any more: the skinned set now
  // binds the whole palette, and each instance selects its slice by an index in
  // its instance record, so two instances of the same character share both the
  // set and the binding. (It had to be in the key while the slice was chosen by
  // dynamic offset -- leaving it out then made every same-texture skinned
  // instance render with the first one's pose.)
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
  // Brackets one pass with GPU timestamps. Nested passes are not supported --
  // each zone is one contiguous span of the command buffer.
  class ScopedGpuZone {
  public:
    ScopedGpuZone(const PassRecorder &recorder, vk::raii::CommandBuffer &command_buffer, GpuZone zone)
        : recorder_(recorder), command_buffer_(command_buffer), zone_(zone) {
      if (recorder_.gpu_profiler != nullptr)
        recorder_.gpu_profiler->begin_zone(command_buffer_, recorder_.profile_frame_index, zone_);
    }
    ~ScopedGpuZone() { close(); }

    // A timestamp cannot be written after vkEndCommandBuffer, and
    // record_main_pass ends the command buffer at more than one exit, so that
    // pass closes its zone explicitly rather than leaving it to scope exit.
    void close() {
      if (closed_ || recorder_.gpu_profiler == nullptr)
        return;
      recorder_.gpu_profiler->end_zone(command_buffer_, recorder_.profile_frame_index, zone_);
      closed_ = true;
    }

    ScopedGpuZone(const ScopedGpuZone &) = delete;
    auto operator=(const ScopedGpuZone &) -> ScopedGpuZone & = delete;
    ScopedGpuZone(ScopedGpuZone &&) = delete;
    auto operator=(ScopedGpuZone &&) -> ScopedGpuZone & = delete;

  private:
    const PassRecorder &recorder_;
    vk::raii::CommandBuffer &command_buffer_;
    GpuZone zone_;
    bool closed_{false};
  };

  const PipelineRegistry &pipelines;
  const DescriptorResources &descriptors;
  const Swapchain &swapchain;
  const DepthImage &depth_image;
  const MsaaColorImage &msaa_color_image;
  const ShadowMap &shadow_map;
  const TextureTable &texture_table;
  const TextureArray &texture_array;
  const std::vector<MeshGpuSlot> &mesh_gpus;
  const BonePalette *bone_palette{nullptr};
  bool msaa_enabled{};
  std::uint32_t shadow_cascade_count{1};
  vk::Extent2D render_extent{};
  const RenderColorImage *render_color_image{nullptr};
  RenderStats *stats{nullptr};
  // Scratch for cull-then-batch, owned by the context so recording allocates nothing.
  std::vector<DrawCommand> *visible_draws{nullptr};
  std::vector<DrawCommand> *shadow_visible_draws{nullptr};
  GpuProfiler *gpu_profiler{nullptr};
  std::uint32_t profile_frame_index{0};

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

  // A zero radius means the draw opted out of culling (skinned meshes animate
  // outside their bind-pose bounds; background geometry ignores the camera
  // translation entirely).
  // Walks a sorted draw list and hands each batchable run to `emit` as
  // (first draw, count). Every pass shares this so the batching rule lives in
  // one place -- a pass that grouped draws slightly differently from the way the
  // instance records were laid out would render the wrong objects.
  template <typename EmitFn>
  static void for_each_batch(const std::vector<DrawCommand> &draws, EmitFn &&emit) {
    std::size_t i = 0;
    while (i < draws.size()) {
      std::size_t end = i + 1;
      while (end < draws.size() && draws_can_batch(draws[end - 1], draws[end]))
        ++end;
      emit(draws[i], static_cast<std::uint32_t>(end - i));
      i = end;
    }
  }

  // Selects the shadow casters for one light or cascade into `out`, so the
  // survivors are contiguous and therefore batchable. `cull` is null for the
  // multiview point-shadow pass: it renders all six cube faces in one draw, so a
  // single-face frustum test would wrongly reject geometry visible from another
  // face (the light's range already bounds it).
  void select_shadow_casters(const std::vector<DrawCommand> &draws, const Frustum *cull,
                             const glm::vec3 *light_pos, float light_range,
                             std::vector<DrawCommand> &out) const {
    out.clear();
    for (const DrawCommand &draw : draws) {
      const bool visible = (cull == nullptr || is_visible(draw, *cull)) &&
                           (light_pos == nullptr || affected_by_light(draw, *light_pos, light_range));
      if (!visible) {
        if (stats != nullptr)
          ++stats->shadow_draws_culled;
        continue;
      }
      if (stats != nullptr)
        ++stats->shadow_draws_submitted;
      out.push_back(draw);
    }
  }

  // Null when the slot is out of range or holds no live geometry (a chunk that
  // streamed out this frame, whose instances have not been retired yet).
  [[nodiscard]] auto mesh_for(std::uint32_t mesh_index) const -> const MeshGpu * {
    if (mesh_index >= mesh_gpus.size() || !mesh_gpus[mesh_index].alive)
      return nullptr;
    return &mesh_gpus[mesh_index].gpu;
  }

  // Conservative light-volume test on the draw's precomputed world sphere. A
  // zero radius means bounds are unknown (skinned or background geometry), which
  // must keep the draw rather than drop it.
  [[nodiscard]] static auto affected_by_light(const DrawCommand &draw, const glm::vec3 &light_pos,
                                              float light_range) -> bool {
    if (draw.world_bounds.radius <= 0.0F)
      return true;
    const glm::vec3 delta = draw.world_bounds.center - light_pos;
    const float reach = draw.world_bounds.radius + light_range;
    return glm::dot(delta, delta) <= reach * reach;
  }

  [[nodiscard]] static auto is_visible(const DrawCommand &draw, const Frustum &cull) -> bool {
    return draw.world_bounds.radius <= 0.0F || cull.intersects(draw.world_bounds);
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

  void bind_mesh_buffers(vk::raii::CommandBuffer &command_buffer, const MeshGpu &mesh,
                         std::uint32_t mesh_index, DrawBindState &state) const {
    if (state.mesh_index == mesh_index)
      return;

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

    const bool is_skinned = is_skinned_pipeline(draw.pipeline);
    const bool bind_bone_set = is_skinned && draw.bone_instance_index != k_invalid_skin_index &&
                               bone_palette != nullptr && bone_palette->valid() &&
                               descriptors.has_skinned_sets();

    const DrawBindState::MaterialKey material_key{
        .texture_source = draw.texture_source,
        .descriptor_index = *descriptor_index,
    };

    if (state.material.has_value() && *state.material == material_key)
      return;

    if (bind_bone_set) {
      command_buffer.bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics,
          pipelines.layout_for(draw.pipeline),
          1,
          descriptors.skinned_set(*descriptor_index),
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

  // Which shadow pass is recording. This used to be inferred from the cascade
  // index -- "cascade_index >= 10 means point light" -- with the point pass
  // passing a literal 10 to say so.
  //
  // That works only while no other pass can reach 10, and the spot pass numbers
  // its cascades 2 + slot. The ninth spot shadow slot is cascade 10, so it bound
  // the point-shadow pipeline inside the spot atlas pass: a multiview pipeline
  // (viewMask 0x3f) in a pass rendering a single view. Nothing had nine spot
  // lights, so nobody found out. Passing the kind removes the collision instead
  // of moving the constant somewhere it collides less often.
  enum class ShadowPassKind : std::uint8_t { Directional, Spot, Point };

   void draw_shadow_mesh(
       vk::raii::CommandBuffer &command_buffer,
       const DrawCommand &draw,
       std::uint32_t cascade_index,
       std::uint32_t frame_index,
       const glm::mat4 &light_vp,
       DrawBindState &state,
       ShadowPassKind kind,
       std::uint32_t instance_count = 1,
       std::uint32_t point_light_index = 0) const {
    const MeshGpu *mesh = mesh_for(draw.mesh_index);
    if (mesh == nullptr)
      return;

    const bool is_skinned = is_skinned_pipeline(draw.pipeline);
    const bool is_point = kind == ShadowPassKind::Point;

    const PipelineId shadow_pipeline = is_point
        ? (is_skinned ? PipelineId::PointShadowDepthSkinned : PipelineId::PointShadowDepth)
        : (is_skinned ? PipelineId::ShadowDepthSkinned : PipelineId::ShadowDepth);
    bind_pipeline(command_buffer, shadow_pipeline, state);

    if (is_skinned && draw.bone_instance_index != k_invalid_skin_index &&
        bone_palette != nullptr && bone_palette->valid() && descriptors.has_skinned_sets()) {
      command_buffer.bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics,
          pipelines.layout_for(shadow_pipeline),
          1,
          descriptors.shadow_bone_set(),
          nullptr);
    }

    bind_mesh_buffers(command_buffer, *mesh, draw.mesh_index, state);

    const TexturedPushConstants push_constants{
        .instance_base = draw.instance_index,
        .shadow_cascade_index = cascade_index,
        .point_light_index = point_light_index,
    };
    command_buffer.pushConstants(
        pipelines.layout_for(shadow_pipeline),
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        0,
        sizeof(TexturedPushConstants),
        &push_constants);

    command_buffer.drawIndexed(mesh->index_count(), instance_count, 0, 0, 0);
  }

  void draw_mesh(
      vk::raii::CommandBuffer &command_buffer,
      const DrawCommand &draw,
      DrawBindState &state,
      std::uint32_t instance_count = 1) const {
    const MeshGpu *mesh = mesh_for(draw.mesh_index);
    if (mesh == nullptr)
      return;

    bind_pipeline(command_buffer, draw.pipeline, state);

    if (is_textured_surface_pipeline(draw.pipeline)) {
      if (!material_descriptor_index(draw))
        return;

      bind_material(command_buffer, draw, state);

      const TexturedPushConstants push_constants{.instance_base = draw.instance_index};
      command_buffer.pushConstants(
          pipelines.layout_for(draw.pipeline),
          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
          0,
          sizeof(TexturedPushConstants),
          &push_constants);
    } else if (draw.pipeline == PipelineId::Background) {
      const TexturedPushConstants push_constants{.instance_base = draw.instance_index};
      command_buffer.pushConstants(
          pipelines.layout_for(draw.pipeline),
          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
          0,
          sizeof(TexturedPushConstants),
          &push_constants);
    }

    bind_mesh_buffers(command_buffer, *mesh, draw.mesh_index, state);
    command_buffer.drawIndexed(mesh->index_count(), instance_count, 0, 0, 0);
  }

  // The stage a barrier must name as its SOURCE the first time a frame touches
  // an acquired swapchain image.
  //
  // The frame's submit waits on the image-available semaphore at
  // COLOR_ATTACHMENT_OUTPUT (see the SubmitInfo2 in VulkanContext::draw_frame).
  // A semaphore wait only orders work against the stages it names, so a barrier
  // that wants to be ordered after the acquire has to name that same stage as
  // its source. Anything else -- and this used to say TOP_OF_PIPE, then
  // BOTTOM_OF_PIPE -- creates no execution dependency with the wait at all, and
  // the layout transition is free to write the image while the presentation
  // engine is still reading it.
  //
  // TOP_OF_PIPE as a source stage is the specific trap: it reads like "the
  // earliest point", but as a source it means "wait for nothing", so the
  // barrier it guards is unordered against everything before it. Found by
  // synchronization validation as a WRITE_AFTER_READ against
  // vkAcquireNextImageKHR; it renders correctly on hardware that happens to
  // schedule the transition late, which is why it survived this long.
  static constexpr vk::PipelineStageFlags2 k_acquire_wait_stage =
      vk::PipelineStageFlagBits2::eColorAttachmentOutput;

  void transition_swapchain_to_color_attachment(
      vk::raii::CommandBuffer &command_buffer,
      std::uint32_t image_index,
      PassLayoutState &layouts) const {
    vk::ImageLayout &tracked_layout = layouts.swapchain_image_layouts.at(image_index);
    const vk::Image image = swapchain.images()[image_index];

    if (tracked_layout == vk::ImageLayout::eUndefined ||
        tracked_layout == vk::ImageLayout::ePresentSrcKHR) {
      transition_image_layout(
          command_buffer,
          image,
          tracked_layout,
          vk::ImageLayout::eColorAttachmentOptimal,
          {},
          vk::AccessFlagBits2::eColorAttachmentWrite,
          k_acquire_wait_stage,
          vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    }

    tracked_layout = vk::ImageLayout::eColorAttachmentOptimal;
  }

  void transition_msaa_to_color_attachment(vk::raii::CommandBuffer &command_buffer, PassLayoutState &layouts) const {
    if (layouts.msaa_color_layout == vk::ImageLayout::eColorAttachmentOptimal) {
      // Self-barrier to order frame N's writes vs frame N+1's writes on shared image
      transition_image_layout(
          command_buffer, msaa_color_image.image(),
          vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal,
          vk::AccessFlagBits2::eColorAttachmentWrite, vk::AccessFlagBits2::eColorAttachmentWrite,
          vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
      return;
    }

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

    if (tracked_layout == vk::ImageLayout::eUndefined ||
        tracked_layout == vk::ImageLayout::ePresentSrcKHR) {
      // Same hazard as transition_swapchain_to_color_attachment, and the same
      // reason: this is the first touch of a freshly acquired image, so it must
      // be ordered against the semaphore wait. See k_acquire_wait_stage.
      //
      // This path only runs under ENGINE_RENDER_SCALE > 1, which is why
      // synchronization validation did not report it on a default run -- the
      // bug was there either way.
      transition_image_layout(
          command_buffer,
          image,
          tracked_layout,
          vk::ImageLayout::eTransferDstOptimal,
          {},
          vk::AccessFlagBits2::eTransferWrite,
          k_acquire_wait_stage,
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
    const ScopedGpuZone gpu_zone(*this, command_buffer, GpuZone::ShadowDirectional);
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
      const glm::mat4 &vp = cascade_vps[cascade_index];
      const Frustum cascade_cull = Frustum::from_view_proj(vp);
      select_shadow_casters(shadow_draws, &cascade_cull, nullptr, 0.0F, *shadow_visible_draws);
      for_each_batch(*shadow_visible_draws, [&](const DrawCommand &first, std::uint32_t count) {
        draw_shadow_mesh(command_buffer, first, cascade_index, frame_index, vp, bind_state,
                         ShadowPassKind::Directional, count);
        if (stats != nullptr)
          ++stats->shadow_batches_submitted;
      });

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
      std::uint32_t atlas_size,
      const ShadowSlotAssignment &shadows,
      std::uint32_t spot_shadow_capacity) const {
    const ScopedGpuZone gpu_zone(*this, command_buffer, GpuZone::ShadowSpot);
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

    // Barrier: clear writes → first light reads
    transition_image_layout(command_buffer, atlas_image,
        vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eDepthAttachmentOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eLateFragmentTests, vk::PipelineStageFlagBits2::eEarlyFragmentTests,
        vk::ImageAspectFlagBits::eDepth);

    std::uint32_t rendered = 0;
    for (std::uint32_t si = 0; si < scene.spot_lights().size(); ++si) {
      const SpotLight &sl = scene.spot_lights()[si];

      // Render into the slot the assignment gave this light, and skip it
      // entirely when it has none. That is the same rule the point pass already
      // follows, and it brings the same two benefits: the tile the shader
      // samples is the tile that was drawn, and a light whose sphere of
      // influence misses the camera costs nothing. Previously this loop ignored
      // the assignment completely and re-derived its own index, so every
      // shadow-casting spot light in the scene was rendered whether or not
      // anything it lit was on screen.
      const std::int32_t slot =
          si < shadows.spot_slots.size() ? shadows.spot_slots[si] : k_no_shadow_slot;
      if (slot == k_no_shadow_slot)
        continue;

      const SpotAtlasTile tile =
          spot_atlas_tile(static_cast<std::uint32_t>(slot), spot_shadow_capacity);
      const auto tile_size = static_cast<std::uint32_t>(tile.width * static_cast<float>(atlas_size));
      const auto x_off = static_cast<int>(tile.u * static_cast<float>(atlas_size));
      const int y_off = static_cast<int>(tile.v * static_cast<float>(atlas_size));
      const vk::Rect2D sub_rect{{x_off, y_off}, {tile_size, tile_size}};

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
      // Square viewport, matching the square projection compute_shadow_view_proj
      // builds (aspect 1.0). The old strip was 1024 wide by 1024/N tall, which
      // squeezed that square projection flat and threw away nearly all of its
      // vertical resolution once there was more than one spot light.
      command_buffer.setViewport(0, vk::Viewport{static_cast<float>(x_off), static_cast<float>(y_off),
          static_cast<float>(tile_size), static_cast<float>(tile_size), 0.0F, 1.0F});
      command_buffer.setScissor(0, sub_rect);
      command_buffer.setDepthBias(k_shadow_depth_bias_constant, 0.0F, k_shadow_depth_bias_slope);

      const std::uint32_t cascade_idx = 2 + static_cast<std::uint32_t>(slot);
      const glm::mat4 spot_vp = LightStorageBuffer::compute_shadow_view_proj(sl);
      const Frustum spot_cull = Frustum::from_view_proj(spot_vp);
      select_shadow_casters(draw_list, &spot_cull, &sl.position, sl.range, *shadow_visible_draws);
      for_each_batch(*shadow_visible_draws, [&](const DrawCommand &first, std::uint32_t count) {
        draw_shadow_mesh(command_buffer, first, cascade_idx, frame_index, spot_vp, bind_state,
                         ShadowPassKind::Spot, count);
        if (stats != nullptr)
          ++stats->shadow_batches_submitted;
      });

      command_buffer.endRendering();
      ++rendered;
      // Between tiles only. The last one is followed by the read-only
      // transition below, which already orders the writes against the sampling.
      if (rendered < shadows.spot_count)
        transition_image_layout(command_buffer, atlas_image,
            vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eDepthAttachmentOptimal,
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits2::eLateFragmentTests, vk::PipelineStageFlagBits2::eEarlyFragmentTests,
            vk::ImageAspectFlagBits::eDepth);
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
      float face_size,
      const ShadowSlotAssignment &shadows) const {
    const ScopedGpuZone gpu_zone(*this, command_buffer, GpuZone::ShadowPoint);
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

    // Iterate lights, but render into the slot the assignment gave each one --
    // never into its scene index. Lights with no slot were skipped because
    // nothing they illuminate is on screen, so their cubemap is not sampled.
    for (std::uint32_t light_index = 0; light_index < scene.point_lights().size(); ++light_index) {
      const std::int32_t slot = light_index < shadows.point_slots.size()
          ? shadows.point_slots[light_index]
          : k_no_shadow_slot;
      if (slot == k_no_shadow_slot)
        continue;
      const auto si = static_cast<std::uint32_t>(slot);
      if (si >= cube_face_views.size())
        continue;
      const PointLight &pl = scene.point_lights()[light_index];

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

      select_shadow_casters(draw_list, nullptr, &pl.position, pl.range, *shadow_visible_draws);
      for_each_batch(*shadow_visible_draws, [&](const DrawCommand &first, std::uint32_t count) {
        draw_shadow_mesh(command_buffer, first, 0, frame_index, glm::mat4(1.0F), bind_state,
                         ShadowPassKind::Point, count, si);
        if (stats != nullptr)
          ++stats->shadow_batches_submitted;
      });

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
  }

  void finish_main_pass(
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
      const Frustum &camera_cull,
      const FrameOverlayCallback *overlay = nullptr,
      std::function<void(vk::raii::CommandBuffer &)> post_geometry = {}) const {
    ScopedGpuZone gpu_zone(*this, command_buffer, GpuZone::MainPass);
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
    // The main pass drew every instance unconditionally until now. With a
    // streaming world most of the scene is behind the camera on any given frame,
    // so this is the cheapest large win available: a plane test per instance
    // against a frustum built once for the pass.
    // Culling runs before batching: a rejected draw in the middle of a run would
    // leave a gap in the instance range, and the shader reads that range
    // contiguously. visible_draws is a member scratch buffer so this does not
    // allocate every frame.
    visible_draws->clear();
    for (const DrawCommand &draw : draw_list) {
      if (!is_visible(draw, camera_cull)) {
        if (stats != nullptr)
          ++stats->draws_culled;
        continue;
      }
      if (stats != nullptr)
        ++stats->draws_submitted;
      visible_draws->push_back(draw);
    }

    for_each_batch(*visible_draws, [&](const DrawCommand &first, std::uint32_t count) {
      draw_mesh(command_buffer, first, bind_state, count);
      if (stats != nullptr)
        ++stats->batches_submitted;
    });

    if (post_geometry) post_geometry(command_buffer);

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
        gpu_zone.close();
        command_buffer.end();
        return;
      }
    }

    if (overlay != nullptr && *overlay) {
      // Self-barrier: main-pass color writes → overlay load on same swapchain image
      transition_image_layout(command_buffer, swapchain.images()[image_index],
          vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal,
          vk::AccessFlagBits2::eColorAttachmentWrite,
          vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
          vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
      record_overlay_pass(command_buffer, frame_index, image_index, *overlay);
    }

    gpu_zone.close();
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
      glm::vec3 cam_up) const {
    if (active_count == 0) return;

    // Colour and size left this struct when they became per-particle. See
    // gpu_particle.hpp.
    struct ParticlePC {
      glm::mat4 viewProj;
      glm::vec4 camRight;
      glm::vec4 camUp;
    } pc{};
    pc.viewProj = view_proj;
    pc.camRight = glm::vec4(cam_right, 0.0F);
    pc.camUp = glm::vec4(cam_up, 0.0F);

    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
    command_buffer.pushConstants<ParticlePC>(
        particle_layout,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
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
