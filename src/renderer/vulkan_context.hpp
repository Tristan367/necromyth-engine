#pragma once

#include "engine_config.hpp"
#include "renderer/depth_image.hpp"
#include "renderer/descriptors.hpp"
#include "renderer/draw_list.hpp"
#include "renderer/graphics_pipeline.hpp"
#include "renderer/image_barrier.hpp"
#include "renderer/mesh_gpu.hpp"
#include "renderer/msaa_color_image.hpp"
#include "renderer/pipeline_id.hpp"
#include "renderer/pipeline_registry.hpp"
#include "renderer/render_settings.hpp"
#include "renderer/swapchain.hpp"
#include "renderer/texture_table.hpp"
#include "renderer/uniform_buffer.hpp"
#include "renderer/vertex.hpp"
#include "renderer/vulkan_device.hpp"
#include "scene/mesh_source.hpp"
#include "scene/scene.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_video.h>

#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace engine {

namespace detail {

constexpr auto max_frames_in_flight = 2U;
constexpr auto resize_debounce_ms = 100U;

} // namespace detail

class VulkanContext {
public:
  VulkanContext(SDL_Window *window, const EngineConfig &config, const Scene &scene)
      : window_(window),
        device_(window, config.msaa, config.gpu_device_index) {
    swapchain_.create(device_, window);
    create_pipeline_cache();
    create_msaa_color_image();
    create_depth_image();
    create_command_pool();
    upload_scene_meshes(scene);
    load_scene_textures(scene);
    create_descriptor_set_layout();
    create_uniform_buffers();
    create_descriptor_pool_and_sets();
    create_pipelines();
    create_command_buffers();
    create_sync_objects();

#ifndef NDEBUG
    if (device_.validation_enabled())
      std::cout << "Vulkan validation: enabled\n";
#endif
    if (device_.msaa_enabled())
      std::cout << "MSAA samples: " << vk::to_string(device_.msaa_samples()) << '\n';
    else
      std::cout << "MSAA: off\n";
  }

  VulkanContext(const VulkanContext &) = delete;
  auto operator=(const VulkanContext &) -> VulkanContext & = delete;
  VulkanContext(VulkanContext &&) = delete;
  auto operator=(VulkanContext &&) -> VulkanContext & = delete;

  ~VulkanContext() {
    if (*device_.device() != nullptr)
      device_.wait_idle();
  }

  [[nodiscard]] auto gpu_name() const -> const std::string & {
    return device_.gpu_name();
  }

  void mark_framebuffer_resized() {
    framebuffer_resized_ = true;
    last_resize_ticks_ = SDL_GetTicks();
  }

  void draw_frame(Scene &scene) {
    if (framebuffer_resized_) {
      if (SDL_GetTicks() - last_resize_ticks_ < detail::resize_debounce_ms)
        return;

      recreate_swapchain();
      framebuffer_resized_ = false;
      return;
    }

    if (device_.device().waitForFences(*in_flight_fences_[frame_index_], vk::True, UINT64_MAX) != vk::Result::eSuccess)
      throw std::runtime_error("Failed to wait for frame fence");

    auto [acquire_result, image_index] = swapchain_.handle().acquireNextImage(
        UINT64_MAX,
        *image_available_semaphores_[frame_index_],
        nullptr);

    if (acquire_result == vk::Result::eErrorOutOfDateKHR) {
      recreate_swapchain();
      return;
    }
    if (acquire_result != vk::Result::eSuccess && acquire_result != vk::Result::eSuboptimalKHR)
      throw std::runtime_error("Failed to acquire swapchain image");

    device_.device().resetFences(*in_flight_fences_[frame_index_]);

    struct FrameSubmitGuard {
      VulkanContext *context{};
      bool submitted{false};

      ~FrameSubmitGuard() {
        if (context != nullptr && !submitted)
          context->submit_empty_frame();
      }
    } submit_guard{this};

    const float aspect = static_cast<float>(swapchain_.extent().width) /
                         static_cast<float>(swapchain_.extent().height);
    scene.camera().set_aspect(aspect);

    uniform_buffers_.write(
        frame_index_,
        FrameUniformBufferObject{
            .view = scene.camera().view_matrix(),
            .proj = scene.camera().projection_matrix(),
            .camera_position = glm::vec4(scene.camera().position(), 1.0F),
            .view_sky = view_without_translation(scene.camera().view_matrix()),
        });

    build_draw_list(scene, draw_list_);

    auto &command_buffer = command_buffers_[frame_index_];
    command_buffer.reset();
    record_draw_commands(command_buffer, image_index, draw_list_);

    const vk::SemaphoreSubmitInfo wait_semaphore_info{
        .semaphore = *image_available_semaphores_[frame_index_],
        .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
    };
    const vk::CommandBufferSubmitInfo command_buffer_info{
        .commandBuffer = *command_buffer,
    };
    const vk::SemaphoreSubmitInfo signal_semaphore_info{
        .semaphore = *render_finished_semaphores_[image_index],
        .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
    };
    const vk::SubmitInfo2 submit_info{
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &wait_semaphore_info,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &command_buffer_info,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signal_semaphore_info,
    };

    device_.graphics_queue().submit2(submit_info, *in_flight_fences_[frame_index_]);
    submit_guard.submitted = true;

    const vk::PresentInfoKHR present_info{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*render_finished_semaphores_[image_index],
        .swapchainCount = 1,
        .pSwapchains = &*swapchain_.handle(),
        .pImageIndices = &image_index,
    };

    const vk::Result present_result = device_.present_queue().presentKHR(present_info);
    if (present_result == vk::Result::eErrorOutOfDateKHR)
      recreate_swapchain();
    else if (present_result == vk::Result::eSuboptimalKHR)
      recreate_swapchain();
    else if (present_result != vk::Result::eSuccess)
      throw std::runtime_error("Failed to present swapchain image");

    frame_index_ = (frame_index_ + 1) % detail::max_frames_in_flight;
  }

private:
  void submit_empty_frame() {
    const vk::SubmitInfo2 submit_info{};
    device_.graphics_queue().submit2(submit_info, *in_flight_fences_[frame_index_]);
  }

  void create_pipeline_cache() {
    pipeline_cache_ = vk::raii::PipelineCache(device_.device(), vk::PipelineCacheCreateInfo{});
  }

  void create_msaa_color_image() {
    msaa_color_image_.create(
        device_.physical_device(),
        device_.device(),
        swapchain_.extent(),
        swapchain_.image_format(),
        device_.msaa_samples());
  }

  void create_depth_image() {
    depth_image_.create(
        device_.physical_device(),
        device_.device(),
        swapchain_.extent(),
        device_.msaa_samples());
  }

  void upload_scene_meshes(const Scene &scene) {
    mesh_gpus_.clear();
    mesh_gpus_.reserve(scene.meshes().size());
    for (const MeshSource &mesh : scene.meshes()) {
      MeshGpu gpu{};
      gpu.upload(
          device_.physical_device(),
          device_.device(),
          command_pool_,
          device_.graphics_queue(),
          mesh);
      mesh_gpus_.push_back(std::move(gpu));
    }
  }

  void load_scene_textures(const Scene &scene) {
    if (scene.texture_paths().empty())
      throw std::runtime_error("Scene must provide at least one texture path");

    texture_table_.load_from_paths(
        device_.physical_device(),
        device_.device(),
        command_pool_,
        device_.graphics_queue(),
        scene.texture_paths());
  }

  void create_descriptor_set_layout() {
    descriptor_resources_.create_layout(device_.device());
  }

  void create_uniform_buffers() {
    uniform_buffers_.create(device_.physical_device(), device_.device(), detail::max_frames_in_flight);
  }

  void create_descriptor_pool_and_sets() {
    descriptor_resources_.create_pool(
        device_.device(),
        detail::max_frames_in_flight,
        texture_table_.count());

    std::array<vk::Buffer, detail::max_frames_in_flight> buffers{};
    for (std::uint32_t i = 0; i < detail::max_frames_in_flight; ++i)
      buffers[i] = uniform_buffers_.buffer(i);

    const auto texture_pointers = texture_table_.texture_pointers();
    descriptor_resources_.allocate_sets(
        device_.device(),
        buffers,
        texture_pointers);
  }

  void create_pipelines() {
    pipelines_.create(
        device_.device(),
        swapchain_.image_format(),
        depth_image_.format(),
        ENGINE_SHADER_SPIRV,
        ENGINE_SKY_SHADER_SPIRV,
        descriptor_resources_.layout(),
        device_.msaa_samples(),
        pipeline_cache_);
  }

  void create_command_pool() {
    const vk::CommandPoolCreateInfo create_info{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = device_.queue_families().graphics,
    };

    command_pool_ = vk::raii::CommandPool(device_.device(), create_info);
  }

  void create_command_buffers() {
    const vk::CommandBufferAllocateInfo allocate_info{
        .commandPool = *command_pool_,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = detail::max_frames_in_flight,
    };

    command_buffers_ = vk::raii::CommandBuffers(device_.device(), allocate_info);
  }

  void create_render_finished_semaphores() {
    render_finished_semaphores_.clear();
    render_finished_semaphores_.reserve(swapchain_.image_count());
    for (std::size_t i = 0; i < swapchain_.image_count(); ++i)
      render_finished_semaphores_.emplace_back(device_.device(), vk::SemaphoreCreateInfo{});
  }

  void create_sync_objects() {
    create_render_finished_semaphores();

    image_available_semaphores_.clear();
    in_flight_fences_.clear();

    image_available_semaphores_.reserve(detail::max_frames_in_flight);
    in_flight_fences_.reserve(detail::max_frames_in_flight);
    for (std::size_t i = 0; i < detail::max_frames_in_flight; ++i) {
      image_available_semaphores_.emplace_back(device_.device(), vk::SemaphoreCreateInfo{});
      in_flight_fences_.emplace_back(device_.device(), vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
    }
  }

  void record_draw_commands(
      vk::raii::CommandBuffer &command_buffer,
      std::uint32_t image_index,
      const std::vector<DrawCommand> &draw_list) const {
    command_buffer.begin({});

    transition_image_layout(
        command_buffer,
        swapchain_.images()[image_index],
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput);

    if (device_.msaa_enabled()) {
      transition_image_layout(
          command_buffer,
          msaa_color_image_.image(),
          vk::ImageLayout::eUndefined,
          vk::ImageLayout::eColorAttachmentOptimal,
          {},
          vk::AccessFlagBits2::eColorAttachmentWrite,
          vk::PipelineStageFlagBits2::eColorAttachmentOutput,
          vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    }

    transition_image_layout(
        command_buffer,
        depth_image_.image(),
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthAttachmentOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        depth_image_.aspect_mask());

    const vk::ClearValue clear_value{vk::ClearColorValue{std::array{0.02F, 0.03F, 0.05F, 1.0F}}};
    const vk::ClearValue clear_depth{vk::ClearDepthStencilValue{1.0F, 0}};

    const vk::RenderingAttachmentInfo color_attachment = device_.msaa_enabled()
        ? vk::RenderingAttachmentInfo{
              .imageView = *msaa_color_image_.view(),
              .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
              .resolveMode = vk::ResolveModeFlagBits::eAverage,
              .resolveImageView = *swapchain_.image_views()[image_index],
              .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
              .loadOp = vk::AttachmentLoadOp::eClear,
              .storeOp = vk::AttachmentStoreOp::eStore,
              .clearValue = clear_value,
          }
        : vk::RenderingAttachmentInfo{
              .imageView = *swapchain_.image_views()[image_index],
              .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
              .loadOp = vk::AttachmentLoadOp::eClear,
              .storeOp = vk::AttachmentStoreOp::eStore,
              .clearValue = clear_value,
          };
    const vk::RenderingAttachmentInfo depth_attachment{
        .imageView = *depth_image_.view(),
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = clear_depth,
    };
    const vk::RenderingInfo rendering_info{
        .renderArea = {.offset = {.x = 0, .y = 0}, .extent = swapchain_.extent()},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment,
        .pDepthAttachment = &depth_attachment,
    };
    command_buffer.beginRendering(rendering_info);
    command_buffer.setViewport(
        0,
        vk::Viewport{
            0.0F,
            0.0F,
            static_cast<float>(swapchain_.extent().width),
            static_cast<float>(swapchain_.extent().height),
            0.0F,
            1.0F,
        });
    command_buffer.setScissor(0, vk::Rect2D{{0, 0}, swapchain_.extent()});

    std::optional<PipelineId> bound_pipeline;
    std::optional<std::uint32_t> bound_texture_index;

    for (const DrawCommand &draw : draw_list) {
      if (draw.mesh_index >= mesh_gpus_.size())
        continue;

      if (!bound_pipeline || *bound_pipeline != draw.pipeline) {
        command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines_.pipeline(draw.pipeline));
        bound_pipeline = draw.pipeline;
        bound_texture_index.reset();
      }

      if (draw.pipeline == PipelineId::TexturedMesh) {
        if (!bound_texture_index || *bound_texture_index != draw.texture_index) {
          if (draw.texture_index >= texture_table_.count())
            continue;

          command_buffer.bindDescriptorSets(
              vk::PipelineBindPoint::eGraphics,
              pipelines_.layout(),
              0,
              descriptor_resources_.set(frame_index_, draw.texture_index),
              nullptr);
          bound_texture_index = draw.texture_index;
        }
      } else if (!bound_texture_index) {
        command_buffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            pipelines_.layout(),
            0,
            descriptor_resources_.set(frame_index_, 0),
            nullptr);
        bound_texture_index = 0;
      }

      const MeshGpu &mesh = mesh_gpus_[draw.mesh_index];
      command_buffer.pushConstants(
          pipelines_.layout(),
          vk::ShaderStageFlagBits::eVertex,
          0,
          sizeof(glm::mat4),
          &draw.model);

      const vk::Buffer vertex_buffers[] = {mesh.vertex_buffer()};
      const vk::DeviceSize offsets[] = {0};
      command_buffer.bindVertexBuffers(0, vertex_buffers, offsets);
      command_buffer.bindIndexBuffer(mesh.index_buffer(), 0, vk::IndexType::eUint32);
      command_buffer.drawIndexed(mesh.index_count(), 1, 0, 0, 0);
    }

    command_buffer.endRendering();

    transition_image_layout(
        command_buffer,
        swapchain_.images()[image_index],
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        {},
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eBottomOfPipe);

    command_buffer.end();
  }

  void wait_for_nonzero_extent() const {
    int width{};
    int height{};
    while (true) {
      if (!SDL_GetWindowSizeInPixels(window_, &width, &height))
        throw std::runtime_error(std::string("Failed to get SDL window pixel size: ") + SDL_GetError());

      if (width > 0 && height > 0)
        return;

      SDL_WaitEvent(nullptr);
    }
  }

  void recreate_swapchain() {
    wait_for_nonzero_extent();
    device_.wait_idle();
    swapchain_.recreate();
    create_msaa_color_image();
    depth_image_.recreate(swapchain_.extent());
    pipelines_.recreate(device_.device(), swapchain_.image_format(), depth_image_.format());
    create_sync_objects();
  }

  SDL_Window *window_{};
  VulkanDevice device_;
  Swapchain swapchain_;
  vk::raii::PipelineCache pipeline_cache_{nullptr};
  vk::raii::CommandPool command_pool_{nullptr};
  vk::raii::CommandBuffers command_buffers_{nullptr};
  std::vector<vk::raii::Semaphore> image_available_semaphores_;
  std::vector<vk::raii::Semaphore> render_finished_semaphores_;
  std::vector<vk::raii::Fence> in_flight_fences_;
  MsaaColorImage msaa_color_image_;
  DepthImage depth_image_;
  TextureTable texture_table_;
  std::vector<MeshGpu> mesh_gpus_;
  UniformBufferSet uniform_buffers_;
  DescriptorResources descriptor_resources_;
  PipelineRegistry pipelines_;
  std::vector<DrawCommand> draw_list_;
  std::uint32_t frame_index_{};
  bool framebuffer_resized_{};
  std::uint64_t last_resize_ticks_{};
};

} // namespace engine
