#pragma once

#include "renderer/buffer.hpp"
#include "renderer/depth_image.hpp"
#include "renderer/descriptors.hpp"
#include "renderer/graphics_pipeline.hpp"
#include "renderer/image_barrier.hpp"
#include "renderer/model_loader.hpp"
#include "renderer/msaa_color_image.hpp"
#include "renderer/swapchain.hpp"
#include "renderer/texture_image.hpp"
#include "renderer/uniform_buffer.hpp"
#include "renderer/vertex.hpp"
#include "renderer/vulkan_device.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_video.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace engine {

namespace detail {

constexpr auto max_frames_in_flight = 2U;
constexpr auto resize_debounce_ms = 100U;

} // namespace detail

class VulkanContext {
public:
  explicit VulkanContext(SDL_Window *window) : window_(window), device_(window) {
    swapchain_.create(device_, window);
    create_msaa_color_image();
    create_depth_image();
    create_command_pool();
    create_descriptor_set_layout();
    load_model();
    create_texture_image();
    create_vertex_buffer();
    create_index_buffer();
    create_uniform_buffers();
    create_descriptor_pool_and_sets();
    create_graphics_pipeline();
    create_command_buffers();
    create_sync_objects();

#ifndef NDEBUG
    if (device_.validation_enabled())
      std::cout << "Vulkan validation: enabled\n";
#endif
    std::cout << "MSAA samples: " << vk::to_string(device_.msaa_samples()) << '\n';
  }

  VulkanContext(const VulkanContext &) = delete;
  auto operator=(const VulkanContext &) -> VulkanContext & = delete;
  VulkanContext(VulkanContext &&) = delete;
  auto operator=(VulkanContext &&) -> VulkanContext & = delete;

  [[nodiscard]] auto gpu_name() const -> const std::string & {
    return device_.gpu_name();
  }

  void mark_framebuffer_resized() {
    framebuffer_resized_ = true;
    last_resize_ticks_ = SDL_GetTicks();
  }

  void draw_frame() {
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

    uniform_buffers_.write(
        frame_index_,
        UniformBufferSet::make_rotating_ubo(swapchain_.extent()));

    auto &command_buffer = command_buffers_[frame_index_];
    command_buffer.reset();
    record_draw_commands(command_buffer, image_index);

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

    const vk::PresentInfoKHR present_info{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*render_finished_semaphores_[image_index],
        .swapchainCount = 1,
        .pSwapchains = &*swapchain_.handle(),
        .pImageIndices = &image_index,
    };

    const vk::Result present_result = device_.present_queue().presentKHR(present_info);
    if (present_result == vk::Result::eErrorOutOfDateKHR || present_result == vk::Result::eSuboptimalKHR)
      recreate_swapchain();
    else if (present_result != vk::Result::eSuccess)
      throw std::runtime_error("Failed to present swapchain image");

    frame_index_ = (frame_index_ + 1) % detail::max_frames_in_flight;
  }

private:
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

  void load_model() {
    mesh_ = load_obj_model(ENGINE_MODEL_PATH);
  }

  void create_texture_image() {
    texture_image_.load_from_file(
        device_.physical_device(),
        device_.device(),
        command_pool_,
        device_.graphics_queue(),
        ENGINE_TEXTURE_PATH);
  }

  void create_descriptor_set_layout() {
    descriptor_resources_.create_layout(device_.device());
  }

  void create_uniform_buffers() {
    uniform_buffers_.create(device_.physical_device(), device_.device(), detail::max_frames_in_flight);
  }

  void create_descriptor_pool_and_sets() {
    descriptor_resources_.create_pool(device_.device(), detail::max_frames_in_flight);

    std::array<vk::Buffer, detail::max_frames_in_flight> buffers{};
    for (std::uint32_t i = 0; i < detail::max_frames_in_flight; ++i)
      buffers[i] = uniform_buffers_.buffer(i);

    descriptor_resources_.allocate_sets(
        device_.device(),
        buffers,
        texture_image_.sampler(),
        texture_image_.view());
  }

  void create_vertex_buffer() {
    vertex_buffer_.upload(
        device_.physical_device(),
        device_.device(),
        command_pool_,
        device_.graphics_queue(),
        static_cast<vk::DeviceSize>(sizeof(MeshVertex) * mesh_.vertices.size()),
        vk::BufferUsageFlagBits::eVertexBuffer,
        mesh_.vertices.data());
  }

  void create_index_buffer() {
    index_count_ = static_cast<std::uint32_t>(mesh_.indices.size());
    index_buffer_.upload(
        device_.physical_device(),
        device_.device(),
        command_pool_,
        device_.graphics_queue(),
        static_cast<vk::DeviceSize>(sizeof(std::uint32_t) * mesh_.indices.size()),
        vk::BufferUsageFlagBits::eIndexBuffer,
        mesh_.indices.data());
  }

  void create_graphics_pipeline() {
    const auto attributes = MeshVertex::attribute_descriptions();
    graphics_pipeline_.create(
        device_.device(),
        swapchain_.image_format(),
        depth_image_.format(),
        ENGINE_SHADER_SPIRV,
        descriptor_resources_.layout(),
        device_.msaa_samples(),
        MeshVertex::binding_description(),
        attributes);
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

  void record_draw_commands(vk::raii::CommandBuffer &command_buffer, std::uint32_t image_index) const {
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

    transition_image_layout(
        command_buffer,
        msaa_color_image_.image(),
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput);

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
    const vk::RenderingAttachmentInfo color_attachment{
        .imageView = *msaa_color_image_.view(),
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .resolveMode = vk::ResolveModeFlagBits::eAverage,
        .resolveImageView = *swapchain_.image_views()[image_index],
        .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
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
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphics_pipeline_.pipeline());
    command_buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *graphics_pipeline_.layout(),
        0,
        descriptor_resources_.set(frame_index_),
        nullptr);
    const vk::Buffer vertex_buffers[] = {vertex_buffer_.handle()};
    const vk::DeviceSize offsets[] = {0};
    command_buffer.bindVertexBuffers(0, vertex_buffers, offsets);
    command_buffer.bindIndexBuffer(index_buffer_.handle(), 0, vk::IndexType::eUint32);
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
    command_buffer.drawIndexed(index_count_, 1, 0, 0, 0);
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
    render_finished_semaphores_.clear();
    swapchain_.recreate();
    msaa_color_image_.recreate(swapchain_.extent());
    depth_image_.recreate(swapchain_.extent());
    create_render_finished_semaphores();
  }

  SDL_Window *window_{};
  VulkanDevice device_;
  Swapchain swapchain_;
  vk::raii::CommandPool command_pool_{nullptr};
  vk::raii::CommandBuffers command_buffers_{nullptr};
  std::vector<vk::raii::Semaphore> image_available_semaphores_;
  std::vector<vk::raii::Semaphore> render_finished_semaphores_;
  std::vector<vk::raii::Fence> in_flight_fences_;
  MsaaColorImage msaa_color_image_;
  DepthImage depth_image_;
  TextureImage texture_image_;
  LoadedMesh mesh_;
  DeviceLocalBuffer vertex_buffer_;
  DeviceLocalBuffer index_buffer_;
  UniformBufferSet uniform_buffers_;
  DescriptorResources descriptor_resources_;
  std::uint32_t index_count_{};
  GraphicsPipeline graphics_pipeline_;
  std::uint32_t frame_index_{};
  bool framebuffer_resized_{};
  std::uint64_t last_resize_ticks_{};
};

} // namespace engine
