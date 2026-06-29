#pragma once

#include "engine_config.hpp"
#include "renderer/bone_buffer.hpp"
#include "renderer/frame_overlay.hpp"
#include "renderer/render_host.hpp"
#include "renderer/depth_image.hpp"
#include "renderer/descriptors.hpp"
#include "renderer/draw_list.hpp"
#include "renderer/msaa_color_image.hpp"
#include "renderer/pass_recorder.hpp"
#include "renderer/pipeline_id.hpp"
#include "renderer/pipeline_registry.hpp"
#include "renderer/render_color_image.hpp"
#include "renderer/scene_gpu.hpp"
#include "renderer/shadow_map.hpp"
#include "renderer/render_settings.hpp"
#include "renderer/swapchain.hpp"
#include "renderer/texture_array.hpp"
#include "renderer/texture_table.hpp"
#include "renderer/uniform_buffer.hpp"
#include "renderer/mesh_gpu.hpp"
#include "renderer/vulkan_device.hpp"
#include "scene/animation_utils.hpp"
#include "scene/scene.hpp"
#include "scene/shadow_utils.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_timer.h>

#include <cstdint>
#include <functional>
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
  VulkanContext(SDL_Window *window, const EngineConfig &config, const Scene &scene)
      : window_(window),
        msaa_config_(resolve_msaa_for_scene(config.msaa, scene_uses_alpha_to_coverage(scene.instances()))),
        msaa_boosted_for_a2c_(!config.msaa.enabled && msaa_config_.enabled),
        startup_point_shadow_filter_(scene.shadow_settings().point_shadow_filter),
        startup_shadow_filter_mode_(scene.shadow_settings().filter_mode),
        startup_cascade_mode_(scene.shadow_settings().cascade_mode),
        startup_render_scale_(config.render_scale),
        startup_shadow_scale_(config.shadow_scale),
        startup_shadow_map_resolution_(scaled_shadow_map_resolution(
            scene.shadow_settings().map_resolution,
            config.shadow_scale)),
        device_(window, msaa_config_, config.gpu_device_index) {
    swapchain_.create(device_, window, config.present_mode);
    create_pipeline_cache();
    create_msaa_color_image();
    create_render_color_image();
    create_depth_image();
    create_shadow_map(
        startup_shadow_map_resolution_,
        shadow_cascade_layer_count(scene.shadow_settings().cascade_mode));
    create_command_pool();
    engine::upload_scene_meshes(
        scene,
        device_.physical_device(),
        device_.device(),
        command_pool_,
        device_.graphics_queue(),
        mesh_gpus_);
    engine::load_scene_textures(
        scene,
        device_.physical_device(),
        device_.device(),
        command_pool_,
        device_.graphics_queue(),
        texture_table_);
    engine::load_texture_array_layers(
        scene,
        device_.physical_device(),
        device_.device(),
        command_pool_,
        device_.graphics_queue(),
        texture_array_);
    create_descriptor_set_layout();
    create_uniform_buffers();
    create_bone_buffers(scene);
    create_descriptor_pool_and_sets(scene);
    create_pipelines(scene);
    create_command_buffers();
    create_sync_objects();
    reset_image_layout_tracking();

#ifndef NDEBUG
    if (device_.validation_enabled())
      std::cout << "Vulkan validation: enabled\n";
#endif
    if (device_.msaa_enabled())
      std::cout << "MSAA samples: " << vk::to_string(device_.msaa_samples()) << '\n';
    else if (!msaa_config_.enabled)
      std::cout << "MSAA: off (set ENGINE_MSAA=4 to enable; A2C foliage needs MSAA)\n";
    else
      std::cout << "MSAA: off (GPU supports up to " << vk::to_string(device_.max_msaa_samples()) << ")\n";
    if (msaa_boosted_for_a2c_)
      std::cout << "MSAA auto-enabled for AlphaToCoverage meshes\n";
    std::cout << "Present mode: " << present_mode_name(swapchain_.present_mode())
              << " (set ENGINE_PRESENT=mailbox to uncap for profiling)\n";
    if (render_scale_active(startup_render_scale_)) {
      const vk::Extent2D internal = render_extent();
      std::cout << "Render scale: " << startup_render_scale_ << " (internal "
                << internal.width << 'x' << internal.height << ", restart to change; ENGINE_RENDER_SCALE)\n";
    }
    if (shadow_scale_active(startup_shadow_scale_))
      std::cout << "Shadow scale: " << startup_shadow_scale_ << " (map "
                << startup_shadow_map_resolution_ << 'x' << startup_shadow_map_resolution_
                << ", restart to change; ENGINE_SHADOW_SCALE)\n";
    else
      std::cout << "Shadow map: " << startup_shadow_map_resolution_ << 'x'
                << startup_shadow_map_resolution_ << '\n';
  }

  [[nodiscard]] auto present_mode() const -> vk::PresentModeKHR {
    return swapchain_.present_mode();
  }

  VulkanContext(const VulkanContext &) = delete;
  auto operator=(const VulkanContext &) -> VulkanContext & = delete;
  VulkanContext(VulkanContext &&) = delete;
  auto operator=(VulkanContext &&) -> VulkanContext & = delete;

  ~VulkanContext() {
    shutdown();
  }

  void shutdown() {
    if (gpu_shutdown_complete_ || *device_.device() == nullptr)
      return;

    for (const vk::raii::Fence &fence : in_flight_fences_) {
      (void)device_.device().waitForFences(*fence, vk::True, UINT64_MAX);
    }

    device_.wait_idle();
    gpu_shutdown_complete_ = true;
  }

  [[nodiscard]] auto gpu_name() const -> const std::string & {
    return device_.gpu_name();
  }

  [[nodiscard]] auto device_ref() -> vk::raii::Device & { return device_.device(); }
  [[nodiscard]] auto color_fmt() const -> vk::Format { return swapchain_.image_format(); }
  [[nodiscard]] auto depth_fmt() const -> vk::Format { return depth_image_.format(); }
  [[nodiscard]] auto frame_layout_obj() const -> vk::DescriptorSetLayout { return descriptor_resources_.frame_layout(); }
  [[nodiscard]] auto sample_count() const -> vk::SampleCountFlagBits { return device_.msaa_samples(); }
  [[nodiscard]] auto frame_set_obj(std::uint32_t i) const -> vk::DescriptorSet { return descriptor_resources_.frame_set(i); }
  [[nodiscard]] auto phys_dev() -> vk::raii::PhysicalDevice { return device_.physical_device(); }
  [[nodiscard]] auto mem_props() -> vk::PhysicalDeviceMemoryProperties { return device_.physical_device().getMemoryProperties(); }

  void mark_framebuffer_resized() {
    framebuffer_resized_ = true;
    last_resize_ticks_ = SDL_GetTicks();
  }

  void set_frame_overlay(FrameOverlayCallback callback) {
    frame_overlay_ = std::move(callback);
  }

  [[nodiscard]] auto render_host_info() const -> RenderHostInfo {
    return RenderHostInfo{
        .instance = *device_.instance(),
        .physical_device = device_.physical_device_handle(),
        .device = device_.device_handle(),
        .graphics_queue = device_.graphics_queue_handle(),
        .graphics_queue_family_index = device_.queue_families().graphics,
        .pipeline_cache = *pipeline_cache_,
        .swapchain_color_format = swapchain_.image_format(),
        .swapchain_extent = swapchain_.extent(),
        .swapchain_image_count = static_cast<std::uint32_t>(swapchain_.image_count()),
        .frames_in_flight = detail::max_frames_in_flight,
    };
  }

  // Upload meshes/textures appended to Scene after construction; rebuilds texture descriptors when needed.
  void sync_scene(const Scene &scene) {
    if (gpu_shutdown_complete_)
      return;

    device_.wait_idle();

    for (std::size_t mesh_index = mesh_gpus_.size(); mesh_index < scene.meshes().size(); ++mesh_index) {
      MeshGpu gpu{};
      gpu.upload(
          device_.physical_device(),
          device_.device(),
          command_pool_,
          device_.graphics_queue(),
          scene.meshes()[mesh_index]);
      mesh_gpus_.push_back(std::move(gpu));
    }

    bool descriptors_changed = false;

    if (scene.texture_paths().size() > texture_table_.count()) {
      for (std::size_t texture_index = texture_table_.count(); texture_index < scene.texture_paths().size();
           ++texture_index) {
        texture_table_.append_from_file(
            device_.physical_device(),
            device_.device(),
            command_pool_,
            device_.graphics_queue(),
            scene.texture_paths()[texture_index]);
      }
      descriptors_changed = true;
    }

    if (count_skinned_instances(scene.instances()) != static_cast<std::uint32_t>(bone_buffers_.size())) {
      create_bone_buffers(scene);
      descriptors_changed = true;
    }

    if (descriptors_changed)
      recreate_descriptor_sets();
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

    const vk::Extent2D internal_extent = render_extent();
    const float aspect = static_cast<float>(internal_extent.width) /
                         static_cast<float>(internal_extent.height);
    scene.camera().set_aspect(aspect);

    const DirectionalLightShadowSettings shadow_settings = effective_shadow_settings(
        scene.shadow_settings(),
        startup_cascade_mode_,
        shadow_map_.extent().width);

    if (!logged_shadow_filter_mode_) {
      std::cout << "Shadow filter: " << shadow_filter_mode_name(startup_shadow_filter_mode_)
                << " (restart to change; ENGINE_SHADOW_FILTER)\n";
      std::cout << "Shadow cascades: " << shadow_cascade_mode_name(startup_cascade_mode_)
                << " (restart to change; ENGINE_SHADOW_CASCADES=1|2)\n";
      logged_shadow_filter_mode_ = true;
    }

    const DirectionalShadowCascadeData cascades = directional_shadow_cascades(
        scene.camera(),
        scene.directional_light(),
        shadow_settings);

    uniform_buffers_.write(
        frame_index_,
        FrameUniformBufferObject{
            .view = scene.camera().view_matrix(),
            .proj = scene.camera().projection_matrix(),
            .camera_position = glm::vec4(scene.camera().position(), 1.0F),
            .view_sky = view_without_translation(scene.camera().view_matrix()),
            .light_direction = glm::vec4(
                glm::normalize(scene.directional_light().direction_toward_light),
                0.0F),
            .light_color = glm::vec4(
                scene.directional_light().color * scene.directional_light().intensity,
                scene.directional_light().ambient),
            .light_view_proj = cascades.light_view_proj,
            .cascade_params = glm::vec4(
                cascades.split_view_z,
                shadow_settings.cascade_blend_range,
                0.0F,
                0.0F),
            .shadow_fade_width = glm::vec4(shadow_settings.coverage_fade_uv_width, 0.0F, 0.0F, 0.0F),
        });

    if (!bone_buffers_.empty()) {
      std::vector<glm::mat4> joint_matrices;
      std::uint32_t bone_buffer_index = 0;
      for (const MeshInstance &instance : scene.instances()) {
        if (instance.skin_index == k_invalid_skin_index || instance.animation_index == k_invalid_skin_index)
          continue;
        if (instance.skin_index >= scene.skeletons().size())
          continue;
        if (instance.animation_index >= scene.animations().size())
          continue;
        if (bone_buffer_index >= bone_buffers_.size())
          break;

        joint_matrices.clear();

        const auto &skel = scene.skeletons()[instance.skin_index];

        if (instance.secondary_joints && !instance.secondary_joints->empty() &&
            instance.next_animation_index < scene.animations().size()) {
          compute_joint_matrices_split(
              skel,
              scene.animations()[instance.animation_index], instance.animation_time,
              scene.animations()[instance.next_animation_index], instance.next_animation_time,
              *instance.secondary_joints,
              joint_matrices);
        } else if (instance.next_animation_index < scene.animations().size()) {
          const AnimationClip &clip = scene.animations()[instance.animation_index];
          const AnimationClip &next_clip = scene.animations()[instance.next_animation_index];

          if (instance.blend_factor < 1.0F)
            compute_joint_matrices_blended(
                skel, clip, instance.animation_time,
                next_clip, instance.next_animation_time,
                instance.blend_factor,
                joint_matrices);
          else
            compute_joint_matrices(
                skel,
                scene.animations()[instance.animation_index],
                instance.animation_time,
                joint_matrices);
        } else {
          compute_joint_matrices(
              skel,
              scene.animations()[instance.animation_index],
              instance.animation_time,
              joint_matrices);
        }

        bone_buffers_[bone_buffer_index].write(frame_index_, joint_matrices);
        ++bone_buffer_index;
      }
    }

    build_draw_list(scene, draw_list_);

    auto &command_buffer = command_buffers_[frame_index_];
    command_buffer.reset();
    command_buffer.begin({});
    pass_recorder().record_shadow_pass(command_buffer, frame_index_, pass_layouts_, draw_list_);
    const FrameOverlayCallback *overlay_ptr = frame_overlay_ ? &frame_overlay_ : nullptr;
    pass_recorder().record_main_pass(
        command_buffer,
        frame_index_,
        image_index,
        pass_layouts_,
        draw_list_,
        overlay_ptr);

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
    else if (present_result != vk::Result::eSuccess && present_result != vk::Result::eSuboptimalKHR)
      throw std::runtime_error("Failed to present swapchain image");

    frame_index_ = (frame_index_ + 1) % detail::max_frames_in_flight;
  }

private:
  [[nodiscard]] auto pass_recorder() const -> PassRecorder {
    return PassRecorder{
        .pipelines = pipelines_,
        .descriptors = descriptor_resources_,
        .swapchain = swapchain_,
        .depth_image = depth_image_,
        .msaa_color_image = msaa_color_image_,
        .shadow_map = shadow_map_,
        .texture_table = texture_table_,
        .texture_array = texture_array_,
        .mesh_gpus = mesh_gpus_,
        .msaa_enabled = device_.msaa_enabled(),
        .shadow_cascade_count = shadow_cascade_layer_count(startup_cascade_mode_),
        .render_extent = render_extent(),
        .render_color_image = render_scale_active(startup_render_scale_) ? &render_color_image_ : nullptr,
    };
  }

  [[nodiscard]] auto render_extent() const -> vk::Extent2D {
    return scaled_render_extent(swapchain_.extent(), startup_render_scale_);
  }

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
        render_extent(),
        swapchain_.image_format(),
        device_.msaa_samples());
  }

  void create_render_color_image() {
    if (!render_scale_active(startup_render_scale_))
      return;

    render_color_image_.create(
        device_.physical_device(),
        device_.device(),
        render_extent(),
        swapchain_.image_format());
  }

  void recreate_render_color_image() {
    if (!render_scale_active(startup_render_scale_))
      return;

    if (render_color_image_.active())
      render_color_image_.recreate(render_extent());
    else
      create_render_color_image();
  }

  void recreate_render_targets() {
    create_msaa_color_image();
    recreate_render_color_image();
    depth_image_.recreate(render_extent());
  }

  void create_shadow_map(std::uint32_t resolution, std::uint32_t layer_count) {
    shadow_map_.create(device_.physical_device(), device_.device(), resolution, layer_count);
  }

  void create_depth_image() {
    depth_image_.create(
        device_.physical_device(),
        device_.device(),
        render_extent(),
        device_.msaa_samples());
  }

  void create_descriptor_set_layout() {
    descriptor_resources_.create_layouts(device_.device());
  }

  void create_uniform_buffers() {
    uniform_buffers_.create(device_.physical_device(), device_.device(), detail::max_frames_in_flight);
  }

  void create_descriptor_pool_and_sets(const Scene &scene) {
    rebuild_descriptor_sets(count_skinned_instances(scene.instances()));
  }

  void rebuild_descriptor_sets(std::uint32_t skinned_count) {
    descriptor_resources_.create_pool(
        device_.device(),
        detail::max_frames_in_flight,
        texture_table_.count(),
        skinned_count);

    std::array<vk::Buffer, detail::max_frames_in_flight> buffers{};
    for (std::uint32_t i = 0; i < detail::max_frames_in_flight; ++i)
      buffers[i] = uniform_buffers_.buffer(i);

    const auto texture_pointers = texture_table_.texture_pointers();
    descriptor_resources_.allocate_sets(
        device_.device(),
        buffers,
        texture_pointers,
        texture_array_.sampler(),
        texture_array_.view(),
        shadow_map_.sampler_for_settings(startup_point_shadow_filter_),
        shadow_map_.view());

    allocate_skinned_descriptor_sets(texture_table_, skinned_count);
  }

  void recreate_descriptor_sets() {
    rebuild_descriptor_sets(bone_buffers_.empty() ? 0U
        : static_cast<std::uint32_t>(bone_buffers_.size()));
  }

  void create_pipelines(const Scene &scene) {
    const PipelineBuildProfile profile{
        .shadow_filter = startup_shadow_filter_mode_,
        .cascade_mode = startup_cascade_mode_,
        .textured_alpha_modes = collect_used_alpha_modes(scene.instances()),
        .build_skinned = has_skinned_instances(scene.instances()),
    };

    std::size_t textured_pipeline_count = 0;
    for (const bool used : profile.textured_alpha_modes) {
      if (used)
        ++textured_pipeline_count;
    }
    const std::size_t skinned_count = profile.build_skinned ? textured_pipeline_count + 1 : 0;
    std::cout << "Graphics pipelines: " << (2 + textured_pipeline_count + skinned_count)
              << " (background + shadow depth + " << textured_pipeline_count << " textured"
              << (profile.build_skinned ? " + " + std::to_string(skinned_count) + " skinned" : "")
              << ")\n";

    pipelines_.create(
        device_.device(),
        swapchain_.image_format(),
        depth_image_.format(),
        shadow_map_.format(),
        ENGINE_SHADER_SPIRV,
        ENGINE_SKY_SHADER_SPIRV,
        ENGINE_SHADOW_DEPTH_SPIRV,
        ENGINE_SKINNED_SHADER_SPIRV,
        ENGINE_SKINNED_SHADOW_DEPTH_SPIRV,
        descriptor_resources_.frame_layout(),
        descriptor_resources_.material_layout(),
        descriptor_resources_.material_skinned_layout(),
        device_.msaa_samples(),
        pipeline_cache_,
        profile);
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

  void reset_image_layout_tracking() {
    pass_layouts_.swapchain_image_layouts.assign(swapchain_.image_count(), vk::ImageLayout::eUndefined);
    pass_layouts_.depth_image_layout = vk::ImageLayout::eUndefined;
    pass_layouts_.msaa_color_layout = vk::ImageLayout::eUndefined;
    pass_layouts_.render_color_layout = vk::ImageLayout::eUndefined;
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
    recreate_render_targets();
    pipelines_.recreate(device_.device(), swapchain_.image_format(), depth_image_.format());
    create_sync_objects();
    reset_image_layout_tracking();
    frame_index_ = 0;
  }

  [[nodiscard]] static auto count_skinned_instances(const std::vector<MeshInstance> &instances) -> std::uint32_t {
    std::uint32_t count = 0;
    for (const MeshInstance &instance : instances) {
      if (instance.skin_index != k_invalid_skin_index)
        ++count;
    }
    return count;
  }

  [[nodiscard]] static auto has_skinned_instances(const std::vector<MeshInstance> &instances) -> bool {
    return count_skinned_instances(instances) > 0;
  }

  void create_bone_buffers(const Scene &scene) {
    bone_buffers_.clear();
    skinned_texture_indices_.clear();
    for (const MeshInstance &instance : scene.instances()) {
      if (instance.skin_index == k_invalid_skin_index)
        continue;

      if (instance.skin_index >= scene.skeletons().size())
        continue;

      const std::uint32_t bone_count =
          static_cast<std::uint32_t>(scene.skeletons()[instance.skin_index].joint_nodes.size());
      if (bone_count == 0)
        continue;

      BoneStorageBufferSet bone_set;
      bone_set.create(
          device_.physical_device(),
          device_.device(),
          bone_count,
          detail::max_frames_in_flight);
      bone_buffers_.push_back(std::move(bone_set));
      skinned_texture_indices_.push_back(instance.texture_index);
    }
  }

  void allocate_skinned_descriptor_sets(const TextureTable &texture_table, std::uint32_t skinned_count) {
    if (skinned_count == 0 || bone_buffers_.empty())
      return;

    std::vector<vk::Buffer> buffers;
    buffers.reserve(bone_buffers_.size() * detail::max_frames_in_flight);
    for (const BoneStorageBufferSet &bone_set : bone_buffers_) {
      for (std::uint32_t frame = 0; frame < detail::max_frames_in_flight; ++frame)
        buffers.push_back(bone_set.buffer(frame));
    }

    descriptor_resources_.allocate_skinned_sets(
        device_.device(),
        buffers,
        texture_table.texture_pointers(),
        skinned_texture_indices_);
  }

  SDL_Window *window_{};
  MsaaSettings msaa_config_{};
  bool msaa_boosted_for_a2c_{false};
  bool startup_point_shadow_filter_{false};
  ShadowFilterMode startup_shadow_filter_mode_{ShadowFilterMode::Pcf3x3};
  ShadowCascadeMode startup_cascade_mode_{ShadowCascadeMode::Dual};
  std::uint32_t startup_render_scale_{1};
  std::uint32_t startup_shadow_scale_{1};
  std::uint32_t startup_shadow_map_resolution_{k_default_shadow_map_resolution};
  VulkanDevice device_;
  Swapchain swapchain_;
  vk::raii::PipelineCache pipeline_cache_{nullptr};
  vk::raii::CommandPool command_pool_{nullptr};
  vk::raii::CommandBuffers command_buffers_{nullptr};
  std::vector<vk::raii::Semaphore> image_available_semaphores_;
  std::vector<vk::raii::Semaphore> render_finished_semaphores_;
  std::vector<vk::raii::Fence> in_flight_fences_;
  MsaaColorImage msaa_color_image_;
  RenderColorImage render_color_image_;
  DepthImage depth_image_;
  ShadowMap shadow_map_;
  TextureTable texture_table_;
  TextureArray texture_array_;
  std::vector<MeshGpu> mesh_gpus_;
  UniformBufferSet uniform_buffers_;
  DescriptorResources descriptor_resources_;
  PipelineRegistry pipelines_;
  std::vector<DrawCommand> draw_list_;
  PassLayoutState pass_layouts_{};
  std::vector<BoneStorageBufferSet> bone_buffers_;
  std::vector<std::uint32_t> skinned_texture_indices_;
  std::uint32_t frame_index_{};
  bool framebuffer_resized_{};
  bool logged_shadow_filter_mode_{false};
  bool gpu_shutdown_complete_{false};
  FrameOverlayCallback frame_overlay_;
  std::uint64_t last_resize_ticks_{};
};

} // namespace engine
