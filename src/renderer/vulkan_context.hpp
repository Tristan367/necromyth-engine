#pragma once

#include "engine_config.hpp"
#include "renderer/bone_buffer.hpp"
#include "renderer/light_buffer.hpp"
#include "renderer/frame_overlay.hpp"
#include "renderer/render_host.hpp"
#include "renderer/deferred_delete.hpp"
#include "renderer/depth_image.hpp"
#include "renderer/descriptors.hpp"
#include "renderer/draw_list.hpp"
#include "renderer/instance_buffer.hpp"
#include "renderer/msaa_color_image.hpp"
#include "renderer/pass_recorder.hpp"
#include "renderer/pipeline_id.hpp"
#include "renderer/pipeline_registry.hpp"
#include "renderer/profiler.hpp"
#include "renderer/render_color_image.hpp"
#include "renderer/scene_gpu.hpp"
#include "renderer/shadow_map.hpp"
#include "renderer/render_settings.hpp"
#include "renderer/swapchain.hpp"
#include "renderer/texture_array.hpp"
#include "renderer/gpu_particle.hpp"
#include "renderer/particle_system.hpp"
#include "renderer/texture_table.hpp"
#include "renderer/uniform_buffer.hpp"
#include "renderer/upload_queue.hpp"
#include "renderer/mesh_gpu.hpp"
#include "renderer/vulkan_device.hpp"
#include "scene/animation_utils.hpp"
#include "scene/scene.hpp"
#include "scene/shadow_assignment.hpp"
#include "scene/shadow_utils.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_timer.h>

#include <cstdint>
#include <functional>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
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
        streaming_alpha_modes_(config.streaming_alpha_modes),
        msaa_config_(resolve_msaa_for_scene(
            config.msaa,
            scene_uses_alpha_to_coverage(scene.instances()) ||
                config.streaming_alpha_modes[static_cast<std::size_t>(
                    MeshAlphaMode::AlphaToCoverage)])),
        startup_point_shadow_filter_(scene.shadow_settings().point_shadow_filter),
        startup_shadow_filter_mode_(scene.shadow_settings().filter_mode),
        startup_cascade_mode_(scene.shadow_settings().cascade_mode),
        startup_render_scale_(config.render_scale),
        startup_shadow_scale_(config.shadow_scale),
        profile_to_stdout_(config.profile_to_stdout),
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
    light_buffer_.create(device_.physical_device(), device_.device(), config.max_total_lights);
    create_spot_atlas(startup_spot_atlas_size_ = scaled_shadow_map_resolution(1024, config.shadow_scale));
    create_point_cubemap(point_cube_face_size_ = scaled_shadow_map_resolution(1024, config.shadow_scale),
                         config.max_point_shadow_lights);
    create_point_light_shadow_ssbo(config.max_point_shadow_lights);
    particle_system_.create(device_.physical_device(), device_.device(), config.max_particles, 2);
    upload_queue_.create(device_.physical_device(), device_.device(),
                         config.staging_bytes_per_frame, detail::max_frames_in_flight);
    instance_buffer_.create(device_.physical_device(), device_.device(),
                            std::max(config.max_draw_instances, 1U), detail::max_frames_in_flight);
    max_skinned_instances_ = std::max(config.max_skinned_instances, 1U);
    bone_palette_.create(device_.physical_device(), device_.device(),
                         max_skinned_instances_, detail::max_frames_in_flight);
    create_command_pool();
    gpu_profiler_.create(device_.physical_device(), device_.device(),
                         detail::max_frames_in_flight, device_.queue_families().graphics);
    sync_mesh_slots(scene);
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
    if (!config.msaa.enabled && msaa_config_.enabled)
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
    retired_meshes_.clear();
    gpu_shutdown_complete_ = true;
  }

  [[nodiscard]] auto gpu_name() const -> const std::string & {
    return device_.gpu_name();
  }

  [[nodiscard]] auto device_ref() -> vk::raii::Device & { return device_.device(); }
  [[nodiscard]] auto color_fmt() const -> vk::Format { return swapchain_.image_format(); }
  [[nodiscard]] auto depth_fmt() const -> vk::Format { return depth_image_.format(); }
  [[nodiscard]] auto frame_layout_obj() const -> vk::DescriptorSetLayout { return descriptor_resources_.frame_layout(); }

  [[nodiscard]] auto &particle_system() { return particle_system_; }
  [[nodiscard]] auto sample_count() const -> vk::SampleCountFlagBits { return device_.msaa_samples(); }
  [[nodiscard]] auto frame_set_obj(std::uint32_t i) const -> vk::DescriptorSet { return descriptor_resources_.frame_set(i); }
  [[nodiscard]] auto phys_dev() -> vk::raii::PhysicalDevice { return device_.physical_device(); }
  [[nodiscard]] auto mem_props() -> vk::PhysicalDeviceMemoryProperties { return device_.physical_device().getMemoryProperties(); }

  [[nodiscard]] static auto pad(const char *label) -> std::string {
    std::string text(label);
    text.resize(std::max<std::size_t>(text.size(), 24), ' ');
    return text;
  }

  [[nodiscard]] static auto format_ms(float ms) -> std::string {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << std::setw(7) << ms << " ms";
    return out.str();
  }

  void mark_framebuffer_resized() {
    framebuffer_resized_ = true;
    last_resize_ticks_ = SDL_GetTicks();
  }

  void set_frame_overlay(FrameOverlayCallback callback) {
    frame_overlay_ = std::move(callback);
  }

  [[nodiscard]] auto gpu_profiler() const -> const GpuProfiler & { return gpu_profiler_; }
  [[nodiscard]] auto cpu_profiler() const -> const CpuProfiler & { return cpu_profiler_; }

  // Human-readable snapshot of the last profiling window. Set ENGINE_PROFILE=1
  // to have it printed periodically, or call this to drive a debug overlay.
  [[nodiscard]] auto profile_report() const -> std::string {
    std::ostringstream out;
    out << "\n-- frame profile (avg / max ms over " << k_profile_window_frames << " frames) --\n";

    if (gpu_profiler_.supported()) {
      float gpu_total = 0.0F;
      for (std::uint32_t i = 0; i < GpuProfiler::k_zone_count; ++i)
        gpu_total += gpu_profiler_.stats(static_cast<GpuZone>(i)).average_ms;
      out << "  GPU (total " << format_ms(gpu_total) << ")\n";
      for (std::uint32_t i = 0; i < GpuProfiler::k_zone_count; ++i) {
        const auto zone = static_cast<GpuZone>(i);
        out << "    " << pad(gpu_zone_name(zone)) << format_ms(gpu_profiler_.stats(zone).average_ms)
            << "  max " << format_ms(gpu_profiler_.stats(zone).max_ms) << '\n';
      }
    } else {
      out << "  GPU timestamps unsupported on this device/queue\n";
    }

    out << "  CPU\n";
    for (std::uint32_t i = 0; i < CpuProfiler::k_zone_count; ++i) {
      const auto zone = static_cast<CpuZone>(i);
      out << "    " << pad(cpu_zone_name(zone)) << format_ms(cpu_profiler_.stats(zone).average_ms)
          << "  max " << format_ms(cpu_profiler_.stats(zone).max_ms) << '\n';
    }

    out << "  draws " << render_stats_.draws_submitted << " submitted, "
        << render_stats_.draws_culled << " culled (main); "
        << render_stats_.shadow_draws_submitted << " submitted, "
        << render_stats_.shadow_draws_culled << " culled (shadow)\n";
    return out.str();
  }

  // Last completed frame's draw/cull counts.
  [[nodiscard]] auto render_stats() const -> const RenderStats & { return render_stats_; }

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
  // How many frames have hit the staging limit and deferred an upload. Not an
  // error -- back-pressure is the design -- but a steady stream of them means
  // geometry is arriving faster than it can be uploaded, and the ring wants to
  // be bigger.
  [[nodiscard]] auto upload_deferrals() const -> std::uint64_t { return upload_deferrals_; }

  // Mesh slots the Scene considers alive whose GPU copy does not match. Nonzero
  // for a frame or two is normal streaming; nonzero and STUCK means geometry
  // that will never appear -- an invisible chunk you can still walk on.
  [[nodiscard]] auto pending_mesh_uploads() const -> std::size_t { return pending_mesh_uploads_; }

  // Writes the next presented frame to an image file.
  //
  // Being able to look at a frame without a human in front of the window is the
  // difference between "something renders wrong" and knowing what. It reads the
  // swapchain image back after the frame is submitted, which is slow and
  // synchronises the device -- fine, because a screenshot is never in a hot
  // path and a correct picture matters more than the frame it costs.
  void request_screenshot(std::string path) { screenshot_path_ = std::move(path); }

  void sync_scene(const Scene &scene) {
    if (gpu_shutdown_complete_)
      return;

    const ScopedCpuZone cpu_zone(cpu_profiler_, CpuZone::SyncScene);
    upload_queue_.begin_frame(frame_counter_, frame_index_);

    // Mesh slots first. Mesh buffers are bound directly (bindVertexBuffers), not
    // through a descriptor set, so creating, remeshing and dropping geometry
    // needs no descriptor rebuild and no device stall -- superseded buffers are
    // retired and freed a few frames later instead. This is the path a
    // streaming world hits every frame.
    retired_meshes_.collect(frame_counter_, detail::max_frames_in_flight);
    sync_mesh_slots(scene);

    // Skinned instances no longer appear here: they share one bone buffer
    // indexed by dynamic offset, so spawning or despawning a character needs no
    // reallocation and no descriptor rewrite. Only textures still live in
    // descriptor sets that in-flight frames are reading, so only textures still
    // need an idle -- and new textures are rare.
    const bool textures_added = scene.texture_paths().size() > texture_table_.count();
    if (!textures_added)
      return;

    device_.wait_idle();

    bool descriptors_changed = false;

    if (textures_added) {
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

    const ScopedCpuZone frame_zone(cpu_profiler_, CpuZone::FrameTotal);

    {
      const ScopedCpuZone wait_zone(cpu_profiler_, CpuZone::WaitForFrame);
      if (device_.device().waitForFences(*in_flight_fences_[frame_index_], vk::True, UINT64_MAX) !=
          vk::Result::eSuccess)
        throw std::runtime_error("Failed to wait for frame fence");
    }

    // The fence above guarantees this slot's previous submission has completed,
    // so its timestamps are readable without ever blocking.
    gpu_profiler_.collect(frame_index_);

    cpu_profiler_.begin(CpuZone::AcquireImage);
    auto [acquire_result, image_index] = swapchain_.handle().acquireNextImage(
        UINT64_MAX,
        *image_available_semaphores_[frame_index_],
        nullptr);
    cpu_profiler_.end(CpuZone::AcquireImage);

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
    if (internal_extent.height == 0 || internal_extent.width == 0)
      return;
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

    // Compute spot shadow VP matrices for the UBO
    std::array<glm::mat4, k_max_spot_shadow_lights> spot_vps{};
    std::size_t spot_vp_count = 0;
    for (const SpotLight &sl : scene.spot_lights()) {
      if (sl.casts_shadow && spot_vp_count < k_max_spot_shadow_lights) {
        // Depth pass VS needs clip-space VP (NO bias remap) for SV_Position.
        spot_vps[spot_vp_count] = LightStorageBuffer::compute_shadow_view_proj(sl);
        ++spot_vp_count;
      }
    }

    // Point shadow: cubemap (Sascha omni model). First shadow-casting
    // Point light shadow data is updated per light during the shadow pass
    // — it varies per light (position, range, face VPs differ). The main pass
    // reads per-light position/range from the SSBO directly, so the UBO's
    // point-light fields are only used by the shadow vertex/fragment shader.

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
            .spot_light_vp = spot_vps,
            .point_light_params = glm::vec4(0.0F, 0.0F, point_cube_face_size_, 1.0F),
            .cascade_params = glm::vec4(
                cascades.split_view_z,
                shadow_settings.cascade_blend_range,
                0.0F,
                0.0F),
            .shadow_fade_width = glm::vec4(shadow_settings.coverage_fade_uv_width, 0.0F, 0.0F, 0.0F),
        });

    // Decide which lights get a shadow map before anything is written or drawn:
    // the light buffer, the point-shadow SSBO and the shadow pass all have to
    // agree on the slot each light was given.
    const Frustum camera_frustum = Frustum::from_view_proj(scene.camera().view_projection_matrix());
    const ShadowSlotAssignment shadow_slots = assign_shadow_slots(
        scene.point_lights(),
        scene.spot_lights(),
        camera_frustum,
        point_shadow_capacity_,
        k_max_spot_shadow_lights);

    light_buffer_.write(frame_index_, scene.point_lights(), scene.spot_lights(), shadow_slots,
                         static_cast<float>(startup_spot_atlas_size_));

    cpu_profiler_.begin(CpuZone::Animation);
    if (bone_palette_.valid()) {
      std::vector<glm::mat4> joint_matrices;
      std::vector<glm::mat4> bone_worlds;
      std::uint32_t bone_slot = 0;
      bool overflowed = false;
      for (MeshInstance &instance : scene.instances()) {
        // Must match every other bone-slot walker exactly — see
        // instance_uses_skinning(). In particular do NOT skip instances without
        // `pose_layers`: they still own a slot, and skipping them here would
        // shift every later instance onto the wrong slice of the bone buffer.
        // compute_joint_matrices_for_instance() emits bind pose for them.
        if (!instance_uses_skinning(instance, scene))
          continue;
        if (bone_slot >= max_skinned_instances_) {
          overflowed = true;
          break;
        }

        joint_matrices.clear();
        bone_worlds.clear();

        compute_joint_matrices_for_instance(
            scene.skeletons()[instance.skin_index],
            instance,
            scene.animations(),
            joint_matrices,
            &bone_worlds);

        // Bone attachments read this; swap rather than copy so a posed
        // character costs no allocation per frame.
        instance.cached_bone_worlds.swap(bone_worlds);

        bone_palette_.write(frame_index_, bone_slot, joint_matrices);
        ++bone_slot;
      }

      // Silently dropping characters would look like a rendering bug, so say so
      // once rather than every frame.
      if (overflowed && !logged_bone_overflow_) {
        std::cout << "Skinned instances exceed max_skinned_instances ("
                  << max_skinned_instances_
                  << "); extra characters render in bind pose. Raise "
                     "EngineConfig::max_skinned_instances.\n";
        logged_bone_overflow_ = true;
      }
    }
    cpu_profiler_.end(CpuZone::Animation);

    cpu_profiler_.begin(CpuZone::BuildDrawLists);
    render_stats_.reset();
    build_draw_list(scene, draw_list_);
    build_shadow_draw_list(draw_list_, shadow_draw_list_);
    write_instance_records(scene);
    cpu_profiler_.end(CpuZone::BuildDrawLists);

    cpu_profiler_.begin(CpuZone::RecordCommands);
    auto &command_buffer = command_buffers_[frame_index_];
    command_buffer.reset();
    upload_queue_.begin_frame(frame_counter_, frame_index_);
    command_buffer.begin({});
    gpu_profiler_.begin_frame(command_buffer, frame_index_);
    // Geometry uploaded since the last frame is copied here, before anything
    // draws it. Same command buffer, so ordering needs no semaphore -- just the
    // one barrier flush() records.
    upload_queue_.flush(command_buffer);
    pass_recorder().record_shadow_pass(command_buffer, frame_index_, pass_layouts_, shadow_draw_list_,
                                         cascades.light_view_proj);

    // A light whose sphere of influence misses the camera frustum lights nothing
    // visible, so it needs no shadow map -- assign_shadow_slots already dropped
    // it, and these counts follow from that.
    const bool has_spot_shadows = shadow_slots.spot_count > 0;
    const bool has_point_shadows = shadow_slots.point_count > 0;
    if (has_spot_shadows)
      pass_recorder().record_spot_shadow_pass(command_buffer, frame_index_, pass_layouts_, scene,
                                                shadow_draw_list_, *spot_atlas_, *spot_atlas_view_,
                                                startup_spot_atlas_size_);
    else
      ensure_shadow_sampler_readable(command_buffer, *spot_atlas_, pass_layouts_.spot_atlas_layout, 1);

    // Point shadow cubemap pass (Sascha omni model)
    if (has_point_shadows) {
      write_point_light_shadows(frame_index_, scene, shadow_slots);
      pass_recorder().record_point_shadow_pass(command_buffer, frame_index_, pass_layouts_, scene,
                                                shadow_draw_list_, *point_cube_,
                                                point_cube_face_views_,
                                                point_cube_face_size_,
                                                shadow_slots);
    } else {
      ensure_shadow_sampler_readable(command_buffer, *point_cube_, pass_layouts_.point_cube_layout,
                                     point_cube_layer_count_);
    }

    const FrameOverlayCallback *overlay_ptr = frame_overlay_ ? &frame_overlay_ : nullptr;

    std::function<void(vk::raii::CommandBuffer &)> post_geometry;
    if (particle_system_.active_count() > 0) {
      particle_system_.upload(frame_index_);
      post_geometry = [this, &scene](vk::raii::CommandBuffer &cmd) {
        pass_recorder().draw_particles(
            cmd, frame_index_, particle_system_.active_count(),
            pipelines_.pipeline(PipelineId::ParticleBillboard),
            pipelines_.pipeline_layout_for_particles(),
            scene.camera().view_projection_matrix(),
            glm::vec3(scene.camera().right()), glm::vec3(scene.camera().up()));
      };
    }

    pass_recorder().record_main_pass(
        command_buffer,
        frame_index_,
        image_index,
        pass_layouts_,
        draw_list_,
        camera_frustum,
        overlay_ptr,
        std::move(post_geometry));

    cpu_profiler_.end(CpuZone::RecordCommands);
    const ScopedCpuZone submit_zone(cpu_profiler_, CpuZone::Submit);

    const vk::SemaphoreSubmitInfo wait_semaphore_info{
        .semaphore = *image_available_semaphores_[frame_index_],
        .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
    };
    const vk::CommandBufferSubmitInfo command_buffer_info{
        .commandBuffer = *command_buffer,
    };
    const vk::SemaphoreSubmitInfo signal_semaphore_info{
        .semaphore = *render_finished_semaphores_[image_index],
        .stageMask = vk::PipelineStageFlagBits2::eAllCommands,
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
    gpu_profiler_.mark_frame_recorded(frame_index_);

    if (!screenshot_path_.empty())
      capture_swapchain_image(image_index);



    const vk::PresentInfoKHR present_info{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*render_finished_semaphores_[image_index],
        .swapchainCount = 1,
        .pSwapchains = &*swapchain_.handle(),
        .pImageIndices = &image_index,
    };

    cpu_profiler_.begin(CpuZone::Present);
    const vk::Result present_result = device_.present_queue().presentKHR(present_info);
    cpu_profiler_.end(CpuZone::Present);
    if (present_result == vk::Result::eErrorOutOfDateKHR)
      recreate_swapchain();
    else if (present_result != vk::Result::eSuccess && present_result != vk::Result::eSuboptimalKHR)
      throw std::runtime_error("Failed to present swapchain image");

    frame_index_ = (frame_index_ + 1) % detail::max_frames_in_flight;
    ++frame_counter_;
    advance_profile_window();
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
        .bone_palette = &bone_palette_,
        .msaa_enabled = device_.msaa_enabled(),
        .shadow_cascade_count = shadow_cascade_layer_count(startup_cascade_mode_),
        .render_extent = render_extent(),
        .render_color_image = render_scale_active(startup_render_scale_) ? &render_color_image_ : nullptr,
        .stats = &render_stats_,
        .visible_draws = &visible_draws_,
        .shadow_visible_draws = &shadow_visible_draws_,
        .gpu_profiler = &gpu_profiler_,
        .profile_frame_index = frame_index_,
    };
  }

  [[nodiscard]] auto render_extent() const -> vk::Extent2D {
    return scaled_render_extent(swapchain_.extent(), startup_render_scale_);
  }

  void submit_empty_frame() noexcept {
    try {
      const vk::SemaphoreSubmitInfo wait_info{
          .semaphore = *image_available_semaphores_[frame_index_],
          .stageMask = vk::PipelineStageFlagBits2::eAllCommands,
      };
      const vk::SubmitInfo2 submit_info{
          .waitSemaphoreInfoCount = 1,
          .pWaitSemaphoreInfos = &wait_info,
      };
      device_.graphics_queue().submit2(submit_info, *in_flight_fences_[frame_index_]);
    } catch (...) { /* device lost — no recovery */ }
  }

  // Writes one instance record per draw, for both lists.
  //
  // The two lists are sorted differently and get separate record ranges, so each
  // is internally contiguous and can be batched independently. Duplicating a
  // shadow caster's model matrix costs 80 bytes and buys batched shadow draws.
  void write_instance_records(const Scene &scene) {
    if (!instance_buffer_.begin_frame(frame_index_))
      return;

    const auto record = [&](DrawCommand &draw) {
      GpuInstance instance;
      instance.model = draw.model;
      instance.texture_layer = draw.texture_index;
      instance.sample_texture_array = draw.texture_source == TextureSource::ArrayLayer ? 1U : 0U;
      instance.bone_base = draw.bone_instance_index != k_invalid_skin_index
          ? bone_palette_.matrix_base(frame_index_, draw.bone_instance_index)
          : k_no_bone_base;
      draw.instance_index = instance_buffer_.push(instance);
    };

    for (DrawCommand &draw : draw_list_)
      record(draw);
    for (DrawCommand &draw : shadow_draw_list_)
      record(draw);

    if (instance_buffer_.overflowed() && !logged_instance_overflow_) {
      std::cout << "Draw instances exceed max_draw_instances ("
                << instance_buffer_.capacity()
                << "); some geometry will not render. Raise "
                   "EngineConfig::max_draw_instances.\n";
      logged_instance_overflow_ = true;
    }
    (void)scene;
  }

  void sync_mesh_slots(const Scene &scene) {
    if (upload_queue_.overflowed()) {
      ++upload_deferrals_;
      upload_queue_.clear_overflow();
      if (!logged_upload_overflow_) {
        std::cout << "Mesh staging ring is full; uploads are being deferred to later "
                     "frames. Raise EngineConfig::upload_bytes_per_frame if this is "
                     "constant.\n";
        logged_upload_overflow_ = true;
      }
    }
    engine::sync_scene_meshes(
        scene,
        device_.physical_device(),
        device_.device(),
        upload_queue_,
        mesh_gpus_,
        [this](MeshGpu retired) { retired_meshes_.retire(std::move(retired), frame_counter_); },
        &pending_mesh_uploads_);
  }

  // Republishes the averaged numbers every window. A window of 120 frames is
  // about two seconds at 60 Hz: long enough to be stable, short enough to react
  // while you are walking around looking for the slow spot.
  static constexpr std::uint32_t k_profile_window_frames = 120;

  void advance_profile_window() {
    if (++profile_window_frames_ < k_profile_window_frames)
      return;
    profile_window_frames_ = 0;
    gpu_profiler_.flush();
    cpu_profiler_.flush();
    if (profile_to_stdout_)
      std::cout << profile_report();
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
    depth_image_.recreate(render_extent());
    create_msaa_color_image();
    recreate_render_color_image();
  }

  void create_shadow_map(std::uint32_t resolution, std::uint32_t layer_count) {
    shadow_map_.create(device_.physical_device(), device_.device(), resolution, layer_count);
  }

  void create_spot_atlas(std::uint32_t size = 1024) {
    const vk::Format fmt = shadow_map_.format();
    const std::uint32_t sz = size;

    vk::ImageCreateInfo img_info{};
    img_info.imageType = vk::ImageType::e2D;
    img_info.format = fmt;
    img_info.extent = vk::Extent3D{sz, sz, 1};
    img_info.mipLevels = 1;
    img_info.arrayLayers = 1;
    img_info.samples = vk::SampleCountFlagBits::e1;
    img_info.tiling = vk::ImageTiling::eOptimal;
    img_info.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled;
    spot_atlas_ = vk::raii::Image(device_.device(), img_info);

    auto reqs = spot_atlas_.getMemoryRequirements();
    auto mt = detail::find_memory_type(device_.physical_device().getMemoryProperties(),
                                        reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::MemoryAllocateInfo mem_info{};
    mem_info.allocationSize = reqs.size;
    mem_info.memoryTypeIndex = mt;
    spot_atlas_mem_ = vk::raii::DeviceMemory(device_.device(), mem_info);
    spot_atlas_.bindMemory(*spot_atlas_mem_, 0);

    vk::ImageViewCreateInfo view_info{};
    view_info.image = *spot_atlas_;
    view_info.viewType = vk::ImageViewType::e2D;
    view_info.format = fmt;
    view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.layerCount = 1;
    spot_atlas_view_ = vk::raii::ImageView(device_.device(), view_info);

    vk::SamplerCreateInfo samp_info{};
    samp_info.magFilter = vk::Filter::eLinear;
    samp_info.minFilter = vk::Filter::eLinear;
    samp_info.compareEnable = VK_TRUE;
    samp_info.compareOp = vk::CompareOp::eLessOrEqual;
    spot_atlas_sampler_ = vk::raii::Sampler(device_.device(), samp_info);
  }

  void create_point_cubemap(std::uint32_t face_size = 1024, std::uint32_t max_point_lights = 1) {
    const std::uint32_t total_layers = max_point_lights * 6;
    point_cube_layer_count_ = total_layers;
    // Shadow slots address CUBES, not image array layers. These differ by a
    // factor of six, and confusing them hands out slots that index past the end
    // of the cube array.
    point_shadow_capacity_ = max_point_lights;

    // D32 depth cubemap array. Depth-only rendering (SV_Depth fragment output).
    // Sampled via comparison sampler in main pass for hardware PCF.
    vk::ImageCreateInfo img_info{};
    img_info.imageType = vk::ImageType::e2D;
    img_info.format = vk::Format::eD32Sfloat;
    img_info.extent = vk::Extent3D{face_size, face_size, 1};
    img_info.mipLevels = 1;
    img_info.arrayLayers = total_layers;
    img_info.samples = vk::SampleCountFlagBits::e1;
    img_info.tiling = vk::ImageTiling::eOptimal;
    img_info.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled;
    img_info.flags = vk::ImageCreateFlagBits::eCubeCompatible;
    point_cube_ = vk::raii::Image(device_.device(), img_info);

    auto reqs = point_cube_.getMemoryRequirements();
    auto mt = detail::find_memory_type(device_.physical_device().getMemoryProperties(),
                                        reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::MemoryAllocateInfo mem_info{};
    mem_info.allocationSize = reqs.size;
    mem_info.memoryTypeIndex = mt;
    point_cube_mem_ = vk::raii::DeviceMemory(device_.device(), mem_info);
    point_cube_.bindMemory(*point_cube_mem_, 0);

    // Per-light D32 2D_ARRAY depth views for multiview rendering
    point_cube_face_views_.clear();
    for (std::uint32_t light = 0; light < max_point_lights; ++light) {
      vk::ImageViewCreateInfo view_info{};
      view_info.image = *point_cube_;
      view_info.viewType = vk::ImageViewType::e2DArray;
      view_info.format = vk::Format::eD32Sfloat;
      view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
      view_info.subresourceRange.baseMipLevel = 0;
      view_info.subresourceRange.levelCount = 1;
      view_info.subresourceRange.baseArrayLayer = light * 6;
      view_info.subresourceRange.layerCount = 6;
      point_cube_face_views_.push_back(vk::raii::ImageView(device_.device(), view_info));
    }

    // CubeArray view for comparison sampling in the fragment shader
    vk::ImageViewCreateInfo cube_view_info{};
    cube_view_info.image = *point_cube_;
    cube_view_info.viewType = vk::ImageViewType::eCubeArray;
    cube_view_info.format = vk::Format::eD32Sfloat;
    cube_view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
    cube_view_info.subresourceRange.baseMipLevel = 0;
    cube_view_info.subresourceRange.levelCount = 1;
    cube_view_info.subresourceRange.baseArrayLayer = 0;
    cube_view_info.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
    point_cube_array_view_ = vk::raii::ImageView(device_.device(), cube_view_info);

    // Comparison sampler: hardware PCF (bilinear 2x2) with LessOrEqual depth test
    vk::SamplerCreateInfo samp_info{};
    samp_info.magFilter = vk::Filter::eLinear;
    samp_info.minFilter = vk::Filter::eLinear;
    samp_info.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samp_info.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samp_info.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    samp_info.compareEnable = vk::True;
    samp_info.compareOp = vk::CompareOp::eLessOrEqual;
    point_cube_sampler_ = vk::raii::Sampler(device_.device(), samp_info);
  }

  void create_point_light_shadow_ssbo(std::uint32_t max_lights) {
    pt_shadow_max_lights_ = max_lights;
    if (max_lights == 0) return;
    const vk::DeviceSize buf_size = max_lights * sizeof(GpuPointLightShadowData);

    for (std::uint32_t i = 0; i < k_pt_shadow_frames; ++i) {
      vk::BufferCreateInfo buf_info{};
      buf_info.size = buf_size;
      buf_info.usage = vk::BufferUsageFlagBits::eStorageBuffer;
      buf_info.sharingMode = vk::SharingMode::eExclusive;
      pt_shadow_buffers_[i] = vk::raii::Buffer(device_.device(), buf_info);

      const auto reqs = pt_shadow_buffers_[i]->getMemoryRequirements();
      const auto mt = detail::find_memory_type(
          device_.physical_device().getMemoryProperties(),
          reqs.memoryTypeBits,
          vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
      vk::MemoryAllocateInfo alloc{};
      alloc.allocationSize = reqs.size;
      alloc.memoryTypeIndex = mt;
      pt_shadow_memory_[i] = vk::raii::DeviceMemory(device_.device(), alloc);
      pt_shadow_buffers_[i]->bindMemory(**pt_shadow_memory_[i], 0);

      pt_shadow_mapped_[i] = static_cast<GpuPointLightShadowData *>(
          pt_shadow_memory_[i]->mapMemory(0, buf_size));
    }
  }

  // Indexed by shadow slot, not by scene light index. Slots are dense and
  // bounded by max_point_shadow_lights, so a scene with more point lights than
  // shadow slots can no longer read past the end of this buffer -- which it did
  // when the shader indexed it by the light's position in the scene array.
  void write_point_light_shadows(std::uint32_t frame_index, const Scene &scene,
                                 const ShadowSlotAssignment &shadows) {
    if (frame_index >= k_pt_shadow_frames || !pt_shadow_mapped_[frame_index])
      return;

    const auto &lights = scene.point_lights();
    const auto &face_views = cubemap_face_views();

    auto *ptr = pt_shadow_mapped_[frame_index];
    for (std::size_t i = 0; i < lights.size() && i < shadows.point_slots.size(); ++i) {
      const std::int32_t slot = shadows.point_slots[i];
      if (slot == k_no_shadow_slot || static_cast<std::size_t>(slot) >= pt_shadow_max_lights_)
        continue;
      const auto &pl = lights[i];
      const glm::mat4 proj = glm::perspective(glm::radians(90.0F), 1.0F, 0.01F, pl.range);
      ptr[slot].view = glm::translate(glm::mat4(1.0F), -glm::vec3(pl.position));
      for (int f = 0; f < 6; ++f)
        ptr[slot].face_vp[f] = proj * face_views[f];
      ptr[slot].pos = glm::vec4(pl.position, 0.0F);
      ptr[slot].range = pl.range;
    }
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
    skinned_pipelines_built_ = has_skinned_instances(scene.instances(), scene);
    rebuild_descriptor_sets(count_skinned_instances(scene.instances(), scene));
  }

  void rebuild_descriptor_sets(bool skinned) {
    descriptor_resources_.create_pool(
        device_.device(),
        detail::max_frames_in_flight,
        texture_table_.count(),
        skinned);

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
    descriptor_resources_.update_light_buffers(device_.device(), {
        light_buffer_.buffer_ptr(0), light_buffer_.buffer_ptr(1)});
    descriptor_resources_.update_spot_shadow_sampler(device_.device(), *spot_atlas_sampler_,
                                                       *spot_atlas_view_);
    descriptor_resources_.update_point_cube_sampler(device_.device(), *point_cube_sampler_,
                                                      *point_cube_array_view_);

    std::array<vk::Buffer, k_pt_shadow_frames> buf_ptrs{};
    for (std::uint32_t i = 0; i < k_pt_shadow_frames; ++i)
      buf_ptrs[i] = pt_shadow_buffers_[i] ? **pt_shadow_buffers_[i] : vk::Buffer{};
    descriptor_resources_.update_point_light_shadow_ssbo(device_.device(), buf_ptrs);

    std::array<vk::Buffer, detail::max_frames_in_flight> instance_bufs{};
    for (std::uint32_t i = 0; i < detail::max_frames_in_flight; ++i)
      instance_bufs[i] = instance_buffer_.buffer(i);
    descriptor_resources_.update_instance_buffers(device_.device(), instance_bufs);

    std::array<vk::Buffer, 2> particle_bufs{};
    for (std::uint32_t i = 0; i < 2; ++i)
      particle_bufs[i] = particle_system_.buffer(i);
    descriptor_resources_.update_particle_ssbo(device_.device(), particle_bufs);

    if (skinned)
      descriptor_resources_.allocate_skinned_sets(
          device_.device(),
          bone_palette_.buffer(),
          bone_palette_.descriptor_range(),
          texture_table_.texture_pointers());
  }

  void recreate_descriptor_sets() {
    rebuild_descriptor_sets(skinned_pipelines_built_);
  }

  void create_pipelines(const Scene &scene) {
    AlphaModeSet alpha_modes = collect_used_alpha_modes(scene.instances());
    for (std::size_t i = 0; i < alpha_modes.size(); ++i)
      alpha_modes[i] = alpha_modes[i] || streaming_alpha_modes_[i];

    const PipelineBuildProfile profile{
        .shadow_filter = startup_shadow_filter_mode_,
        .cascade_mode = startup_cascade_mode_,
        .textured_alpha_modes = alpha_modes,
        .build_skinned = has_skinned_instances(scene.instances(), scene),
        .has_point_shadows = has_point_shadow_lights(scene.point_lights()),
    };

    std::size_t textured_pipeline_count = 0;
    for (const bool used : profile.textured_alpha_modes) {
      if (used)
        ++textured_pipeline_count;
    }
    const std::size_t skinned_count = profile.build_skinned ? textured_pipeline_count + 1 : 0;
    const std::size_t point_count = profile.has_point_shadows ? 2 : 0; // PointShadowDepth + skinned
    std::cout << "Graphics pipelines: " << (2 + textured_pipeline_count + skinned_count + point_count)
              << " (background + shadow depth + " << textured_pipeline_count << " textured"
              << (profile.build_skinned ? " + " + std::to_string(skinned_count) + " skinned" : "")
              << (profile.has_point_shadows ? " + " + std::to_string(point_count) + " point shadow" : "")
              << ")\n";

    const auto textured_spirv = profile.has_point_shadows
        ? std::string_view(ENGINE_SHADER_SPIRV)
        : std::string_view(ENGINE_NO_POINT_SHADER_SPIRV);

    pipelines_.create(
        device_.device(),
        swapchain_.image_format(),
        depth_image_.format(),
        shadow_map_.format(),
        textured_spirv,
        ENGINE_SKY_SHADER_SPIRV,
        ENGINE_SHADOW_DEPTH_SPIRV,
        ENGINE_SKINNED_SHADER_SPIRV,
        ENGINE_SKINNED_SHADOW_DEPTH_SPIRV,
        ENGINE_POINT_SHADOW_SPIRV,
        ENGINE_PARTICLE_BILLBOARD_SPIRV,
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

  // When a shadow pass is skipped (no lights of that type), the depth image is
  // never transitioned out of UNDEFINED, yet the main-pass fragment shader still
  // samples its descriptor (written as DEPTH_STENCIL_READ_ONLY_OPTIMAL). Transition
  // it once so the sampled layout matches the descriptor. Tracked so it fires only
  // on the first frame (or after a layout reset).
  void ensure_shadow_sampler_readable(vk::raii::CommandBuffer &command_buffer, vk::Image image,
                                      vk::ImageLayout &tracked_layout,
                                      std::uint32_t layer_count) const {
    if (tracked_layout == vk::ImageLayout::eDepthStencilReadOnlyOptimal)
      return;

    transition_image_layout(
        command_buffer, image,
        tracked_layout, vk::ImageLayout::eDepthStencilReadOnlyOptimal,
        tracked_layout == vk::ImageLayout::eUndefined ? vk::AccessFlagBits2{}
                                                      : vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits2::eShaderRead,
        tracked_layout == vk::ImageLayout::eUndefined
            ? vk::PipelineStageFlagBits2::eTopOfPipe
            : vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eDepth, 0, 1, 0, layer_count);
    tracked_layout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
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

      SDL_Event e;
      while (SDL_PollEvent(&e)) {
        if (e.type == SDL_EVENT_QUIT)
          throw std::runtime_error("Quit requested while waiting for window extent");
      }
      SDL_Delay(16);
    }
  }

  void recreate_swapchain() {
    wait_for_nonzero_extent();
    device_.wait_idle();
    frame_index_ = 0;
    swapchain_.recreate();
    recreate_render_targets();
    pipelines_.recreate(device_.device(), swapchain_.image_format(), depth_image_.format());
    create_sync_objects();
    reset_image_layout_tracking();
  }

  [[nodiscard]] static auto count_skinned_instances(const std::vector<MeshInstance> &instances,
                                                     const Scene &scene) -> std::uint32_t {
    std::uint32_t count = 0;
    for (const MeshInstance &instance : instances) {
      // Must match every other bone-slot walker exactly — see instance_uses_skinning().
      if (instance_uses_skinning(instance, scene))
        ++count;
    }
    return count;
  }

  [[nodiscard]] static auto has_skinned_instances(const std::vector<MeshInstance> &instances,
                                                     const Scene &scene) -> bool {
    return count_skinned_instances(instances, scene) > 0;
  }

  [[nodiscard]] static auto has_point_shadow_lights(const std::vector<PointLight> &point_lights) -> bool {
    for (const PointLight &pl : point_lights)
      if (pl.casts_shadow) return true;
    return false;
  }

  SDL_Window *window_{};
  mutable RenderStats render_stats_{};
  // CpuProfiler holds no Vulkan objects, so it is free of the ordering
  // constraint that keeps GpuProfiler below device_.
  mutable CpuProfiler cpu_profiler_;
  std::uint32_t profile_window_frames_{0};
  // Declared before msaa_config_ on purpose: the MSAA decision reads it.
  AlphaModeSet streaming_alpha_modes_{};
  MsaaSettings msaa_config_{};
  bool startup_point_shadow_filter_{false};
  ShadowFilterMode startup_shadow_filter_mode_{ShadowFilterMode::Pcf3x3};
  ShadowCascadeMode startup_cascade_mode_{ShadowCascadeMode::Dual};
  std::uint32_t startup_render_scale_{1};
  std::uint32_t startup_shadow_scale_{1};
  bool profile_to_stdout_{false};
  std::uint32_t startup_shadow_map_resolution_{k_default_shadow_map_resolution};
  std::uint32_t startup_spot_atlas_size_{1024};
  VulkanDevice device_;
  Swapchain swapchain_;
  // MUST stay below device_: members are destroyed in reverse declaration
  // order, so anything owning a Vulkan handle has to be declared after the
  // device that created it or it will outlive the device and leak.
  mutable GpuProfiler gpu_profiler_;
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
  LightStorageBuffer light_buffer_;
  vk::raii::DeviceMemory spot_atlas_mem_{nullptr};
  vk::raii::Image spot_atlas_{nullptr};
  vk::raii::ImageView spot_atlas_view_{nullptr};
  vk::raii::Sampler spot_atlas_sampler_{nullptr};
  vk::raii::DeviceMemory point_cube_mem_{nullptr};
  vk::raii::Image point_cube_{nullptr};
  std::vector<vk::raii::ImageView> point_cube_face_views_;
  vk::raii::ImageView point_cube_array_view_{nullptr};
  vk::raii::Sampler point_cube_sampler_{nullptr};
  float point_cube_face_size_{1024.0F};
  struct GpuPointLightShadowData {
    glm::mat4 view{1.0F};
    std::array<glm::mat4, 6> face_vp{};
    glm::vec4 pos{};
    float range{0.0F};
    float _pad0{0.0F};
    float _pad1{0.0F};
    float _pad2{0.0F};
  };
  static_assert(sizeof(GpuPointLightShadowData) == 480,
                "GpuPointLightShadowData must be 480 bytes to match Slang SSBO layout");
  static constexpr std::uint32_t k_pt_shadow_frames = 2;
  std::array<std::optional<vk::raii::Buffer>, k_pt_shadow_frames> pt_shadow_buffers_{};
  std::array<std::optional<vk::raii::DeviceMemory>, k_pt_shadow_frames> pt_shadow_memory_{};
  std::array<GpuPointLightShadowData *, k_pt_shadow_frames> pt_shadow_mapped_{};
  std::uint32_t pt_shadow_max_lights_{0};
  std::uint32_t point_cube_layer_count_{0};   // image array layers = cubes * 6
  std::uint32_t point_shadow_capacity_{0};    // cubes, i.e. shadow-casting lights
  ParticleSystem particle_system_;
  TextureTable texture_table_;
  TextureArray texture_array_;
  BonePalette bone_palette_;
  InstanceBuffer instance_buffer_;
  mutable std::vector<DrawCommand> visible_draws_;
  mutable std::vector<DrawCommand> shadow_visible_draws_;
  bool logged_instance_overflow_{false};
  std::uint32_t max_skinned_instances_{0};
  bool skinned_pipelines_built_{false};
  std::vector<MeshGpuSlot> mesh_gpus_;
  DeferredDelete<MeshGpu> retired_meshes_;
  void capture_swapchain_image(std::uint32_t image_index) {
    const std::string path = screenshot_path_;
    screenshot_path_.clear();

    device_.device().waitIdle();

    const vk::Extent2D extent = swapchain_.extent();
    const vk::DeviceSize bytes =
        static_cast<vk::DeviceSize>(extent.width) * extent.height * 4;

    vk::raii::Buffer staging(device_.device(), vk::BufferCreateInfo{
        .size = bytes,
        .usage = vk::BufferUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive,
    });
    const vk::MemoryRequirements requirements = staging.getMemoryRequirements();
    vk::raii::DeviceMemory memory(device_.device(), vk::MemoryAllocateInfo{
        .allocationSize = requirements.size,
        .memoryTypeIndex = detail::find_memory_type(
            device_.physical_device().getMemoryProperties(), requirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
    });
    staging.bindMemory(*memory, 0);

    vk::raii::CommandBuffers buffers(device_.device(), vk::CommandBufferAllocateInfo{
        .commandPool = *command_pool_,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    });
    vk::raii::CommandBuffer &copy = buffers.front();
    copy.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    const vk::Image image = swapchain_.images()[image_index];
    const auto barrier = [&](vk::ImageLayout from, vk::ImageLayout to,
                             vk::AccessFlags2 src, vk::AccessFlags2 dst) {
      const vk::ImageMemoryBarrier2 b{
          .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
          .srcAccessMask = src,
          .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
          .dstAccessMask = dst,
          .oldLayout = from,
          .newLayout = to,
          .image = image,
          .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
      };
      copy.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                               .pImageMemoryBarriers = &b});
    };

    barrier(vk::ImageLayout::ePresentSrcKHR, vk::ImageLayout::eTransferSrcOptimal,
            vk::AccessFlagBits2::eMemoryRead, vk::AccessFlagBits2::eTransferRead);
    copy.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, *staging,
                           vk::BufferImageCopy{
                               .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                               .imageExtent = {extent.width, extent.height, 1},
                           });
    barrier(vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eTransferRead, vk::AccessFlagBits2::eMemoryRead);
    copy.end();

    device_.graphics_queue().submit(vk::SubmitInfo{
        .commandBufferCount = 1, .pCommandBuffers = &*copy});
    device_.graphics_queue().waitIdle();

    const auto *pixels = static_cast<const unsigned char *>(memory.mapMemory(0, bytes));
    const bool bgr = swapchain_.image_format() == vk::Format::eB8G8R8A8Srgb ||
                     swapchain_.image_format() == vk::Format::eB8G8R8A8Unorm;

    // Plain PPM: no encoder to depend on, and every viewer and script reads it.
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << extent.width << " " << extent.height << "\n255\n";
    for (std::uint32_t i = 0; i < extent.width * extent.height; ++i) {
      const unsigned char *p = pixels + static_cast<std::size_t>(i) * 4;
      const char rgb[3] = {static_cast<char>(bgr ? p[2] : p[0]), static_cast<char>(p[1]),
                           static_cast<char>(bgr ? p[0] : p[2])};
      out.write(rgb, 3);
    }
    memory.unmapMemory();
    std::cout << "screenshot written to " << path << " (" << extent.width << "x"
              << extent.height << ")\n";
  }

  std::string screenshot_path_;
  UploadQueue upload_queue_;
  std::uint64_t upload_deferrals_{0};
  std::size_t pending_mesh_uploads_{0};
  bool logged_upload_overflow_{false};
  // Monotonic frame number, for deciding when retired GPU resources are safe
  // to free. Distinct from frame_index_, which cycles 0..max_frames_in_flight-1.
  std::uint64_t frame_counter_{0};
  UniformBufferSet uniform_buffers_;
  DescriptorResources descriptor_resources_;
  PipelineRegistry pipelines_;
  std::vector<DrawCommand> draw_list_;
  std::vector<DrawCommand> shadow_draw_list_;
  PassLayoutState pass_layouts_{};
  bool logged_bone_overflow_{false};
  std::uint32_t frame_index_{};
  bool framebuffer_resized_{};
  bool logged_shadow_filter_mode_{false};
  bool gpu_shutdown_complete_{false};
  FrameOverlayCallback frame_overlay_;
  std::uint64_t last_resize_ticks_{};
};

} // namespace engine
