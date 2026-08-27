// Does the upload queue ever hand vkCmdCopyBuffer a handle that is already
// dead?
//
// It did. A staged copy holds a raw VkBuffer and the copy is replayed into the
// frame's command buffer later, so between the two the destination has to stay
// alive -- and there are two ways it did not:
//
//   re-upload   upload_deferred destroys and recreates the buffer before it
//               stages, so a slot uploaded twice before a flush leaves the
//               first copy pointing at a buffer that no longer exists.
//   dropped frame
//               draw_frame returns early on a swapchain resize, a minimised
//               window and an out-of-date acquire, all of them AFTER sync_scene
//               staged this frame's geometry. The copies sit in the queue while
//               the next sync_scene rebuilds the meshes underneath them.
//
// Neither one crashes and neither one is visible: the copy either writes into
// freed memory or is skipped, and what you get is a chunk of geometry that is
// subtly wrong or missing, which looks exactly like a mesher bug. The only
// thing that ever said otherwise was one line from the validation layer.
//
// So the test drives both shapes directly and lets the layer be the oracle.
// Needs a real device.

#include "gpu_test_support.hpp"

#include "engine_config.hpp"
#include "platform/engine_runtime.hpp"
#include "renderer/buffer.hpp"
#include "renderer/device_allocator.hpp"
#include "renderer/upload_queue.hpp"
#include "scene/scene.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char *what) {
  if (!condition) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

} // namespace

auto main() -> int {
  const char *texture = std::getenv("VCE_STRESS_TEXTURE");
  if (texture == nullptr) {
    std::printf("Set VCE_STRESS_TEXTURE to any .png before running.\n");
    return EXIT_FAILURE;
  }

  engine::Scene scene;
  (void)scene.add_texture(texture);
  (void)scene.add_texture_array_layer(texture);

  engine::test::request_validation();
  engine::EngineConfig config = engine::engine_config_from_environment();
  config.window_title = "upload queue test";
  engine::EngineRuntime runtime(config, scene);
  const engine::test::ValidationGuard validation(runtime.vulkan());

  auto &device = runtime.vulkan().device_ref();

  engine::DeviceAllocator allocator;
  allocator.create(runtime.vulkan().phys_dev(), device, 4ULL * 1024 * 1024);

  engine::UploadQueue uploads;
  // Three segments so the frame index can move off a segment and back onto it,
  // which is the case begin_frame has to get right.
  uploads.create(runtime.vulkan().phys_dev(), device, 1ULL * 1024 * 1024, 3);

  // A command buffer of our own. The layer reports a dead VkBuffer when the
  // copy is RECORDED, so this never has to be submitted -- which keeps the test
  // free of fences and of any dependency on the frame loop.
  const vk::raii::CommandPool pool(device, vk::CommandPoolCreateInfo{
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = runtime.vulkan().gpu().queue_families().graphics,
  });
  vk::raii::CommandBuffers command_buffers(device, vk::CommandBufferAllocateInfo{
      .commandPool = *pool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1,
  });
  vk::raii::CommandBuffer &command_buffer = command_buffers.front();

  const std::vector<std::uint32_t> payload(256, 0xABCDEF01U);
  const auto payload_bytes = static_cast<vk::DeviceSize>(payload.size() * sizeof(std::uint32_t));

  const auto stage_into = [&](engine::DeviceLocalBuffer &buffer) {
    return buffer.upload_deferred(allocator, device, uploads, payload_bytes,
                                  vk::BufferUsageFlagBits::eVertexBuffer, payload.data());
  };

  const auto record = [&] {
    command_buffer.reset();
    command_buffer.begin({});
    uploads.flush(command_buffer);
    command_buffer.end();
  };

  // --- 1. Destroyed before the flush. The queue must forget it. ---
  uploads.begin_frame(1, 0);
  {
    engine::DeviceLocalBuffer doomed;
    check(stage_into(doomed), "staged into a buffer that is about to die");
    check(uploads.pending_count() == 1, "one copy queued");
  }
  check(uploads.pending_count() == 0, "destroying the buffer withdrew its copy");
  record();

  // --- 2. Re-uploaded before the flush. Only the live handle survives. ---
  uploads.begin_frame(2, 1);
  {
    engine::DeviceLocalBuffer remeshed;
    check(stage_into(remeshed), "first upload staged");
    check(stage_into(remeshed), "second upload staged");
    // upload_deferred replaces the VkBuffer, so the first copy's destination is
    // gone. Exactly one copy may be left, and it must be to the new handle.
    check(uploads.pending_count() == 1, "the superseded copy was withdrawn");
    record();
  }

  // --- 3. A frame that stages and never records, then a frame that does. ---
  //
  // The staged data has to still be there. Landing back on the same segment is
  // the case that used to overwrite it, so the frame indices below go 0, 1, 0.
  {
    engine::DeviceLocalBuffer survivor_a;
    engine::DeviceLocalBuffer survivor_b;

    uploads.begin_frame(10, 0);
    check(stage_into(survivor_a), "staged in the frame that gets dropped");
    // No record() here: this is the swapchain-resize early-out.

    uploads.begin_frame(11, 1);
    check(uploads.pending_count() == 1, "the dropped frame's copy is still queued");
    check(stage_into(survivor_b), "staged in the frame that follows");
    check(uploads.pending_count() == 2, "both copies queued");
    record();
    check(uploads.pending_count() == 0, "flush cleared the queue");
  }

  // --- 4. Same, but the next frame lands back on the same segment. ---
  {
    engine::DeviceLocalBuffer survivor_a;
    engine::DeviceLocalBuffer survivor_b;

    uploads.begin_frame(20, 2);
    check(stage_into(survivor_a), "staged in the frame that gets dropped");
    const vk::DeviceSize used = uploads.bytes_used();

    uploads.begin_frame(21, 2); // the frame index did not advance
    check(uploads.bytes_used() >= used,
          "staging carried on past the dropped frame's bytes instead of over them");
    check(stage_into(survivor_b), "staged in the frame that follows");
    record();
  }

  const bool clean = validation.check("upload queue");
  if (failures != 0 || !clean) {
    std::printf("test_upload_queue: FAILED (%d checks)\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("test_upload_queue: OK\n");
  return EXIT_SUCCESS;
}
