#pragma once

#include "renderer/device_memory.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

namespace engine {

// Batches buffer uploads into the frame's own command buffer.
//
// The previous path created a staging buffer, allocated a one-shot command
// buffer, submitted it and then WAITED ON A FENCE -- per buffer, and a mesh is
// two buffers. That is fine for a scene built once at startup and ruinous for a
// streaming world: with chunks arriving continuously it measured 9.8 ms average
// and 23.9 ms peak on the main thread, which is a dropped frame every time the
// horizon fills in. The GPU was idle for most of it; the cost was round-trip
// latency, not bandwidth.
//
// Staging the copies and replaying them into the frame's command buffer removes
// every fence wait. Ordering is free: the copies are recorded before the passes
// that read the buffers, in the same command buffer, so one barrier covers them
// all.
//
// Staging memory is one persistent, permanently mapped ring rather than a buffer
// allocated per upload. vkAllocateMemory is a heavyweight driver call, and at
// two per mesh a streaming world makes tens of them a frame -- that alone still
// cost 2.6 ms after the fence waits were gone. The ring is split into one
// segment per frame in flight, so a segment is only reused once the frame that
// read it has completed.
class UploadQueue {
public:
  void create(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      vk::DeviceSize bytes_per_frame,
      std::uint32_t frame_count) {
    segment_size_ = std::max<vk::DeviceSize>(bytes_per_frame, 1024);
    segment_count_ = std::max(frame_count, 1U);
    const vk::DeviceSize total = segment_size_ * segment_count_;

    ring_ = vk::raii::Buffer(device, vk::BufferCreateInfo{
        .size = total,
        .usage = vk::BufferUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive,
    });
    const vk::MemoryRequirements requirements = ring_.getMemoryRequirements();
    memory_ = vk::raii::DeviceMemory(device, vk::MemoryAllocateInfo{
        .allocationSize = requirements.size,
        .memoryTypeIndex = detail::find_memory_type(
            physical_device.getMemoryProperties(),
            requirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
    });
    ring_.bindMemory(*memory_, 0);
    mapped_ = static_cast<std::byte *>(memory_.mapMemory(0, total));
  }

  // Selects this frame's segment. Idempotent within a frame, so it is safe to
  // call from both sync_scene and draw_frame.
  void begin_frame(std::uint64_t frame_counter, std::uint32_t frame_index) {
    if (frame_counter == current_frame_ && started_)
      return;
    current_frame_ = frame_counter;
    started_ = true;
    const auto base = static_cast<vk::DeviceSize>(frame_index % segment_count_) * segment_size_;

    // Copies left over from a frame that never rendered are NOT dropped.
    //
    // draw_frame returns early on a swapchain resize, a minimised window and an
    // out-of-date acquire -- all of them after sync_scene has already staged
    // this frame's geometry. The mesh slots have been marked uploaded by then,
    // so dropping the copies would leave those buffers holding whatever the
    // pool last had in them and nothing would ever ask for them again. They
    // have to survive to the next frame that actually records.
    //
    // Their source bytes live at absolute offsets in the ring, so a different
    // segment leaves them alone. Landing back on the SAME segment is the one
    // case that would overwrite them, and there the offset has to carry over
    // instead of restarting -- which costs the frame a little staging room and
    // nothing else, because the segment is sized for a frame's worth of
    // uploads and a frame that never rendered did not spend its share.
    if (copies_.empty() || base != segment_base_)
      offset_ = 0;
    segment_base_ = base;
  }

  // Copies `data` into this frame's staging segment and records a copy into
  // `destination`. Returns false when the segment is full -- the caller should
  // stop uploading and try again next frame rather than growing without bound.
  auto stage(vk::Buffer destination, vk::DeviceSize size, const void *data) -> bool {
    if (size == 0)
      return true;
    if (mapped_ == nullptr)
      return false;

    // Copy offsets want to be 4-byte aligned; over-aligning costs nothing here.
    const vk::DeviceSize aligned = (offset_ + 15) & ~vk::DeviceSize{15};
    // The ring is a physical limit and nothing else. How much one frame SHOULD
    // upload is a question about time, not bytes, and it is answered by the
    // clock in sync_scene_meshes -- see the note there on why a byte cap was
    // the wrong instrument.
    if (aligned + size > segment_size_) {
      overflowed_ = true;
      return false;
    }
    offset_ = aligned;

    std::memcpy(mapped_ + segment_base_ + offset_, data, static_cast<std::size_t>(size));
    copies_.push_back(Copy{.source_offset = segment_base_ + offset_,
                           .destination = destination,
                           .size = size});
    offset_ += size;
    return true;
  }

  // Forgets every queued copy into `destination`. Called by the buffer that
  // owns the handle, on its way out.
  //
  // A staged copy holds a raw VkBuffer and flush() replays it later -- so
  // between the two, that buffer has to still exist. It did not always. A mesh
  // slot re-uploaded before the flush destroys and recreates its buffer inside
  // upload_deferred, and a frame that never rendered carries its copies into
  // the next one, where sync_scene rebuilds the geometry underneath them.
  // Either way vkCmdCopyBuffer is handed a handle that no longer names
  // anything, which is what the validation layer was reporting.
  //
  // Cancelling at the owner rather than validating at the flush is the only
  // version that can be right: once the buffer is gone there is nothing left
  // to ask.
  void cancel(vk::Buffer destination) {
    if (copies_.empty())
      return;
    std::erase_if(copies_, [destination](const Copy &copy) {
      return copy.destination == destination;
    });
  }

  [[nodiscard]] auto overflowed() const -> bool { return overflowed_; }
  void clear_overflow() { overflowed_ = false; }
  [[nodiscard]] auto bytes_used() const -> vk::DeviceSize { return offset_; }

  [[nodiscard]] auto empty() const -> bool { return copies_.empty(); }
  [[nodiscard]] auto pending_count() const -> std::size_t { return copies_.size(); }

  // Records every queued copy, then one barrier making the results visible to
  // vertex and index fetch. Must be called before any pass that draws them.
  void flush(vk::raii::CommandBuffer &command_buffer) {
    if (copies_.empty())
      return;

    for (const Copy &copy : copies_)
      command_buffer.copyBuffer(*ring_, copy.destination,
                                vk::BufferCopy{.srcOffset = copy.source_offset, .size = copy.size});

    // One barrier for all of them: a per-copy barrier would serialise transfers
    // that the GPU is free to overlap.
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eVertexAttributeInput |
                        vk::PipelineStageFlagBits2::eIndexInput,
        .dstAccessMask = vk::AccessFlagBits2::eVertexAttributeRead |
                         vk::AccessFlagBits2::eIndexRead,
    };
    command_buffer.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });

    copies_.clear();
  }

private:
  struct Copy {
    vk::DeviceSize source_offset;
    vk::Buffer destination;
    vk::DeviceSize size;
  };

  vk::raii::Buffer ring_{nullptr};
  vk::raii::DeviceMemory memory_{nullptr};
  std::byte *mapped_{nullptr};
  vk::DeviceSize segment_size_{0};
  vk::DeviceSize segment_base_{0};
  vk::DeviceSize offset_{0};
  std::uint32_t segment_count_{1};
  std::uint64_t current_frame_{0};
  bool started_{false};
  bool overflowed_{false};
  std::vector<Copy> copies_;
};

} // namespace engine
