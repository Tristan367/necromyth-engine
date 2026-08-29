#pragma once

#include "renderer/device_allocator.hpp"
#include "renderer/device_memory.hpp"
#include "renderer/upload_queue.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace engine {

namespace detail {

[[nodiscard]] inline auto create_mipmapped_sampler(
    const vk::raii::Device &device,
    const vk::raii::PhysicalDevice &physical_device,
    std::uint32_t mip_levels) -> vk::raii::Sampler {
  const vk::PhysicalDeviceProperties properties = physical_device.getProperties();
  const vk::PhysicalDeviceFeatures features = physical_device.getFeatures();
  const bool anisotropy = features.samplerAnisotropy == vk::True;
  return vk::raii::Sampler(
      device,
      vk::SamplerCreateInfo{
          .magFilter = vk::Filter::eLinear,
          .minFilter = vk::Filter::eLinear,
          .mipmapMode = vk::SamplerMipmapMode::eLinear,
          .addressModeU = vk::SamplerAddressMode::eRepeat,
          .addressModeV = vk::SamplerAddressMode::eRepeat,
          .addressModeW = vk::SamplerAddressMode::eRepeat,
          .anisotropyEnable = anisotropy ? vk::True : vk::False,
          .maxAnisotropy = anisotropy ? properties.limits.maxSamplerAnisotropy : 1.0F,
          .maxLod = static_cast<float>(mip_levels),
      });
}

inline void copy_buffer(
    vk::raii::Device &device,
    vk::raii::CommandPool &command_pool,
    vk::raii::Queue &queue,
    vk::Buffer src,
    vk::Buffer dst,
    vk::DeviceSize size) {
  const vk::CommandBufferAllocateInfo allocate_info{
      .commandPool = *command_pool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1,
  };

  vk::raii::CommandBuffer command_buffer = std::move(
      vk::raii::CommandBuffers(device, allocate_info).front());

  command_buffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  command_buffer.copyBuffer(src, dst, vk::BufferCopy{.size = size});
  command_buffer.end();

  const vk::CommandBufferSubmitInfo command_buffer_info{.commandBuffer = *command_buffer};
  const vk::SubmitInfo2 submit_info{
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &command_buffer_info,
  };

  vk::raii::Fence fence(device, vk::FenceCreateInfo{});
  queue.submit2(submit_info, *fence);
  if (device.waitForFences(*fence, vk::True, UINT64_MAX) != vk::Result::eSuccess)
    throw std::runtime_error("Failed to wait for buffer upload fence");
}

} // namespace detail

// A device-local buffer whose memory comes from the shared DeviceAllocator
// rather than a vkAllocateMemory of its own. See device_allocator.hpp for the
// measurement that motivated the change.
//
// The move operations are written out rather than defaulted, and have to be: a
// defaulted move would copy the allocator pointer and the allocation handle and
// leave the source still holding both, so the range would be returned to the
// pool twice -- and the second return would hand out memory another buffer is
// using. Retired meshes are moved (DeferredDelete<MeshGpu>), so this is the
// common path, not an edge case.
class DeviceLocalBuffer {
public:
  DeviceLocalBuffer() = default;
  ~DeviceLocalBuffer() { release(); }

  DeviceLocalBuffer(const DeviceLocalBuffer &) = delete;
  auto operator=(const DeviceLocalBuffer &) -> DeviceLocalBuffer & = delete;

  DeviceLocalBuffer(DeviceLocalBuffer &&other) noexcept
      : buffer_(std::move(other.buffer_)),
        allocator_(std::exchange(other.allocator_, nullptr)),
        allocation_(std::exchange(other.allocation_, {})),
        pending_queue_(std::exchange(other.pending_queue_, nullptr)) {}

  auto operator=(DeviceLocalBuffer &&other) noexcept -> DeviceLocalBuffer & {
    if (this != &other) {
      release();
      buffer_ = std::move(other.buffer_);
      allocator_ = std::exchange(other.allocator_, nullptr);
      allocation_ = std::exchange(other.allocation_, {});
      pending_queue_ = std::exchange(other.pending_queue_, nullptr);
    }
    return *this;
  }

  void upload(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      vk::raii::CommandPool &command_pool,
      vk::raii::Queue &queue,
      vk::DeviceSize size,
      vk::BufferUsageFlags usage,
      const void *data) {
    const detail::Staging staging = detail::make_staging(device, physical_device, size, data);
    const auto memory_properties = physical_device.getMemoryProperties();

    const vk::BufferCreateInfo device_info{
        .size = size,
        .usage = usage | vk::BufferUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    buffer_ = vk::raii::Buffer(device, device_info);
    const vk::MemoryRequirements device_requirements = buffer_.getMemoryRequirements();
    memory_ = vk::raii::DeviceMemory{
        device,
        vk::MemoryAllocateInfo{
            .allocationSize = device_requirements.size,
            .memoryTypeIndex = detail::find_memory_type(
                memory_properties,
                device_requirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal),
        }};

    buffer_.bindMemory(*memory_, 0);
    detail::copy_buffer(device, command_pool, queue, *staging.buffer, *buffer_, size);
  }

  // Allocates the device buffer and queues the copy, without waiting. The
  // caller flushes the queue into the frame's command buffer. This is the path
  // streaming geometry must use -- the blocking upload() above costs a GPU
  // round trip per buffer, which is a dropped frame once chunks arrive
  // continuously.
  // Returns false when the frame's staging segment is full. The caller should
  // leave the slot unmarked so it retries next frame -- back-pressure rather
  // than either a stall or a buffer full of nothing.
  auto upload_deferred(
      DeviceAllocator &allocator,
      vk::raii::Device &device,
      UploadQueue &uploads,
      vk::DeviceSize size,
      vk::BufferUsageFlags usage,
      const void *data) -> bool {
    if (size == 0)
      return true;

    // Release before allocating, not after. A remesh replaces this buffer's
    // contents, and holding the old range while asking for a new one makes the
    // pool carry both -- which for a chunk being remeshed every frame is a
    // steadily growing pool that looks exactly like a leak.
    release();

    buffer_ = vk::raii::Buffer(device, vk::BufferCreateInfo{
        .size = size,
        .usage = usage | vk::BufferUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive,
    });
    const vk::MemoryRequirements requirements = buffer_.getMemoryRequirements();
    allocation_ = allocator.allocate(requirements, vk::MemoryPropertyFlagBits::eDeviceLocal);
    allocator_ = &allocator;
    buffer_.bindMemory(allocation_.memory, allocation_.offset);

    if (!uploads.stage(*buffer_, size, data))
      return false;
    // Remember where the copy is queued, so this buffer can withdraw it if it
    // is destroyed or re-uploaded before the queue is flushed. See
    // UploadQueue::cancel.
    pending_queue_ = &uploads;
    return true;
  }

  [[nodiscard]] auto handle() const -> vk::Buffer {
    return *buffer_;
  }

private:
  void release() {
    // Withdraw the queued copy BEFORE the handle goes away, not after: once
    // vk::raii::Buffer has run its destructor there is no handle left to name.
    if (pending_queue_ != nullptr) {
      pending_queue_->cancel(*buffer_);
      pending_queue_ = nullptr;
    }
    if (allocator_ != nullptr)
      allocator_->free(allocation_);
    allocator_ = nullptr;
    allocation_ = {};
  }

  vk::raii::Buffer buffer_{nullptr};
  // The queue holding a copy into this buffer that has not been recorded yet,
  // or null when there is none outstanding. Set by upload_deferred, cleared by
  // release. Not an ownership pointer: the queue lives in the VulkanContext and
  // outlives every buffer that stages into it.
  UploadQueue *pending_queue_{nullptr};
  // Only set on the pooled path (upload_deferred). The blocking upload() above
  // is startup-only and keeps its own dedicated allocation.
  vk::raii::DeviceMemory memory_{nullptr};
  DeviceAllocator *allocator_{nullptr};
  DeviceAllocator::Allocation allocation_{};
};

} // namespace engine
