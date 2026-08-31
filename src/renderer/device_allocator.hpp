#pragma once

#include "renderer/device_memory.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <future>
#include <limits>
#include <stdexcept>
#include <vector>

namespace engine {

// Sub-allocates buffer memory out of a few large blocks instead of calling
// vkAllocateMemory per buffer.
//
// Why this exists, measured rather than assumed (tests/bench_allocation.cpp,
// RTX 3060, 2000 buffers of 256 KB):
//
//   dedicated allocation   0.1280 ms each
//   ...and freeing it      0.1012 ms each
//   suballocated           0.0054 ms each
//
// A mesh is two buffers, so under the old scheme creating one cost about
// 0.26 ms and dropping it another 0.20 ms. Ten chunks remeshing in a frame --
// an ordinary moment in a streaming voxel world -- spent something like 4.6 ms
// inside the memory allocator, most of a 60 Hz budget, for no drawing
// whatsoever. Sub-allocation makes the same work roughly 24 times cheaper.
//
// The allocation COUNT is the other half of the argument and matters more off
// this machine: this GPU reports maxMemoryAllocationCount = 4294967295, but the
// limit is commonly 4096 elsewhere, and two allocations per chunk mesh reaches
// that inside a normal render distance.
//
// BUFFERS ONLY, deliberately. Mixing buffers and optimally-tiled images in one
// block drags in bufferImageGranularity: the two kinds must not share a
// granularity page, and getting that wrong produces corruption that no
// validation layer is obliged to catch. Buffers are always linear, so a
// buffer-only pool has no such rule to break. Images in this engine are few and
// long-lived -- depth, shadow maps, the texture table -- so they are not where
// the cost was, and they keep their dedicated allocations.
//
// Not thread safe. Every caller is the main thread, which is also the only
// thread that touches the device.
class DeviceAllocator {
public:
  static constexpr std::uint32_t k_invalid_block = std::numeric_limits<std::uint32_t>::max();

  // A slice of a block. `memory` is borrowed -- the allocator owns it, and this
  // handle is only valid until free() is called with it.
  //
  // `offset` is where the caller binds; `reserved_offset` and `reserved_size`
  // are the range the block actually gave up, which starts earlier when
  // alignment forced padding in front. Freeing has to return the reserved
  // range, not the usable one: returning [offset, offset + size) would strand
  // the padding at the front forever and overrun the end by the same amount --
  // a slow leak plus a corrupted free list, from one confused pair of numbers.
  struct Allocation {
    vk::DeviceMemory memory{};
    vk::DeviceSize offset{};
    vk::DeviceSize size{};
    vk::DeviceSize reserved_offset{};
    vk::DeviceSize reserved_size{};
    std::uint32_t block{k_invalid_block};

    [[nodiscard]] auto valid() const -> bool { return block != k_invalid_block; }
  };
  // Big enough that a normal render distance fits in a handful of blocks, small
  // enough that a mostly-empty pool is not holding a quarter of a gigabyte.
  static constexpr vk::DeviceSize k_default_block_bytes = 64ULL * 1024 * 1024;

  void create(const vk::raii::PhysicalDevice &physical_device,
              vk::raii::Device &device,
              vk::DeviceSize block_bytes = k_default_block_bytes) {
    device_ = &device;
    memory_properties_ = physical_device.getMemoryProperties();
    block_bytes_ = std::max<vk::DeviceSize>(block_bytes, 1024 * 1024);
  }

  // A prefetch in flight holds the device and returns a VkDeviceMemory, so it
  // has to be finished with before either can go away. std::async's future does
  // that on destruction and `prefetch_` is the last member declared -- so it is
  // the first destroyed -- but the ordering is load-bearing enough to say out
  // loud rather than leave to whoever next adds a member.
  ~DeviceAllocator() {
    if (prefetch_.valid())
      prefetch_.wait();
  }
  DeviceAllocator() = default;
  DeviceAllocator(DeviceAllocator &&) = default;
  auto operator=(DeviceAllocator &&) -> DeviceAllocator & = default;
  DeviceAllocator(const DeviceAllocator &) = delete;
  auto operator=(const DeviceAllocator &) -> DeviceAllocator & = delete;

  [[nodiscard]] auto created() const -> bool { return device_ != nullptr; }

  // Finds room for `requirements`, opening a new block if nothing fits.
  //
  // A request larger than the block size gets its own exactly-sized block. That
  // is a dedicated allocation wearing the same interface, which is the right
  // answer -- rounding a 100 MB buffer up to a 128 MB block would waste more
  // than the call costs.
  [[nodiscard]] auto allocate(const vk::MemoryRequirements &requirements,
                              vk::MemoryPropertyFlags properties) -> Allocation {
    const std::uint32_t memory_type = detail::find_memory_type(
        memory_properties_, requirements.memoryTypeBits, properties);
    const vk::DeviceSize alignment = std::max<vk::DeviceSize>(requirements.alignment, 1);

    adopt_prefetched_block();

    for (std::uint32_t index = 0; index < blocks_.size(); ++index) {
      Block &block = blocks_[index];
      if (block.size == 0 || block.memory_type != memory_type || block.dedicated)
        continue;
      if (const Allocation allocation = take_from(block, index, requirements.size, alignment);
          allocation.valid()) {
        request_prefetch(memory_type);
        return allocation;
      }
    }

    const bool dedicated = requirements.size > block_bytes_;
    const vk::DeviceSize block_size = dedicated ? requirements.size : block_bytes_;
    open_block(memory_type, block_size, dedicated);

    const Allocation allocation =
        take_from(blocks_.back(), static_cast<std::uint32_t>(blocks_.size() - 1),
                  requirements.size, alignment);
    if (!allocation.valid())
      throw std::runtime_error("DeviceAllocator: fresh block could not satisfy its own request");
    request_prefetch(memory_type);
    return allocation;
  }

  // Returns the range to its block and merges it with any neighbours.
  //
  // Coalescing is the whole reason this keeps sorted ranges rather than a plain
  // list: a streaming world frees and reallocates constantly, and without a
  // merge the free list fragments into a sawtooth that eventually cannot fit a
  // mesh despite having room for ten.
  void free(const Allocation &allocation) {
    if (!allocation.valid() || allocation.block >= blocks_.size())
      return;

    Block &block = blocks_[allocation.block];
    block.in_use -= std::min(block.in_use, allocation.reserved_size);

    const auto at =
        std::ranges::lower_bound(block.free_ranges, allocation.reserved_offset, {}, &Range::offset);
    const auto inserted = block.free_ranges.insert(
        at, Range{.offset = allocation.reserved_offset, .size = allocation.reserved_size});

    // Merge forward first: doing it in this order means the backward merge sees
    // the already-merged range and both collapse in one pass.
    const auto next = std::next(inserted);
    if (next != block.free_ranges.end() && inserted->offset + inserted->size == next->offset) {
      inserted->size += next->size;
      block.free_ranges.erase(next);
    }
    if (inserted != block.free_ranges.begin()) {
      const auto previous = std::prev(inserted);
      if (previous->offset + previous->size == inserted->offset) {
        previous->size += inserted->size;
        block.free_ranges.erase(inserted);
      }
    }
    reclaim_empty_blocks();
  }

  // Give a wholly-unused block back to the driver.
  //
  // Blocks were never released, so the pool was a permanent high-water mark:
  // walk somewhere dense once and the memory is held for the rest of the
  // session. At a 32-chunk render distance that reached eleven of twelve
  // gigabytes of VRAM and starved everything else on the machine -- reported
  // by another application refusing to start.
  //
  // One spare is kept. Freeing the last empty block the instant it empties
  // would hand it back and immediately re-allocate it on the next mesh, and a
  // 64MB vkAllocateMemory is the 1.8ms call this engine already went to some
  // trouble to move off the frame.
  // The SLOT stays; only the memory goes back.
  //
  // Allocation::block is an index into blocks_, so erasing an element would
  // renumber every allocation that came after it and corrupt the next free().
  // An emptied slot is left in place with size 0 and no memory; allocate()
  // skips it and open_block() reuses it.
  void reclaim_empty_blocks() {
    std::size_t empty_seen = 0;
    for (Block &block : blocks_) {
      if (block.size == 0 || block.dedicated || block.in_use != 0)
        continue;
      if (++empty_seen <= 1)
        continue; // keep one spare, see above
      block.memory = nullptr; // frees it: vk::raii owns the handle
      block.size = 0;
      block.free_ranges.clear();
      ++blocks_released_;
    }
  }

  // Observability, because an allocator you cannot see inside is one you cannot
  // tell apart from a leak.
  [[nodiscard]] auto block_count() const -> std::size_t { return blocks_.size(); }

  [[nodiscard]] auto bytes_reserved() const -> vk::DeviceSize {
    vk::DeviceSize total = 0;
    for (const Block &block : blocks_)
      total += block.size;
    return total;
  }

  [[nodiscard]] auto bytes_in_use() const -> vk::DeviceSize {
    vk::DeviceSize total = 0;
    for (const Block &block : blocks_)
      total += block.in_use;
    return total;
  }

  // How many vkAllocateMemory calls this has made in its lifetime. Under the
  // old scheme this equalled the number of buffers ever created; the point of
  // the exercise is that it no longer does.
  [[nodiscard]] auto device_allocations() const -> std::uint64_t { return device_allocations_; }

  // The slowest vkAllocateMemory this pool has made ON THE CALLING THREAD, in
  // milliseconds. Prefetched blocks are deliberately not counted: they cost the
  // frame nothing, which is the entire point of them.
  [[nodiscard]] auto worst_block_open_ms() const -> float { return worst_open_ms_; }
  // Blocks that were opened ahead of need rather than mid-frame.
  [[nodiscard]] auto prefetched_blocks() const -> std::uint64_t { return prefetched_blocks_; }
  // Blocks handed back to the driver once nothing was using them.
  [[nodiscard]] auto blocks_released() const -> std::uint64_t { return blocks_released_; }
  // For tests that want the old, entirely synchronous behaviour.
  void set_prefetch_enabled(bool enabled) { prefetch_enabled_ = enabled; }

  // How many separate free ranges a block has been broken into.
  //
  // This is the fragmentation measure, and the reason it is exposed: an
  // allocator that is losing to fragmentation still reports plenty of free
  // bytes, and only the shape of the free list says whether any of them are
  // usable. A block with everything returned should be back to exactly one.
  [[nodiscard]] auto free_range_count(std::size_t block) const -> std::size_t {
    return block < blocks_.size() ? blocks_[block].free_ranges.size() : 0;
  }

  // Every free range is non-empty, in ascending order, non-overlapping, within
  // the block, and not adjacent to its neighbour (adjacent means a coalesce was
  // missed). Costs a walk of the list, so it is for tests, not for frames.
  [[nodiscard]] auto free_list_is_well_formed() const -> bool {
    for (const Block &block : blocks_) {
      vk::DeviceSize previous_end = 0;
      bool first = true;
      for (const Range &range : block.free_ranges) {
        if (range.size == 0)
          return false;
        if (range.offset + range.size > block.size)
          return false;
        if (!first) {
          if (range.offset < previous_end)
            return false;             // overlapping or out of order
          if (range.offset == previous_end)
            return false;             // touching: should have been merged
        }
        previous_end = range.offset + range.size;
        first = false;
      }
    }
    return true;
  }

private:
  struct Range {
    vk::DeviceSize offset{};
    vk::DeviceSize size{};
  };

  struct Block {
    vk::raii::DeviceMemory memory{nullptr};
    vk::DeviceSize size{};
    vk::DeviceSize in_use{};
    std::uint32_t memory_type{};
    bool dedicated{false};
    std::vector<Range> free_ranges;
  };

  // Opens the next block on a worker thread, BEFORE the pool runs out of room.
  //
  // Measured, autowalking 3,000 frames: the costliest single mesh upload took
  // 1.81ms, of which 1.78ms was one vkAllocateMemory for a fresh 64 MB block.
  // That is the whole cost -- the memcpy, the two vkCreateBuffers and the
  // suballocation together are the remaining 0.03ms. It lands mid-frame,
  // whenever the pool happens to run out, which is while you are walking into
  // new terrain. A per-frame time budget cannot help: it can only stop BETWEEN
  // meshes, and this is one mesh.
  //
  // Doing it on another thread is legal, checked against the registry rather
  // than assumed: vk.xml lists no externsync parameter for vkAllocateMemory --
  // not even VkDevice -- so it may be called concurrently with anything else on
  // the same device. (vkFreeMemory externally syncs on `memory`; nothing here
  // frees a block.) Only the worker touches Vulkan; blocks_ stays main-thread.
  void request_prefetch(std::uint32_t memory_type) {
    if (!prefetch_enabled_ || prefetch_in_flight_)
      return;
    // Only worth doing while the pool is running low. Above the threshold the
    // next allocation will land in an existing block and no new one is due.
    if (largest_free_range(memory_type) > block_bytes_ / 4)
      return;

    prefetch_in_flight_ = true;
    prefetch_memory_type_ = memory_type;
    prefetch_ = std::async(std::launch::async, [this, memory_type] {
      return vk::raii::DeviceMemory(*device_, vk::MemoryAllocateInfo{
          .allocationSize = block_bytes_,
          .memoryTypeIndex = memory_type,
      });
    });
  }

  // Takes a prefetched block into the pool if one is ready. Never waits: a
  // block that has not arrived yet is simply not there this time round, and the
  // synchronous path in allocate() still covers the case.
  void adopt_prefetched_block() {
    if (!prefetch_in_flight_ || !prefetch_.valid())
      return;
    if (prefetch_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
      return;

    Block block;
    block.memory = prefetch_.get();
    block.size = block_bytes_;
    block.memory_type = prefetch_memory_type_;
    block.dedicated = false;
    block.free_ranges.push_back(Range{.offset = 0, .size = block_bytes_});
    bool placed = false;
    for (Block &slot : blocks_) {
      if (slot.size == 0 && !slot.dedicated) {
        slot = std::move(block);
        placed = true;
        break;
      }
    }
    if (!placed)
      blocks_.push_back(std::move(block));
    ++device_allocations_;
    ++prefetched_blocks_;
    prefetch_in_flight_ = false;
  }

  [[nodiscard]] auto largest_free_range(std::uint32_t memory_type) const -> vk::DeviceSize {
    vk::DeviceSize largest = 0;
    for (const Block &block : blocks_) {
      if (block.memory_type != memory_type || block.dedicated)
        continue;
      for (const Range &range : block.free_ranges)
        largest = std::max(largest, range.size);
    }
    return largest;
  }

  void open_block(std::uint32_t memory_type, vk::DeviceSize size, bool dedicated) {
    Block block;
    const auto started = std::chrono::steady_clock::now();
    block.memory = vk::raii::DeviceMemory(*device_, vk::MemoryAllocateInfo{
        .allocationSize = size,
        .memoryTypeIndex = memory_type,
    });
    // Opening a block is the one thing here that can cost a millisecond, and it
    // happens mid-frame, whenever the pool runs out. Worth knowing when a
    // single mesh upload turns out to be expensive: it is usually this.
    const float ms =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - started).count();
    if (ms > worst_open_ms_)
      worst_open_ms_ = ms;
    block.size = size;
    block.memory_type = memory_type;
    block.dedicated = dedicated;
    block.free_ranges.push_back(Range{.offset = 0, .size = size});
    // Reuse a reclaimed slot if there is one, so blocks_ does not grow without
    // bound as areas are entered and left.
    for (Block &slot : blocks_) {
      if (slot.size == 0 && !slot.dedicated) {
        slot = std::move(block);
        ++device_allocations_;
        return;
      }
    }
    blocks_.push_back(std::move(block));
    ++device_allocations_;
  }

  // First fit. Best fit would pack tighter, but first fit over a coalesced list
  // is O(ranges) with a tiny constant and the blocks here hold tens of ranges,
  // not thousands -- the measurement that motivated all of this was 0.128 ms
  // per call, and this is nowhere near that.
  [[nodiscard]] static auto take_from(Block &block, std::uint32_t index, vk::DeviceSize size,
                                      vk::DeviceSize alignment) -> Allocation {
    for (auto range = block.free_ranges.begin(); range != block.free_ranges.end(); ++range) {
      const vk::DeviceSize aligned = (range->offset + alignment - 1) & ~(alignment - 1);
      const vk::DeviceSize padding = aligned - range->offset;
      if (padding + size > range->size)
        continue;

      const Allocation allocation{
          .memory = *block.memory,
          .offset = aligned,
          .size = size,
          // The padding is charged to this allocation, so freeing it hands back
          // every byte it made unusable rather than stranding the gap in front.
          .reserved_offset = range->offset,
          .reserved_size = padding + size,
          .block = index,
      };

      const vk::DeviceSize remaining = range->size - (padding + size);
      if (remaining == 0)
        block.free_ranges.erase(range);
      else {
        range->offset = aligned + size;
        range->size = remaining;
      }

      block.in_use += allocation.reserved_size;
      return allocation;
    }

    return Allocation{};
  }

  vk::raii::Device *device_{nullptr};
  vk::PhysicalDeviceMemoryProperties memory_properties_{};
  vk::DeviceSize block_bytes_{k_default_block_bytes};
  std::vector<Block> blocks_;
  std::uint64_t device_allocations_{0};
  std::uint64_t prefetched_blocks_{0};
  std::uint64_t blocks_released_{0};
  float worst_open_ms_{0.0F};
  bool prefetch_enabled_{true};
  bool prefetch_in_flight_{false};
  std::uint32_t prefetch_memory_type_{0};
  std::future<vk::raii::DeviceMemory> prefetch_;
};

} // namespace engine
