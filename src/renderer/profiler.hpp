#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace engine {

// GPU work, measured with timestamps written into the command buffer.
enum class GpuZone : std::uint8_t {
  // Copying newly meshed geometry into device-local buffers. Unmeasured until
  // now, and it is the one pass whose cost scales with how fast you are walking
  // into new terrain -- which is exactly when the game staggers.
  Upload,
  ShadowDirectional,
  ShadowSpot,
  ShadowPoint,
  MainPass,
  Count,
};

// CPU work, measured on the host. WaitForFrame and AcquireImage are the two that
// say whether you are GPU-bound or presentation-bound: if the frame is cheap on
// both CPU and GPU but WaitForFrame is large, you are waiting on the display,
// not on rendering, and optimising draw submission will change nothing.
enum class CpuZone : std::uint8_t {
  WaitForFrame,
  AcquireImage,
  SyncScene,
  Animation,
  BuildDrawLists,
  RecordCommands,
  Submit,
  Present,
  FrameTotal,
  Count,
};

[[nodiscard]] inline auto gpu_zone_name(GpuZone zone) -> const char * {
  switch (zone) {
  case GpuZone::Upload:            return "geometry upload";
  case GpuZone::ShadowDirectional: return "shadow (directional)";
  case GpuZone::ShadowSpot:        return "shadow (spot)";
  case GpuZone::ShadowPoint:       return "shadow (point)";
  case GpuZone::MainPass:          return "main pass";
  case GpuZone::Count:             break;
  }
  return "?";
}

[[nodiscard]] inline auto cpu_zone_name(CpuZone zone) -> const char * {
  switch (zone) {
  case CpuZone::WaitForFrame:   return "wait for frame fence";
  case CpuZone::AcquireImage:   return "acquire swapchain image";
  case CpuZone::SyncScene:      return "sync scene";
  case CpuZone::Animation:      return "animation / skinning";
  case CpuZone::BuildDrawLists: return "build draw lists";
  case CpuZone::RecordCommands: return "record commands";
  case CpuZone::Submit:         return "submit";
  case CpuZone::Present:        return "present";
  case CpuZone::FrameTotal:     return "FRAME TOTAL (cpu)";
  case CpuZone::Count:          break;
  }
  return "?";
}

// Rolling statistics for one zone. Averaged over a window rather than smoothed
// exponentially so the number on screen is one you could have counted by hand,
// and so `max` means "the worst frame in the last window" -- which is what a
// hitch actually is.
struct ZoneStats {
  float average_ms{0.0F};
  float max_ms{0.0F};
};

namespace detail {

class ZoneAccumulator {
public:
  void add(float sample_ms) {
    sum_ += sample_ms;
    peak_ = std::max(peak_, sample_ms);
    ++count_;
  }

  void flush() {
    published_.average_ms = count_ > 0 ? sum_ / static_cast<float>(count_) : 0.0F;
    published_.max_ms = peak_;
    sum_ = 0.0F;
    peak_ = 0.0F;
    count_ = 0;
  }

  [[nodiscard]] auto stats() const -> const ZoneStats & { return published_; }

private:
  ZoneStats published_{};
  float sum_{0.0F};
  float peak_{0.0F};
  std::uint32_t count_{0};
};

} // namespace detail

// Timestamps every pass on the GPU.
//
// Timestamps are written into the command buffer and read back only once the
// frame's fence has been waited on, so collection never blocks: the results
// gathered at the top of a frame belong to that slot's previous submission.
// Zones whose pass did not run that frame report unavailable and are shown as
// zero rather than as a stale value.
class GpuProfiler {
public:
  static constexpr std::uint32_t k_zone_count = static_cast<std::uint32_t>(GpuZone::Count);
  static constexpr std::uint32_t k_queries_per_frame = k_zone_count * 2;

  void create(const vk::raii::PhysicalDevice &physical_device,
              vk::raii::Device &device,
              std::uint32_t frame_count,
              std::uint32_t graphics_queue_family) {
    const vk::PhysicalDeviceProperties properties = physical_device.getProperties();
    timestamp_period_ns_ = properties.limits.timestampPeriod;

    const auto families = physical_device.getQueueFamilyProperties();
    const bool queue_supports_timestamps =
        graphics_queue_family < families.size() && families[graphics_queue_family].timestampValidBits > 0;

    // timestampPeriod of 0 means the device does not support timestamps at all.
    if (timestamp_period_ns_ <= 0.0F || !queue_supports_timestamps) {
      supported_ = false;
      return;
    }

    frame_count_ = std::max(frame_count, 1U);
    pool_ = vk::raii::QueryPool(
        device,
        vk::QueryPoolCreateInfo{
            .queryType = vk::QueryType::eTimestamp,
            .queryCount = k_queries_per_frame * frame_count_,
        });
    frame_written_.assign(frame_count_, false);
    supported_ = true;
  }

  [[nodiscard]] auto supported() const -> bool { return supported_; }

  // Must precede any timestamp write for this frame slot: a query has to be
  // reset before it is written again.
  void begin_frame(vk::raii::CommandBuffer &command_buffer, std::uint32_t frame_index) {
    if (!supported_)
      return;
    command_buffer.resetQueryPool(*pool_, frame_index * k_queries_per_frame, k_queries_per_frame);
    for (bool &written : zone_written_)
      written = false;
  }

  void begin_zone(vk::raii::CommandBuffer &command_buffer, std::uint32_t frame_index, GpuZone zone) {
    if (!supported_)
      return;
    command_buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, *pool_,
                                   query_index(frame_index, zone, false));
  }

  void end_zone(vk::raii::CommandBuffer &command_buffer, std::uint32_t frame_index, GpuZone zone) {
    if (!supported_)
      return;
    command_buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *pool_,
                                   query_index(frame_index, zone, true));
    zone_written_[static_cast<std::uint32_t>(zone)] = true;
  }

  void mark_frame_recorded(std::uint32_t frame_index) {
    if (supported_ && frame_index < frame_written_.size())
      frame_written_[frame_index] = true;
  }

  // Call after this slot's fence has been waited on, before the slot is reused.
  // Reads the previous submission's results; never waits.
  void collect(std::uint32_t frame_index) {
    if (!supported_ || frame_index >= frame_written_.size() || !frame_written_[frame_index])
      return;

    // Two values per query: the timestamp and whether it was ever written.
    // Passes that did not run this frame legitimately have no result.
    std::array<std::uint64_t, k_queries_per_frame * 2> results{};
    const vk::Result result = pool_.getResults(
        frame_index * k_queries_per_frame,
        k_queries_per_frame,
        sizeof(results),
        results.data(),
        sizeof(std::uint64_t) * 2,
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
    if (result != vk::Result::eSuccess && result != vk::Result::eNotReady)
      return;

    for (std::uint32_t zone = 0; zone < k_zone_count; ++zone) {
      const std::size_t begin = static_cast<std::size_t>(zone) * 4;      // (value, avail) x2
      const std::size_t end = begin + 2;
      const bool available = results[begin + 1] != 0 && results[end + 1] != 0;
      if (!available || results[end] < results[begin]) {
        accumulators_[zone].add(0.0F);
        continue;
      }
      const double ticks = static_cast<double>(results[end] - results[begin]);
      accumulators_[zone].add(static_cast<float>(ticks * timestamp_period_ns_ * 1e-6));
    }
  }

  void flush() {
    for (detail::ZoneAccumulator &accumulator : accumulators_)
      accumulator.flush();
  }

  [[nodiscard]] auto stats(GpuZone zone) const -> const ZoneStats & {
    return accumulators_[static_cast<std::uint32_t>(zone)].stats();
  }

private:
  [[nodiscard]] static auto query_index(std::uint32_t frame_index, GpuZone zone, bool end)
      -> std::uint32_t {
    return frame_index * k_queries_per_frame + static_cast<std::uint32_t>(zone) * 2 + (end ? 1U : 0U);
  }

  vk::raii::QueryPool pool_{nullptr};
  std::array<detail::ZoneAccumulator, k_zone_count> accumulators_{};
  std::array<bool, k_zone_count> zone_written_{};
  std::vector<bool> frame_written_;
  float timestamp_period_ns_{0.0F};
  std::uint32_t frame_count_{0};
  bool supported_{false};
};

// Host-side phase timings.
class CpuProfiler {
public:
  static constexpr std::uint32_t k_zone_count = static_cast<std::uint32_t>(CpuZone::Count);

  using Clock = std::chrono::steady_clock;

  void begin(CpuZone zone) { starts_[static_cast<std::uint32_t>(zone)] = Clock::now(); }

  void end(CpuZone zone) {
    const auto index = static_cast<std::uint32_t>(zone);
    const auto elapsed = std::chrono::duration<float, std::milli>(Clock::now() - starts_[index]);
    accumulators_[index].add(elapsed.count());
  }

  void flush() {
    for (detail::ZoneAccumulator &accumulator : accumulators_)
      accumulator.flush();
  }

  [[nodiscard]] auto stats(CpuZone zone) const -> const ZoneStats & {
    return accumulators_[static_cast<std::uint32_t>(zone)].stats();
  }

private:
  std::array<Clock::time_point, k_zone_count> starts_{};
  std::array<detail::ZoneAccumulator, k_zone_count> accumulators_{};
};

// RAII timer, so an early return cannot leave a zone open.
class ScopedCpuZone {
public:
  ScopedCpuZone(CpuProfiler &profiler, CpuZone zone) : profiler_(&profiler), zone_(zone) {
    profiler_->begin(zone_);
  }
  ~ScopedCpuZone() { profiler_->end(zone_); }

  ScopedCpuZone(const ScopedCpuZone &) = delete;
  auto operator=(const ScopedCpuZone &) -> ScopedCpuZone & = delete;
  ScopedCpuZone(ScopedCpuZone &&) = delete;
  auto operator=(ScopedCpuZone &&) -> ScopedCpuZone & = delete;

private:
  CpuProfiler *profiler_;
  CpuZone zone_;
};

} // namespace engine
