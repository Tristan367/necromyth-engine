#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace engine {

// Holds GPU resources that have been replaced or removed but may still be
// referenced by command buffers the GPU has not finished with.
//
// The alternative is a full device wait_idle on every change, which is what the
// engine used to do. That is tolerable for a scene built once at startup and
// completely unworkable for a streaming world, where chunk meshes are created,
// remeshed and dropped continuously -- a stall per change means a stall per
// frame.
//
// A command buffer submitted on frame F is known complete once its fence has
// been waited on, which happens when that frame slot comes round again at
// F + frames_in_flight. Collecting one frame beyond that keeps the rule easy to
// verify by eye at negligible cost (a resource lives a few milliseconds longer).
template <typename T>
class DeferredDelete {
public:
  void retire(T resource, std::uint64_t current_frame) {
    pending_.push_back(Entry{.resource = std::move(resource), .retired_on = current_frame});
  }

  void collect(std::uint64_t current_frame, std::uint32_t frames_in_flight) {
    const std::uint64_t margin = static_cast<std::uint64_t>(frames_in_flight) + 1U;
    if (current_frame < margin)
      return;
    const std::uint64_t safe_before = current_frame - margin;

    std::size_t keep = 0;
    for (std::size_t i = 0; i < pending_.size(); ++i) {
      if (pending_[i].retired_on > safe_before) {
        if (keep != i)
          pending_[keep] = std::move(pending_[i]);
        ++keep;
      }
    }
    pending_.resize(keep);
  }

  // Only safe once the device is idle.
  void clear() { pending_.clear(); }

  [[nodiscard]] auto pending_count() const -> std::size_t { return pending_.size(); }

private:
  struct Entry {
    T resource;
    std::uint64_t retired_on{};
  };

  std::vector<Entry> pending_;
};

} // namespace engine
