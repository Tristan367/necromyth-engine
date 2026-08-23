#pragma once

// Shared scaffolding for the tests that need a real GPU.
//
// The important piece is validation_guard: the validation layer used to be
// something you had to remember to enable and then read the output of, which
// means in practice nobody did either. Counting the messages turns it into a
// pass/fail result, so a hazard introduced next month fails the test that runs
// after it instead of surviving until it corrupts a frame on somebody else's
// card.

#include "renderer/vulkan_context.hpp"
#include "renderer/vulkan_device.hpp"

#include <cstdio>
#include <cstdlib>

namespace engine::test {

// Asks for validation before the device is created, so a caller does not have
// to remember the environment variables. Returns what was actually requested,
// which is not the same thing -- the layer may not be installed.
inline void request_validation(bool synchronization = true) {
  // setenv rather than a constructor argument because VulkanDevice reads the
  // environment: the test and a hand-run of the same binary then take the same
  // path, and there is no second way to configure validation to keep in sync.
  ::setenv("ENGINE_VALIDATION", "1", 0);
  if (synchronization)
    ::setenv("ENGINE_SYNC_VALIDATION", "1", 0);
}

// Fails the test if the layer reported anything at warning or above.
//
// Construct after the device exists and check() at the end. Deliberately not a
// destructor: a test should fail by returning a status, not by whatever a
// throwing destructor does.
//
// Counts from process start rather than from construction. It cannot do
// otherwise and be honest: pipelines, images and descriptor layouts are all
// built inside the VulkanContext constructor, so a guard that baselined itself
// afterwards would skip every message from device setup -- which is exactly
// what it did on its first run, printing "validation clean" over a pipeline
// warning sitting in stderr. request_validation() runs before the device is
// created, so every message in the process belongs to this test.
class ValidationGuard {
public:
  explicit ValidationGuard(const VulkanContext &context)
      : enabled_(context.validation_enabled()),
        synchronization_(context.sync_validation_enabled()) {
    if (!enabled_)
      std::printf(
          "WARNING: validation layer unavailable -- this run proves much less than it looks.\n");
  }

  [[nodiscard]] auto check(const char *label) const -> bool {
    if (!enabled_)
      return true;

    const std::uint64_t reported = VulkanDevice::validation_messages();
    if (reported == 0) {
      std::printf("%s: validation clean%s\n", label,
                  synchronization_ ? " (including synchronization)" : "");
      return true;
    }

    std::printf("FAIL: %s produced %llu validation message(s) -- see stderr above\n", label,
                static_cast<unsigned long long>(reported));
    return false;
  }

private:
  bool enabled_;
  bool synchronization_;
};

} // namespace engine::test
