// Does the buffer pool actually hand out memory that nobody else is using?
//
// A suballocator fails quietly. Overlapping two buffers does not crash, does not
// trip the validation layers, and does not look like a memory bug -- it looks
// like geometry that is subtly wrong, which is indistinguishable from a mesher
// bug and is where you would spend the next two days. So the invariants get
// asserted directly rather than inferred from "the stress test still passes".
//
// Needs a real device (vkAllocateMemory is the thing under test), so it is not
// part of `make test`.

#include "gpu_test_support.hpp"

#include "engine_config.hpp"
#include "platform/engine_runtime.hpp"
#include "renderer/device_allocator.hpp"
#include "scene/scene.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

struct Live {
  engine::DeviceAllocator::Allocation allocation;
  vk::DeviceSize requested{};
};

int failures = 0;

void check(bool condition, const char *what) {
  if (!condition) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

// The invariant that matters: no two live allocations share a byte.
//
// Compares only allocations in the same VkDeviceMemory -- ranges in different
// blocks are allowed to have identical offsets and frequently do, which is
// exactly the sort of thing a naive overlap check gets wrong and then reports
// as a bug in the allocator instead of in itself.
[[nodiscard]] auto any_overlap(const std::vector<Live> &live) -> bool {
  std::vector<Live> sorted = live;
  std::ranges::sort(sorted, [](const Live &a, const Live &b) {
    if (a.allocation.memory != b.allocation.memory)
      return a.allocation.memory < b.allocation.memory;
    return a.allocation.offset < b.allocation.offset;
  });

  for (std::size_t i = 1; i < sorted.size(); ++i) {
    const Live &previous = sorted[i - 1];
    const Live &current = sorted[i];
    if (previous.allocation.memory != current.allocation.memory)
      continue;
    if (previous.allocation.offset + previous.requested > current.allocation.offset)
      return true;
  }
  return false;
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
  config.window_title = "device allocator test";
  engine::EngineRuntime runtime(config, scene);
  const engine::test::ValidationGuard validation(runtime.vulkan());

  auto &device = runtime.vulkan().device_ref();

  engine::DeviceAllocator allocator;
  // A small block on purpose: 4 MB forces the pool to open several blocks and
  // to roll over between them, which a 64 MB block would hide entirely at this
  // scale. Bugs live at the boundaries.
  allocator.create(runtime.vulkan().phys_dev(), device, 4ULL * 1024 * 1024);

  // Probe real requirements rather than inventing an alignment: the whole point
  // is to honour what the driver asks for.
  const vk::raii::Buffer probe(device, vk::BufferCreateInfo{
      .size = 1024,
      .usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .sharingMode = vk::SharingMode::eExclusive,
  });
  const vk::MemoryRequirements probe_requirements = probe.getMemoryRequirements();
  std::printf("alignment=%llu memoryTypeBits=0x%x\n",
              static_cast<unsigned long long>(probe_requirements.alignment),
              probe_requirements.memoryTypeBits);

  std::mt19937 rng(12345);
  // Spanning three orders of magnitude, because same-size allocations are the
  // case a broken free list survives: every range fits every hole.
  std::uniform_int_distribution<vk::DeviceSize> size_dist(256, 512 * 1024);

  std::vector<Live> live;
  const auto allocate_one = [&] {
    vk::MemoryRequirements requirements = probe_requirements;
    requirements.size = size_dist(rng);
    const auto allocation =
        allocator.allocate(requirements, vk::MemoryPropertyFlagBits::eDeviceLocal);
    check(allocation.valid(), "allocation succeeded");
    check(allocation.offset % probe_requirements.alignment == 0, "allocation is aligned");
    live.push_back(Live{.allocation = allocation, .requested = requirements.size});
  };

  // --- Fill, then churn: free a random half and refill, repeatedly. ---
  for (int i = 0; i < 400; ++i)
    allocate_one();
  check(!any_overlap(live), "no overlap after initial fill");
  check(allocator.free_list_is_well_formed(), "free list well formed after initial fill");

  for (int round = 0; round < 20; ++round) {
    std::ranges::shuffle(live, rng);
    const std::size_t drop = live.size() / 2;
    for (std::size_t i = 0; i < drop; ++i)
      allocator.free(live[i].allocation);
    live.erase(live.begin(), live.begin() + static_cast<std::ptrdiff_t>(drop));

    for (std::size_t i = 0; i < drop; ++i)
      allocate_one();

    check(!any_overlap(live), "no overlap after churn round");
    check(allocator.free_list_is_well_formed(), "free list well formed after churn round");
  }

  const std::size_t blocks_after_churn = allocator.block_count();
  const std::uint64_t device_allocations = allocator.device_allocations();
  std::printf("after churn: %zu blocks, %llu device allocations for %d requests\n",
              blocks_after_churn, static_cast<unsigned long long>(device_allocations), 400 + 20 * 200);

  // --- Everything returned: each block should coalesce back to one range. ---
  for (const Live &entry : live)
    allocator.free(entry.allocation);
  live.clear();

  check(allocator.bytes_in_use() == 0, "pool reports nothing in use once all is freed");
  check(allocator.free_list_is_well_formed(), "free list well formed when empty");
  for (std::size_t block = 0; block < allocator.block_count(); ++block) {
    if (allocator.free_range_count(block) != 1) {
      std::printf("FAIL: block %zu coalesced to %zu ranges, expected 1\n", block,
                  allocator.free_range_count(block));
      ++failures;
    }
  }

  // --- A request larger than the block size gets its own block. ---
  {
    const std::size_t before = allocator.block_count();
    vk::MemoryRequirements oversized = probe_requirements;
    oversized.size = 6ULL * 1024 * 1024; // block size is 4 MB
    const auto allocation =
        allocator.allocate(oversized, vk::MemoryPropertyFlagBits::eDeviceLocal);
    check(allocation.valid(), "oversized allocation succeeded");
    check(allocator.block_count() == before + 1, "oversized allocation opened its own block");
    allocator.free(allocation);
  }

  runtime.shutdown();

  if (!validation.check("device allocator"))
    return EXIT_FAILURE;
  if (failures > 0) {
    std::printf("\n%d check(s) failed\n", failures);
    return EXIT_FAILURE;
  }

  std::printf("\nok\n");
  return EXIT_SUCCESS;
}
