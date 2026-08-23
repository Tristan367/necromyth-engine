// How much does one vkAllocateMemory per buffer actually cost?
//
// MeshGpu allocates dedicated device memory for its vertex buffer and again for
// its index buffer, so every chunk that streams in costs two of these, and every
// remesh costs two more. That reads like a problem -- vkAllocateMemory is a
// kernel round trip, and the count is capped -- but "reads like a problem" is
// not a measurement, and a sub-allocator is only worth writing if the number
// says so.
//
// So: time N dedicated allocations against N sub-allocations from one block,
// at a size typical of a chunk mesh, and print both. Also print
// maxMemoryAllocationCount, since on some drivers that is the wall rather than
// the latency.

#include "gpu_test_support.hpp"

#include "engine_config.hpp"
#include "platform/engine_runtime.hpp"
#include "scene/scene.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] auto ms_since(Clock::time_point start) -> double {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

} // namespace

auto main(int argc, char **argv) -> int {
  const int count = argc > 1 ? std::atoi(argv[1]) : 2000;
  // A chunk mesh in this game runs to a few hundred KB of vertices; 256 KB is a
  // fair stand-in and is large enough that any per-call overhead is not simply
  // lost in the noise of a tiny allocation.
  constexpr vk::DeviceSize k_buffer_bytes = 256 * 1024;

  const char *texture = std::getenv("VCE_STRESS_TEXTURE");
  if (texture == nullptr) {
    std::printf("Set VCE_STRESS_TEXTURE to any .png before running.\n");
    return EXIT_FAILURE;
  }

  engine::Scene scene;
  scene.camera().set_position({0.0F, 2.0F, 6.0F});
  (void)scene.add_texture(texture);
  (void)scene.add_texture_array_layer(texture);

  engine::test::request_validation();
  engine::EngineConfig config = engine::engine_config_from_environment();
  config.window_title = "allocation benchmark";
  engine::EngineRuntime runtime(config, scene);
  const engine::test::ValidationGuard validation(runtime.vulkan());

  auto &device = runtime.vulkan().device_ref();
  const auto memory_properties = runtime.vulkan().mem_props();

  std::printf("GPU: %s\n", runtime.vulkan().gpu_name().c_str());
  std::printf("maxMemoryAllocationCount: %u\n",
              runtime.vulkan().phys_dev().getProperties().limits.maxMemoryAllocationCount);
  std::printf("allocating %d buffers of %llu KB\n\n", count,
              static_cast<unsigned long long>(k_buffer_bytes / 1024));

  const vk::BufferCreateInfo buffer_info{
      .size = k_buffer_bytes,
      .usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .sharingMode = vk::SharingMode::eExclusive,
  };

  // --- What the engine does today: one allocation per buffer. ---
  {
    std::vector<vk::raii::Buffer> buffers;
    std::vector<vk::raii::DeviceMemory> memories;
    buffers.reserve(count);
    memories.reserve(count);

    const auto start = Clock::now();
    for (int i = 0; i < count; ++i) {
      vk::raii::Buffer buffer(device, buffer_info);
      const vk::MemoryRequirements requirements = buffer.getMemoryRequirements();
      vk::raii::DeviceMemory memory(device, vk::MemoryAllocateInfo{
          .allocationSize = requirements.size,
          .memoryTypeIndex = engine::detail::find_memory_type(
              memory_properties, requirements.memoryTypeBits,
              vk::MemoryPropertyFlagBits::eDeviceLocal),
      });
      buffer.bindMemory(*memory, 0);
      buffers.push_back(std::move(buffer));
      memories.push_back(std::move(memory));
    }
    const double elapsed = ms_since(start);
    std::printf("  dedicated allocation: %8.3f ms total, %7.4f ms each\n", elapsed,
                elapsed / count);

    const auto free_start = Clock::now();
    memories.clear();
    buffers.clear();
    std::printf("  ...freeing them:      %8.3f ms total, %7.4f ms each\n", ms_since(free_start),
                ms_since(free_start) / count);
  }

  // --- One block, sub-allocated: what a suballocator would cost instead. ---
  {
    // Probe the requirements once so the block is the right memory type.
    const vk::raii::Buffer probe(device, buffer_info);
    const vk::MemoryRequirements requirements = probe.getMemoryRequirements();
    const vk::DeviceSize stride =
        (k_buffer_bytes + requirements.alignment - 1) & ~(requirements.alignment - 1);

    std::vector<vk::raii::Buffer> buffers;
    std::vector<vk::raii::DeviceMemory> blocks;
    buffers.reserve(count);

    // Real suballocators cap block size; 64 MB is a common choice, so this also
    // measures the cost of rolling over to a new block.
    constexpr vk::DeviceSize k_block_bytes = 64ULL * 1024 * 1024;
    const auto per_block = static_cast<int>(k_block_bytes / stride);

    const auto start = Clock::now();
    vk::DeviceSize offset = 0;
    for (int i = 0; i < count; ++i) {
      if (blocks.empty() || i % per_block == 0) {
        blocks.emplace_back(device, vk::MemoryAllocateInfo{
            .allocationSize = k_block_bytes,
            .memoryTypeIndex = engine::detail::find_memory_type(
                memory_properties, requirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal),
        });
        offset = 0;
      }
      vk::raii::Buffer buffer(device, buffer_info);
      buffer.bindMemory(*blocks.back(), offset);
      offset += stride;
      buffers.push_back(std::move(buffer));
    }
    const double elapsed = ms_since(start);
    std::printf("\n  suballocated:         %8.3f ms total, %7.4f ms each  (%zu blocks)\n", elapsed,
                elapsed / count, blocks.size());
  }

  runtime.shutdown();

  if (!validation.check("allocation benchmark"))
    return EXIT_FAILURE;

  std::printf("\nok\n");
  return EXIT_SUCCESS;
}
