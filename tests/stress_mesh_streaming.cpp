// GPU stress test for mesh streaming.
//
// Opens a real window and device, then churns mesh slots the way a voxel world
// does: every frame some chunks are created, some remeshed in place, and some
// dropped, while instances referencing them come and go. Run it with the
// validation layers enabled -- the failure modes it targets (freeing a buffer an
// in-flight command buffer still references, drawing from a recycled slot) are
// exactly what the layers catch and what a silent corruption would otherwise
// hide until it showed up as garbage geometry.
//
// Not part of `make test`: it needs a GPU and a display. Build and run it by
// hand, or from CI on a machine that has both.

#include "gpu_test_support.hpp"

#include "engine_config.hpp"
#include "platform/engine_runtime.hpp"
#include "scene/scene.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

// A cube, standing in for a meshed chunk. `size` varies so bounds changes are
// observable and so each revision really is different geometry.
[[nodiscard]] auto make_chunk_mesh(float size) -> engine::MeshSource {
  engine::MeshSource mesh;
  const float h = size * 0.5F;
  const float corners[8][3] = {
      {-h, -h, -h}, {h, -h, -h}, {h, h, -h}, {-h, h, -h},
      {-h, -h, h},  {h, -h, h},  {h, h, h},  {-h, h, h},
  };
  for (const auto &c : corners) {
    engine::MeshVertex v{};
    v.pos[0] = c[0];
    v.pos[1] = c[1];
    v.pos[2] = c[2];
    v.normal[1] = 1.0F;
    v.color[0] = v.color[1] = v.color[2] = 1.0F;
    mesh.vertices.push_back(v);
  }
  const std::uint32_t faces[36] = {
      0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 0, 4, 7, 7, 3, 0,
      1, 5, 6, 6, 2, 1, 3, 2, 6, 6, 7, 3, 0, 1, 5, 5, 4, 0,
  };
  mesh.indices.assign(std::begin(faces), std::end(faces));
  return mesh;
}

} // namespace

auto main(int argc, char **argv) -> int {
  const int frames = argc > 1 ? std::atoi(argv[1]) : 600;

  const char *texture = std::getenv("VCE_STRESS_TEXTURE");
  if (texture == nullptr) {
    std::printf("Set VCE_STRESS_TEXTURE to any .png before running.\n");
    return EXIT_FAILURE;
  }

  engine::Scene scene;
  (void)scene.add_texture(texture);
  (void)scene.add_texture_array_layer(texture);
  scene.camera().look_at({0.0F, 20.0F, 40.0F}, {0.0F, 0.0F, 0.0F});
  scene.directional_light().direction_toward_light = {0.4F, 1.0F, 0.3F};

  // A resident mesh that is never touched. If streaming ever corrupts unrelated
  // slots, this is what stops rendering.
  const std::uint32_t resident = scene.add_mesh(make_chunk_mesh(4.0F));
  (void)scene.add_instance({.mesh_index = resident, .texture_index = 0});

  // Validation on by default here: this test exists to catch exactly the
  // hazards the layer reports, and a run without it proves much less.
  engine::test::request_validation();
  engine::EngineConfig config = engine::engine_config_from_environment();
  config.window_title = "VCE mesh streaming stress";
  engine::EngineRuntime runtime(config, scene);
  const engine::test::ValidationGuard validation(runtime.vulkan());

  // Slot -> instance, for the chunks currently streamed in.
  struct Chunk {
    std::uint32_t mesh{};
    engine::InstanceHandle instance{};
  };
  std::vector<Chunk> live;

  std::size_t peak_slots = 0;
  std::size_t total_created = 0;
  std::size_t total_removed = 0;
  std::size_t total_updated = 0;

  for (int frame = 0; frame < frames && !engine::EngineRuntime::quit_requested(); ++frame) {
    // Stream in.
    for (int i = 0; i < 3; ++i) {
      const float size = 1.0F + static_cast<float>((frame + i) % 5);
      const std::uint32_t mesh = scene.add_mesh(make_chunk_mesh(size));
      glm::mat4 model(1.0F);
      model[3] = glm::vec4(static_cast<float>((frame * 3 + i) % 40) - 20.0F, 0.0F,
                           static_cast<float>((frame + i) % 40) - 20.0F, 1.0F);
      const engine::InstanceHandle instance = scene.add_instance({
          .mesh_index = mesh,
          .texture_index = 0,
          .model = model,
      });
      live.push_back({mesh, instance});
      ++total_created;
    }

    // Remesh in place -- the voxel-edit / LOD-swap path.
    for (std::size_t i = 0; i < live.size(); i += 7) {
      scene.update_mesh(live[i].mesh, make_chunk_mesh(1.0F + static_cast<float>(frame % 6)));
      ++total_updated;
    }

    // Stream out, oldest first, keeping a bounded working set.
    while (live.size() > 60) {
      scene.remove_instance(live.front().instance);
      scene.remove_mesh(live.front().mesh);
      live.erase(live.begin());
      ++total_removed;
    }

    peak_slots = std::max(peak_slots, scene.mesh_count());

    runtime.vulkan().sync_scene(scene);
    runtime.vulkan().draw_frame(scene);
  }

  runtime.shutdown();

  std::printf("frames=%d created=%zu updated=%zu removed=%zu\n",
              frames, total_created, total_updated, total_removed);
  std::printf("mesh slot capacity=%zu (live=%zu) -- must stay bounded, not grow with `created`\n",
              scene.mesh_count(), scene.live_mesh_count());

  // The buffer pool has to behave the same way: bounded by the working set, not
  // by total churn. Two numbers say whether it does -- how many device
  // allocations it ever made (under the old dedicated scheme this was two per
  // upload, so 14238 here), and how many bytes it is still holding out.
  const auto &pool = runtime.vulkan().buffer_pool();
  std::printf("buffer pool: %llu device allocations for %zu uploads, %zu blocks, "
              "%.1f MB in use of %.1f MB reserved\n",
              static_cast<unsigned long long>(pool.device_allocations()),
              (total_created + total_updated) * 2, pool.block_count(),
              static_cast<double>(pool.bytes_in_use()) / (1024.0 * 1024.0),
              static_cast<double>(pool.bytes_reserved()) / (1024.0 * 1024.0));

  if (pool.device_allocations() > (total_created + total_updated) / 4) {
    std::printf("FAIL: pool made %llu device allocations -- it is not reusing memory\n",
                static_cast<unsigned long long>(pool.device_allocations()));
    return EXIT_FAILURE;
  }

  // The point of slot reuse: capacity tracks the working set, not total churn.
  if (total_created > 200 && peak_slots > total_created / 2) {
    std::printf("FAIL: slot capacity %zu grew with churn (%zu created) -- slots are not being reused\n",
                peak_slots, total_created);
    return EXIT_FAILURE;
  }

  if (!validation.check("mesh streaming"))
    return EXIT_FAILURE;

  std::printf("ok\n");
  return EXIT_SUCCESS;
}
