// Deterministic point-shadow benchmark.
//
// The demo cannot answer "did that make point shadows faster?" because its
// camera is wherever you left it, and the cost depends entirely on how many
// lights are in view. This runs a fixed scene from two fixed camera positions:
// one looking at the lights, one looking away from them.
//
// Looking away is the case light culling exists for. Looking at them is the
// case it must not slow down or break.
//
// Needs a GPU and a display; not part of `make test`.

#include "gpu_test_support.hpp"

#include "engine_config.hpp"
#include "platform/engine_runtime.hpp"
#include "scene/scene.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <string>
#include <vector>

namespace {

[[nodiscard]] auto make_box(float size) -> engine::MeshSource {
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

struct Measurement {
  float point_shadow_ms{};
  float gpu_total_ms{};
  float cpu_total_ms{};
  std::uint32_t shadow_draws{};
};

[[nodiscard]] auto measure(engine::EngineRuntime &runtime, engine::Scene &scene,
                           const glm::vec3 &eye, const glm::vec3 &target, int frames) -> Measurement {
  scene.camera().look_at(eye, target);
  // Discard a warm-up window so the reported average covers only this camera.
  for (int i = 0; i < frames; ++i)
    runtime.vulkan().draw_frame(scene);

  const engine::GpuProfiler &gpu = runtime.vulkan().gpu_profiler();
  const engine::CpuProfiler &cpu = runtime.vulkan().cpu_profiler();

  Measurement result;
  result.point_shadow_ms = gpu.stats(engine::GpuZone::ShadowPoint).average_ms;
  for (std::uint32_t i = 0; i < engine::GpuProfiler::k_zone_count; ++i)
    result.gpu_total_ms += gpu.stats(static_cast<engine::GpuZone>(i)).average_ms;
  result.cpu_total_ms = cpu.stats(engine::CpuZone::FrameTotal).average_ms;
  result.shadow_draws = runtime.vulkan().render_stats().shadow_draws_submitted;
  return result;
}

} // namespace

auto main(int argc, char **argv) -> int {
  const int light_count = argc > 1 ? std::atoi(argv[1]) : 10;

  const char *texture = std::getenv("VCE_STRESS_TEXTURE");
  if (texture == nullptr) {
    std::printf("Set VCE_STRESS_TEXTURE to any .png before running.\n");
    return EXIT_FAILURE;
  }

  engine::Scene scene;
  (void)scene.add_texture(texture);
  (void)scene.add_texture_array_layer(texture);
  scene.directional_light().direction_toward_light = {0.4F, 1.0F, 0.3F};

  const std::uint32_t box = scene.add_mesh(make_box(1.0F));
  const std::uint32_t ground = scene.add_mesh(make_box(60.0F));

  glm::mat4 ground_xform(1.0F);
  ground_xform[3] = glm::vec4(0.0F, -30.5F, 0.0F, 1.0F);
  (void)scene.add_instance({.mesh_index = ground, .texture_index = 0, .model = ground_xform});

  // Clutter for the shadow passes to chew on.
  for (int i = 0; i < 80; ++i) {
    const float angle = static_cast<float>(i) * 0.4F;
    glm::mat4 xform(1.0F);
    xform[3] = glm::vec4(std::cos(angle) * (2.0F + static_cast<float>(i % 7)),
                         0.5F,
                         std::sin(angle) * (2.0F + static_cast<float>(i % 5)),
                         1.0F);
    (void)scene.add_instance({.mesh_index = box, .texture_index = 0, .model = xform});
  }

  // Shadow-casting point lights, clustered near the origin like the demo's
  // stress block.
  for (int i = 0; i < light_count; ++i) {
    const float angle = static_cast<float>(i) * std::numbers::pi_v<float> * 2.0F /
                        static_cast<float>(light_count);
    scene.point_lights().push_back({
        .position = {std::cos(angle) * 4.0F, 1.5F, std::sin(angle) * 4.0F},
        .color = {1.0F, 0.9F, 0.8F},
        .intensity = 2.0F,
        .range = 8.0F,
        .casts_shadow = true,
    });
  }

  // Validation on by default here: this test exists to catch exactly the
  // hazards the layer reports, and a run without it proves much less.
  engine::test::request_validation();
  engine::EngineConfig config = engine::engine_config_from_environment();
  config.window_title = "VCE point shadow benchmark";
  engine::EngineRuntime runtime(config, scene);
  const engine::test::ValidationGuard validation(runtime.vulkan());

  constexpr int k_window = 120; // matches the profiler's averaging window

  // Two full windows each, so the reported average covers only the second.
  const Measurement toward =
      measure(runtime, scene, {0.0F, 6.0F, 16.0F}, {0.0F, 0.0F, 0.0F}, k_window * 2);
  const Measurement away =
      measure(runtime, scene, {0.0F, 6.0F, 160.0F}, {0.0F, 6.0F, 400.0F}, k_window * 2);

  runtime.shutdown();

  std::printf("\npoint shadow benchmark -- %d shadow-casting point lights\n", light_count);
  std::printf("  %-22s %10s %10s %10s %8s\n", "camera", "pt shadow", "gpu total", "cpu total", "sh.draws");
  std::printf("  %-22s %8.3f ms %8.3f ms %8.3f ms %8u\n", "toward the lights",
              toward.point_shadow_ms, toward.gpu_total_ms, toward.cpu_total_ms, toward.shadow_draws);
  std::printf("  %-22s %8.3f ms %8.3f ms %8.3f ms %8u\n", "away from the lights",
              away.point_shadow_ms, away.gpu_total_ms, away.cpu_total_ms, away.shadow_draws);

  if (away.point_shadow_ms > toward.point_shadow_ms * 0.5F) {
    std::printf("\nFAIL: looking away did not reduce point shadow cost -- lights outside the\n"
                "      camera frustum are still being rendered.\n");
    return EXIT_FAILURE;
  }
  if (!validation.check("point shadows"))
    return EXIT_FAILURE;

  std::printf("\nok\n");
  return EXIT_SUCCESS;
}
