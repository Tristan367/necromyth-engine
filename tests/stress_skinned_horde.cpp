// GPU stress test for a horde of skinned characters.
//
// Every instance shares one skeleton and one texture, which is precisely the
// arrangement that used to break: the draw list sorts same-texture instances
// adjacent, the set-1 descriptor cache keyed only on the texture, and so every
// character after the first rendered with the first one's bone matrices in the
// main pass while the shadow pass stayed correct.
//
// It is also the arrangement the old bone storage scaled worst on -- two buffers
// and four descriptor sets per instance, all rebuilt behind a device wait_idle
// whenever the count changed.
//
// Run with the validation layers enabled. Characters are posed at staggered
// animation times, so if bone slices were aliased the horde would visibly move
// in lockstep.
//
// Not part of `make test`: needs a GPU and a display.

#include "engine_config.hpp"
#include "platform/engine_runtime.hpp"
#include "renderer/gltf_loader.hpp"
#include "scene/animation_state_machine.hpp"
#include "scene/scene.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

auto main(int argc, char **argv) -> int {
  const int frames = argc > 1 ? std::atoi(argv[1]) : 600;
  const int horde_size = argc > 2 ? std::atoi(argv[2]) : 120;

  const char *model_path = std::getenv("VCE_STRESS_MODEL");
  const char *texture_path = std::getenv("VCE_STRESS_TEXTURE");
  if (model_path == nullptr || texture_path == nullptr) {
    std::printf("Set VCE_STRESS_MODEL to a skinned .glb and VCE_STRESS_TEXTURE to a .png.\n");
    return EXIT_FAILURE;
  }

  engine::Scene scene;
  const std::uint32_t texture = scene.add_texture(texture_path);
  (void)scene.add_texture_array_layer(texture_path);

  engine::LoadedGltfModel model = engine::load_gltf_model(model_path);
  if (model.primitives.empty() || model.skeletons.empty() || model.animations.empty()) {
    std::printf("FAIL: %s has no skinned primitive with animations\n", model_path);
    return EXIT_FAILURE;
  }

  const std::uint32_t mesh = scene.add_mesh({.vertices = model.primitives[0].mesh.vertices,
                                             .indices = model.primitives[0].mesh.indices});
  const std::uint32_t skin = scene.add_skeleton(model.skeletons[0]);
  std::vector<std::uint32_t> clips;
  for (const engine::AnimationClip &clip : model.animations)
    clips.push_back(scene.add_animation(clip));

  // One state machine per character, so each owns its own pose stack. Held by
  // pointer because MeshInstance::pose_layers is a raw pointer into it -- a
  // vector that reallocates would dangle every one of them.
  std::vector<std::unique_ptr<engine::AnimStateMachine>> minds;
  std::vector<std::uint32_t> instances;

  const int side = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(horde_size))));
  for (int i = 0; i < horde_size; ++i) {
    auto mind = std::make_unique<engine::AnimStateMachine>();
    mind->add_state({"move", clips[0], true});
    mind->start("move");
    // Stagger the phase: identical poses would hide slice aliasing entirely.
    for (int step = 0; step < i % 37; ++step)
      mind->tick(1.0F / 60.0F, scene.animations());

    glm::mat4 xform(1.0F);
    xform[3] = glm::vec4(static_cast<float>(i % side) * 2.0F - static_cast<float>(side),
                         0.0F,
                         static_cast<float>(i / side) * 2.0F - static_cast<float>(side),
                         1.0F);

    const std::uint32_t index = scene.add_instance({
        .mesh_index = mesh,
        .texture_index = texture, // deliberately shared across the whole horde
        .model = xform,
        .skin_index = skin,
    });
    scene.instance(index).pose_layers = &mind->layers();
    minds.push_back(std::move(mind));
    instances.push_back(index);
  }

  scene.camera().look_at({0.0F, 12.0F, static_cast<float>(side) * 2.5F}, {0.0F, 0.0F, 0.0F});
  scene.directional_light().direction_toward_light = {0.4F, 1.0F, 0.3F};

  engine::EngineConfig config = engine::engine_config_from_environment();
  config.window_title = "VCE skinned horde stress";
  config.max_skinned_instances = static_cast<std::uint32_t>(horde_size) + 16U;
  engine::EngineRuntime runtime(config, scene);

  std::size_t despawned = 0;
  std::size_t respawned = 0;

  for (int frame = 0; frame < frames && !engine::EngineRuntime::quit_requested(); ++frame) {
    for (auto &mind : minds)
      mind->tick(1.0F / 60.0F, scene.animations());

    // Churn the horde: despawn and respawn characters mid-list so the bone slot
    // assignment shifts under the renderer every frame. This is what used to
    // reallocate every buffer and descriptor set behind a stall, and what
    // shifted every later instance onto the wrong skeleton.
    if (frame % 30 == 15 && instances.size() > 8) {
      scene.remove_instance(instances[instances.size() / 2]);
      ++despawned;
    }
    if (frame % 30 == 25) {
      auto mind = std::make_unique<engine::AnimStateMachine>();
      mind->add_state({"move", clips[0], true});
      mind->start("move");
      const std::uint32_t index = scene.add_instance({
          .mesh_index = mesh,
          .texture_index = texture,
          .skin_index = skin,
      });
      scene.instance(index).pose_layers = &mind->layers();
      minds.push_back(std::move(mind));
      instances.push_back(index);
      ++respawned;
    }

    runtime.vulkan().sync_scene(scene);
    runtime.vulkan().draw_frame(scene);
  }

  const engine::RenderStats stats = runtime.vulkan().render_stats();
  runtime.shutdown();

  std::printf("frames=%d horde=%d despawned=%zu respawned=%zu\n",
              frames, horde_size, despawned, respawned);
  std::printf("last frame: %u draws submitted, %u culled\n",
              stats.draws_submitted, stats.draws_culled);

  if (stats.main_pass_total() == 0) {
    std::printf("FAIL: nothing was drawn\n");
    return EXIT_FAILURE;
  }
  std::printf("ok\n");
  return EXIT_SUCCESS;
}
