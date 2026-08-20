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

  // One state machine per character, in a plain vector that reallocates as the
  // horde grows. That is deliberate: MeshInstance::pose_layers shares ownership
  // of the pose stack, so relocating the state machines moves only the pointers
  // and every instance keeps working. Back when pose_layers was a raw pointer
  // into the state machine, this vector had to be a vector of unique_ptr or
  // every character would have been reading freed memory after the first regrow.
  std::vector<engine::AnimStateMachine> minds;
  std::vector<engine::InstanceHandle> instances;

  const int side = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(horde_size))));
  for (int i = 0; i < horde_size; ++i) {
    engine::AnimStateMachine mind;
    mind.add_state({"move", clips[0], true});
    mind.start("move");
    // Stagger the phase: identical poses would hide slice aliasing entirely.
    for (int step = 0; step < i % 37; ++step)
      mind.tick(1.0F / 60.0F, scene.animations());

    glm::mat4 xform(1.0F);
    xform[3] = glm::vec4(static_cast<float>(i % side) * 2.0F - static_cast<float>(side),
                         0.0F,
                         static_cast<float>(i / side) * 2.0F - static_cast<float>(side),
                         1.0F);

    const engine::InstanceHandle handle = scene.add_instance({
        .mesh_index = mesh,
        .texture_index = texture, // deliberately shared across the whole horde
        .model = xform,
        .skin_index = skin,
    });
    scene.instance(handle).pose_layers = mind.shared_layers();
    minds.push_back(std::move(mind));
    instances.push_back(handle);
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
    for (engine::AnimStateMachine &mind : minds)
      mind.tick(1.0F / 60.0F, scene.animations());

    // Churn the horde: despawn and respawn characters mid-list so the bone slot
    // assignment shifts under the renderer every frame. This is what used to
    // reallocate every buffer and descriptor set behind a stall, and what
    // shifted every later instance onto the wrong skeleton.
    if (frame % 30 == 15 && instances.size() > 8) {
      scene.remove_instance(instances[instances.size() / 2]);
      ++despawned;
    }
    if (frame % 30 == 25) {
      engine::AnimStateMachine mind;
      mind.add_state({"move", clips[0], true});
      mind.start("move");
      const engine::InstanceHandle handle = scene.add_instance({
          .mesh_index = mesh,
          .texture_index = texture,
          .skin_index = skin,
      });
      scene.instance(handle).pose_layers = mind.shared_layers();
      minds.push_back(std::move(mind));
      instances.push_back(handle);
      ++respawned;
    }

    runtime.vulkan().sync_scene(scene);
    runtime.vulkan().draw_frame(scene);
  }

  const engine::RenderStats stats = runtime.vulkan().render_stats();

  // Batching these into one draw call is only correct if each character still
  // reads its own slice of the shared bone palette. The characters were started
  // at staggered animation phases, so their poses must differ -- if the bone
  // base were wrong they would all show the first instance's pose, which renders
  // perfectly happily and looks like a horde marching in lockstep.
  std::size_t compared = 0;
  std::size_t distinct = 0;
  const engine::MeshInstance *reference = nullptr;
  for (const engine::InstanceHandle handle : instances) {
    const engine::MeshInstance *instance = scene.try_instance(handle);
    if (instance == nullptr || instance->cached_bone_worlds.empty())
      continue;
    if (reference == nullptr) {
      reference = instance;
      continue;
    }
    ++compared;
    if (instance->cached_bone_worlds != reference->cached_bone_worlds)
      ++distinct;
  }

  runtime.shutdown();

  std::printf("frames=%d horde=%d despawned=%zu respawned=%zu\n",
              frames, horde_size, despawned, respawned);
  std::printf("last frame, main pass: %u instances submitted in %u draw calls (%u culled)\n",
              stats.draws_submitted, stats.batches_submitted, stats.draws_culled);
  std::printf("last frame, shadow:    %u instances submitted in %u draw calls\n",
              stats.shadow_draws_submitted, stats.shadow_batches_submitted);
  if (stats.batches_submitted > 0) {
    std::printf("instances per draw call: %.1f\n",
                static_cast<double>(stats.draws_submitted) /
                    static_cast<double>(stats.batches_submitted));
  }

  std::printf("distinct poses: %zu of %zu characters differ from the first\n", distinct, compared);

  if (stats.main_pass_total() == 0) {
    std::printf("FAIL: nothing was drawn\n");
    return EXIT_FAILURE;
  }
  if (compared == 0 || distinct * 2 < compared) {
    std::printf("FAIL: the horde is posed in lockstep -- characters are not reading\n"
                "      their own slice of the bone palette.\n");
    return EXIT_FAILURE;
  }
  std::printf("ok\n");
  return EXIT_SUCCESS;
}
