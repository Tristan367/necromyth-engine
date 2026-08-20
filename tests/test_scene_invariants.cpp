// CPU-side engine invariants — no Vulkan device required.
//
// These cover the contracts that used to be enforced only by four hand-written
// predicates agreeing with each other, and that failed silently (wrong pose on
// the wrong model, a light that would not turn off) rather than crashing.

#include "renderer/draw_list.hpp"
#include "scene/scene.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string &what) {
  if (condition) {
    std::printf("  ok   %s\n", what.c_str());
    return;
  }
  std::printf("  FAIL %s\n", what.c_str());
  ++g_failures;
}

[[nodiscard]] auto make_skeleton(std::uint32_t bone_count) -> engine::SkeletonAsset {
  engine::SkeletonAsset skeleton;
  skeleton.joint_nodes.resize(bone_count);
  skeleton.inverse_bind_matrices.resize(bone_count, glm::mat4(1.0F));
  skeleton.node_parents.resize(bone_count, 0U);
  return skeleton;
}

[[nodiscard]] auto make_instance(std::uint32_t skin_index) -> engine::MeshInstance {
  engine::MeshInstance instance;
  instance.mesh_index = 0;
  instance.texture_index = 0; // deliberately shared: see the bone-slot test below
  instance.skin_index = skin_index;
  return instance;
}

// The bone-slot invariant.
//
// Bone buffers, skinned descriptor sets, the per-frame joint-matrix upload and
// DrawCommand::bone_instance_index are four separate sequential walks over the
// same instance list. They agree only if every one of them advances on exactly
// instance_uses_skinning(). When they disagreed, instances silently rendered
// with another model's pose.
//
// build_draw_list is the only one of the four reachable without a GPU, so this
// pins it to an independently computed expectation.
void test_bone_slot_indices_are_dense_and_ordered() {
  std::printf("bone slot indices\n");

  engine::Scene scene;
  (void)scene.add_mesh(engine::MeshSource{});
  const std::uint32_t skin_a = scene.add_skeleton(make_skeleton(4));
  const std::uint32_t skin_b = scene.add_skeleton(make_skeleton(7));
  const std::uint32_t skin_empty = scene.add_skeleton(make_skeleton(0));

  std::vector<engine::PoseLayer> layers(1);

  // A deliberately awkward mix: skinned and unskinned, alive and dead, with and
  // without a pose stack, plus a skeleton that carries no joints at all.
  const std::uint32_t inst_skinned_0 = scene.add_instance(make_instance(skin_a));
  (void)scene.add_instance(make_instance(engine::k_invalid_skin_index)); // static
  const std::uint32_t inst_no_layers = scene.add_instance(make_instance(skin_b));
  const std::uint32_t inst_dead = scene.add_instance(make_instance(skin_a));
  (void)scene.add_instance(make_instance(skin_empty));                   // jointless
  const std::uint32_t inst_skinned_last = scene.add_instance(make_instance(skin_b));
  (void)scene.add_instance(make_instance(engine::k_invalid_skin_index)); // static

  // Only some skinned instances get a pose stack. The ones without must still
  // own a slot — skipping them was the original off-by-N.
  scene.instance(inst_skinned_0).pose_layers = &layers;
  scene.instance(inst_skinned_last).pose_layers = &layers;
  scene.instance(inst_no_layers).pose_layers = nullptr;
  scene.remove_instance(inst_dead);

  std::vector<engine::DrawCommand> draws;
  engine::build_draw_list(scene, draws);

  // Expectation computed independently of build_draw_list.
  std::uint32_t expected_slots = 0;
  for (const engine::MeshInstance &instance : scene.instances())
    if (engine::instance_uses_skinning(instance, scene))
      ++expected_slots;

  check(expected_slots == 3,
        "3 instances own a bone slot (2 with a pose stack, 1 without)");

  std::vector<std::uint32_t> slots;
  for (const engine::DrawCommand &draw : draws)
    if (draw.bone_instance_index != engine::k_invalid_skin_index)
      slots.push_back(draw.bone_instance_index);

  check(slots.size() == expected_slots,
        "draw list emits exactly one bone slot per skinned instance");

  // Dense 0..N-1 with no gaps and no duplicates. A gap means some walker skipped
  // an instance the others counted; a duplicate means two instances share a
  // bone buffer.
  std::vector<bool> seen(expected_slots, false);
  bool dense = slots.size() == expected_slots;
  for (const std::uint32_t slot : slots) {
    if (slot >= expected_slots || seen[slot]) {
      dense = false;
      break;
    }
    seen[slot] = true;
  }
  check(dense, "bone slots are a dense 0..N-1 permutation (no gaps, no aliases)");

  // The dead instance must not be drawn at all.
  bool dead_drawn = false;
  for (const engine::DrawCommand &draw : draws)
    if (draw.skin_index == skin_a && draw.bone_instance_index == engine::k_invalid_skin_index)
      dead_drawn = true;
  check(!dead_drawn, "removed instance contributes no draw");

  // A skeleton with zero joints is not skinned geometry — it must not consume a
  // slot, or every later instance shifts onto the wrong bone buffer.
  bool jointless_has_slot = false;
  for (const engine::DrawCommand &draw : draws)
    if (draw.skin_index == skin_empty && draw.bone_instance_index != engine::k_invalid_skin_index)
      jointless_has_slot = true;
  check(!jointless_has_slot, "jointless skeleton consumes no bone slot");

  (void)inst_skinned_0;
}

// Same-texture skinned instances are adjacent after the draw-list sort, which is
// what let a descriptor-set cache keyed only on the texture collapse them onto
// one bone buffer. The sort must still order them by bone slot so the recorder
// can tell them apart.
void test_same_texture_skinned_instances_stay_distinct() {
  std::printf("same-texture skinned instances\n");

  engine::Scene scene;
  (void)scene.add_mesh(engine::MeshSource{});
  const std::uint32_t skin = scene.add_skeleton(make_skeleton(5));
  std::vector<engine::PoseLayer> layers(1);

  constexpr std::uint32_t k_horde = 6;
  for (std::uint32_t i = 0; i < k_horde; ++i) {
    const std::uint32_t index = scene.add_instance(make_instance(skin));
    scene.instance(index).pose_layers = &layers; // all share texture_index 0
  }

  std::vector<engine::DrawCommand> draws;
  engine::build_draw_list(scene, draws);

  check(draws.size() == k_horde, "every instance of the horde is drawn");

  std::vector<std::uint32_t> slots;
  for (const engine::DrawCommand &draw : draws)
    slots.push_back(draw.bone_instance_index);

  bool all_distinct = true;
  for (std::size_t i = 0; i < slots.size(); ++i) {
    if (slots[i] != static_cast<std::uint32_t>(i))
      all_distinct = false;
  }
  check(all_distinct,
        "identical-texture instances keep distinct, ordered bone slots");
}

// A default-constructed PointLight is a white, intensity-1, range-5 lamp at the
// world origin, so "remove" used to leave a glowing orb at (0,0,0).
void test_removed_lights_stop_emitting() {
  std::printf("light removal\n");

  engine::Scene scene;
  engine::PointLight point;
  point.position = {3.0F, 4.0F, 5.0F};
  point.intensity = 2.0F;
  point.casts_shadow = true;
  const std::uint32_t point_index = scene.add_point_light(point);

  engine::SpotLight spot;
  spot.position = {1.0F, 2.0F, 3.0F};
  spot.intensity = 3.0F;
  spot.casts_shadow = true;
  const std::uint32_t spot_index = scene.add_spot_light(spot);

  scene.remove_point_light(point_index);
  scene.remove_spot_light(spot_index);

  const engine::PointLight &p = scene.point_lights()[point_index];
  check(p.intensity == 0.0F && p.range == 0.0F && !p.casts_shadow,
        "removed point light emits nothing and casts no shadow");

  const engine::SpotLight &s = scene.spot_lights()[spot_index];
  check(s.intensity == 0.0F && s.range == 0.0F && !s.casts_shadow,
        "removed spot light emits nothing and casts no shadow");

  // Indices must stay stable: a shadow-casting point light's cubemap layer IS
  // its array index, so removal may not compact the vector.
  check(scene.point_lights().size() == 1 && scene.spot_lights().size() == 1,
        "light removal preserves indices of remaining lights");
}

// Background geometry must never be handed a skinned pipeline, and shadow
// casters must be a subset of the main draw list.
void test_draw_list_layer_and_shadow_partition() {
  std::printf("draw list partitioning\n");

  engine::Scene scene;
  (void)scene.add_mesh(engine::MeshSource{});

  engine::MeshInstance sky = make_instance(engine::k_invalid_skin_index);
  sky.layer = engine::RenderLayer::Background;
  (void)scene.add_instance(sky);
  (void)scene.add_instance(make_instance(engine::k_invalid_skin_index));

  std::vector<engine::DrawCommand> draws;
  engine::build_draw_list(scene, draws);

  std::vector<engine::DrawCommand> shadow_draws;
  engine::build_shadow_draw_list(draws, shadow_draws);

  bool background_is_background_pipeline = true;
  for (const engine::DrawCommand &draw : draws)
    if (draw.layer == engine::RenderLayer::Background &&
        draw.pipeline != engine::PipelineId::Background)
      background_is_background_pipeline = false;
  check(background_is_background_pipeline,
        "background layer always uses the background pipeline");

  check(shadow_draws.size() <= draws.size(),
        "shadow casters are a subset of the main draw list");

  bool no_background_shadows = true;
  for (const engine::DrawCommand &draw : shadow_draws)
    if (draw.pipeline == engine::PipelineId::Background)
      no_background_shadows = false;
  check(no_background_shadows, "background geometry casts no shadow");
}

} // namespace

auto main() -> int {
  test_bone_slot_indices_are_dense_and_ordered();
  test_same_texture_skinned_instances_stay_distinct();
  test_removed_lights_stop_emitting();
  test_draw_list_layer_and_shadow_partition();

  if (g_failures != 0) {
    std::printf("\n%d check(s) failed\n", g_failures);
    return EXIT_FAILURE;
  }
  std::printf("\nall checks passed\n");
  return EXIT_SUCCESS;
}
