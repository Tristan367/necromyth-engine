// CPU-side engine invariants — no Vulkan device required.
//
// These cover the contracts that used to be enforced only by four hand-written
// predicates agreeing with each other, and that failed silently (wrong pose on
// the wrong model, a light that would not turn off) rather than crashing.

#include "renderer/bone_buffer.hpp"
#include "renderer/deferred_delete.hpp"
#include "renderer/draw_list.hpp"
#include "renderer/frustum.hpp"
#include "scene/camera.hpp"
#include "scene/animation_state_machine.hpp"
#include "scene/animation_utils.hpp"
#include "scene/shadow_assignment.hpp"
#include "scene/shadow_utils.hpp"
#include "scene/scene.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
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

  const auto layers = std::make_shared<std::vector<engine::PoseLayer>>(1);

  // A deliberately awkward mix: skinned and unskinned, alive and dead, with and
  // without a pose stack, plus a skeleton that carries no joints at all.
  const engine::InstanceHandle inst_skinned_0 = scene.add_instance(make_instance(skin_a));
  (void)scene.add_instance(make_instance(engine::k_invalid_skin_index)); // static
  const engine::InstanceHandle inst_no_layers = scene.add_instance(make_instance(skin_b));
  const engine::InstanceHandle inst_dead = scene.add_instance(make_instance(skin_a));
  (void)scene.add_instance(make_instance(skin_empty));                   // jointless
  const engine::InstanceHandle inst_skinned_last = scene.add_instance(make_instance(skin_b));
  (void)scene.add_instance(make_instance(engine::k_invalid_skin_index)); // static

  // Only some skinned instances get a pose stack. The ones without must still
  // own a slot — skipping them was the original off-by-N.
  scene.instance(inst_skinned_0).pose_layers = layers;
  scene.instance(inst_skinned_last).pose_layers = layers;
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
  const auto layers = std::make_shared<std::vector<engine::PoseLayer>>(1);

  constexpr std::uint32_t k_horde = 6;
  for (std::uint32_t i = 0; i < k_horde; ++i) {
    const engine::InstanceHandle handle = scene.add_instance(make_instance(skin));
    scene.instance(handle).pose_layers = layers; // all share texture_index 0
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


// Frustum culling. A wrong near-plane convention (the OpenGL [-1,1] form on a
// Vulkan [0,1] projection) culls geometry directly in front of the camera, and a
// sign error culls everything or nothing -- both look like "the world vanished".
void test_frustum_culls_behind_not_in_front() {
  std::printf("frustum culling\n");

  engine::Camera camera;
  camera.set_aspect(16.0F / 9.0F);
  camera.look_at({0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, -1.0F});

  const engine::Frustum frustum =
      engine::Frustum::from_view_proj(camera.view_projection_matrix());

  check(frustum.intersects_sphere({0.0F, 0.0F, -10.0F}, 1.0F),
        "sphere straight ahead is visible");
  check(!frustum.intersects_sphere({0.0F, 0.0F, 10.0F}, 1.0F),
        "sphere directly behind the camera is culled");
  check(frustum.intersects_sphere({0.0F, 0.0F, 1.0F}, 5.0F),
        "large sphere straddling the camera is kept (radius reaches the frustum)");
  check(!frustum.intersects_sphere({10000.0F, 0.0F, -10.0F}, 1.0F),
        "sphere far off to the side is culled");

  // The near plane is the one that regresses when the depth convention is wrong.
  check(frustum.intersects_sphere({0.0F, 0.0F, -0.5F}, 0.4F),
        "sphere just past the near plane survives (Vulkan [0,1] depth convention)");
}

// Culling must never drop geometry that opted out: skinned meshes animate
// outside their bind-pose bounds, and background geometry ignores camera
// translation entirely.
void test_cull_opt_out_is_respected() {
  std::printf("cull opt-out\n");

  engine::Scene scene;
  (void)scene.add_mesh(engine::MeshSource{});
  const std::uint32_t skin = scene.add_skeleton(make_skeleton(3));
  const auto layers = std::make_shared<std::vector<engine::PoseLayer>>(1);

  engine::MeshInstance sky = make_instance(engine::k_invalid_skin_index);
  sky.layer = engine::RenderLayer::Background;
  (void)scene.add_instance(sky);

  const engine::InstanceHandle skinned = scene.add_instance(make_instance(skin));
  scene.instance(skinned).pose_layers = layers;

  std::vector<engine::DrawCommand> draws;
  engine::build_draw_list(scene, draws);

  bool all_opted_out = true;
  for (const engine::DrawCommand &draw : draws)
    if (draw.world_bounds.radius > 0.0F)
      all_opted_out = false;
  check(all_opted_out,
        "skinned and background draws carry radius 0 (never culled)");
}


[[nodiscard]] auto make_mesh(float extent) -> engine::MeshSource {
  engine::MeshSource mesh;
  for (int i = 0; i < 2; ++i) {
    engine::MeshVertex v{};
    const float sign = i == 0 ? -1.0F : 1.0F;
    v.pos[0] = sign * extent;
    v.pos[1] = sign * extent;
    v.pos[2] = sign * extent;
    mesh.vertices.push_back(v);
  }
  mesh.indices = {0, 1, 0};
  return mesh;
}

// Mesh slot lifecycle. A streaming world creates, remeshes and drops geometry
// continuously, so slots must be reusable and indices must stay stable -- an
// instance holds a slot index, not a position.
void test_mesh_slot_lifecycle() {
  std::printf("mesh slot lifecycle\n");

  engine::Scene scene;
  const std::uint32_t a = scene.add_mesh(make_mesh(1.0F));
  const std::uint32_t b = scene.add_mesh(make_mesh(2.0F));

  check(a != b, "distinct meshes get distinct slots");
  check(scene.live_mesh_count() == 2, "two live meshes");
  check(scene.mesh_alive(a) && scene.mesh_alive(b), "both slots alive");

  const std::uint32_t rev_before = scene.meshes()[a].revision;
  scene.update_mesh(a, make_mesh(8.0F));
  check(scene.meshes()[a].revision != rev_before,
        "update_mesh bumps the revision so the renderer re-uploads");
  check(scene.mesh_bounds(a).max.x == 8.0F, "update_mesh recomputes bounds");

  scene.remove_mesh(a);
  check(!scene.mesh_alive(a), "removed slot is not alive");
  check(scene.live_mesh_count() == 1, "live count drops");
  check(scene.mesh_bounds(a).empty(), "removed slot has no bounds");

  const std::uint32_t recycled = scene.add_mesh(make_mesh(3.0F));
  check(recycled == a, "a freed slot is reused instead of growing the vector");
  check(scene.mesh_count() == 2, "slot capacity did not grow");
  check(scene.mesh_alive(recycled), "recycled slot is alive again");

  // Removing twice must not push the slot onto the free list twice, or two
  // different meshes would later be handed the same index.
  scene.remove_mesh(recycled);
  scene.remove_mesh(recycled);
  const std::uint32_t x = scene.add_mesh(make_mesh(1.0F));
  const std::uint32_t y = scene.add_mesh(make_mesh(1.0F));
  check(x != y, "double remove does not hand the same slot out twice");
}

// An instance left pointing at a freed slot must draw nothing: the slot may
// already hold different geometry, or none.
void test_instances_of_removed_meshes_do_not_draw() {
  std::printf("instances of removed meshes\n");

  engine::Scene scene;
  const std::uint32_t mesh = scene.add_mesh(make_mesh(1.0F));
  engine::MeshInstance instance = make_instance(engine::k_invalid_skin_index);
  instance.mesh_index = mesh;
  (void)scene.add_instance(instance);

  std::vector<engine::DrawCommand> draws;
  engine::build_draw_list(scene, draws);
  check(draws.size() == 1, "instance draws while its mesh is alive");

  scene.remove_mesh(mesh);
  engine::build_draw_list(scene, draws);
  check(draws.empty(), "instance draws nothing once its mesh slot is freed");
}

// Deferred deletion timing. Freeing too early lets the GPU read destroyed
// buffers; never freeing is a leak. Both are silent.
void test_deferred_delete_holds_for_frames_in_flight() {
  std::printf("deferred delete\n");

  constexpr std::uint32_t k_frames_in_flight = 2;
  engine::DeferredDelete<int> queue;

  queue.retire(1, /*current_frame=*/10);
  check(queue.pending_count() == 1, "retired resource is held");

  queue.collect(10, k_frames_in_flight);
  check(queue.pending_count() == 1, "not freed on the frame it was retired");

  queue.collect(12, k_frames_in_flight);
  check(queue.pending_count() == 1,
        "still held while a frame in flight could reference it");

  queue.collect(13, k_frames_in_flight);
  check(queue.pending_count() == 0,
        "freed once every in-flight frame has cycled past");

  // Early frames must not underflow the frame arithmetic into freeing at once.
  engine::DeferredDelete<int> early;
  early.retire(1, 0);
  early.collect(1, k_frames_in_flight);
  check(early.pending_count() == 1, "no underflow during the first frames");
}


// Bone slice addressing. Every skinned instance reads its matrices out of one
// shared buffer at a dynamic offset, so two slots colliding means two characters
// share a pose -- silently, with no validation error. Alignment matters too: a
// dynamic offset that is not a multiple of minStorageBufferOffsetAlignment is
// invalid usage.
void test_bone_slot_offsets_are_distinct_and_aligned() {
  std::printf("bone slice addressing\n");

  constexpr std::uint32_t k_capacity = 64;
  constexpr std::uint32_t k_frames = 2;

  std::vector<std::uint32_t> bases;
  for (std::uint32_t frame = 0; frame < k_frames; ++frame)
    for (std::uint32_t slot = 0; slot < k_capacity; ++slot)
      bases.push_back(engine::bone_slot_matrix_base(frame, slot, k_capacity));

  std::ranges::sort(bases);
  check(std::ranges::adjacent_find(bases) == bases.end(),
        "every (frame, slot) pair maps to a distinct matrix base");

  // Slices must not overlap: consecutive bases have to differ by a full palette,
  // or one character's matrices spill into the next character's.
  bool no_overlap = true;
  for (std::size_t i = 1; i < bases.size(); ++i)
    if (bases[i] - bases[i - 1] < engine::k_bone_slot_matrices)
      no_overlap = false;
  check(no_overlap, "slices are a whole palette apart, so they cannot overlap");

  check(engine::k_bone_slot_matrices >= engine::k_max_bones,
        "a slice holds a full bone palette");

  // The last slice must end exactly at the end of the buffer.
  const std::uint32_t total = k_frames * k_capacity * engine::k_bone_slot_matrices;
  const std::uint32_t last = engine::bone_slot_matrix_base(k_frames - 1, k_capacity - 1, k_capacity);
  check(last + engine::k_bone_slot_matrices == total,
        "the last slice ends exactly at the end of the palette");

  // Frames must not overlap: a slot's frame-0 and frame-1 slices are what make
  // double buffering safe while a frame is still in flight.
  check(engine::bone_slot_matrix_base(0, 5, k_capacity) !=
            engine::bone_slot_matrix_base(1, 5, k_capacity),
        "the same slot occupies different memory in different frames");
}


// Shadow slot assignment. The cubemap layer a light renders into used to be its
// index in the scene array, which meant a shadow-caster past the capacity was
// skipped by the renderer while the shader kept sampling its nonexistent layer,
// and no light could be skipped for being irrelevant without the shader reading
// whatever was underneath. Slots are now explicit, so they must be dense,
// in-range, and given only to lights that need one.
void test_shadow_slot_assignment() {
  std::printf("shadow slot assignment\n");

  engine::Camera camera;
  camera.set_aspect(16.0F / 9.0F);
  camera.look_at({0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, -1.0F});
  const engine::Frustum frustum =
      engine::Frustum::from_view_proj(camera.view_projection_matrix());

  const auto lamp = [](glm::vec3 position, bool casts) {
    engine::PointLight light;
    light.position = position;
    light.range = 5.0F;
    light.casts_shadow = casts;
    return light;
  };

  std::vector<engine::PointLight> points{
      lamp({0.0F, 0.0F, -10.0F}, true),   // 0: in view, casts    -> slot
      lamp({0.0F, 0.0F, -12.0F}, false),  // 1: in view, no cast   -> none
      lamp({0.0F, 0.0F, 500.0F}, true),   // 2: behind camera      -> none
      lamp({0.0F, 0.0F, -14.0F}, true),   // 3: in view, casts     -> slot
  };
  points[1].casts_shadow = false;

  const std::vector<engine::SpotLight> spots;
  auto assignment = engine::assign_shadow_slots(points, spots, frustum, 8, 8);

  check(assignment.point_count == 2, "only in-view shadow casters get a slot");
  check(assignment.point_slots[0] == 0, "first caster takes slot 0");
  check(assignment.point_slots[1] == engine::k_no_shadow_slot, "non-caster gets no slot");
  check(assignment.point_slots[2] == engine::k_no_shadow_slot,
        "light behind the camera gets no slot");
  // Dense: the caster after a skipped light must take slot 1, not slot 3. Using
  // the scene index here is what wasted cube layers and overran the capacity.
  check(assignment.point_slots[3] == 1, "slots are dense, not the light's scene index");

  // Capacity is in cubes. Every handed-out slot must be addressable.
  auto capped = engine::assign_shadow_slots(points, spots, frustum, 1, 8);
  check(capped.point_count == 1, "assignment stops at capacity");
  bool within_capacity = true;
  for (const std::int32_t slot : capped.point_slots)
    if (slot != engine::k_no_shadow_slot && slot >= 1)
      within_capacity = false;
  check(within_capacity, "no slot is ever handed out past capacity");

  // A light with no reach cannot illuminate anything, shadow or not.
  std::vector<engine::PointLight> dark{lamp({0.0F, 0.0F, -10.0F}, true)};
  dark[0].intensity = 0.0F;
  const auto none = engine::assign_shadow_slots(dark, spots, frustum, 8, 8);
  check(none.point_count == 0, "a zero-intensity light gets no shadow map");
}


// Instance handles. Slot reuse means a bare index can silently refer to a
// different entity after a despawn -- the generation is what makes that
// detectable instead of merely wrong.
void test_instance_handles_detect_reuse() {
  std::printf("instance handles\n");

  engine::Scene scene;
  (void)scene.add_mesh(engine::MeshSource{});

  const engine::InstanceHandle a = scene.add_instance(make_instance(engine::k_invalid_skin_index));
  const engine::InstanceHandle b = scene.add_instance(make_instance(engine::k_invalid_skin_index));
  check(scene.is_valid(a) && scene.is_valid(b), "fresh handles are valid");
  check(!(a == b), "distinct instances get distinct handles");
  check(scene.live_instance_count() == 2, "two live instances");

  scene.remove_instance(a);
  check(!scene.is_valid(a), "a removed handle stops being valid");
  check(scene.try_instance(a) == nullptr, "try_instance returns null for a removed handle");
  check(scene.is_valid(b), "removing one instance does not disturb another");
  check(scene.live_instance_count() == 1, "live count drops");

  // The slot comes back, but as a different entity.
  const engine::InstanceHandle c = scene.add_instance(make_instance(engine::k_invalid_skin_index));
  check(c.index == a.index, "the freed slot is reused rather than growing the vector");
  check(c.generation != a.generation, "reuse bumps the generation");
  check(!scene.is_valid(a), "the stale handle is still rejected after its slot is reused");
  check(scene.is_valid(c), "the new handle for that slot is valid");

  // Double remove must not free the same slot twice, or two live entities would
  // later be handed the same index.
  scene.remove_instance(c);
  scene.remove_instance(c);
  const engine::InstanceHandle d = scene.add_instance(make_instance(engine::k_invalid_skin_index));
  const engine::InstanceHandle e = scene.add_instance(make_instance(engine::k_invalid_skin_index));
  check(d.index != e.index, "double remove does not hand the same slot out twice");

  const engine::InstanceHandle unset;
  check(!scene.is_valid(unset), "a default-constructed handle is never valid");
}

// A bone attachment holds a handle to the object it drives (a weapon in a hand).
// If that object was removed and its slot reused, driving it would move an
// unrelated entity every frame.
void test_bone_attachment_ignores_stale_target() {
  std::printf("bone attachment targets\n");

  engine::Scene scene;
  (void)scene.add_mesh(engine::MeshSource{});
  const std::uint32_t skin = scene.add_skeleton(make_skeleton(4));

  const engine::InstanceHandle weapon = scene.add_instance(make_instance(engine::k_invalid_skin_index));
  const engine::InstanceHandle character = scene.add_instance(make_instance(skin));
  scene.instance(character).cached_bone_worlds.assign(4, glm::mat4(1.0F));
  scene.instance(character).bone_attachments.push_back(
      engine::BoneAttachment{.joint_index = 1, .target_instance = weapon});

  glm::mat4 moved(1.0F);
  moved[3] = glm::vec4(5.0F, 0.0F, 0.0F, 1.0F);
  scene.instance(character).model = moved;

  engine::update_bone_attachments(scene.instances());
  check(scene.instance(weapon).model[3].x == 5.0F, "a live attachment target follows the bone");

  // Remove the weapon and let something else take its slot.
  scene.remove_instance(weapon);
  const engine::InstanceHandle bystander = scene.add_instance(make_instance(engine::k_invalid_skin_index));
  check(bystander.index == weapon.index, "the bystander took the weapon's slot");

  glm::mat4 moved_again(1.0F);
  moved_again[3] = glm::vec4(99.0F, 0.0F, 0.0F, 1.0F);
  scene.instance(character).model = moved_again;
  engine::update_bone_attachments(scene.instances());

  check(scene.instance(bystander).model[3].x == 0.0F,
        "a stale attachment target does not drag an unrelated instance around");
}


// Pose stacks are shared, not pointed at. A raw pointer into an AnimStateMachine
// dangled whenever the game relocated one -- a std::vector of them growing was
// enough -- and the result was every character reading freed memory, silently.
void test_pose_stack_survives_owner_relocation() {
  std::printf("pose stack ownership\n");

  engine::Scene scene;
  (void)scene.add_mesh(engine::MeshSource{});
  const std::uint32_t skin = scene.add_skeleton(make_skeleton(3));

  std::vector<engine::AnimStateMachine> minds;
  minds.reserve(1); // force a reallocation on the second push_back

  engine::AnimStateMachine first;
  first.add_state({"idle", 0, true});
  first.start("idle");
  const auto shared = first.shared_layers();
  minds.push_back(std::move(first));

  const engine::InstanceHandle character = scene.add_instance(make_instance(skin));
  scene.instance(character).pose_layers = minds[0].shared_layers();
  const void *before = scene.instance(character).pose_layers.get();

  // Reallocate the container the state machines live in.
  for (int i = 0; i < 8; ++i) {
    engine::AnimStateMachine extra;
    extra.add_state({"idle", 0, true});
    extra.start("idle");
    minds.push_back(std::move(extra));
  }

  const void *after = scene.instance(character).pose_layers.get();
  check(before == after, "the instance's pose stack did not move with its owner");
  check(minds[0].shared_layers().get() == after,
        "the relocated state machine still drives the same pose stack");
  check(shared.use_count() > 1, "ownership is genuinely shared, not copied");

  // Destroying every state machine must freeze the pose, not free it underneath
  // the renderer.
  minds.clear();
  check(scene.instance(character).pose_layers != nullptr,
        "the pose stack outlives its state machine rather than dangling");
  check(scene.instance(character).pose_layers->size() >= 1,
        "the last pose remains readable after the owner is gone");
}

// Both halves of the spot shadow atlas -- the pass that renders a tile and the
// light buffer that tells the shader where to sample -- used to compute the
// layout independently. They agreed by luck, and they agreed on a layout that
// was wrong in two ways: it divided the atlas by the number of spot lights in
// the SCENE (so a light casting no shadow shrank everyone else's map), and it
// produced strips, when the light's projection is built with aspect 1.0 and
// wants a square.
//
// The layout now lives in one function. This pins its contract, because with
// one spot light every version of it looks identical -- and one spot light is
// all the demo has, so nothing else would notice a regression.
void test_spot_atlas_tiles_are_square_and_disjoint() {
  constexpr std::uint32_t k_slots = engine::k_max_spot_shadow_lights;
  const std::uint32_t side = engine::spot_atlas_grid_side(k_slots);
  check(side * side >= k_slots, "grid is big enough for every slot");

  std::vector<engine::SpotAtlasTile> tiles;
  for (std::uint32_t slot = 0; slot < k_slots; ++slot)
    tiles.push_back(engine::spot_atlas_tile(slot, k_slots));

  bool square = true;
  bool inside = true;
  for (const engine::SpotAtlasTile &tile : tiles) {
    if (tile.width != tile.height)
      square = false;
    if (tile.u < 0.0F || tile.v < 0.0F || tile.u + tile.width > 1.0F + 1e-6F ||
        tile.v + tile.height > 1.0F + 1e-6F)
      inside = false;
  }
  check(square, "every tile is square, matching the light's aspect-1.0 projection");
  check(inside, "every tile lies inside the atlas");

  // The one that actually matters: two lights must never share texels, or each
  // one's shadow bleeds into the other's lookup.
  bool disjoint = true;
  for (std::size_t a = 0; a < tiles.size(); ++a) {
    for (std::size_t b = a + 1; b < tiles.size(); ++b) {
      const bool separated_x = tiles[a].u + tiles[a].width <= tiles[b].u + 1e-6F ||
                               tiles[b].u + tiles[b].width <= tiles[a].u + 1e-6F;
      const bool separated_y = tiles[a].v + tiles[a].height <= tiles[b].v + 1e-6F ||
                               tiles[b].v + tiles[b].height <= tiles[a].v + 1e-6F;
      if (!separated_x && !separated_y)
        disjoint = false;
    }
  }
  check(disjoint, "no two slots overlap in the atlas");

  // Tile size must not depend on how many lights the scene happens to hold --
  // that dependency was the original bug.
  const engine::SpotAtlasTile first = engine::spot_atlas_tile(0, k_slots);
  check(first.width == 1.0F / static_cast<float>(side),
        "tile size is fixed by the slot count, not by the scene's light count");
}

} // namespace

auto main() -> int {
  test_bone_slot_indices_are_dense_and_ordered();
  test_same_texture_skinned_instances_stay_distinct();
  test_removed_lights_stop_emitting();
  test_draw_list_layer_and_shadow_partition();
  test_frustum_culls_behind_not_in_front();
  test_cull_opt_out_is_respected();
  test_mesh_slot_lifecycle();
  test_instances_of_removed_meshes_do_not_draw();
  test_deferred_delete_holds_for_frames_in_flight();
  test_bone_slot_offsets_are_distinct_and_aligned();
  test_shadow_slot_assignment();
  test_spot_atlas_tiles_are_square_and_disjoint();
  test_instance_handles_detect_reuse();
  test_bone_attachment_ignores_stale_target();
  test_pose_stack_survives_owner_relocation();

  if (g_failures != 0) {
    std::printf("\n%d check(s) failed\n", g_failures);
    return EXIT_FAILURE;
  }
  std::printf("\nall checks passed\n");
  return EXIT_SUCCESS;
}
