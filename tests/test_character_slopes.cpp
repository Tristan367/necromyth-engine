// Character controller: slopes and steps.
//
// A voxel world's smallest feature is one metre, and its terrain is marching
// cubes, so a cell that rises a full voxel across its own width is a 45 degree
// ramp. Walking up one has to just work. The settings that decide this are easy
// to get wrong in a way that never crashes and never shows up in a screenshot:
// the character simply stops against a hill and stands there.
//
// The speed rule this pins down is deliberate rather than realistic. A
// controller left to itself projects velocity onto the ground plane, so
// climbing costs ground speed. Enemies here path over the terrain at a flat
// rate, so a player who slows down on a hill is being chased uphill by
// something that is not. Horizontal speed is therefore constant on any slope
// the character can climb at all, and what stops you is the slope limit.

#include "physics/physics_world.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string &what) {
  std::printf("  %s %s\n", condition ? "ok  " : "FAIL", what.c_str());
  if (!condition)
    ++g_failures;
}

constexpr float k_dt = 1.0F / 60.0F;
constexpr float k_speed = 4.0F;
// The ceiling on the constant-speed compensation, matching the game's
// NM_SLOPE_CAP default. 1/cos^2 is 2 at 45 degrees and 4 at the controller's
// 60 degree limit, and a 4x boost turns a walk into a launch off the next
// crest. Godot's version of this rule cannot run away, because it tops the
// motion up by the distance actually lost and that is bounded by the distance
// asked for.
constexpr float k_slope_cap = 2.0F;
// The game's own numbers, from nm::game::Player. Kept here rather than included
// because the engine does not depend on the game -- and kept in step for the
// same reason the controller settings are: a harness measuring different
// numbers from the ones shipped is measuring a controller nobody plays.
constexpr float k_sprint = 7.5F;
constexpr float k_accelerate = 12.0F;
constexpr float k_decelerate = 26.0F;

// Overrides, so the knobs can be swept in seconds against a deterministic
// surface instead of over a town whose save drifts between runs.
[[nodiscard]] auto tuned(const char *name, float fallback) -> float {
  const char *value = std::getenv(name);
  return value == nullptr ? fallback : std::strtof(value, nullptr);
}

// A ground plane that ramps upward at `rise` metres per metre along +X, built
// as a trimesh the way terrain colliders are.
auto make_ramp(float rise, float length = 60.0F, float width = 24.0F) -> engine::MeshSource {
  engine::MeshSource mesh;
  const auto height_at = [rise](float x) { return x <= 0.0F ? 0.0F : x * rise; };

  const int steps = static_cast<int>(length);
  for (int i = 0; i <= steps; ++i) {
    const auto x = static_cast<float>(i) - 8.0F;
    for (int side = 0; side < 2; ++side) {
      const float z = side == 0 ? -width * 0.5F : width * 0.5F;
      engine::MeshVertex v{};
      v.pos[0] = x;
      v.pos[1] = height_at(x);
      v.pos[2] = z;
      mesh.vertices.push_back(v);
    }
  }
  for (int i = 0; i < steps; ++i) {
    const auto base = static_cast<std::uint32_t>(i * 2);
    mesh.indices.insert(mesh.indices.end(),
                        {base, base + 1, base + 2, base + 1, base + 3, base + 2});
  }
  return mesh;
}

struct Walk {
  float distance{0.0F};  // horizontal ground covered
  float climbed{0.0F};   // net height gained
  bool stalled{false};   // stopped making progress while still trying to move
  // Vertical motion in a single frame that the character's OWN velocity does
  // not account for: the stair sweep lifting it, the floor snap pulling it
  // down. Both happen instantaneously, which is what reads as teleporting up a
  // step and as jitter on uneven ground.
  float worst_teleport{0.0F};
  int teleports{0};
};

// Drives the character along +X for `seconds`, using the same rule the game
// uses: velocity is horizontal, full stop, and gravity only applies in the air.
//
// This used to project the velocity onto the ground normal and rescale it to
// hold horizontal speed -- which is what the game did once, and stopped doing,
// because rescaling scales the VERTICAL part too and turns a steep contact into
// a launch. The harness kept the old rule and its comment kept claiming it was
// the game's, so it was measuring a controller nobody plays: walking into a
// 20 cm step flung it upward at 5 m/s and over a metre into the air. Any number
// this file produced about steps was about that, not about the game.
auto walk_along(engine::physics::PhysicsWorld &world, engine::physics::Character &character,
                float seconds) -> Walk {
  const glm::vec3 start = character.position();
  glm::vec3 previous = start;
  int stalled_frames = 0;
  float worst_teleport = 0.0F;
  int teleports = 0;

  const int frames = static_cast<int>(seconds / k_dt);
  for (int frame = 0; frame < frames; ++frame) {
    const glm::vec3 ground_normal = character.ground_normal();
    glm::vec3 velocity{k_speed, 0.0F, 0.0F};
    if (character.is_on_ground()) {
      // The game's constant-speed rule. Gated on walkable ground, and it
      // scales a horizontal vector rather than rotating one, so it cannot
      // launch. Capped, because 1/cos^2 runs away to 4 at the slope limit --
      // see k_slope_cap.
      const float up = ground_normal.y;
      if (up > 0.5F)
        velocity *= std::min(1.0F / (up * up), tuned("SLOPECAP", k_slope_cap));
    } else {
      velocity.y = std::max(character.linear_velocity().y - 9.81F * k_dt, -50.0F);
    }

    character.set_velocity(velocity);
    // Godot's floor-snap rule, which is the game's: you were on the ground and
    // you are not trying to leave it, so hold on to it -- by a tenth of a
    // metre, not by more. Same for the stair sweep. Keeping this in step with
    // main.cpp is the whole value of the harness; it drifted once already and
    // spent a session measuring a controller nobody plays.
    const bool hold_ground = character.is_on_ground() && velocity.y <= 0.0F;
    character.update(k_dt, hold_ground ? tuned("SNAP", engine::physics::Character::k_floor_snap) : 0.0F,
                     hold_ground ? tuned("STEP", engine::physics::Character::k_step_height) : 0.0F);
    world.step(k_dt);

    const glm::vec3 now = character.position();
    const float moved = std::hypot(now.x - previous.x, now.z - previous.z);
    stalled_frames = moved < k_speed * k_dt * 0.25F ? stalled_frames + 1 : 0;
    if (std::getenv("TRACE_Y") != nullptr)
      std::printf("    f%-3d y=%.4f  vy=%+.3f  dy=%+.4f  vy*dt=%+.4f  ground=%d\n", frame,
                  static_cast<double>(now.y), static_cast<double>(velocity.y),
                  static_cast<double>(now.y - previous.y),
                  static_cast<double>(velocity.y * k_dt),
                  static_cast<int>(character.is_on_ground()));
    // How far the character rose that the GROUND does not account for.
    //
    // Comparing against the character's own vertical velocity is the wrong
    // question here, because the velocity is horizontal by design and the
    // capsule does all the climbing -- so on any slope at all, every
    // centimetre of the climb scores as a teleport. Comparing against the
    // surface does the right thing: given the horizontal step and the slope
    // underfoot, the ground says exactly how far up it should have gone.
    const glm::vec3 step = now - previous;
    float teleport = 0.0F;
    if (ground_normal.y > 0.05F) {
      const float expected =
          -(ground_normal.x * step.x + ground_normal.z * step.z) / ground_normal.y;
      teleport = std::abs(step.y - expected);
    }
    if (teleport > 0.002F) {
      ++teleports;
      worst_teleport = std::max(worst_teleport, teleport);
    }
    previous = now;
  }

  const glm::vec3 end = character.position();
  return {std::hypot(end.x - start.x, end.z - start.z), end.y - start.y, stalled_frames > 30,
          worst_teleport, teleports};
}

// Drops a character onto the ramp at x = -4 (the flat part) and lets it settle.
auto spawn_on(engine::physics::PhysicsWorld &world, const engine::MeshSource &ground)
    -> std::unique_ptr<engine::physics::Character> {
  world.create_static_mesh(ground, glm::vec3(0.0F));
  auto character = std::make_unique<engine::physics::Character>(world, glm::vec3(-4.0F, 2.0F, 0.0F));
  character->set_penetration_recovery(tuned("RECOVERY", character->penetration_recovery()));
  for (int i = 0; i < 120; ++i) {
    character->set_velocity({0.0F, std::min(character->linear_velocity().y - 9.81F * k_dt, 0.0F),
                             0.0F});
    character->update(k_dt);
    world.step(k_dt);
  }
  return character;
}


// A flat floor with a single one-metre step in it at x = 0: the wall the
// character walks into.
auto make_step(float length = 60.0F, float width = 24.0F, float rise = 1.0F)
    -> engine::MeshSource {
  engine::MeshSource mesh;
  const auto quad = [&mesh](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d) {
    const auto base = static_cast<std::uint32_t>(mesh.vertices.size());
    for (const glm::vec3 &p : {a, b, c, d}) {
      engine::MeshVertex v{};
      v.pos[0] = p.x;
      v.pos[1] = p.y;
      v.pos[2] = p.z;
      mesh.vertices.push_back(v);
    }
    mesh.indices.insert(mesh.indices.end(),
                        {base, base + 1, base + 2, base, base + 2, base + 3});
  };
  const float h = width * 0.5F;
  // Lower floor, the riser, and the upper floor.
  quad({-length, 0, -h}, {0, 0, -h}, {0, 0, h}, {-length, 0, h});
  quad({0, 0, -h}, {0, rise, -h}, {0, rise, h}, {0, 0, h});
  quad({0, rise, -h}, {length, rise, -h}, {length, rise, h}, {0, rise, h});
  return mesh;
}

// A flat floor with a ceiling `headroom` metres above it.
auto make_room(float headroom, float size = 40.0F) -> engine::MeshSource {
  engine::MeshSource mesh;
  const auto quad = [&mesh](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d) {
    const auto base = static_cast<std::uint32_t>(mesh.vertices.size());
    for (const glm::vec3 &p : {a, b, c, d}) {
      engine::MeshVertex v{};
      v.pos[0] = p.x;
      v.pos[1] = p.y;
      v.pos[2] = p.z;
      mesh.vertices.push_back(v);
    }
    mesh.indices.insert(mesh.indices.end(),
                        {base, base + 1, base + 2, base, base + 2, base + 3});
  };
  const float h = size * 0.5F;
  quad({-h, 0, -h}, {h, 0, -h}, {h, 0, h}, {-h, 0, h});
  quad({-h, headroom, -h}, {h, headroom, -h}, {h, headroom, h}, {-h, headroom, h});
  return mesh;
}

// Drops a character at the origin and lets it settle.
auto spawn_at(engine::physics::PhysicsWorld &world, const engine::MeshSource &ground,
              const glm::vec3 &at) -> std::unique_ptr<engine::physics::Character> {
  world.create_static_mesh(ground, glm::vec3(0.0F));
  auto character = std::make_unique<engine::physics::Character>(world, at);
  for (int i = 0; i < 120; ++i) {
    character->set_velocity(
        {0.0F, std::min(character->linear_velocity().y - 9.81F * k_dt, 0.0F), 0.0F});
    character->update(k_dt);
    world.step(k_dt);
  }
  return character;
}

// Walking into a one-metre step must not climb it.
//
// The smallest feature this world can have is a whole voxel, so free-climbing
// one means brushing against any block and ending up on top of it. What made it
// happen was stair-walking running while the character was in the air: a nudge
// against the block plus a few centimetres of jump, and the controller finished
// the climb.
void test_walking_into_a_one_metre_step_does_not_climb_it() {
  std::printf("a one metre step\n");

  engine::physics::PhysicsWorld world;
  auto character = spawn_at(world, make_step(), {-4.0F, 2.0F, 0.0F});
  check(character->is_on_ground(), "the character lands on the lower floor");
  const float floor_y = character->position().y;

  float highest = floor_y;
  for (int frame = 0; frame < 300; ++frame) {
    glm::vec3 velocity = character->linear_velocity();
    velocity.x = k_speed;
    velocity.z = 0.0F;
    const bool grounded = character->is_on_ground();
    if (grounded)
      velocity.y = std::min(velocity.y, 0.0F);
    else
      velocity.y = std::max(velocity.y - 9.81F * k_dt, -50.0F);
    character->set_velocity(velocity);
    // The game passes zero step-up whenever the character is not grounded.
    character->update(k_dt, engine::physics::Character::k_stick_to_floor,
                      grounded ? engine::physics::Character::k_step_height : 0.0F);
    world.step(k_dt);
    highest = std::max(highest, character->position().y);
  }
  std::printf("    rose %.2f m over five seconds of walking into it\n",
              static_cast<double>(highest - floor_y));
  check(highest - floor_y < 0.3F, "walking into a one metre step does not climb it");
}

// ...but a jump does, because a jump clears a metre on its own.
void test_a_jump_still_gets_onto_the_step() {
  std::printf("jumping the step\n");

  engine::physics::PhysicsWorld world;
  auto character = spawn_at(world, make_step(), {-2.0F, 2.0F, 0.0F});
  const float floor_y = character->position().y;

  for (int frame = 0; frame < 300; ++frame) {
    glm::vec3 velocity = character->linear_velocity();
    velocity.x = k_speed;
    velocity.z = 0.0F;
    const bool grounded = character->is_on_ground();
    // Held for a few frames, the way a key is. Setting it on one frame and
    // zeroing it the next -- which is what "else if grounded, y = 0" does --
    // cancels the jump before it has left the ground, because the controller
    // still reports contact on the frame after.
    if (frame >= 20 && frame <= 23)
      velocity.y = 8.2F;
    else if (grounded)
      velocity.y = std::min(velocity.y, 0.0F);
    else
      velocity.y = std::max(velocity.y - 24.0F * k_dt, -50.0F);
    character->set_velocity(velocity);
    character->update(k_dt, engine::physics::Character::k_stick_to_floor,
                      grounded ? engine::physics::Character::k_step_height : 0.0F);
    world.step(k_dt);
  }
  std::printf("    ended %.2f m above the lower floor at x = %.1f\n",
              static_cast<double>(character->position().y - floor_y),
              static_cast<double>(character->position().x));
  check(character->position().y - floor_y > 0.7F, "a jump gets onto the step");
}

// A jump into a ceiling has to stop rising.
//
// Nothing was cancelling upward velocity against a ceiling, so the character
// was pinned to it for the rest of the jump's duration while gravity worked the
// speed off -- the jump lasted exactly as long whether or not there was a roof.
void test_bumping_your_head_stops_the_jump() {
  std::printf("bumping your head\n");

  engine::physics::PhysicsWorld world;
  // Three metres of room and a spawn near the floor, so the capsule is not
  // already intersecting the ceiling when the test starts -- at 2.6 m it was,
  // and a character jammed into the roof does not jump, which reads as the
  // ceiling working when nothing was being tested at all.
  auto character = spawn_at(world, make_room(3.0F), {0.0F, 0.3F, 0.0F});
  check(character->is_on_ground(), "the character lands on the floor");
  const float floor_y = character->position().y;

  int frames_at_peak = 0;
  float peak = floor_y;
  for (int frame = 0; frame < 200; ++frame) {
    glm::vec3 velocity = character->linear_velocity();
    velocity.x = 0.0F;
    velocity.z = 0.0F;
    const bool grounded = character->is_on_ground();
    if (frame >= 10 && frame <= 13)
      velocity.y = 8.2F;
    else if (grounded)
      velocity.y = std::min(velocity.y, 0.0F);
    else
      velocity.y = std::max(velocity.y - 24.0F * k_dt, -50.0F);
    character->set_velocity(velocity);

    const float before = character->position().y;
    character->update(k_dt, engine::physics::Character::k_stick_to_floor,
                      grounded ? engine::physics::Character::k_step_height : 0.0F);
    world.step(k_dt);

    // The fix, as the game applies it: if we meant to rise and did not, stop
    // trying. Testing "did we go where we said we were going" rather than
    // querying contacts keeps it true whatever the character hit.
    if (velocity.y > 0.0F && character->position().y - before < velocity.y * k_dt * 0.5F) {
      const glm::vec3 after = character->linear_velocity();
      character->set_velocity({after.x, 0.0F, after.z});
    }

    const float here = character->position().y;
    if (frame > 10) {
      peak = std::max(peak, here);
      if (peak > floor_y + 0.2F && here > peak - 0.05F)
        ++frames_at_peak;
    }
  }
  std::printf("    rose %.2f m, spent %d frames within 5 cm of the top\n",
              static_cast<double>(peak - floor_y), frames_at_peak);
  // Well short of the 1.4 m an unobstructed jump reaches, because the ceiling
  // is doing its job -- what matters is that it left the ground at all.
  check(peak - floor_y > 0.3F, "the jump gets off the ground");
  check(peak - floor_y < 1.3F, "and the ceiling stops it short of a free jump");
  check(frames_at_peak < 12, "and it comes straight back down instead of hanging there");
}

void test_walks_up_a_45_degree_slope() {
  std::printf("45 degree slope\n");

  engine::physics::PhysicsWorld world;
  const engine::MeshSource ramp = make_ramp(1.0F); // 1 up per 1 along = 45 degrees
  auto character = spawn_on(world, ramp);

  check(character->is_on_ground(), "the character lands on the ramp's flat approach");

  const Walk walk = walk_along(world, *character, 4.0F);
  check(!walk.stalled, "walking into a 45 degree slope does not get stuck against it");
  check(walk.climbed > 2.0F, "and actually gains height");
  check(walk.distance > 8.0F, "while continuing to cover ground");
}

void test_speed_is_the_same_uphill_as_on_the_flat() {
  std::printf("constant speed\n");

  const auto distance_on = [](float rise) {
    engine::physics::PhysicsWorld world;
    const engine::MeshSource ground = make_ramp(rise);
    auto character = spawn_on(world, ground);
    return walk_along(world, *character, 3.0F).distance;
  };

  const float flat = distance_on(0.0F);
  const float gentle = distance_on(0.5F);
  const float steep = distance_on(1.0F);

  check(flat > 8.0F, "the flat baseline covers ground");
  // Within a tolerance: the controller still has to resolve contacts, and the
  // first frames spend a little settling onto the slope.
  std::printf("    flat %.1f m, 27 degrees %.1f m (%.0f%%), 45 degrees %.1f m (%.0f%%)\n",
              static_cast<double>(flat), static_cast<double>(gentle),
              static_cast<double>(gentle / flat * 100.0F), static_cast<double>(steep),
              static_cast<double>(steep / flat * 100.0F));
  check(std::abs(gentle - flat) < flat * 0.15F,
        "a 27 degree climb covers the same ground as the flat");
  check(std::abs(steep - flat) < flat * 0.15F,
        "and so does a 45 degree climb -- no uphill tax");
}

void test_a_wall_is_still_a_wall() {
  std::printf("slope limit\n");

  engine::physics::PhysicsWorld world;
  // Far steeper than the controller's limit: this must stop the character.
  const engine::MeshSource cliff = make_ramp(6.0F);
  auto character = spawn_on(world, cliff);

  const Walk walk = walk_along(world, *character, 3.0F);
  check(walk.climbed < 2.0F, "the slope limit stops the character walking up a cliff");
}

void test_a_single_block_step_is_taken_in_stride() {
  std::printf("one voxel step\n");

  engine::physics::PhysicsWorld world;
  // Flat, then one abrupt voxel up, then flat again -- the smallest feature a
  // voxel world can have, and the one the old 0.4 step height could not clear.
  engine::MeshSource ground;
  const auto quad = [&ground](float x0, float x1, float y) {
    const auto base = static_cast<std::uint32_t>(ground.vertices.size());
    for (const auto &corner : {std::pair{x0, -12.0F}, std::pair{x0, 12.0F},
                               std::pair{x1, -12.0F}, std::pair{x1, 12.0F}}) {
      engine::MeshVertex v{};
      v.pos[0] = corner.first;
      v.pos[1] = y;
      v.pos[2] = corner.second;
      ground.vertices.push_back(v);
    }
    ground.indices.insert(ground.indices.end(),
                          {base, base + 1, base + 2, base + 1, base + 3, base + 2});
  };
  quad(-10.0F, 0.0F, 0.0F);
  quad(0.0F, 40.0F, 1.0F);
  // The riser, so there is no gap to fall into.
  {
    const auto base = static_cast<std::uint32_t>(ground.vertices.size());
    for (const auto &corner : {std::tuple{0.0F, 0.0F, -12.0F}, std::tuple{0.0F, 0.0F, 12.0F},
                               std::tuple{0.0F, 1.0F, -12.0F}, std::tuple{0.0F, 1.0F, 12.0F}}) {
      engine::MeshVertex v{};
      v.pos[0] = std::get<0>(corner);
      v.pos[1] = std::get<1>(corner);
      v.pos[2] = std::get<2>(corner);
      ground.vertices.push_back(v);
    }
    ground.indices.insert(ground.indices.end(),
                          {base, base + 1, base + 2, base + 1, base + 3, base + 2});
  }

  auto character = spawn_on(world, ground);
  const Walk walk = walk_along(world, *character, 3.0F);
  // This used to assert the character ends up on TOP of a one voxel step, back
  // when the step height was 1.05. That was the wrong call: one voxel is the
  // smallest thing the world can contain, so free-climbing one means brushing
  // against any block in the game and finding yourself standing on it. Getting
  // up a block is a jump. See test_walking_into_a_one_metre_step_does_not_climb_it.
  check(walk.climbed < 0.4F, "walking alone does not get you on top of a one voxel step");
}

} // namespace

// A 45 degree ramp -- the shape our stair voxels actually collide as -- running
// up to a flat landing at the top. This is the reported bug: "before going all
// the way up the stairs, we just kinda skip over to the landing from the side
// of the stair. We just, like, teleport up on top of it really quick."
auto make_ramp_to_landing(float height = 3.0F, float width = 24.0F) -> engine::MeshSource {
  engine::MeshSource mesh;
  const auto height_at = [height](float x) {
    if (x <= 0.0F) return 0.0F;            // flat approach
    if (x >= height) return height;        // the landing
    return x;                              // 45 degrees, one up per one along
  };

  const int steps = 60;
  for (int i = 0; i <= steps; ++i) {
    const auto x = static_cast<float>(i) - 8.0F;
    for (int side = 0; side < 2; ++side) {
      engine::MeshVertex v{};
      v.pos[0] = x;
      v.pos[1] = height_at(x);
      v.pos[2] = side == 0 ? -width * 0.5F : width * 0.5F;
      mesh.vertices.push_back(v);
    }
  }
  for (int i = 0; i < steps; ++i) {
    const auto base = static_cast<std::uint32_t>(i * 2);
    mesh.indices.insert(mesh.indices.end(),
                        {base, base + 1, base + 2, base + 1, base + 3, base + 2});
  }
  return mesh;
}

void test_climbing_to_a_landing_is_continuous() {
  std::printf("climbing a stair ramp onto its landing\n");

  engine::physics::PhysicsWorld world;
  const engine::MeshSource ground = make_ramp_to_landing();
  auto character = spawn_on(world, ground);

  const Walk walk = walk_along(world, *character, 5.0F);
  check(walk.climbed > 2.5F, "the character gets up onto the landing");

  // Godot's CharacterBody3D has no step-up at all; its only vertical assist is
  // a downward floor snap, 0.1 m by default. A climb that is really a slide up
  // a slope should produce nothing bigger than that.
  std::printf("  (worst single-frame teleport %.3f m over %d frames)\n",
              static_cast<double>(walk.worst_teleport), walk.teleports);
  check(walk.worst_teleport < 0.10F,
        "no single frame moves the character further than Godot's floor snap");
}

// The reported bug, isolated: a riser too short to be a wall and too tall to be
// terrain -- the side of a stair tread, or the lip of a landing you brush while
// still on the ramp. Jolt's stair sweep fires whenever horizontal motion is
// blocked and lifts the character its whole step height in ONE frame, which is
// the "we just teleport up on top of it really quick" the player sees.
//
// Godot's CharacterBody3D does not do this at all. It has no step-up mechanic:
// you go up by sliding along a walkable slope, and its only vertical assist is
// a downward floor snap of 0.1 m. Whatever we lift the character by, no single
// frame should move it further than that, or the motion reads as a teleport
// rather than as walking.
// A lip in the ground -- the kind marching cubes leaves where two cells meet.
//
// This used to walk a 20 cm and a 30 cm riser and call them "the side of a stair
// tread", which was measuring something the game does not contain: a stair
// collides as a smooth ramp from foot to head, so there is no tread to catch on.
// The 30 cm case was the only thing in the whole suite that needed the stair
// sweep, and it was keeping a mechanic alive for a shape with no source.
//
// What IS real is a small lip, and the capsule's own rounded bottom takes those
// without any sweep at all.
void test_a_lip_in_the_ground_is_taken_in_stride() {
  std::printf("a lip in the ground\n");

  for (const float rise : {0.10F, 0.20F}) {
    engine::physics::PhysicsWorld world;
    const engine::MeshSource ground = make_step(60.0F, 24.0F, rise);
    auto character = spawn_on(world, ground);
    const Walk walk = walk_along(world, *character, 3.0F);

    const auto label = std::to_string(static_cast<int>(rise * 100.0F)) + " cm lip";
    std::printf("  %s: climbed %.2f m, worst single frame %.3f m\n", label.c_str(),
                static_cast<double>(walk.climbed), static_cast<double>(walk.worst_teleport));
    check(walk.climbed > rise * 0.8F, "a " + label + " is walked over rather than blocked");
    // The BODY has to resolve collision within the frame, so some of this is
    // irreducible. There is no longer a low-pass on the eye hiding it -- the
    // camera rides the body the way a Godot camera rides its CharacterBody3D,
    // because a filter that lags the eye behind the body turns every jump into
    // a soar. So this bound is the real one now: it is what the player sees.
    check(walk.worst_teleport < 0.10F, "a " + label + " is not hopped over in one frame");
  }
}

// Rolling ground, the way marching cubes actually makes it: a surface whose
// height varies continuously, so the slope under the character changes every
// frame.
//
// This is the case the player described as jitter, and the flat-ramp tests
// cannot reach it. A 45 degree ramp has one slope from end to end; real terrain
// has a different one every step, and every change is a chance for the
// controller to resolve the difference by moving the capsule somewhere the
// surface did not put it.
auto make_rolling(float amplitude, float wavelength, float length = 80.0F,
                  float width = 24.0F) -> engine::MeshSource {
  engine::MeshSource mesh;
  const auto height_at = [&](float x) {
    if (x <= 0.0F)
      return 0.0F;
    // Two waves rather than one, so the relief is not periodic with the
    // character's stride and the crests are not all the same shape.
    return amplitude * (std::sin(x * 6.2831853F / wavelength) +
                        0.4F * std::sin(x * 6.2831853F / (wavelength * 0.37F)));
  };

  const int steps = static_cast<int>(length * 2.0F); // half-metre cells
  for (int i = 0; i <= steps; ++i) {
    const float x = static_cast<float>(i) * 0.5F - 8.0F;
    for (int side = 0; side < 2; ++side) {
      const float z = side == 0 ? -width * 0.5F : width * 0.5F;
      engine::MeshVertex v{};
      v.pos[0] = x;
      v.pos[1] = height_at(x);
      v.pos[2] = z;
      mesh.vertices.push_back(v);
    }
  }
  for (int i = 0; i < steps; ++i) {
    const auto base = static_cast<std::uint32_t>(i * 2);
    mesh.indices.insert(mesh.indices.end(),
                        {base, base + 1, base + 2, base + 1, base + 3, base + 2});
  }
  return mesh;
}

void test_walking_over_rolling_ground_stays_on_it() {
  std::printf("walking over rolling ground stays on it\n");

  // What to measure, after three metrics that measured the terrain instead.
  //
  // "How far did the controller move the character that the ground does not
  // explain" is the right question and it has no honest answer on a trimesh: a
  // capsule crossing a facet edge pivots on the EDGE, so its height above
  // either facet is genuinely discontinuous there, and every candidate measure
  // -- second difference of height, rise against the last normal, ride height
  // above the surface -- is dominated by that. Each one was tried and each one
  // reported the mesh's own faceting as jitter, at four, eight and twenty-three
  // centimetres respectively.
  //
  // The invariant that IS clean: walking over rolling ground, the character
  // should stay on it. Leaving the ground at a crest and being pulled back down
  // is precisely what the floor snap exists to prevent, it needs no reference
  // surface to detect, and it is what jitter is made of -- the eye rises with a
  // hop it did not ask for and is then yanked back.
  //
  // The old settings could not prevent it, because the snap was not running.
  // stick_down was `rising > 0 ? 1.2 : 0` with `rising` read straight after
  // set_velocity, and velocity.y is zero whenever the character is grounded and
  // not jumping -- so during an ordinary walk the snap was OFF, and during a
  // jump's ascent it was 1.2 m, fighting the jump. Godot's rule is the other
  // way round and its length is a tenth of a metre.
  constexpr float k_amplitude = 0.25F;
  constexpr float k_wavelength = 4.0F;
  engine::physics::PhysicsWorld world;
  const engine::MeshSource ground = make_rolling(k_amplitude, k_wavelength);
  auto character = spawn_on(world, ground);
  check(character->is_on_ground(), "the character lands on the flat approach");

  int airborne = 0;
  int walked_frames = 0;
  const int frames = static_cast<int>(12.0F / k_dt);
  const glm::vec3 start = character->position();

  for (int frame = 0; frame < frames; ++frame) {
    glm::vec3 velocity{k_speed, 0.0F, 0.0F};
    const bool grounded = character->is_on_ground();
    if (grounded) {
      const float up = character->ground_normal().y;
      if (up > 0.5F)
        velocity *= std::min(1.0F / (up * up), tuned("SLOPECAP", k_slope_cap));
    } else {
      velocity.y = std::max(character->linear_velocity().y - 9.81F * k_dt, -50.0F);
    }
    character->set_velocity(velocity);
    const bool hold_ground = grounded && velocity.y <= 0.0F;
    character->update(k_dt,
                      hold_ground ? tuned("SNAP", engine::physics::Character::k_floor_snap) : 0.0F,
                      hold_ground ? tuned("STEP", engine::physics::Character::k_step_height) : 0.0F);
    world.step(k_dt);

    if (character->position().x < 2.0F)
      continue; // still on the flat approach
    ++walked_frames;
    if (!character->is_on_ground())
      ++airborne;
  }

  const float distance = character->position().x - start.x;
  check(distance > 30.0F, "it covers ground the whole way");
  std::printf("    airborne on %d of %d frames walking %.1f m of rolling ground\n", airborne,
              walked_frames, static_cast<double>(distance));
  // Nothing here is steeper than 40 degrees and the character is walking, not
  // running off a cliff. Any frame off the ground is a hop over a crest.
  check(airborne * 100 < walked_frames * 2,
        "it does not bounce off the crests it is walking over");
}

// The constant-speed rule must not become a launcher near the slope limit.
//
// 1/cos^2 is 2 at 45 degrees and 4 at 60, and 60 is exactly where the
// controller stops calling a surface walkable -- so the compensation is largest
// precisely where the ground is steepest, and 4x on a 60 degree face is four
// metres a second going up at seven. The moment the slope eases, that is a jump
// nobody asked for.
void test_the_slope_compensation_cannot_run_away() {
  std::printf("the slope compensation cannot run away\n");

  float flat_distance = 0.0F;
  for (const float rise : {0.0F, 0.5F, 1.0F, 1.6F}) { // 0, 27, 45, 58 degrees
    engine::physics::PhysicsWorld world;
    const engine::MeshSource ground = make_ramp(rise);
    auto character = spawn_on(world, ground);
    const Walk walk = walk_along(world, *character, 2.5F);

    const std::string label = "rise " + std::to_string(static_cast<int>(rise * 100.0F)) + "cm";
    if (rise == 0.0F) {
      flat_distance = walk.distance;
      check(flat_distance > 8.0F, "the flat baseline covers ground");
      continue;
    }
    std::printf("    %s: %.2f m against %.2f m flat\n", label.c_str(),
                static_cast<double>(walk.distance), static_cast<double>(flat_distance));
    // Never faster than walking on the flat. Slower is honest -- a steep climb
    // costing you something is fine -- but faster is the compensation
    // overshooting, and overshoot is what turns a crest into a launch.
    check(walk.distance <= flat_distance * 1.05F,
          "a " + label + " is never covered FASTER than flat ground");
    check(walk.distance > flat_distance * 0.45F, "and is not reduced to a crawl either");
  }
}

// One stair voxel, collided the way the game collides one: a closed WEDGE --
// floor, head wall, two triangular sides and the slope across the top. See
// collide_stair in the game's voxel_mesher.
//
// The sides are what this is about. They are vertical triangles rising from
// nothing at the foot of the wedge to a full metre at its head, so walking into
// one from the side is walking into a wall whose height depends on exactly
// where along it you arrive.
auto make_stair_wedge(float width = 24.0F) -> engine::MeshSource {
  engine::MeshSource mesh;
  const auto tri = [&mesh](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 outward) {
    if (glm::dot(glm::cross(b - a, c - a), outward) < 0.0F)
      std::swap(b, c);
    const auto base = static_cast<std::uint32_t>(mesh.vertices.size());
    for (const glm::vec3 &p : {a, b, c}) {
      engine::MeshVertex v{};
      v.pos[0] = p.x;
      v.pos[1] = p.y;
      v.pos[2] = p.z;
      mesh.vertices.push_back(v);
    }
    mesh.indices.insert(mesh.indices.end(), {base, base + 1, base + 2});
  };
  const auto quad = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d, glm::vec3 outward) {
    tri(a, b, c, outward);
    tri(a, c, d, outward);
  };

  const float h = width * 0.5F;
  quad({-20, 0, -h}, {20, 0, -h}, {20, 0, h}, {-20, 0, h}, {0, 1, 0}); // the floor it stands on

  // One voxel, climbing toward +x, spanning z in [0, 1].
  quad({0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}, {0, -1, 0});
  quad({1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {1, 0, 1}, {1, 0, 0});
  quad({0, 0, 0}, {1, 1, 0}, {1, 1, 1}, {0, 0, 1}, {0, 1, 0});
  tri({0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 0, -1});
  tri({0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 0, 1});
  return mesh;
}

// Walking into the SIDE of a stair must not put you on top of it.
//
// Reported as: "instead of going up the ramp like you normally would, you go up
// the non-ramp side, like the side of it, and you just instantly teleport to the
// middle of the ramp on top of it."
//
// Swept along the whole length of the wedge rather than sampled at one place,
// because the side's height varies from nothing to a metre across it and the
// interesting part is wherever the stair sweep decides it is a step. The sweep
// does not put you down where it lifted you: it goes UP, then FORWARD, then
// DOWN, and what it lands on is the sloped top of the wedge.
void test_walking_into_the_side_of_a_stair_does_not_climb_it() {
  std::printf("the side of a stair is a wall\n");

  float worst_climb = 0.0F;
  float worst_at = 0.0F;
  for (int i = 1; i < 20; ++i) {
    const float x = static_cast<float>(i) * 0.05F;
    engine::physics::PhysicsWorld world;
    world.create_static_mesh(make_stair_wedge(), glm::vec3(0.0F));
    auto character =
        std::make_unique<engine::physics::Character>(world, glm::vec3(x, 2.0F, -3.0F));
    for (int f = 0; f < 120; ++f) { // settle on the floor beside the wedge
      character->set_velocity({0.0F, std::min(character->linear_velocity().y - 9.81F * k_dt, 0.0F),
                               0.0F});
      character->update(k_dt, engine::physics::Character::k_floor_snap, 0.0F);
      world.step(k_dt);
    }
    const float floor_y = character->position().y;

    float highest = floor_y;
    for (int f = 0; f < 150; ++f) {
      glm::vec3 velocity{0.0F, 0.0F, k_speed};
      const bool grounded = character->is_on_ground();
      if (!grounded)
        velocity.y = std::max(character->linear_velocity().y - 9.81F * k_dt, -50.0F);
      character->set_velocity(velocity);
      const bool hold = grounded && velocity.y <= 0.0F;
      character->update(k_dt, hold ? tuned("SNAP", engine::physics::Character::k_floor_snap) : 0.0F,
                        hold ? tuned("STEP", engine::physics::Character::k_step_height) : 0.0F);
      world.step(k_dt);
      highest = std::max(highest, character->position().y);
    }
    const float climbed = highest - floor_y;
    if (climbed > worst_climb) {
      worst_climb = climbed;
      worst_at = x;
    }
    if (std::getenv("TRACE_STAIR") != nullptr)
      std::printf("      x = %.2f: climbed %.2f m\n", static_cast<double>(x),
                  static_cast<double>(climbed));
  }

  std::printf("    worst climb %.2f m, walking in at x = %.2f\n",
              static_cast<double>(worst_climb), static_cast<double>(worst_at));
  // The step-up is what you are allowed to walk over. Getting onto a stair from
  // its side is a jump, the same as getting onto a block.
  // With no stair sweep this is the capsule's rounded bottom riding the corner
  // where the wedge's side is genuinely only a few centimetres tall, which is
  // the same thing that gets it over a lip in the ground. A third of a metre is
  // the bar because getting ONTO a stair from the side is a jump.
  check(worst_climb < 0.35F, "walking into the side of a stair does not climb onto it");
}

// Run into a wall, then immediately walk back. You have to move at once.
//
// Reported as: "if I run and sprint into a wall and then immediately try to walk
// backwards, I'm froze and stuck into the wall for a couple seconds". The cause
// is not the wall -- the controller stops the character dead every frame. It is
// that the game's own speed accumulator never hears about it: it ramps toward
// the input and nothing tells it the last four metres a second did not happen,
// so reversing has to unwind a velocity the character never had.
void test_a_wall_takes_your_speed_away() {
  std::printf("a wall takes your speed away\n");

  engine::physics::PhysicsWorld world;
  const engine::MeshSource ground = make_step(60.0F, 24.0F, 4.0F); // a 4 m wall at x = 0
  auto character = spawn_on(world, ground);
  check(character->is_on_ground(), "the character lands on the floor");

  // Sprint into the wall for a second and a half.
  glm::vec3 speed{0.0F};
  const auto drive = [&](float wish_x, int frames) {
    for (int i = 0; i < frames; ++i) {
      const glm::vec3 before = character->position();
      const glm::vec3 wish{wish_x, 0.0F, 0.0F};
      const glm::vec3 towards = wish - speed;
      const float gap = glm::length(towards);
      const float rate = glm::length(wish) > glm::length(speed) ? k_accelerate : k_decelerate;
      if (gap > 0.0001F)
        speed += (towards / gap) * std::min(rate * k_dt, gap);

      glm::vec3 velocity{speed.x, 0.0F, speed.z};
      const bool grounded = character->is_on_ground();
      if (!grounded)
        velocity.y = std::max(character->linear_velocity().y - 9.81F * k_dt, -50.0F);
      character->set_velocity(velocity);
      const bool hold = grounded && velocity.y <= 0.0F;
      character->update(k_dt, hold ? engine::physics::Character::k_floor_snap : 0.0F,
                        hold ? engine::physics::Character::k_step_height : 0.0F);
      world.step(k_dt);

      // Godot's move_and_slide writes the resolved velocity back into
      // `velocity`; this is that write-back. Whatever the character was stopped
      // from doing is taken out of the accumulator, along the direction it
      // failed in -- which leaves speed along a wall untouched and speed into
      // one at zero.
      if (!tuned("NOWRITEBACK", 0.0F)) {
        const glm::vec3 moved = character->position() - before;
        const glm::vec3 wanted{velocity.x * k_dt, 0.0F, velocity.z * k_dt};
        const glm::vec3 blocked{wanted.x - moved.x, 0.0F, wanted.z - moved.z};
        const float missed = glm::length(blocked);
        if (missed > 0.001F) {
          const glm::vec3 n = blocked / missed;
          speed -= n * std::max(0.0F, glm::dot(speed, n));
        }
      }
    }
  };

  drive(k_sprint, 90);
  const float into_wall = speed.x;
  std::printf("    after a second and a half against the wall, speed into it is %.2f m/s\n",
              static_cast<double>(into_wall));

  // Now walk back, and count how long before the character actually moves.
  const float x_at_wall = character->position().x;
  int frames_stuck = 0;
  for (int i = 0; i < 120; ++i) {
    const float before_x = character->position().x;
    drive(-k_speed, 1);
    if (character->position().x < before_x - 0.001F)
      break;
    ++frames_stuck;
  }
  std::printf("    %d frames (%.2f s) before it started backing away from x = %.2f\n", frames_stuck,
              static_cast<double>(static_cast<float>(frames_stuck) * k_dt),
              static_cast<double>(x_at_wall));

  check(into_wall < 0.5F, "a wall does not leave you holding the speed you hit it with");
  // A tenth of a second is a couple of frames of the acceleration ramp. Seconds
  // is the bug.
  check(frames_stuck < 8, "and backing off starts at once rather than after a wind-down");
}

int main() {
  test_walks_up_a_45_degree_slope();
  test_speed_is_the_same_uphill_as_on_the_flat();
  test_a_wall_is_still_a_wall();
  test_a_single_block_step_is_taken_in_stride();
  test_climbing_to_a_landing_is_continuous();
  test_a_lip_in_the_ground_is_taken_in_stride();
  test_walking_over_rolling_ground_stays_on_it();
  test_the_slope_compensation_cannot_run_away();
  test_walking_into_the_side_of_a_stair_does_not_climb_it();
  test_a_wall_takes_your_speed_away();
  test_walking_into_a_one_metre_step_does_not_climb_it();
  test_a_jump_still_gets_onto_the_step();
  test_bumping_your_head_stops_the_jump();

  if (g_failures > 0) {
    std::printf("\n%d check(s) failed\n", g_failures);
    return EXIT_FAILURE;
  }
  std::printf("\nall character slope checks passed\n");
  return EXIT_SUCCESS;
}
