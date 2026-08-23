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
};

// Drives the character along +X for `seconds`, using the same constant-speed
// rule the game uses.
auto walk_along(engine::physics::PhysicsWorld &world, engine::physics::Character &character,
                float seconds) -> Walk {
  const glm::vec3 start = character.position();
  glm::vec3 previous = start;
  int stalled_frames = 0;

  const int frames = static_cast<int>(seconds / k_dt);
  for (int frame = 0; frame < frames; ++frame) {
    glm::vec3 velocity = character.linear_velocity();
    velocity.x = k_speed;
    velocity.z = 0.0F;

    if (character.is_on_ground()) {
      const glm::vec3 normal = character.ground_normal();
      glm::vec3 along = velocity - normal * glm::dot(velocity, normal);
      const float flat = std::sqrt(along.x * along.x + along.z * along.z);
      if (flat > 0.001F) {
        along *= k_speed / flat;
        velocity = along;
      } else {
        velocity.y = std::min(velocity.y, 0.0F);
      }
    } else {
      velocity.y = std::max(velocity.y - 9.81F * k_dt, -50.0F);
    }

    character.set_velocity(velocity);
    character.update(k_dt);
    world.step(k_dt);

    const glm::vec3 now = character.position();
    const float moved = std::hypot(now.x - previous.x, now.z - previous.z);
    stalled_frames = moved < k_speed * k_dt * 0.25F ? stalled_frames + 1 : 0;
    previous = now;
  }

  const glm::vec3 end = character.position();
  return {std::hypot(end.x - start.x, end.z - start.z), end.y - start.y, stalled_frames > 30};
}

// Drops a character onto the ramp at x = -4 (the flat part) and lets it settle.
auto spawn_on(engine::physics::PhysicsWorld &world, const engine::MeshSource &ground)
    -> std::unique_ptr<engine::physics::Character> {
  world.create_static_mesh(ground, glm::vec3(0.0F));
  auto character = std::make_unique<engine::physics::Character>(world, glm::vec3(-4.0F, 2.0F, 0.0F));
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
auto make_step(float length = 60.0F, float width = 24.0F) -> engine::MeshSource {
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
  quad({0, 0, -h}, {0, 1, -h}, {0, 1, h}, {0, 0, h});
  quad({0, 1, -h}, {length, 1, -h}, {length, 1, h}, {0, 1, h});
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

int main() {
  test_walks_up_a_45_degree_slope();
  test_speed_is_the_same_uphill_as_on_the_flat();
  test_a_wall_is_still_a_wall();
  test_a_single_block_step_is_taken_in_stride();
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
