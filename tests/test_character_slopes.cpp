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
  check(!walk.stalled, "a one voxel step does not stop the character");
  check(walk.climbed > 0.5F, "the character ends up on top of it");
}

} // namespace

int main() {
  test_walks_up_a_45_degree_slope();
  test_speed_is_the_same_uphill_as_on_the_flat();
  test_a_wall_is_still_a_wall();
  test_a_single_block_step_is_taken_in_stride();

  if (g_failures > 0) {
    std::printf("\n%d check(s) failed\n", g_failures);
    return EXIT_FAILURE;
  }
  std::printf("\nall character slope checks passed\n");
  return EXIT_SUCCESS;
}
