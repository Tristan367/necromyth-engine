#pragma once

#include "physics/physics_world.hpp"

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include <algorithm>
#include <cmath>

namespace engine::physics {

// The ground-movement rule, in one place.
//
// It lives here rather than in the game loop because it had been living in two
// places -- the loop and the engine's slope harness -- and they drifted apart
// twice. The second time cost a session: the harness was measuring a
// constant-speed rule the game had removed, so every number it produced about
// steps was about a controller nobody played. A rule with two copies is a rule
// with one of them wrong.
//
// What it is: Godot's move_and_slide, as much of it as one shot at Jolt's
// CharacterVirtual can be. Three pieces, and each is Godot's:
//
//   the accumulator   velocity is STATE, ramped toward the input, not recomputed
//                     from it. Godot keeps it on the node for the same reason.
//   constant speed    climbing must not cost ground speed, because the things
//                     chasing you path over terrain at a flat rate.
//   the write-back    whatever the character was stopped from doing comes out of
//                     the velocity. Godot's velocity.slide(wall_normal).
//
// Godot's slide loop runs several sweeps per frame, which is what lets it top
// the motion up using distance already travelled THIS frame rather than guessing
// in advance. So this runs two: see top_up().
class GroundMotion {
public:
  // Godot's max_slides is 6. Four extra passes takes a 45 degree climb from 83%
  // of flat speed to within a percent of it, and each further pass halves what
  // is left, so more would be measuring noise.
  static constexpr int k_max_climb_passes = 4;

  GroundMotion(float acceleration, float deceleration)
      : acceleration_(acceleration), deceleration_(deceleration) {}

  // The horizontal velocity to hand the controller this frame.
  //
  // `wish` is where the player is asking to go, already at the speed they are
  // asking for. Vertical is the caller's business -- gravity and jumping are
  // not movement rules.
  [[nodiscard]] auto plan(const Character &character, const glm::vec3 &wish, float dt)
      -> glm::vec3 {
    // Get up to speed, do not appear at it. Instant top speed is what lets a
    // player strafe-dance and dodge like nothing with mass. Stopping is quicker
    // than starting, because digging your heels in is easier than getting
    // moving.
    const glm::vec3 flat_wish{wish.x, 0.0F, wish.z};
    const float rate =
        glm::length(flat_wish) > glm::length(velocity_) ? acceleration_ : deceleration_;
    const glm::vec3 towards = flat_wish - velocity_;
    if (const float gap = glm::length(towards); gap > 0.0001F)
      velocity_ += (towards / gap) * std::min(rate * dt, gap);

    const float speed = glm::length(velocity_);
    if (speed < 0.0001F) {
      planned_ = 0.0F;
      return {0.0F, 0.0F, 0.0F};
    }

    planned_ = speed * dt;
    const glm::vec3 direction = velocity_ / speed;
    direction_ = direction;
    return direction * speed;
  }

  // The extra motion to sweep AFTER the first one, or zero if there is none.
  //
  // This is Godot's constant-speed rule, and it is the reason there are two
  // sweeps rather than one:
  //
  //     motion = motion.normalized() * MAX(0, motion_slide_up.length()
  //                                           - travel_slide_up.length())
  //
  // -- the horizontal distance asked for, minus the horizontal distance ALREADY
  // TRAVELLED this frame. A measurement, not a prediction, and that is the whole
  // difference. Climbing costs ground speed because the ground goes up; giving
  // it back has to be conditioned on the climb having actually happened, and
  // only a second look inside the same frame can know that.
  //
  // Two things were tried before this and both failed at the crest, which is
  // exactly where a prediction has to be wrong:
  //
  //   scale by 1/cos^2   Correct about the geometry, wrong about the timing. The
  //                      normal under the capsule still says ramp for several
  //                      frames after the ground has flattened, so the scale
  //                      keeps being applied to ground that does not need it.
  //                      Measured 5.6 m/s against a 4.0 m/s walk for five frames
  //                      -- "I shoot forward a little bit... I go fast".
  //   carry the shortfall
  //                      Repay last frame's lost distance this frame. Converges
  //                      to exactly constant speed on a steady slope, and still
  //                      overshoots at the crest, because the frame that
  //                      straddles it spends a debt earned on the ramp over
  //                      ground that is already flat. 5.7 m/s.
  //
  // Measuring within the frame cannot overshoot: the top-up is bounded by
  // distance the character demonstrably failed to cover, so on flat ground there
  // is nothing to give back and the rule does not fire at all.
  //
  // Gated on walkable ground for the reason Godot gates its top-up on the
  // surface having been classified as floor: a momentary contact with an edge or
  // a wall face reports a steep normal, and topping up against one of those is
  // how a character gets fired off a kerb.
  // Runs the rest of Godot's slide loop and returns the extra distance covered.
  //
  // `sweep` performs one controller move and returns the displacement it
  // produced -- set_velocity, update, and the difference in position. It must
  // NOT step the rigid-body world: those have had their frame, and this only
  // moves the character.
  //
  // It iterates because one extra sweep is not enough. The top-up motion goes up
  // the same slope the first one did, so it loses the same fraction: on a 45
  // degree ramp the first sweep covers half the distance asked for, the second
  // covers half of what is left, and the total is three quarters. Measured that
  // way a 45 degree climb ran at 83% of flat ground -- better than the 71% of no
  // compensation at all, and still an uphill tax. Godot iterates up to
  // max_slides for exactly this reason; four passes puts a 45 degree climb
  // within a percent, and the loop exits as soon as there is nothing owed, so
  // flat ground pays for one comparison and nothing else.
  template <typename Sweep>
  auto finish_climb(const Character &character, glm::vec3 moved, float dt, const Sweep &sweep)
      -> glm::vec3 {
    glm::vec3 extra{0.0F};
    for (int pass = 0; pass < k_max_climb_passes; ++pass) {
      const glm::vec3 velocity = top_up(character, moved, dt);
      if (glm::dot(velocity, velocity) <= 0.0F)
        break;
      const glm::vec3 step = sweep(velocity);
      extra += step;
      moved += step;
      // No progress means something is in the way rather than under us, and
      // asking again would only press harder into it.
      if (std::hypot(step.x, step.z) < 0.0005F)
        break;
    }
    return extra;
  }

  [[nodiscard]] auto top_up(const Character &character, const glm::vec3 &moved, float dt) const
      -> glm::vec3 {
    if (dt <= 0.0F || planned_ <= 0.0F)
      return {0.0F, 0.0F, 0.0F};
    if (!character.is_on_ground() || character.ground_normal().y <= 0.5F)
      return {0.0F, 0.0F, 0.0F};

    const float travelled = std::hypot(moved.x, moved.z);
    const float owed = planned_ - travelled;
    // A millimetre is the controller's own margin, not a climb.
    if (owed <= 0.001F)
      return {0.0F, 0.0F, 0.0F};
    return direction_ * (owed / dt);
  }

  // Call once both sweeps are done.
  void resolve(const Character &character) {
    // The write-back: Godot's velocity.slide(wall_normal).
    //
    // Speed into a wall goes, speed along it stays. Without this the accumulator
    // never hears that the wall stopped the character, so sprinting into one
    // leaves it holding the full speed and reversing has to unwind a velocity
    // the character never had. Reported as "I'm froze and stuck into the wall
    // for a couple seconds... it's as if the wall itself is not stopping my
    // velocity", which is exactly what was happening.
    //
    // Asking the controller which contacts are WALLS matters as much as the
    // slide itself. The first version of this inferred the obstruction from the
    // shortfall in distance -- what we asked for minus what we got -- and that
    // cannot work, because climbing costs horizontal distance for the same
    // reason a wall does. It took the accumulator from 4.00 m/s to 0.20 the
    // instant the character set foot on a ramp, and made the player walk up
    // every slope in the game from a standing start.
    const glm::vec3 wall = character.wall_normal();
    if (glm::dot(wall, wall) > 0.0F)
      velocity_ -= wall * glm::dot(velocity_, wall);
  }

  [[nodiscard]] auto velocity() const -> const glm::vec3 & { return velocity_; }

  // Respawn, teleport, world rebase: none of the state means anything after one.
  void stop() {
    velocity_ = {0.0F, 0.0F, 0.0F};
    direction_ = {0.0F, 0.0F, 0.0F};
    planned_ = 0.0F;
  }

private:
  float acceleration_;
  float deceleration_;
  glm::vec3 velocity_{0.0F};
  glm::vec3 direction_{0.0F};
  float planned_{0.0F};
};

} // namespace engine::physics
