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
//   the write-back    whatever the character was stopped from doing comes out of
//                     the velocity. Godot's velocity.slide(wall_normal).
//
// Godot's third piece, floor_constant_speed, is deliberately NOT here. See the
// note further down: it works, and it made diagonal climbing feel like being
// shoved sideways.
class GroundMotion {
public:
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

  // There is no constant-speed rule here, deliberately, and this is the second
  // time this has been rewritten.
  //
  // Godot has one -- floor_constant_speed -- and this used to port it: run the
  // sweep, measure the horizontal distance actually covered against the distance
  // asked for, and sweep the shortfall again, up to four passes. It worked as
  // specified. A 45 degree climb ran at 98% of flat ground instead of 71%, with
  // no overshoot at the crest.
  //
  // It felt wrong anyway, and the report says exactly why:
  //
  //   "if I go up a ramp diagonally, and I'm going forward and to the left, then
  //   it pushes me to the left a little bit... I guess slide more to the left,
  //   because going up is harder for me than going sideways. But because you
  //   have it where our speed never changes, it feels like I'm dashing to the
  //   side."
  //
  // That is the rule working as designed, and the design being wrong for this
  // game. Climbing diagonally, the sweep is resisted along the uphill component
  // and not at all across it, so the character is deflected sideways. Restoring
  // the total DISTANCE then feeds that deflection: you keep full speed, and the
  // speed you keep is pointing further sideways than you asked for. Constant
  // speed does not mean constant direction, and it is direction the hands feel.
  //
  // So: climbing costs speed now, which is what the ground is for. Nothing is
  // topped up, nothing is deflected by being topped up, and the whole rule is
  // one sweep.
  //
  // The reason it was ported in the first place was fairness -- "the things
  // chasing you path over terrain at a flat rate", so a player slowed by a hill
  // is a player the horde catches for free. That is answered where it should
  // have been answered in the first place: zombies pay the same tax. See
  // Mob::slope_speed_factor.

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
