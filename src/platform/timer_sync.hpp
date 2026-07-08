#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace engine {

// Port of Godot's MainTimerSync::advance_core (main_timer_sync.cpp:349-550).
// History-smoothed physics step count eliminates the 1↔2 oscillation that raw
// accumulator loops produce when frame deltas jitter.
class TimerSync {
public:
  struct Result {
    int physics_steps{0};
    double interpolation_fraction{0.0};
  };

  Result advance(double frame_delta) {
    // Accumulate time
    accumulator_ += frame_delta;

    // Raw step count
    int desired = static_cast<int>(std::floor(accumulator_ / k_physics_step));
    if (desired < 0) desired = 0;

    // History-smoothed typical step count (running average over last N frames)
    // Godot: typical_physics_steps (main_timer_sync.cpp:374)
    int smooth = static_cast<int>(std::round(typical_steps_));
    if (smooth < 1) smooth = 1;

    // Godot jitter fix (main_timer_sync.cpp:388-404):
    // If the raw count is one more than the smoothed average, defer it.
    // Save the excess in time_deficit; only let the count rise when the
    // deficit accumulates past physics_jitter_fix * dt.
    if (desired == smooth + 1 && desired >= 2) {
      time_deficit_ += accumulator_ - smooth * k_physics_step;
      desired = smooth;
    }

    // If we're more than one step ahead of the smooth average, the load
    // genuinely changed — reset the ring buffer and accept the new count.
    if (desired > smooth + 1) {
      typical_steps_ = static_cast<double>(desired);
      time_deficit_ = 0.0;
      ring_filled_ = 0;
    }

    // Allow desired to increment on the next frame if enough time_deficit
    // has built up (Godot: "compensate and call"). This keeps it toggling
    // on the right beat instead of randomly.
    if (time_deficit_ >= k_jitter_fix * k_physics_step)
      time_deficit_ = 0.0; // next frame: desired can rise

    // Clamp to max steps per frame (Godot: max_physics_steps_per_frame, default 8)
    int steps = std::min(desired, k_max_steps);

    // If we truly can't keep up, discard excess (Godot: main.cpp:4953)
    if (desired > k_max_steps)
      accumulator_ = 0.0;

    // Drain consumed time
    accumulator_ -= steps * k_physics_step;
    if (accumulator_ < 0.0) accumulator_ = 0.0;

    // Interpolation fraction: always [0, 1]
    double fraction = accumulator_ / k_physics_step;
    fraction = std::clamp(fraction, 0.0, 1.0);

    push_ring(steps);

    return {steps, fraction};
  }

  void reset() {
    accumulator_ = 0.0;
    time_deficit_ = 0.0;
    typical_steps_ = 1.0;
    ring_filled_ = 0;
    ring_pos_ = 0;
  }

private:
  static constexpr int k_control_steps = 12;      // Godot: CONTROL_STEPS
  static constexpr double k_jitter_fix = 0.5;      // Godot: physics_jitter_fix
  static constexpr int k_max_steps = 2;             // max steps per frame
  static constexpr double k_physics_step = 1.0 / 60.0;

  double accumulator_{0.0};
  double time_deficit_{0.0};              // Godot: deficit before allowing step count rise
  double typical_steps_{1.0};             // Godot: typical_physics_steps (smoothed)
  int ring_buffer_[k_control_steps]{};
  int ring_pos_{0};
  int ring_filled_{0};

  void push_ring(int steps) {
    ring_buffer_[ring_pos_] = steps;
    ring_pos_ = (ring_pos_ + 1) % k_control_steps;
    if (ring_filled_ < k_control_steps)
      ++ring_filled_;

    double sum = 0.0;
    for (int i = 0; i < ring_filled_; ++i)
      sum += static_cast<double>(ring_buffer_[i]);
    typical_steps_ = sum / ring_filled_;
  }
};

} // namespace engine
