#pragma once

#include <glm/vec3.hpp>

namespace engine {

struct DirectionalLight {
  // Unit vector from a surface point toward the sun (same vector for shading: N·L, sky disc, shadows).
  // Some engines expose the opposite (direction light rays travel, e.g. Godot -Z, Sascha -lightPos); negate to convert.
  glm::vec3 direction_toward_light{0.371F, 0.928F, 0.278F};  // unit vector toward sun
  glm::vec3 color{1.0F, 0.98F, 0.92F};
  float intensity{1.0F};
  float ambient{0.18F};
  // How high the real sun is, -1 to 1, regardless of where the shading light
  // points. At night `direction_toward_light` is swung to a moon vector well
  // above the horizon so that shadows still have a direction -- which left the
  // sky shader believing the sun was up and painting a blue midnight. This is
  // the sky's clock, and nothing else reads it.
  float sun_elevation{1.0F};
  // How much of winter there is, 0 to 1. Snow on upward faces and a cold cast
  // over everything; see applySnow and applyCold in shaders/lib/surface.slang.
  // The C# flips this as a boolean per material (VoxelClient.SetCold); a float
  // lets it arrive and leave gradually.
  float snow{0.0F};
  // How much of the sky has cloud in it. The C#'s `cloudiness`: 0.333 is an
  // ordinary day, and it moves toward 1.333 when it is about to rain, which is
  // past full cover on purpose -- the extra range is what makes an overcast sky
  // uniformly grey instead of merely crowded. See shaders/Clouds.gdshader.
  float cloudiness{0.333F};
  // How far the star field has wheeled, in radians. Driven by the GAME CLOCK
  // at a constant rate -- the C#'s starsRot, SkyManager.cs:1244 -- and never
  // by the light vector: the shading direction is swung to a synthetic moon at
  // nightfall so shadows keep a source, and a star map keyed to that vector
  // lurched every dusk. Stars turn because time passes, not because the light
  // moved.
  float star_angle{0.0F};
  // Where the moon is. Driven by the game clock at the C#'s rate: one lap
  // per day MINUS a seventh, so the phase against the sun cycles weekly
  // (SkyManager.cs MOON_DEGREES_PER_DAY). The sky lights the disc off the
  // sun direction, so this vector is all the phase machinery there is.
  glm::vec3 moon_direction{0.3F, 0.85F, 0.42F};
  // 0 = an ordinary moon, 1 = full blood. The sky reddens the disc by it.
  float blood_moon{0.0F};
  // The REAL sun's angle around its track, radians. lightDirection is swung
  // to a synthetic moon at night so shadows keep a source; the sky's moon
  // must be lit by the actual sun -- below the horizon and all -- or every
  // night is new moon. The angle travels instead of the vector because one
  // float slot was free and the track's shape is a constant.
  float sun_angle{0.0F};
};

} // namespace engine
