#pragma once

#include <string>

namespace engine {

// One layer of a texture array: a base image, optionally with a second image
// composited over it as it loads.
//
// The overlay exists so a game can ship a single wear/crack/frost image and get
// a worn variant of every tile for free, rather than authoring and shipping one
// per tile. Compositing at load time rather than sampling a second texture in
// the fragment shader means the cost is paid once instead of every frame, and
// the worn variant stays a plain array layer -- same draw call, same binding.
struct TextureArrayLayer {
  std::string path;
  std::string overlay; // empty for a plain layer
};

} // namespace engine
