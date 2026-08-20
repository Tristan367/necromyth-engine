#pragma once

// Vulkan's depth range is [0, 1]; OpenGL's is [-1, 1]. GLM picks between them
// with GLM_FORCE_DEPTH_ZERO_TO_ONE, and it latches that choice the first time
// <glm/detail/setup.hpp> runs in a translation unit (one-time include guard).
//
// That makes a `#define GLM_FORCE_DEPTH_ZERO_TO_ONE` at the top of an
// individual engine header useless: if any other header pulled in glm first,
// the define arrives too late and is silently ignored. The convention has to
// come from the build system, which is why VCE::Core sets it as an INTERFACE
// compile definition for every consumer.
//
// The failure mode when it goes missing is the reason for this file. Nothing
// fails to compile — glm just quietly builds [-1, 1] projection matrices for a
// [0, 1] depth buffer, and the symptom is geometry clipped at the near plane
// and inverted shadow depth, several layers away from the cause. A stale CMake
// cache configured before the definitions were added is enough to trigger it.
//
// Fail at build time instead.

#if !defined(GLM_FORCE_DEPTH_ZERO_TO_ONE)
#error "GLM_FORCE_DEPTH_ZERO_TO_ONE is not defined for this target. Link VCE::Engine (or VCE::Core), which sets it for the whole target. If you already do, your CMake cache predates it -- delete the build directory and reconfigure. Without it every projection matrix uses the OpenGL [-1,1] depth range on a Vulkan [0,1] depth buffer."
#endif

#if !defined(GLM_FORCE_RADIANS)
#error "GLM_FORCE_RADIANS is not defined for this target. Link VCE::Engine (or VCE::Core), which sets it for the whole target."
#endif
