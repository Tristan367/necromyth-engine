# Vulkan C++ Engine

Minimal Vulkan game engine in modern C++.

## Requirements

- CMake 3.24+
- A C++23 compiler (GCC 13+, Clang 16+, or similar)
- [Vulkan SDK](https://vulkan.lunarg.com/) with the LunarG validation layer
- [SDL3](https://github.com/libsdl-org/SDL) development libraries

Install these via your package manager, or build/install them locally and point CMake at them (see below).

## Configure Dependencies

CMake discovers dependencies through standard paths and environment variables:

| Dependency | How CMake finds it |
|---|---|
| Vulkan | `VULKAN_SDK` environment variable, or system install |
| SDL3 | `CMAKE_PREFIX_PATH`, or system install |

Example with locally installed dependencies:

```bash
export VULKAN_SDK="$HOME/opt/vulkan-sdk/default/x86_64"
export CMAKE_PREFIX_PATH="$HOME/opt/SDL"
cmake -S . -B build
```

You can also pass the Vulkan path directly:

```bash
cmake -S . -B build -DVULKAN_SDK_ROOT="$VULKAN_SDK"
```

If SDL3 is installed system-wide (e.g. Arch `sdl3` package), no extra prefix is needed.

## Build

```bash
make              # Release build (default)
make debug        # Debug build with Vulkan validation layers
make release      # Release build explicitly
make clean        # Remove build directory
```

## Run

```bash
make run
```

Prefer an external terminal for running the app. Close the window or press Ctrl+C to exit.

## Reference Projects

Several Vulkan reference projects are available locally at `/home/tristan/Projects/vulkan examples/` for architecture study. These are not required to build or run this project.

## Project Goals

The goal is to build a small, fast Vulkan engine with AI assistance and open source references.

The engine stays minimal:

- No scripting language
- Probably no editor window
- Shadow mapping, 3D animation, Blender-to-game asset workflow
- Sky shader or traditional skybox mesh
- Khronos tutorial fundamentals: MSAA, mip mapping, texture sampling, depth buffering, model loading

The renderer avoids unnecessary complexity — no PBR, no normal maps unless direction changes.

## Code Organization

Keep the project uncluttered. Avoid hundreds of tiny files, but also avoid one massive source file. Use minimal abstractions that make ownership and rendering flow clear.

Prefer modern C++ and modern Vulkan (`vk::raii::*`). Single `main.cpp` with engine code in `.hpp` files. Use RAII for all resource ownership.
