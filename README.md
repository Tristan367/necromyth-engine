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

By default the Makefile uses `~/opt/vulkan-sdk/default/x86_64`. Override for one build:

```bash
make VULKAN_SDK_ROOT="$HOME/opt/vulkan-sdk/default/x86_64" debug
```

Example with locally installed SDL:

```bash
export CMAKE_PREFIX_PATH="$HOME/opt/SDL"
cmake -S . -B build -DVULKAN_SDK_ROOT="$HOME/opt/vulkan-sdk/default/x86_64"
```

**Shell setup is optional.** `make run` / `make debug` set `VULKAN_SDK`, `LD_LIBRARY_PATH`, and `VK_ADD_LAYER_PATH` from the CMake SDK path. You do not need `source setup-env.sh` in `.bashrc` for this project. If you keep a global SDK in `.bashrc`, point it at the `default` symlink so it stays in sync:

```bash
source ~/opt/vulkan-sdk/default/setup-env.sh
```

Older SDK installs (e.g. `1.4.313.0`) can be removed once nothing references them; run `make clean` and reconfigure after switching SDKs.

If SDL3 is installed system-wide (e.g. Arch `sdl3` package), no extra prefix is needed.

## Build

```bash
make              # Release build (default)
make debug        # Debug build + run (Vulkan validation layers enabled)
make release      # Release build explicitly
make run          # Release build + run
make clean        # Remove build directory
```

Shaders are written in Slang (Khronos tutorial style) and compiled automatically by CMake via `slangc` from the Vulkan SDK. SPIR-V output goes to `build/shaders/slang.spv`.

## Run

```bash
make run          # Release
make debug        # Debug with validation
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

Omit braces on single-statement `if`, `else`, and loop bodies:

```cpp
if (condition)
  do_one_thing();

if (framebuffer_resized_) {
  recreate_swapchain();
  framebuffer_resized_ = false;
  return;
}
```

Keep braces when the body is multiple statements.

## Vulkan Conventions

These are project defaults. If a change contradicts them, there should be a deliberate reason.

**Runtime API baseline is Vulkan 1.3** (`vk::ApiVersion13`, `dynamicRendering`, `synchronization2`, `submit2`). Do not hard-require Vulkan 1.4 or `maintenance5`. Shader `-profile spirv_1_4` is SPIR-V bytecode version, not the runtime API level.

**Do not define `VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS`.** Handle `vk::Result::eErrorOutOfDateKHR` and `eSuboptimalKHR` explicitly in the frame loop and swapchain recreation path.

**Physical device selection** filters first (`is_device_suitable`), then scores:
- `+1000` for discrete GPU
- `+ maxImageDimension2D` as a capability tie-breaker

Both score terms are required. Do not pick the first suitable device.

**Queue families:** first-match for graphics/compute/transfer/present, then prefer a unified family that supports graphics, compute, transfer, and present together.

**Swapchain:** prefer `Mailbox`, fall back to `FIFO`; recreate with `oldSwapchain`; debounce window resize (~100 ms); wait for non-zero extent before recreating.

**Synchronization:** signal semaphores at `eColorAttachmentOutput`, not `eAllGraphics`. Dynamic rendering color attachments use `eColorAttachmentOptimal`.

**Validation:** Debug builds require `VK_LAYER_KHRONOS_validation` and fail fast if it is missing. Release builds run without validation. No validation output on startup usually means the layer is active and found nothing wrong — debug builds print `Vulkan validation: enabled` to confirm.

**Extensions:** verify required instance and device extensions exist before create, not only at link time.

**Helper order:** define lower-level helpers (e.g. `has_name`) before helpers that call them (e.g. `supports_all_names`).
