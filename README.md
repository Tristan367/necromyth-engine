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

**Shell setup is optional.** `make run` / `make debug` set `VULKAN_SDK`, `LD_LIBRARY_PATH`, and `VK_ADD_LAYER_PATH` from the CMake SDK path.

If SDL3 is installed system-wide (e.g. Arch `sdl3` package), no extra prefix is needed.

## Build

```bash
make              # Release build (default)
make debug        # Debug build and run (validation layers)
make release      # Release build explicitly
make run          # Release build and run
make clean        # Remove build directory
```

To compile without launching the app: `make BUILD_TYPE=Debug build` or `make BUILD_TYPE=Release build`.

Shaders are written in Slang and compiled by CMake via `slangc` from the Vulkan SDK. SPIR-V output goes to `build/shaders/slang.spv`.

## Run

```bash
make run          # Release build and run
make debug        # Debug build and run (validation layers)
```

Prefer an external terminal for running the app. Close the window or press Ctrl+C to exit.

## Reference Projects

Several Vulkan reference projects live at `/home/tristan/Projects/vulkan examples/` for study. They are not required to build this project.

Use them with intent:

| Repo | Use for |
|---|---|
| **Khronos Vulkan-Tutorial** | Step-by-step feature sequence (what to build next). Not engine architecture — each chapter is a self-contained demo. |
| **Sascha Willems `Vulkan/`** | Engine-adjacent patterns: split modules, dynamic rendering (`trianglevulkan13`), depth, buffers, descriptors. |
| **Vulkan-Tutorial `simple_engine`** | Holochip's split layout (`VulkanDevice`, `SwapChain`, `Renderer`) — closer to a real engine than the chapter attachments. |
| **HowToVulkan** | SDL3 + modern Vulkan 1.3 in one file; good for swapchain/depth resize behavior, not structure. |

**Two-repo rule:** adopt a technique only when at least two reference projects do it the same way (or one repo does it clearly and a second confirms the pattern). If only the tutorial does it, treat it as a learning step, not a permanent convention.

## Project Goals

The goal is to build a small, fast Vulkan engine with AI assistance and open source references.

The engine stays minimal:

- No scripting language
- Probably no editor window
- Shadow mapping, 3D animation, Blender-to-game asset workflow
- Sky shader or traditional skybox mesh
- Khronos tutorial fundamentals: MSAA, mip mapping, texture sampling, depth buffering, model loading

Current renderer modules: `vulkan_context` (orchestrator), `graphics_pipeline`, `depth_image`, `buffer`, `vertex`, `image_barrier`, `platform/sdl_window`.

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

**Swapchain:** prefer `Mailbox`, fall back to `FIFO`; recreate with `oldSwapchain`; debounce window resize (~100 ms); wait for non-zero extent before recreating; recreate depth image on swapchain resize (Khronos ch. 27, HowToVulkan, Sascha `trianglevulkan13`).

**Depth:** format selection tries `D32_SFLOAT`, then `D32_SFLOAT_S8_UINT`, then `D24_UNORM_S8_UINT`; device-local optimal image; clear to 1.0; `CompareOp::eLessOrEqual` (Sascha, HowToVulkan).

**Synchronization:** signal semaphores at `eColorAttachmentOutput`, not `eAllGraphics`. Dynamic rendering color attachments use `eColorAttachmentOptimal`.

**Validation:** Debug builds require `VK_LAYER_KHRONOS_validation` and fail fast if it is missing. Release builds run without validation. No validation output on startup usually means the layer is active and found nothing wrong — debug builds print `Vulkan validation: enabled` to confirm.

**Extensions:** verify required instance and device extensions exist before create, not only at link time.

**Helper order:** define lower-level helpers (e.g. `has_name`) before helpers that call them (e.g. `supports_all_names`).
