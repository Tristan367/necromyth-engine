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

Engine library + shaders only:

```bash
make              # Build VCE::Engine + compile shaders
make shaders      # Shaders only
make clean
```

**Run the demo** from the sibling app repo:

```bash
cd ../Vulkan-C-App && make debug
```

Optional in-tree executable: `cmake -DVCE_BUILD_IN_TREE_APP=ON ..` (legacy).

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

Current renderer modules: `vulkan_context` (orchestrator), `vulkan_device`, `swapchain`, `render_settings`, `engine_config`, `scene` (`camera`, `scene`, `mesh_instance`, `render_layer`, `directional_light`, `shadow_utils`, `floor_mesh`, `sky_mesh`), `draw_list`, `mesh_gpu`, `pipeline_id`, `graphics_pipeline`, `depth_image`, `msaa_color_image`, `shadow_map`, `buffer`, `vertex`, `model_loader`, `gltf_loader`, `texture_image`, `uniform_buffer`, `descriptors`, `image_barrier`, `platform/sdl_window`, `platform/gpu_cli`.

The renderer avoids unnecessary complexity — no PBR, no normal maps unless direction changes.

**Engine vs game:** This repo is the **engine library** (`VCE::Engine` CMake target). The [Vulkan-C-App](../Vulkan-C-App) repo is the demo/game client — fly camera, scene setup, and game logic live there. Link with `add_subdirectory(../Vulkan-C-Engine)` and `target_link_libraries(... VCE::Engine)`.

**Repo split (current):** Renderer + scene/core live here as one library. A separate *renderer-only* repo makes sense when you add a second backend (e.g. Metal). Until then, keep renderer and engine unified to avoid submodule friction.

**Models:** `load_gltf_model()` loads glTF 2.0 static meshes (tinygltf, Sascha-style CPU path). `load_obj_model()` remains for simple `.obj` assets. glTF sidecar textures (`.png`/`.jpg` next to the `.gltf`) resolve via material paths; `.glb` bundles everything into one file.

**Lighting:** `Scene::directional_light()` feeds sun direction, color, intensity, and ambient into the frame UBO. The textured mesh shader applies Lambert diffuse modulated by a directional shadow map.

**Shadow mapping:** View-frustum-fitted orthographic directional shadow (single split from Sascha `shadowmappingcascade`). Two passes per frame:

1. **Shadow pass** — depth-only into `2048×2048` (`ShadowMap::k_default_size`). Slope-scaled polygon offset on casters.
2. **Main pass** — PCF with slope-scaled depth bias + normal offset on receivers (`triangle.slang`).

`directional_light_view_projection()` fits an ortho box to the camera view wedge up to `max_distance` (default **50**). Geometry outside that wedge is not shadowed. `min_ortho_extent` (default **12**) keeps the map from collapsing when the camera is close.

Tune via `Scene::shadow_settings()` or `ENGINE_SHADOW_DISTANCE`.

**Descriptors:** Set 0 holds per-frame UBO, texture array, and shadow map; set 1 holds the table texture for the current draw (Sascha multi-set pattern). This avoids allocating duplicate frame/shadow bindings for every table texture.

Final shading: `baseColor * (ambient + diffuse * shadow)` with standard Lambert `max(N·L, 0)`.

**Textures:** Each `MeshInstance` uses `texture_source` + `texture_index`:
- `TextureSource::Table` — array of textures (separate images, Sascha `descriptorsets/`)
- `TextureSource::ArrayLayer` — layer index into `Scene::texture_array_layer_paths()` (same-size `texture2DArray`, voxel-atlas path)

**Draw order:** Instances carry a `RenderLayer` (`Background`, `Opaque`, `Transparent`, `Overlay`). The renderer sorts by layer, then pipeline, then mesh. Draw `Background` first for skyboxes; a dedicated no-depth pipeline comes later for true “always behind” geometry (Godot-style depth-off backgrounds).

**Platforms:** SDL3 + Vulkan targets Linux and Windows with the same code. macOS requires MoltenVK — the engine enables `VK_KHR_portability_enumeration` and `VK_KHR_portability_subset` when available. Android/iOS are possible via SDL mobile targets but not set up yet.

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
- `+1` per GiB of device-local VRAM (simple_engine pattern)

Both score terms are required. Do not pick the first suitable device. Override with `-g <index>`, list with `-gl`, or interactively pick when multiple GPUs are present and stdin is a TTY (`--pick-gpu` forces the prompt).

**Queue families:** first-match for graphics/compute/transfer/present, then prefer a unified family that supports graphics, compute, transfer, and present together.

**Swapchain:** prefer `FIFO` (vsync to display refresh); fall back to `Mailbox` if unavailable; recreate with `oldSwapchain`; debounce window resize (~100 ms); wait for non-zero extent before recreating; recreate depth image on swapchain resize (Khronos ch. 27, HowToVulkan, Sascha `trianglevulkan13`).

**Depth:** format selection tries `D32_SFLOAT`, then `D32_SFLOAT_S8_UINT`, then `D24_UNORM_S8_UINT`; device-local optimal image; clear to 1.0; `CompareOp::eLessOrEqual` (Sascha, HowToVulkan).

**Synchronization:** signal semaphores at `eColorAttachmentOutput`, not `eAllGraphics`. Dynamic rendering color attachments use `eColorAttachmentOptimal`.

**Validation:** Debug builds require `VK_LAYER_KHRONOS_validation` and fail fast if it is missing. Release builds run without validation. No validation output on startup usually means the layer is active and found nothing wrong — debug builds print `Vulkan validation: enabled` to confirm.

**MSAA:** Configured at startup via `MsaaSettings` (`render_settings.hpp`). Default is enabled with device maximum samples. When disabled (`samples == 1`), the renderer draws directly to the swapchain with no multisampled color image or resolve pass. Quick override: `ENGINE_MSAA=0` (off), `ENGINE_MSAA=4`, or `ENGINE_MSAA=8` (clamped to GPU support). Runtime toggle can be added later; changing MSAA today requires restart.

**Swapchain surface:** probe `compositeAlpha`, prefer identity `preTransform`, and add `TRANSFER_DST` usage when supported (Sascha, Vulkan-Samples).

**Pipeline cache:** Created once per device; all graphics pipelines reuse it so later pipeline variants compile faster (Sascha pattern).

**Extensions:** verify required instance and device extensions exist before create, not only at link time.

**Helper order:** define lower-level helpers (e.g. `has_name`) before helpers that call them (e.g. `supports_all_names`).
