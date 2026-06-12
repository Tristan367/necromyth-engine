# Necromyth Engine

Minimal Vulkan 1.3 renderer in modern C++ (`VCE::Engine`).

**Demo client:** [necromyth-engine-demo](https://github.com/Tristan367/necromyth-engine-demo) · **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md) (pull requests welcome)

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
cd ../necromyth-engine-demo && make debug
```

Run the demo from the sibling app repo (`make run` in necromyth-engine-demo).

## Reference Projects

Reference clones live at `/home/tristan/Projects/vulkan examples/`. They are not required to build this project. See also **`AGENTS.md`** for a short guide for AI assistants.

| Repo | Use for |
|------|---------|
| **Vulkan/** (Sascha Willems) | Primary code reference: dynamic rendering (`trianglevulkan13`), descriptors, **shadowmappingcascade** (directional/CSM), glTF loaders |
| **Vulkan-Tutorial** | Step-by-step feature sequence; **`simple_engine`** module split (`VulkanDevice`, `SwapChain`, `Renderer`) |
| **Vulkan-Guide** | Concepts and explanations (companion reading, not copy-paste architecture) |
| **Vulkan-Samples** (Khronos) | Official sample patterns, extensions, sync |
| **HowToVulkan** | SDL3 + Vulkan 1.3 in one file; swapchain/depth resize behavior |
| **godot** | Production renderer behavior (e.g. directional shadow distance, stability) — read, do not port wholesale |
| **VulkanDemos** | RTX / advanced demos — future, not current baseline |

**Two-repo rule:** adopt a technique only when at least two reference projects do it the same way (or one repo does it clearly and a second confirms). Tutorial-only patterns are learning steps, not permanent conventions.

**Shadow authority:** directional shadows → Sascha **`shadowmappingcascade`**. Do **not** use basic **`shadowmapping`** (perspective / orbiting light).

## Project Goals

The goal is to build a small, fast Vulkan engine with AI assistance and open source references.

The engine stays minimal:

- No scripting language
- Probably no editor window
- Shadow mapping, 3D animation, Blender-to-game asset workflow
- Sky shader or traditional skybox mesh
- Khronos tutorial fundamentals: MSAA, mip mapping, texture sampling, depth buffering, model loading

Current renderer modules: `vulkan_context` (frame loop, init), `pass_recorder` (shadow/main pass recording), `scene_gpu` (mesh/texture upload helpers), `vulkan_device`, `swapchain`, `render_settings`, `engine_config`, `scene` (`camera`, `scene`, `mesh_instance`, `render_layer`, `directional_light`, `shadow_utils`, `sky_mesh`), `draw_list`, `mesh_gpu`, `pipeline_id`, `graphics_pipeline`, `pipeline_registry`, `depth_image`, `msaa_color_image`, `shadow_map`, `buffer`, `vertex`, `model_loader`, `gltf_loader`, `texture_image`, `uniform_buffer`, `descriptors`, `image_barrier`, `platform/sdl_window`, `platform/gpu_cli`.

**Compiled sources:** `gltf_loader_impl.cpp` (tinygltf in one TU), `texture_image_stb.cpp` (stb_image in one TU). Everything else is headers.

**No bundled assets.** Meshes, textures, and procedural demo geometry (floor quads, etc.) belong in the demo/game repo.

The renderer avoids unnecessary complexity — no PBR, no normal maps unless direction changes.

**Engine vs game:** This repo is the **engine library** (`VCE::Engine` CMake target). The [necromyth-engine-demo](https://github.com/Tristan367/necromyth-engine-demo) repo is the demo client — fly camera, scene setup, and game logic live there. Link with `add_subdirectory(../necromyth-engine)` (or a sibling checkout) and `target_link_libraries(... VCE::Engine)`.

**Repo split (current):** Renderer + scene/core live here as one library. A separate *renderer-only* repo makes sense when you add a second backend (e.g. Metal). Until then, keep renderer and engine unified to avoid submodule friction.

**Models:** `load_gltf_model()` loads glTF 2.0 static meshes (tinygltf, Sascha-style CPU path). `load_obj_model()` remains for simple `.obj` assets. glTF sidecar textures (`.png`/`.jpg` next to the `.gltf`) resolve via material paths; `.glb` bundles everything into one file.

**Lighting:** `Scene::directional_light()` feeds sun direction, color, intensity, and ambient into the frame UBO. The textured mesh shader applies Lambert diffuse modulated by a directional shadow map.

**Shadow mapping:** Dual-cascade directional shadows (default). Per frame:

1. **Shadow pass(es)** — depth-only into a `2048×2048` 2D array (`ShadowMap::k_default_size`, 1 or 2 layers). Camera-footprint ortho, texel snapping, slope-scaled polygon offset on opaque casters.
2. **Main pass** — 3×3 PCF depth compare by default (startup `Hard` = single tap), receiver bias + normal offset in `shaders/lib/shadow.slang`, optional UV coverage fade at map edges.

Focus: **camera footprint** ortho on camera XZ — stable when rotating (Sascha-style, not view-frustum fit).

**Defaults:** PCF 3×3, bilinear compare fetch, texel snapping on, **dual cascades**, cascade blend **3** m, `max_distance` **100** (split placement only), single `ortho_half_extent` **64**, dual far footprint **127**, coverage fade **0.08**. Startup env: `ENGINE_SHADOW_DISTANCE`, `ENGINE_SHADOW_FILTER`, `ENGINE_SHADOW_POINT_FILTER`, `ENGINE_SHADOW_TEXEL_SNAP`, `ENGINE_SHADOW_FADE_WIDTH`, `ENGINE_SHADOW_CASCADES=1|2` (default **2**).

**Startup-only** (restart to change): `filter_mode`, `point_shadow_filter`, `cascade_mode`. **Runtime:** `texel_snapping`, `coverage_fade_uv_width`, `cascade_blend_range` (dual).

**Dual cascades:** two full shadow depth passes, view-Z split with band-limited cross-fade (dual-samples only inside the blend band). **Single cascade** (`ENGINE_SHADOW_CASCADES=1`): one layer, no blending, smaller footprint.

**Cutout/A2C:** cast **opaque** silhouettes in the shadow pass (VS-only). Optional follow-up: alpha-threshold discard in shadow FS.

**Alpha surfaces:** cutout discard or alpha-to-coverage with MSAA — no true alpha blend pass. Use `RenderLayer::AlphaTested` for ordered foliage/fences.

**Descriptors:** Set 0 holds per-frame UBO, texture array, and shadow map; set 1 holds the table texture for the current draw (Sascha multi-set pattern). This avoids allocating duplicate frame/shadow bindings for every table texture.

Final shading: `baseColor * (ambient + diffuse * shadow)` with standard Lambert `max(N·L, 0)`.

**Textures:** Each `MeshInstance` uses `texture_source` + `texture_index`. Opaque-surface draws also set `MeshAlphaMode` (`Opaque`, `Cutout`, `AlphaToCoverage`); the engine creates a textured pipeline only for alpha modes used in the scene, using the shadow filter from startup settings.
- `TextureSource::Table` — array of textures (separate images, Sascha `descriptorsets/`)
- `TextureSource::ArrayLayer` — layer index into `Scene::texture_array_layer_paths()` (same-size `texture2DArray`, voxel-atlas path)

**Shaders:** Stock GLSL-like Slang sources under `shaders/`; shared helpers live in `shaders/lib/` (`frame_uniforms`, `shadow`, `surface`, etc.) and are `#include`d by entry shaders. Mods can reuse the same includes. Build output is SPIR-V (`slangc`); runtime user-shader compilation is planned, not implemented yet.

**Draw order:** Instances carry a `RenderLayer` (`Background`, `Opaque`, `AlphaTested`, `Overlay`). The main pass sorts by layer → pipeline → texture source → texture index → mesh. The shadow pass re-sorts opaque draws by layer → mesh so consecutive instances share vertex/index bindings. `PassRecorder` tracks bound pipeline, material (set 1), and mesh buffers and skips redundant binds (Sascha `gltfscenerendering` multi-set pattern).

**Platforms:** SDL3 + Vulkan targets Linux and Windows with the same code. macOS requires MoltenVK — the engine enables `VK_KHR_portability_enumeration` and `VK_KHR_portability_subset` when available. Android/iOS are possible via SDL mobile targets but not set up yet.

## Code Organization

Keep the project uncluttered. Avoid hundreds of tiny files, but also avoid one massive source file. Use minimal abstractions that make ownership and rendering flow clear.

Prefer modern C++ and modern Vulkan (`vk::raii::*`). Engine code lives in `.hpp` files; the demo app owns `main`. Use RAII for all resource ownership.

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

**Swapchain:** default `FIFO` (vsync); set `ENGINE_PRESENT=mailbox` to uncap FPS for profiling (falls back to `Immediate` or `FIFO` if mailbox is unavailable). Recreate with `oldSwapchain`; debounce window resize (~100 ms); wait for non-zero extent before recreating; recreate depth image on swapchain resize (Khronos ch. 27, HowToVulkan, Sascha `trianglevulkan13`).

**Depth:** format selection tries `D32_SFLOAT`, then `D32_SFLOAT_S8_UINT`, then `D24_UNORM_S8_UINT`; device-local optimal image; clear to 1.0; `CompareOp::eLessOrEqual` (Sascha, HowToVulkan).

**Synchronization:** signal semaphores at `eColorAttachmentOutput`, not `eAllGraphics`. Dynamic rendering color attachments use `eColorAttachmentOptimal`.

**Validation:** Debug builds require `VK_LAYER_KHRONOS_validation` and fail fast if it is missing. Release builds run without validation. No validation output on startup usually means the layer is active and found nothing wrong — debug builds print `Vulkan validation: enabled` to confirm.

**MSAA:** Configured at startup via `MsaaSettings` (`render_settings.hpp`). Default is enabled, capped at **4×** (not device max). Override: `ENGINE_MSAA=0` (off), `ENGINE_MSAA=2`, `ENGINE_MSAA=4`, or `ENGINE_MSAA=8` (clamped to GPU support). Changing MSAA requires restart.

**Swapchain surface:** probe `compositeAlpha`, prefer identity `preTransform`, and add `TRANSFER_DST` usage when supported (Sascha, Vulkan-Samples).

**Pipeline cache:** Created once per device; all graphics pipelines reuse it so later pipeline variants compile faster (Sascha pattern).

**Extensions:** verify required instance and device extensions exist before create, not only at link time.

**Helper order:** define lower-level helpers (e.g. `has_name`) before helpers that call them (e.g. `supports_all_names`).

## License

[MIT](LICENSE) — use, modify, and distribute freely; keep the copyright notice. Contributions are welcome via pull request; see [CONTRIBUTING.md](CONTRIBUTING.md).
