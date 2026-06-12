# Notes for AI assistants

Read this and `README.md` before large renderer changes.

## Architecture

- **Engine** (`VCE::Engine`): header-only library + compiled `vce_gltf_impl` + Slang SPIR-V. `vulkan_context.hpp` owns init/frame loop; `pass_recorder.hpp` records shadow/main passes.
- **App** ([necromyth-engine-demo](https://github.com/Tristan367/necromyth-engine-demo)): demo client only (fly camera, `demo_scene.cpp`). Game logic does not belong in the engine repo.
- **Shaders**: Slang → SPIR-V via `slangc`, `-profile spirv_1_4`. Runtime Vulkan **1.3** (dynamic rendering, sync2). Do not require 1.4.

## Shadows (current)

**Fast path** (`DirectionalLightShadowSettings`, default): single ortho cascade, `CameraFootprint` focus, texel snap **on**, **bilinear** compare fetch, **Pcf3x3** filter. Filter ladder: `Hard` → `Pcf3x3` → (future PCSS / CSM). `max_distance` default **100**; `ortho_half_extent` default **127** world units (~254m box).

**Alpha policy:** cutout or alpha-to-coverage only — no true alpha blend pass. `RenderLayer::AlphaTested` for ordered cutout/A2C draws.

**Future (optional):** 2-cascade CSM as a **second shadow pipeline family** (texture array, two depth passes). Single-cascade fast path stays default. ViewWedge focus may be removed if CSM + footprint is enough.

- `shadow_utils.hpp`: matrix + snap logic
- `shaders/lib/shadow.slang`: separate fragment entry points per filter; optional **coverage edge fade** (`coverage_fade`, `coverage_fade_uv_width` → UBO `shadowFadeParams`)
- `pipeline_registry.hpp`: `alpha_to_coverage` enabled on A2C pipelines when MSAA > 1
- `frame_overlay.hpp`: optional app callback recorded after the main pass (ImGui lives in the app)
- Shadow pass polygon offset: `k_shadow_depth_bias_*`

## References (`~/Projects/vulkan examples/`)

| Repo | When to use |
|------|-------------|
| **Vulkan/** (Sascha) | Patterns we already follow: dynamic rendering, descriptors, shadowmappingcascade, glTF |
| **Vulkan-Tutorial** | Feature order; `simple_engine` layout |
| **Vulkan-Guide** | Concept explanations |
| **Vulkan-Samples** | Khronos official samples (sync, extensions) |
| **HowToVulkan** | SDL3 + resize edge cases |
| **godot** | Production renderer behavior (directional shadows, not copy-paste) |
| **VulkanDemos** | RTX demos — out of scope until RT |

**Two-repo rule** (from README): adopt a pattern only if two references agree, or one is clearly authoritative for that feature.

## Known follow-ups

1. Split `vulkan_context.hpp` further (init vs resources) if it grows again.
2. **2-cascade CSM** — separate pipeline from single-cascade.
3. glTF skinning / animation (Sascha `gltfskinning`).

## Do not

- Use Sascha **`shadowmapping`** (perspective point light) as directional shadow authority.
- Commit without user asking.
- Add PBR/normal maps unless direction changes.
