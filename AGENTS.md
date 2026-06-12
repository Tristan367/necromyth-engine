# Notes for AI assistants

Read this and `README.md` before large renderer changes.

## Architecture

- **Engine** (`VCE::Engine`): header-only library + compiled `vce_gltf_impl` + Slang SPIR-V. `vulkan_context.hpp` owns init/frame loop; `pass_recorder.hpp` records shadow/main passes.
- **App** ([necromyth-engine-demo](https://github.com/Tristan367/necromyth-engine-demo)): demo client only (fly camera, `demo_scene.cpp`). Game logic does not belong in the engine repo.
- **Shaders**: Slang → SPIR-V via `slangc`, `-profile spirv_1_4`. Runtime Vulkan **1.3** (dynamic rendering, sync2). Do not require 1.4.

## Shadows (current)

**Fast path** (`DirectionalLightShadowSettings`, default): single ortho cascade, `CameraFootprint` focus, texel snap **on**, **bilinear** compare fetch, **Pcf3x3** filter, **coverage edge fade** (`coverage_fade_uv_width`, 0 = hard edge). Filter ladder: `Hard` → `Pcf3x3`. `max_distance` default **100**; `ortho_half_extent` default **127**.

**Dual cascade** (`ENGINE_SHADOW_CASCADES=2`, startup-only): depth **texture array** (2 layers), two full shadow passes, separate textured pipeline entries (`*Csm2`), split blend between cascades (Godot/Sascha standard). Single-cascade path unchanged.

**Alpha policy:** cutout or alpha-to-coverage only — no true alpha blend pass. `RenderLayer::AlphaTested` for ordered cutout/A2C draws.

- `shadow_utils.hpp`: matrix + snap + cascade splits
- `shaders/lib/shadow.slang`: single vs dual visibility paths; `Sampler2DArray` depth compare
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
2. glTF skinning / animation (Sascha `gltfskinning`).
3. Remove `ViewWedge` if dual CSM + footprint is enough in practice.

## Do not

- Use Sascha **`shadowmapping`** (perspective point light) as directional shadow authority.
- Commit without user asking.
- Add PBR/normal maps unless direction changes.
