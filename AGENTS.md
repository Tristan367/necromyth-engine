# Notes for AI assistants

Read this and `README.md` before large renderer changes.

## Architecture

- **Engine** (`VCE::Engine`): header-only library + compiled `vce_gltf_impl` + Slang SPIR-V. `vulkan_context.hpp` owns init/frame loop; `pass_recorder.hpp` records shadow/main passes.
- **App** ([necromyth-engine-demo](https://github.com/Tristan367/necromyth-engine-demo)): demo client only (fly camera, `demo_scene.cpp`). Game logic does not belong in the engine repo.
- **Shaders**: Slang → SPIR-V via `slangc`, `-profile spirv_1_4`. Runtime Vulkan **1.3** (dynamic rendering, sync2). Do not require 1.4.

## Shadows (current)

**Fast path** (`DirectionalLightShadowSettings`, default): **dual** cascades, camera footprint ortho, **texel snap always on**, **bilinear** compare fetch, **Pcf3x3** filter, cascade blend **3** m, coverage edge fade. Single cascade: `ortho_half_extent` **64**; dual far footprint **127**; `max_distance` **100** (dual split only).

**Dual cascade** (default, startup-only): depth **texture array** (2 layers), two shadow depth passes, separate textured pipeline entries (`*Csm2`), band-limited split blend. Single-cascade: `ENGINE_SHADOW_CASCADES=1`.

**Startup-only in `VulkanContext`:** `filter_mode`, `point_shadow_filter`, `cascade_mode`, map resolution/layer count. Runtime on `Scene::shadow_settings()`: coverage fade, blend width.

**Alpha policy:** cutout or alpha-to-coverage in the main pass only — no true alpha blend. Cutout/A2C meshes cast **opaque** silhouettes in the VS-only shadow pass; alpha-threshold shadow discard is an optional follow-up.

- `shadow_utils.hpp`: footprint matrices, texel snap, cascade splits, `effective_shadow_settings()`
- `shaders/lib/shadow.slang`: single/dual visibility, band-limited blend, PCF
- `pass_recorder.hpp`: shadow pass barriers (all array layers), per-cascade dynamic rendering
- Caster bias: `k_shadow_depth_bias_*`; receiver bias: `shadow.slang`

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

1. **glTF skinning / animation** (Sascha `gltfskinning`) — next major feature; design for physics-driven root transforms and optional ragdoll/hit-reaction later.
2. **Physics wrapper** (e.g. Jolt) — sync rigid bodies to `MeshInstance` transforms; collision rig parented to skeleton for headshots.
3. Alpha-threshold shadow discard (optional; opaque silhouettes are fine for now).
4. Split `vulkan_context.hpp` further if it grows again.
5. Hard cleanup pass after animation lands (dead abstractions).

## Engine vs demo boundaries

**Engine stock:** camera, directional light + shadows, sky cube mesh helper, loaders, renderer. **Not engine:** floor/quad procedural meshes, game assets, scene layout — those live in [necromyth-engine-demo](https://github.com/Tristan367/necromyth-engine-demo).

## Do not

- Use Sascha **`shadowmapping`** (perspective point light) as directional shadow authority.
- Commit without user asking.
- Add PBR/normal maps unless direction changes.
