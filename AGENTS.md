# Notes for AI assistants

Read this and `README.md` before large renderer changes.

## Architecture

- **Engine** (`VCE::Engine`): header-only library + compiled `vce_gltf_impl` + Slang SPIR-V. `vulkan_context.hpp` owns init/frame loop; `pass_recorder.hpp` records shadow/main passes.
- **App** ([necromyth-engine-demo](https://github.com/Tristan367/necromyth-engine-demo)): demo client only (fly camera, `demo_scene.cpp`). Game logic does not belong in the engine repo.
- **Shaders**: Slang → SPIR-V via `slangc`, `-profile spirv_1_4`. Runtime Vulkan **1.3** (dynamic rendering, sync2). Do not require 1.4.

## Shadows (current)

**Fast path** (`DirectionalLightShadowSettings`, default): **dual** cascades, camera footprint ortho, **texel snap always on**, **bilinear** compare fetch, **Pcf3x3** filter, cascade blend **3** m, coverage edge fade. Single cascade: `ortho_half_extent` **64**; dual far footprint **127**; `max_distance` **100** (dual split only).

**Dual cascade** (default, startup-only): depth **texture array** (2 layers), two shadow depth passes, separate textured pipeline entries (`*Csm2`), band-limited split blend. Single-cascade: `ENGINE_SHADOW_CASCADES=1`.

**Startup-only in `VulkanContext`:** `filter_mode`, `point_shadow_filter`, `cascade_mode`, map resolution (via `ENGINE_SHADOW_SCALE` on scene base size). Runtime on `Scene::shadow_settings()`: coverage fade, blend width.

**Profiling knobs (startup env, restart):** `ENGINE_RENDER_SCALE` (main pass resolution), `ENGINE_SHADOW_SCALE` (shadow map resolution), `ENGINE_PRESENT=mailbox` (uncap FPS).

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

## Animation / skinning (implemented)

**Vertex layout:** `MeshVertex` extended with `joint_indices[4]` + `joint_weights[4]` (float[4] each, locations 4-5, `R32G32B32A32_SFLOAT`). Static meshes use `static_attribute_descriptions()` (4 attrs) — no joint fetch overhead for non-skinned draws.

**glTF loader:** reads `JOINTS_0` (uint8/uint16), `WEIGHTS_0`, `model.skins` (inverse bind matrices, joint nodes), `model.animations` (clips, channels, samplers). `SkeletonAsset` stores IBM + joint_nodes + `node_parents` map; `AnimationClip`, `AnimationSampler`, `AnimationChannel` in `animation_types.hpp`. Skinned node transforms NOT baked into primitives (identity instead — bone matrices handle placement).

**GPU path:** `BoneStorageBufferSet` — per-instance, host-visible + host-coherent SSBOs, double-buffered (2 per frame-in-flight). Exact bone-count sizing (not k_max_bones). `BoneTRS`-based sampling with SLERP (quaternion) and LERP (translation/scale). Binary keyframe search (`std::upper_bound`). Pre-built per-node channel maps. Thread-local chain vector (zero heap-alloc per frame).

**Scene API:** `MeshInstance` has `skin_index`, `animation_index`, `animation_time`, `animation_speed`, `animation_loop`. Blending: `next_animation_index`, `blend_factor`, `blend_duration`. `Scene` stores `skeletons_` and `animations_` (add_skeleton/add_animation). `Scene::instance()` returns mutable ref for runtime control.

**Pipeline:** 4 new `PipelineId` variants: `TexturedOpaqueSkinned`, `TexturedCutoutSkinned`, `TexturedAlphaToCoverageSkinned`, `ShadowDepthSkinned` (IDs 5-8). Skinned main pipelines use separate VS SPIR-V (`triangle_skinned.spv`, `vertMainSkinned`) + reuse fragment shaders from `triangle.spv`. Skinned shadow uses dedicated `shadow_depth_skinned.spv` with 3-attr vertex input (pos + joints, no normal/color/tex).

**Descriptor layouts:** `material_skinned_layout_` (set 1: sampler b=0 + SSBO b=1) for main pass skinned; shadow skinned reuses same layout (dummy sampler at b=0, SSBO at b=1). Non-skinned uses `material_layout_` (set 1: sampler b=0 only). Zero overhead for non-skinned scenes — `build_skinned` flag gates skinned pipeline creation.

**Animation blending:** `compute_joint_matrices_blended()` blends at TRS level (lerp T/S, slerp R) before world matrix construction. Single-animation path delegates to blended with `blend_factor = 1.0F`. Crossfade auto-advances `blend_factor += delta / blend_duration`, promotes `next_animation_index` to `animation_index` on completion.

### key files

- `animation_types.hpp`: `SkeletonAsset`, `AnimationClip`, `AnimationSampler`, `AnimationChannel`, `k_max_bones = 128`
- `animation_utils.hpp`: `sample_animation_trs()`, `compute_joint_matrices()`, `compute_joint_matrices_blended()`, `BoneTRS`, `trs_to_mat4()`
- `bone_buffer.hpp`: `BoneStorageBufferSet` (host-visible SSBO per instance)
- `vulkan_context.hpp`: animation update loop in `draw_frame()`, `create_bone_buffers()`
- `gltf_loader.hpp`: `read_joint_accessor()`, `load_skeletons()`, `load_animations()`, `node_parents` map
- `pass_recorder.hpp`: `bind_material()` handles skinned sets, `draw_shadow_mesh()` binds bone SSBO
- `pipeline_id.hpp`: `textured_pipeline(alpha_mode, skinned)`, `is_skinned_pipeline()`
- Shaders: `triangle_skinned.slang`, `shadow_depth_skinned.slang`, `lib/mesh_types_skinned.slang`

### Blender → glTF export (test model)

- **+Y up** on export (glTF default; matches engine).
- Apply transforms; triangulate; export **skin weights** (4 influences max is fine).
- **Animations:** NLA or actions → glTF; name clips clearly (`idle`, `walk`).
- Export **.glb** for simplest path (mesh + rig + clips in one file).
- Drop test asset in demo `assets/models/`; wire one instance in `demo_scene.cpp`.

## Known follow-ups

1. **Physics wrapper** (e.g. Jolt) — after skinning; sync root transform; collision rig on bones later.
2. Alpha-threshold shadow discard (optional; opaque silhouettes are fine for now).
3. Split `vulkan_context.hpp` further if it grows again.
4. Per-skin bone buffers instead of per-instance (shared skeletons).
5. Cubic spline interpolation for glTF animations (rare in practice).

## Engine vs demo boundaries

**Engine stock:** camera, directional light + shadows, sky cube mesh helper, loaders, renderer. **Not engine:** floor/quad procedural meshes, game assets, scene layout — those live in [necromyth-engine-demo](https://github.com/Tristan367/necromyth-engine-demo).

## Do not

- Use Sascha **`shadowmapping`** (perspective point light) as directional shadow authority.
- Commit without user asking.
- Add PBR/normal maps unless direction changes.
