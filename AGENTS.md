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

**Pipeline:** 6 new `PipelineId` variants: `TexturedOpaqueSkinned`, `TexturedCutoutSkinned`, `TexturedAlphaToCoverageSkinned`, `ShadowDepthSkinned`, `PointShadowDepth`, `PointShadowDepthSkinned` (IDs 5-10). Skinned main pipelines use separate VS SPIR-V (`triangle_skinned.spv`, `vertMainSkinned`) + reuse fragment shaders from `triangle.spv`. Skinned shadow uses dedicated `shadow_depth_skinned.spv` with 3-attr vertex input (pos + joints, no normal/color/tex). Point shadow uses `point_shadow.spv` (non-skinned + skinned variants, both VS + FS, multiview).

**Descriptor layouts:** `material_skinned_layout_` (set 1: sampler b=0 + SSBO b=1) for main pass skinned; shadow skinned reuses same layout (dummy sampler at b=0, SSBO at b=1). Non-skinned uses `material_layout_` (set 1: sampler b=0 only). Zero overhead for non-skinned scenes — `build_skinned` flag gates skinned pipeline creation.

**Pose-layer stack (PREFERRED path):** `PoseLayer` (`animation_types.hpp`) = clip + internal A→B crossfade (`xfade_index/time/weight`) + compositing `weight` + optional bone `mask`. `evaluate_pose_layers()` composites an ordered stack per joint: layer 0 = full-body locomotion; higher layers = masked overrides (e.g. upper-body "hold weapon"), blended OVER via `blend_bone_trs(accum, sampled, weight)`. Set `MeshInstance::pose_layers` (non-null, non-empty) and `compute_joint_matrices_for_instance()` routes through the layer evaluator, ignoring all legacy blend/split fields. This is the Godot/Unity model: layers are additive-composite, never mutually-exclusive modes, and every layer crossfades its own transitions (no snapping). `AnimStateMachine` owns the stack (`layers()`), drives layer 0's crossfade, and exposes `add_override_layer()`, `set_override_clip()`, `set_override_weight()` (weight fades smoothly).

**Legacy animation blending (fallback, no `pose_layers`):** `compute_joint_matrices_blended()` blends at TRS level (lerp T/S, slerp R). Crossfade auto-advances `blend_factor += delta / blend_duration`, promotes `next_animation_index` to `animation_index`. **Historical bug:** `next_animation_index`/`blend_factor` were overloaded for BOTH crossfade target AND per-bone split's clip_b — they collided and forced instant switches. Do not reintroduce that aliasing; use pose layers instead.

**Legacy animation split (fallback):** `MeshInstance::secondary_joints` + `secondary_animation_index` — `compute_joint_matrices_split()` hard-picks clip_a or clip_b per joint (no blend). Still used by demo model2. Prefer a masked `PoseLayer` for new work.

### key files

- `animation_types.hpp`: `SkeletonAsset`, `AnimationClip`, `AnimationSampler`, `AnimationChannel`, `k_max_bones = 128`, `BoneTRS`, `HitboxAttachment`, `HitboxShape`, `BodyColliderDef`
- `animation_utils.hpp`: `evaluate_pose_layers()` (preferred), `sample_animation_trs()`, `compute_joint_matrices()`, `compute_joint_matrices_blended()`, `compute_joint_matrices_split()`, `BoneTRS`, `trs_to_mat4()`
- `animation_state_machine.hpp`: `AnimStateMachine` owns `layers()` (`PoseLayer` stack), base-layer crossfade + masked override layers
- `bone_buffer.hpp`: `BoneStorageBufferSet` (host-visible SSBO per instance)
- `vulkan_context.hpp`: animation update loop in `draw_frame()`, `create_bone_buffers()`, split check before blended/single path
- `gltf_loader.hpp`: `read_joint_accessor()`, `load_skeletons()` (stores `joint_names` from glTF node names), `load_animations()`, `node_parents` map
- `pass_recorder.hpp`: `bind_material()` handles skinned sets, `draw_shadow_mesh()` binds bone SSBO
- `pipeline_id.hpp`: `textured_pipeline(alpha_mode, skinned)`, `is_skinned_pipeline()`
- Shaders: `triangle_skinned.slang`, `shadow_depth_skinned.slang`, `lib/mesh_types_skinned.slang`

### Blender → glTF export (test model)

- **+Y up** on export (glTF default; matches engine).
- Apply transforms; triangulate; export **skin weights** (4 influences max is fine).
- **Animations:** NLA or actions → glTF; name clips clearly (`idle`, `walk`).
- Export **.glb** for simplest path (mesh + rig + clips in one file).
- Drop test asset in demo `assets/models/`; wire one instance in `demo_scene.cpp`.

## Hitbox / bone-attached colliders (implemented)

**Two collider types per model** (defined in `<model_name>.json` sidecar next to `.glb`):

| | Body Collider | Hitboxes |
|---|---|---|
| Jolt body type | Dynamic/Kinematic, `IsSensor = false` | Kinematic, `IsSensor = true` |
| Object layer | `kMoving` | `kHitbox` (new) |
| Purpose | Physics: ground, rigidbodies, character push | Detection: raycasting, weapon sweeps, per-bone queries |
| Raycast participation | Only when no hitboxes configured | Always |
| Count per model | 1 | 0..N |
| Transform source | Skeleton root (from scene instance model) | Per-bone world transform (from `build_world_matrices` bone_worlds) |

**Layer design:** `kHitbox` = ObjectLayer 2, BroadPhaseLayer 2. Layer pair filter returns false for any pair involving kHitbox (sensor-only, no collision). BroadPhaseLayer still includes kHitbox for raycast/spatial queries.

**Data model** (`animation_types.hpp`): `HitboxShape` enum (Box/Sphere/Capsule), `HitboxAttachment` (name, joint_index, shape, offset, rotation, half_extent, half_height), `BodyColliderDef` (shape, half_height, radius, half_extent, offset). `SkeletonAsset` gets `joint_names`, `hitboxes`, optional `body_collider`. `find_joint_index(name)` helper.

**HitboxManager** (`physics/hitbox_manager.hpp`): Creates kinematic sensor bodies on `kHitbox` layer from `HitboxAttachment` definitions. `update()` syncs body transforms from bone world transforms each frame. Exposes `find_name(BodyID)` for hit-to-hitbox lookup.

**Bone world transforms:** `build_world_matrices()` accepts optional `out_bone_worlds` parameter — stores `inverse_skin_node_transform * world` (before IBM multiplication) for hitbox placement. Functions: `compute_joint_matrices()`, `compute_joint_matrices_blended()`, `compute_joint_matrices_split()` all propagate `out_bone_worlds`.

**JSON format** (`<model>.json`):
```json
{
  "body": {"shape": "capsule", "half_height": 0.4, "radius": 0.5, "offset": [0,0.4,0]},
  "hitboxes": [
    {"name": "Head", "bone": 4, "shape": "sphere", "offset": [0,0,0], "radius": 0.25}
  ]
}
```
`"bone"` accepts string name or integer joint index.

**Tapered shape CoM:** Jolt centers `TaperedCylinderShape` and `TaperedCapsuleShape` on center-of-mass (not geometric center). Use `Shape::GetCenterOfMass()` and rotate offset by body rotation for correct visual alignment. See `sync_body_to_instance` in demo.

## Roadmap (planned features, in priority order)

1. **GPU particle system** — vertex-shader billboard quads with lifetime/velocity/color-over-life/gravity
2. **Bone attachment system** — attach objects (weapons, particles, lights) to skeleton bones with hitbox support

## Implemented (formerly planned)

- **Point + spot lights** — forward shading with attenuation, D32 depth cubemap shadow atlas with hardware PCF. 10 lights @ 190 FPS on RTX 3060.
- **Audio engine** — miniaudio integration, positional 3D audio, looping music, multi-track.

## Known knowns (deferred, not urgent)

1. Alpha-threshold shadow discard (optional; opaque silhouettes are fine for now).
2. Split `vulkan_context.hpp` further if it grows again.
3. Per-skin bone buffers instead of per-instance (shared skeletons).
5. Shared skeleton hitbox bodies (currently per-instance).

## Jolt CharacterVirtual — Critical Properties

**These are easy to confuse. Read this before touching the character controller.**

### `mMass` vs `mMaxStrength` — completely separate paths

| Property | What it controls | Jolt source |
|----------|-----------------|-------------|
| `mMass` | **Downward gravity crush** on the body underfoot (`mGroundBodyID`). Applies `mMass * g * dt` as a downward impulse at the off-center contact point. Only fires when standing ON something. | `CharacterVirtual.cpp:1474` |
| `mMaxStrength` | **Horizontal pushing** when walking into dynamic bodies. Clamps the impulse per tick: `max_impulse = mMaxStrength * dt`. | `CharacterVirtual.cpp:799` |

**`mMass` does NOT affect pushing. `mMaxStrength` does NOT affect standing weight.** They are orthogonal code paths. Do not use `mMass` to tune pushing, and do not use `mMaxStrength` to stop cubes from tumbling when you stand on them.

- To stop cubes from tumbling underfoot: **`mMass = 0`**
- To control how hard the character pushes props: **tune `mMaxStrength`** (default 100 N)

### `mCanReceiveImpulses` vs `mCanPushCharacter`

| Property | Direction |
|----------|-----------|
| `mCanReceiveImpulses` | Character → Body (character pushes the body) |
| `mCanPushCharacter` | Body → Character (body pushes the character) |

Both default to `true`. Set via `CharacterContactSettings` in a `CharacterContactListener::OnContactAdded/OnContactPersisted` callback.

### `mInnerBodyShape`

An optional inner kinematic body for sensor/raycast presence. Set to `nullptr` to disable (no force transfer). Separate from the main collision shape.

### Mesh collision edge jitter

`mEnhancedInternalEdgeRemoval=true` is required but NOT sufficient. Jolt detects internal edges **topologically** — two triangles must share the same vertex indices. glTF models export split vertices (same position, different index per face). Vertex welding (collapsing coincident positions to shared indices) is required to make enhanced edge removal work. `create_static_mesh` in `physics_world.hpp` does this at 0.1mm grid resolution.

### Frame interpolation

Dynamic body transforms must be interpolated (`prev/curr + lerp at accumulator alpha`), not written directly in `tick()`. Otherwise bodies snap at the physics tick rate while the camera glides. Same alpha for camera and all bodies. Character position uses `render_state_.prev/curr` with the same interpolation.

### Accumulator overflow

When capping ticks per frame (max 2), drain the overflow: `accumulator = fmod(accumulator, k_fixed_dt)`. Otherwise leftover carries forward and creates permanent latency.

### Reference

- Jolt sample: `/home/tristan/opt/JoltPhysics/Samples/Tests/Character/CharacterVirtualTest.cpp`
- `HandleInput()` lines 112-187 — canonical velocity formula
- `OnContactSolve()` lines 386-394 — anti-slide callback for idle-on-slope stability
- Jolt docs: https://jrouwe.github.io/JoltPhysics/index.html#character-controllers

## Engine vs demo boundaries

**Engine stock:** camera, directional light + shadows, sky cube mesh helper, loaders, renderer. **Not engine:** floor/quad procedural meshes, game assets, scene layout — those live in [necromyth-engine-demo](https://github.com/Tristan367/necromyth-engine-demo).

## Do not

- Use Sascha **`shadowmapping`** (perspective point light) as directional shadow authority.
- Commit without user asking.
- Add PBR/normal maps unless direction changes.

## Deferred refactors (design decisions needed)

1. **Graphics pipeline dedup** — ~~`graphics_pipeline.hpp` has 3 functions (`create_graphics_pipeline` x2 + `create_depth_only_graphics_pipeline`) sharing ~90% setup code. ~200 lines of triplication. Extracting shared logic would touch all pipeline paths (main, shadow, point, particle) — needs thorough re-testing.~~ Done. Extracted `detail::build_graphics_pipeline`. File: 425 → 275 lines.

2. **Two-step init → RAII** — `Swapchain`, `DepthImage`, `RenderColorImage`, `ShadowMap`, `TextureImage`, `TextureArray`, `ParticleSystem`, `BoneStorageBufferSet`, `UniformBufferSet` all default-construct empty, then require `create()`. Moving to RAII constructors prevents use-before-init bugs. Some (Swapchain) need `recreate()` anyway for resize.

3. **`MeshInstance::pose_layers` raw pointer** — pointer to `AnimStateMachine::layers()` owned by user. If the state machine moves/relocates/reallocates, the pointer dangles silently. Options: `std::shared_ptr`, `std::reference_wrapper`, or keep raw with debug assert.

4. **`PipelineId` insertion safety** — adding a pipeline in the middle of the enum breaks `is_textured_surface_pipeline(id >= E_First && id <= E_Last)` and `is_skinned_pipeline`. Options: explicit check sets (`std::unordered_set<PipelineId>`) or sentinel `Count` values.

5. **Bone buffer instance mapping** — `bone_instance_index` is computed sequentially. If instances are reordered (insertion/removal in the middle), bone buffers silently map to wrong skeletons. Slot-map or stable `bone_handle` per instance would fix both this and dead-instance sequencing issues at the root.

6. **`TextureImage/TextureArray::create_sampler` identical** — ~~character-for-character duplicate (~20 lines each). Extract to `detail::create_mipmapped_sampler`.~~ Done.

7. **`scene_uses_alpha_to_coverage` startup-only** — checked once in `VulkanContext` ctor. New A2C meshes added via `sync_scene` won't enable MSAA. Could be checked in `sync_scene` or deferred to a `render_settings` flag.
