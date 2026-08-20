# Notes for AI assistants

Read this and `README.md` before large renderer changes.

## Verify your work before handing it back

`make` in this repo compiles **every** engine header — once in a single TU
(catches cross-header conflicts) and once per header on its own (catches headers
that only compile because a consumer included their dependencies first). `make
test` additionally runs the CPU-side invariants in `tests/`. Neither needs a GPU.

```bash
make        # compiles all engine headers + shaders
make test   # the above, then ctest
```

Both are warning-free, C++ **and** Slang. Keep them that way -- a build with a
handful of accepted warnings is a build where a new one goes unnoticed. Slang
warning 39001 (`explicit binding overlap`) is disabled by id in CMakeLists,
because a Texture and its Sampler sharing one `[[vk::binding]]` is how a combined
image sampler is written; nothing else is suppressed.

Before this existed, `make` built three third-party TUs and reported success
while the entire header-only engine went unchecked — breakage surfaced as a
build error in the demo, or at runtime in the game. If you add a header it is
picked up automatically (CONFIGURE_DEPENDS glob); keep it self-contained.

A GPU-side change still needs `cd ../necromyth-engine-demo && make debug` and an
actual run with validation layers on.

## Point light shadows: cubemap, not dual paraboloid

The cubemap is the right choice; do not go back to dual paraboloid.

DP's seam is not a bug that can be fixed. DP projects vertices through a
non-linear warp, and the rasteriser then interpolates *linearly* across each
triangle, so the error is worst where a triangle spans the hemisphere boundary --
and it scales with triangle size. Engines that ship DP hide it with heavy
tessellation and clamping, which costs back whatever DP saved. A cubemap has no
seam because each face is an ordinary perspective projection.

Rendered with multiview (`viewMask = 0b111111`), all six faces come from one draw
per light, which is why the cubemap is also competitive on speed.

**Shadow slots are explicit.** A light's cubemap is addressed by a slot assigned
per frame in `assign_shadow_slots()` (`scene/shadow_assignment.hpp`), written into
the light buffer, and read by the shader. It is NOT the light's index in
`Scene::point_lights()`. Three things depend on agreeing about it -- the light
buffer, the point-shadow SSBO, and the shadow pass -- so all three take the
assignment. Note the capacity is in **cubes**, not image array layers; those
differ by six.

Lights whose sphere of influence misses the camera frustum get no slot and are
not rendered: attenuation is zero past `range`, so nothing they light is visible.

## Profiling

`ENGINE_PROFILE=1` prints a per-pass CPU/GPU breakdown every 120 frames.
`VulkanContext::profile_report()` returns the same text for a debug overlay, and
`gpu_profiler()` / `cpu_profiler()` expose the raw per-zone numbers.

GPU zones come from real timestamps written into the command buffer and read back
only after that frame slot's fence has been waited on, so collection never
blocks. CPU zones are host timers around each phase of the frame.

Read it like this:

- **`acquire swapchain image` large, everything else small** — you are waiting on
  the display, not rendering. At 60 Hz FIFO this is the idle remainder of the
  frame and is exactly what you want to see. Profile with `ENGINE_PRESENT=mailbox`
  to remove it and see the real cost.
- **GPU total high** — look at which pass dominates before touching anything.
- **`record commands` high with a low GPU total** — you are CPU-bound on draw
  submission; fewer draw calls (batching, culling) will help, a cheaper shader
  will not.

Measure before and after, in mailbox mode, and quote the numbers. Eyeballing the
frame rate cannot distinguish a 5 ms pass from a 0.1 ms one.

## Scaling rule: the demo is not the workload

The demo builds its scene once at startup, gives every skinned instance a pose
stack, and never removes anything. The target game is a streaming voxel world
(8-chunk render distance, ~289 chunk meshes plus LOD tiers, continuous remeshing,
hordes of skinned characters, dimension swaps that replace the whole world).

Two silent skinning bugs and a per-frame GPU stall lived here for a long time
purely because the demo never exercised the paths the game will. When adding a
feature, size it for the game: many instances, continuous churn, and removal --
not for the one-of-each case the demo happens to show.

## Instanced draws

Everything per-object -- model matrix, texture layer, bone palette base -- lives
in a per-frame storage buffer (`instance_buffer.hpp`, set 0 binding 8), NOT in
push constants. The vertex shader reads record `pushConstants.instanceBase +
SV_InstanceID`, so draws that share a pipeline, mesh and material collapse into
one `drawIndexed` with `instanceCount > 1`. Measured: 255 skinned characters
sharing a mesh and texture render in **one** draw call.

Rules, all of which exist because breaking them renders the wrong objects rather
than failing:

- **Cull before batching.** A draw rejected in the middle of a run leaves a gap
  in the instance range, and the shader reads that range contiguously. Every
  pass selects survivors into a scratch list first, then batches.
- **Batch with `draws_can_batch()`** (`draw_list.hpp`) and iterate with
  `for_each_batch()`. It requires adjacent instance records, not just matching
  state.
- **The main and shadow lists sort differently and get separate record ranges.**
  Each is contiguous within itself; a shadow caster's matrix is written twice, on
  purpose.
- Push constants are 16 bytes and carry only `instanceBase`, the shadow cascade,
  and the point light index. Do not put per-object data back in them.
- Bone slices are chosen by index (`bone_base` in the instance record), not by a
  dynamic descriptor offset. A dynamic offset is fixed for a whole bind, so it
  cannot vary across the instances of one instanced draw.

`tests/stress_skinned_horde.cpp` checks both halves: that the horde batches, and
that the characters still hold distinct poses. A wrong bone base renders happily
and just looks like a horde marching in lockstep, so the test asserts the poses
actually differ.

## Instance handles

`Scene::add_instance()` returns an `InstanceHandle` (index + generation), not a
bare index. Instance slots are recycled like mesh slots, so an index alone is not
a safe reference: after a despawn the same index belongs to a different entity,
and code holding the old one would silently move, animate or delete the wrong
thing.

- `scene.instance(handle)` throws on a stale handle.
- `scene.try_instance(handle)` returns null -- use it wherever the entity may
  legitimately be gone (a projectile whose target despawned mid-flight).
- `BoneAttachment::target_instance` is a handle and is generation-checked every
  frame, so a dropped weapon cannot start dragging an unrelated instance around.

`MeshInstance::pose_layers` and `joint_overrides` are `shared_ptr`, not raw
pointers into caller storage. Hand them `AnimStateMachine::shared_layers()`. The
state machine can then be moved or relocated freely (a `std::vector` of them
growing is fine), and if it is destroyed the character freezes in its last pose
instead of reading freed memory.

Do not store `handle.index` on its own and index `instances()` with it later.
Iterating `instances()` directly is fine -- that is what the renderer does -- but
a reference kept *across frames* has to be a handle.

## Mesh streaming

`Scene` mesh storage is slot-based, not append-only:

| | |
|---|---|
| `add_mesh` | takes a free slot if there is one, otherwise grows |
| `update_mesh` | replaces geometry in place, index stays valid (remesh / LOD swap) |
| `remove_mesh` | frees the slot for reuse, drops the CPU geometry |

Each slot carries a `revision`, bumped on every content change. The renderer
records the revision it last uploaded and re-uploads on mismatch, so create,
update and remove are one uniform path (`sync_scene_meshes`).

**Mesh changes must not stall the GPU.** Mesh buffers are bound directly
(`bindVertexBuffers`), never through a descriptor set, so changing geometry needs
no descriptor rebuild and no `wait_idle`. Superseded buffers go to
`DeferredDelete` and are freed once every in-flight frame has cycled past them.
Only texture and bone-buffer changes -- which do rewrite descriptor sets -- still
take an idle, and those are rare.

`tests/stress_mesh_streaming.cpp` is the GPU-side check: it churns slots for
hundreds of frames under the validation layers and asserts slot capacity tracks
the working set rather than total churn. It needs a GPU and a display, so it is
built but not run by `ctest`:

```bash
VCE_STRESS_TEXTURE=path/to/any.png ./build/vce_stress_mesh_streaming 600
```

## The bone-slot invariant (read before touching skinning)

Four separate sequential walks over `Scene::instances()` must produce the *same*
slot numbering, because each one indexes the same per-instance bone resources:

| Walker | File |
|---|---|
| bone buffer creation | `create_bone_buffers()` in `vulkan_context.hpp` |
| skinned descriptor set count | `count_skinned_instances()` in `vulkan_context.hpp` |
| per-frame joint-matrix upload | `draw_frame()` in `vulkan_context.hpp` |
| `DrawCommand::bone_instance_index` | `build_draw_list()` in `draw_list.hpp` |

All four call **`instance_uses_skinning()`** (`scene/scene.hpp`) and advance on
exactly that predicate. Do not inline a variant of the check, and do not add an
extra condition to one site. When two of them disagreed, instances silently
rendered with another model's pose — no crash, no validation error.

Specifically: an instance without `pose_layers` **still owns a slot**. It renders
in bind pose (`compute_joint_matrices_for_instance()` handles the null case).
Skipping it in the upload loop shifts every later instance onto the wrong buffer.

Second half of the same invariant: the set-1 descriptor cache in
`PassRecorder::bind_material` keys on `MaterialKey`, which **must** include
`bone_instance_index`. Without it, same-texture skinned instances are adjacent
after the draw-list sort and all collapse onto the first one's bone matrices in
the main pass, while the shadow pass (which binds unconditionally) uses the
correct ones — the symptom is a shadow that animates while the mesh does not.

`tests/test_scene_invariants.cpp` covers both halves.

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

**GPU path:** `BonePalette` — ONE shared host-visible buffer for the whole scene, laid out `[frame][slot]`, bound through a `eStorageBufferDynamic` descriptor. A draw selects its instance's slice with a dynamic offset, so the descriptor set only varies by texture: a horde of 250 costs the same descriptors as one character, and spawning or despawning needs no reallocation, no descriptor rewrite and no device stall. Fixed per-slot stride of `k_max_bones` matrices, rounded up to `minStorageBufferOffsetAlignment` — wasteful for small skeletons, but an offset is then pure arithmetic with no table that can fall out of step with slot assignment. Capacity is `EngineConfig::max_skinned_instances` (default 256); overflow logs once and renders the extras in bind pose. `BoneTRS`-based sampling with SLERP (quaternion) and LERP (translation/scale). Binary keyframe search (`std::upper_bound`). Pre-built per-node channel maps. Thread-local chain vector (zero heap-alloc per frame).

**Scene API:** `MeshInstance` has `skin_index`, `animation_index`, `animation_time`, `animation_speed`, `animation_loop`. Blending: `next_animation_index`, `blend_factor`, `blend_duration`. `Scene` stores `skeletons_` and `animations_` (add_skeleton/add_animation). `Scene::instance()` returns mutable ref for runtime control.

**Pipeline:** 6 new `PipelineId` variants: `TexturedOpaqueSkinned`, `TexturedCutoutSkinned`, `TexturedAlphaToCoverageSkinned`, `ShadowDepthSkinned`, `PointShadowDepth`, `PointShadowDepthSkinned` (IDs 5-10). Skinned main pipelines use separate VS SPIR-V (`triangle_skinned.spv`, `vertMainSkinned`) + reuse fragment shaders from `triangle.spv`. Skinned shadow uses dedicated `shadow_depth_skinned.spv` with 3-attr vertex input (pos + joints, no normal/color/tex). Point shadow uses `point_shadow.spv` (non-skinned + skinned variants, both VS + FS, multiview).

**Descriptor layouts:** `material_skinned_layout_` (set 1: sampler b=0 + SSBO b=1) for main pass skinned; shadow skinned reuses same layout (dummy sampler at b=0, SSBO at b=1). Non-skinned uses `material_layout_` (set 1: sampler b=0 only). Zero overhead for non-skinned scenes — `build_skinned` flag gates skinned pipeline creation.

**Ownership:** `AnimStateMachine` holds its pose stack in a `shared_ptr`; give
`MeshInstance::pose_layers` the result of `shared_layers()`. Never `&layers()`.

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

1. **Impostor (Y-billboard) LOD** — see below. Instanced draws (its prerequisite) are done.
2. **GPU particle system** — vertex-shader billboard quads with lifetime/velocity/color-over-life/gravity
3. **Bone attachment system** — attach objects (weapons, particles, lights) to skeleton bones with hitbox support

## Impostor / Y-billboard LOD (design, not yet implemented)

Goal: render distant animated characters (zombies) as pre-rendered billboards
instead of skinned meshes, so the horde stops costing skinning and geometry.
Target is on the order of 10k characters as a small number of draw calls.
Generalises to any model, and is the intended basis of an automated LOD step.

**Offline bake** (a tool, not part of the runtime):

- Input: a skinned glTF plus the clips to bake.
- Render with an orthographic camera at a fixed elevation, rotating around Y in
  `N` steps (8 or 16; 8 is usually enough for ground-level viewing), sampling
  each clip at `M` uniform times (8-16 per loop).
- Output: a texture array layer per `(angle, frame)` pair, plus a small manifest
  (clip names, `N`, `M`, clip lengths, world-space quad size).
- Bake albedo with alpha cutout. Optionally a second layer set with a baked
  world-space normal, if the impostors need to respond to the sun rather than
  carrying flat baked lighting.

**Runtime**:

- A dedicated pipeline, closest in shape to the existing particle billboard path.
- **Y-billboard**: the quad yaws to face the camera but keeps world up, so it
  does not tip when the camera looks down. Build it in the vertex shader from the
  instance position, world up, and the camera's XZ-projected right vector -- NOT
  the camera's true right, which is what makes a billboard roll.
- Layer selection, per instance, in the vertex shader:
  - `angle = round(N * frac((atan2 of camera-to-instance in XZ - instance yaw) / 2pi)) % N`
  - `frame = floor(animTime / clipLength * M) % M`
  - `layer = atlasBase + angle * M + frame`
  where `atlasBase` selects the character variant, so different zombies with
  different textures and different animations coexist in one draw.
- Per-instance data (position, yaw, scale, `atlasBase`, clip, `animTime`) lives in
  the same per-instance buffer instancing uses, so the whole horde is one
  instanced draw.

**Notes and pitfalls**:

- One baked elevation looks wrong from directly above. Fine for a ground game;
  bake a second elevation row if the camera ever gets height.
- Shadows: impostors casting shadows need the same alpha-cutout treatment in the
  shadow pass, or they cast rectangles.
- Memory is the real constraint: `N x M x resolution`. 16 angles x 16 frames at
  128x128 RGBA is ~16 MB per variant. 8 angles halves it. Budget per variant, not
  per character -- variants are shared.
- Transition from skinned mesh to impostor: a hard switch at a distance threshold
  is usually acceptable and much cheaper than a cross-fade, provided the impostor
  frame is picked from the same clip time.
- **Temporal animation LOD** (updating distant characters' animation less often)
  is probably unnecessary once impostors exist: for an impostor, "advancing the
  animation" is choosing a frame index, which is nearly free. Measure before
  building it.

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
- Add PBR/normal maps unless direction changes.

## Git

Commit after every coherent change set — small, self-contained commits, so a bad
change can be reverted without unpicking good ones. No need to ask first.
Push occasionally rather than per commit (this is a big repo; pushing every
commit wastes time), but do push before wrapping up a session's work.

## Deferred refactors (design decisions needed)

1. **Graphics pipeline dedup** — ~~`graphics_pipeline.hpp` has 3 functions (`create_graphics_pipeline` x2 + `create_depth_only_graphics_pipeline`) sharing ~90% setup code. ~200 lines of triplication. Extracting shared logic would touch all pipeline paths (main, shadow, point, particle) — needs thorough re-testing.~~ Done. Extracted `detail::build_graphics_pipeline`. File: 425 → 275 lines.

2. **Two-step init → RAII** — `Swapchain`, `DepthImage`, `RenderColorImage`, `ShadowMap`, `TextureImage`, `TextureArray`, `ParticleSystem`, `BoneStorageBufferSet`, `UniformBufferSet` all default-construct empty, then require `create()`. Moving to RAII constructors prevents use-before-init bugs. Some (Swapchain) need `recreate()` anyway for resize.

3. **`MeshInstance::pose_layers` raw pointer** — pointer to `AnimStateMachine::layers()` owned by user. If the state machine moves/relocates/reallocates, the pointer dangles silently. Options: `std::shared_ptr`, `std::reference_wrapper`, or keep raw with debug assert.

4. **`PipelineId` insertion safety** — adding a pipeline in the middle of the enum breaks `is_textured_surface_pipeline(id >= E_First && id <= E_Last)` and `is_skinned_pipeline`. Options: explicit check sets (`std::unordered_set<PipelineId>`) or sentinel `Count` values.

5. **Bone buffer instance mapping** — `bone_instance_index` is computed sequentially by four separate walkers. ~~They disagreed (on `alive`, on `pose_layers`, on empty `joint_nodes`), so dead or pose-stack-less skinned instances shifted every later instance onto the wrong bone buffer.~~ All four now share `instance_uses_skinning()` and are covered by `tests/test_scene_invariants.cpp` — see **The bone-slot invariant** above.

   Still open at the design level: the mapping is *positional*, so it is only correct because `remove_instance()` tombstones rather than erases. A stable `bone_handle` (slot map) per instance would remove the ordering dependency entirely and is the real fix if instance storage ever compacts.

6. **`TextureImage/TextureArray::create_sampler` identical** — ~~character-for-character duplicate (~20 lines each). Extract to `detail::create_mipmapped_sampler`.~~ Done.

7. **`scene_uses_alpha_to_coverage` startup-only** — checked once in `VulkanContext` ctor. New A2C meshes added via `sync_scene` won't enable MSAA. Could be checked in `sync_scene` or deferred to a `render_settings` flag.
