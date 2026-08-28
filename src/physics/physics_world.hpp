#pragma once

#include "scene/mesh_instance.hpp"
#include "scene/mesh_source.hpp"

#include <Jolt/Jolt.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/TaperedCapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/TaperedCylinderShape.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace engine {
namespace physics {

JPH_SUPPRESS_WARNINGS

namespace Layers {
static constexpr JPH::ObjectLayer kNonMoving = 0;
static constexpr JPH::ObjectLayer kMoving = 1;
static constexpr JPH::ObjectLayer kHitbox = 2;  // sensor-only body-part hitbox
static constexpr JPH::ObjectLayer kWeapon = 3;  // sensor-only weapon hitbox, overlaps kHitbox
static constexpr JPH::ObjectLayer kNumLayers = 4;
} // namespace Layers

namespace BroadPhaseLayers {
static constexpr JPH::BroadPhaseLayer kNonMoving(0);
static constexpr JPH::BroadPhaseLayer kMoving(1);
static constexpr JPH::BroadPhaseLayer kHitbox(2);
static constexpr JPH::BroadPhaseLayer kWeapon(3);
static constexpr std::uint32_t kNumLayers = 4;
} // namespace BroadPhaseLayers

class PhysicsWorld {
public:
  PhysicsWorld(std::uint32_t max_bodies = 1024) {
    static std::once_flag s_jolt_init;
    std::call_once(s_jolt_init, [] {
      JPH::RegisterDefaultAllocator();
      JPH::Factory::sInstance = new JPH::Factory();
      JPH::RegisterTypes();
    });

    temp_allocator_ = std::make_unique<JPH::TempAllocatorImpl>(10 * 1024 * 1024);
    job_system_ = std::make_unique<JPH::JobSystemThreadPool>(
        JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers,
        std::max(std::thread::hardware_concurrency(), 1U) - 1);

    physics_system_.Init(
        max_bodies, 0, 1024, 1024,
        broad_phase_layer_interface_,
        object_vs_broadphase_layer_filter_,
        object_vs_object_layer_filter_);

    body_interface_ = &physics_system_.GetBodyInterface();
    physics_system_.SetContactListener(&sensor_contact_tracker_);
  }

  ~PhysicsWorld() noexcept {
    for (const JPH::BodyID id : body_ids_) {
      try {
        body_interface_->RemoveBody(id);
        body_interface_->DestroyBody(id);
      } catch (...) {}
    }
  }

  void step(float delta_time) {
    physics_system_.Update(delta_time, 1, temp_allocator_.get(), job_system_.get());
  }

  void remove_body(JPH::BodyID body_id) {
    const auto it = std::ranges::find(body_ids_, body_id);
    if (it == body_ids_.end()) return;
    body_interface_->RemoveBody(body_id);
    body_interface_->DestroyBody(body_id);
    body_ids_.erase(it);
  }

  [[nodiscard]] auto create_box(const glm::vec3 &half_extent, const glm::vec3 &position,
                                 JPH::EMotionType motion_type, JPH::ObjectLayer layer,
                                 const glm::quat &rotation = glm::quat(1.0F, 0.0F, 0.0F, 0.0F),
                                 float mass_override_kg = 0.0F, float friction = 0.5F)
      -> JPH::BodyID {
    JPH::BoxShapeSettings shape_settings(
        JPH::Vec3(half_extent.x, half_extent.y, half_extent.z));
    shape_settings.SetEmbedded();
    JPH::ShapeRefC shape = shape_settings.Create().Get();

    JPH::BodyCreationSettings settings(
        shape,
        JPH::RVec3(position.x, position.y, position.z),
        JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w),
        motion_type, layer);

    settings.mFriction = friction;

    if (mass_override_kg > 0.0F) {
      settings.mOverrideMassProperties = JPH::EOverrideMassProperties::MassAndInertiaProvided;
      settings.mMassPropertiesOverride.mMass = mass_override_kg;
    }

    JPH::BodyID id = body_interface_->CreateAndAddBody(
        settings,
        motion_type == JPH::EMotionType::Dynamic ? JPH::EActivation::Activate
                                                   : JPH::EActivation::DontActivate);
    body_ids_.push_back(id);
    return id;
  }

  // Drives a kinematic body to a new position, sweeping rather than teleporting.
  //
  // MoveKinematic rather than SetPosition, because the difference is whether
  // anything standing on it comes along: MoveKinematic gives the body a velocity
  // for the step, so contacts resolve and a character riding it is carried.
  // SetPosition teleports, and whatever was standing on it is left behind in the
  // air for a frame and then falls through.
  //
  // `delta_time` is the step the move should happen over -- pass the same value
  // the next step() gets, or the implied velocity is wrong.
  void move_kinematic(JPH::BodyID body_id, const glm::vec3 &position, float delta_time,
                      const glm::quat &rotation = glm::quat(1.0F, 0.0F, 0.0F, 0.0F)) {
    if (body_id.IsInvalid() || delta_time <= 0.0F)
      return;
    body_interface_->MoveKinematic(body_id,
                                   JPH::RVec3(position.x, position.y, position.z),
                                   JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w),
                                   delta_time);
  }

  // Building the shape and adding the body are separate because the first is
  // pure computation and the second is not.
  //
  // Welding the vertices and building the BVH is 75% of what a terrain collider
  // costs -- measured at 22.0ms against 7.3ms for turning voxels into triangles,
  // over a 9,000 frame walk -- and none of it touches the physics system. It
  // reads a mesh and returns a shape, so it can run on a worker while the main
  // thread gets on with the frame. Adding the body cannot: BodyInterface is
  // shared, and body_ids_ is ours.
  [[nodiscard]] static auto build_static_mesh_shape(const MeshSource &mesh) -> JPH::ShapeRefC {

    // Vertex welding: collapse coincident positions to shared indices at
    // 0.1 mm grid resolution so Jolt's mEnhancedInternalEdgeRemoval can
    // detect internal edges topologically (matching AGENTS.md documentation).
    static constexpr float kWeldTol = 0.0001F;
    const float inv_tol = 1.0F / kWeldTol;
    auto hash = [inv_tol](const glm::vec3 &p) -> std::uint64_t {
      return (static_cast<std::uint64_t>(static_cast<std::int64_t>(p.x * inv_tol)) << 42) ^
             (static_cast<std::uint64_t>(static_cast<std::int64_t>(p.y * inv_tol)) << 21) ^
             static_cast<std::uint64_t>(static_cast<std::int64_t>(p.z * inv_tol));
    };

    std::unordered_map<std::uint64_t, std::uint32_t> pos_map;
    JPH::VertexList welded_vertices;
    JPH::IndexedTriangleList welded_triangles;
    welded_vertices.reserve(mesh.vertices.size());
    welded_triangles.reserve(mesh.indices.size() / 3);

    for (const MeshVertex &v : mesh.vertices) {
      const std::uint64_t key = hash(glm::vec3{v.pos[0], v.pos[1], v.pos[2]});
      auto [it, inserted] = pos_map.try_emplace(key, static_cast<std::uint32_t>(welded_vertices.size()));
      if (inserted)
        welded_vertices.emplace_back(v.pos[0], v.pos[1], v.pos[2]);
    }

    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
      auto idx = [&](std::uint32_t vi) -> std::uint32_t {
        if (vi >= mesh.vertices.size()) return 0;  // corrupt mesh guard
        const MeshVertex &v = mesh.vertices[vi];
        return pos_map.at(hash(glm::vec3{v.pos[0], v.pos[1], v.pos[2]}));
      };
      welded_triangles.emplace_back(idx(mesh.indices[i]), idx(mesh.indices[i+1]),
                                    idx(mesh.indices[i+2]), 0);
    }

    JPH::MeshShapeSettings shape_settings(welded_vertices, welded_triangles);
    shape_settings.SetEmbedded();
    const auto create_result = shape_settings.Create();
    if (create_result.HasError())
      return {};  // degenerate mesh — return invalid BodyID
    JPH::ShapeRefC shape = create_result.Get();

    return shape;
  }

  // Takes a shape built anywhere -- this thread or a worker -- and gives it a
  // body. Cheap, and the only half that has to be here.
  [[nodiscard]] auto add_static_mesh_body(
      const JPH::ShapeRefC &shape, const glm::vec3 &position,
      const glm::quat &rotation = glm::quat(1.0F, 0.0F, 0.0F, 0.0F),
      JPH::EMotionType motion_type = JPH::EMotionType::Static) -> JPH::BodyID {
    if (shape == nullptr)
      return {};
    // A kinematic mesh still belongs on the non-moving layer: it is something
    // the world collides WITH, not something that collides with the world. What
    // kinematic buys is the right to be driven -- see move_kinematic -- which is
    // what a falling piece of building needs and a chunk of terrain does not.
    const bool kinematic = motion_type == JPH::EMotionType::Kinematic;
    JPH::BodyCreationSettings settings(
        shape,
        JPH::RVec3(position.x, position.y, position.z),
        JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w),
        motion_type,
        Layers::kNonMoving);

    settings.mFriction = 0.8F;

    JPH::BodyID id = body_interface_->CreateAndAddBody(
        settings, kinematic ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);
    body_ids_.push_back(id);
    return id;
  }

  [[nodiscard]] auto create_static_mesh(const MeshSource &mesh, const glm::vec3 &position,
                                        const glm::quat &rotation = glm::quat(1.0F, 0.0F, 0.0F, 0.0F),
                                        JPH::EMotionType motion_type = JPH::EMotionType::Static)
        -> JPH::BodyID {
    return add_static_mesh_body(build_static_mesh_shape(mesh), position, rotation, motion_type);
  }

  [[nodiscard]] auto add_dynamic_body(const JPH::ShapeSettings &shape_settings,
                                       const glm::vec3 &position) -> JPH::BodyID {
    JPH::ShapeRefC shape = shape_settings.Create().Get();
    JPH::BodyCreationSettings settings(
        shape,
        JPH::RVec3(position.x, position.y, position.z),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        Layers::kMoving);
    settings.mFriction = 0.7F;
    settings.mOverrideMassProperties = JPH::EOverrideMassProperties::MassAndInertiaProvided;
    settings.mMassPropertiesOverride.mMass = 1.0F;
    JPH::BodyID id = body_interface_->CreateAndAddBody(settings, JPH::EActivation::Activate);
    body_ids_.push_back(id);
    return id;
  }

  [[nodiscard]] auto add_sphere(float radius, const glm::vec3 &position) -> JPH::BodyID {
    JPH::SphereShapeSettings s(radius); s.SetEmbedded();
    return add_dynamic_body(s, position);
  }

  [[nodiscard]] auto add_capsule(float half_height, float radius, const glm::vec3 &position) -> JPH::BodyID {
    JPH::CapsuleShapeSettings s(half_height, radius); s.SetEmbedded();
    return add_dynamic_body(s, position);
  }

  [[nodiscard]] auto add_cylinder(float half_height, float radius, const glm::vec3 &position) -> JPH::BodyID {
    JPH::CylinderShapeSettings s(half_height, radius); s.SetEmbedded();
    return add_dynamic_body(s, position);
  }

  [[nodiscard]] auto add_tapered_capsule(float half_height, float top_radius, float bottom_radius,
                                          const glm::vec3 &position) -> JPH::BodyID {
    JPH::TaperedCapsuleShapeSettings s(half_height, top_radius, bottom_radius); s.SetEmbedded();
    return add_dynamic_body(s, position);
  }

  [[nodiscard]] auto add_tapered_cylinder(float half_height, float top_radius, float bottom_radius,
                                           const glm::vec3 &position) -> JPH::BodyID {
    float convex = std::min(0.05F, std::min(top_radius, bottom_radius) * 0.5F);
    JPH::TaperedCylinderShapeSettings s(half_height, top_radius, bottom_radius, convex); s.SetEmbedded();
    return add_dynamic_body(s, position);
  }

  void sync_body_to_instance(JPH::BodyID body_id, engine::MeshInstance &instance) const {
    JPH::BodyLockRead lock(physics_system_.GetBodyLockInterface(), body_id);
    if (!lock.Succeeded()) return;
    const JPH::Vec3 pos = lock.GetBody().GetPosition();
    const JPH::Quat rot = lock.GetBody().GetRotation();

    glm::mat4 model(1.0F);
    model = glm::translate(model, glm::vec3(pos.GetX(), pos.GetY(), pos.GetZ()));
    model *= glm::mat4_cast(glm::quat(rot.GetW(), rot.GetX(), rot.GetY(), rot.GetZ()));
    instance.model = model;
  }

  [[nodiscard]] auto body_interface() -> JPH::BodyInterface & { return *body_interface_; }
  [[nodiscard]] auto physics_system() -> JPH::PhysicsSystem & { return physics_system_; }
  [[nodiscard]] auto temp_allocator() -> JPH::TempAllocator & { return *temp_allocator_; }

  void set_body_user_data(JPH::BodyID id, std::uint64_t data) {
    JPH::BodyLockWrite lock(physics_system_.GetBodyLockInterface(), id);
    if (lock.Succeeded())
      lock.GetBody().SetUserData(data);
  }

  [[nodiscard]] auto get_body_user_data(JPH::BodyID id) const -> std::uint64_t {
    JPH::BodyLockRead lock(physics_system_.GetBodyLockInterface(), id);
    if (!lock.Succeeded()) return 0;
    return lock.GetBody().GetUserData();
  }

  [[nodiscard]] auto shape_center_of_mass(JPH::BodyID body_id) const -> JPH::Vec3 {
    JPH::BodyLockRead lock(physics_system_.GetBodyLockInterface(), body_id);
    if (lock.Succeeded())
      return lock.GetBody().GetShape()->GetCenterOfMass();
    return JPH::Vec3::sZero();
  }

  void set_sensor_transform(JPH::BodyID body_id, const glm::vec3 &pos, const glm::quat &rot) {
    body_interface_->SetPositionAndRotation(
        body_id,
        JPH::RVec3(pos.x, pos.y, pos.z),
        JPH::Quat(rot.x, rot.y, rot.z, rot.w),
        JPH::EActivation::DontActivate);
  }

  [[nodiscard]] auto get_sensor_overlaps(JPH::BodyID body_id) const -> std::vector<JPH::BodyID> {
    return sensor_contact_tracker_.get_overlaps(body_id);
  }

private:
  JPH::PhysicsSystem physics_system_;
  JPH::BodyInterface *body_interface_{nullptr};
  std::unique_ptr<JPH::TempAllocatorImpl> temp_allocator_;
  std::unique_ptr<JPH::JobSystemThreadPool> job_system_;
  std::vector<JPH::BodyID> body_ids_;

  class BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface {
  public:
    BPLayerInterfaceImpl() {
      mObjectToBroadPhase[Layers::kNonMoving] = BroadPhaseLayers::kNonMoving;
      mObjectToBroadPhase[Layers::kMoving] = BroadPhaseLayers::kMoving;
      mObjectToBroadPhase[Layers::kHitbox] = BroadPhaseLayers::kHitbox;
      mObjectToBroadPhase[Layers::kWeapon] = BroadPhaseLayers::kWeapon;
    }
    std::uint32_t GetNumBroadPhaseLayers() const override { return BroadPhaseLayers::kNumLayers; }
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override {
      JPH_ASSERT(inLayer < Layers::kNumLayers);
      return mObjectToBroadPhase[inLayer];
    }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char *GetBroadPhaseLayerName(JPH::BroadPhaseLayer layer) const override {
      if (layer == BroadPhaseLayers::kNonMoving) return "NON_MOVING";
      if (layer == BroadPhaseLayers::kMoving) return "MOVING";
      if (layer == BroadPhaseLayers::kHitbox) return "HITBOX";
      if (layer == BroadPhaseLayers::kWeapon) return "WEAPON";
      return "";
    }
#endif
  private:
    JPH::BroadPhaseLayer mObjectToBroadPhase[Layers::kNumLayers];
  };

  class ObjectVsBroadPhaseLayerFilterImpl final : public JPH::ObjectVsBroadPhaseLayerFilter {
  public:
    bool ShouldCollide(JPH::ObjectLayer inLayer, JPH::BroadPhaseLayer inBroadPhase) const override {
      if (inLayer == Layers::kHitbox)
        return inBroadPhase == BroadPhaseLayers::kWeapon;  // hitbox ↔ weapon only
      if (inLayer == Layers::kWeapon)
        return inBroadPhase == BroadPhaseLayers::kHitbox;  // weapon ↔ hitbox only
      return true;
    }
  };

  class ObjectLayerPairFilterImpl final : public JPH::ObjectLayerPairFilter {
  public:
    bool ShouldCollide(JPH::ObjectLayer inLayer1, JPH::ObjectLayer inLayer2) const override {
      if (inLayer1 == Layers::kWeapon && inLayer2 == Layers::kHitbox) return true;
      if (inLayer1 == Layers::kHitbox && inLayer2 == Layers::kWeapon) return true;
      if (inLayer1 == Layers::kHitbox || inLayer2 == Layers::kHitbox) return false;
      if (inLayer1 == Layers::kWeapon || inLayer2 == Layers::kWeapon) return false;
      return true;
    }
  };

  class SensorContactTracker : public JPH::ContactListener {
  public:
    void OnContactAdded(const JPH::Body &body1, const JPH::Body &body2,
                        const JPH::ContactManifold &, JPH::ContactSettings &) override {
      if (body1.IsSensor() && body2.IsSensor())
        add_pair(body1.GetID(), body2.GetID());
    }
    void OnContactPersisted(const JPH::Body &, const JPH::Body &,
                            const JPH::ContactManifold &, JPH::ContactSettings &) override {}
    void OnContactRemoved(const JPH::SubShapeIDPair &pair) override {
      remove_pair(pair.GetBody1ID(), pair.GetBody2ID());
    }

    [[nodiscard]] auto get_overlaps(JPH::BodyID body) const -> std::vector<JPH::BodyID> {
      std::vector<JPH::BodyID> result;
      auto it = overlaps_.find(body.GetIndexAndSequenceNumber());
      if (it == overlaps_.end()) return result;
      for (std::uint32_t raw : it->second)
        result.push_back(JPH::BodyID(raw));
      return result;
    }

  private:
    void add_pair(JPH::BodyID a, JPH::BodyID b) {
      const auto ka = a.GetIndexAndSequenceNumber();
      const auto kb = b.GetIndexAndSequenceNumber();
      overlaps_[ka].insert(kb);
      overlaps_[kb].insert(ka);
    }
    void remove_pair(JPH::BodyID a, JPH::BodyID b) {
      const auto ka = a.GetIndexAndSequenceNumber();
      const auto kb = b.GetIndexAndSequenceNumber();
      if (auto it = overlaps_.find(ka); it != overlaps_.end())
        it->second.erase(kb);
      if (auto it = overlaps_.find(kb); it != overlaps_.end())
        it->second.erase(ka);
    }

    std::unordered_map<std::uint32_t, std::unordered_set<std::uint32_t>> overlaps_;
  };

  BPLayerInterfaceImpl broad_phase_layer_interface_{};
  ObjectVsBroadPhaseLayerFilterImpl object_vs_broadphase_layer_filter_{};
  ObjectLayerPairFilterImpl object_vs_object_layer_filter_{};
  SensorContactTracker sensor_contact_tracker_{};
};

class Character {
public:
  // Zero. The controller does not lift the character over anything.
  //
  // This is Godot's answer -- CharacterBody3D has no step-up mechanic at all --
  // and it took a bug report to see why it is the right one. Jolt's stair sweep
  // goes UP by this much, then FORWARD, then DOWN, and it commits to whatever
  // the down-sweep lands on. Against anything whose height varies across the
  // capsule's own footprint, that ratchets: it finds the low corner, steps up,
  // lands higher, and does it again next frame.
  //
  // The side of a staircase is exactly that shape. A stair collides as a wedge,
  // so its side is a triangle rising from nothing at the foot to a metre at the
  // head, and walking into one from the side climbed it -- reported as "you go
  // up the non-ramp side and you just instantly teleport to the middle of the
  // ramp on top of it". Measured against a single stair voxel, walking in from
  // the side:
  //
  //     step-up 0.35   climbed 0.83 m
  //     step-up 0.10   climbed 0.57 m
  //     step-up 0      climbed 0.28 m
  //
  // Note the amplification. The climb is never the step height; it is roughly
  // twice it plus what the capsule's own rounded bottom manages. So no value of
  // this number is safe, which is why it is zero rather than smaller.
  //
  // What it cost: a 30 cm vertical riser is no longer walkable. Nothing in the
  // game has one. Stairs collide as a smooth ramp from foot to head -- that is
  // collide_stair's whole point -- terrain rises across a marching-cubes cell
  // as a 45 degree slope, and a block is a metre, which is a jump. The only
  // thing that ever needed the sweep was a shape the world does not contain.
  //
  // Everything the sweep genuinely did still works without it: a 20 cm lip is
  // walked over by the capsule's rounded bottom alone, rolling ground is
  // unchanged frame for frame, and 35 of the 36 character checks pass either
  // way.
  static constexpr float k_step_height = 0.0F;

  // How far the controller may pull the character back down to keep contact.
  //
  // Deliberately smaller than the step-up. Matching them meant a descent was
  // resolved by teleporting down more than a metre every frame, which is faster
  // than gravity and feels like being sucked into the hill. Half a voxel is
  // enough to stay glued to a slope you are walking down at a normal pace, and
  // little enough that running off a real drop lets gravity take over.
  static constexpr float k_stick_to_floor = 0.5F;

  // How far the controller pulls the character back down to keep contact with
  // ground it was already standing on. Godot's floor_snap_length, and Godot's
  // default value.
  //
  // Godot is the reference here on purpose. Its CharacterBody3D has no step-up
  // mechanic at all and exactly one vertical assist -- _snap_on_floor, which
  // moves DOWN by floor_snap_length and only when the body was on the floor,
  // is not on it now, and is not trying to move up. Ten centimetres. That is
  // the whole of it, and it is the controller people describe as feeling
  // right.
  //
  // Ours had 1.2 m, applied under the opposite condition. Anything this size
  // is a teleport with a friendly name: the controller resolves it inside one
  // frame, so the view moves a tenth of a metre between two frames and there
  // is nothing gradual about it.
  static constexpr float k_floor_snap = 0.1F;

  // Climbing: stick hard, so cresting a rise cannot throw the character.
  static constexpr float k_stick_climbing = 1.2F;
  // Descending: barely stick at all, so gravity does the work and a drop reads
  // as falling rather than as being pulled.
  static constexpr float k_stick_descending = 0.15F;
  static constexpr float k_max_slope_degrees = 60.0F;

  Character(PhysicsWorld &world, const glm::vec3 &position,
            float radius = 0.5F, float height = 1.5F)
      : world_{world} {
    JPH::Ref<JPH::CapsuleShape> shape = new JPH::CapsuleShape(0.5F * height, radius);
    JPH::CharacterVirtualSettings settings;
    settings.mShape = shape;
    // Steep enough to walk any marching-cubes slope the terrain generates --
    // a cell that rises a full voxel across its own width is 45 degrees -- and
    // shallow enough that a wall is still a wall.
    settings.mMaxSlopeAngle = JPH::DegreesToRadians(k_max_slope_degrees);
    settings.mEnhancedInternalEdgeRemoval = true;
    settings.mInnerBodyShape = nullptr;
    settings.mMass = 0.0F;                    // disables gravity push-down onto bodies underfoot
    // Full-speed recovery. At half speed a capsule that sinks a little into a
    // slope takes several frames to be pushed back out, and while it is in
    // there it is wedged.
    settings.mPenetrationRecoverySpeed = 1.0F;
    character_ = new JPH::CharacterVirtual(
        &settings,
        JPH::RVec3(position.x, position.y, position.z),
        JPH::Quat::sIdentity(),
        &world_.physics_system());
    character_->SetListener(&contact_listener_);
    current_height_ = height + 2.0F * radius;
  }

  ~Character() { delete character_; }

  Character(const Character &) = delete;
  Character &operator=(const Character &) = delete;
  Character(Character &&) = delete;
  Character &operator=(Character &&) = delete;

  // `stick_down` is how far the controller may pull the character back down to
  // keep contact with the ground this step.
  //
  // It wants to be asymmetric. Going UP a slope, sticking hard is what stops the
  // character being thrown off the crest -- there is no downside, because the
  // ground is rising to meet it anyway. Going DOWN, sticking hard is exactly
  // what makes a descent feel like being sucked into the hill, because the
  // controller teleports the character down faster than gravity would take it.
  // One number cannot be right for both, so the caller picks per step.
  // `step_up` is how high the controller may lift the character over an
  // obstacle it has walked into. **Pass zero whenever the character is not
  // standing on something**, or it will climb while airborne: walk into a
  // one-metre block, jump a few centimetres, and the stair logic finishes the
  // climb for you and plants you on top. That reads as teleporting onto a block
  // you only brushed against, and no amount of tuning the height fixes it,
  // because the height is not the problem -- running it at all while in the air
  // is.
  void update(float delta, float stick_down = k_stick_to_floor,
              float step_up = k_step_height) {
    JPH::CharacterVirtual::ExtendedUpdateSettings settings;
    // Both of these are measured in world units against a world made of one
    // metre voxels, so anything below 1.0 cannot deal with a single voxel.
    //
    // They were 0.5 and 0.4, which meant a step of one voxel -- the smallest
    // feature the world can possibly have -- stopped the character dead, and
    // the lip where two marching-cubes cells meet was enough to catch on. That
    // reads as "walks for a bit, then gets stuck against a hill".
    settings.mStickToFloorStepDown = JPH::Vec3(0.0F, -std::max(stick_down, 0.0F), 0.0F);
    settings.mWalkStairsStepUp = JPH::Vec3(0.0F, std::max(step_up, 0.0F), 0.0F);

    character_->ExtendedUpdate(delta,
                               JPH::Vec3(0.0F, -9.81F, 0.0F),
                               settings,
                               world_.physics_system().GetDefaultBroadPhaseLayerFilter(Layers::kMoving),
                               world_.physics_system().GetDefaultLayerFilter(Layers::kMoving),
                               {}, {},
                               world_.temp_allocator());
  }

  void update_ground_velocity() { character_->UpdateGroundVelocity(); }

  [[nodiscard]] auto ground_velocity() const -> glm::vec3 {
    const JPH::Vec3 v = character_->GetGroundVelocity();
    return {v.GetX(), v.GetY(), v.GetZ()};
  }

  [[nodiscard]] auto position() const -> glm::vec3 {
    const JPH::RVec3 pos = character_->GetPosition();
    return {pos.GetX(), pos.GetY(), pos.GetZ()};
  }

  [[nodiscard]] auto linear_velocity() const -> glm::vec3 {
    const JPH::Vec3 v = character_->GetLinearVelocity();
    return {v.GetX(), v.GetY(), v.GetZ()};
  }

  void set_velocity(const glm::vec3 &v) {
    character_->SetLinearVelocity(JPH::Vec3(v.x, v.y, v.z));
  }

  // Teleports the character. Needed for respawn, and for rebasing when the
  // world origin shifts under a streaming world.
  void set_position(const glm::vec3 &p) {
    character_->SetPosition(JPH::RVec3(p.x, p.y, p.z));
  }

  // Surface normal of whatever is underfoot, or straight up when airborne.
  // Callers that want a constant speed regardless of slope need this: the
  // movement direction has to be projected onto the ground plane and rescaled,
  // or walking uphill silently costs you speed.
  [[nodiscard]] auto ground_normal() const -> glm::vec3 {
    if (!is_on_ground())
      return {0.0F, 1.0F, 0.0F};
    const JPH::Vec3 n = character_->GetGroundNormal();
    return {n.GetX(), n.GetY(), n.GetZ()};
  }

  // The wall the character is pressed against, or zero if it is not touching
  // one. Points away from the wall, toward the character.
  //
  // A WALL, specifically: a contact too steep to stand on. That distinction is
  // the whole reason this exists. Speed lost to a wall should be gone, and speed
  // lost to a slope should not -- climbing costs horizontal distance because the
  // ground goes up, not because anything blocked you -- and the two are
  // indistinguishable if you try to infer them from how far the character
  // actually moved, which is what the first version of the write-back did. It
  // zeroed your speed the moment you set foot on a ramp.
  //
  // Godot classifies every collision the same way and by the same test
  // (character_body_3d.cpp, _set_collision_direction): floor if the normal is
  // within the slope limit of up, ceiling if it is within the limit of down,
  // wall otherwise. Several walls at once are averaged, which is what stops a
  // corner picking one of its two faces and jittering between them.
  [[nodiscard]] auto wall_normal() const -> glm::vec3 {
    const float limit = std::cos(JPH::DegreesToRadians(k_max_slope_degrees));
    JPH::Vec3 sum = JPH::Vec3::sZero();
    int walls = 0;
    for (const JPH::CharacterContact &contact : character_->GetActiveContacts()) {
      if (!contact.mHadCollision)
        continue;
      const JPH::Vec3 n = contact.mSurfaceNormal;
      const float up = n.GetY();
      if (up >= limit || up <= -limit)
        continue; // floor or ceiling, not a wall
      sum += n;
      ++walls;
    }
    if (walls == 0)
      return {0.0F, 0.0F, 0.0F};
    if (sum.IsNearZero())
      return {0.0F, 0.0F, 0.0F}; // two opposed walls: wedged, no direction to slide
    sum = sum.Normalized();
    return {sum.GetX(), sum.GetY(), sum.GetZ()};
  }

  [[nodiscard]] auto is_on_ground() const -> bool {
    return character_->GetGroundState() == JPH::CharacterBase::EGroundState::OnGround;
  }

  // Swaps the capsule for a shorter one, or back, keeping the FEET where they
  // are. Returns false if the new shape does not fit, which is the whole point
  // of asking: standing up under a ceiling has to fail rather than push the
  // character through it.
  //
  // Jolt's position is the shape's origin, and the shape is a capsule centred on
  // it, so the centre is half the total height above the feet. Changing the
  // height therefore has to move the position by half the difference or the
  // character grows and shrinks from its middle -- which reads as sinking into
  // the floor when you go down and hopping when you get up.
  auto try_set_height(float total_height, float radius) -> bool {
    const float half = std::max(0.5F * total_height - radius, 0.01F);
    JPH::Ref<JPH::CapsuleShape> shape = new JPH::CapsuleShape(half, radius);

    const float was = current_height_;
    const float rise = 0.5F * (total_height - was);
    const JPH::RVec3 at = character_->GetPosition();
    character_->SetPosition({at.GetX(), at.GetY() + rise, at.GetZ()});

    const bool fitted = character_->SetShape(
        shape, k_shape_penetration,
        world_.physics_system().GetDefaultBroadPhaseLayerFilter(Layers::kMoving),
        world_.physics_system().GetDefaultLayerFilter(Layers::kMoving), {}, {},
        world_.temp_allocator());
    if (!fitted) {
      character_->SetPosition(at); // put it back exactly where it was
      return false;
    }
    current_height_ = total_height;
    return true;
  }

  [[nodiscard]] auto height() const -> float { return current_height_; }

  void set_max_strength(float s) { character_->SetMaxStrength(s); }
  // How much of a penetration the controller works off per step. 1 means all of
  // it, in one frame, which is a teleport whenever the capsule has sunk into
  // anything. Exposed so a run can sweep it -- see NM_RECOVERY.
  void set_penetration_recovery(float speed) {
    character_->SetPenetrationRecoverySpeed(speed);
  }
  [[nodiscard]] auto penetration_recovery() const -> float {
    return character_->GetPenetrationRecoverySpeed();
  }
  void set_allow_sliding(bool allow) { contact_listener_.allow_sliding_ = allow; }

private:
  // How much the new shape may already be inside something and still be
  // accepted. A shape swap is not a teleport: a couple of centimetres is the
  // controller's own margin and penetration recovery will work it off, where a
  // larger tolerance would let a character stand up into a ceiling.
  static constexpr float k_shape_penetration = 0.02F;

  PhysicsWorld &world_;
  JPH::CharacterVirtual *character_{nullptr};
  float current_height_{0.0F};

  class ContactBlocker : public JPH::CharacterContactListener {
  public:
    void OnContactAdded(const JPH::CharacterVirtual *, const JPH::CharacterContact &,
                        JPH::CharacterContactSettings &) override {}
    void OnContactPersisted(const JPH::CharacterVirtual *, const JPH::CharacterContact &,
                            JPH::CharacterContactSettings &) override {}
    void OnContactSolve(const JPH::CharacterVirtual *inCharacter, const JPH::BodyID &,
                        const JPH::SubShapeID &, JPH::RVec3Arg, JPH::Vec3Arg inContactNormal,
                        JPH::Vec3Arg inContactVelocity, const JPH::PhysicsMaterial *,
                        JPH::Vec3Arg, JPH::Vec3 &ioNewCharacterVelocity) override {
      if (!allow_sliding_ && inContactVelocity.IsNearZero() &&
          !inCharacter->IsSlopeTooSteep(inContactNormal))
        ioNewCharacterVelocity = JPH::Vec3::sZero();
    }
    bool allow_sliding_{true};
  } contact_listener_;
};

} // namespace physics
} // namespace engine
