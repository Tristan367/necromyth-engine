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

  [[nodiscard]] auto create_static_mesh(const MeshSource &mesh, const glm::vec3 &position,
                                        const glm::quat &rotation = glm::quat(1.0F, 0.0F, 0.0F, 0.0F))
        -> JPH::BodyID {
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

    JPH::BodyCreationSettings settings(
        shape,
        JPH::RVec3(position.x, position.y, position.z),
        JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w),
        JPH::EMotionType::Static,
        Layers::kNonMoving);

    settings.mFriction = 0.8F;

    JPH::BodyID id = body_interface_->CreateAndAddBody(settings, JPH::EActivation::DontActivate);
    body_ids_.push_back(id);
    return id;
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
  Character(PhysicsWorld &world, const glm::vec3 &position,
            float radius = 0.5F, float height = 1.5F)
      : world_{world} {
    JPH::Ref<JPH::CapsuleShape> shape = new JPH::CapsuleShape(0.5F * height, radius);
    JPH::CharacterVirtualSettings settings;
    settings.mShape = shape;
    settings.mMaxSlopeAngle = JPH::DegreesToRadians(60.0F);
    settings.mEnhancedInternalEdgeRemoval = true;
    settings.mInnerBodyShape = nullptr;
    settings.mMass = 0.0F;                    // disables gravity push-down onto bodies underfoot
    settings.mPenetrationRecoverySpeed = 0.5F;
    character_ = new JPH::CharacterVirtual(
        &settings,
        JPH::RVec3(position.x, position.y, position.z),
        JPH::Quat::sIdentity(),
        &world_.physics_system());
    character_->SetListener(&contact_listener_);
  }

  ~Character() { delete character_; }

  Character(const Character &) = delete;
  Character &operator=(const Character &) = delete;
  Character(Character &&) = delete;
  Character &operator=(Character &&) = delete;

  void update(float delta) {
    JPH::CharacterVirtual::ExtendedUpdateSettings settings;
    settings.mStickToFloorStepDown = JPH::Vec3(0.0F, -0.5F, 0.0F);
    settings.mWalkStairsStepUp = JPH::Vec3(0.0F, 0.4F, 0.0F);

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

  [[nodiscard]] auto is_on_ground() const -> bool {
    return character_->GetGroundState() == JPH::CharacterBase::EGroundState::OnGround;
  }

  void set_max_strength(float s) { character_->SetMaxStrength(s); }
  void set_allow_sliding(bool allow) { contact_listener_.allow_sliding_ = allow; }

private:
  PhysicsWorld &world_;
  JPH::CharacterVirtual *character_{nullptr};

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
