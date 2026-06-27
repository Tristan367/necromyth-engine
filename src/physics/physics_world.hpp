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
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace engine {
namespace physics {

JPH_SUPPRESS_WARNINGS

namespace Layers {
static constexpr JPH::ObjectLayer kNonMoving = 0;
static constexpr JPH::ObjectLayer kMoving = 1;
static constexpr JPH::ObjectLayer kNumLayers = 2;
} // namespace Layers

namespace BroadPhaseLayers {
static constexpr JPH::BroadPhaseLayer kNonMoving(0);
static constexpr JPH::BroadPhaseLayer kMoving(1);
static constexpr std::uint32_t kNumLayers = 2;
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
  }

  ~PhysicsWorld() {
    for (const JPH::BodyID id : body_ids_)
      body_interface_->RemoveBody(id);
    for (const JPH::BodyID id : body_ids_)
      body_interface_->DestroyBody(id);
  }

  void step(float delta_time) {
    physics_system_.Update(delta_time, 1, temp_allocator_.get(), job_system_.get());
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
    // Weld coincident vertices (0.1 mm grid) so internal mesh edges are
    // properly detected by Jolt's enhanced internal edge removal.
    auto key = [](const float *p) -> std::uint64_t {
      auto q = [](float v) { return static_cast<std::int64_t>(std::llround(v * 10000.0)); };
      return (static_cast<std::uint64_t>(static_cast<std::uint16_t>(q(p[0])))) |
             (static_cast<std::uint64_t>(static_cast<std::uint16_t>(q(p[1]))) << 16) |
             (static_cast<std::uint64_t>(static_cast<std::uint16_t>(q(p[2]))) << 32);
    };

    JPH::VertexList vertex_list;
    JPH::IndexedTriangleList tri_list;
    std::unordered_map<std::uint64_t, std::uint32_t> welded;
    std::vector<std::uint32_t> remap(mesh.vertices.size());

    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
      const float *p = mesh.vertices[i].pos;
      std::uint64_t k = key(p);
      auto it = welded.find(k);
      if (it == welded.end()) {
        std::uint32_t ni = static_cast<std::uint32_t>(vertex_list.size());
        welded[k] = ni;
        remap[i] = ni;
        vertex_list.emplace_back(p[0], p[1], p[2]);
      } else {
        remap[i] = it->second;
      }
    }

    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
      std::uint32_t a = remap[mesh.indices[i]];
      std::uint32_t b = remap[mesh.indices[i + 1]];
      std::uint32_t c = remap[mesh.indices[i + 2]];
      if (a != b && b != c && a != c)
        tri_list.emplace_back(a, b, c);
    }

    JPH::MeshShapeSettings shape_settings(vertex_list, tri_list);
    shape_settings.SetEmbedded();
    JPH::ShapeRefC shape = shape_settings.Create().Get();

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

  void sync_body_to_instance(JPH::BodyID body_id, engine::MeshInstance &instance) const {
    const JPH::Vec3 pos = body_interface_->GetPosition(body_id);
    const JPH::Quat rot = body_interface_->GetRotation(body_id);

    glm::mat4 model(1.0F);
    model = glm::translate(model, glm::vec3(pos.GetX(), pos.GetY(), pos.GetZ()));
    model *= glm::mat4_cast(glm::quat(rot.GetW(), rot.GetX(), rot.GetY(), rot.GetZ()));
    instance.model = model;
  }

  [[nodiscard]] auto body_interface() -> JPH::BodyInterface & { return *body_interface_; }
  [[nodiscard]] auto physics_system() -> JPH::PhysicsSystem & { return physics_system_; }
  [[nodiscard]] auto temp_allocator() -> JPH::TempAllocator & { return *temp_allocator_; }

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
    }
    std::uint32_t GetNumBroadPhaseLayers() const override { return BroadPhaseLayers::kNumLayers; }
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override {
      return mObjectToBroadPhase[inLayer];
    }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char *GetBroadPhaseLayerName(JPH::BroadPhaseLayer) const override { return ""; }
#endif
  private:
    JPH::BroadPhaseLayer mObjectToBroadPhase[Layers::kNumLayers];
  };

  class ObjectVsBroadPhaseLayerFilterImpl final : public JPH::ObjectVsBroadPhaseLayerFilter {
  public:
    bool ShouldCollide(JPH::ObjectLayer, JPH::BroadPhaseLayer) const override { return true; }
  };

  class ObjectLayerPairFilterImpl final : public JPH::ObjectLayerPairFilter {
  public:
    bool ShouldCollide(JPH::ObjectLayer, JPH::ObjectLayer) const override { return true; }
  };

  BPLayerInterfaceImpl broad_phase_layer_interface_{};
  ObjectVsBroadPhaseLayerFilterImpl object_vs_broadphase_layer_filter_{};
  ObjectLayerPairFilterImpl object_vs_object_layer_filter_{};
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
  } contact_listener_;};

} // namespace physics
} // namespace engine
