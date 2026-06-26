#pragma once

#include "scene/mesh_instance.hpp"
#include "scene/mesh_source.hpp"

#include <Jolt/Jolt.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/ShapeCast.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
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
    JPH::VertexList vertex_list;
    vertex_list.reserve(mesh.vertices.size());
    for (const MeshVertex &v : mesh.vertices)
      vertex_list.emplace_back(v.pos[0], v.pos[1], v.pos[2]);

    JPH::IndexedTriangleList tri_list;
    tri_list.reserve(mesh.indices.size() / 3);
    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3)
      tri_list.emplace_back(mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]);

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
    JPH::BodyCreationSettings settings(
        shape,
        JPH::RVec3(position.x, position.y, position.z),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Kinematic,
        Layers::kMoving);
    settings.mFriction = 0.0F;
    body_id_ = world_.body_interface().CreateAndAddBody(settings, JPH::EActivation::Activate);
  }

  ~Character() {
    world_.body_interface().RemoveBody(body_id_);
    world_.body_interface().DestroyBody(body_id_);
  }

  void update(float delta) {
    JPH::BodyInterface &bi = world_.body_interface();
    JPH::RVec3 pos = bi.GetPosition(body_id_);
    const JPH::Vec3 motion = velocity_ * delta;
    const JPH::Shape *shape = bi.GetShape(body_id_);
    JPH::IgnoreSingleBodyFilter body_filter(body_id_);

    // Godot-style: recovery (4 iterations, 40% per iter, margin 0.001)
    for (int iter = 0; iter < 4; ++iter) {
      JPH::CollideShapeSettings rec_settings;
      rec_settings.mMaxSeparationDistance = 0.001F;
      JPH::AllHitCollisionCollector<JPH::CollideShapeCollector> rec_collector;
      world_.physics_system().GetNarrowPhaseQuery().CollideShapeWithInternalEdgeRemoval(
          shape, JPH::Vec3::sReplicate(1.0F),
          JPH::RMat44::sTranslation(pos), rec_settings,
          JPH::RVec3::sZero(), rec_collector, {}, {}, body_filter);

      if (!rec_collector.HadHit())
        break;

      JPH::Vec3 push(0, 0, 0);
      bool had_penetration = false;
      for (const JPH::CollideShapeResult &hit : rec_collector.mHits) {
        if (hit.mPenetrationDepth > 0) {
          push -= hit.mPenetrationAxis.Normalized() * hit.mPenetrationDepth * 0.4F;
          had_penetration = true;
        }
      }
      if (!had_penetration)
        break;
      pos += JPH::RVec3(push);
    }

    // Godot-style: binary search sweep with CollideShape (works with meshes)
    float safe_fraction = 1.0f;
    {
      const int steps = std::clamp(static_cast<int>(std::log2(1000.0f * motion.Length())), 4, 16);
      float lo = 0.0f, hi = 1.0f, coeff = 0.5f;
      JPH::CollideShapeSettings sweep_settings;
      sweep_settings.mMaxSeparationDistance = 0.001F;
      bool collided = false;

      for (int j = 0; j < steps; ++j) {
        float fraction = lo + (hi - lo) * coeff;
        JPH::RVec3 probe = pos + JPH::RVec3(motion * fraction);
        JPH::ClosestHitCollisionCollector<JPH::CollideShapeCollector> step_collector;
        world_.physics_system().GetNarrowPhaseQuery().CollideShape(
            shape, JPH::Vec3::sReplicate(1.0F),
            JPH::RMat44::sTranslation(probe), sweep_settings,
            JPH::RVec3::sZero(), step_collector, {}, {}, body_filter);

        if (step_collector.HadHit()) {
          collided = true;
          if (step_collector.mHit.mPenetrationAxis.GetY() > 0.5F)
            ground_state_ = true;
          hi = fraction;
          coeff = (j == 0 || lo > 0.0f) ? 0.5f : 0.25f;
        } else {
          lo = fraction;
          coeff = (j == 0 || hi < 1.0f) ? 0.5f : 0.75f;
        }
      }
      safe_fraction = std::max(0.0f, lo - 0.001f);
    }

    JPH::RVec3 target = pos + JPH::RVec3(motion * safe_fraction);

    bi.SetPosition(body_id_, target, JPH::EActivation::Activate);
  }

  void update_ground_velocity() {}
  [[nodiscard]] auto ground_velocity() const -> glm::vec3 { return {0, 0, 0}; }

  [[nodiscard]] auto position() const -> glm::vec3 {
    const JPH::RVec3 pos = world_.body_interface().GetPosition(body_id_);
    return {pos.GetX(), pos.GetY(), pos.GetZ()};
  }

  [[nodiscard]] auto linear_velocity() const -> glm::vec3 {
    return {velocity_.GetX(), velocity_.GetY(), velocity_.GetZ()};
  }

  void set_velocity(const glm::vec3 &v) {
    velocity_ = JPH::Vec3(v.x, v.y, v.z);
  }

  [[nodiscard]] auto is_on_ground() const -> bool {
    return ground_state_;
  }

  void set_max_strength(float) {}

private:
  PhysicsWorld &world_;
  JPH::BodyID body_id_;
  JPH::Vec3 velocity_{0, 0, 0};
  bool ground_state_{false};
};

} // namespace physics
} // namespace engine
