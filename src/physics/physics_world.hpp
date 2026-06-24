#pragma once

#include "scene/mesh_instance.hpp"
#include "scene/scene.hpp"

#include <Jolt/Jolt.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#define GLM_FORCE_RADIANS
#include <glm/gtc/type_ptr.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
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
    JPH::RegisterDefaultAllocator();
    JPH::Factory::sInstance = new JPH::Factory();
    JPH::RegisterTypes();

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
    // Remove all bodies before destroying
    for (const JPH::BodyID id : body_ids_)
      body_interface_->RemoveBody(id);
    for (const JPH::BodyID id : body_ids_)
      body_interface_->DestroyBody(id);

    JPH::UnregisterTypes();
    delete JPH::Factory::sInstance;
    JPH::Factory::sInstance = nullptr;
  }

  void step(float delta_time) {
    physics_system_.Update(delta_time, 1, temp_allocator_.get(), job_system_.get());
  }

  [[nodiscard]] auto create_static_box(const glm::vec3 &half_extent, const glm::vec3 &position,
                                        const glm::quat &rotation = glm::quat(1.0F, 0.0F, 0.0F, 0.0F))
      -> JPH::BodyID {
    JPH::BoxShapeSettings shape_settings(JPH::Vec3(half_extent.x, half_extent.y, half_extent.z));
    shape_settings.SetEmbedded();
    JPH::ShapeRefC shape = shape_settings.Create().Get();

    const JPH::BodyCreationSettings settings(
        shape,
        JPH::RVec3(position.x, position.y, position.z),
        JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w),
        JPH::EMotionType::Static,
        Layers::kNonMoving);

    JPH::BodyID id = body_interface_->CreateAndAddBody(settings, JPH::EActivation::DontActivate);
    body_ids_.push_back(id);
    return id;
  }

  [[nodiscard]] auto create_dynamic_box(const glm::vec3 &half_extent, const glm::vec3 &position,
                                         const glm::quat &rotation = glm::quat(1.0F, 0.0F, 0.0F, 0.0F))
      -> JPH::BodyID {
    JPH::BoxShapeSettings shape_settings(JPH::Vec3(half_extent.x, half_extent.y, half_extent.z));
    shape_settings.SetEmbedded();
    JPH::ShapeRefC shape = shape_settings.Create().Get();

    const JPH::BodyCreationSettings settings(
        shape,
        JPH::RVec3(position.x, position.y, position.z),
        JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w),
        JPH::EMotionType::Dynamic,
        Layers::kMoving);

    JPH::BodyID id = body_interface_->CreateAndAddBody(settings, JPH::EActivation::Activate);
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

private:
  JPH::PhysicsSystem physics_system_;
  JPH::BodyInterface *body_interface_{nullptr};
  std::unique_ptr<JPH::TempAllocatorImpl> temp_allocator_;
  std::unique_ptr<JPH::JobSystemThreadPool> job_system_;
  std::vector<JPH::BodyID> body_ids_;

  // Layer interfaces
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

} // namespace physics
} // namespace engine
