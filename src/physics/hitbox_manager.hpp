#pragma once

#include "physics/physics_world.hpp"
#include "scene/animation_types.hpp"

#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace engine {
namespace physics {

class HitboxManager {
public:
  struct HitboxBody {
    JPH::BodyID body_id;
    std::string name;
  };

  HitboxManager(PhysicsWorld &world, const engine::SkeletonAsset &skeleton)
      : world_(world) {
    for (const engine::HitboxAttachment &hb : skeleton.hitboxes) {
      JPH::ShapeRefC shape;
      switch (hb.shape) {
      case engine::HitboxShape::Box: {
        JPH::BoxShapeSettings s(
            JPH::Vec3(hb.half_extent.x, hb.half_extent.y, hb.half_extent.z));
        s.SetEmbedded();
        shape = s.Create().Get();
        break;
      }
      case engine::HitboxShape::Sphere: {
        JPH::SphereShapeSettings s(hb.half_extent.x);
        s.SetEmbedded();
        shape = s.Create().Get();
        break;
      }
      case engine::HitboxShape::Capsule: {
        JPH::CapsuleShapeSettings s(hb.half_height, hb.half_extent.x);
        s.SetEmbedded();
        shape = s.Create().Get();
        break;
      }
      }

      JPH::BodyCreationSettings body_settings(
          shape, JPH::RVec3::sZero(), JPH::Quat::sIdentity(),
          JPH::EMotionType::Kinematic, Layers::kHitbox);
      body_settings.mIsSensor = true;

      JPH::BodyID id = world_.body_interface().CreateAndAddBody(
          body_settings, JPH::EActivation::DontActivate);
      bodies_.push_back({id, hb.name});
    }
  }

  void update(const engine::SkeletonAsset &skeleton,
              const std::vector<glm::mat4> &bone_worlds) {
    for (std::size_t i = 0; i < bodies_.size() && i < skeleton.hitboxes.size(); ++i) {
      const engine::HitboxAttachment &hb = skeleton.hitboxes[i];
      if (hb.joint_index >= bone_worlds.size()) continue;

      const glm::mat4 bone_world = bone_worlds[hb.joint_index];
      const glm::mat4 local = glm::translate(glm::mat4(1.0F), hb.offset) *
                               glm::mat4_cast(hb.rotation);
      const glm::mat4 world = bone_world * local;

      glm::vec3 pos(world[3]);
      const glm::quat rot = glm::quat_cast(world);

      world_.set_sensor_transform(bodies_[i].body_id, pos, rot);
    }
  }

  [[nodiscard]] auto hitbox_bodies() const -> const std::vector<HitboxBody> & { return bodies_; }
  [[nodiscard]] auto find_name(JPH::BodyID id) const -> const std::string * {
    for (const HitboxBody &b : bodies_)
      if (b.body_id == id)
        return &b.name;
    return nullptr;
  }

private:
  PhysicsWorld &world_;
  std::vector<HitboxBody> bodies_;
};

} // namespace physics
} // namespace engine
