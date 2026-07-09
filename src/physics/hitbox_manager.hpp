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
    std::uint32_t joint_index{};
    glm::vec3 offset{0.0F};
    glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
    bool active{true};
  };

  explicit HitboxManager(PhysicsWorld &world) : world_(world) {}

  ~HitboxManager() {
    JPH::BodyInterface &bi = world_.body_interface();
    for (const HitboxBody &b : bodies_) {
      bi.RemoveBody(b.body_id);
      bi.DestroyBody(b.body_id);
    }
  }

  void add_hitboxes_from_json(const engine::SkeletonAsset &skeleton) {
    for (const engine::HitboxAttachment &hb : skeleton.hitboxes) {
      const std::uint32_t idx = add_hitbox(hb.name, hb);
      (void)idx;
    }
  }

  auto add_hitbox(const std::string &name, const engine::HitboxAttachment &hb,
                  JPH::ObjectLayer layer = Layers::kHitbox) -> std::uint32_t {
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
        JPH::EMotionType::Kinematic, layer);
    body_settings.mIsSensor = true;

    JPH::BodyID id = world_.body_interface().CreateAndAddBody(
        body_settings, JPH::EActivation::DontActivate);

    bodies_.push_back({id, name, hb.joint_index, hb.offset, hb.rotation, true});
    return static_cast<std::uint32_t>(bodies_.size() - 1);
  }

  void remove_hitbox(std::uint32_t index) {
    if (index >= bodies_.size()) return;
    JPH::BodyInterface &bi = world_.body_interface();
    bi.RemoveBody(bodies_[index].body_id);
    bi.DestroyBody(bodies_[index].body_id);
    bodies_.erase(bodies_.begin() + static_cast<std::ptrdiff_t>(index));
  }

  void set_active(std::uint32_t index, bool active) {
    if (index >= bodies_.size()) return;
    bodies_[index].active = active;
  }

  void update(const std::vector<glm::mat4> &bone_worlds) {
    for (const HitboxBody &hb : bodies_) {
      if (!hb.active) continue;
      if (hb.joint_index >= bone_worlds.size()) continue;

      const glm::mat4 bone_world = bone_worlds[hb.joint_index];
      const glm::mat4 local = glm::translate(glm::mat4(1.0F), hb.offset) *
                               glm::mat4_cast(hb.rotation);
      const glm::mat4 world = bone_world * local;

      glm::vec3 pos(world[3]);
      const glm::quat rot = glm::quat_cast(world);

      world_.set_sensor_transform(hb.body_id, pos, rot);
    }
  }

  [[nodiscard]] auto get_active_overlaps(std::uint32_t index) const -> std::vector<JPH::BodyID> {
    if (index >= bodies_.size()) return {};
    if (!bodies_[index].active) return {};
    return world_.get_sensor_overlaps(bodies_[index].body_id);
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
