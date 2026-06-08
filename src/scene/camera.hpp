#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace engine {

class Camera {
public:
  void set_aspect(float aspect) {
    if (aspect_ != aspect) {
      aspect_ = aspect;
      proj_dirty_ = true;
    }
  }

  void look_at(const glm::vec3 &eye, const glm::vec3 &center, const glm::vec3 &up = {0.0F, 1.0F, 0.0F}) {
    position_ = eye;
    target_ = center;
    up_ = up;
    view_dirty_ = true;
  }

  void set_perspective(float fov_y_degrees, float near_plane, float far_plane) {
    fov_y_degrees_ = fov_y_degrees;
    near_plane_ = near_plane;
    far_plane_ = far_plane;
    proj_dirty_ = true;
  }

  [[nodiscard]] auto position() const -> const glm::vec3 & {
    return position_;
  }

  [[nodiscard]] auto look_direction() const -> glm::vec3 {
    return glm::normalize(target_ - position_);
  }

  [[nodiscard]] auto view_matrix() -> const glm::mat4 & {
    if (view_dirty_)
      rebuild_view();
    return view_;
  }

  [[nodiscard]] auto projection_matrix() -> const glm::mat4 & {
    if (proj_dirty_)
      rebuild_projection();
    return proj_;
  }

private:
  void rebuild_view() {
    view_ = glm::lookAt(position_, target_, up_);
    view_dirty_ = false;
  }

  void rebuild_projection() {
    proj_ = glm::perspective(glm::radians(fov_y_degrees_), aspect_, near_plane_, far_plane_);
    proj_[1][1] *= -1.0F;
    proj_dirty_ = false;
  }

  glm::vec3 position_{2.0F, 2.0F, 2.0F};
  glm::vec3 target_{};
  glm::vec3 up_{0.0F, 1.0F, 0.0F};
  float fov_y_degrees_{45.0F};
  float near_plane_{0.05F};
  float far_plane_{2000.0F};
  float aspect_{1.0F};
  glm::mat4 view_{1.0F};
  glm::mat4 proj_{1.0F};
  bool view_dirty_{true};
  bool proj_dirty_{true};
};

} // namespace engine
