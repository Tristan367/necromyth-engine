#pragma once

#include "renderer/model_loader.hpp"
#include "scene/mesh_instance.hpp"
#include "scene/scene.hpp"

#define GLM_FORCE_RADIANS
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>

namespace engine {

inline void populate_demo_scene(Scene &scene) {
  const LoadedMesh loaded = load_obj_model(ENGINE_MODEL_PATH);
  scene.add_mesh({
      .vertices = loaded.vertices,
      .indices = loaded.indices,
  });

  scene.camera().look_at({2.0F, 2.0F, 2.0F}, {0.0F, 0.0F, 0.0F});
  scene.camera().set_perspective(45.0F, 0.1F, 10.0F);

  scene.add_instance({
      .mesh_index = 0,
      .model = glm::mat4(1.0F),
      .layer = RenderLayer::Opaque,
  });

  scene.add_instance({
      .mesh_index = 0,
      .model = glm::translate(glm::mat4(1.0F), glm::vec3(0.0F, -0.5F, 0.0F)) *
               glm::scale(glm::mat4(1.0F), glm::vec3(0.35F)),
      .layer = RenderLayer::Opaque,
  });
}

[[nodiscard]] inline auto create_demo_scene() -> Scene {
  Scene scene;
  populate_demo_scene(scene);
  return scene;
}

inline void update_demo_scene(Scene &scene) {
  static const auto start_time = std::chrono::high_resolution_clock::now();
  const auto current_time = std::chrono::high_resolution_clock::now();
  const float time = std::chrono::duration<float>(current_time - start_time).count();

  if (scene.instances().size() >= 2)
    scene.instance(1).model = glm::rotate(glm::mat4(1.0F), time * glm::radians(90.0F), glm::vec3(0.0F, 0.0F, 1.0F)) *
                              glm::translate(glm::mat4(1.0F), glm::vec3(0.0F, -0.5F, 0.0F)) *
                              glm::scale(glm::mat4(1.0F), glm::vec3(0.35F));
}

} // namespace engine
