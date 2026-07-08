#pragma once

#include "renderer/buffer.hpp"
#include "renderer/gpu_particle.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace engine {

class ParticleSystem {
public:
  struct Particle {
    glm::vec3 pos{};
    glm::vec3 vel{};
    float age{0.0F};
    bool alive{false};
    float lifetime{1.0F};
  };

  struct Emitter {
    glm::vec3 position{};
    float rate{10.0F}; // particles per second
    float accumulator{0.0F};
    // Called when a new particle spawns. Set pos, vel, lifetime.
    std::function<void(Particle &)> on_emit;
    // Called each frame per alive particle. Return false to recycle.
    std::function<bool(Particle &, float dt)> on_update;
  };

  ParticleSystem() = default;

  void create(const vk::raii::PhysicalDevice &physical_device,
              vk::raii::Device &device, std::uint32_t max_particles,
              std::uint32_t frame_count = 2) {
    max_particles_ = max_particles;
    particles_.resize(max_particles);
    free_list_.reserve(max_particles);
    for (std::uint32_t i = 0; i < max_particles; ++i)
      free_list_.push_back(max_particles - 1 - i);

    const vk::DeviceSize buf_size = max_particles * sizeof(GpuParticle);
    const auto memory_properties = physical_device.getMemoryProperties();
    buffers_.reserve(frame_count);
    memories_.reserve(frame_count);
    mapped_.reserve(frame_count);

    for (std::uint32_t i = 0; i < frame_count; ++i) {
      vk::BufferCreateInfo buf_info{};
      buf_info.size = buf_size;
      buf_info.usage = vk::BufferUsageFlagBits::eStorageBuffer;
      buf_info.sharingMode = vk::SharingMode::eExclusive;
      buffers_.emplace_back(device, buf_info);

      const auto reqs = buffers_.back().getMemoryRequirements();
      const auto mt = detail::find_memory_type(
          memory_properties, reqs.memoryTypeBits,
          vk::MemoryPropertyFlagBits::eHostVisible |
              vk::MemoryPropertyFlagBits::eHostCoherent);
      vk::MemoryAllocateInfo alloc{};
      alloc.allocationSize = reqs.size;
      alloc.memoryTypeIndex = mt;
      memories_.emplace_back(device, alloc);
      buffers_.back().bindMemory(*memories_.back(), 0);
      mapped_.push_back(static_cast<GpuParticle *>(
          memories_.back().mapMemory(0, buf_size)));
    }
  }

  auto add_emitter(const Emitter &e) -> std::uint32_t {
    auto idx = static_cast<std::uint32_t>(emitters_.size());
    emitters_.push_back(e);
    return idx;
  }

  auto emitter(std::uint32_t index) -> Emitter & {
    return emitters_[index];
  }

  void update(float dt) {
    for (auto &e : emitters_) {
      if (e.on_update) {
        for (auto &p : particles_) {
          if (!p.alive) continue;
          if (!e.on_update(p, dt)) {
            p.alive = false;
            free_list_.push_back(static_cast<std::uint32_t>(&p - particles_.data()));
          }
        }
      }

      if (e.rate <= 0.0F) continue;

      e.accumulator += dt * e.rate;
      while (e.accumulator >= 1.0F && !free_list_.empty()) {
        e.accumulator -= 1.0F;
        std::uint32_t idx = free_list_.back();
        free_list_.pop_back();
        Particle &p = particles_[idx];
        p.pos = e.position;
        p.vel = glm::vec3{};
        p.age = 0.0F;
        p.lifetime = 1.0F;
        if (e.on_emit) e.on_emit(p);
        p.alive = true;
      }
    }

    active_count_ = 0;
    for (const auto &p : particles_)
      if (p.alive) ++active_count_;
  }

  void upload(std::uint32_t frame_index) const {
    if (frame_index >= mapped_.size()) return;
    auto *dst = mapped_[frame_index];
    std::uint32_t out = 0;
    for (const auto &p : particles_) {
      if (!p.alive) continue;
      if (out >= max_particles_) break;
      dst[out++].pos = glm::vec4(p.pos, 0.0F);
    }
  }

  [[nodiscard]] auto buffer(std::uint32_t frame_index) const -> vk::Buffer {
    return *buffers_[frame_index];
  }

  [[nodiscard]] auto active_count() const -> std::uint32_t {
    return active_count_;
  }

private:
  std::vector<Particle> particles_;
  std::vector<std::uint32_t> free_list_;
  std::vector<Emitter> emitters_;
  std::uint32_t active_count_{0};
  std::uint32_t max_particles_{0};

  std::vector<vk::raii::Buffer> buffers_;
  std::vector<vk::raii::DeviceMemory> memories_;
  std::vector<GpuParticle *> mapped_;
};

} // namespace engine
