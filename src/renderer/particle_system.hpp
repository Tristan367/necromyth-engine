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
    float lifetime{1.0F};
    // Written by on_emit and free to change every frame in on_update, which is
    // what lets a flame cool from yellow to red as it rises and a spark shrink
    // as it dies. Both are per particle; see gpu_particle.hpp.
    glm::vec4 color{1.0F};
    float size{0.1F};
    std::uint32_t emitted_by{};
  };

  struct Emitter {
    glm::vec3 position{};
    float rate{10.0F}; // particles per second
    float accumulator{0.0F};
    // What a particle from this emitter starts as. on_emit may overwrite both,
    // and usually should -- these are the default so an emitter that only wants
    // one colour does not need a callback to say so.
    glm::vec4 color{1.0F};
    float size{0.1F};
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
    // Reserved once and never grown past. Everything below relies on the
    // vector not reallocating, and on it holding only LIVE particles: the
    // update loop, the upload and the active count are all just its size.
    particles_.reserve(max_particles);

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

  // Slots are recycled, and that is load-bearing rather than tidy.
  //
  // A caller with one emitter per burning voxel or per bleeding mob adds and
  // removes them constantly, and an emitter list that only ever grows is a leak
  // with a per-frame cost attached: update() walks every emitter every frame,
  // so ten thousand dead ones is ten thousand branches a frame forever.
  auto add_emitter(const Emitter &e) -> std::uint32_t {
    if (!free_emitters_.empty()) {
      const std::uint32_t idx = free_emitters_.back();
      free_emitters_.pop_back();
      emitters_[idx] = e;
      return idx;
    }
    auto idx = static_cast<std::uint32_t>(emitters_.size());
    emitters_.push_back(e);
    return idx;
  }

  auto emitter(std::uint32_t index) -> Emitter & {
    return emitters_.at(index);
  }

  void remove_emitter(std::uint32_t index) {
    if (index >= emitters_.size())
      return;
    // Its particles go with it. They are indexed by emitter, so leaving them
    // alive would hand them to whoever takes the slot next -- a flame that
    // outlives its fire and then starts being updated as somebody's rain.
    for (std::size_t i = 0; i < particles_.size();) {
      if (particles_[i].emitted_by == index)
        remove_at(i);
      else
        ++i;
    }
    emitters_[index] = {};
    emitters_[index].rate = 0.0F;
    free_emitters_.push_back(index);
  }

  // One pass over the live particles, then one over the emitters.
  //
  // It used to be one pass over ALL particle slots PER EMITTER -- so a hundred
  // burning voxels meant a hundred walks of a sixty-five-thousand-element array
  // every frame, six and a half million iterations to move a few hundred
  // sparks. The particle array is dense and holds only live particles now, so
  // the work is proportional to what is actually on screen, and a world with
  // nothing alight in it does nothing at all.
  void update(float dt) {
    for (std::size_t i = 0; i < particles_.size();) {
      Particle &p = particles_[i];
      Emitter &e = emitters_[p.emitted_by];
      if (e.on_update && !e.on_update(p, dt))
        remove_at(i);
      else
        ++i;
    }

    for (std::size_t eidx = 0; eidx < emitters_.size(); ++eidx) {
      Emitter &e = emitters_[eidx];
      if (e.rate <= 0.0F)
        continue;

      e.accumulator += dt * e.rate;
      while (e.accumulator >= 1.0F && particles_.size() < max_particles_) {
        e.accumulator -= 1.0F;
        Particle p{};
        p.pos = e.position;
        p.age = 0.0F;
        p.lifetime = 1.0F;
        p.color = e.color;
        p.size = e.size;
        p.emitted_by = static_cast<std::uint32_t>(eidx);
        if (e.on_emit)
          e.on_emit(p);
        particles_.push_back(p);
      }
      // Whatever could not be spawned this frame is dropped rather than owed.
      // Carrying the debt means that the instant a slot frees up, every emitter
      // that has been starved fires at once.
      if (particles_.size() >= max_particles_)
        e.accumulator = 0.0F;
    }
  }

  void upload(std::uint32_t frame_index) const {
    if (frame_index >= mapped_.size()) return;
    auto *dst = mapped_[frame_index];
    for (std::size_t i = 0; i < particles_.size(); ++i) {
      dst[i].pos_size = glm::vec4(particles_[i].pos, particles_[i].size);
      dst[i].color = particles_[i].color;
    }
  }

  [[nodiscard]] auto buffer(std::uint32_t frame_index) const -> vk::Buffer {
    return *buffers_[frame_index];
  }

  [[nodiscard]] auto active_count() const -> std::uint32_t {
    return static_cast<std::uint32_t>(particles_.size());
  }

private:
  // Swap-remove. Order does not matter to a cloud of sparks, and it keeps the
  // array dense in constant time.
  void remove_at(std::size_t i) {
    particles_[i] = particles_.back();
    particles_.pop_back();
  }

  std::vector<Particle> particles_;
  std::vector<Emitter> emitters_;
  std::vector<std::uint32_t> free_emitters_;
  std::uint32_t max_particles_{0};

  std::vector<vk::raii::Buffer> buffers_;
  std::vector<vk::raii::DeviceMemory> memories_;
  std::vector<GpuParticle *> mapped_;
};

} // namespace engine
