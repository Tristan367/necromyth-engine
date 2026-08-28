#pragma once

#include "renderer/frames_in_flight.hpp"

#include "renderer/texture_image.hpp"
#include "renderer/uniform_buffer.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace engine {

// Set 0: per-frame UBO, texture array, shadow map. Set 1: table texture (Sascha multi-set pattern).
class DescriptorResources {
public:
  void create_layouts(vk::raii::Device &device) {
    const std::array frame_bindings{
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 3,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 4,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 5,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 6,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 7,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
        },
        // Per-draw instance records: model matrix, material selection and bone
        // palette base. Read by every vertex shader.
        vk::DescriptorSetLayoutBinding{
            .binding = 8,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
        },
        // The HUD: one storage buffer of screen-space quads and the pixel-font
        // atlas they sample. In the FRAME set rather than a set of their own
        // because the HUD is per frame and nothing else about it varies -- a
        // second set would be a second thing to allocate, bind and keep in step
        // for one buffer and one texture.
        vk::DescriptorSetLayoutBinding{
            .binding = 9,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 10,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
        },
    };

    const std::array material_bindings{
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
        },
    };

    const std::array material_skinned_bindings{
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
        },
        // The whole shared bone palette. Each instance's slice is selected by
        // an index carried in its instance record, not by a dynamic offset: a
        // dynamic offset is fixed for a whole bind, so it cannot vary across the
        // instances of a single instanced draw.
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
        },
    };

    frame_layout_ = vk::raii::DescriptorSetLayout(
        device,
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<std::uint32_t>(frame_bindings.size()),
            .pBindings = frame_bindings.data(),
        });

    material_layout_ = vk::raii::DescriptorSetLayout(
        device,
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<std::uint32_t>(material_bindings.size()),
            .pBindings = material_bindings.data(),
        });

    material_skinned_layout_ = vk::raii::DescriptorSetLayout(
        device,
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<std::uint32_t>(material_skinned_bindings.size()),
            .pBindings = material_skinned_bindings.data(),
        });
  }

  // `skinned` enables the skinned sets: one per texture for the main pass, plus
  // a single shared set for the shadow pass. This no longer scales with the
  // number of skinned instances -- the bone buffer is shared and indexed by
  // dynamic offset -- so a horde costs the same descriptors as one character.
  void create_pool(
      vk::raii::Device &device,
      std::uint32_t frame_count,
      std::uint32_t texture_count,
      bool skinned = false) {
    if (texture_count == 0)
      throw std::runtime_error("At least one texture is required for descriptor allocation");

    frame_count_ = frame_count;
    texture_count_ = texture_count;

    const std::uint32_t skinned_sets = skinned ? texture_count + 1 : 0;

    const std::array pool_sizes{
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = frame_count,
        },
        vk::DescriptorPoolSize{
            // Four per frame, plus one for the pixel-font atlas at binding 10.
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = frame_count * 5 + texture_count + skinned_sets,
        },
        vk::DescriptorPoolSize{
            // Four per frame, plus one for the HUD's quads at binding 9.
            .type = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = frame_count * 5 + skinned_sets,
        },
    };

    const std::uint32_t max_sets = frame_count + texture_count + skinned_sets;

    descriptor_pool_ = vk::raii::DescriptorPool(
        device,
        vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = max_sets,
            .poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size()),
            .pPoolSizes = pool_sizes.data(),
        });
  }

  void allocate_sets(
      vk::raii::Device &device,
      std::span<const vk::Buffer> uniform_buffers,
      std::span<const TextureImage *const> textures,
      vk::Sampler texture_array_sampler,
      vk::ImageView texture_array_view,
      vk::Sampler shadow_sampler,
      vk::ImageView shadow_view) {
    if (uniform_buffers.size() != frame_count_)
      throw std::runtime_error("Uniform buffer count does not match frame count");
    if (textures.size() != texture_count_)
      throw std::runtime_error("Texture count does not match descriptor allocation");

    frame_sets_.clear();
    material_sets_.clear();

    std::vector<vk::DescriptorSetLayout> frame_layouts(frame_count_, *frame_layout_);
    const vk::DescriptorSetAllocateInfo frame_allocate{
        .descriptorPool = *descriptor_pool_,
        .descriptorSetCount = frame_count_,
        .pSetLayouts = frame_layouts.data(),
    };
    frame_sets_ = device.allocateDescriptorSets(frame_allocate);

    std::vector<vk::DescriptorSetLayout> material_layouts(texture_count_, *material_layout_);
    const vk::DescriptorSetAllocateInfo material_allocate{
        .descriptorPool = *descriptor_pool_,
        .descriptorSetCount = texture_count_,
        .pSetLayouts = material_layouts.data(),
    };
    material_sets_ = device.allocateDescriptorSets(material_allocate);

    const vk::DescriptorImageInfo array_image_info{
        .sampler = texture_array_sampler,
        .imageView = texture_array_view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    const vk::DescriptorImageInfo shadow_image_info{
        .sampler = shadow_sampler,
        .imageView = shadow_view,
        .imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal,
    };

    for (std::uint32_t frame_index = 0; frame_index < frame_count_; ++frame_index) {
      const vk::DescriptorBufferInfo buffer_info{
          .buffer = uniform_buffers[frame_index],
          .offset = 0,
          .range = sizeof(FrameUniformBufferObject),
      };

      const std::array writes{
          vk::WriteDescriptorSet{
              .dstSet = frame_sets_[frame_index],
              .dstBinding = 0,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eUniformBuffer,
              .pBufferInfo = &buffer_info,
          },
          vk::WriteDescriptorSet{
              .dstSet = frame_sets_[frame_index],
              .dstBinding = 1,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo = &array_image_info,
          },
          vk::WriteDescriptorSet{
              .dstSet = frame_sets_[frame_index],
              .dstBinding = 2,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo = &shadow_image_info,
          },
      };

      device.updateDescriptorSets(writes, nullptr);
    }

    for (std::uint32_t texture_index = 0; texture_index < texture_count_; ++texture_index) {
      const vk::DescriptorImageInfo table_image_info{
          .sampler = textures[texture_index]->sampler(),
          .imageView = textures[texture_index]->view(),
          .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
      };

      const std::array writes{
          vk::WriteDescriptorSet{
              .dstSet = material_sets_[texture_index],
              .dstBinding = 0,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo = &table_image_info,
          },
      };

      device.updateDescriptorSets(writes, nullptr);
    }
  }

  [[nodiscard]] auto frame_layout() const -> vk::DescriptorSetLayout {
    return *frame_layout_;
  }

  [[nodiscard]] auto material_layout() const -> vk::DescriptorSetLayout {
    return *material_layout_;
  }

  [[nodiscard]] auto material_skinned_layout() const -> vk::DescriptorSetLayout {
    return *material_skinned_layout_;
  }

  [[nodiscard]] auto pipeline_set_layouts() const -> std::array<vk::DescriptorSetLayout, 2> {
    return {*frame_layout_, *material_layout_};
  }

  [[nodiscard]] auto skinned_pipeline_set_layouts() const -> std::array<vk::DescriptorSetLayout, 2> {
    return {*frame_layout_, *material_skinned_layout_};
  }

  [[nodiscard]] auto frame_set(std::uint32_t frame_index) const -> vk::DescriptorSet {
    return frame_sets_.at(frame_index);
  }

  [[nodiscard]] auto material_set(std::uint32_t texture_index) const -> vk::DescriptorSet {
    return material_sets_.at(texture_index);
  }

  [[nodiscard]] auto has_skinned_sets() const -> bool { return !skinned_sets_.empty(); }

  [[nodiscard]] auto skinned_set(std::uint32_t texture_index) const -> vk::DescriptorSet {
    return *skinned_sets_.at(texture_index % skinned_sets_.size());
  }

  [[nodiscard]] auto shadow_bone_set() const -> vk::DescriptorSet { return *shadow_bone_set_; }

  void update_light_buffers(vk::raii::Device &device,
                            const std::array<vk::Buffer, k_frames_in_flight> &light_buffers) {
    for (std::uint32_t i = 0; i < frame_count_ && i < k_frames_in_flight; ++i) {
      const vk::DescriptorBufferInfo light_info{
          .buffer = light_buffers[i],
          .offset = 0,
          .range = vk::WholeSize,
      };
      device.updateDescriptorSets(
          vk::WriteDescriptorSet{
              .dstSet = frame_sets_[i],
              .dstBinding = 3,
              .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eStorageBuffer,
              .pBufferInfo = &light_info,
          },
          nullptr);
    }
  }

  // One set per texture (main pass) plus one shared set (shadow pass). Each
  // binds the whole bone buffer; the draw picks its instance with a dynamic
  // offset. Nothing here scales with instance count, so spawning or despawning
  // a skinned instance needs no descriptor work at all.
  void allocate_skinned_sets(
      vk::raii::Device &device,
      vk::Buffer bone_buffer,
      vk::DeviceSize bone_range,
      std::span<const TextureImage *const> textures) {
    skinned_sets_.clear();
    shadow_bone_set_ = vk::raii::DescriptorSet{nullptr};
    if (bone_buffer == vk::Buffer{} || textures.empty())
      return;

    const vk::DescriptorBufferInfo bone_info{
        .buffer = bone_buffer,
        .offset = 0,
        .range = bone_range,
    };

    const auto allocate_one = [&]() -> vk::raii::DescriptorSet {
      const vk::DescriptorSetLayout layout = *material_skinned_layout_;
      const vk::DescriptorSetAllocateInfo allocate{
          .descriptorPool = *descriptor_pool_,
          .descriptorSetCount = 1,
          .pSetLayouts = &layout,
      };
      return std::move(device.allocateDescriptorSets(allocate).front());
    };

    const auto write_set = [&](const vk::raii::DescriptorSet &set, const TextureImage &texture) {
      const vk::DescriptorImageInfo image_info{
          .sampler = texture.sampler(),
          .imageView = texture.view(),
          .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
      };
      const std::array writes{
          vk::WriteDescriptorSet{
              .dstSet = *set,
              .dstBinding = 0,
              .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo = &image_info,
          },
          vk::WriteDescriptorSet{
              .dstSet = *set,
              .dstBinding = 1,
              .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eStorageBuffer,
              .pBufferInfo = &bone_info,
          },
      };
      device.updateDescriptorSets(writes, nullptr);
    };

    skinned_sets_.reserve(textures.size());
    for (const TextureImage *texture : textures) {
      skinned_sets_.push_back(allocate_one());
      write_set(skinned_sets_.back(), *texture);
    }

    // The shadow pass is vertex-only; binding 0 exists solely to satisfy the
    // shared layout, so any texture will do.
    shadow_bone_set_ = allocate_one();
    write_set(shadow_bone_set_, *textures[0]);
  }

  void update_spot_shadow_sampler(vk::raii::Device &device, vk::Sampler sampler, vk::ImageView view) {
    const vk::DescriptorImageInfo info{
        .sampler = sampler,
        .imageView = view,
        .imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal,
    };
    for (const vk::raii::DescriptorSet &set : frame_sets_) {
      device.updateDescriptorSets(
          vk::WriteDescriptorSet{
              .dstSet = *set, .dstBinding = 4, .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo = &info,
          },
          nullptr);
    }
  }

  void update_point_cube_sampler(vk::raii::Device &device, vk::Sampler sampler, vk::ImageView view) {
    const vk::DescriptorImageInfo info{
        .sampler = sampler,
        .imageView = view,
        .imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal,
    };
    for (const vk::raii::DescriptorSet &set : frame_sets_) {
      device.updateDescriptorSets(
          vk::WriteDescriptorSet{
              .dstSet = *set, .dstBinding = 5, .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo = &info,
          },
          nullptr);
    }
  }

  void update_point_light_shadow_ssbo(
      vk::raii::Device &device,
      std::span<const vk::Buffer> buffers) {
    for (std::size_t i = 0; i < frame_sets_.size() && i < buffers.size(); ++i) {
      const vk::DescriptorBufferInfo info{
          .buffer = buffers[i],
          .offset = 0,
          .range = vk::WholeSize,
      };
      device.updateDescriptorSets(
          vk::WriteDescriptorSet{
              .dstSet = *frame_sets_[i], .dstBinding = 6, .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eStorageBuffer,
              .pBufferInfo = &info,
          },
          nullptr);
    }
  }

  void update_instance_buffers(vk::raii::Device &device, std::span<const vk::Buffer> buffers) {
    for (std::size_t i = 0; i < frame_sets_.size() && i < buffers.size(); ++i) {
      if (buffers[i] == vk::Buffer{})
        continue;
      const vk::DescriptorBufferInfo info{.buffer = buffers[i], .offset = 0, .range = vk::WholeSize};
      device.updateDescriptorSets(
          vk::WriteDescriptorSet{
              .dstSet = *frame_sets_[i], .dstBinding = 8, .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eStorageBuffer,
              .pBufferInfo = &info,
          },
          nullptr);
    }
  }

  // The HUD's quads and the pixel-font atlas, bindings 9 and 10.
  //
  // Written for EVERY frame set even when the HUD has nothing to draw. A
  // descriptor a bound set declares must point at something valid whether or not
  // any shader reads it, and "the HUD happened to be empty this frame" is not a
  // state the validation layer distinguishes from a bug.
  void update_ui(vk::raii::Device &device, std::span<const vk::Buffer> buffers,
                 vk::Sampler atlas_sampler, vk::ImageView atlas_view) {
    for (std::size_t i = 0; i < frame_sets_.size() && i < buffers.size(); ++i) {
      if (buffers[i] == vk::Buffer{})
        continue;
      const vk::DescriptorBufferInfo quads{
          .buffer = buffers[i], .offset = 0, .range = vk::WholeSize};
      const vk::DescriptorImageInfo atlas{
          .sampler = atlas_sampler,
          .imageView = atlas_view,
          .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
      };
      const std::array writes{
          vk::WriteDescriptorSet{
              .dstSet = *frame_sets_[i], .dstBinding = 9, .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eStorageBuffer,
              .pBufferInfo = &quads,
          },
          vk::WriteDescriptorSet{
              .dstSet = *frame_sets_[i], .dstBinding = 10, .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo = &atlas,
          },
      };
      device.updateDescriptorSets(writes, nullptr);
    }
  }

  void update_particle_ssbo(vk::raii::Device &device,
                            std::span<const vk::Buffer> buffers) {
    for (std::size_t i = 0; i < frame_sets_.size() && i < buffers.size(); ++i) {
      const vk::DescriptorBufferInfo info{
          .buffer = buffers[i], .offset = 0, .range = vk::WholeSize};
      device.updateDescriptorSets(
          vk::WriteDescriptorSet{
              .dstSet = *frame_sets_[i], .dstBinding = 7, .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eStorageBuffer,
              .pBufferInfo = &info,
          },
          nullptr);
    }
  }

private:
  std::uint32_t frame_count_{};
  std::uint32_t texture_count_{};
  vk::raii::DescriptorSetLayout frame_layout_{nullptr};
  vk::raii::DescriptorSetLayout material_layout_{nullptr};
  vk::raii::DescriptorSetLayout material_skinned_layout_{nullptr};
  vk::raii::DescriptorPool descriptor_pool_{nullptr};
  std::vector<vk::raii::DescriptorSet> frame_sets_;
  std::vector<vk::raii::DescriptorSet> material_sets_;
  std::vector<vk::raii::DescriptorSet> skinned_sets_;
  vk::raii::DescriptorSet shadow_bone_set_{nullptr};
};

} // namespace engine
