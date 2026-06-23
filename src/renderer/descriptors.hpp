#pragma once

#include "renderer/texture_image.hpp"
#include "renderer/texture_array.hpp"
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

  void create_pool(
      vk::raii::Device &device,
      std::uint32_t frame_count,
      std::uint32_t texture_count,
      std::uint32_t skinned_instance_count = 0) {
    if (texture_count == 0)
      throw std::runtime_error("At least one texture is required for descriptor allocation");

    frame_count_ = frame_count;
    texture_count_ = texture_count;
    skinned_instance_count_ = skinned_instance_count;

    const std::array pool_sizes{
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = frame_count,
        },
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = frame_count * 2 + texture_count + skinned_instance_count * 2,
        },
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = skinned_instance_count * 4,
        },
    };

    const std::uint32_t max_sets = frame_count + texture_count + skinned_instance_count * 4;

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

  [[nodiscard]] auto skinned_set(std::uint32_t instance, std::uint32_t frame) const -> vk::DescriptorSet {
    return skinned_sets_.at(static_cast<std::size_t>(instance) * 2 + frame);
  }

  [[nodiscard]] auto shadow_bone_set(std::uint32_t instance, std::uint32_t frame) const -> vk::DescriptorSet {
    return shadow_bone_sets_.at(static_cast<std::size_t>(instance) * 2 + frame);
  }

  void allocate_skinned_sets(
      vk::raii::Device &device,
      std::span<const vk::Buffer> bone_buffers,
      std::span<const TextureImage *const> textures,
      std::span<const std::uint32_t> texture_indices) {
    if (bone_buffers.empty())
      return;

    const std::uint32_t instance_count = static_cast<std::uint32_t>(bone_buffers.size()) / 2;
    skinned_sets_.clear();
    shadow_bone_sets_.clear();
    skinned_sets_.reserve(instance_count * 2);
    shadow_bone_sets_.reserve(instance_count * 2);

    for (std::uint32_t i = 0; i < instance_count; ++i) {
      const std::uint32_t tex_idx = i < texture_indices.size()
          ? texture_indices[i]
          : 0U;

      const vk::DescriptorImageInfo table_image_info{
          .sampler = textures[tex_idx % textures.size()]->sampler(),
          .imageView = textures[tex_idx % textures.size()]->view(),
          .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
      };

      for (std::uint32_t frame = 0; frame < 2; ++frame) {
        const vk::DescriptorBufferInfo bone_info{
            .buffer = bone_buffers[i * 2 + frame],
            .offset = 0,
            .range = VK_WHOLE_SIZE,
        };

        // Main pass skinned set (texture + bone SSBO)
        {
          const vk::DescriptorSetLayout layout = *material_skinned_layout_;
          const vk::DescriptorSetAllocateInfo allocate{
              .descriptorPool = *descriptor_pool_,
              .descriptorSetCount = 1,
              .pSetLayouts = &layout,
          };
          skinned_sets_.push_back(std::move(device.allocateDescriptorSets(allocate).front()));

          const std::array writes{
              vk::WriteDescriptorSet{
                  .dstSet = skinned_sets_.back(),
                  .dstBinding = 0,
                  .descriptorCount = 1,
                  .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                  .pImageInfo = &table_image_info,
              },
              vk::WriteDescriptorSet{
                  .dstSet = skinned_sets_.back(),
                  .dstBinding = 1,
                  .descriptorCount = 1,
                  .descriptorType = vk::DescriptorType::eStorageBuffer,
                  .pBufferInfo = &bone_info,
              },
          };
          device.updateDescriptorSets(writes, nullptr);
        }

        // Shadow pass skinned set (dummy texture + bone SSBO)
        {
          const vk::DescriptorSetLayout layout = *material_skinned_layout_;
          const vk::DescriptorSetAllocateInfo allocate{
              .descriptorPool = *descriptor_pool_,
              .descriptorSetCount = 1,
              .pSetLayouts = &layout,
          };
          shadow_bone_sets_.push_back(std::move(device.allocateDescriptorSets(allocate).front()));

          const vk::DescriptorImageInfo dummy_sampler_info{
              .sampler = textures[0]->sampler(),
              .imageView = textures[0]->view(),
              .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
          };

          const std::array writes{
              vk::WriteDescriptorSet{
                  .dstSet = shadow_bone_sets_.back(),
                  .dstBinding = 0,
                  .descriptorCount = 1,
                  .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                  .pImageInfo = &dummy_sampler_info,
              },
              vk::WriteDescriptorSet{
                  .dstSet = shadow_bone_sets_.back(),
                  .dstBinding = 1,
                  .descriptorCount = 1,
                  .descriptorType = vk::DescriptorType::eStorageBuffer,
                  .pBufferInfo = &bone_info,
              },
          };
          device.updateDescriptorSets(writes, nullptr);
        }
      }
    }
  }

private:
  std::uint32_t frame_count_{};
  std::uint32_t texture_count_{};
  std::uint32_t skinned_instance_count_{};
  vk::raii::DescriptorSetLayout frame_layout_{nullptr};
  vk::raii::DescriptorSetLayout material_layout_{nullptr};
  vk::raii::DescriptorSetLayout material_skinned_layout_{nullptr};
  vk::raii::DescriptorPool descriptor_pool_{nullptr};
  std::vector<vk::raii::DescriptorSet> frame_sets_;
  std::vector<vk::raii::DescriptorSet> material_sets_;
  std::vector<vk::raii::DescriptorSet> skinned_sets_;
  std::vector<vk::raii::DescriptorSet> shadow_bone_sets_;
};

} // namespace engine
