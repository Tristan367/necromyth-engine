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

class DescriptorResources {
public:
  void create_layout(vk::raii::Device &device) {
    const std::array bindings{
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

    descriptor_set_layout_ = vk::raii::DescriptorSetLayout(
        device,
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<std::uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        });
  }

  void create_pool(vk::raii::Device &device, std::uint32_t frame_count, std::uint32_t texture_count) {
    if (texture_count == 0)
      throw std::runtime_error("At least one texture is required for descriptor allocation");

    frame_count_ = frame_count;
    texture_count_ = texture_count;
    const std::uint32_t set_count = frame_count * texture_count;

    const std::array pool_sizes{
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = set_count,
        },
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = set_count * 2,
        },
    };

    descriptor_pool_ = vk::raii::DescriptorPool(
        device,
        vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = set_count,
            .poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size()),
            .pPoolSizes = pool_sizes.data(),
        });
  }

  void allocate_sets(
      vk::raii::Device &device,
      std::span<const vk::Buffer> uniform_buffers,
      std::span<const TextureImage *const> textures,
      vk::Sampler texture_array_sampler,
      vk::ImageView texture_array_view) {
    if (uniform_buffers.size() != frame_count_)
      throw std::runtime_error("Uniform buffer count does not match frame count");
    if (textures.size() != texture_count_)
      throw std::runtime_error("Texture count does not match descriptor allocation");

    const std::uint32_t set_count = frame_count_ * texture_count_;
    std::vector<vk::DescriptorSetLayout> layouts(set_count, *descriptor_set_layout_);

    const vk::DescriptorSetAllocateInfo allocate_info{
        .descriptorPool = *descriptor_pool_,
        .descriptorSetCount = set_count,
        .pSetLayouts = layouts.data(),
    };

    auto allocated = device.allocateDescriptorSets(allocate_info);
    descriptor_sets_ = std::move(allocated);

    const vk::DescriptorImageInfo array_image_info{
        .sampler = texture_array_sampler,
        .imageView = texture_array_view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    for (std::uint32_t texture_index = 0; texture_index < texture_count_; ++texture_index) {
      for (std::uint32_t frame_index = 0; frame_index < frame_count_; ++frame_index) {
        const vk::DescriptorBufferInfo buffer_info{
            .buffer = uniform_buffers[frame_index],
            .offset = 0,
            .range = sizeof(FrameUniformBufferObject),
        };
        const vk::DescriptorImageInfo table_image_info{
            .sampler = textures[texture_index]->sampler(),
            .imageView = textures[texture_index]->view(),
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        const vk::DescriptorSet descriptor_set = descriptor_sets_[set_index(texture_index, frame_index)];
        const std::array writes{
            vk::WriteDescriptorSet{
                .dstSet = descriptor_set,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &buffer_info,
            },
            vk::WriteDescriptorSet{
                .dstSet = descriptor_set,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &table_image_info,
            },
            vk::WriteDescriptorSet{
                .dstSet = descriptor_set,
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &array_image_info,
            },
        };

        device.updateDescriptorSets(writes, nullptr);
      }
    }
  }

  [[nodiscard]] auto layout() const -> vk::DescriptorSetLayout {
    return *descriptor_set_layout_;
  }

  [[nodiscard]] auto set(std::uint32_t frame_index, std::uint32_t texture_index) const -> vk::DescriptorSet {
    return descriptor_sets_.at(set_index(texture_index, frame_index));
  }

private:
  [[nodiscard]] auto set_index(std::uint32_t texture_index, std::uint32_t frame_index) const -> std::size_t {
    return static_cast<std::size_t>(texture_index) * frame_count_ + frame_index;
  }

  std::uint32_t frame_count_{};
  std::uint32_t texture_count_{};
  vk::raii::DescriptorSetLayout descriptor_set_layout_{nullptr};
  vk::raii::DescriptorPool descriptor_pool_{nullptr};
  std::vector<vk::raii::DescriptorSet> descriptor_sets_;
};

} // namespace engine
