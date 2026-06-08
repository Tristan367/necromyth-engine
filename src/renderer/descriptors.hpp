#pragma once

#include "renderer/uniform_buffer.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <cstdint>
#include <span>
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
    };

    descriptor_set_layout_ = vk::raii::DescriptorSetLayout(
        device,
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<std::uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        });
  }

  void create_pool(vk::raii::Device &device, std::uint32_t frame_count) {
    const std::array pool_sizes{
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = frame_count,
        },
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = frame_count,
        },
    };

    descriptor_pool_ = vk::raii::DescriptorPool(
        device,
        vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = frame_count,
            .poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size()),
            .pPoolSizes = pool_sizes.data(),
        });
  }

  void allocate_sets(
      vk::raii::Device &device,
      std::span<const vk::Buffer> uniform_buffers,
      vk::Sampler texture_sampler,
      vk::ImageView texture_view) {
    std::vector<vk::DescriptorSetLayout> layouts(uniform_buffers.size(), *descriptor_set_layout_);

    const vk::DescriptorSetAllocateInfo allocate_info{
        .descriptorPool = *descriptor_pool_,
        .descriptorSetCount = static_cast<std::uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data(),
    };

    descriptor_sets_ = device.allocateDescriptorSets(allocate_info);

    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(uniform_buffers.size()); ++i) {
      const vk::DescriptorBufferInfo buffer_info{
          .buffer = uniform_buffers[i],
          .offset = 0,
          .range = sizeof(FrameUniformBufferObject),
      };
      const vk::DescriptorImageInfo image_info{
          .sampler = texture_sampler,
          .imageView = texture_view,
          .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
      };

      const std::array writes{
          vk::WriteDescriptorSet{
              .dstSet = descriptor_sets_[i],
              .dstBinding = 0,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eUniformBuffer,
              .pBufferInfo = &buffer_info,
          },
          vk::WriteDescriptorSet{
              .dstSet = descriptor_sets_[i],
              .dstBinding = 1,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo = &image_info,
          },
      };

      device.updateDescriptorSets(writes, nullptr);
    }
  }

  [[nodiscard]] auto layout() const -> vk::DescriptorSetLayout {
    return *descriptor_set_layout_;
  }

  [[nodiscard]] auto set(std::uint32_t frame_index) const -> vk::DescriptorSet {
    return descriptor_sets_[frame_index];
  }

private:
  vk::raii::DescriptorSetLayout descriptor_set_layout_{nullptr};
  vk::raii::DescriptorPool descriptor_pool_{nullptr};
  std::vector<vk::raii::DescriptorSet> descriptor_sets_;
};

} // namespace engine
