#pragma once

#include "renderer/uniform_buffer.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace engine {

class DescriptorResources {
public:
  void create_layout(vk::raii::Device &device) {
    const vk::DescriptorSetLayoutBinding ubo_binding{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
    };

    descriptor_set_layout_ = vk::raii::DescriptorSetLayout(
        device,
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = 1,
            .pBindings = &ubo_binding,
        });
  }

  void create_pool(vk::raii::Device &device, std::uint32_t frame_count) {
    const vk::DescriptorPoolSize pool_size{
        .type = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = frame_count,
    };

    descriptor_pool_ = vk::raii::DescriptorPool(
        device,
        vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = frame_count,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        });
  }

  void allocate_sets(vk::raii::Device &device, std::span<const vk::Buffer> uniform_buffers) {
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
          .range = sizeof(UniformBufferObject),
      };
      const vk::WriteDescriptorSet write{
          .dstSet = descriptor_sets_[i],
          .dstBinding = 0,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eUniformBuffer,
          .pBufferInfo = &buffer_info,
      };

      device.updateDescriptorSets(write, nullptr);
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
