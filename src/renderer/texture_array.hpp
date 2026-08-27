#pragma once

#include "renderer/texture_image.hpp"
#include "scene/texture_array_layer.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine {

namespace detail {

// sRGB <-> linear. A GPU blit on an sRGB image linearises before it filters, so
// a CPU mip chain has to as well or every mip comes out darker than the one the
// hardware would have produced. Alpha is not sRGB-encoded and is averaged as-is.
[[nodiscard]] inline auto srgb_to_linear(float c) -> float {
  return c <= 0.04045F ? c / 12.92F : std::pow((c + 0.055F) / 1.055F, 2.4F);
}

[[nodiscard]] inline auto linear_to_srgb(float c) -> float {
  return c <= 0.0031308F ? c * 12.92F : 1.055F * std::pow(c, 1.0F / 2.4F) - 0.055F;
}

// How much a mip level's alpha is lifted per halving.
//
// Straight from the C#'s BuildMipChainAlphaAware, which box-filters the 2x2 and
// then does `aOut = Mathf.Min(aOut * 1.6f, 1)`.
//
// It exists because averaging alpha is what makes cutout detail dissolve with
// distance. A window's mullion is a few pixels of opaque in a field of nothing;
// halve the texture a few times and its alpha averages below the cutoff, and
// the mullion -- and the frame, and the boarding on a window -- stops being
// drawn at all. Lifting alpha as the level shrinks keeps thin opaque things
// thin instead of letting them evaporate.
//
// Opaque textures are unaffected: their alpha is already 1 and min(1.6, 1) is 1.
// That is why the C# turns this on for the WHOLE voxel atlas rather than
// picking out the cutout layers, and why this does too.
// The alpha test the cutout shader applies. Must match kAlphaCutoff in
// lib/surface.slang: this whole scheme is about how much of the texture is on
// the passing side of that number, so the two have to be the same number.
inline constexpr float k_alpha_cutoff = 0.5F;

// What fraction of the texels would survive the alpha test if their alpha were
// multiplied by `scale`.
[[nodiscard]] inline auto alpha_coverage(const std::uint8_t *px, int w, int h, float scale)
    -> float {
  const std::size_t count = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  if (count == 0)
    return 0.0F;
  std::size_t passed = 0;
  for (std::size_t i = 0; i < count; ++i)
    if (std::min(static_cast<float>(px[i * 4 + 3]) / 255.0F * scale, 1.0F) >= k_alpha_cutoff)
      ++passed;
  return static_cast<float>(passed) / static_cast<float>(count);
}

// Makes this mip cover exactly as much of its area as the full-size image did,
// by choosing WHICH texels are opaque rather than by scaling all of them.
//
// This replaced two earlier attempts, and the reason it had to is worth keeping.
//
// First there was a flat 1.6x lift per halving. Wrong in the way the player
// described: "it's greedy towards the colored pixels... the window cross texture
// gets wider." A fixed multiplier does not know how much alpha there was to
// begin with, so it dilates every level and the dilation compounds.
//
// Then there was Ignacio Castano's technique proper: bisect for the single alpha
// SCALE whose coverage matches the original's. That is what Unity and Unreal do
// for alpha-tested foliage, and on a photographic leaf texture it works. On
// these tiles it does not, and dumping the mip chain to images -- rather than
// walking around the world looking at windows -- showed exactly why.
//
// The tile alpha is BINARY. Box-filtering a hard-edged cutout lands enormous
// numbers of texels on exactly 0.5: at 8x8 on the window tile, sixteen of
// sixty-four sit precisely there. Coverage as a function of scale is therefore a
// step function that steps straight over the target -- that level could produce
// 18.75% coverage or 43.75%, and nothing in between, when it wanted 33.98%.
// Successive levels landed on opposite sides of the target, and the level where
// the muntin cross finally lost the argument kept a single opaque texel in the
// middle of a transparent tile. That is the artifact as reported: "windows have
// this weird floating dot on them from a distance."
//
// So this does not look for a scale. It takes the N highest-alpha texels, N
// being the count that matches the base coverage, and puts the threshold
// between the Nth and the N+1th. Coverage is then exact at every level by
// construction, and cannot swing.
//
// Ties -- and at 0.5 there are hundreds -- are broken by how much alpha
// SURROUNDS a texel. Breaking them in raster order scatters the survivors;
// keeping the ones whose neighbours are also solid keeps the frame connected,
// which is the difference between a window and a handful of speckles. The
// neighbourhood wraps, because these are tiles.
inline void match_coverage(std::vector<float> &alpha, int w, int h, float target) {
  const auto count = static_cast<int>(alpha.size());
  if (count == 0 || target <= 0.0F || target >= 1.0F)
    return;

  const int want = std::clamp(
      static_cast<int>(std::lround(target * static_cast<float>(count))), 0, count);

  std::vector<float> around(alpha.size(), 0.0F);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float sum = 0.0F;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
          sum += alpha[static_cast<std::size_t>((y + dy + h) % h) * w + ((x + dx + w) % w)];
      around[static_cast<std::size_t>(y) * w + x] = sum;
    }

  std::vector<int> order(alpha.size());
  for (int i = 0; i < count; ++i)
    order[static_cast<std::size_t>(i)] = i;
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    const auto i = static_cast<std::size_t>(a);
    const auto j = static_cast<std::size_t>(b);
    if (alpha[i] != alpha[j])
      return alpha[i] > alpha[j];
    return around[i] > around[j];
  });

  // Nudged just clear of the cutoff rather than set to 0 and 1. Which side of
  // the line a texel is on is all a cutout pass reads, and leaving the value
  // otherwise alone keeps the edge shading that alpha-to-coverage would use.
  constexpr float k_margin = 0.02F;
  std::vector<float> out(alpha.size());
  for (int i = 0; i < count; ++i) {
    const auto t = static_cast<std::size_t>(order[static_cast<std::size_t>(i)]);
    out[t] = i < want ? std::max(alpha[t], k_alpha_cutoff + k_margin)
                      : std::min(alpha[t], k_alpha_cutoff - k_margin);
  }
  alpha.swap(out);
}

// One box-filtered halving of an RGBA8 image.
inline void halve_alpha_aware(const std::uint8_t *src, int src_w, int src_h,
                              std::uint8_t *dst, int dst_w, int dst_h) {
  for (int y = 0; y < dst_h; ++y) {
    for (int x = 0; x < dst_w; ++x) {
      float r = 0.0F, g = 0.0F, b = 0.0F, a = 0.0F;
      int samples = 0;
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          const int sx = x * 2 + dx;
          const int sy = y * 2 + dy;
          if (sx >= src_w || sy >= src_h)
            continue;
          const std::uint8_t *p = src + (static_cast<std::size_t>(sy) * src_w + sx) * 4;
          r += srgb_to_linear(p[0] / 255.0F);
          g += srgb_to_linear(p[1] / 255.0F);
          b += srgb_to_linear(p[2] / 255.0F);
          a += p[3] / 255.0F;
          ++samples;
        }
      }
      const float inv = 1.0F / static_cast<float>(std::max(samples, 1));
      // A plain average. The alpha is corrected afterwards, per level, by
      // scale_for_coverage -- doing it here would mean the correction compounded
      // down the chain instead of being measured against the original.
      const float out_a = std::min(a * inv, 1.0F);
      std::uint8_t *o = dst + (static_cast<std::size_t>(y) * dst_w + x) * 4;
      o[0] = static_cast<std::uint8_t>(std::lround(linear_to_srgb(r * inv) * 255.0F));
      o[1] = static_cast<std::uint8_t>(std::lround(linear_to_srgb(g * inv) * 255.0F));
      o[2] = static_cast<std::uint8_t>(std::lround(linear_to_srgb(b * inv) * 255.0F));
      o[3] = static_cast<std::uint8_t>(std::lround(out_a * 255.0F));
    }
  }
}

} // namespace detail

class TextureArray {
public:
  void load_from_files(
      const vk::raii::PhysicalDevice &physical_device,
      vk::raii::Device &device,
      vk::raii::CommandPool &command_pool,
      vk::raii::Queue &queue,
      std::span<const TextureArrayLayer> paths) {
    if (paths.empty())
      throw std::runtime_error("Texture array requires at least one layer path");

    physical_device_ = &physical_device;
    device_ = &device;
    format_ = vk::Format::eR8G8B8A8Srgb;

    std::vector<detail::RgbaImageData> layers;
    layers.reserve(paths.size());

    for (const TextureArrayLayer &source : paths) {
      detail::RgbaImageData layer = detail::load_rgba_image(source.path);
      if (!source.overlay.empty())
        detail::composite_over(layer, detail::load_rgba_image(source.overlay), source.path);
      if (layers.empty()) {
        extent_ = vk::Extent3D{
            .width = static_cast<std::uint32_t>(layer.width),
            .height = static_cast<std::uint32_t>(layer.height),
            .depth = 1,
        };
      } else if (static_cast<std::uint32_t>(layer.width) != extent_.width ||
                 static_cast<std::uint32_t>(layer.height) != extent_.height)
        throw std::runtime_error("Texture array layers must share dimensions: " + source.path);

      layers.push_back(std::move(layer));
    }

    layer_count_ = static_cast<std::uint32_t>(layers.size());
    mip_levels_ = detail::mip_level_count(static_cast<std::int32_t>(extent_.width), static_cast<std::int32_t>(extent_.height));

    // Every mip level of every layer is built on the CPU and uploaded, rather
    // than blitted down on the GPU, so the alpha lift in halve_alpha_aware can
    // be applied at each halving. See k_alpha_mip_gain for why.
    std::vector<vk::Extent2D> level_size;
    std::vector<vk::DeviceSize> level_offset; // within one layer
    vk::DeviceSize chain_size = 0;
    for (std::uint32_t level = 0; level < mip_levels_; ++level) {
      const vk::Extent2D size{std::max(extent_.width >> level, 1U),
                              std::max(extent_.height >> level, 1U)};
      level_size.push_back(size);
      level_offset.push_back(chain_size);
      chain_size += static_cast<vk::DeviceSize>(size.width) * size.height * 4;
    }

    const vk::DeviceSize layer_size = chain_size;
    const vk::DeviceSize staging_size = layer_size * layer_count_;

    const vk::BufferCreateInfo staging_info{
        .size = staging_size,
        .usage = vk::BufferUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    vk::raii::Buffer staging_buffer{device, staging_info};
    const vk::MemoryRequirements staging_requirements = staging_buffer.getMemoryRequirements();
    const auto memory_properties = physical_device.getMemoryProperties();

    vk::raii::DeviceMemory staging_memory{
        device,
        vk::MemoryAllocateInfo{
            .allocationSize = staging_requirements.size,
            .memoryTypeIndex = detail::find_memory_type(
                memory_properties,
                staging_requirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
        }};

    staging_buffer.bindMemory(*staging_memory, 0);

    void *mapped = staging_memory.mapMemory(0, staging_size);
    for (std::uint32_t layer = 0; layer < layer_count_; ++layer) {
      auto *chain = static_cast<std::uint8_t *>(mapped)
                  + static_cast<std::size_t>(layer) * static_cast<std::size_t>(layer_size);
      std::memcpy(chain, layers[layer].pixels.data(),
                  static_cast<std::size_t>(level_size[0].width) * level_size[0].height * 4);
      // Coverage is measured on the full-size image ONCE and every level is
      // matched to it. Matching each level to the one above instead lets the
      // error compound.
      const float base_coverage = detail::alpha_coverage(
          chain, static_cast<int>(level_size[0].width), static_cast<int>(level_size[0].height),
          1.0F);

      // The UNCORRECTED alpha of the previous level, which is what the next
      // halving reads.
      //
      // This is the part the old comment claimed and the old code did not do.
      // It halved level N from level N-1 *after* writing the corrected alpha
      // back into it, so every correction fed the next downsample and the error
      // compounded down the chain -- the exact failure the comment said was
      // being avoided.
      std::vector<float> previous_alpha(
          static_cast<std::size_t>(level_size[0].width) * level_size[0].height);
      for (std::size_t i = 0; i < previous_alpha.size(); ++i)
        previous_alpha[i] = static_cast<float>(chain[i * 4 + 3]) / 255.0F;

      for (std::uint32_t level = 1; level < mip_levels_; ++level) {
        const int width = static_cast<int>(level_size[level].width);
        const int height = static_cast<int>(level_size[level].height);
        const int src_w = static_cast<int>(level_size[level - 1].width);
        const int src_h = static_cast<int>(level_size[level - 1].height);

        // Colour from the previous level's colour; alpha from its raw alpha.
        detail::halve_alpha_aware(chain + level_offset[level - 1], src_w, src_h,
                                  chain + level_offset[level], width, height);

        std::vector<float> alpha(static_cast<std::size_t>(width) * height);
        for (int y = 0; y < height; ++y)
          for (int x = 0; x < width; ++x) {
            float sum = 0.0F;
            int taken = 0;
            for (int dy = 0; dy < 2; ++dy)
              for (int dx = 0; dx < 2; ++dx) {
                const int sx = x * 2 + dx;
                const int sy = y * 2 + dy;
                if (sx >= src_w || sy >= src_h)
                  continue;
                sum += previous_alpha[static_cast<std::size_t>(sy) * src_w + sx];
                ++taken;
              }
            alpha[static_cast<std::size_t>(y) * width + x] =
                sum / static_cast<float>(std::max(taken, 1));
          }
        previous_alpha = alpha; // raw, for the next level down

        detail::match_coverage(alpha, width, height, base_coverage);
        std::uint8_t *dst = chain + level_offset[level];
        for (std::size_t i = 0; i < alpha.size(); ++i)
          dst[i * 4 + 3] = static_cast<std::uint8_t>(
              std::lround(std::clamp(alpha[i], 0.0F, 1.0F) * 255.0F));
      }
    }
    staging_memory.unmapMemory();

    const vk::ImageCreateInfo image_info{
        .imageType = vk::ImageType::e2D,
        .format = format_,
        .extent = extent_,
        .mipLevels = mip_levels_,
        .arrayLayers = layer_count_,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
                 vk::ImageUsageFlagBits::eSampled,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    image_ = vk::raii::Image(device, image_info);
    const vk::MemoryRequirements image_requirements = image_.getMemoryRequirements();
    memory_ = vk::raii::DeviceMemory{
        device,
        vk::MemoryAllocateInfo{
            .allocationSize = image_requirements.size,
            .memoryTypeIndex = detail::find_memory_type(
                memory_properties,
                image_requirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal),
        }};
    image_.bindMemory(*memory_, 0);

    detail::execute_one_time_commands(device, command_pool, queue, [&](vk::raii::CommandBuffer &command_buffer) {
      transition_image_layout(
          command_buffer,
          *image_,
          vk::ImageLayout::eUndefined,
          vk::ImageLayout::eTransferDstOptimal,
          {},
          vk::AccessFlagBits2::eTransferWrite,
          vk::PipelineStageFlagBits2::eTopOfPipe,
          vk::PipelineStageFlagBits2::eTransfer,
          vk::ImageAspectFlagBits::eColor,
          0,
          mip_levels_,
          0,
          layer_count_);

      for (std::uint32_t layer = 0; layer < layer_count_; ++layer) {
        for (std::uint32_t level = 0; level < mip_levels_; ++level) {
          const vk::BufferImageCopy region{
              .bufferOffset = layer_size * layer + level_offset[level],
              .imageSubresource = {
                  .aspectMask = vk::ImageAspectFlagBits::eColor,
                  .mipLevel = level,
                  .baseArrayLayer = layer,
                  .layerCount = 1,
              },
              .imageExtent = {level_size[level].width, level_size[level].height, 1},
          };
          command_buffer.copyBufferToImage(*staging_buffer, *image_,
                                           vk::ImageLayout::eTransferDstOptimal, region);
        }
      }

      transition_image_layout(
          command_buffer,
          *image_,
          vk::ImageLayout::eTransferDstOptimal,
          vk::ImageLayout::eShaderReadOnlyOptimal,
          vk::AccessFlagBits2::eTransferWrite,
          vk::AccessFlagBits2::eShaderRead,
          vk::PipelineStageFlagBits2::eTransfer,
          vk::PipelineStageFlagBits2::eFragmentShader,
          vk::ImageAspectFlagBits::eColor,
          0,
          mip_levels_,
          0,
          layer_count_);
    });

    view_ = vk::raii::ImageView(
        device,
        vk::ImageViewCreateInfo{
            .image = *image_,
            .viewType = vk::ImageViewType::e2DArray,
            .format = format_,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = mip_levels_,
                .baseArrayLayer = 0,
                .layerCount = layer_count_,
            },
        });

    create_sampler(physical_device);
  }

  [[nodiscard]] auto sampler() const -> vk::Sampler {
    return *sampler_;
  }

  [[nodiscard]] auto view() const -> vk::ImageView {
    return *view_;
  }

  [[nodiscard]] auto layer_count() const -> std::uint32_t {
    return layer_count_;
  }

private:

  void create_sampler(const vk::raii::PhysicalDevice &physical_device) {
    sampler_ = detail::create_mipmapped_sampler(*device_, physical_device, mip_levels_);
  }

  const vk::raii::PhysicalDevice *physical_device_{nullptr};
  vk::raii::Device *device_{nullptr};
  vk::Format format_{vk::Format::eUndefined};
  vk::Extent3D extent_{};
  std::uint32_t layer_count_{};
  std::uint32_t mip_levels_{1};
  vk::raii::Image image_{nullptr};
  vk::raii::DeviceMemory memory_{nullptr};
  vk::raii::ImageView view_{nullptr};
  vk::raii::Sampler sampler_{nullptr};
};

} // namespace engine
