#pragma once

#include "renderer/pixel_font.hpp"

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include <cstdint>
#include <string_view>
#include <vector>

namespace engine::ui {

// What the GPU draws: one axis-aligned rectangle with a texture window and a
// colour. Matches UiQuad in shaders/ui.slang, field for field.
struct Quad {
  glm::vec4 rect{0.0F};  // x, y, w, h in pixels from the top left
  glm::vec4 uv{0.0F};    // u0, v0, u1, v1
  glm::vec4 color{1.0F}; // straight alpha
};

// The atlas the shader samples.
//
// A single row of glyphs plus one solid texel at the far left, which is what
// untextured rectangles point at. Keeping the white texel IN the font atlas is
// what lets the whole HUD -- panels, bars, text, the crosshair -- be one draw
// call with one texture bound.
inline constexpr int k_atlas_padding = 1;
inline constexpr int k_atlas_glyphs = k_last_glyph - k_first_glyph + 1;
// The solid cell, then every glyph, each with a column of padding after it so
// linear filtering at non-integer scales cannot bleed one glyph into the next.
inline constexpr int k_atlas_width =
    1 + k_atlas_padding + k_atlas_glyphs * (k_glyph_width + k_atlas_padding);
inline constexpr int k_atlas_height = k_glyph_height;

// Builds the atlas as one byte of coverage per texel. Deliberately R8: the
// glyph is a mask and the colour lives on the quad, so the same glyph can be
// drawn in any colour without a second texture and the whole atlas is under a
// kilobyte.
[[nodiscard]] inline auto build_atlas() -> std::vector<std::uint8_t> {
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(k_atlas_width) * k_atlas_height, 0);
  const auto put = [&pixels](int x, int y, std::uint8_t value) {
    if (x < 0 || x >= k_atlas_width || y < 0 || y >= k_atlas_height)
      return;
    pixels[static_cast<std::size_t>(y) * k_atlas_width + static_cast<std::size_t>(x)] = value;
  };

  // The solid texel, full height so a rectangle sampling it never catches an
  // edge whatever its size.
  for (int y = 0; y < k_atlas_height; ++y)
    put(0, y, 255);

  for (int i = 0; i < k_atlas_glyphs; ++i) {
    const Glyph &glyph = k_glyphs[static_cast<std::size_t>(i)];
    const int x0 = 1 + k_atlas_padding + i * (k_glyph_width + k_atlas_padding);
    for (int y = 0; y < k_glyph_height; ++y)
      for (int x = 0; x < k_glyph_width; ++x)
        if (lit(glyph, x, y))
          put(x0 + x, y, 255);
  }
  return pixels;
}

// Where the solid texel is, as a uv rect. A hair inside it, so no amount of
// filtering can reach the transparent column beside it.
[[nodiscard]] inline auto solid_uv() -> glm::vec4 {
  const float half = 0.5F / static_cast<float>(k_atlas_width);
  return {half, 0.5F / static_cast<float>(k_atlas_height), half,
          1.0F - 0.5F / static_cast<float>(k_atlas_height)};
}

[[nodiscard]] inline auto glyph_uv(char c) -> glm::vec4 {
  const int index = (c < k_first_glyph || c > k_last_glyph) ? 0 : c - k_first_glyph;
  const auto x0 = static_cast<float>(1 + k_atlas_padding +
                                     index * (k_glyph_width + k_atlas_padding));
  const auto width = static_cast<float>(k_atlas_width);
  return {x0 / width, 0.0F, (x0 + static_cast<float>(k_glyph_width)) / width, 1.0F};
}

// Everything the HUD wants to draw this frame, in pixels.
//
// Rebuilt from nothing every frame. That is not a compromise -- it is what makes
// a HUD easy to reason about, because there is no retained state to get out of
// step with the game, and at this scale the whole list is a few hundred quads
// that cost one memcpy to upload.
class DrawList {
public:
  // Integer scale, and it has to be an integer.
  //
  // A pixel font drawn at 1.5x has some rows two pixels tall and some one, which
  // reads as a font with a stutter in it. Doubling and tripling is how a pixel
  // font is meant to be enlarged, and it is why this is called out here rather
  // than left to whatever the caller passes: the type says integer so the bug
  // cannot be written.
  void begin(int screen_width, int screen_height, int scale) {
    quads_.clear();
    width_ = screen_width;
    height_ = screen_height;
    scale_ = scale < 1 ? 1 : scale;
  }

  [[nodiscard]] auto scale() const -> int { return scale_; }
  [[nodiscard]] auto screen() const -> glm::ivec2 { return {width_, height_}; }
  [[nodiscard]] auto quads() const -> const std::vector<Quad> & { return quads_; }
  [[nodiscard]] auto empty() const -> bool { return quads_.empty(); }

  // A filled rectangle, in unscaled pixels. Scaling is applied here so a caller
  // lays the HUD out in font-sized units and never multiplies anything itself.
  void rect(int x, int y, int w, int h, const glm::vec4 &colour) {
    const auto s = static_cast<float>(scale_);
    quads_.push_back({.rect = {static_cast<float>(x) * s, static_cast<float>(y) * s,
                               static_cast<float>(w) * s, static_cast<float>(h) * s},
                      .uv = solid_uv(),
                      .color = colour});
  }

  // A one-pixel-thick outline, drawn as four rectangles rather than as a
  // stretched texture: at integer scale the corners then land exactly.
  void frame(int x, int y, int w, int h, const glm::vec4 &colour) {
    rect(x, y, w, 1, colour);
    rect(x, y + h - 1, w, 1, colour);
    rect(x, y + 1, 1, h - 2, colour);
    rect(x + w - 1, y + 1, 1, h - 2, colour);
  }

  // One line of text. Returns where the next character would go, so callers can
  // run several colours across one line without measuring anything.
  auto text(int x, int y, std::string_view s, const glm::vec4 &colour) -> int {
    const auto sc = static_cast<float>(scale_);
    int pen = x;
    for (const char c : s) {
      if (c == ' ') { // nothing to draw, and the commonest character there is
        pen += k_glyph_advance;
        continue;
      }
      quads_.push_back(
          {.rect = {static_cast<float>(pen) * sc, static_cast<float>(y) * sc,
                    static_cast<float>(k_glyph_width) * sc,
                    static_cast<float>(k_glyph_height) * sc},
           .uv = glyph_uv(c),
           .color = colour});
      pen += k_glyph_advance;
    }
    return pen;
  }

  // Text with a hard drop shadow one pixel down and right.
  //
  // Not decoration. A HUD is drawn over whatever the world happens to be, and
  // pale text on a pale wall is unreadable; a shadow is how a pixel font stays
  // legible without a panel behind every label. One pixel, at the same integer
  // scale, so it never blurs.
  auto shadowed(int x, int y, std::string_view s, const glm::vec4 &colour) -> int {
    text(x + 1, y + 1, s, {0.0F, 0.0F, 0.0F, colour.a * 0.7F});
    return text(x, y, s, colour);
  }

  [[nodiscard]] static auto measure(std::string_view s) -> int {
    return static_cast<int>(s.size()) * k_glyph_advance;
  }

private:
  std::vector<Quad> quads_;
  int width_{0};
  int height_{0};
  int scale_{1};
};

} // namespace engine::ui
