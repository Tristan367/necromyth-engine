// The mip chain, checked by looking at what it produces rather than by walking
// around the world looking at windows.
//
// "You could easily do this yourself. You just run the code to generate the
// mipmaps and literally just look at those images. You don't need to start the
// world and generate a world and then find a window and then walk away from it."
// Quite right, and this is that, as assertions.
#include "renderer/texture_array.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char *what) {
  std::printf("  %-4s %s\n", condition ? "ok" : "FAIL", what);
  if (!condition)
    ++failures;
}

// A four-pane window: a border and a cross, on transparent glass. The same
// shape as the real tile and the one whose muntins collapsed to a dot.
auto window_tile(int size) -> std::vector<float> {
  std::vector<float> alpha(static_cast<std::size_t>(size) * size, 0.0F);
  const int bar = size / 16;      // muntin half-width
  const int mid = size / 2;
  for (int y = 0; y < size; ++y)
    for (int x = 0; x < size; ++x) {
      const bool border = x < bar || y < bar || x >= size - bar || y >= size - bar;
      const bool cross = std::abs(x - mid) < bar || std::abs(y - mid) < bar;
      alpha[static_cast<std::size_t>(y) * size + x] = (border || cross) ? 1.0F : 0.0F;
    }
  return alpha;
}

auto coverage_of(const std::vector<float> &a) -> float {
  int passed = 0;
  for (const float v : a)
    if (v >= engine::detail::k_alpha_cutoff)
      ++passed;
  return static_cast<float>(passed) / static_cast<float>(a.size());
}

auto halve(const std::vector<float> &src, int w, int h) -> std::vector<float> {
  const int nw = std::max(w / 2, 1);
  const int nh = std::max(h / 2, 1);
  std::vector<float> out(static_cast<std::size_t>(nw) * nh);
  for (int y = 0; y < nh; ++y)
    for (int x = 0; x < nw; ++x) {
      float sum = 0.0F;
      int taken = 0;
      for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
          const int sx = x * 2 + dx;
          const int sy = y * 2 + dy;
          if (sx >= w || sy >= h)
            continue;
          sum += src[static_cast<std::size_t>(sy) * w + sx];
          ++taken;
        }
      out[static_cast<std::size_t>(y) * nw + x] = sum / static_cast<float>(std::max(taken, 1));
    }
  return out;
}

// How many separate pieces the opaque part of the tile is in.
//
// This is the artifact stated precisely. "Windows have this weird floating dot
// on them from a distance" is not a texel being dim -- it is the muntin cross
// DETACHING from the frame and surviving on its own. A window frame is one
// connected shape; the moment a level reports two, one of them is the dot.
//
// Eight-connected and wrapping, because these are tiles.
auto components(const std::vector<float> &a, int w, int h) -> int {
  std::vector<char> seen(a.size(), 0);
  int found = 0;
  std::vector<int> stack;
  for (int start = 0; start < static_cast<int>(a.size()); ++start) {
    if (a[static_cast<std::size_t>(start)] < engine::detail::k_alpha_cutoff ||
        seen[static_cast<std::size_t>(start)] != 0)
      continue;
    ++found;
    stack.assign(1, start);
    seen[static_cast<std::size_t>(start)] = 1;
    while (!stack.empty()) {
      const int at = stack.back();
      stack.pop_back();
      const int y = at / w;
      const int x = at % w;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int sx = (x + dx + w) % w;
          const int sy = (y + dy + h) % h;
          const auto k = static_cast<std::size_t>(sy) * w + sx;
          if (a[k] >= engine::detail::k_alpha_cutoff && seen[k] == 0) {
            seen[k] = 1;
            stack.push_back(static_cast<int>(k));
          }
        }
    }
  }
  return found;
}

void test_the_window_never_becomes_a_dot() {
  std::printf("mipping a four-pane window\n");
  constexpr int k_size = 256;
  std::vector<float> alpha = window_tile(k_size);
  const float base = coverage_of(alpha);
  std::printf("    base coverage %.4f\n", base);

  int w = k_size, h = k_size;
  int worst_pieces = 1;
  float worst_drift = 0.0F;
  std::vector<float> raw = alpha;
  while (w > 1 || h > 1) {
    raw = halve(raw, w, h);
    w = std::max(w / 2, 1);
    h = std::max(h / 2, 1);

    std::vector<float> shipped = raw;
    engine::detail::match_coverage(shipped, w, h, base);

    const float coverage = coverage_of(shipped);
    const int pieces = components(shipped, w, h);
    // One texel of slack: a 4x4 level cannot represent 43.16% exactly.
    const float drift = std::abs(coverage - base) -
                        1.0F / static_cast<float>(shipped.size());
    worst_drift = std::max(worst_drift, drift);
    // Only down to 16x16. Below that a four-pane window is not a shape that
    // fits -- a 8x8 tile has no room for a border and a cross both -- and
    // asserting otherwise would be asserting something untrue.
    if (w >= 16)
      worst_pieces = std::max(worst_pieces, pieces);
    std::printf("    %3dx%-3d coverage %.4f  %d piece(s)\n", w, h, coverage, pieces);
  }

  check(worst_drift <= 0.0F, "every level covers the same fraction as the original");
  check(worst_pieces == 1,
        "and the frame stays in ONE piece down to 16x16 -- no detached dot");
}

// The other half of the same claim: a fully opaque texture must come through
// untouched. Every non-cutout layer in the atlas goes through this code.
void test_an_opaque_texture_is_left_alone() {
  std::printf("mipping an opaque texture\n");
  std::vector<float> alpha(64 * 64, 1.0F);
  engine::detail::match_coverage(alpha, 64, 64, 1.0F);
  bool untouched = true;
  for (const float v : alpha)
    untouched = untouched && v == 1.0F;
  check(untouched, "a fully opaque level is returned exactly as it came in");
}

} // namespace

int main() {
  test_the_window_never_becomes_a_dot();
  test_an_opaque_texture_is_left_alone();
  std::printf(failures == 0 ? "\nall mipmap tests passed\n" : "\n%d FAILED\n", failures);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
