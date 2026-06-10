# Contributing

Pull requests are welcome. This repo is public; you do not need special access to fork and open a PR.

## Quick start

1. Fork [necromyth-engine](https://github.com/Tristan367/necromyth-engine) on GitHub.
2. Clone your fork and create a branch from `master`.
3. Build and run the [demo client](https://github.com/Tristan367/necromyth-engine-demo) against your engine checkout (`VCE_ROOT` or sibling directory).
4. Keep changes focused — one logical change per PR when possible.
5. Open a pull request against `master` with a short description and how you tested (e.g. `make debug build`, `make debug run` on Linux).

## Scope

**In scope for this repo:** renderer, shaders, scene types, build/docs for `VCE::Engine`.

**Out of scope:** the Necromyth game itself (private), and most demo-only UI/scene work (use [necromyth-engine-demo](https://github.com/Tristan367/necromyth-engine-demo) instead).

## Code style

Match the surrounding files: modern C++23, minimal diffs, no unrelated refactors.

## License

By contributing, you agree that your contributions will be licensed under the same [MIT License](LICENSE) as the project.
