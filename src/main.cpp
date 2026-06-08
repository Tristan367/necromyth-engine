#define VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include "app.hpp"
#include "platform/gpu_cli.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>

auto main(int argc, char **argv) -> int {
  try {
    const engine::CliParseResult cli = engine::parse_engine_cli(argc, argv);
    if (cli.exit_after_list_gpus) {
      engine::list_physical_devices();
      return EXIT_SUCCESS;
    }

    engine::App app(cli.config);
    app.run();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
