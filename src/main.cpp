#define VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include "app.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>

auto main() -> int {
    try {
        engine::App app;
        app.run();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
