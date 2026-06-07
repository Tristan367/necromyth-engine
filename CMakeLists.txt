cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

set(NAME VulkanCppEngine)
project(${NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT VULKAN_SDK_ROOT AND DEFINED ENV{VULKAN_SDK})
    set(VULKAN_SDK_ROOT "$ENV{VULKAN_SDK}" CACHE PATH "Vulkan SDK root directory")
endif()

if(VULKAN_SDK_ROOT AND IS_DIRECTORY "${VULKAN_SDK_ROOT}")
    list(PREPEND CMAKE_PREFIX_PATH "${VULKAN_SDK_ROOT}")
    set(ENV{VULKAN_SDK} "${VULKAN_SDK_ROOT}")
    message(STATUS "Using Vulkan SDK: ${VULKAN_SDK_ROOT}")
endif()

find_package(Vulkan REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

find_program(SLANGC_EXECUTABLE slangc
    HINTS
        "${VULKAN_SDK_ROOT}/bin"
        "$ENV{VULKAN_SDK}/bin"
    REQUIRED
)

set(SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.slang")
set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
set(SHADER_SPIRV "${SHADER_OUTPUT_DIR}/slang.spv")

add_custom_command(
    OUTPUT "${SHADER_SPIRV}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}"
    COMMAND "${SLANGC_EXECUTABLE}"
        "${SHADER_SOURCE}"
        -target spirv
        -profile spirv_1_4
        -emit-spirv-directly
        -fvk-use-entrypoint-name
        -entry vertMain
        -entry fragMain
        -o "${SHADER_SPIRV}"
    DEPENDS "${SHADER_SOURCE}"
    COMMENT "Compiling Slang shaders"
    VERBATIM
)

add_custom_target(shaders DEPENDS "${SHADER_SPIRV}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(${NAME}
    src/main.cpp
    src/app.hpp
    src/platform/sdl_window.hpp
    src/renderer/vulkan_context.hpp
    src/renderer/graphics_pipeline.hpp
    src/renderer/image_barrier.hpp
    src/renderer/depth_image.hpp
    src/renderer/buffer.hpp
    src/renderer/vertex.hpp
    src/renderer/uniform_buffer.hpp
    src/renderer/descriptors.hpp
)

add_dependencies(${NAME} shaders)

target_compile_features(${NAME} PRIVATE cxx_std_23)

target_compile_definitions(${NAME} PRIVATE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    ENGINE_SHADER_SPIRV="${SHADER_SPIRV}"
)

target_include_directories(${NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(${NAME} PRIVATE
    SDL3::SDL3
    Vulkan::Vulkan
    glm::glm
)

if(MSVC)
    target_compile_options(${NAME} PRIVATE /W4 /permissive-)
else()
    target_compile_options(${NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(${NAME} PROPERTIES
    DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

set(_run_env_vars "")
if(VULKAN_SDK_ROOT AND IS_DIRECTORY "${VULKAN_SDK_ROOT}/lib")
    list(APPEND _run_env_vars "VULKAN_SDK=${VULKAN_SDK_ROOT}")
    list(APPEND _run_env_vars "LD_LIBRARY_PATH=${VULKAN_SDK_ROOT}/lib")
    list(APPEND _run_env_vars "VK_ADD_LAYER_PATH=${VULKAN_SDK_ROOT}/share/vulkan/explicit_layer.d")
endif()

add_custom_target(run
    COMMAND "${CMAKE_COMMAND}" -E env ${_run_env_vars} "$<TARGET_FILE:${NAME}>"
    DEPENDS ${NAME}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
)
