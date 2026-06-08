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

find_path(STB_INCLUDE_DIR NAMES stb_image.h PATH_SUFFIXES stb)
if(NOT STB_INCLUDE_DIR)
  message(FATAL_ERROR "stb_image.h not found; install the stb package")
endif()

find_program(SLANGC_EXECUTABLE slangc
    HINTS
        "${VULKAN_SDK_ROOT}/bin"
        "$ENV{VULKAN_SDK}/bin"
    REQUIRED
)

set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.slang")

set(SHADER_SPIRV_OUTPUTS "")
set(DEFAULT_SHADER_SPIRV "")

foreach(SHADER_SOURCE ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME "${SHADER_SOURCE}" NAME_WE)
    set(SHADER_SPIRV "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")

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
        COMMENT "Compiling ${SHADER_NAME}.slang"
        VERBATIM
    )

    list(APPEND SHADER_SPIRV_OUTPUTS "${SHADER_SPIRV}")
    if(SHADER_NAME STREQUAL "triangle")
        set(DEFAULT_SHADER_SPIRV "${SHADER_SPIRV}")
    endif()
endforeach()

if(NOT DEFAULT_SHADER_SPIRV)
    message(FATAL_ERROR "Expected shaders/triangle.slang for the default textured mesh pipeline")
endif()

add_custom_target(shaders DEPENDS ${SHADER_SPIRV_OUTPUTS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(${NAME}
    src/main.cpp
    src/app.hpp
    src/demo_scene.hpp
    src/engine_config.hpp
    src/platform/gpu_cli.hpp
    src/platform/sdl_window.hpp
    src/scene/camera.hpp
    src/scene/mesh_instance.hpp
    src/scene/mesh_source.hpp
    src/scene/render_layer.hpp
    src/scene/scene.hpp
    src/renderer/render_settings.hpp
    src/renderer/pipeline_id.hpp
    src/renderer/draw_list.hpp
    src/renderer/mesh_gpu.hpp
    src/renderer/vulkan_device.hpp
    src/renderer/swapchain.hpp
    src/renderer/vulkan_context.hpp
    src/renderer/graphics_pipeline.hpp
    src/renderer/image_barrier.hpp
    src/renderer/depth_image.hpp
    src/renderer/msaa_color_image.hpp
    src/renderer/buffer.hpp
    src/renderer/vertex.hpp
    src/renderer/model_loader.hpp
    src/renderer/uniform_buffer.hpp
    src/renderer/descriptors.hpp
    src/renderer/texture_image.hpp
)

add_dependencies(${NAME} shaders)

target_compile_features(${NAME} PRIVATE cxx_std_23)

target_compile_definitions(${NAME} PRIVATE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    ENGINE_SHADER_SPIRV="${DEFAULT_SHADER_SPIRV}"
    ENGINE_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/models/viking_room.obj"
    ENGINE_TEXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/textures/viking_room.png"
)

target_include_directories(${NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    "${STB_INCLUDE_DIR}"
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
