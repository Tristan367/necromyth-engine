cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(VulkanCppEngine LANGUAGES CXX)

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
set(VCE_SHADER_SPIRV "")
set(VCE_SKY_SHADER_SPIRV "")
set(VCE_SHADOW_DEPTH_SPIRV "")

foreach(SHADER_SOURCE ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME "${SHADER_SOURCE}" NAME_WE)
    set(SHADER_SPIRV "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")

    if(SHADER_NAME STREQUAL "shadow_depth")
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
                -o "${SHADER_SPIRV}"
            DEPENDS "${SHADER_SOURCE}"
            COMMENT "Compiling ${SHADER_NAME}.slang"
            VERBATIM
        )
    else()
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
    endif()

    list(APPEND SHADER_SPIRV_OUTPUTS "${SHADER_SPIRV}")
    if(SHADER_NAME STREQUAL "triangle")
        set(VCE_SHADER_SPIRV "${SHADER_SPIRV}")
    elseif(SHADER_NAME STREQUAL "sky")
        set(VCE_SKY_SHADER_SPIRV "${SHADER_SPIRV}")
    elseif(SHADER_NAME STREQUAL "shadow_depth")
        set(VCE_SHADOW_DEPTH_SPIRV "${SHADER_SPIRV}")
    endif()
endforeach()

if(NOT VCE_SHADER_SPIRV)
    message(FATAL_ERROR "Expected shaders/triangle.slang")
endif()

if(NOT VCE_SKY_SHADER_SPIRV)
    message(FATAL_ERROR "Expected shaders/sky.slang")
endif()

if(NOT VCE_SHADOW_DEPTH_SPIRV)
    message(FATAL_ERROR "Expected shaders/shadow_depth.slang")
endif()

add_custom_target(vce_shaders DEPENDS ${SHADER_SPIRV_OUTPUTS})

add_library(vce_engine INTERFACE)
add_library(VCE::Engine ALIAS vce_engine)

target_compile_features(vce_engine INTERFACE cxx_std_23)

target_compile_definitions(vce_engine INTERFACE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    ENGINE_SHADER_SPIRV="${VCE_SHADER_SPIRV}"
    ENGINE_SKY_SHADER_SPIRV="${VCE_SKY_SHADER_SPIRV}"
    ENGINE_SHADOW_DEPTH_SPIRV="${VCE_SHADOW_DEPTH_SPIRV}"
)

target_include_directories(vce_engine SYSTEM INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>"
    "$<BUILD_INTERFACE:${STB_INCLUDE_DIR}>"
)

target_link_libraries(vce_engine INTERFACE
    SDL3::SDL3
    Vulkan::Vulkan
    glm::glm
)

add_dependencies(vce_engine vce_shaders)

if(MSVC)
    target_compile_options(vce_engine INTERFACE /W4 /permissive-)
else()
    target_compile_options(vce_engine INTERFACE -Wall -Wextra -Wpedantic)
endif()

set(_run_env_vars "")
if(VULKAN_SDK_ROOT AND IS_DIRECTORY "${VULKAN_SDK_ROOT}/lib")
    list(APPEND _run_env_vars "VULKAN_SDK=${VULKAN_SDK_ROOT}")
    list(APPEND _run_env_vars "LD_LIBRARY_PATH=${VULKAN_SDK_ROOT}/lib")
    list(APPEND _run_env_vars "VK_ADD_LAYER_PATH=${VULKAN_SDK_ROOT}/share/vulkan/explicit_layer.d")
endif()

option(VCE_BUILD_IN_TREE_APP "Build the legacy in-tree demo executable" OFF)
if(VCE_BUILD_IN_TREE_APP)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_executable(VulkanCppEngine src/main.cpp)
    target_link_libraries(VulkanCppEngine PRIVATE VCE::Engine)
    add_dependencies(VulkanCppEngine vce_shaders)
    add_custom_target(run
        COMMAND "${CMAKE_COMMAND}" -E env ${_run_env_vars} "$<TARGET_FILE:VulkanCppEngine>"
        DEPENDS VulkanCppEngine
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        USES_TERMINAL
    )
endif()
