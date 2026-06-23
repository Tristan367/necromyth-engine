cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(NecromythEngine LANGUAGES CXX)

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
set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS "${SHADER_SOURCE_DIR}/*.slang")
file(GLOB SHADER_LIB_SOURCES CONFIGURE_DEPENDS "${SHADER_SOURCE_DIR}/lib/*.slang")

set(SHADER_SPIRV_OUTPUTS "")
set(VCE_SHADER_SPIRV "")
set(VCE_SKY_SHADER_SPIRV "")
set(VCE_SHADOW_DEPTH_SPIRV "")
set(VCE_SKINNED_SHADER_SPIRV "")
set(VCE_SKINNED_SHADOW_DEPTH_SPIRV "")

foreach(SHADER_SOURCE ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME "${SHADER_SOURCE}" NAME_WE)
    set(SHADER_SPIRV "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")

    if(SHADER_NAME STREQUAL "shadow_depth")
        add_custom_command(
            OUTPUT "${SHADER_SPIRV}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}"
            COMMAND "${SLANGC_EXECUTABLE}"
                "${SHADER_SOURCE}"
                -I "${SHADER_SOURCE_DIR}"
                -target spirv
                -profile spirv_1_4
                -emit-spirv-directly
                -fvk-use-entrypoint-name
                -entry vertMain
                -o "${SHADER_SPIRV}"
            DEPENDS "${SHADER_SOURCE}" ${SHADER_LIB_SOURCES}
            COMMENT "Compiling ${SHADER_NAME}.slang"
            VERBATIM
        )
    elseif(SHADER_NAME STREQUAL "triangle")
        add_custom_command(
            OUTPUT "${SHADER_SPIRV}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}"
            COMMAND "${SLANGC_EXECUTABLE}"
                "${SHADER_SOURCE}"
                -I "${SHADER_SOURCE_DIR}"
                -target spirv
                -profile spirv_1_4
                -emit-spirv-directly
                -fvk-use-entrypoint-name
                -entry vertMain
                -entry fragOpaqueHard
                -entry fragCutoutHard
                -entry fragA2CHard
                -entry fragOpaquePcf
                -entry fragCutoutPcf
                -entry fragA2CPcf
                -entry fragOpaqueHardCsm2
                -entry fragCutoutHardCsm2
                -entry fragA2CHardCsm2
                -entry fragOpaquePcfCsm2
                -entry fragCutoutPcfCsm2
                -entry fragA2CPcfCsm2
                -o "${SHADER_SPIRV}"
            DEPENDS "${SHADER_SOURCE}" ${SHADER_LIB_SOURCES}
            COMMENT "Compiling ${SHADER_NAME}.slang"
            VERBATIM
        )
    elseif(SHADER_NAME STREQUAL "triangle_skinned")
        add_custom_command(
            OUTPUT "${SHADER_SPIRV}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}"
            COMMAND "${SLANGC_EXECUTABLE}"
                "${SHADER_SOURCE}"
                -I "${SHADER_SOURCE_DIR}"
                -target spirv
                -profile spirv_1_4
                -emit-spirv-directly
                -fvk-use-entrypoint-name
                -entry vertMainSkinned
                -o "${SHADER_SPIRV}"
            DEPENDS "${SHADER_SOURCE}" ${SHADER_LIB_SOURCES}
            COMMENT "Compiling ${SHADER_NAME}.slang"
            VERBATIM
        )
    elseif(SHADER_NAME STREQUAL "shadow_depth_skinned")
        add_custom_command(
            OUTPUT "${SHADER_SPIRV}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}"
            COMMAND "${SLANGC_EXECUTABLE}"
                "${SHADER_SOURCE}"
                -I "${SHADER_SOURCE_DIR}"
                -target spirv
                -profile spirv_1_4
                -emit-spirv-directly
                -fvk-use-entrypoint-name
                -entry vertMainSkinned
                -o "${SHADER_SPIRV}"
            DEPENDS "${SHADER_SOURCE}" ${SHADER_LIB_SOURCES}
            COMMENT "Compiling ${SHADER_NAME}.slang"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT "${SHADER_SPIRV}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}"
            COMMAND "${SLANGC_EXECUTABLE}"
                "${SHADER_SOURCE}"
                -I "${SHADER_SOURCE_DIR}"
                -target spirv
                -profile spirv_1_4
                -emit-spirv-directly
                -fvk-use-entrypoint-name
                -entry vertMain
                -entry fragMain
                -o "${SHADER_SPIRV}"
            DEPENDS "${SHADER_SOURCE}" ${SHADER_LIB_SOURCES}
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
    elseif(SHADER_NAME STREQUAL "triangle_skinned")
        set(VCE_SKINNED_SHADER_SPIRV "${SHADER_SPIRV}")
    elseif(SHADER_NAME STREQUAL "shadow_depth_skinned")
        set(VCE_SKINNED_SHADOW_DEPTH_SPIRV "${SHADER_SPIRV}")
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

add_library(vce_gltf_impl STATIC src/renderer/gltf_loader_impl.cpp)
target_include_directories(vce_gltf_impl SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    "${STB_INCLUDE_DIR}"
)
target_compile_features(vce_gltf_impl PRIVATE cxx_std_23)

add_library(vce_stb STATIC src/renderer/texture_image_stb.cpp)
target_include_directories(vce_stb SYSTEM PRIVATE "${STB_INCLUDE_DIR}")
target_compile_features(vce_stb PRIVATE cxx_std_23)

add_library(vce_core INTERFACE)
add_library(VCE::Core ALIAS vce_core)

target_compile_features(vce_core INTERFACE cxx_std_23)

target_include_directories(vce_core SYSTEM INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>"
    "$<BUILD_INTERFACE:${STB_INCLUDE_DIR}>"
)

target_link_libraries(vce_core INTERFACE
    glm::glm
    vce_gltf_impl
    vce_stb
)

if(MSVC)
    target_compile_options(vce_core INTERFACE /W4 /permissive-)
else()
    target_compile_options(vce_core INTERFACE -Wall -Wextra -Wpedantic)
endif()

add_library(vce_renderer INTERFACE)
add_library(VCE::Renderer ALIAS vce_renderer)

target_compile_definitions(vce_renderer INTERFACE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    ENGINE_SHADER_SPIRV="${VCE_SHADER_SPIRV}"
    ENGINE_SKY_SHADER_SPIRV="${VCE_SKY_SHADER_SPIRV}"
    ENGINE_SHADOW_DEPTH_SPIRV="${VCE_SHADOW_DEPTH_SPIRV}"
    ENGINE_SKINNED_SHADER_SPIRV="${VCE_SKINNED_SHADER_SPIRV}"
    ENGINE_SKINNED_SHADOW_DEPTH_SPIRV="${VCE_SKINNED_SHADOW_DEPTH_SPIRV}"
)

target_link_libraries(vce_renderer INTERFACE
    vce_core
    SDL3::SDL3
    Vulkan::Vulkan
)

add_dependencies(vce_renderer vce_shaders)

# Backward compatibility alias
add_library(vce_engine INTERFACE)
add_library(VCE::Engine ALIAS vce_engine)
target_link_libraries(vce_engine INTERFACE vce_renderer)
