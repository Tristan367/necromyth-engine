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

# 39001 "explicit binding overlap" fires on every combined image sampler: a
# Texture and its Sampler deliberately share one [[vk::binding]], which is how
# VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER is expressed with separate Slang
# types. It is the intended idiom here and matches the descriptor layout, so it
# is silenced by id rather than blanket-disabling warnings -- anything else the
# shader compiler has to say should still be visible.
set(VCE_SLANGC_WARNING_FLAGS -warnings-disable 39001)

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
set(VCE_DEBUG_LINE_SPIRV "")
set(VCE_POINT_SHADOW_SPIRV "")
set(VCE_NO_POINT_SHADER_SPIRV "")
set(VCE_PARTICLE_BILLBOARD_SPIRV "")

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
                ${VCE_SLANGC_WARNING_FLAGS}
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
                ${VCE_SLANGC_WARNING_FLAGS}
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
    elseif(SHADER_NAME STREQUAL "triangle_nopt")
        add_custom_command(
            OUTPUT "${SHADER_SPIRV}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}"
            COMMAND "${SLANGC_EXECUTABLE}"
                "${SHADER_SOURCE}"
                -I "${SHADER_SOURCE_DIR}"
                -target spirv
                -profile spirv_1_4
                ${VCE_SLANGC_WARNING_FLAGS}
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
                ${VCE_SLANGC_WARNING_FLAGS}
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
                ${VCE_SLANGC_WARNING_FLAGS}
                -emit-spirv-directly
                -fvk-use-entrypoint-name
                -entry vertMainSkinned
                -o "${SHADER_SPIRV}"
            DEPENDS "${SHADER_SOURCE}" ${SHADER_LIB_SOURCES}
            COMMENT "Compiling ${SHADER_NAME}.slang"
            VERBATIM
        )
    elseif(SHADER_NAME STREQUAL "point_shadow")
        add_custom_command(
            OUTPUT "${SHADER_SPIRV}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}"
            COMMAND "${SLANGC_EXECUTABLE}"
                "${SHADER_SOURCE}"
                -I "${SHADER_SOURCE_DIR}"
                -target spirv
                -profile spirv_1_4
                ${VCE_SLANGC_WARNING_FLAGS}
                -emit-spirv-directly
                -fvk-use-entrypoint-name
                -entry vertMain
                -entry vertMainSkinned
                -entry fragMain
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
                ${VCE_SLANGC_WARNING_FLAGS}
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
    elseif(SHADER_NAME STREQUAL "triangle_nopt")
        set(VCE_NO_POINT_SHADER_SPIRV "${SHADER_SPIRV}")
    elseif(SHADER_NAME STREQUAL "sky")
        set(VCE_SKY_SHADER_SPIRV "${SHADER_SPIRV}")
    elseif(SHADER_NAME STREQUAL "shadow_depth")
        set(VCE_SHADOW_DEPTH_SPIRV "${SHADER_SPIRV}")
    elseif(SHADER_NAME STREQUAL "triangle_skinned")
        set(VCE_SKINNED_SHADER_SPIRV "${SHADER_SPIRV}")
    elseif(SHADER_NAME STREQUAL "shadow_depth_skinned")
        set(VCE_SKINNED_SHADOW_DEPTH_SPIRV "${SHADER_SPIRV}")
    elseif(SHADER_NAME STREQUAL "debug_line")
        set(VCE_DEBUG_LINE_SPIRV "${SHADER_SPIRV}")
    elseif(SHADER_NAME STREQUAL "point_shadow")
        set(VCE_POINT_SHADOW_SPIRV "${SHADER_SPIRV}")
    elseif(SHADER_NAME STREQUAL "particle_billboard")
        set(VCE_PARTICLE_BILLBOARD_SPIRV "${SHADER_SPIRV}")
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

# ALL: without it `make` in this repo silently skipped shader compilation, so an
# edited .slang kept its stale .spv until someone built the demo.
add_custom_target(vce_shaders ALL DEPENDS ${SHADER_SPIRV_OUTPUTS})

add_library(vce_gltf_impl STATIC src/renderer/gltf_loader_impl.cpp)
target_include_directories(vce_gltf_impl SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    "${STB_INCLUDE_DIR}"
)
target_compile_features(vce_gltf_impl PRIVATE cxx_std_23)

add_library(vce_stb STATIC src/renderer/texture_image_stb.cpp)
target_include_directories(vce_stb SYSTEM PRIVATE "${STB_INCLUDE_DIR}")
target_compile_features(vce_stb PRIVATE cxx_std_23)

# miniaudio audio engine (compiled TU, single-header library)
add_library(vce_audio STATIC src/audio/audio_engine_impl.cpp)
target_include_directories(vce_audio SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/miniaudio")
target_include_directories(vce_audio PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(vce_audio PRIVATE cxx_std_23)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(vce_audio PRIVATE dl pthread m)
endif()

# Jolt Physics (third-party, built as static library)
set(JOLT_DIR "$ENV{HOME}/opt/JoltPhysics")
if(NOT EXISTS "${JOLT_DIR}")
    message(FATAL_ERROR "Jolt Physics not found at ${JOLT_DIR}. Clone it: git clone https://github.com/jrouwe/JoltPhysics.git ${JOLT_DIR}")
endif()
add_subdirectory("${JOLT_DIR}/Build" "${CMAKE_BINARY_DIR}/jolt" EXCLUDE_FROM_ALL)

add_library(vce_core INTERFACE)
add_library(VCE::Core ALIAS vce_core)

target_compile_features(vce_core INTERFACE cxx_std_23)

# GLM latches its clip-control convention on its first include per translation
# unit (setup.hpp uses a one-time include guard). Individual headers that
# #define GLM_FORCE_DEPTH_ZERO_TO_ONE are ineffective if any glm header is
# included first without it. Define globally so every TU uses Vulkan 0..1 depth
# regardless of include order.
target_compile_definitions(vce_core INTERFACE
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)

target_include_directories(vce_core SYSTEM INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>"
    "$<BUILD_INTERFACE:${STB_INCLUDE_DIR}>"
)

target_link_libraries(vce_core INTERFACE
    glm::glm
    Jolt::Jolt
    vce_gltf_impl
    vce_stb
    vce_audio
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
    VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS=1
    ENGINE_SHADER_SPIRV="${VCE_SHADER_SPIRV}"
    ENGINE_SKY_SHADER_SPIRV="${VCE_SKY_SHADER_SPIRV}"
    ENGINE_SHADOW_DEPTH_SPIRV="${VCE_SHADOW_DEPTH_SPIRV}"
    ENGINE_SKINNED_SHADER_SPIRV="${VCE_SKINNED_SHADER_SPIRV}"
    ENGINE_SKINNED_SHADOW_DEPTH_SPIRV="${VCE_SKINNED_SHADOW_DEPTH_SPIRV}"
    ENGINE_POINT_SHADOW_SPIRV="${VCE_POINT_SHADOW_SPIRV}"
    ENGINE_NO_POINT_SHADER_SPIRV="${VCE_NO_POINT_SHADER_SPIRV}"
    ENGINE_DEBUG_LINE_SPIRV="${VCE_DEBUG_LINE_SPIRV}"
    ENGINE_PARTICLE_BILLBOARD_SPIRV="${VCE_PARTICLE_BILLBOARD_SPIRV}"
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

# ---------------------------------------------------------------------------
# Header compile gate
#
# The engine is header-only, so INTERFACE targets alone compile none of it:
# `make` in this repo used to build three third-party TUs and report success
# while every renderer/scene/physics header was syntactically unchecked. Breakage
# surfaced later as a build error in the demo or the game — or at runtime.
#
# vce_header_check compiles every engine header, in one TU and again
# individually, as part of the default target. A header that does not compile,
# or that only compiles when some other header was included first, fails HERE.
# It builds no engine functionality; it exists purely to make `make` mean
# something in this repo.
# ---------------------------------------------------------------------------
# Only when this repo is the thing being built. A downstream project already
# compiles the headers it uses; making it also compile all 57 of them twice,
# plus the tests and GPU benchmarks, would just be a slower build for no signal.
option(VCE_BUILD_HEADER_CHECK "Compile every engine header as a build gate" ${PROJECT_IS_TOP_LEVEL})

if(VCE_BUILD_HEADER_CHECK)
file(GLOB_RECURSE VCE_PUBLIC_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp")
list(SORT VCE_PUBLIC_HEADERS)

set(VCE_HEADER_CHECK_DIR "${CMAKE_BINARY_DIR}/header_check")
set(VCE_HEADER_CHECK_SOURCES "")

# One TU including everything: catches cross-header conflicts (ODR, macro
# collisions, ambiguous overloads between headers).
set(VCE_HEADER_CHECK_ALL_BODY "// Generated by CMake. Do not edit.\n")
foreach(HEADER ${VCE_PUBLIC_HEADERS})
    file(RELATIVE_PATH REL "${CMAKE_CURRENT_SOURCE_DIR}/src" "${HEADER}")
    string(APPEND VCE_HEADER_CHECK_ALL_BODY "#include \"${REL}\"\n")
endforeach()
file(GENERATE
    OUTPUT "${VCE_HEADER_CHECK_DIR}/all_headers.cpp"
    CONTENT "${VCE_HEADER_CHECK_ALL_BODY}")
list(APPEND VCE_HEADER_CHECK_SOURCES "${VCE_HEADER_CHECK_DIR}/all_headers.cpp")

# One TU per header: catches headers that are not self-contained, i.e. that
# compile only because a consumer happened to include their dependencies first.
foreach(HEADER ${VCE_PUBLIC_HEADERS})
    file(RELATIVE_PATH REL "${CMAKE_CURRENT_SOURCE_DIR}/src" "${HEADER}")
    string(REPLACE "/" "_" STEM "${REL}")
    string(REPLACE ".hpp" "" STEM "${STEM}")
    file(GENERATE
        OUTPUT "${VCE_HEADER_CHECK_DIR}/solo_${STEM}.cpp"
        CONTENT "// Generated by CMake. Do not edit.\n#include \"${REL}\"\n")
    list(APPEND VCE_HEADER_CHECK_SOURCES "${VCE_HEADER_CHECK_DIR}/solo_${STEM}.cpp")
endforeach()

list(LENGTH VCE_PUBLIC_HEADERS VCE_HEADER_COUNT)
message(STATUS "Header compile gate: ${VCE_HEADER_COUNT} engine headers")

add_library(vce_header_check STATIC ${VCE_HEADER_CHECK_SOURCES})
target_link_libraries(vce_header_check PRIVATE vce_renderer)
set_target_properties(vce_header_check PROPERTIES
    # Headers are pure declarations here; nothing links against this target.
    INTERPROCEDURAL_OPTIMIZATION OFF)
# The generated TUs only declare things, so an empty archive member is expected.
if(NOT MSVC)
    target_compile_options(vce_header_check PRIVATE -Wno-missing-field-initializers)
endif()
add_dependencies(vce_header_check vce_shaders)
endif()

# ---------------------------------------------------------------------------
# Tests
#
# CPU-side invariants only — these run without a Vulkan device, so `make test`
# works on any machine and in CI. Renderer behaviour that genuinely needs a GPU
# stays the demo's job.
# ---------------------------------------------------------------------------
option(VCE_BUILD_TESTS "Build the engine's CPU-side tests" ${PROJECT_IS_TOP_LEVEL})

if(VCE_BUILD_TESTS)
    enable_testing()
    add_executable(vce_tests tests/test_scene_invariants.cpp)
    target_link_libraries(vce_tests PRIVATE vce_renderer)
    if(NOT MSVC)
        target_compile_options(vce_tests PRIVATE -Wno-missing-field-initializers)
    endif()
    add_test(NAME scene_invariants COMMAND vce_tests)

    add_executable(vce_tests_character tests/test_character_slopes.cpp)
    target_link_libraries(vce_tests_character PRIVATE vce_renderer)
    if(NOT MSVC)
        target_compile_options(vce_tests_character PRIVATE -Wno-missing-field-initializers)
    endif()
    add_test(NAME character_slopes COMMAND vce_tests_character)

    # GPU stress test: needs a device and a display, so it is built but never
    # run by ctest. Exercises mesh streaming (create / remesh / drop under
    # frames in flight) against the validation layers, which is the only way to
    # catch a buffer freed while still referenced.
    foreach(GPU_PROG stress_mesh_streaming stress_skinned_horde bench_point_shadows)
        add_executable(vce_${GPU_PROG} tests/${GPU_PROG}.cpp)
        target_link_libraries(vce_${GPU_PROG} PRIVATE vce_renderer)
        # For gpu_test_support.hpp, which turns validation output into a
        # pass/fail result rather than something a human has to notice.
        target_include_directories(vce_${GPU_PROG} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
        if(NOT MSVC)
            target_compile_options(vce_${GPU_PROG} PRIVATE -Wno-missing-field-initializers)
        endif()
    endforeach()
endif()
