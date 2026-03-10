cmake_minimum_required(VERSION 3.22)
project(monti LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Options ---
option(MONTI_BUILD_APPS "Build application executables (monti_view, monti_datagen)" OFF)
option(MONTI_DOWNLOAD_TEST_ASSETS "Download Khronos glTF sample models at configure time" ON)
option(MONTI_DOWNLOAD_BENCHMARK_SCENES "Download heavy benchmark scenes (opt-in)" OFF)

# --- Fetch dependencies ---
include(cmake/FetchDependencies.cmake)

# --- Find Vulkan SDK ---
find_package(Vulkan REQUIRED)

# --- Compile GLSL shaders to SPIR-V ---
find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin")
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK or set VULKAN_SDK.")
endif()

# Shader compilation macro — used by renderer and denoiser as they add shaders
function(monti_compile_shaders TARGET_NAME SHADER_DIR SHADER_SOURCES)
    set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

    set(SHADER_SPIRV_OUTPUTS "")
    foreach(SHADER_SRC ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER_SRC} NAME)
        set(SHADER_SPV "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")
        add_custom_command(
            OUTPUT ${SHADER_SPV}
            COMMAND ${GLSLC} --target-env=vulkan1.2 -I ${SHADER_DIR} -O -o ${SHADER_SPV} ${SHADER_SRC}
            DEPENDS ${SHADER_SRC}
            COMMENT "Compiling ${SHADER_NAME} -> SPIR-V"
        )
        list(APPEND SHADER_SPIRV_OUTPUTS ${SHADER_SPV})
    endforeach()

    if(SHADER_SPIRV_OUTPUTS)
        add_custom_target(${TARGET_NAME}_shaders DEPENDS ${SHADER_SPIRV_OUTPUTS})
    endif()
endfunction()

# ============================================================================
# Library: deni_vulkan (standalone denoiser)
# ============================================================================
add_library(deni_vulkan STATIC
    denoise/src/vulkan/Denoiser.cpp
)
target_include_directories(deni_vulkan PUBLIC
    denoise/include
)
target_include_directories(deni_vulkan PRIVATE
    denoise/src
)
target_link_libraries(deni_vulkan PUBLIC
    Vulkan::Headers
    GPUOpen::VulkanMemoryAllocator
)
target_compile_definitions(deni_vulkan PRIVATE
    _CRT_SECURE_NO_WARNINGS
)

# ============================================================================
# Library: monti_scene (platform-agnostic scene layer)
# ============================================================================
add_library(monti_scene STATIC
    scene/src/Scene.cpp
    scene/src/gltf/GltfLoader.cpp
    ${mikktspace_SOURCE_DIR}/mikktspace.c
)
# Suppress warnings in third-party MikkTSpace C source
if(MSVC)
    set_source_files_properties(${mikktspace_SOURCE_DIR}/mikktspace.c
        PROPERTIES COMPILE_FLAGS "/W0")
else()
    set_source_files_properties(${mikktspace_SOURCE_DIR}/mikktspace.c
        PROPERTIES COMPILE_FLAGS "-w")
endif()
target_include_directories(monti_scene PUBLIC
    scene/include
)
target_include_directories(monti_scene PRIVATE
    scene/src
    ${cgltf_SOURCE_DIR}
    ${stb_SOURCE_DIR}
    ${mikktspace_SOURCE_DIR}
)
target_link_libraries(monti_scene PUBLIC
    glm::glm
)

# ============================================================================
# Library: monti_vulkan (path tracer renderer)
# ============================================================================
add_library(monti_vulkan STATIC
    renderer/src/vulkan/Renderer.cpp
)
target_include_directories(monti_vulkan PUBLIC
    renderer/include
)
target_include_directories(monti_vulkan PRIVATE
    renderer/src
)
target_link_libraries(monti_vulkan PUBLIC
    monti_scene
    Vulkan::Headers
    GPUOpen::VulkanMemoryAllocator
)
target_compile_definitions(monti_vulkan PRIVATE
    _CRT_SECURE_NO_WARNINGS
)

# ============================================================================
# Library: monti_capture (OpenEXR writer)
# ============================================================================
add_library(monti_capture STATIC
    capture/src/Writer.cpp
)
target_include_directories(monti_capture PUBLIC
    capture/include
)
target_include_directories(monti_capture PRIVATE
    capture/src
    ${tinyexr_SOURCE_DIR}
)
target_link_libraries(monti_capture PUBLIC
    tinyexr
)

# ============================================================================
# Download Khronos glTF sample models for integration tests
# ============================================================================
set(MONTI_TEST_ASSETS_DIR "${CMAKE_SOURCE_DIR}/tests/assets")
if(MONTI_DOWNLOAD_TEST_ASSETS)
    set(_GLTF_BASE_URL "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models")
    set(_GLTF_ASSETS
        "Box/glTF-Binary/Box.glb"
        "DamagedHelmet/glTF-Binary/DamagedHelmet.glb"
        "DragonAttenuation/glTF-Binary/DragonAttenuation.glb"
        "MosquitoInAmber/glTF-Binary/MosquitoInAmber.glb"
        "ClearCoatTest/glTF-Binary/ClearCoatTest.glb"
        "MaterialsVariantsShoe/glTF-Binary/MaterialsVariantsShoe.glb"
        "MorphPrimitivesTest/glTF-Binary/MorphPrimitivesTest.glb"
    )
    foreach(_ASSET ${_GLTF_ASSETS})
        get_filename_component(_FILENAME ${_ASSET} NAME)
        set(_LOCAL_PATH "${MONTI_TEST_ASSETS_DIR}/${_FILENAME}")
        if(NOT EXISTS "${_LOCAL_PATH}")
            message(STATUS "Downloading test asset: ${_FILENAME}")
            file(DOWNLOAD
                "${_GLTF_BASE_URL}/${_ASSET}"
                "${_LOCAL_PATH}"
                STATUS _DL_STATUS
            )
            list(GET _DL_STATUS 0 _DL_CODE)
            if(NOT _DL_CODE EQUAL 0)
                list(GET _DL_STATUS 1 _DL_MSG)
                message(WARNING "Failed to download ${_FILENAME}: ${_DL_MSG}")
            endif()
        endif()
    endforeach()
endif()

# ============================================================================
# Test target: monti_tests
# ============================================================================
add_executable(monti_tests
    tests/main_test.cpp
    tests/scene_integration_test.cpp
    tests/gltf_loader_test.cpp
    tests/scenes/CornellBox.cpp
)
target_link_libraries(monti_tests PRIVATE
    Catch2::Catch2WithMain
    flip_lib
    deni_vulkan
    monti_scene
    monti_vulkan
    monti_capture
    CLI11::CLI11
)
target_include_directories(monti_tests PRIVATE
    tests
    scene/src
)
target_compile_definitions(monti_tests PRIVATE
    _CRT_SECURE_NO_WARNINGS
    MONTI_TEST_ASSETS_DIR="${MONTI_TEST_ASSETS_DIR}"
)

# ============================================================================
# Compiler warnings (warnings as errors)
# ============================================================================
set(MONTI_TARGETS deni_vulkan monti_scene monti_vulkan monti_capture monti_tests)
foreach(TARGET_NAME ${MONTI_TARGETS})
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    elseif(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /W4 /WX /permissive-)
    endif()
endforeach()

# ============================================================================
# CTest integration
# ============================================================================
include(CTest)
include(Catch)
catch_discover_tests(monti_tests)
