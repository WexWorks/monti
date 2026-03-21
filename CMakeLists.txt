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

    # Collect shared include files so shaders rebuild when includes change
    file(GLOB SHADER_INCLUDES "${SHADER_DIR}/include/*.glsl")

    set(SHADER_SPIRV_OUTPUTS "")
    foreach(SHADER_SRC ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER_SRC} NAME)
        set(SHADER_SPV "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")
        add_custom_command(
            OUTPUT ${SHADER_SPV}
            COMMAND ${GLSLC} --target-env=vulkan1.2 -I ${SHADER_DIR} -O -o ${SHADER_SPV} ${SHADER_SRC}
            DEPENDS ${SHADER_SRC} ${SHADER_INCLUDES}
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

# Compile deni shaders to SPIR-V
set(DENI_SHADER_DIR "${CMAKE_SOURCE_DIR}/denoise/src/vulkan/shaders")
set(DENI_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/deni_shaders")
file(MAKE_DIRECTORY ${DENI_SHADER_OUTPUT_DIR})
set(DENI_SHADER_SOURCES
    ${DENI_SHADER_DIR}/passthrough_denoise.comp
)
set(DENI_SHADER_SPIRV_OUTPUTS "")
foreach(SHADER_SRC ${DENI_SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER_SRC} NAME)
    set(SHADER_SPV "${DENI_SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SHADER_SPV}
        COMMAND ${GLSLC} --target-env=vulkan1.2 -O -o ${SHADER_SPV} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Compiling deni ${SHADER_NAME} -> SPIR-V"
    )
    list(APPEND DENI_SHADER_SPIRV_OUTPUTS ${SHADER_SPV})
endforeach()
add_custom_target(deni_vulkan_shaders DEPENDS ${DENI_SHADER_SPIRV_OUTPUTS})
add_dependencies(deni_vulkan deni_vulkan_shaders)

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
    ${dds_ktx_SOURCE_DIR}
)
target_link_libraries(monti_scene PUBLIC
    glm::glm
)

# ============================================================================
# Library: monti_vulkan (path tracer renderer)
# ============================================================================
add_library(monti_vulkan STATIC
    renderer/src/vulkan/Renderer.cpp
    renderer/src/vulkan/Buffer.cpp
    renderer/src/vulkan/Image.cpp
    renderer/src/vulkan/Upload.cpp
    renderer/src/vulkan/GpuScene.cpp
    renderer/src/vulkan/GpuBufferUtils.cpp
    renderer/src/vulkan/GeometryManager.cpp
    renderer/src/vulkan/EnvironmentMap.cpp
    renderer/src/vulkan/BlueNoise.cpp
    renderer/src/vulkan/RaytracePipeline.cpp
)

# Compile ray tracing shaders
set(MONTI_SHADER_DIR "${CMAKE_SOURCE_DIR}/shaders")
set(MONTI_SHADER_SOURCES
    ${MONTI_SHADER_DIR}/raygen.rgen
    ${MONTI_SHADER_DIR}/miss.rmiss
    ${MONTI_SHADER_DIR}/closesthit.rchit
    ${MONTI_SHADER_DIR}/anyhit.rahit
)
monti_compile_shaders(monti_vulkan ${MONTI_SHADER_DIR} "${MONTI_SHADER_SOURCES}")
add_dependencies(monti_vulkan monti_vulkan_shaders)
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
    VK_NO_PROTOTYPES
)

# ============================================================================
# Library: monti_capture (OpenEXR writer)
# ============================================================================
add_library(monti_capture STATIC
    capture/src/Writer.cpp
    capture/src/GpuReadback.cpp
    capture/src/GpuAccumulator.cpp
)
target_include_directories(monti_capture PUBLIC
    capture/include
)
target_include_directories(monti_capture PRIVATE
    capture/src
    renderer/include
    ${tinyexr_SOURCE_DIR}
)
target_link_libraries(monti_capture PUBLIC
    tinyexr
    glm::glm
    GPUOpen::VulkanMemoryAllocator
    Vulkan::Headers
)
target_compile_definitions(monti_capture PRIVATE
    VK_NO_PROTOTYPES
)

# Compile capture shaders to SPIR-V
set(CAPTURE_SHADER_DIR "${CMAKE_SOURCE_DIR}/capture/shaders")
set(CAPTURE_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/capture_shaders")
file(MAKE_DIRECTORY ${CAPTURE_SHADER_OUTPUT_DIR})
set(CAPTURE_SHADER_SOURCES
    ${CAPTURE_SHADER_DIR}/accumulate.comp
)
set(CAPTURE_SHADER_SPIRV_OUTPUTS "")
foreach(SHADER_SRC ${CAPTURE_SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER_SRC} NAME)
    set(SHADER_SPV "${CAPTURE_SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SHADER_SPV}
        COMMAND ${GLSLC} --target-env=vulkan1.2 -O -o ${SHADER_SPV} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Compiling capture ${SHADER_NAME} -> SPIR-V"
    )
    list(APPEND CAPTURE_SHADER_SPIRV_OUTPUTS ${SHADER_SPV})
endforeach()
add_custom_target(capture_shaders DEPENDS ${CAPTURE_SHADER_SPIRV_OUTPUTS})
add_dependencies(monti_capture capture_shaders)

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
# ============================================================================
# Application targets (behind MONTI_BUILD_APPS)
# ============================================================================
if(MONTI_BUILD_APPS)
    # ImGui sources (no CMakeLists.txt in the imgui repo)
    set(IMGUI_SOURCES
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
        ${imgui_SOURCE_DIR}/misc/freetype/imgui_freetype.cpp
    )

    # Compile app shaders to SPIR-V
    set(APP_SHADER_DIR "${CMAKE_SOURCE_DIR}/app/shaders")
    set(APP_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/app_shaders")
    file(MAKE_DIRECTORY ${APP_SHADER_OUTPUT_DIR})
    set(APP_SHADER_SOURCES
        ${APP_SHADER_DIR}/tonemap.comp
    )
    set(APP_SHADER_SPIRV_OUTPUTS "")
    foreach(SHADER_SRC ${APP_SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER_SRC} NAME)
        set(SHADER_SPV "${APP_SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")
        add_custom_command(
            OUTPUT ${SHADER_SPV}
            COMMAND ${GLSLC} --target-env=vulkan1.2 -O -o ${SHADER_SPV} ${SHADER_SRC}
            DEPENDS ${SHADER_SRC}
            COMMENT "Compiling app ${SHADER_NAME} -> SPIR-V"
        )
        list(APPEND APP_SHADER_SPIRV_OUTPUTS ${SHADER_SPV})
    endforeach()
    add_custom_target(app_shaders DEPENDS ${APP_SHADER_SPIRV_OUTPUTS})

    # Shared app core sources
    set(CORE_SOURCES
        app/core/vulkan_context.cpp
        app/core/frame_resources.cpp
        app/core/GBufferImages.cpp
        app/core/ToneMapper.cpp
        app/core/EnvironmentLoader.cpp
    )

    set(CORE_LIBS
        monti_scene
        monti_vulkan
        deni_vulkan
        volk::volk
        GPUOpen::VulkanMemoryAllocator
        glm::glm
        Vulkan::Headers
        tinyexr
    )

    set(CORE_DEFS
        VK_NO_PROTOTYPES
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        _CRT_SECURE_NO_WARNINGS
    )

    # --- monti_view (interactive viewer) ---
    set(VIEW_SOURCES
        app/view/main.cpp
        app/view/swapchain.cpp
        app/view/CameraController.cpp
        app/view/UiRenderer.cpp
        app/view/Panels.cpp
    )

    add_executable(monti_view ${CORE_SOURCES} ${VIEW_SOURCES} ${IMGUI_SOURCES})

    target_link_libraries(monti_view PRIVATE
        ${CORE_LIBS}
        SDL3::SDL3-static
        CLI11::CLI11
        freetype
        nlohmann_json::nlohmann_json
    )

    target_include_directories(monti_view PRIVATE
        app/core
        app/view
        scene/src
        ${tinyexr_SOURCE_DIR}
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
    )

    target_compile_definitions(monti_view PRIVATE
        ${CORE_DEFS}
        APP_SHADER_SPV_DIR="${APP_SHADER_OUTPUT_DIR}"
        MONTI_SHADER_SPV_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders"
        DENI_SHADER_SPV_DIR="${CMAKE_CURRENT_BINARY_DIR}/deni_shaders"
        MONTI_FONT_DIR="${CMAKE_SOURCE_DIR}/app/assets/fonts"
        IMGUI_ENABLE_FREETYPE
    )
    add_dependencies(monti_view app_shaders)

    # --- monti_datagen (headless training data generator) ---
    set(DATAGEN_SOURCES
        app/datagen/main.cpp
        app/datagen/GenerationSession.cpp
    )

    add_executable(monti_datagen ${CORE_SOURCES} ${DATAGEN_SOURCES})

    target_link_libraries(monti_datagen PRIVATE
        ${CORE_LIBS}
        monti_capture
        CLI11::CLI11
        nlohmann_json::nlohmann_json
    )

    target_include_directories(monti_datagen PRIVATE
        app/core
        scene/src
        ${tinyexr_SOURCE_DIR}
    )

    target_compile_definitions(monti_datagen PRIVATE
        ${CORE_DEFS}
        MONTI_SHADER_SPV_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders"
        CAPTURE_SHADER_SPV_DIR="${CAPTURE_SHADER_OUTPUT_DIR}"
    )
    add_dependencies(monti_datagen capture_shaders)
endif()

# ============================================================================
# Test target: monti_tests
# ============================================================================
set(MONTI_TEST_SOURCES
    tests/main_test.cpp
    tests/scene_integration_test.cpp
    tests/gltf_loader_test.cpp
    tests/capture_writer_test.cpp
    tests/phase11b_test.cpp
    tests/scenes/CornellBox.cpp
    tests/scenes/Primitives.cpp
)
set(MONTI_TEST_LIBS
    Catch2::Catch2WithMain
    flip_lib
    deni_vulkan
    monti_scene
    monti_vulkan
    monti_capture
    CLI11::CLI11
)
set(MONTI_TEST_INCLUDES
    tests
    scene/src
    ${stb_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}
)
set(MONTI_TEST_DEFS
    _CRT_SECURE_NO_WARNINGS
    MONTI_TEST_ASSETS_DIR="${MONTI_TEST_ASSETS_DIR}"
)

# Vulkan context integration test requires volk (app dependency)
if(MONTI_BUILD_APPS)
    list(APPEND MONTI_TEST_SOURCES
        tests/vulkan_context_test.cpp
        tests/gpu_scene_test.cpp
        tests/geometry_manager_test.cpp
        tests/phase7a_test.cpp
        tests/phase7b_test.cpp
        tests/phase7c_test.cpp
        tests/phase8b_test.cpp
        tests/phase8c_test.cpp
        tests/phase8d_test.cpp
        tests/phase8e_test.cpp
        tests/phase8f_test.cpp
        tests/phase8h_test.cpp
        tests/phase8i_test.cpp
        tests/phase8l_test.cpp
        tests/phase8g_test.cpp
        tests/phase8m_test.cpp
        tests/phase8n_test.cpp
        tests/phase9b_test.cpp
        tests/phase10a_test.cpp
        tests/deni_passthrough_test.cpp
        app/core/vulkan_context.cpp
        app/core/GBufferImages.cpp
        app/core/ToneMapper.cpp
    )
    list(APPEND MONTI_TEST_LIBS
        volk::volk
    )
    list(APPEND MONTI_TEST_INCLUDES
        app/core
        renderer/src
    )
    list(APPEND MONTI_TEST_DEFS
        VK_NO_PROTOTYPES
        MONTI_SHADER_SPV_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders"
        DENI_SHADER_SPV_DIR="${CMAKE_CURRENT_BINARY_DIR}/deni_shaders"
        APP_SHADER_SPV_DIR="${CMAKE_CURRENT_BINARY_DIR}/app_shaders"
    )
endif()

add_executable(monti_tests ${MONTI_TEST_SOURCES})
target_link_libraries(monti_tests PRIVATE ${MONTI_TEST_LIBS})
target_include_directories(monti_tests PRIVATE ${MONTI_TEST_INCLUDES})
target_compile_definitions(monti_tests PRIVATE ${MONTI_TEST_DEFS})
if(TARGET monti_vulkan_shaders)
    add_dependencies(monti_tests monti_vulkan_shaders)
endif()
if(TARGET deni_vulkan_shaders)
    add_dependencies(monti_tests deni_vulkan_shaders)
endif()
if(TARGET app_shaders)
    add_dependencies(monti_tests app_shaders)
endif()

# ============================================================================
# Compiler warnings (warnings as errors)
# ============================================================================
set(MONTI_TARGETS deni_vulkan monti_scene monti_vulkan monti_capture monti_tests)
if(MONTI_BUILD_APPS)
    list(APPEND MONTI_TARGETS monti_view monti_datagen)
endif()
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
