cmake_minimum_required(VERSION 3.22)
project(monti LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Options ---
option(MONTI_BUILD_APPS "Build application executables (monti_view, monti_datagen)" ON)
option(MONTI_DOWNLOAD_TEST_ASSETS "Download Khronos glTF sample models at configure time" ON)
option(MONTI_DOWNLOAD_BENCHMARK_SCENES "Download heavy benchmark scenes (opt-in)" ON)
option(MONTI_DOWNLOAD_EXTENDED_SCENES "Download Cauldron-Media extended scenes via Git sparse checkout" ON)

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
    denoise/src/vulkan/WeightLoader.cpp
    denoise/src/vulkan/MlInference.cpp
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
    ${DENI_SHADER_DIR}/conv.comp
    ${DENI_SHADER_DIR}/group_norm_reduce.comp
    ${DENI_SHADER_DIR}/group_norm_apply.comp
    ${DENI_SHADER_DIR}/encoder_input_conv.comp
    ${DENI_SHADER_DIR}/downsample.comp
    ${DENI_SHADER_DIR}/upsample_concat.comp
    ${DENI_SHADER_DIR}/output_conv.comp
    ${DENI_SHADER_DIR}/reproject.comp
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

# Copy model to build directory from the denoiser's own models/ folder.
# The model is distributed with the denoiser library, not the application.
set(DENI_MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/deni_models")
set(DENI_MODEL_SOURCE "${CMAKE_SOURCE_DIR}/denoise/models/deni_v1.denimodel")
if(EXISTS "${DENI_MODEL_SOURCE}")
    file(MAKE_DIRECTORY ${DENI_MODEL_DIR})
    add_custom_command(
        OUTPUT "${DENI_MODEL_DIR}/deni_v1.denimodel"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DENI_MODEL_SOURCE}"
            "${DENI_MODEL_DIR}/deni_v1.denimodel"
        DEPENDS "${DENI_MODEL_SOURCE}"
        COMMENT "Copying deni_v1.denimodel to build directory"
    )
    add_custom_target(deni_model DEPENDS "${DENI_MODEL_DIR}/deni_v1.denimodel")
    add_dependencies(deni_vulkan deni_model)
    target_compile_definitions(deni_vulkan PRIVATE
        DENI_MODEL_DIR="${DENI_MODEL_DIR}"
    )
endif()

# ============================================================================
# Library: monti_scene (platform-agnostic scene layer)
# ============================================================================
add_library(monti_scene STATIC
    renderer/scene/src/Scene.cpp
    renderer/scene/src/gltf/GltfLoader.cpp
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
    renderer/scene/include
)
target_include_directories(monti_scene PRIVATE
    renderer/scene/src
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
    renderer/src/vulkan/EmissiveLightExtractor.cpp
    renderer/src/vulkan/vma_impl.cpp
)

# Compile ray tracing shaders
set(MONTI_SHADER_DIR "${CMAKE_SOURCE_DIR}/renderer/src/vulkan/shaders")
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
    capture/src/Luminance.cpp
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
set(MONTI_TEST_ASSETS_DIR "${CMAKE_SOURCE_DIR}/scenes/khronos")
set(MONTI_DEBUG_SCENES_DIR "${CMAKE_SOURCE_DIR}/scenes/debug")
if(MONTI_DOWNLOAD_TEST_ASSETS)
    set(_GLTF_BASE_URL "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models")
    set(_GLTF_ASSETS
        "ABeautifulGame/glTF-Binary/ABeautifulGame.glb"
        "AntiqueCamera/glTF-Binary/AntiqueCamera.glb"
        "BoomBox/glTF-Binary/BoomBox.glb"
        "ClearCoatTest/glTF-Binary/ClearCoatTest.glb"
        "DamagedHelmet/glTF-Binary/DamagedHelmet.glb"
        "DragonAttenuation/glTF-Binary/DragonAttenuation.glb"
        "Lantern/glTF-Binary/Lantern.glb"
        "MaterialsVariantsShoe/glTF-Binary/MaterialsVariantsShoe.glb"
        "MosquitoInAmber/glTF-Binary/MosquitoInAmber.glb"
        "SheenChair/glTF-Binary/SheenChair.glb"
        "ToyCar/glTF-Binary/ToyCar.glb"
        "WaterBottle/glTF-Binary/WaterBottle.glb"
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

    # Box.glb is too simple for training; download to scenes/debug/ instead
    set(_BOX_PATH "${MONTI_DEBUG_SCENES_DIR}/Box.glb")
    if(NOT EXISTS "${_BOX_PATH}")
        message(STATUS "Downloading debug asset: Box.glb")
        file(DOWNLOAD
            "${_GLTF_BASE_URL}/Box/glTF-Binary/Box.glb"
            "${_BOX_PATH}"
            STATUS _DL_STATUS
        )
        list(GET _DL_STATUS 0 _DL_CODE)
        if(NOT _DL_CODE EQUAL 0)
            list(GET _DL_STATUS 1 _DL_MSG)
            message(WARNING "Failed to download Box.glb: ${_DL_MSG}")
        endif()
    endif()

    set(_MORPH_PATH "${MONTI_DEBUG_SCENES_DIR}/MorphPrimitivesTest.glb")
    if(NOT EXISTS "${_MORPH_PATH}")
        message(STATUS "Downloading debug asset: MorphPrimitivesTest.glb")
        file(DOWNLOAD
            "${_GLTF_BASE_URL}/MorphPrimitivesTest/glTF-Binary/MorphPrimitivesTest.glb"
            "${_MORPH_PATH}"
            STATUS _DL_STATUS
        )
        list(GET _DL_STATUS 0 _DL_CODE)
        if(NOT _DL_CODE EQUAL 0)
            list(GET _DL_STATUS 1 _DL_MSG)
            message(WARNING "Failed to download MorphPrimitivesTest.glb: ${_DL_MSG}")
        endif()
    endif()

    set(_GLASS_PATH "${MONTI_DEBUG_SCENES_DIR}/GlassHurricaneCandleHolder.glb")
    if(NOT EXISTS "${_GLASS_PATH}")
        message(STATUS "Downloading debug asset: GlassHurricaneCandleHolder.glb")
        file(DOWNLOAD
            "${_GLTF_BASE_URL}/GlassHurricaneCandleHolder/glTF-Binary/GlassHurricaneCandleHolder.glb"
            "${_GLASS_PATH}"
            STATUS _DL_STATUS
        )
        list(GET _DL_STATUS 0 _DL_CODE)
        if(NOT _DL_CODE EQUAL 0)
            list(GET _DL_STATUS 1 _DL_MSG)
            message(WARNING "Failed to download GlassHurricaneCandleHolder.glb: ${_DL_MSG}")
        endif()
    endif()

    # Multi-file glTF models (FlightHelmet, Sponza) — downloaded via Git sparse checkout
    set(_GLTF_MULTI_MODELS FlightHelmet Sponza)
    set(_ALL_MULTI_PRESENT TRUE)
    foreach(_MODEL ${_GLTF_MULTI_MODELS})
        if(NOT EXISTS "${MONTI_TEST_ASSETS_DIR}/${_MODEL}/${_MODEL}.gltf")
            set(_ALL_MULTI_PRESENT FALSE)
        endif()
    endforeach()

    if(NOT _ALL_MULTI_PRESENT)
        find_program(GIT_EXECUTABLE git REQUIRED)
        set(_GLTF_CLONE_DIR "${CMAKE_BINARY_DIR}/_gltf_sample_assets")

        if(NOT EXISTS "${_GLTF_CLONE_DIR}/.git")
            message(STATUS "Cloning glTF-Sample-Assets for multi-file models...")
            execute_process(
                COMMAND ${GIT_EXECUTABLE} clone --filter=blob:none --sparse --depth=1
                        https://github.com/KhronosGroup/glTF-Sample-Assets.git
                        "${_GLTF_CLONE_DIR}"
                RESULT_VARIABLE _GIT_RESULT
            )
            if(NOT _GIT_RESULT EQUAL 0)
                message(WARNING "Failed to clone glTF-Sample-Assets")
            endif()
        endif()

        if(EXISTS "${_GLTF_CLONE_DIR}/.git")
            # Build sparse checkout paths
            set(_SPARSE_PATHS "")
            foreach(_MODEL ${_GLTF_MULTI_MODELS})
                list(APPEND _SPARSE_PATHS "Models/${_MODEL}/glTF")
            endforeach()

            execute_process(
                COMMAND ${GIT_EXECUTABLE} sparse-checkout set ${_SPARSE_PATHS}
                WORKING_DIRECTORY "${_GLTF_CLONE_DIR}"
                RESULT_VARIABLE _GIT_SPARSE_RESULT
            )

            if(_GIT_SPARSE_RESULT EQUAL 0)
                foreach(_MODEL ${_GLTF_MULTI_MODELS})
                    set(_SRC "${_GLTF_CLONE_DIR}/Models/${_MODEL}/glTF")
                    set(_DST "${MONTI_TEST_ASSETS_DIR}/${_MODEL}")
                    if(EXISTS "${_SRC}/${_MODEL}.gltf" AND NOT EXISTS "${_DST}/${_MODEL}.gltf")
                        message(STATUS "Installing multi-file model: ${_MODEL}")
                        file(COPY "${_SRC}/" DESTINATION "${_DST}")
                    endif()
                endforeach()
            else()
                message(WARNING "Failed to sparse-checkout glTF-Sample-Assets")
            endif()
        endif()
    endif()

    # Generate DDS test data for Phase 8N tests
    set(_DDS_TEST_DIR "${MONTI_TEST_ASSETS_DIR}/dds")
    if(NOT EXISTS "${_DDS_TEST_DIR}/test_bc7.dds")
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_FOUND)
            set(_DDS_GEN_ARGS
                --output-dir "${_DDS_TEST_DIR}"
            )
            set(_BISTRO_DIR "${CMAKE_SOURCE_DIR}/scenes/extended/Cauldron-Media/BistroInterior")
            if(EXISTS "${_BISTRO_DIR}/Cloth_BaseColor.dds")
                list(APPEND _DDS_GEN_ARGS --bistro-dir "${_BISTRO_DIR}")
            endif()
            message(STATUS "Generating DDS test data for Phase 8N...")
            execute_process(
                COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_SOURCE_DIR}/tests/generate_dds_test_data.py"
                    ${_DDS_GEN_ARGS}
                RESULT_VARIABLE _DDS_GEN_RESULT
            )
            if(NOT _DDS_GEN_RESULT EQUAL 0)
                message(WARNING "Failed to generate DDS test data")
            endif()
        else()
            message(WARNING "Python3 not found — cannot generate DDS test data for Phase 8N tests")
        endif()
    endif()
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
        ${APP_SHADER_DIR}/luminance.comp
        ${APP_SHADER_DIR}/luminance_resolve.comp
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
        app/core/AutoExposure.cpp
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
        renderer/scene/src
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
        renderer/scene/src
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
    tests/luminance_test.cpp
    tests/ml_weight_loader_test.cpp
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
    renderer/scene/src
    ${stb_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}
)
set(MONTI_TEST_DEFS
    _CRT_SECURE_NO_WARNINGS
    MONTI_TEST_ASSETS_DIR="${MONTI_TEST_ASSETS_DIR}"
    MONTI_DEBUG_SCENES_DIR="${MONTI_DEBUG_SCENES_DIR}"
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
        tests/phase8j_test.cpp
        tests/phase8k_test.cpp
        tests/phase9b_test.cpp
        tests/phase10a_test.cpp
        tests/extended_scene_test.cpp
        tests/nan_regression_test.cpp
        tests/golden_test.cpp
        tests/deni_passthrough_test.cpp
        tests/ml_inference_test.cpp
        tests/ml_inference_numerical_test.cpp
        tests/ml_denoiser_integration_test.cpp
        tests/temporal_reproject_test.cpp
        tests/ml_e2e_quality_test.cpp
        tests/auto_exposure_test.cpp
        tests/datagen_path_order_test.cpp
        tests/datagen_session_e2e_test.cpp
        app/datagen/GenerationSession.cpp
        app/core/vulkan_context.cpp
        app/core/GBufferImages.cpp
        app/core/ToneMapper.cpp
        app/core/AutoExposure.cpp
        app/core/EnvironmentLoader.cpp
    )
    list(APPEND MONTI_TEST_LIBS
        volk::volk
        nlohmann_json::nlohmann_json
    )
    list(APPEND MONTI_TEST_INCLUDES
        app/core
        renderer/src
        ${dds_ktx_SOURCE_DIR}
    )
    list(APPEND MONTI_TEST_DEFS
        VK_NO_PROTOTYPES
        MONTI_ENABLE_VALIDATION
        MONTI_SHADER_SPV_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders"
        DENI_SHADER_SPV_DIR="${CMAKE_CURRENT_BINARY_DIR}/deni_shaders"
        DENI_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data"
        DENI_MODEL_DIR="${DENI_MODEL_DIR}"
        APP_SHADER_SPV_DIR="${CMAKE_CURRENT_BINARY_DIR}/app_shaders"
        CAPTURE_SHADER_SPV_DIR="${CAPTURE_SHADER_OUTPUT_DIR}"
        MONTI_ENVIRONMENTS_DIR="${CMAKE_SOURCE_DIR}/scenes/environments"
    )
    list(APPEND MONTI_TEST_DEFS
        MONTI_EXTENDED_SCENES_DIR="${CMAKE_SOURCE_DIR}/scenes/extended/Cauldron-Media"
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
if(TARGET capture_shaders)
    add_dependencies(monti_tests capture_shaders)
endif()
if(TARGET deni_model)
    add_dependencies(monti_tests deni_model)
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

# ============================================================================
# Batched test targets
# ============================================================================
# Running all tests in a single process can exhaust GPU resources (VMA
# fragmentation / BLAS compaction failures after ~150 heavy tests share one
# VkDevice).  These targets split the test suite into batches, each in its
# own process, avoiding the cascade.
#
#   cmake --build build --config Release --target run_all_tests
#
set(MONTI_TEST_EXE "$<TARGET_FILE:monti_tests>")

add_custom_target(run_tests_core
    COMMAND ${MONTI_TEST_EXE} "~[extended]~[golden]~[deni]~[numerical]"
    DEPENDS monti_tests
    COMMENT "Batch 1/4: Core tests"
    VERBATIM
)
add_custom_target(run_tests_golden
    COMMAND ${MONTI_TEST_EXE} "[golden]"
    DEPENDS monti_tests
    COMMENT "Batch 2/4: Golden tests"
    VERBATIM
)
add_custom_target(run_tests_extended
    COMMAND ${MONTI_TEST_EXE} "[extended]"
    DEPENDS monti_tests
    COMMENT "Batch 3/4: Extended scene tests"
    VERBATIM
)
add_custom_target(run_tests_deni
    COMMAND ${MONTI_TEST_EXE} "[deni],[numerical]"
    DEPENDS monti_tests
    COMMENT "Batch 4/4: Denoiser and numerical tests"
    VERBATIM
)
add_custom_target(run_all_tests
    DEPENDS run_tests_core run_tests_golden run_tests_extended run_tests_deni
    COMMENT "All test batches complete"
)

# ── Scene patching ────────────────────────────────────────────────────────
# Manual target:  cmake --build build --target patch_bistro_glass
find_program(MONTI_PYTHON NAMES python3 python)
if(MONTI_PYTHON)
    set(_BISTRO_GLTF "${CMAKE_SOURCE_DIR}/scenes/extended/Cauldron-Media/BistroInterior/scene.gltf")
    set(_PATCH_SCRIPT "${CMAKE_SOURCE_DIR}/scenes/patch_bistro_glass.py")
    add_custom_target(patch_bistro_glass
        COMMAND "${MONTI_PYTHON}" "${_PATCH_SCRIPT}" "${_BISTRO_GLTF}"
        COMMENT "Patching BistroInterior glass materials with transmission extensions"
        VERBATIM
    )
endif()
