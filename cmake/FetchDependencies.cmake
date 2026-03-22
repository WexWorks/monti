include(FetchContent)

# Allow older CMake minimum versions in third-party dependencies
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

# --- VMA (Vulkan Memory Allocator) ---
FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.2.1
    GIT_SHALLOW    TRUE
)

# --- GLM ---
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    TRUE
)

# --- cgltf (header-only glTF loader) ---
FetchContent_Declare(
    cgltf
    GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
    GIT_TAG        v1.14
    GIT_SHALLOW    TRUE
)

# --- tinyexr (header-only EXR loader) ---
FetchContent_Declare(
    tinyexr
    GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
    GIT_TAG        v1.0.9
    GIT_SHALLOW    TRUE
)

# --- Catch2 (test framework) ---
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
    GIT_SHALLOW    TRUE
)

# --- CLI11 (argument parsing) ---
FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.4.2
    GIT_SHALLOW    TRUE
)

# tinyexr options
set(TINYEXR_BUILD_SAMPLE OFF CACHE BOOL "" FORCE)

# Catch2 options
set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(VulkanMemoryAllocator glm cgltf tinyexr Catch2 CLI11)

# stb, FLIP, and MikkTSpace have no usable CMakeLists.txt — download only via
# FetchContent_Populate. CMP0169 OLD allows direct FetchContent_Populate() calls
# (deprecated but necessary when DOWNLOAD_ONLY doesn't prevent add_subdirectory
# in CMake 4.x).
cmake_policy(PUSH)
cmake_policy(SET CMP0169 OLD)

# stb — header-only, no CMakeLists.txt
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
    FetchContent_Populate(stb)
endif()

# dds-ktx — header-only DDS/KTX texture file parser (BC compressed formats)
FetchContent_Declare(
    dds_ktx
    GIT_REPOSITORY https://github.com/septag/dds-ktx.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(dds_ktx)
if(NOT dds_ktx_POPULATED)
    FetchContent_Populate(dds_ktx)
endif()

# MikkTSpace — tangent generation for glTF meshes missing TANGENT attributes
FetchContent_Declare(
    mikktspace
    GIT_REPOSITORY https://github.com/mmikk/MikkTSpace.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(mikktspace)
if(NOT mikktspace_POPULATED)
    FetchContent_Populate(mikktspace)
endif()

# FLIP — header-only core in cpp/; skip its CMakeLists (Python binding deps)
FetchContent_Declare(
    flip
    GIT_REPOSITORY https://github.com/NVlabs/flip.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(flip)
if(NOT flip_POPULATED)
    FetchContent_Populate(flip)
endif()

cmake_policy(POP)

add_library(flip_lib INTERFACE)
target_include_directories(flip_lib SYSTEM INTERFACE "${flip_SOURCE_DIR}/src/cpp")

# --- App-only dependencies (volk, SDL3, ImGui, FreeType, nlohmann/json) ---
# volk is the host app's Vulkan function loader. Libraries (deni_vulkan,
# monti_vulkan) do NOT depend on volk — they accept a PFN_vkGetDeviceProcAddr
# from the host and fall back to the linked Vulkan loader.
if(MONTI_BUILD_APPS)
    # --- volk (Vulkan function pointer loader — app only) ---
    FetchContent_Declare(
        volk
        GIT_REPOSITORY https://github.com/zeux/volk.git
        GIT_TAG        1.4.304
        GIT_SHALLOW    TRUE
    )
    set(VOLK_INSTALL OFF CACHE BOOL "" FORCE)

    # --- SDL3 ---
    FetchContent_Declare(
        SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG        release-3.2.8
        GIT_SHALLOW    TRUE
    )
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    set(SDL_VULKAN ON CACHE BOOL "" FORCE)

    # --- Dear ImGui ---
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.91.8
        GIT_SHALLOW    TRUE
    )

    # --- FreeType ---
    FetchContent_Declare(
        freetype
        GIT_REPOSITORY https://github.com/freetype/freetype.git
        GIT_TAG        VER-2-13-3
        GIT_SHALLOW    TRUE
    )
    set(FT_DISABLE_BROTLI ON CACHE BOOL "" FORCE)
    set(FT_DISABLE_BZIP2 ON CACHE BOOL "" FORCE)
    set(FT_DISABLE_HARFBUZZ ON CACHE BOOL "" FORCE)
    set(FT_DISABLE_PNG ON CACHE BOOL "" FORCE)
    set(FT_DISABLE_ZLIB ON CACHE BOOL "" FORCE)

    # --- nlohmann/json ---
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
        GIT_SHALLOW    TRUE
    )
    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    set(JSON_Install OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(volk SDL3 freetype nlohmann_json)

    # Dear ImGui has no CMakeLists.txt — populate manually
    FetchContent_GetProperties(imgui)
    if(NOT imgui_POPULATED)
        FetchContent_Populate(imgui)
    endif()

    # --- Inter font (SIL Open Font License) ---
    set(INTER_FONT_DIR "${CMAKE_SOURCE_DIR}/app/assets/fonts")
    set(INTER_FONT_PATH "${INTER_FONT_DIR}/Inter-Regular.ttf")
    if(NOT EXISTS "${INTER_FONT_PATH}")
        file(MAKE_DIRECTORY "${INTER_FONT_DIR}")
        message(STATUS "Downloading Inter-Regular.ttf...")
        file(DOWNLOAD
            "https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip"
            "${CMAKE_BINARY_DIR}/Inter-4.1.zip"
            STATUS _INTER_DL_STATUS
        )
        list(GET _INTER_DL_STATUS 0 _INTER_DL_CODE)
        if(_INTER_DL_CODE EQUAL 0)
            file(ARCHIVE_EXTRACT
                INPUT "${CMAKE_BINARY_DIR}/Inter-4.1.zip"
                DESTINATION "${CMAKE_BINARY_DIR}/inter_extract"
                PATTERNS "extras/ttf/Inter-Regular.ttf"
            )
            # Find the extracted TTF
            set(_INTER_TTF_PATH "${CMAKE_BINARY_DIR}/inter_extract/extras/ttf/Inter-Regular.ttf")
            if(EXISTS "${_INTER_TTF_PATH}")
                file(COPY "${_INTER_TTF_PATH}" DESTINATION "${INTER_FONT_DIR}")
                message(STATUS "Inter-Regular.ttf installed to ${INTER_FONT_DIR}")
            else()
                message(WARNING "Inter-Regular.ttf not found in extracted archive")
            endif()
        else()
            list(GET _INTER_DL_STATUS 1 _INTER_DL_MSG)
            message(WARNING "Failed to download Inter font: ${_INTER_DL_MSG}")
        endif()
    endif()
endif()

# ============================================================================
# Extended scene download (Cauldron-Media via Git sparse checkout)
# ============================================================================
if(MONTI_DOWNLOAD_EXTENDED_SCENES)
    set(_CAULDRON_DIR "${CMAKE_SOURCE_DIR}/tests/assets/extended/Cauldron-Media")
    set(_CAULDRON_SCENES AbandonedWarehouse BistroInterior Brutalism)

    # Check if all scene directories already exist (idempotent)
    set(_ALL_PRESENT TRUE)
    foreach(_SCENE ${_CAULDRON_SCENES})
        if(NOT EXISTS "${_CAULDRON_DIR}/${_SCENE}")
            set(_ALL_PRESENT FALSE)
        endif()
    endforeach()

    if(NOT _ALL_PRESENT)
        find_program(GIT_EXECUTABLE git REQUIRED)
        message(STATUS "Downloading Cauldron-Media extended scenes via Git sparse checkout...")

        if(NOT EXISTS "${_CAULDRON_DIR}/.git")
            file(MAKE_DIRECTORY "${_CAULDRON_DIR}")
            # Clone with blob filter (no file content until checkout) and sparse mode
            execute_process(
                COMMAND ${GIT_EXECUTABLE} clone --filter=blob:none --sparse
                        https://github.com/GPUOpen-LibrariesAndSDKs/Cauldron-Media.git
                        "${_CAULDRON_DIR}"
                RESULT_VARIABLE _GIT_CLONE_RESULT
                OUTPUT_VARIABLE _GIT_CLONE_OUT
                ERROR_VARIABLE _GIT_CLONE_ERR
            )
            if(NOT _GIT_CLONE_RESULT EQUAL 0)
                message(WARNING "Git sparse clone failed: ${_GIT_CLONE_ERR}")
            endif()
        endif()

        if(EXISTS "${_CAULDRON_DIR}/.git")
            # Configure sparse checkout to fetch only the 3 target directories
            execute_process(
                COMMAND ${GIT_EXECUTABLE} sparse-checkout set ${_CAULDRON_SCENES}
                WORKING_DIRECTORY "${_CAULDRON_DIR}"
                RESULT_VARIABLE _GIT_SPARSE_RESULT
                OUTPUT_VARIABLE _GIT_SPARSE_OUT
                ERROR_VARIABLE _GIT_SPARSE_ERR
            )
            if(NOT _GIT_SPARSE_RESULT EQUAL 0)
                message(WARNING "Git sparse-checkout set failed: ${_GIT_SPARSE_ERR}")
            else()
                message(STATUS "Cauldron-Media scenes downloaded to ${_CAULDRON_DIR}")
            endif()
        endif()
    else()
        message(STATUS "Cauldron-Media extended scenes already present at ${_CAULDRON_DIR}")
    endif()
endif()
