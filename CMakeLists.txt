cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0015 NEW)

project(nova)

# Languages and Packages
find_package(OpenMP REQUIRED)

set(BACKEND "OpenCL" CACHE STRING "Backend \(OpenCL, CUDA)")
string(TOLOWER ${BACKEND} BACKEND)
if (BACKEND STREQUAL "opencl")
    find_package(OpenCL REQUIRED)
elseif (BACKEND STREQUAL "cuda")
    enable_language(CUDA)
else()
    message(FATAL_ERROR "Unknown backend ${BACKEND}")
endif()

# Library includes
set(LIBS_NAME includes)
file(GLOB_RECURSE LIBS_SOURCES include/*)
add_library(${LIBS_NAME} STATIC ${LIBS_SOURCES})
target_include_directories(${LIBS_NAME} PRIVATE include)
target_compile_definitions(${LIBS_NAME} PRIVATE -DGLM_FORCE_INTRINSICS -DGLM_FORCE_SILENT_WARNINGS)
target_compile_options(${LIBS_NAME} PRIVATE -march=native)

# Source files
file(GLOB_RECURSE SOURCES src/*/*)
set_source_files_properties(${SOURCES} PROPERTIES COMPILE_FLAGS
    "-Wall -Wextra -Werror -Wpedantic")

# Link and build
set(PROJECT_LIBS
    ${LIBS_NAME}
    assimp
    OpenMP::OpenMP_CXX
    stdc++fs
)
if (OpenCL_FOUND)
    set(PROJECT_LIBS ${PROJECT_LIBS} ${OpenCL_LIBRARIES})
endif()
add_executable(${PROJECT_NAME} ${SOURCES} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_LIBS})
target_compile_options(${PROJECT_NAME} PRIVATE -std=c++17 -march=native)
target_include_directories(${PROJECT_NAME} PRIVATE src include)

# Options
option(RELEASE "Build in release mode" ON)
option(LOG "Enable logging" OFF)
option(PROFILE "Enable profiling" OFF)

if (RELEASE)
    target_compile_options(${PROJECT_NAME} PRIVATE -O3)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DNDEBUG)
    set(LOG OFF)
    set(PROFILE OFF)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -g -O3)
    set(LOG ON)
    set(PROFILE ON)
endif()

if (LOG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DLOG)
endif()

if (PROFILE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DPROFILE)
endif()

# Defines
if (OpenCL_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        -DCL_HPP_ENABLE_EXCEPTIONS -D__CL_ENABLE_EXCEPTIONS -DBACKEND_OPENCL)
    if (${OpenCL_VERSION_MAJOR} EQUAL 2)
        target_compile_definitions(${PROJECT_NAME} PRIVATE -DCL_HPP_TARGET_OPENCL_VERSION=200)
        target_compile_definitions(${PROJECT_NAME} PRIVATE -DOPENCL_2)
    else()
        target_compile_definitions(${PROJECT_NAME} PRIVATE -DCL_HPP_TARGET_OPENCL_VERSION=120)
    endif()
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DBACKEND_CUDA)
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE -DSRC_PATH="${CMAKE_SOURCE_DIR}/")
target_compile_definitions(${PROJECT_NAME} PRIVATE -DKERNELS_PATH="${CMAKE_SOURCE_DIR}/kernels/")
target_compile_definitions(${PROJECT_NAME} PRIVATE -DASSETS_PATH="${CMAKE_SOURCE_DIR}/assets/")

# Status
message("Build Options -----------------------------------")
message("RELEASE ----------------------------------------- ${RELEASE}")
message("LOG --------------------------------------------- ${LOG}")
message("PROFILE ----------------------------------------- ${PROFILE}")
if (OpenCL_FOUND)
    message("OpenCL Version ---------------------------------- ${OpenCL_VERSION_STRING}")
else()
    message("CUDA Version ------------------------------------ ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

add_subdirectory(tests)

# Assimp
set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT CACHE BOOL OFF)
set(ASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT CACHE BOOL OFF)
set(ASSIMP_NO_EXPORT ON)
set(ASSIMP_BUILD_ASSIMP_TOOLS CACHE BOOL OFF)
set(ASSIMP_BUILD_TESTS CACHE BOOL OFF)
set(ASSIMP_BUILD_OBJ_IMPORTER ON)
set(ASSIMP_BUILD_3DS_IMPORTER ON)
set(ASSIMP_BUILD_BLEND_IMPORTER ON)
set(ASSIMP_BUILD_FBX_IMPORTER ON)
set(ASSIMP_BUILD_GLTF_IMPORTER ON)
add_subdirectory(assimp)
