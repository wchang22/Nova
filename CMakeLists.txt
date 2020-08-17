cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0015 NEW)

project(nova)
set(CMAKE_CXX_STANDARD 17)

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

# Dependencies that require compilation
add_subdirectory(dependencies)

# Library includes
set(LIBS_NAME includes)
file(GLOB_RECURSE LIBS_SOURCES include/*)
add_library(${LIBS_NAME} STATIC ${LIBS_SOURCES})
target_link_libraries(${LIBS_NAME} PRIVATE ${CMAKE_DL_LIBS} glfw)
target_include_directories(${LIBS_NAME} PRIVATE include ${GLFW_INCLUDE_PATH})
target_compile_definitions(${LIBS_NAME} PRIVATE
    GLM_FORCE_INTRINSICS
    GLM_FORCE_SILENT_WARNINGS
    IMGUI_IMPL_OPENGL_LOADER_GLAD
)
target_compile_options(${LIBS_NAME} PRIVATE -march=native -O3)

# Source files
file(GLOB_RECURSE SOURCES
    src/backend/common/*
    src/backend/accelerator.h
    src/backend/types.h
    src/bvh/*
    src/camera/*
    src/core/*
    src/intersectables/*
    src/kernel_types/*
    src/material/*
    src/model/*
    src/scene/*
    src/util/*
    src/vector/*
    src/window/*
    src/constants.h
)
if (OpenCL_FOUND)
    file(GLOB_RECURSE BACKEND_SOURCES src/backend/opencl/*)
else()
    file(GLOB_RECURSE BACKEND_SOURCES src/backend/cuda/*)
endif()
set(SOURCES ${SOURCES} ${BACKEND_SOURCES})

set_source_files_properties(${SOURCES} PROPERTIES COMPILE_FLAGS
    "-Wall -Wextra -Werror -Wpedantic -Wno-missing-braces")

# Kernels
if (OpenCL_FOUND)
    set(OPENCL_LIB_NAME opencl)
    set(OPENCL_KERNEL_BINARY ${PROJECT_NAME}.spv)
    if (RELEASE)
        set(OPENCL_NDEBUG -DNDEBUG)
    else()
        set(OPENCL_KERNEL_DEBUG -DKERNEL_DEBUG)
    endif()
    add_custom_target(
        ${OPENCL_LIB_NAME} ALL
        COMMAND ${CMAKE_CXX_COMPILER} -c -target spir64-unknown-unknown -emit-llvm -Xclang 
        -finclude-default-header -cl-std=clc++
        -I${CMAKE_SOURCE_DIR}/src -I${CMAKE_SOURCE_DIR}
        -DKERNEL_BACKEND_OPENCL ${OPENCL_NDEBUG} ${OPENCL_KERNEL_DEBUG}
        ${CMAKE_SOURCE_DIR}/kernels/entrypoint/opencl/entrypoint.cpp -o ${PROJECT_NAME}.bc &&
        llvm-spirv --spirv-max-version=1.0 ${PROJECT_NAME}.bc
    )
else()
    set(CUDA_LIB_NAME cuda)
    file(GLOB_RECURSE CUDA_SOURCES kernels/*)
    add_library(${CUDA_LIB_NAME} STATIC ${CUDA_SOURCES})
    target_include_directories(${CUDA_LIB_NAME} PRIVATE src ${CMAKE_SOURCE_DIR})
    target_compile_options(${CUDA_LIB_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math 
        --gpu-architecture=sm_50
        -maxrregcount=64
        >
    )
    target_compile_definitions(${CUDA_LIB_NAME} PRIVATE KERNEL_BACKEND_CUDA)
    set_target_properties(${CUDA_LIB_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    if (RELEASE)
        target_compile_definitions(${CUDA_LIB_NAME} PRIVATE NDEBUG)
    else()
        target_compile_definitions(${CUDA_LIB_NAME} PRIVATE KERNEL_DEBUG)
    endif()
endif()

# Link and build
set(PROJECT_LIBS
    ${LIBS_NAME}
    assimp
    ${OpenCL_LIBRARIES}
    ${CUDA_LIB_NAME}
    OpenMP::OpenMP_CXX
    stdc++fs
    glfw
)
add_executable(${PROJECT_NAME} ${SOURCES} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_LIBS})
target_compile_options(${PROJECT_NAME} PRIVATE -march=native -O3)
target_include_directories(${PROJECT_NAME} PRIVATE
    src
    include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    kernels
    ${GLFW_INCLUDE_PATH}
)

# Options
option(RELEASE "Build in release mode" ON)
option(LOG "Enable logging" OFF)
option(PROFILE "Enable profiling" OFF)

if (RELEASE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE NDEBUG)
    set(LOG OFF)
    set(PROFILE OFF)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -g)
    set(LOG ON)
    set(PROFILE ON)
endif()

if (LOG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE LOG)
endif()
if (PROFILE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE PROFILE)
endif()

# Defines
macro(set_backend_defines TARGET)
    if (OpenCL_FOUND)
        target_compile_definitions(${TARGET} PRIVATE
            CL_HPP_ENABLE_EXCEPTIONS __CL_ENABLE_EXCEPTIONS BACKEND_OPENCL)
        if (${OpenCL_VERSION_MAJOR} EQUAL 2)
            target_compile_definitions(${TARGET} PRIVATE
                CL_HPP_TARGET_OPENCL_VERSION=200
                OPENCL_2
            )
        else()
            target_compile_definitions(${TARGET} PRIVATE CL_HPP_TARGET_OPENCL_VERSION=120)
        endif()
    else()
        target_compile_definitions(${TARGET} PRIVATE BACKEND_CUDA)
    endif()
endmacro()
set_backend_defines(${PROJECT_NAME})

target_compile_definitions(${PROJECT_NAME} PRIVATE
    SRC_PATH="${CMAKE_SOURCE_DIR}/"
    OPENCL_KERNEL_BINARY="${OPENCL_KERNEL_BINARY}"
    ASSETS_PATH="${CMAKE_SOURCE_DIR}/assets/"
)

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
