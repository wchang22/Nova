cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0015 NEW)

project(clraytracer)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(assimp REQUIRED)

# Library includes
set(LIBS_NAME cllib)
file(GLOB_RECURSE LIBS_SOURCES include/*)
add_library(${LIBS_NAME} ${LIBS_SOURCES})
target_include_directories(${LIBS_NAME} PRIVATE include)

# Source files
file(GLOB_RECURSE SOURCES src/*)
set_source_files_properties(${SOURCES} PROPERTIES COMPILE_FLAGS
    "-Wall -Wextra -Werror -Wpedantic")

# Link and build
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${LIBS_NAME} ${OpenCL_LIBRARIES} ${ASSIMP_LIBRARIES} OpenMP::OpenMP_CXX)
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
target_compile_definitions(${PROJECT_NAME} PRIVATE -DCL_HPP_ENABLE_EXCEPTIONS)
target_compile_definitions(${PROJECT_NAME} PRIVATE -DCL_HPP_TARGET_OPENCL_VERSION=200)
target_compile_definitions(${PROJECT_NAME} PRIVATE -DGLM_FORCE_AVX -DGLM_FORCE_SILENT_WARNINGS)
target_compile_definitions(${PROJECT_NAME} PRIVATE -DKERNELS_PATH="${CMAKE_SOURCE_DIR}/kernels/")
target_compile_definitions(${PROJECT_NAME} PRIVATE -DASSETS_PATH="${CMAKE_SOURCE_DIR}/assets/")

# Status
message("Build Options -----------------------------------")
message("RELEASE ----------------------------------------- ${RELEASE}")
message("LOG --------------------------------------------- ${LOG}")
message("PROFILE ----------------------------------------- ${PROFILE}")
