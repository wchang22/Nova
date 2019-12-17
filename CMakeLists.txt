cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0015 NEW)

project(clraytracer)

find_package(OpenCL REQUIRED)
find_package(glm REQUIRED)

file(GLOB_RECURSE SOURCES src/* kernels/*)

set_source_files_properties(${SOURCES} PROPERTIES COMPILE_FLAGS
    "-Wall -Wextra -Werror -Wpedantic")

file(GLOB_RECURSE SOURCES ${SOURCES} include/*)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_options(${PROJECT_NAME} PRIVATE -lOpenCL)
target_compile_options(${PROJECT_NAME} PRIVATE -std=c++17)
target_include_directories(${PROJECT_NAME} PRIVATE src include)
target_link_directories(${PROJECT_NAME} PRIVATE include)

option(RELEASE "Build in release mode" ON)
option(LOG "Enable logging" OFF)
option(PROFILE "Enable profiling" OFF)

if (RELEASE)
    target_compile_options(${PROJECT_NAME} PRIVATE -O3)
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

target_compile_definitions(${PROJECT_NAME} PRIVATE -DCL_HPP_ENABLE_EXCEPTIONS)
target_compile_definitions(${PROJECT_NAME} PRIVATE -DCL_HPP_TARGET_OPENCL_VERSION=200)
target_compile_definitions(${PROJECT_NAME} PRIVATE -DKERNEL_PATH="${CMAKE_SOURCE_DIR}/kernels/")

message("Build Options -----------------------------------")
message("RELEASE ----------------------------------------- ${RELEASE}")
message("LOG --------------------------------------------- ${LOG}")
message("PROFILE ----------------------------------------- ${PROFILE}")
