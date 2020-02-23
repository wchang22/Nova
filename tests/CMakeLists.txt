set(TEST_NAME tests)
file(GLOB_RECURSE TEST_SOURCES src/*)

add_executable(${TEST_NAME} EXCLUDE_FROM_ALL ${TEST_SOURCES} ${SOURCES})
target_link_libraries(${TEST_NAME} PRIVATE ${PROJECT_LIBS})
target_compile_options(${TEST_NAME} PRIVATE -std=c++17 -march=native -g -O3)
target_include_directories(${TEST_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/kernels
)
set_backend_defines(${TEST_NAME})