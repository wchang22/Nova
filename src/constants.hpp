#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#ifndef SRC_PATH
  #define SRC_PATH ""
#endif
#ifndef OPENCL_KERNEL_BINARY
  #define OPENCL_KERNEL_BINARY ""
#endif
#ifndef ASSETS_PATH
  #define ASSETS_PATH ""
#endif

#ifndef NDEBUG
  #define NUM_PROFILE_ITERATIONS 10
#else
  #define NUM_PROFILE_ITERATIONS 1
#endif

#include <cstddef>
#include <utility>

#include "shared_constants.hpp"

constexpr char APP_DESCRIPTION[] = "High performance GPU accelerated ray tracer using OpenCL/CUDA";

// Window constants
constexpr std::pair<int, int> MAX_RESOLUTION(7680, 4320);
constexpr char FONT_PATH[] = ASSETS_PATH "/fonts/SourceCodePro-Bold.ttf";
constexpr float FONT_SIZE = 14.0f;

// Raytracer constants
constexpr char SCENE_PATH[] = SRC_PATH "scene.toml";
constexpr char MODEL_FILE_TYPES[] = ".glb\0.gltf\0.obj\0";
constexpr char SKY_FILE_TYPES[] = ".hdr\0";
constexpr char IMAGE_EXTENSION[] = ".jpg";

#endif // CONSTANTS_HPP
