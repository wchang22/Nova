#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#ifndef SRC_PATH
  #define SRC_PATH ""
#endif
#ifndef KERNELS_PATH
  #define KERNELS_PATH
#endif
#ifndef KERNELS_PATH_STR
  #define KERNELS_PATH_STR ""
#endif
#ifndef ASSETS_PATH
  #define ASSETS_PATH ""
#endif

#ifndef NDEBUG
  #define NUM_PROFILE_ITERATIONS 100
#else
  #define NUM_PROFILE_ITERATIONS 1
#endif

#include <cstddef>
#include <utility>

constexpr char APP_DESCRIPTION[] = "High performance GPU accelerated ray tracer using OpenCL/CUDA";

// Window constants
constexpr std::pair<int, int> MAX_RESOLUTION(7680, 4320);
constexpr char FONT_PATH[] = ASSETS_PATH "/fonts/SourceCodePro-Bold.ttf";
constexpr float FONT_SIZE = 14.0f;

// Raytracer constants
constexpr char SCENE_PATH[] = SRC_PATH "scene.toml";
constexpr char MODEL_FILE_TYPES[] = ".obj";
constexpr char IMAGE_EXTENSION[] = ".jpg";

// BVH/Triangle constants
constexpr size_t TRIANGLES_PER_LEAF_BITS = 6;
constexpr size_t TRIANGLE_NUM_SHIFT = 32 - TRIANGLES_PER_LEAF_BITS;
constexpr size_t TRIANGLE_OFFSET_MASK =
  (0xFFFFFFFF << TRIANGLES_PER_LEAF_BITS) >> TRIANGLES_PER_LEAF_BITS;
constexpr size_t MIN_TRIANGLES_PER_LEAF = 8;
constexpr float MAX_BINS = 1024.f;
constexpr size_t MAX_TRIANGLES = (1 << (32 - TRIANGLES_PER_LEAF_BITS)) - 1;

static_assert(TRIANGLES_PER_LEAF_BITS <= 32);
static_assert(MIN_TRIANGLES_PER_LEAF < (1 << TRIANGLES_PER_LEAF_BITS));

#endif // CONSTANTS_HPP
