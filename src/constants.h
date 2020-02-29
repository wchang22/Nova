#ifndef CONSTANTS_H
#define CONSTANTS_H

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

// Raytracer constants
constexpr char SCENE_PATH[] = SRC_PATH"scene.toml";

// BVH/Triangle constants
constexpr size_t TRIANGLES_PER_LEAF_BITS = 12;
constexpr size_t TRIANGLE_NUM_SHIFT = 32 - TRIANGLES_PER_LEAF_BITS;
constexpr size_t TRIANGLE_OFFSET_MASK =
  (0xFFFFFFFF << TRIANGLES_PER_LEAF_BITS) >> TRIANGLES_PER_LEAF_BITS;
constexpr size_t MIN_TRIANGLES_PER_LEAF = 8;
constexpr size_t MAX_TRIANGLES_PER_LEAF = (1 << TRIANGLES_PER_LEAF_BITS) - 1;
constexpr float MAX_BINS = 1024.f;
constexpr float OVERLAP_TOLERANCE = 1e-5f;
constexpr size_t MAX_TRIANGLES = (1 << (32 - TRIANGLES_PER_LEAF_BITS)) - 1;

static_assert(TRIANGLES_PER_LEAF_BITS <= 32);
static_assert(MIN_TRIANGLES_PER_LEAF < (1 << TRIANGLES_PER_LEAF_BITS));

#endif // CONSTANTS_H
