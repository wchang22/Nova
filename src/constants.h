#ifndef CONSTANTS_H
#define CONSTANTS_H

#ifndef SRC_PATH
  #define SRC_PATH ""
#endif
#ifndef KERNELS_PATH
  #define KERNELS_PATH ""
#endif
#ifndef ASSETS_PATH
  #define ASSETS_PATH ""
#endif

#ifndef NDEBUG
  #define NUM_PROFILE_ITERATIONS 100
#else
  #define NUM_PROFILE_ITERATIONS 1
#endif

#define STRINGIFY(x) #x

// Raytracer constants
constexpr char KERNEL_PATH[] = KERNELS_PATH"raytrace.cl";
constexpr char SCENE_PATH[] = SRC_PATH"scene.toml";

// BVH/Triangle constants
constexpr size_t TRIANGLES_PER_LEAF_BITS = 6;
constexpr size_t MIN_TRIANGLES_PER_LEAF = 8;
constexpr float MAX_BINS = 1024.f;
constexpr size_t MAX_TRIANGLES = (1 << (32 - TRIANGLES_PER_LEAF_BITS)) - 1;

static_assert(TRIANGLES_PER_LEAF_BITS <= 32);
static_assert(MIN_TRIANGLES_PER_LEAF < (1 << TRIANGLES_PER_LEAF_BITS));

#endif // CONSTANTS_H
