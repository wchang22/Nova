#ifndef CONFIGURATION_H
#define CONFIGURATION_H

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

#ifdef OPENCL_2
  #include <CL/cl2.hpp>
#else
  #ifdef __APPLE__
    #include <OpenCL/cl.hpp>
  #else
    #include <CL/cl.hpp>
  #endif
#endif

#include <glm/glm.hpp>

using namespace glm;

// Raytracer constants
constexpr cl_device_type DEVICE_TYPE = CL_DEVICE_TYPE_GPU;
const cl::NDRange LOCAL_SIZE(16, 16);
constexpr char MODEL_PATH[] = ASSETS_PATH"aircraft/aircraft.obj";
constexpr char KERNEL_PATH[] = KERNELS_PATH"raytrace.cl";

// Camera constants
const vec3 CAMERA_POSITION(-4, 2.8, 5);
const vec3 CAMERA_FORWARD(1, -0.5, -1);
const vec3 CAMERA_UP(0, 1, 0);
constexpr int CAMERA_FOVY = 45;

// BVH/Triangle constants
constexpr size_t TRIANGLES_PER_LEAF_BITS = 6;
constexpr size_t MIN_TRIANGLES_PER_LEAF = 8;
constexpr float MAX_BINS = 1024.f;
constexpr size_t MAX_TRIANGLES = (1 << (32 - TRIANGLES_PER_LEAF_BITS)) - 1;

static_assert(TRIANGLES_PER_LEAF_BITS <= 32);
static_assert(MIN_TRIANGLES_PER_LEAF < (1 << TRIANGLES_PER_LEAF_BITS));

#endif // CONFIGURATION_H
