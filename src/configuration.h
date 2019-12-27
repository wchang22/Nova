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

#include <CL/cl2.hpp>
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

// BVH constants
constexpr size_t MIN_TRIANGLES_PER_LEAF = 15;
constexpr float MAX_BINS = 1024.f;

#endif // CONFIGURATION_H
