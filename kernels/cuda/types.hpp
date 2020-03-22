#ifndef CUDA_KERNEL_TYPES_HPP
#define CUDA_KERNEL_TYPES_HPP

#include "math_constants.h"
#include "vector_math.h"

struct Ray {
  float3 origin;
  float3 direction;
  float3 inv_direction;
  float3 nio;
};

__device__ inline Ray create_ray(float3 point, float3 direction, float epsilon) {
  float3 origin = point + direction * epsilon;
  float3 inv_direction = 1.0f / direction;
  float3 nio = -origin * inv_direction;
  return { origin, direction, inv_direction, nio };
}

struct Intersection {
  float3 barycentric;
  float length;
  int tri_index;
};

__device__ inline Intersection no_intersection() {
  return { make_float3(0.0f), CUDART_NORM_HUGE_F, -1 };
};

#endif // CUDA_KERNEL_TYPES_HPP