#ifndef CUDA_KERNEL_TRANSFORMS_H
#define CUDA_KERNEL_TRANSFORMS_H

#include "vector_math.h"

__device__
inline float3 uint3_to_float3(float3 u) {
  return u / 255.0f;
}

__device__
inline float3 triangle_interpolate(float3 barycentric_coords, float3 a, float3 b, float3 c) {
  return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

__device__
inline float2 triangle_interpolate(float3 barycentric_coords, float2 a, float2 b, float2 c) {
  return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

#endif // CUDA_KERNEL_TRANSFORMS_H
