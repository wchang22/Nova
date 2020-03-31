#ifndef CUDA_KERNEL_TRANSFORMS_HPP
#define CUDA_KERNEL_TRANSFORMS_HPP

#include "vector_math.h"

namespace nova {

__device__ inline float3 uint3_to_float3(float3 u) { return u / 255.0f; }

__device__ inline float3
triangle_interpolate(float3 barycentric_coords, float3 a, float3 b, float3 c) {
  return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

__device__ inline float2
triangle_interpolate(float3 barycentric_coords, float2 a, float2 b, float2 c) {
  return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

__device__ inline int linear_index(int2 p, int width) { return p.y * width + p.x; }

__device__ inline float3 tone_map(float3 x) { return x / (x + 1.0f); }

__device__ inline float3 gamma_correct(float3 x) { return pow(x, 1.0f / 2.2f); }

}

#endif // CUDA_KERNEL_TRANSFORMS_HPP
