#ifndef CUDA_KERNEL_TRANSFORMS_HPP
#define CUDA_KERNEL_TRANSFORMS_HPP

#include "kernels/backend/kernel.hpp"
#include "kernels/backend/vector.hpp"

namespace nova {

__device__ inline float3 uint3_to_float3(const uint3& u) { return make_vector<float3>(u) / 255.0f; }

__device__ inline float3 uchar3_to_float3(const uchar3& u) {
  return make_vector<float3>(u) / 255.0f;
}

__device__ inline uchar3 float3_to_uchar3(const float3& u) {
  return make_vector<uchar3, float3>(u * 255.0f);
}

template <typename U>
__device__ inline U
triangle_interpolate(const float3& barycentric_coords, const U& a, const U& b, const U& c) {
  return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

__device__ inline int linear_index(const int2& p, int width) { return p.y * width + p.x; }

__device__ inline float3 tone_map(const float3& x) { return x / (x + 1.0f); }

__device__ inline float3 gamma_correct(const float3& x) { return pow(x, 1.0f / 2.2f); }

__device__ inline float rgb_to_luma(const float3& rgb) {
  return dot(rgb, make_vector<float3>(0.299f, 0.587f, 0.114f));
}

}

#endif // CUDA_KERNEL_TRANSFORMS_HPP
