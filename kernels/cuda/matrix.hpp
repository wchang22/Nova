#ifndef CUDA_KERNEL_MATRIX_HPP
#define CUDA_KERNEL_MATRIX_HPP

#include "vector_math.h"
#include "kernel_types/matrix.hpp"

__device__
inline Mat3x3 make_mat3x3(const Mat3x4& mat) {
  return {
    make_float3(mat.x),
    make_float3(mat.y),
    make_float3(mat.z),
  };
}

__device__
inline float3 operator*(const Mat3x3& mat, float3 vec) {
  return {
    dot(mat.x, vec),
    dot(mat.y, vec),
    dot(mat.z, vec)
  };
}

__device__
inline Mat3x3 transpose(const Mat3x3& mat) {
  return {
    make_float3(mat.x.x, mat.y.x, mat.z.x),
    make_float3(mat.x.y, mat.y.y, mat.z.y),
    make_float3(mat.x.z, mat.y.z, mat.z.z)
  };
}

__device__
inline float3 operator*(const Mat3x4& mat, float3 vec) {
  float4 vec4 = make_float4(vec, 1.0f);
  return {
    dot(mat.x, vec4),
    dot(mat.y, vec4),
    dot(mat.z, vec4)
  };
}

#endif // CUDA_KERNEL_MATRIX_HPP
