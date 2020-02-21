#ifndef CUDA_KERNEL_VECTOR_CONVERSIONS_H
#define CUDA_KERNEL_VECTOR_CONVERSIONS_H

inline __host__ __device__ uchar4 convert_uchar4(float4 a) {
  return make_uchar4(a.x, a.y, a.z, a.w);
}

inline __host__ __device__ float4 convert_float4(float3 a, float b) {
  return make_float4(a.x, a.y, a.z, b);
}

#endif