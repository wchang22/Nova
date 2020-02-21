#ifndef VECTOR_CONVERSIONS_H
#define VECTOR_CONVERSIONS_H

inline __host__ __device__ uchar4 convert_uchar4(float4 a) {
  return make_uchar4(a.x, a.y, a.z, a.w);
}

#endif