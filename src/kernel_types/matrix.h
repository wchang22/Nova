#ifndef KERNEL_TYPE_MATRIX_H
#define KERNEL_TYPE_MATRIX_H

#include "backend/types.h"

struct Mat3x3 {
  float3 x;
  float3 y;
  float3 z;
};

struct Mat3x4 {
  float4 x;
  float4 y;
  float4 z;
};

#endif // KERNEL_TYPE_MATRIX_H