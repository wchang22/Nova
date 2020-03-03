#ifndef KERNEL_RAY_H
#define KERNEL_RAY_H

#include "backend/types.h"

typedef struct {
  float4 origin_index;
  float3 direction;
} PackedRay;

#endif // KERNEL_RAY_H
