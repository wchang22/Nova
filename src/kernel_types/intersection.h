#ifndef KERNEL_INTERSECTION_H
#define KERNEL_INTERSECTION_H

#include "backend/types.h"

struct Intersection {
  float3 barycentric;
  float length;
  int32_t tri_index;
  int32_t ray_index;
};

#endif // KERNEL_INTERSECTION_H
