#ifndef KERNEL_RAY_H
#define KERNEL_RAY_H

#include "backend/types.h"

struct Ray {
  float3 origin;
  float3 direction;
  uint32_t image_index;
};

#endif // KERNEL_RAY_H
