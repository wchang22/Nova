#ifndef KERNEL_RANDOM_HPP
#define KERNEL_RANDOM_HPP

#include "kernels/backend/kernel.hpp"

namespace nova {

// https://raytracey.blogspot.com/2015/10/gpu-path-tracing-tutorial-1-drawing.html
DEVICE inline float rng(uint& seed1, uint& seed2) {
  // hash the seeds using bitwise AND and bitshifts
  seed1 = 36969 * (seed1 & 65535) + (seed1 >> 16);
  seed2 = 18000 * (seed2 & 65535) + (seed2 >> 16);

  uint ires = ((seed1) << 16) + (seed2);

  // Convert to float
  union {
    float f;
    uint ui;
  } res;

  res.ui = (ires & 0x007fffff) | 0x40000000;

  return (res.f - 2.0f) / 2.0f;
}

}

#endif // KERNEL_RANDOM_HPP
